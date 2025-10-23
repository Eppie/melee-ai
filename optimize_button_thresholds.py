#!/usr/bin/env python3
"""Optimize per-button decision thresholds on the validation set."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from column_map import ColumnMap, CONTROLLER_KEY_GROUPS
from config import get_config, init_config
from controller_quantization import quantize_targets
from feature_transforms import feature_spec_from_config
from model.nano_gpt import GPT
from train import build_inputs_for_gpt, _BUTTON_PRETTY
from utils import _resolve_device
from window_dataset import WindowDataset, worker_init_fn


SUPPORTED_METRICS = {"f1", "accuracy", "balanced_accuracy", "mcc"}


def _ensure_absolute(path: Path, anchor: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (anchor / path).resolve()


def _make_loader(data_root: Path, batch_size: int, num_workers: int, pin_memory: bool,
                 prefetch_factor: int | None, persistent_workers: bool) -> tuple[DataLoader, WindowDataset]:
    config = get_config()
    feature_spec = feature_spec_from_config(config.features)
    dataset = WindowDataset(
        data_dir=str(data_root),
        feature_transforms=feature_spec,
        return_numpy=False,
    )

    mp_ctx = None
    if num_workers > 0:
        try:
            mp_ctx = torch.multiprocessing.get_context("spawn")
        except RuntimeError:
            mp_ctx = None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        worker_init_fn=worker_init_fn,
        drop_last=False,
        multiprocessing_context=mp_ctx,
    )
    return loader, dataset


def _load_checkpoint(path: Path, device: torch.device) -> GPT:
    model = GPT(get_config())
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _gather_statistics(
        model: GPT,
        loader: DataLoader,
        colmap: ColumnMap,
        device: torch.device,
        thresholds: torch.Tensor,
        max_batches: int | None,
        progress: bool,
) -> Dict[str, torch.Tensor]:
    num_thresholds = thresholds.numel()
    num_buttons = len(CONTROLLER_KEY_GROUPS["buttons"])

    tp = torch.zeros(num_buttons, num_thresholds, dtype=torch.float64)
    fp = torch.zeros_like(tp)
    fn = torch.zeros_like(tp)
    total_frames = 0

    thresholds = thresholds.to(device)

    total_batches = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):
        X = batch["X"].to(device, non_blocking=True)
        Y = batch["Y"].to(device, non_blocking=True)

        inputs_td = build_inputs_for_gpt(X, colmap)
        target_info = quantize_targets(Y, colmap, input_domain="unit11")

        pred = model(inputs_td)
        logits_btn = pred["buttons"]  # [B,L,K]
        probs = torch.sigmoid(logits_btn)

        true = target_info["buttons"].to(device=device).bool()  # [B,L,K]
        total_frames += true.numel() // num_buttons

        probs_exp = probs.unsqueeze(-1)  # [B,L,K,1]
        thr = thresholds.view(1, 1, 1, -1)
        pred_mask = probs_exp > thr  # [B,L,K,T]

        true_exp = true.unsqueeze(-1)

        tp_batch = (pred_mask & true_exp).sum(dim=(0, 1)).cpu().to(tp.dtype)
        fp_batch = (pred_mask & (~true_exp)).sum(dim=(0, 1)).cpu().to(fp.dtype)
        fn_batch = ((~pred_mask) & true_exp).sum(dim=(0, 1)).cpu().to(fn.dtype)

        tp += tp_batch
        fp += fp_batch
        fn += fn_batch

        if progress:
            pct = (batch_idx / total_batches) * 100.0
            print(
                f"[{batch_idx}/{total_batches}] {pct:5.1f}% | frames {total_frames:,}",
                end="\r",
                flush=True,
            )

        if max_batches is not None and batch_idx >= max_batches:
            break

    if progress:
        print()

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_frames": total_frames,
    }


def _metric_values(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, total_frames: float, metric: str) -> np.ndarray:
    if metric == "f1":
        numerator = 2.0 * tp
        denominator = 2.0 * tp + fp + fn
        return np.divide(numerator, denominator, out=np.zeros_like(tp), where=denominator > 0)
    if metric == "accuracy":
        if total_frames <= 0:
            return np.zeros_like(tp)
        tn = total_frames - tp - fp - fn
        return (tp + tn) / total_frames
    if metric == "balanced_accuracy":
        tn = total_frames - tp - fp - fn
        tpr = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        tnr = np.divide(tn, tn + fp, out=np.zeros_like(tp), where=(tn + fp) > 0)
        return 0.5 * (tpr + tnr)
    if metric == "mcc":
        tn = total_frames - tp - fp - fn
        numerator = tp * tn - fp * fn
        denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        denom = np.sqrt(np.clip(denom, a_min=0.0, a_max=None))
        return np.divide(numerator, denom, out=np.zeros_like(numerator), where=denom > 0)
    raise ValueError(f"Unsupported metric '{metric}'")


def _compute_best_thresholds(stats: Dict[str, torch.Tensor], thresholds: torch.Tensor,
                             metrics: Sequence[str]) -> Dict[str, List[Dict[str, float]]]:
    tp = stats["tp"].numpy()
    fp = stats["fp"].numpy()
    fn = stats["fn"].numpy()
    total_frames = float(stats["total_frames"])

    thr_np = thresholds.cpu().numpy()
    results: Dict[str, List[Dict[str, float]]] = {}

    for metric in metrics:
        values = _metric_values(tp, fp, fn, total_frames, metric)
        best_info: List[Dict[str, float]] = []
        for button_idx in range(tp.shape[0]):
            scores = values[button_idx]
            best_idx = int(np.nanargmax(scores))
            best_thr = float(thr_np[best_idx])
            best_score = float(scores[best_idx])

            tp_b = float(tp[button_idx, best_idx])
            fp_b = float(fp[button_idx, best_idx])
            fn_b = float(fn[button_idx, best_idx])
            tn_b = float(total_frames - tp_b - fp_b - fn_b)

            precision = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0
            recall = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
            f1 = (2 * tp_b) / (2 * tp_b + fp_b + fn_b) if (2 * tp_b + fp_b + fn_b) > 0 else 0.0
            accuracy = (tp_b + tn_b) / total_frames if total_frames else 0.0
            tnr = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0
            balanced_accuracy = 0.5 * (recall + tnr)
            denom = (tp_b + fp_b) * (tp_b + fn_b) * (tn_b + fp_b) * (tn_b + fn_b)
            if denom > 0:
                mcc = (tp_b * tn_b - fp_b * fn_b) / float(np.sqrt(max(denom, 0.0)))
            else:
                mcc = 0.0

            best_info.append({
                "threshold": best_thr,
                "score": best_score,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "mcc": mcc,
            })
        results[metric] = best_info

    return results


def _baseline_metrics(stats: Dict[str, torch.Tensor], thresholds: torch.Tensor, base_threshold: float = 0.5) -> List[Dict[str, float]]:
    tp = stats["tp"].numpy()
    fp = stats["fp"].numpy()
    fn = stats["fn"].numpy()
    total_frames = float(stats["total_frames"])

    thr_np = thresholds.cpu().numpy()
    base_idx = int(np.argmin(np.abs(thr_np - base_threshold)))

    tn = total_frames - tp - fp - fn

    baseline = []
    for button_idx in range(tp.shape[0]):
        tp_b = float(tp[button_idx, base_idx])
        fp_b = float(fp[button_idx, base_idx])
        fn_b = float(fn[button_idx, base_idx])
        tn_b = float(tn[button_idx, base_idx])

        precision = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0
        recall = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
        f1 = (2 * tp_b) / (2 * tp_b + fp_b + fn_b) if (2 * tp_b + fp_b + fn_b) > 0 else 0.0
        accuracy = (tp_b + tn_b) / total_frames if total_frames else 0.0
        tnr = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0
        balanced_accuracy = 0.5 * (recall + tnr)
        denom = (tp_b + fp_b) * (tp_b + fn_b) * (tn_b + fp_b) * (tn_b + fn_b)
        if denom > 0:
            mcc = (tp_b * tn_b - fp_b * fn_b) / float(np.sqrt(max(denom, 0.0)))
        else:
            mcc = 0.0

        baseline.append({
            "threshold": float(thr_np[base_idx]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "mcc": mcc,
        })

    return baseline


def _button_labels() -> List[str]:
    labels = []
    for name in CONTROLLER_KEY_GROUPS["buttons"]:
        labels.append(_BUTTON_PRETTY.get(name, name.upper()))
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize per-button thresholds on the validation set")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path; defaults to latest in train.out_dir")
    parser.add_argument("--data-root", type=Path, default=None, help="Validation dataset root (defaults to validation_set)")
    parser.add_argument("--device", default="auto", help="Torch device to use")
    parser.add_argument("--batch-size", type=int, default=None, help="Override validation batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override validation num workers")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="Override validation prefetch factor")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pin_memory")
    parser.add_argument("--threshold-samples", type=int, default=201, help="Number of thresholds between 0 and 1 to evaluate")
    parser.add_argument("--metrics", nargs="+", default=None, choices=sorted(SUPPORTED_METRICS),
                        help="Optimization metrics (any combination of f1, accuracy, balanced_accuracy, mcc)")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional cap on number of batches")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    parser.add_argument("--no-progress", action="store_true", help="Disable batch progress prints")
    parser.add_argument("--save-path", type=Path, default=None, help="Path to save F1-optimal thresholds (JSON)")
    return parser.parse_args()


def main() -> None:
    init_config()
    config = get_config()
    args = parse_args()

    project_root = Path(__file__).resolve().parent

    data_root = args.data_root or project_root / "validation_set"
    data_root = _ensure_absolute(data_root, project_root)
    if not data_root.exists():
        print(f"ERROR: validation data root {data_root} not found", file=sys.stderr)
        sys.exit(1)

    checkpoint = args.checkpoint
    if checkpoint is None:
        ckpt_dir = _ensure_absolute(Path(config.train.out_dir), project_root)
        candidates = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print(f"ERROR: no checkpoints found in {ckpt_dir}", file=sys.stderr)
            sys.exit(1)
        checkpoint = candidates[0]
        print(f"Loading latest checkpoint: {checkpoint}")
    else:
        checkpoint = _ensure_absolute(checkpoint, Path.cwd())

    device = _resolve_device()

    batch_size = args.batch_size or config.train.batch_size
    num_workers = args.num_workers if args.num_workers is not None else config.train.num_workers
    prefetch_factor = args.prefetch_factor if args.prefetch_factor is not None else config.train.prefetch_factor
    pin_memory = config.train.pin_memory and not args.no_pin_memory
    persistent_workers = config.train.persistent_workers and num_workers > 0

    loader, dataset = _make_loader(data_root, batch_size, num_workers, pin_memory, prefetch_factor, persistent_workers)
    colmap = ColumnMap.from_dataset(dataset)

    model = _load_checkpoint(checkpoint, device)

    num_thresholds = max(2, args.threshold_samples)
    thresholds = torch.linspace(0.0, 1.0, steps=num_thresholds)

    # if args.max_batches is not None:
    #     loader = list(loader)[:args.max_batches]

    stats = _gather_statistics(
        model,
        loader,
        colmap,
        device,
        thresholds,
        args.max_batches,
        progress=not args.no_progress,
    )

    if args.metrics:
        metrics = list(dict.fromkeys(args.metrics))  # remove duplicates preserving order
    else:
        metrics = list(SUPPORTED_METRICS)

    best = _compute_best_thresholds(stats, thresholds, metrics)
    baseline = _baseline_metrics(stats, thresholds)

    button_labels = _button_labels()

    button_keys = list(CONTROLLER_KEY_GROUPS["buttons"])

    if args.json:
        output = {
            "checkpoint": str(checkpoint),
            "buttons": button_keys,
            "metrics": metrics,
            "baseline": baseline,
            "optimized": best,
        }
        print(json.dumps(output, indent=2))
        return

    print("Validation threshold search summary")
    print(f"Checkpoint: {checkpoint}")
    print(f"Frames evaluated: {stats['total_frames']:,}")
    print(f"Threshold grid: {num_thresholds} samples between 0 and 1")

    print("\nBaseline metrics at threshold 0.50:")
    for label, metrics_dict in zip(button_labels, baseline):
        print(
            f"  {label:<6} thr {metrics_dict['threshold']:.3f} | "
            f"prec {metrics_dict['precision']:.3f} | rec {metrics_dict['recall']:.3f} | "
            f"f1 {metrics_dict['f1']:.3f} | acc {metrics_dict['accuracy']:.3f} | "
            f"bal {metrics_dict['balanced_accuracy']:.3f} | mcc {metrics_dict['mcc']:.3f}"
        )

    for metric in metrics:
        print(f"\nBest thresholds maximizing {metric.upper()}:")
        for label, info in zip(button_labels, best[metric]):
            print(
                f"  {label:<6} thr {info['threshold']:.3f} | "
                f"prec {info['precision']:.3f} | rec {info['recall']:.3f} | "
                f"f1 {info['f1']:.3f} | acc {info['accuracy']:.3f} | "
                f"bal {info['balanced_accuracy']:.3f} | mcc {info['mcc']:.3f} | {metric} {info['score']:.3f}"
            )

    save_path: Path = args.save_path or (project_root / "button_thresholds.json")
    f1_info = best.get("f1")
    if not f1_info:
        print("WARNING: Cannot save thresholds because F1 metric was not evaluated.", file=sys.stderr)
    else:
        thresholds_list = [info["threshold"] for info in f1_info]
        thresholds_map = {key: float(info["threshold"]) for key, info in zip(button_keys, f1_info)}
        payload = {
            "checkpoint": str(checkpoint),
            "metric": "f1",
            "buttons": button_keys,
            "thresholds": thresholds_list,
            "threshold_map": thresholds_map,
        }
        save_path = _ensure_absolute(save_path, Path.cwd())
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved F1-optimal thresholds to {save_path}")


if __name__ == "__main__":
    main()
