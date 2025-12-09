"""Visualize future-position predictions efficiently with a single persistent matplotlib figure.

We render the stage once, keep artists alive (heatmap + markers), and update data each frame
using blitting to avoid per-frame figure creation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple
import time
import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm
from torch.serialization import add_safe_globals
from tqdm import tqdm

add_safe_globals([pathlib.PosixPath])

from libmelee.melee import enums
from libmelee.melee.stages import (
    BLASTZONES,
    EDGE_GROUND_POSITION,
    top_platform_position,
    right_platform_position,
    left_platform_position,
)

from future_position.config import Config
from future_position.constants import (
    CONTEXT_LENGTH,
    HORIZONS,
    STAGE_HALF_WIDTH,
    STAGE_HALF_HEIGHT,
    PLAYER_FEATURES_DIM,
)
from future_position.data.parse_slp import parse_slp_to_features
from future_position.model import FuturePositionPredictor
from future_position.train.train import resolve_device, load_checkpoint


STAGE_NAMES = {
    enums.Stage.BATTLEFIELD: "Battlefield",
    enums.Stage.FINAL_DESTINATION: "Final Destination",
    enums.Stage.DREAMLAND: "Dreamland",
    enums.Stage.FOUNTAIN_OF_DREAMS: "Fountain of Dreams",
    enums.Stage.POKEMON_STADIUM: "Pokemon Stadium",
    enums.Stage.YOSHIS_STORY: "Yoshi's Story",
}


def stage_edge_positions(stage: enums.Stage) -> tuple[Optional[float], Optional[float]]:
    STAGE_EDGES = {
        enums.Stage.BATTLEFIELD: (-68.4, 68.4),
        enums.Stage.FINAL_DESTINATION: (-85.5656, 85.5656),
        enums.Stage.DREAMLAND: (-77.2713, 77.2713),
        enums.Stage.FOUNTAIN_OF_DREAMS: (-63.3499, 63.3499),
        enums.Stage.POKEMON_STADIUM: (-87.7501, 87.7501),
        enums.Stage.YOSHIS_STORY: (-56.0, 56.0),
    }
    return STAGE_EDGES.get(stage, (None, None))


def get_stage_bounds(stage: enums.Stage) -> tuple[float, float, float, float]:
    """Blastzone boundaries (left, right, top, bottom)."""
    if stage not in BLASTZONES:
        return BLASTZONES.get(enums.Stage.FINAL_DESTINATION)
    return BLASTZONES[stage]


def get_gameplay_bounds(stage: enums.Stage) -> tuple[float, float, float, float]:
    """Focused gameplay area."""
    GAMEPLAY_BOUNDS = {
        enums.Stage.BATTLEFIELD: (-80, 80, 90, -60),
        enums.Stage.FINAL_DESTINATION: (-90, 90, 90, -60),
        enums.Stage.DREAMLAND: (-85, 85, 95, -60),
        enums.Stage.FOUNTAIN_OF_DREAMS: (-75, 75, 85, -60),
        enums.Stage.POKEMON_STADIUM: (-85, 85, 85, -60),
        enums.Stage.YOSHIS_STORY: (-70, 70, 75, -55),
    }
    if stage not in GAMEPLAY_BOUNDS:
        left, right, top, bottom = (-90, 90, 90, -60)
    else:
        left, right, top, bottom = GAMEPLAY_BOUNDS[stage]
    width = right - left
    height = top - bottom
    margin_x = width * 0.10
    margin_y = height * 0.10
    return (left - margin_x, right + margin_x, top + margin_y, bottom - margin_y)


def load_model(checkpoint_path: Path, device: str) -> FuturePositionPredictor:
    config = Config()
    model = FuturePositionPredictor(
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        context_length=config.data.context_length,
    )
    model.to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    return model


def predict_batch(
    model: FuturePositionPredictor,
    context: torch.Tensor,  # [B, T, F]
    horizon_idx: int,
    device: str,
) -> Tuple[np.ndarray, dict]:
    """Run model and return predicted absolute deltas for P1 and mixture."""
    with torch.inference_mode():
        p1_list, _ = model(context.to(device))
        p1_mix = p1_list[horizon_idx]

        w = p1_mix["weights"]
        mu_x = p1_mix["mu_x"]
        mu_y = p1_mix["mu_y"]
        dx = (w * mu_x).sum(dim=-1)
        dy = (w * mu_y).sum(dim=-1)
        p1_delta = torch.stack([dx, dy], dim=-1)  # [B,2]

        scale = torch.tensor([STAGE_HALF_WIDTH, STAGE_HALF_HEIGHT], device=device, dtype=p1_delta.dtype)
        p1_delta_stage = p1_delta * scale
        return (
            p1_delta_stage.cpu().numpy(),
            {k: v.detach() for k, v in p1_mix.items()},
        )


def mixture_to_heatmap(
    mix: dict,
    origin: Tuple[float, float],
    bounds: Tuple[float, float, float, float],
    resolution: int = 96,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Convert a P1 mixture into a heatmap over absolute positions."""
    w = mix["weights"][0]
    mu_x = mix["mu_x"][0] * STAGE_HALF_WIDTH + origin[0]
    mu_y = mix["mu_y"][0] * STAGE_HALF_HEIGHT + origin[1]
    sigma_x = mix["sigma_x"][0] * STAGE_HALF_WIDTH
    sigma_y = mix["sigma_y"][0] * STAGE_HALF_HEIGHT

    left, right, top, bottom = bounds
    xs = np.linspace(left, right, resolution)
    ys = np.linspace(bottom, top, resolution)
    xv, yv = np.meshgrid(xs, ys)
    heat = np.zeros_like(xv, dtype=np.float32)

    for k in range(len(w)):
        wx = w[k].item()
        if wx < 1e-6:
            continue
        sx = max(sigma_x[k].item(), 1e-3)
        sy = max(sigma_y[k].item(), 1e-3)
        mx = mu_x[k].item()
        my = mu_y[k].item()
        gauss = wx * np.exp(-0.5 * (((xv - mx) / sx) ** 2 + ((yv - my) / sy) ** 2))
        gauss /= (2 * np.pi * sx * sy)
        heat += gauss

    if heat.max() > 0:
        heat /= heat.max()
    return heat, (left, right, bottom, top)


class BlitViz:
    """Hold persistent matplotlib artists and provide fast per-frame updates via blitting."""

    def __init__(self, stage_enum: Optional[enums.Stage], figsize=(8, 7), dpi=100):
        self.stage_enum = stage_enum
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvasAgg(self.fig)
        stage_name = STAGE_NAMES.get(stage_enum, str(stage_enum)) if stage_enum else "Unknown"
        self.ax.set_title(f"{stage_name} — future horizon", fontsize=14)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect("equal")

        # Static background: stage/platforms
        if stage_enum:
            left, right, top, bottom = get_gameplay_bounds(stage_enum)
            self.ax.set_xlim(left, right)
            self.ax.set_ylim(bottom, top)
            # Use EDGE_GROUND_POSITION to draw main platform width accurately
            edge = EDGE_GROUND_POSITION.get(stage_enum)
            if edge is None:
                stage_left, stage_right, blast_top, blast_bottom = get_stage_bounds(stage_enum)
                edge = (stage_right - stage_left) / 2.0
            stage_left = -edge
            stage_right = edge
            stage_height = 10
            stage_rect = Rectangle(
                (stage_left, -stage_height),
                stage_right - stage_left,
                stage_height,
                fill=True,
                facecolor='dimgray',
                edgecolor='white',
                linewidth=1.5,
                alpha=0.6,
                zorder=1,
            )
            self.ax.add_patch(stage_rect)
            platform_thickness = 4.0
            for pos_fn in (top_platform_position, left_platform_position, right_platform_position):
                h, l, r = pos_fn(stage_enum)
                if h is not None:
                    self.ax.add_patch(Rectangle(
                        (l, h - platform_thickness / 2),
                        r - l,
                        platform_thickness,
                        fill=True,
                        facecolor='gray',
                        edgecolor='white',
                        linewidth=1,
                        alpha=0.6,
                        zorder=3,
                    ))
        else:
            self.ax.set_xlim(-90, 90)
            self.ax.set_ylim(-80, 120)

        # Dynamic artists
        self.heatmap_im = self.ax.imshow(
            np.zeros((2, 2)),
            extent=(0, 1, 0, 1),
            origin="lower",
            cmap="hot",
            alpha=0.45,
            zorder=0,
            aspect="auto",
        )
        self.p1_now, = self.ax.plot([], [], 'o', color='green', markersize=6, zorder=5)
        self.p2_now, = self.ax.plot([], [], 'o', color='red', markersize=6, zorder=5)
        self.p1_future, = self.ax.plot([], [], '*', color='green', markersize=12, zorder=6)

        self.fig.tight_layout()
        self.canvas.draw()
        self.bg = self.canvas.copy_from_bbox(self.ax.bbox)

    def update(
        self,
        p1_now: Tuple[float, float],
        p2_now: Tuple[float, float],
        p1_true: Tuple[float, float],
        heatmap: np.ndarray,
        extent: Tuple[float, float, float, float],
    ) -> np.ndarray:
        self.canvas.restore_region(self.bg)

        self.heatmap_im.set_data(heatmap)
        self.heatmap_im.set_extent(extent)
        vmax = float(heatmap.max())
        if vmax <= 0:
            vmax = 1.0
        self.heatmap_im.set_clim(0.0, vmax)
        self.p1_now.set_data([p1_now[0]], [p1_now[1]])
        self.p2_now.set_data([p2_now[0]], [p2_now[1]])
        self.p1_future.set_data([p1_true[0]], [p1_true[1]])

        self.ax.draw_artist(self.heatmap_im)
        self.ax.draw_artist(self.p1_now)
        self.ax.draw_artist(self.p2_now)
        self.ax.draw_artist(self.p1_future)
        self.canvas.blit(self.ax.bbox)
        img = np.asarray(self.canvas.buffer_rgba())
        return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize future position predictions (blitting).")
    parser.add_argument("--replay", type=Path, required=True, help="Path to .slp file")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path")
    parser.add_argument("--horizon", type=int, default=30, help="Horizon (frames) to visualize (must be in HORIZONS)")
    parser.add_argument("--max-frames", type=int, default=2000, help="Max frames to render")
    args = parser.parse_args()

    if args.horizon not in HORIZONS:
        raise ValueError(f"Horizon {args.horizon} not in HORIZONS {HORIZONS}")
    horizon_idx = HORIZONS.index(args.horizon)

    # Timers
    t_parse = 0.0
    t_model_load = 0.0
    t_infer = 0.0
    t_heatmap = 0.0
    t_render = 0.0
    t_cv2 = 0.0

    device = resolve_device("auto")
    t0 = time.time()
    model = load_model(args.checkpoint, device)
    t_model_load = time.time() - t0

    t0 = time.time()
    data = parse_slp_to_features(args.replay)
    t_parse = time.time() - t0

    stage_value = data["stage"]
    try:
        stage_enum = enums.Stage(stage_value)
    except Exception:
        stage_enum = None

    fps = 60
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Initialize viz
    viz = BlitViz(stage_enum)
    dummy = viz.update((0, 0), (0, 0), (0, 0), np.zeros((2, 2)), (0, 1, 0, 1))
    h, w = dummy.shape[:2]
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (w, h))

    feats = data["features"]
    context_len = CONTEXT_LENGTH
    max_idx = min(len(feats) - args.horizon - 1, args.max_frames + context_len)
    frames_rendered = 0

    for t in tqdm(range(context_len, max_idx), desc="Rendering", unit="f"):
        context_np = feats[t - context_len : t]  # [T, F]
        base_np = feats[t]  # current frame for delta origin

        context = torch.from_numpy(context_np).unsqueeze(0).float()  # [1,T,F]
        t_start = time.time()
        p1_delta, p1_mix = predict_batch(model, context, horizon_idx, device)
        t_infer += time.time() - t_start

        p1_now = (base_np[0], base_np[1])
        p2_now = (base_np[PLAYER_FEATURES_DIM], base_np[PLAYER_FEATURES_DIM + 1])
        gt = feats[t + args.horizon]
        p1_true = (gt[0], gt[1])

        bounds = get_gameplay_bounds(stage_enum) if stage_enum else (-100, 100, 100, -80)
        t_start = time.time()
        heat, extent = mixture_to_heatmap(p1_mix, p1_now, bounds)
        t_heatmap += time.time() - t_start

        t_start = time.time()
        img = viz.update(p1_now, p2_now, p1_true, heat, extent)
        t_render += time.time() - t_start

        t_start = time.time()
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))
        t_cv2 += time.time() - t_start

        frames_rendered += 1
        if frames_rendered >= args.max_frames:
            break

    writer.release()
    total_time = t_parse + t_model_load + t_infer + t_heatmap + t_render + t_cv2
    eps = frames_rendered / max(total_time, 1e-9)
    print(f"Wrote {frames_rendered} frames to {args.output}")
    print("Timings (seconds):")
    print(f"  parse_slp      : {t_parse:.3f}")
    print(f"  model_load     : {t_model_load:.3f}")
    print(f"  model_inference: {t_infer:.3f}")
    print(f"  heatmap        : {t_heatmap:.3f}")
    print(f"  render/mpl     : {t_render:.3f}")
    print(f"  cv2_write      : {t_cv2:.3f}")
    print(f"  total          : {total_time:.3f}")
    print(f"  overall fps    : {eps:.2f} frames/sec (including all stages)")


if __name__ == "__main__":
    main()
