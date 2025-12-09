"""Validation utilities."""

import torch
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm

from ..model import FuturePositionPredictor, mixture_nll_loss
from ..config import Config


def validate(
    model: FuturePositionPredictor,
    val_loader: DataLoader,
    device: str = 'cuda',
) -> Dict[str, float]:
    """Run validation on validation set.

    Args:
        model: Model to validate
        val_loader: Validation dataloader
        device: Device to run on

    Returns:
        Dict of validation metrics:
            - loss: Average NLL loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    metric_sums: Dict[str, float] = {}

    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="Validation"):
            metrics = validate_step(model, batch, device)
            total_loss += metrics['loss']
            num_batches += 1
            # Aggregate extra metrics if present
            for k, v in metrics.items():
                if k == 'loss':
                    continue
                if k == 'per_horizon':
                    if 'per_horizon' not in metric_sums or not isinstance(metric_sums.get('per_horizon'), list):
                        metric_sums['per_horizon'] = []
                    metric_sums['per_horizon'].append(v)
                else:
                    if k not in metric_sums:
                        metric_sums[k] = 0.0
                    metric_sums[k] += v

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    results = {'loss': avg_loss}
    for k, v in metric_sums.items():
        if k == 'per_horizon':
            # Average per-horizon metrics across batches
            merged = {}
            counts = {}
            for batch_dict in v:
                for h, metrics in batch_dict.items():
                    if h not in merged:
                        merged[h] = {mk: 0.0 for mk in metrics}
                        counts[h] = 0
                    for mk, mv in metrics.items():
                        merged[h][mk] += mv
                    counts[h] += 1
            results['per_horizon'] = {
                h: {mk: mv / max(counts[h], 1) for mk, mv in md.items()}
                for h, md in merged.items()
            }
        else:
            results[k] = v / num_batches
    return results


@torch.inference_mode()
def validate_step(
    model: FuturePositionPredictor,
    batch: tuple,
    device: str,
) -> dict:
    """Single validation step.

    Args:
        model: Model
        batch: (context, targets, valid_mask)
        device: Device

    Returns:
        Dict with metrics for this batch
    """
    context, targets, valid_mask = batch
    
    # Move to device and convert to bf16 (if supported, otherwise float32)
    # Assuming we want to validate with the same precision as training if possible,
    # but for safety on all platforms, we'll let pytorch handle autocast or just move to device.
    # For consistency with train.py plan which uses bf16 explicitly:
    if device.startswith('cuda') and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    context = context.to(device, dtype=dtype)
    targets = targets.to(device, dtype=dtype)
    valid_mask = valid_mask.to(device)

    p1_trajectories, p2_trajectories = model(context)

    # Compute loss over all horizons
    loss = 0
    n_horizons = len(p1_trajectories)
    
    for i in range(n_horizons):
        loss += mixture_nll_loss(
            p1_trajectories[i], targets[:, i, :2], valid_mask[:, i]
        )
        loss += mixture_nll_loss(
            p2_trajectories[i], targets[:, i, 2:4], valid_mask[:, i]
        )
    
    loss = loss / (2 * n_horizons)

    # Extra metrics
    metrics = {'loss': loss.item()}
    from future_position.train.train import compute_batch_metrics  # local import to avoid cycle
    metrics.update(compute_batch_metrics(p1_trajectories, p2_trajectories, targets, valid_mask))

    return metrics
