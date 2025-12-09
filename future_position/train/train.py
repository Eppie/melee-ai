"""Training loop for future position prediction model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from typing import Optional, Tuple
import json
from tqdm import tqdm
import sys

from loguru import logger

# Add the parent directory to sys.path to import the local future_position package
sys.path.insert(0, str(Path(__file__).parents[2]))

from future_position.model import FuturePositionPredictor, mixture_nll_loss
from future_position.data import NPZDataset
from future_position.constants import STAGE_HALF_WIDTH, STAGE_HALF_HEIGHT
from future_position.config import Config
from future_position.train.validate import validate


def resolve_device(preferred: str) -> str:
    """Resolve training device with fallbacks.

    - If preferred is "cuda", use CUDA when available, else fall back to MPS then CPU.
    - If preferred is "mps", use MPS when available, else CPU.
    - If preferred is "auto", try CUDA -> MPS -> CPU.
    - Otherwise return the preferred string unchanged.
    """
    preferred = preferred.lower()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if preferred == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if has_mps:
            return "mps"
        return "cpu"

    if preferred == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("CUDA requested but not available; falling back to MPS or CPU.")
        if has_mps:
            return "mps"
        return "cpu"

    if preferred == "mps":
        if has_mps:
            return "mps"
        logger.warning("MPS requested but not available; falling back to CPU.")
        return "cpu"

    return preferred


def train(config: Config, resume_from: Optional[Path] = None) -> None:
    """Main training loop.

    Args:
        config: Training configuration
        resume_from: Optional checkpoint path to resume from

    Notes:
        - Uses bf16 precision (no grad scaler)
        - torch.compile for optimization
        - Saves checkpoints every N steps
        - Logs metrics every N steps
    """
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=True, enqueue=True)

    device = resolve_device(config.training.device)
    config.training.device = device
    logger.info(f"Selected device: {device}")

    torch.manual_seed(config.data.random_seed)

    # Setup directories
    if config.training.checkpoint_dir:
        checkpoint_dir = config.training.checkpoint_dir
    elif config.data.output_dir:
        checkpoint_dir = config.data.output_dir / "checkpoints"
    else:
        checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoints will be written to: {checkpoint_dir}")

    # Data
    train_loader, val_loader = setup_dataloaders(config)
    logger.info(
        f"Data loaded from {config.data.output_dir}; "
        f"train samples: {len(train_loader.dataset)}, val samples: {len(val_loader.dataset)}; "
        f"batch size: {config.training.batch_size}"
    )
    logger.info(
        f"Steps per epoch: train={len(train_loader)}, val={len(val_loader)}"
    )

    # Model
    model = setup_model(config, device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {total_params:,} parameters.")

    # Optimizer and Scheduler
    optimizer, scheduler = setup_optimizer(model, config)
    # If we know the true steps per epoch, update scheduler T_max/warmup based on that
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.training.max_epochs
    if hasattr(scheduler, "warmup_steps"):
        warmup_steps = min(int(config.training.warmup_frac * total_steps), 1000)
        scheduler.warmup_steps = warmup_steps
        scheduler.T_max = max(total_steps - warmup_steps, 1)
        logger.info(f"LR schedule: total_steps={total_steps}, warmup_steps={warmup_steps}, T_max={scheduler.T_max}")
    logger.info(
        f"Optimizer: AdamW lr={config.training.learning_rate}, "
        f"weight_decay={config.training.weight_decay}; "
        f"Scheduler: CosineAnnealingLR (eta_min={config.training.min_lr})"
    )

    start_epoch = 0
    step = 0

    if resume_from:
        meta = load_checkpoint(resume_from, model, optimizer, scheduler)
        start_epoch = meta.get('epoch', 0)
        step = meta.get('step', 0)
        logger.info(f"Resumed from {resume_from} at epoch {start_epoch}, step {step}")

    model.train()
    
    for epoch in range(start_epoch, config.training.max_epochs):
        logger.info(f"Starting epoch {epoch}")
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            metrics = train_step(model, batch, optimizer, scheduler, config)
            step += 1
            # Manual warmup: linearly ramp lr for first scheduler.warmup_steps
            if hasattr(scheduler, "warmup_steps") and scheduler.warmup_steps > 0 and step <= scheduler.warmup_steps:
                warmup_ratio = step / scheduler.warmup_steps
                for pg in optimizer.param_groups:
                    base_lr = config.training.learning_rate
                    pg['lr'] = base_lr * warmup_ratio

            if step % config.training.log_every == 0:
                # Simple logging to progress bar
                progress_bar.set_postfix(loss=f"{metrics['loss']:.4f}", lr=f"{metrics['lr']:.2e}")
                log_metrics(metrics, step)

            if step % config.training.save_every == 0:
                save_checkpoint(
                    model, optimizer, scheduler, step, metrics['loss'], config, checkpoint_dir
                )

        # Validation at end of epoch
        val_metrics = validate(model, val_loader, device)
    val_msg_parts = [f"Epoch {epoch} Validation: Loss {val_metrics['loss']:.4f}"]
    for key in ('mae_norm', 'rmse_norm', 'mae_stage', 'rmse_stage', 'sigma_mean', 'weight_entropy'):
        if key in val_metrics:
            val_msg_parts.append(f"{key}={val_metrics[key]:.4f}")
    logger.info(" ".join(val_msg_parts))
    if 'per_horizon' in val_metrics and val_metrics['per_horizon']:
        ph = val_metrics['per_horizon']
        parts = []
        for h, vals in ph.items():
            parts.append(
                f"h{h}:mae_norm={vals['mae_norm']:.3f},rmse_norm={vals['rmse_norm']:.3f},"
                f"mae_stage={vals['mae_stage']:.2f},rmse_stage={vals['rmse_stage']:.2f},"
                f"sigma={vals['sigma_mean']:.2f}"
            )
        logger.info(" | ".join(parts))
        model.train() # Switch back to train mode

        # Save epoch checkpoint
        save_checkpoint(
            model, optimizer, scheduler, step, val_metrics['loss'], config, checkpoint_dir, prefix=f"epoch_{epoch}"
        )


def setup_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    """Setup train and validation dataloaders.

    Args:
        config: Configuration

    Returns:
        (train_loader, val_loader)
    """
    data_dir = config.data.output_dir # Assuming data is in output_dir/episodes or similar structure as per build_dataset
    # Check if data_dir is directly the one containing .npz files or index.json
    if not (data_dir / 'index.json').exists():
        # Fallback check if it's inside 'processed' or similar if passed generically
        if (config.data.output_dir / 'processed' / 'index.json').exists():
            data_dir = config.data.output_dir / 'processed'
        else:
            logger.error(f"index.json not found in {data_dir} or {data_dir / 'processed'}")
            raise FileNotFoundError(f"index.json not found in {data_dir}")
    logger.info(f"Loading dataset index from {data_dir / 'index.json'}")
    
    dataset = NPZDataset(
        data_dir=data_dir,
        context_length=config.data.context_length,
        max_episodes=config.training.max_episodes,
    )

    if len(dataset) == 0:
        logger.error(f"No samples found in dataset at {data_dir}")
        raise ValueError(f"No samples found in dataset at {data_dir}")

    # Split train/val
    val_size = int(len(dataset) * config.data.val_fraction)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config.data.random_seed)
    )
    logger.info(
        f"Dataset split: train={train_size} samples, val={val_size} samples "
        f"(episodes used: {len(dataset.episodes)})"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        prefetch_factor=config.training.prefetch_factor,
        persistent_workers=config.training.persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        prefetch_factor=config.training.prefetch_factor,
        persistent_workers=config.training.persistent_workers,
    )

    return train_loader, val_loader


def setup_model(config: Config, device: str) -> nn.Module:
    """Initialize and setup model.

    Args:
        config: Model configuration
        device: Device to place model on

    Returns:
        Model (optionally compiled)
    """
    model = FuturePositionPredictor(
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        context_length=config.data.context_length,
    )
    model.to(device)
    logger.info(f"Model moved to device {device}")
    
    # Use BF16 for training
    # Note: Model weights are kept in float32 typically, but we cast inputs to bf16
    # and use autocast or rely on operations handling it.
    # The plan says "BF16 Without GradScaler", meaning we run forward in bf16.
    # We can cast the model to bf16 if we want pure bf16 weights, but usually mixed precision is safer.
    # However, the plan says "context = context.to('cuda', dtype=torch.bfloat16)".
    # Let's keep weights in fp32 and cast inputs, or cast model to bf16 if appropriate.
    # PyTorch layers handle mixed types well.
    # For "BF16 native training", we often can convert the whole model.
    # model.to(dtype=torch.bfloat16) 

    if config.training.use_compile and device.startswith("cuda"):
        try:
            model = torch.compile(model, mode=config.training.compile_mode)
            logger.info(f"Model compiled with mode={config.training.compile_mode}")
        except Exception as exc:
            logger.warning(f"torch.compile failed ({exc}); continuing without compilation.")
    elif config.training.use_compile:
        logger.info("Skipping torch.compile because device is not CUDA.")
    
    return model


def setup_optimizer(model: nn.Module, config: Config):
    """Setup optimizer and scheduler.

    Args:
        model: Model to optimize
        config: Training configuration

    Returns:
        (optimizer, scheduler)
    """
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        fused=True if config.training.device == 'cuda' else False
    )

    # Warmup + cosine schedule: warmup_steps = warmup_frac * total_steps, capped at 1000
    # If train_loader length is available, we override total_steps_est later in train().
    total_steps_est = config.training.max_epochs * 1000  # fallback estimate if loader length unknown
    warmup_steps = min(int(config.training.warmup_frac * total_steps_est), 1000)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps_est - warmup_steps, 1),
        eta_min=config.training.min_lr,
    )

    # Simple warmup hook: linearly ramp lr to base over warmup_steps
    for pg in optimizer.param_groups:
        pg["initial_lr"] = 0.0
    scheduler.warmup_steps = warmup_steps

    return optimizer, scheduler


def train_step(
    model: nn.Module,
    batch: tuple,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: Config,
) -> dict:
    """Single training step.

    Args:
        model: Model
        batch: (context, targets, valid_mask)
        optimizer: Optimizer
        scheduler: LR scheduler
        config: Config

    Returns:
        Dict with loss and metrics
    """
    device = config.training.device
    context, targets, valid_mask = batch

    # Move to device and cast to bf16
    if device.startswith('cuda') and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    context = context.to(device, dtype=dtype)
    targets = targets.to(device, dtype=dtype)
    valid_mask = valid_mask.to(device)

    optimizer.zero_grad(set_to_none=True)

    # Forward pass
    p1_trajectories, p2_trajectories = model(context)

    # Compute loss
    loss = compute_total_loss(
        p1_trajectories,
        p2_trajectories,
        targets,
        valid_mask,
        sigma_penalty_weight=config.training.sigma_penalty,
    )

    # Backward pass
    loss.backward()

    # Gradient clipping
    if config.training.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

    optimizer.step()
    scheduler.step()

    # Compute additional metrics without tracking gradients
    with torch.no_grad():
        extra = compute_batch_metrics(p1_trajectories, p2_trajectories, targets, valid_mask)

    return {
        'loss': loss.item(),
        'lr': scheduler.get_last_lr()[0],
        **extra,
    }


def compute_total_loss(
    p1_trajectories: list,
    p2_trajectories: list,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    sigma_penalty_weight: float = 0.0,
) -> torch.Tensor:
    """Compute total loss over all horizons.

    Args:
        p1_trajectories: List of P1 mixture params (one per horizon)
        p2_trajectories: List of P2 mixture params
        targets: [B, n_horizons, 4] ground truth deltas
        valid_mask: [B, n_horizons] validity mask

    Returns:
        Scalar loss (averaged over horizons and players)
    """
    loss = 0
    n_horizons = len(p1_trajectories)
    
    # targets shape: [B, n_horizons, 4] -> p1_dx, p1_dy, p2_dx, p2_dy
    sigma_pen = 0.0
    
    for i in range(n_horizons):
        loss += mixture_nll_loss(
            p1_trajectories[i], targets[:, i, :2], valid_mask[:, i]
        )
        loss += mixture_nll_loss(
            p2_trajectories[i], targets[:, i, 2:4], valid_mask[:, i]
        )
        if sigma_penalty_weight > 0:
            sigma_pen += compute_sigma_penalty(
                p1_trajectories[i],
                p2_trajectories[i],
                valid_mask[:, i],
            )
    
    # Average over horizons and players (2 players)
    loss = loss / (2 * n_horizons)
    if sigma_penalty_weight > 0 and n_horizons > 0:
        loss = loss + sigma_penalty_weight * (sigma_pen / n_horizons)
    return loss


def compute_sigma_penalty(
    p1_mix: dict,
    p2_mix: dict,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Penalty to discourage collapsing sigmas."""
    def masked_inv_sigma(mix: dict) -> torch.Tensor:
        sigma = torch.cat([mix['sigma_x'], mix['sigma_y']], dim=-1)  # [B, 2K]
        if mask.any():
            sigma = sigma[mask]
        return (1.0 / sigma).mean()

    return masked_inv_sigma(p1_mix) + masked_inv_sigma(p2_mix)


@torch.no_grad()
def compute_batch_metrics(
    p1_trajectories: list,
    p2_trajectories: list,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict:
    """Compute quick diagnostic metrics for a batch."""
    n_horizons = len(p1_trajectories)

    def mode_prediction(mixture: dict) -> torch.Tensor:
        """Take argmax component as point prediction."""
        idx = mixture['weights'].argmax(dim=-1)  # [B]
        mu_x = mixture['mu_x'].gather(1, idx.unsqueeze(1)).squeeze(1)
        mu_y = mixture['mu_y'].gather(1, idx.unsqueeze(1)).squeeze(1)
        return torch.stack([mu_x, mu_y], dim=-1)  # [B, 2]

    mae_sum = 0.0
    rmse_sum = 0.0
    mae_stage_sum = 0.0
    rmse_stage_sum = 0.0
    count = 0.0
    sigma_vals = []
    entropy_vals = []
    per_horizon = {}

    for i in range(n_horizons):
        mask = valid_mask[:, i]
        if mask.any():
            p1_pred = mode_prediction(p1_trajectories[i])
            p2_pred = mode_prediction(p2_trajectories[i])

            p1_err = (p1_pred - targets[:, i, :2]).pow(2).sum(dim=-1).sqrt()
            p2_err = (p2_pred - targets[:, i, 2:4]).pow(2).sum(dim=-1).sqrt()

            p1_mae = (p1_pred - targets[:, i, :2]).abs().sum(dim=-1)
            p2_mae = (p2_pred - targets[:, i, 2:4]).abs().sum(dim=-1)

            mae_sum += (p1_mae + p2_mae)[mask].sum()
            rmse_sum += (p1_err + p2_err)[mask].sum()

            # De-normalize for stage-unit diagnostics (x*stage_width, y*stage_height)
            stage_scale = torch.tensor([STAGE_HALF_WIDTH, STAGE_HALF_HEIGHT], device=targets.device, dtype=targets.dtype)
            p1_stage_err = ((p1_pred - targets[:, i, :2]) * stage_scale).pow(2).sum(dim=-1).sqrt()
            p2_stage_err = ((p2_pred - targets[:, i, 2:4]) * stage_scale).pow(2).sum(dim=-1).sqrt()
            p1_stage_mae = ((p1_pred - targets[:, i, :2]).abs() * stage_scale).sum(dim=-1)
            p2_stage_mae = ((p2_pred - targets[:, i, 2:4]).abs() * stage_scale).sum(dim=-1)

            mae_stage_sum += (p1_stage_mae + p2_stage_mae)[mask].sum()
            rmse_stage_sum += (p1_stage_err + p2_stage_err)[mask].sum()
            count += 2 * mask.sum()

            sigma_vals.append(torch.cat([p1_trajectories[i]['sigma_x'], p1_trajectories[i]['sigma_y'],
                                         p2_trajectories[i]['sigma_x'], p2_trajectories[i]['sigma_y']], dim=1)[mask].mean())
            # Mixture weight entropy per sample, averaged (masked)
            p1_w = p1_trajectories[i]['weights'][mask]
            p2_w = p2_trajectories[i]['weights'][mask]
            if p1_w.numel() > 0:
                entropy_vals.append((-(p1_w * (p1_w + 1e-8).log()).sum(dim=-1)).mean())
            if p2_w.numel() > 0:
                entropy_vals.append((-(p2_w * (p2_w + 1e-8).log()).sum(dim=-1)).mean())

            # Per-horizon metrics
            h_count = mask.sum().item() * 2  # two players
            per_horizon[i] = {
                'mae_norm': float((p1_mae[mask].sum() + p2_mae[mask].sum()) / max(h_count, 1)),
                'rmse_norm': float((p1_err[mask].sum() + p2_err[mask].sum()) / max(h_count, 1)),
                'mae_stage': float((p1_stage_mae[mask].sum() + p2_stage_mae[mask].sum()) / max(h_count, 1)),
                'rmse_stage': float((p1_stage_err[mask].sum() + p2_stage_err[mask].sum()) / max(h_count, 1)),
                'sigma_mean': float(torch.cat([
                    p1_trajectories[i]['sigma_x'][mask],
                    p1_trajectories[i]['sigma_y'][mask],
                    p2_trajectories[i]['sigma_x'][mask],
                    p2_trajectories[i]['sigma_y'][mask],
                ], dim=1).mean()),
            }

    mae = (mae_sum / count).item() if count > 0 else 0.0
    rmse = (rmse_sum / count).item() if count > 0 else 0.0
    sigma_mean = torch.stack(sigma_vals).mean().item() if sigma_vals else 0.0
    entropy_mean = torch.stack(entropy_vals).mean().item() if entropy_vals else 0.0
    mae_stage = (mae_stage_sum / count).item() if count > 0 else 0.0
    rmse_stage = (rmse_stage_sum / count).item() if count > 0 else 0.0

    return {
        'mae_norm': mae,
        'rmse_norm': rmse,
        'mae_stage': mae_stage,
        'rmse_stage': rmse_stage,
        'sigma_mean': sigma_mean,
        'weight_entropy': entropy_mean,
        'per_horizon': per_horizon,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
    loss: float,
    config: Config,
    checkpoint_dir: Path,
    prefix: str = "checkpoint"
) -> None:
    """Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        step: Current training step
        loss: Current loss
        config: Configuration
        checkpoint_dir: Directory to save checkpoint
        prefix: Filename prefix
    """
    checkpoint_path = checkpoint_dir / f"{prefix}_{step}.pt"
    
    state = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': config.to_dict(),
    }
    torch.save(state, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path} (step={step}, loss={loss:.4f})")
    
    # Save config JSON
    config.save(checkpoint_dir / "config.json")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> dict:
    """Load checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state

    Returns:
        Dict with metadata (step, loss, etc.)
    """
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict']

    # Allow partial load and handle stage embedding size changes gracefully
    stage_reused = False
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        # Strip stage_embed if shape mismatch, then retry
        if 'stage_embed' in str(exc):
            filtered = {k: v for k, v in state_dict.items() if not k.startswith('feature_embedder.stage_embed')}
            logger.warning("Stage embedding shape mismatch; loading remaining weights and reinitializing stage embedding.")
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            # Reuse overlapping rows from old embedding if available
            old_weight = state_dict.get('feature_embedder.stage_embed.weight')
            if old_weight is not None:
                new_weight = model.feature_embedder.stage_embed.weight.data
                rows_to_copy = min(new_weight.shape[0], old_weight.shape[0])
                new_weight[:rows_to_copy].copy_(old_weight[:rows_to_copy])
                stage_reused = True
        else:
            raise

    if missing or unexpected:
        logger.warning(f"Checkpoint load with non-strict match. Missing: {missing}, Unexpected: {unexpected}")
    if stage_reused:
        logger.info(f"Stage embedding rows reused: {rows_to_copy}/{old_weight.shape[0]}")

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    return {
        'step': checkpoint.get('step', 0),
        'epoch': checkpoint.get('epoch', 0), # In case we save epoch
        'loss': checkpoint.get('loss', 0.0),
    }


def log_metrics(metrics: dict, step: int) -> None:
    """Log training metrics.

    Args:
        metrics: Dict of metrics to log
        step: Training step
    """
    msg_parts = [
        f"Step {step}",
        f"loss={metrics.get('loss'):.4f}",
        f"lr={metrics.get('lr'):.2e}",
    ]
    if 'mae_norm' in metrics:
        msg_parts.append(f"mae_norm={metrics['mae_norm']:.4f}")
    if 'rmse_norm' in metrics:
        msg_parts.append(f"rmse_norm={metrics['rmse_norm']:.4f}")
    if 'mae_stage' in metrics:
        msg_parts.append(f"mae_stage={metrics['mae_stage']:.2f}")
    if 'rmse_stage' in metrics:
        msg_parts.append(f"rmse_stage={metrics['rmse_stage']:.2f}")
    if 'sigma_mean' in metrics:
        msg_parts.append(f"sigma_mean={metrics['sigma_mean']:.2f}")
    if 'weight_entropy' in metrics:
        msg_parts.append(f"w_entropy={metrics['weight_entropy']:.3f}")
    logger.info(", ".join(msg_parts))
    # Log per-horizon summary in compact form if present
    if 'per_horizon' in metrics and metrics['per_horizon']:
        ph = metrics['per_horizon']
        parts = []
        for h, vals in ph.items():
            parts.append(
                f"h{h}:mae_norm={vals['mae_norm']:.3f},rmse_norm={vals['rmse_norm']:.3f},"
                f"mae_stage={vals['mae_stage']:.2f},rmse_stage={vals['rmse_stage']:.2f},"
                f"sigma={vals['sigma_mean']:.2f}"
            )
        logger.info(" | ".join(parts))


if __name__ == '__main__':
    """Command-line interface for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train future position prediction model.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing processed NPZ data")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save checkpoints")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--resume-from", type=Path, default=None, help="Checkpoint to resume from")
    parser.add_argument("--device", type=str, default="auto", help="Device to use: auto, cuda, mps, or cpu")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit number of episodes to load (useful for quick experiments).",
    )

    args = parser.parse_args()

    # Create config from args
    config = Config()
    config.data.output_dir = args.data_dir # Using data-dir as source, assumes preprocessed
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.max_epochs = args.epochs
    config.training.device = args.device
    config.training.max_episodes = args.max_episodes
    # Separate checkpoint destination
    config.training.checkpoint_dir = args.output_dir

    train(config, resume_from=args.resume_from)
