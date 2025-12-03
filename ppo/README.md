# PPO Reinforcement Learning System

Hierarchical PPO training system for Melee bot with self-play and opponent diversity.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Coordinator (CRD)                          │
│                    [1 GPU Process]                            │
│                                                               │
│  • Batched inference (96 envs)                               │
│  • PPO training loop                                          │
│  • Opponent pool management                                   │
│  • Checkpoint saving                                          │
└───────────────────┬──────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┬─────────────┐
        │                       │             │
┌───────▼────────┐    ┌────────▼──────┐     ...  (12 shards)
│  ArenaShard 0  │    │ ArenaShard 1  │
│  [CPU Process] │    │  [CPU Process]│
│                │    │               │
│  8 ENV threads │    │  8 ENV threads│
│  8 Dolphins    │    │  8 Dolphins   │
└────────────────┘    └───────────────┘

Total: 1 CRD + 12 S8 = 96 Dolphin environments
```

## Features

### Self-Play with Opponent Diversity
- **80% Historical Opponents**: Sample from pool of past checkpoints
- **20% Self-Play**: Current policy plays against itself
- Opponent reassignment every rollout (1024 frames)

### Zero-Copy Data Flow
- Shared memory slabs for features and actions
- Pinned host memory for async H2D transfers
- Incremental updates (only new frame column)

### PPO Training
- GAE (λ=0.95) for advantage estimation
- Clipped objective (ε=0.2)
- Value function bootstrap
- Entropy bonus for exploration

## Quick Start

### 1. Train with Imitation Learning First

Before PPO, you need a good initialization:

```bash
# Train on Slippi replays first
python train.py --set train.epochs=100
```

### 2. Launch PPO Training

```bash
# Full scale (96 environments)
python train_ppo_rl.py --init-checkpoint checkpoints/latest.pt

# Small scale for testing (16 environments)
python train_ppo_rl.py \
    --init-checkpoint checkpoints/latest.pt \
    --num-shards 2 \
    --envs-per-shard 8
```

### 3. Monitor Training

Training will print:
- FPS (frames per second across all environments)
- PPO loss metrics (policy, value, entropy)
- Importance sampling ratios
- Checkpoint saves

### 4. Stop Training

Press `Ctrl+C` for graceful shutdown. The coordinator will:
1. Signal all shards to stop
2. Wait for processes to exit
3. Close shared memory
4. Save final checkpoint

## Configuration

### Command Line Arguments

```bash
python train_ppo_rl.py --help
```

Key options:
- `--init-checkpoint`: Initial policy (required)
- `--num-shards`: Number of S8 processes (default: 12)
- `--envs-per-shard`: Dolphins per shard (default: 8)
- `--lr`: Learning rate (default: 3e-4)
- `--opponent-pool-size`: Historical checkpoints to keep (default: 20)
- `--opponent-sample-prob`: Probability of historical opponent (default: 0.8)
- `--checkpoint-interval`: Steps between saves (default: 10)

### File Structure

```
ppo/
├── config.py              # PPOConfig with all hyperparameters
├── coordinator.py         # GPU process (inference + training)
├── arena_shard.py         # S8 process (manages 8 ENV threads)
├── env_worker.py          # ENV thread (single Dolphin lifecycle)
├── shared_memory.py       # Zero-copy data structures
├── opponent.py            # Opponent pool & matchmaking
├── rollout.py             # Rollout buffer & GAE computation
├── ppo_loss.py            # PPO loss functions
├── ipc.py                 # IPC primitives
└── tests/                 # Comprehensive test suite
```

## Performance

### Expected Throughput

With 96 environments on M2 Max (MPS):
- **Inference**: ~5-10ms per step (batched)
- **Environment**: ~16ms per frame (60 FPS target)
- **Throughput**: ~5000-10000 frames/sec total

Bottleneck is typically Dolphin emulation, not inference.

### Memory Usage

- **Shared Memory**: ~7.4 MiB per shard × 12 = ~90 MiB
- **GPU Ring Buffer**: 96 × 256 × 908 × 2 bytes (bf16) = ~44 MiB
- **Model**: ~50-200 MiB depending on size
- **Total**: ~200-400 MiB (very efficient!)

## Troubleshooting

### "Checkpoint not found"
Provide a valid checkpoint from imitation learning:
```bash
python train_ppo_rl.py --init-checkpoint checkpoints/checkpoint_epoch_100.pt
```

### "Failed to connect to Dolphin"
Check Dolphin path and ISO path:
```bash
python train_ppo_rl.py \
    --dolphin-path "/Applications/Slippi Dolphin.app" \
    --iso-path "~/Documents/SSBM.iso"
```

### Shards dying
Check memory usage and Dolphin stability. Reduce scale:
```bash
python train_ppo_rl.py --num-shards 4 --envs-per-shard 4
```

### Slow FPS
- Check if Dolphin is in headless mode
- Reduce number of environments
- Profile with `python -m cProfile`

## Development

### Running Tests

```bash
# All PPO tests
pytest ppo/tests/ -v

# Specific test file
pytest ppo/tests/test_opponent_pool.py -v

# Integration test (requires Dolphin)
pytest ppo/tests/test_single_s8.py -v
```

### Adding New Features

1. **New config parameter**: Add to `ppo/config.py`
2. **New component**: Create module in `ppo/`
3. **Tests**: Add to `ppo/tests/`
4. **Documentation**: Update this README

## FAQ

**Q: How long should I train?**
A: Start with 10-100k training steps. Monitor policy loss and value estimates.

**Q: What's a good learning rate?**
A: Start with 3e-4. If ratios diverge, reduce to 1e-4.

**Q: Should I use more environments?**
A: More environments = more diverse data but higher memory. 96 is a good balance.

**Q: How often should checkpoints save?**
A: Every 10-100 training steps. Frequent saves → better opponent diversity.

**Q: Can I resume training?**
A: Not yet implemented. Checkpoints save state but load logic is TODO.

## References

- [Proximal Policy Optimization (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [Generalized Advantage Estimation (Schulman et al., 2016)](https://arxiv.org/abs/1506.02438)
- [Emergent Complexity via Multi-Agent Competition (Bansal et al., 2018)](https://arxiv.org/abs/1710.03748)
