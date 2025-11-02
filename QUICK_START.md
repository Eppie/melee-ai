# PPO Self-Play Quick Start Guide

## Installation Complete! ✓

Your PPO self-play training system is now fully implemented and ready to use.

## What You Got

### Core Components
1. **PPO Configuration** (`config.py`)
   - `PPOConfig` with all hyperparameters
   - Pool size: 5 (configurable)
   - Clip ratio: 0.2, GAE lambda: 0.95, entropy: 0.01

2. **Opponent Pool** (`ppo/opponent_pool.py`)
   - Manages up to 5 past model versions
   - Random sampling for diversity
   - Automatic FIFO removal when full

3. **Trajectory Collection** (`ppo/trajectory.py`)
   - Step-by-step experience storage
   - GAE (Generalized Advantage Estimation)
   - Advantage normalization

4. **PPO Loss** (`ppo/ppo_loss.py`)
   - Clipped surrogate objective
   - Multi-head action space (sticks, buttons, shoulder)
   - Value function loss
   - Entropy bonus

5. **Self-Play Environment** (`ppo/selfplay_env.py`)
   - Libmelee integration
   - Two-player control (learner vs opponent)
   - Reward computation
   - Episode management

6. **Training Script** (`train_ppo.py`)
   - Complete training loop
   - Wandb logging
   - Checkpointing
   - Pool management

## Quick Start (2 Steps)

### Step 1: Activate Your Environment

```bash
cd /Users/eppie/PycharmProjects/nano-melee
source venv/bin/activate  # or your virtualenv path
```

### Step 2: Start Training

```bash
python train_ppo.py \
    --dolphin-path /home/eppie/slippi-Ishiiruka/build/Binaries/dolphin-emu \
    --iso /path/to/melee.iso \
    --checkpoint checkpoints/model_ep013_060001.pt \
    --num-episodes 100 \
    --save-every 10 \
    --add-to-pool-every 5
```

Or use the helper script:

```bash
# Edit paths in run_ppo.sh first
./run_ppo.sh
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    PPO Training Loop                     │
└─────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐   ┌──────────────┐  ┌──────────┐
    │ Learner  │   │  Environment │  │ Opponent │
    │  Model   │◄──┤   (libmelee) │──►│   Pool  │
    │(training)│   │              │  │(frozen)  │
    └──────────┘   └──────────────┘  └──────────┘
          │                │                │
          └────────────────┼────────────────┘
                          ▼
                 ┌──────────────────┐
                 │   Trajectories   │
                 │  (experiences)   │
                 └──────────────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │  GAE + Returns   │
                 │  (advantages)    │
                 └──────────────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │    PPO Loss      │
                 │ Policy+Value+Ent │
                 └──────────────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Update Learner   │
                 │  Add to Pool     │
                 └──────────────────┘
```

## Episode Flow

```
1. Reset Episode
   ├─ Sample opponent from pool
   ├─ Load weights (frozen)
   └─ Clear buffers

2. Navigate to Game
   ├─ Select characters
   └─ Start match

3. Play Episode (max 18000 frames)
   ├─ Both players act
   ├─ Learner: explores (samples)
   ├─ Opponent: exploits (argmax)
   ├─ Compute rewards
   └─ Store trajectory

4. Episode Ends (stocks=0 or timeout)
   ├─ Compute GAE
   └─ Calculate returns

5. PPO Training (4 epochs)
   ├─ Shuffle data
   ├─ Minibatch updates
   ├─ Clip policy ratio
   └─ Update value function

6. Update Pool & Save
   ├─ Add model to pool (every 5 episodes)
   └─ Save checkpoint (every 10 episodes)
```

## Key Algorithms

### GAE (Generalized Advantage Estimation)

```python
# TD error
δ(t) = r(t) + γ·V(t+1) - V(t)

# GAE advantage (backward pass)
A(t) = δ(t) + (γ·λ)·δ(t+1) + (γ·λ)²·δ(t+2) + ...

# Returns for value target
R(t) = A(t) + V(t)
```

### PPO Loss

```python
# Ratio
ratio = π_new(a|s) / π_old(a|s)

# Clipped surrogate
L_clip = -min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)

# Value loss
L_value = (V - R)²

# Entropy bonus
L_entropy = -β·H(π)

# Total
L_total = L_clip + c₁·L_value + c₂·L_entropy
```

## Configuration

All parameters in `config.py`:

```python
ppo:
  pool_size: 5                 # opponent pool size
  clip_ratio: 0.2              # PPO clipping ε
  entropy_coef: 0.01           # entropy bonus β
  gae_lambda: 0.95             # GAE λ parameter
  ppo_epochs: 4                # training epochs per episode
  minibatch_size: 64           # minibatch size
  max_grad_norm: 0.5           # gradient clipping
  max_episode_frames: 18000    # ~5 min per episode
  normalize_advantages: true   # normalize advantages
  value_clip: null             # optional value clipping
```

Override via CLI:
```bash
--set ppo.pool_size=10
--set ppo.clip_ratio=0.15
```

## Monitoring

### Weights & Biases Dashboard

**Episode Metrics:**
- `episode/frames`: Number of frames played
- `episode/total_reward`: Cumulative reward
- `episode/learner_stocks`: Remaining stocks (learner)
- `episode/opponent_stocks`: Remaining stocks (opponent)
- `episode/learner_percent`: Final damage %
- `episode/opponent_percent`: Final damage %

**Training Metrics:**
- `ppo/policy_loss`: Policy gradient loss
- `ppo/value_loss`: Critic MSE loss
- `ppo/entropy`: Current policy entropy
- `ppo/ratio_mean`: Average policy ratio
- `ppo/clipped_fraction`: % of ratios clipped
- `ppo/advantages_mean`: Average advantage

**Pool Metrics:**
- `pool/size`: Current pool size
- `pool/total_created`: Total opponents created

## File Structure

```
nano-melee/
├── config.py               # ✨ PPOConfig added
├── train_ppo.py            # ✨ Main training script
├── run_ppo.sh              # ✨ Helper launcher
├── test_ppo_imports.py     # ✨ Verification script
├── QUICK_START.md          # ✨ This guide
├── PPO_IMPLEMENTATION.md   # ✨ Detailed docs
└── ppo/
    ├── __init__.py         # ✨ Module init
    ├── README.md           # ✨ Usage guide
    ├── opponent_pool.py    # ✨ Pool manager
    ├── trajectory.py       # ✨ Experience storage + GAE
    ├── ppo_loss.py         # ✨ Loss computation
    └── selfplay_env.py     # ✨ Environment wrapper
```

## Troubleshooting

### Import Errors
**Problem**: `ModuleNotFoundError: No module named 'torch'`  
**Solution**: Activate your virtual environment:
```bash
source venv/bin/activate
```

### Dolphin Not Found
**Problem**: `Failed to connect to console`  
**Solution**: Check paths:
```bash
--dolphin-path /correct/path/to/dolphin-emu
--iso /correct/path/to/SSBM.iso
```

### Pool Empty Warning
**Problem**: "Opponent pool is empty. Using self-play..."  
**Solution**: This is normal on first run! The system will add models to the pool automatically.

### Out of Memory
**Problem**: CUDA out of memory  
**Solution**: Reduce batch sizes:
```bash
--set ppo.minibatch_size=32
--set ppo.max_episode_frames=10000
```

### Slow Training
**Problem**: Episodes taking too long  
**Solution**: Check:
- Is fast-forward enabled? (should be)
- Are graphics disabled? (should be "Null" backend)
- Are you on the right device? (CUDA > MPS > CPU)

## Next Steps

1. ✅ **Verify setup**: `python test_ppo_imports.py`
2. ✅ **Start training**: `./run_ppo.sh` or `python train_ppo.py ...`
3. ✅ **Monitor progress**: Check Wandb dashboard
4. ✅ **Tune hyperparameters**: Adjust in `config.py`
5. ✅ **Evaluate**: Test against CPU or other models

## Expected Results

After 100-200 episodes, you should see:
- ✓ Increasing win rate against older opponents
- ✓ Stable policy ratios (close to 1.0)
- ✓ Gradually decreasing entropy (convergence)
- ✓ Diverse opponent pool (different strategies)
- ✓ Improved tech skill execution

## Documentation

- **This file**: Quick start guide
- **`ppo/README.md`**: Detailed usage & configuration
- **`PPO_IMPLEMENTATION.md`**: Full implementation details
- **Code**: All modules have extensive docstrings

## Support

The implementation follows your exact specifications:
- ✅ Self-play with opponent pool (configurable size)
- ✅ Player 1 = learner (current model)
- ✅ Player 2 = opponent (frozen from pool)
- ✅ Trajectory collection per frame
- ✅ GAE for advantage computation
- ✅ PPO loss (clipped + value + entropy)
- ✅ Update at end of each episode
- ✅ Pool management (add/remove oldest)
- ✅ Full integration with existing codebase

Everything is ready to go! Happy training! 🚀🎮

