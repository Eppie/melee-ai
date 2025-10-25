# PPO Self-Play Training

This directory contains the implementation of Proximal Policy Optimization (PPO) for self-play training of the Melee AI.

## Overview

The PPO implementation includes:

1. **Opponent Pool**: Maintains a pool of past model versions to train against
2. **Self-Play Environment**: Manages two players (learner vs opponent) in libmelee
3. **Trajectory Collection**: Stores experiences during gameplay
4. **GAE (Generalized Advantage Estimation)**: Computes advantages for policy updates
5. **PPO Loss**: Clipped surrogate objective with value function and entropy bonus

## Quick Start

### Prerequisites

- Dolphin Slippi (Ishiiruka build) installed
- Melee ISO file
- Trained initial checkpoint (optional)

### Running PPO Training

```bash
python train_ppo.py \
    --dolphin-path /path/to/dolphin-emu \
    --iso /path/to/melee.iso \
    --checkpoint checkpoints/model_ep013_060001.pt \
    --num-episodes 1000 \
    --save-every 10 \
    --add-to-pool-every 5 \
    --out-dir checkpoints/ppo
```

### Configuration

PPO parameters can be configured in `config.py` under the `PPOConfig` class:

- `pool_size`: Number of opponents to keep (default: 5)
- `clip_ratio`: PPO clipping epsilon (default: 0.2)
- `entropy_coef`: Entropy bonus coefficient (default: 0.01)
- `gae_lambda`: GAE lambda parameter (default: 0.95)
- `ppo_epochs`: Training epochs per episode (default: 4)
- `minibatch_size`: Minibatch size for updates (default: 64)
- `max_episode_frames`: Max frames per episode (default: 18000)

You can override these via command line:

```bash
python train_ppo.py ... --set ppo.pool_size=10 --set ppo.clip_ratio=0.15
```

## How It Works

### Episode Flow

1. **Episode Start**: 
   - Random opponent is selected from the pool
   - Opponent model is loaded with frozen weights
   - Buffers are cleared

2. **Gameplay**:
   - Both players act based on model outputs
   - Learner samples actions (exploration)
   - Opponent uses deterministic policy
   - Experiences stored in trajectory buffer
   - Rewards computed per-frame

3. **Episode End**:
   - Trajectory buffer finalized
   - GAE advantages computed
   - PPO training performed
   - Model optionally added to opponent pool

### Reward Structure

Rewards are computed using the existing reward system from `train/value_head.py`:

- **Damage**: +0.01 per % dealt, -0.01 per % taken
- **Stocks**: +0.3 for taking stock, -0.3 for losing stock
- **Hitlag**: +0.02 when hitting opponent, -0.02 when being hit
- **Shield**: Penalty for low shield (scales with depletion)
- **Per-frame**: Small constant penalty (-0.001) to discourage stalling

### PPO Loss

Total loss = Policy Loss + Value Loss + Entropy Bonus

1. **Policy Loss** (clipped surrogate):
   ```
   L_policy = -E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)]
   where ratio = π_new(a|s) / π_old(a|s)
   ```

2. **Value Loss** (MSE):
   ```
   L_value = E[(V(s) - R)²]
   ```

3. **Entropy Bonus**:
   ```
   L_entropy = -β * E[H(π(·|s))]
   ```

## File Structure

```
ppo/
├── __init__.py              # Module init
├── opponent_pool.py         # Opponent pool management
├── trajectory.py            # Trajectory collection & GAE
├── ppo_loss.py             # PPO loss computation
├── selfplay_env.py         # Self-play environment wrapper
└── README.md               # This file

train_ppo.py                # Main training script
```

## Monitoring

Training metrics are logged to Weights & Biases:

- **Episode metrics**: frames, rewards, stocks, damage
- **Policy metrics**: entropy, clipping fraction, ratio statistics
- **Value metrics**: value predictions, errors
- **Pool metrics**: pool size, opponent statistics

## Tips

### Starting from Scratch

If you don't have a trained model:
1. First train with supervised learning (use `train.py`)
2. Then switch to PPO for self-play improvement

### Pool Size

- Larger pools (10-20) provide more diverse opponents but slower adaptation
- Smaller pools (3-5) adapt quickly but risk overfitting to recent strategies

### Hyperparameter Tuning

Key parameters to tune:
- `clip_ratio`: Higher (0.3) = more aggressive updates, Lower (0.1) = more conservative
- `entropy_coef`: Higher = more exploration, Lower = more exploitation
- `gae_lambda`: Higher (0.99) = less bias but more variance, Lower (0.9) = more bias but less variance

### Episode Length

- Shorter episodes (5000 frames): Faster iteration, less diverse data
- Longer episodes (20000 frames): More diverse, but slower iteration

## Troubleshooting

### "Opponent pool is empty"

The system will use self-play (current model vs itself) until the pool has at least one model. This is normal on first run.

### Game not starting

Check that:
- Dolphin path is correct
- ISO path is valid
- No other Dolphin instances are running

### Out of memory

Reduce:
- `ppo.minibatch_size`
- `max_episode_frames`
- `model.seq_len`

### Poor performance

- Check reward structure aligns with desired behavior
- Verify advantage normalization is enabled
- Try adjusting `gae_lambda` for better credit assignment
- Increase exploration with higher `entropy_coef`

