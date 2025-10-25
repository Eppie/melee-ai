#!/bin/bash
# Example script for running PPO training
#
# Usage: ./run_ppo.sh
#
# Note: Parallel trajectory collection is configured in config.py (ppo.num_workers)
# Set num_workers to the number of CPU cores for parallel collection (e.g., 8 for an 8-core machine)
# Each worker runs a separate Dolphin instance and collects one trajectory in parallel

# Configuration
DOLPHIN_PATH="${DOLPHIN_PATH:-/home/eppie/slippi-Ishiiruka/build/Binaries/dolphin-emu}"
ISO_PATH="${ISO_PATH:-/home/eppie/melee-ai/melee.iso}"
CHECKPOINT="${CHECKPOINT:-checkpoints/model_ep010_065002.pt}"
OUT_DIR="checkpoints/ppo"

# PPO parameters
NUM_EPISODES=1000
SAVE_EVERY=10
ADD_TO_POOL_EVERY=5

# Check if paths exist
if [ ! -f "$DOLPHIN_PATH" ]; then
    echo "Error: Dolphin executable not found at: $DOLPHIN_PATH"
    echo "Set DOLPHIN_PATH environment variable or edit this script"
    exit 1
fi

if [ ! -f "$ISO_PATH" ]; then
    echo "Error: Melee ISO not found at: $ISO_PATH"
    echo "Set ISO_PATH environment variable or edit this script"
    exit 1
fi

# Optional: check checkpoint
if [ ! -z "$CHECKPOINT" ] && [ ! -f "$CHECKPOINT" ]; then
    echo "Warning: Checkpoint not found at: $CHECKPOINT"
    echo "Will start training from scratch"
    CHECKPOINT=""
fi

# Build command
CMD="python train_ppo.py \
    --dolphin-path '$DOLPHIN_PATH' \
    --iso '$ISO_PATH' \
    --num-episodes $NUM_EPISODES \
    --save-every $SAVE_EVERY \
    --add-to-pool-every $ADD_TO_POOL_EVERY \
    --out-dir '$OUT_DIR'"

# Add checkpoint if provided
if [ ! -z "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint '$CHECKPOINT'"
fi

echo "Starting PPO training..."
echo "Command: $CMD"
echo ""

# Run training
eval $CMD

