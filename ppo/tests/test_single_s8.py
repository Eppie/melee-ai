"""
Minimal test script for single S8 with 1 Dolphin.

This tests:
- EnvWorker initialization
- Dolphin subprocess management
- Feature extraction
- Shared memory communication
- Basic control loop

Run with:
    python -m ppo.tests.test_single_s8
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from column_map import ColumnMap
from config import Config
from ppo.config import PPOConfig
from ppo.env_worker import EnvWorker
from ppo.shared_memory import ActionData, SharedMemorySlab


def test_single_env():
    """Test a single ENV worker without full S8 coordination."""
    print("=" * 60)
    print("Testing Single EnvWorker (1 Dolphin)")
    print("=" * 60)

    # Load config
    config = Config()

    # Create PPO config (minimal for testing)
    ppo_config = PPOConfig(
        num_shards=1,
        envs_per_shard=1,
        dolphin_path="/Applications/Slippi Dolphin.app",
        iso_path="~/Documents/SSBM.iso",
        character="FOX",
        stages=["FD"],
        bot_port=1,
        opp_port=2,
        warmup_frames=10,  # Short warmup for testing
        restart_interval=1000,  # Short for testing
        init_checkpoint=Path("checkpoints/dummy.pt"),  # Dummy for now
    )

    # Create column map
    column_map = ColumnMap(config)

    # Create shared memory slab
    print("\n[TEST] Creating shared memory slab...")
    slab = SharedMemorySlab(
        shard_id=0,
        envs_per_shard=1,
        context_length=ppo_config.context_length,
        create=True,
    )

    try:
        # Create ENV worker
        print("[TEST] Creating EnvWorker...")
        worker = EnvWorker(
            shard_id=0,
            env_id=0,
            shared_slab=slab,
            config=ppo_config,
            column_map=column_map,
        )

        print("\n[TEST] Starting Dolphin...")
        worker._start_dolphin()

        print("\n[TEST] Running test loop (10 frames)...")
        for i in range(10):
            print(f"\n--- Frame {i} ---")

            # Get gamestate
            gamestate = worker.console.step()
            if gamestate is None:
                print("[TEST] No gamestate yet, skipping")
                time.sleep(0.1)
                continue

            # Featurize
            features = worker._featurize(gamestate)
            print(f"[TEST] Features shape: {features.shape}")
            print(f"[TEST] Features mean: {features.mean():.4f}, std: {features.std():.4f}")

            # Write to shared memory
            slab.features[0, worker.t_mod, :] = features
            print(f"[TEST] Wrote features to ring position {worker.t_mod}")

            # Simulate CRD providing action (dummy action)
            dummy_action = ActionData(
                main_idx=0,  # Neutral
                c_idx=4,  # Center
                shoulder_idx=0,  # No shoulder
                buttons=np.array([False] * 5),
                logp=-1.0,
                value=0.0,
            )
            slab.write_action(0, dummy_action)
            print("[TEST] Wrote dummy action to shared memory")

            # Apply action
            worker._apply_action(dummy_action, is_ego=True)
            print("[TEST] Applied action to controller")

            # Update ring position
            worker.t_mod = (worker.t_mod + 1) % ppo_config.context_length
            worker.t_local += 1

            time.sleep(0.1)  # Slow down for observation

        print("\n" + "=" * 60)
        print("[TEST] SUCCESS: Basic control loop functional")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")
    except Exception as e:
        print(f"\n[TEST] ERROR: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\n[TEST] Cleaning up...")
        if worker.console:
            worker.console.stop()
        slab.close()
        slab.unlink()
        print("[TEST] Cleanup complete")


if __name__ == "__main__":
    test_single_env()
