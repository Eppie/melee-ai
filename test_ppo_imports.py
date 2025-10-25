#!/usr/bin/env python3
"""Quick test to verify PPO implementation imports correctly."""


def test_imports():
    """Test that all PPO components can be imported."""
    print("Testing PPO imports...")

    try:
        print("  ✓ Importing config...")
        from config import PPOConfig, get_config, init_config

        print("  ✓ Importing trajectory...")
        from ppo.trajectory import Step, Trajectory, TrajectoryBuffer

        print("  ✓ Importing opponent_pool...")
        from ppo.opponent_pool import OpponentPool

        print("  ✓ Importing ppo_loss...")
        from ppo.ppo_loss import (
            compute_log_probs,
            compute_entropy,
            compute_ppo_loss,
            compute_value_loss,
            compute_total_ppo_loss,
        )

        print("  ✓ Importing selfplay_env...")
        from ppo.selfplay_env import SelfPlayEnvironment

        print("\n✓ All imports successful!")
        return True

    except Exception as e:
        print(f"\n✗ Import failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_config():
    """Test that PPO config works."""
    print("\nTesting PPO configuration...")

    try:
        from config import init_config, get_config

        # Initialize config
        init_config(freeze=False)
        config = get_config()

        # Check PPO config exists
        assert hasattr(config, "ppo"), "Config missing 'ppo' attribute"

        ppo_cfg = config.ppo
        print(f"  ✓ Pool size: {ppo_cfg.pool_size}")
        print(f"  ✓ Clip ratio: {ppo_cfg.clip_ratio}")
        print(f"  ✓ Entropy coef: {ppo_cfg.entropy_coef}")
        print(f"  ✓ GAE lambda: {ppo_cfg.gae_lambda}")
        print(f"  ✓ PPO epochs: {ppo_cfg.ppo_epochs}")
        print(f"  ✓ Minibatch size: {ppo_cfg.minibatch_size}")
        print(f"  ✓ Max episode frames: {ppo_cfg.max_episode_frames}")

        print("\n✓ Configuration successful!")
        return True

    except Exception as e:
        print(f"\n✗ Configuration failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_trajectory():
    """Test trajectory and GAE computation."""
    print("\nTesting trajectory computation...")

    try:
        import torch
        from ppo.trajectory import Step, Trajectory

        # Create a simple trajectory
        steps = []
        for i in range(10):
            step = Step(
                state=torch.randn(130),
                action_logits={"main_stick": torch.randn(64)},
                action_taken={"main_stick": torch.tensor(0)},
                log_prob=torch.tensor(-2.0),
                value=torch.tensor(0.5),
                reward=0.01 * i,
                done=(i == 9),
            )
            steps.append(step)

        traj = Trajectory(steps=steps)
        print(f"  ✓ Created trajectory with {len(traj)} steps")

        # Compute GAE
        traj.compute_gae(gamma=0.99, gae_lambda=0.95, normalize=True)
        print(f"  ✓ Computed GAE advantages: shape {traj.advantages.shape}")
        print(f"  ✓ Computed returns: shape {traj.returns.shape}")

        print("\n✓ Trajectory computation successful!")
        return True

    except Exception as e:
        print(f"\n✗ Trajectory computation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("PPO Implementation Verification")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Trajectory", test_trajectory),
    ]

    results = []
    for name, test_fn in tests:
        success = test_fn()
        results.append((name, success))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(success for _, success in results)

    if all_passed:
        print("\n🎉 All tests passed! PPO implementation is ready.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
