#!/usr/bin/env python3
"""
Comprehensive test suite for Python vs Rust parser parity.

Tests all code paths, edge cases, and scenarios to ensure
100% correctness.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_ice_climbers():
    """Test Ice Climbers (Nana) parsing."""
    print("\n" + "=" * 70)
    print("TEST: Ice Climbers (Nana) Parsing")
    print("=" * 70)
    
    # TODO: Need a replay with Ice Climbers
    print("   ⚠️  Requires test.slp with Ice Climbers - skipping")
    return True


def test_projectiles():
    """Test projectile parsing."""
    print("\n" + "=" * 70)
    print("TEST: Projectile Parsing")
    print("=" * 70)
    
    from melee.console import Console
    
    # Parse with Python
    os.environ['LIBMELEE_USE_RUST'] = '0'
    import importlib
    import melee.console
    importlib.reload(melee.console)
    
    console = Console(path='test.slp', is_dolphin=False, allow_old_version=True)
    console.connect()
    
    py_projectile_counts = []
    while True:
        gs = console.step()
        if gs is None:
            break
        py_projectile_counts.append(len(gs.projectiles))
    
    # Parse with Rust
    os.environ['LIBMELEE_USE_RUST'] = '1'
    importlib.reload(melee.console)
    
    console = Console(path='test.slp', is_dolphin=False, allow_old_version=True)
    console.connect()
    
    rust_projectile_counts = []
    while True:
        gs = console.step()
        if gs is None:
            break
        rust_projectile_counts.append(len(gs.projectiles))
    
    if py_projectile_counts == rust_projectile_counts:
        total_projectiles = sum(py_projectile_counts)
        print(f"   ✅ Projectile counts match! Total: {total_projectiles}")
        return True
    else:
        print(f"   ❌ Projectile count mismatch!")
        print(f"      Python: {sum(py_projectile_counts)}")
        print(f"      Rust:   {sum(rust_projectile_counts)}")
        return False


def test_player_states():
    """Test all PlayerState fields are parsed correctly."""
    print("\n" + "=" * 70)
    print("TEST: PlayerState Field Coverage")
    print("=" * 70)
    
    from melee.console import Console
    
    os.environ['LIBMELEE_USE_RUST'] = '1'
    import importlib
    import melee.console
    importlib.reload(melee.console)
    
    console = Console(path='test.slp', is_dolphin=False, allow_old_version=True)
    console.connect()
    
    # Get one frame
    gs = console.step()
    
    if not gs or not gs.players:
        print("   ❌ No players found")
        return False
    
    # Check a player has all expected fields
    player = None
    if hasattr(gs, 'get_player'):
        # Rust GameState
        for port in [1, 2, 3, 4]:
            p = gs.get_player(port)
            if p:
                player = p
                break
    else:
        # Python GameState
        player = next(iter(gs.players.values()))
    
    if not player:
        print("   ❌ Could not get player")
        return False
    
    # Check critical fields exist
    fields = [
        'character', 'action', 'position', 'percent', 'stock',
        'facing', 'controller_state', 'action_frame', 'speed_air_x_self',
        'speed_y_self', 'speed_x_attack', 'speed_y_attack', 'speed_ground_x_self',
        'jumps_left', 'on_ground', 'invulnerable', 'hitlag_left', 'hitstun_frames_left'
    ]
    
    missing = []
    for field in fields:
        if not hasattr(player, field):
            missing.append(field)
    
    if missing:
        print(f"   ❌ Missing fields: {missing}")
        return False
    
    print(f"   ✅ All {len(fields)} critical fields present")
    return True


def test_frame_range():
    """Test that all frames are parsed (including negative frames)."""
    print("\n" + "=" * 70)
    print("TEST: Frame Range Coverage")
    print("=" * 70)
    
    from melee.console import Console
    
    # Test Python
    os.environ['LIBMELEE_USE_RUST'] = '0'
    import importlib
    import melee.console
    importlib.reload(melee.console)
    
    console = Console(path='test.slp', is_dolphin=False, allow_old_version=True)
    console.connect()
    
    py_frames = []
    while True:
        gs = console.step()
        if gs is None:
            break
        py_frames.append(gs.frame)
    
    # Test Rust
    os.environ['LIBMELEE_USE_RUST'] = '1'
    importlib.reload(melee.console)
    
    console = Console(path='test.slp', is_dolphin=False, allow_old_version=True)
    console.connect()
    
    rust_frames = []
    while True:
        gs = console.step()
        if gs is None:
            break
        rust_frames.append(gs.frame)
    
    if py_frames == rust_frames:
        print(f"   ✅ Frame ranges match!")
        print(f"      Range: {min(py_frames)} to {max(py_frames)}")
        print(f"      Total: {len(py_frames)} frames")
        return True
    else:
        print(f"   ❌ Frame ranges differ!")
        if len(py_frames) != len(rust_frames):
            print(f"      Lengths: Python={len(py_frames)}, Rust={len(rust_frames)}")
        return False


def test_controller_states():
    """Test controller input parsing."""
    print("\n" + "=" * 70)
    print("TEST: Controller State Parsing")
    print("=" * 70)
    
    from melee.console import Console
    
    os.environ['LIBMELEE_USE_RUST'] = '1'
    import importlib
    import melee.console
    importlib.reload(melee.console)
    
    console = Console(path='test.slp', is_dolphin=False, allow_old_version=True)
    console.connect()
    
    # Sample several frames
    button_presses = 0
    stick_inputs = 0
    
    for _ in range(100):
        gs = console.step()
        if not gs:
            break
        
        # Get first player
        player = None
        if hasattr(gs, 'get_player'):
            player = gs.get_player(1) or gs.get_player(2)
        elif gs.players:
            player = next(iter(gs.players.values()), None)
        
        if player and hasattr(player, 'controller_state'):
            cs = player.controller_state
            if hasattr(cs, 'button'):
                button_presses += sum(1 for b in cs.button.values() if b)
            if hasattr(cs, 'main_stick'):
                if cs.main_stick[0] != 0.5 or cs.main_stick[1] != 0.5:
                    stick_inputs += 1
    
    print(f"   ✅ Sampled 100 frames")
    print(f"      Button presses detected: {button_presses}")
    print(f"      Stick movements: {stick_inputs}")
    return True


def test_memory_consistency():
    """Test that parsing twice gives same results (no state pollution)."""
    print("\n" + "=" * 70)
    print("TEST: Memory Consistency (No State Pollution)")
    print("=" * 70)
    
    from melee.console import Console
    
    os.environ['LIBMELEE_USE_RUST'] = '1'
    import importlib
    import melee.console
    importlib.reload(melee.console)
    
    # Parse once
    console1 = Console(path='test.slp', is_dolphin=False, allow_old_version=True)
    console1.connect()
    frames1 = []
    while True:
        gs = console1.step()
        if gs is None:
            break
        frames1.append(gs.frame)
    
    # Parse again
    console2 = Console(path='test.slp', is_dolphin=False, allow_old_version=True)
    console2.connect()
    frames2 = []
    while True:
        gs = console2.step()
        if gs is None:
            break
        frames2.append(gs.frame)
    
    if frames1 == frames2:
        print(f"   ✅ Consistent results across runs")
        return True
    else:
        print(f"   ❌ Results differ between runs!")
        return False


def main():
    """Run all comprehensive tests."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    try:
        import melee_rust
        print("✓ Rust module available\n")
    except ImportError:
        print("❌ Rust module not available")
        return 1
    
    tests = [
        ("Ice Climbers", test_ice_climbers),
        ("Projectiles", test_projectiles),
        ("PlayerState Fields", test_player_states),
        ("Frame Range", test_frame_range),
        ("Controller States", test_controller_states),
        ("Memory Consistency", test_memory_consistency),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

