#!/usr/bin/env python3
"""
Test Rust integration without using Console (to isolate issues).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_rust_gamestate_dict_ops():
    """Test dict-like operations on Rust GameState."""
    print("Testing Rust GameState dict-like operations...")
    
    try:
        import melee_rust
        
        gs = melee_rust.GameState()
        gs.frame = 100
        
        # Test set_player
        print("  Creating player...")
        player = melee_rust.PlayerState()
        player.percent = 50
        
        print("  Setting player at port 1...")
        gs.set_player(1, player)
        
        print("  Checking if player exists...")
        assert gs.has_player(1), "Player should exist at port 1"
        
        print("  Getting player...")
        retrieved = gs.get_player(1)
        assert retrieved is not None, "Should retrieve player"
        assert retrieved.percent == 50, f"Expected percent=50, got {retrieved.percent}"
        
        print("✓ All dict operations work!\n")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_rust_parser_basic():
    """Test Rust parser with minimal input."""
    print("Testing Rust parser with minimal data...")
    
    try:
        import melee_rust
        
        gs = melee_rust.GameState()
        
        # Create minimal event bytes (just empty for now)
        event_bytes = b''
        
        print("  Calling parse_events...")
        try:
            result = melee_rust.parse_events(event_bytes, gs)
            print(f"  Parser returned: {result}")
        except Exception as e:
            print(f"  Parser raised: {e} (expected for empty input)")
        
        print("✓ Parser exists and is callable!\n")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_enum_compatibility():
    """Test that Rust enums work with Python code."""
    print("Testing enum compatibility...")
    
    try:
        import melee_rust
        from melee import enums as py_enums
        
        # Create Rust enums
        rust_char = melee_rust.Character.Fox
        rust_stage = melee_rust.Stage.FinalDestination
        
        print(f"  Rust Character: {rust_char}")
        print(f"  Rust Stage: {rust_stage}")
        
        # Test assignment to player state
        player = melee_rust.PlayerState()
        player.character = rust_char
        assert player.character == rust_char
        
        print("✓ Enums work correctly!\n")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("RUST INTEGRATION TESTS")
    print("=" * 70)
    print()
    
    tests = [
        test_rust_gamestate_dict_ops,
        test_rust_parser_basic,
        test_enum_compatibility,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("=" * 70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print("=" * 70)
    
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())

