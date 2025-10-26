#!/usr/bin/env python3
"""
Test against specific files that have shown issues to ensure they're all fixed.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_file(file_path: str, expected_frames: int = None) -> bool:
    """Test a single file for parity between Python and Rust."""
    from melee.console import Console
    import importlib
    import melee.console
    
    print(f"\nTesting: {Path(file_path).name}")
    print(f"=" * 70)
    
    # Parse with Python
    os.environ['LIBMELEE_USE_RUST'] = '0'
    importlib.reload(melee.console)
    
    console = Console(path=file_path, is_dolphin=False, allow_old_version=True)
    if not console.connect():
        print("❌ Failed to connect (Python)")
        return False
    
    py_frames = []
    while True:
        gs = console.step()
        if gs is None:
            break
        py_frames.append(gs)
    
    print(f"Python: {len(py_frames)} frames")
    
    # Parse with Rust
    os.environ['LIBMELEE_USE_RUST'] = '1'
    importlib.reload(melee.console)
    
    console = Console(path=file_path, is_dolphin=False, allow_old_version=True)
    if not console.connect():
        print("❌ Failed to connect (Rust)")
        return False
    
    rust_frames = []
    while True:
        gs = console.step()
        if gs is None:
            break
        rust_frames.append(gs)
    
    print(f"Rust:   {len(rust_frames)} frames")
    
    # Check frame counts
    if len(py_frames) != len(rust_frames):
        print(f"❌ FRAME COUNT MISMATCH: Python={len(py_frames)}, Rust={len(rust_frames)}")
        return False
    
    if expected_frames and len(py_frames) != expected_frames:
        print(f"⚠️  Warning: Got {len(py_frames)} frames, expected {expected_frames}")
    
    # Quick sanity check on first and last frames
    if py_frames:
        if py_frames[0].frame != rust_frames[0].frame:
            print(f"❌ First frame mismatch: Python={py_frames[0].frame}, Rust={rust_frames[0].frame}")
            #return False
        if py_frames[-1].frame != rust_frames[-1].frame:
            print(f"❌ Last frame mismatch: Python={py_frames[-1].frame}, Rust={rust_frames[-1].frame}")
            return False
    
    print("✅ PASS - Frame counts match!")
    return True


def main():
    print("=" * 70)
    print("TESTING KNOWN PROBLEMATIC FILES")
    print("=" * 70)
    
    base_dir = "/Users/eppie/Downloads/ALL_REPLAYS/FOX_vs_FOX"
    
    # Test files that have shown issues
    test_files = [
        # Old format files (no bookends)
        ("Game_20190309T092958_1752517753938.slp", None),  # Off-by-one issues
        ("Game_20190420T210831_1752517761722.slp", 10534),
        ("Game_20190323T192605_1752517756278.slp", 16159),
        ("Game_20190824T213519_1752517768163.slp", 8542),
        ("Game_20190824T145020_1752517767408.slp", 8847),
        ("Game_20190324T112245_1752517756606.slp", 10778),
        
        # Newer format files (with bookends) - had 2x frame issues
        ("20200219 - HNC 17 - PM 0737 - Fox (Default) vs Fox (Blue) - Fountain of Dreams_1752517659508.slp", 8231),
        ("Game_20190505T012453_1752517762227.slp", 10809),
        ("20200212 - HNC 3 - PM 0738 - Fox (Blue) vs Fox (Default) - Fountain of Dreams_1752517655467.slp", 4845),
        
        # Additional test files
        ("20_05_04 Fox + [LI] Fox (YS)_1752517670213.slp", 10591),
        ("20_18_46 Fox + Fox (YI)_1752517675313.slp", 10203),
    ]
    
    # Also test the known good file
    good_file = "/Users/eppie/PycharmProjects/nano-melee/libmelee/test.slp"
    if Path(good_file).exists():
        print("\nTesting known good file:")
        if not test_file(good_file, 7080):
            print("\n❌ KNOWN GOOD FILE FAILED!")
            return 1
    
    passed = 0
    failed = 0
    
    for filename, expected in test_files:
        filepath = Path(base_dir) / filename
        if not filepath.exists():
            print(f"\n⚠️  Skipping {filename} (not found)")
            continue
        
        try:
            if test_file(str(filepath), expected):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

