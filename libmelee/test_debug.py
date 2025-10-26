#!/usr/bin/env python3
"""
Test script to demonstrate debug logging.

Usage:
    LIBMELEE_DEBUG=1 python test_debug.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def main():
    # Enable Rust parsing
    os.environ['LIBMELEE_USE_RUST'] = '1'
    
    # Check if debug logging is enabled
    if os.environ.get('LIBMELEE_DEBUG') == '1':
        print("=" * 70)
        print("DEBUG LOGGING ENABLED")
        print("=" * 70)
        print()
    else:
        print("=" * 70)
        print("Debug logging NOT enabled")
        print("To enable, run: LIBMELEE_DEBUG=1 python test_debug.py")
        print("=" * 70)
        print()
    
    from melee.console import Console
    
    # Test with the known good file (newer format with bookends)
    test_file = "test.slp"
    
    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        return 1
    
    print(f"Testing: {test_file}")
    print()
    
    console = Console(path=test_file, is_dolphin=False, allow_old_version=True)
    if not console.connect():
        print("❌ Failed to connect")
        return 1
    
    frame_count = 0
    max_frames = 5  # Only parse first 5 frames to keep output manageable
    
    while frame_count < max_frames:
        gs = console.step()
        if gs is None:
            break
        frame_count += 1
        print(f"\n[PYTHON] Got frame {gs.frame}, players: {list(gs.players.keys())}")
    
    print()
    print("=" * 70)
    print(f"Parsed {frame_count} frames")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

