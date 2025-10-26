#!/usr/bin/env python3
"""Debug script to see why specific files fail."""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Test one of the failing files
failing_file = "/Users/eppie/Downloads/ALL_REPLAYS/FOX_vs_FOX/Game_20190309T092958_1752517753938.slp"

print("Testing failing file:", failing_file)
print()

# First check Python works
print("=" * 70)
print("PYTHON PARSE")
print("=" * 70)

os.environ['LIBMELEE_USE_RUST'] = '0'
import importlib
import melee.console
importlib.reload(melee.console)

from melee.console import Console

console = Console(path=failing_file, is_dolphin=False, allow_old_version=True)
if not console.connect():
    print("Failed to connect")
    sys.exit(1)

frames = 0
for _ in range(10):  # Just first 10 frames
    gs = console.step()
    if gs is None:
        break
    frames += 1
    print(f"Frame {gs.frame}: {len(gs.players if hasattr(gs.players, '__len__') else gs.players.keys())} players")

print(f"Total (first 10): {frames} frames\n")

# Now test Rust with error output
print("=" * 70)
print("RUST PARSE (with errors)")
print("=" * 70)

os.environ['LIBMELEE_USE_RUST'] = '1'
importlib.reload(melee.console)

console = Console(path=failing_file, is_dolphin=False, allow_old_version=True)
if not console.connect():
    print("Failed to connect")
    sys.exit(1)

# This should show the errors from Rust
frames = 0
try:
    for _ in range(10):
        gs = console.step()
        if gs is None:
            print(f"Got None at frame {frames}")
            break
        frames += 1
        print(f"Frame {gs.frame}: players={list(gs.get_player(p) for p in [1,2,3,4] if gs.get_player(p))}")
except Exception as e:
    print(f"\n❌ Exception raised: {type(e).__name__}")
    print(f"   Message: {e}")
    import traceback
    traceback.print_exc()

print(f"Total (first 10): {frames} frames")

if frames == 0:
    print("\n❌ Rust parser returned 0 frames - check errors above")
else:
    print(f"\n✓ Rust parser worked - got {frames} frames")

