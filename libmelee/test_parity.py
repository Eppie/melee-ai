#!/usr/bin/env python3
"""
Test parity between Python and Rust parsers.

This compares the output of Python vs Rust parsing on the same SLP file
and reports any differences.
"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from melee.console import Console
from melee import enums


@dataclass
class FrameDiff:
    frame_num: int
    field_path: str
    python_value: Any
    rust_value: Any


def compare_values(path: str, py_val, rust_val, diffs: List[FrameDiff], frame_num: int):
    """Recursively compare two values and record differences."""
    # Handle None
    if py_val is None and rust_val is None:
        return
    if py_val is None or rust_val is None:
        diffs.append(FrameDiff(frame_num, path, py_val, rust_val))
        return
    
    # Handle enums - compare by value, not name (Rust uses different casing)
    if isinstance(py_val, enums.Character) or isinstance(py_val, enums.Action) or \
       isinstance(py_val, enums.Stage) or isinstance(py_val, enums.Menu):
        # Compare enum values
        py_value = py_val.value if hasattr(py_val, 'value') else py_val
        rust_value = rust_val.value if hasattr(rust_val, 'value') else rust_val
        if py_value != rust_value:
            diffs.append(FrameDiff(frame_num, path, py_val, rust_val))
        return
    
    # Handle floats with tolerance
    if isinstance(py_val, float) and isinstance(rust_val, float):
        if abs(py_val - rust_val) > 0.001:  # Small tolerance for float precision
            diffs.append(FrameDiff(frame_num, path, py_val, rust_val))
        return
    
    # Handle other primitives
    if isinstance(py_val, (int, bool, str)):
        if py_val != rust_val:
            diffs.append(FrameDiff(frame_num, path, py_val, rust_val))
        return
    
    # Handle tuples (like stick positions)
    if isinstance(py_val, tuple) and isinstance(rust_val, tuple):
        for i, (pv, rv) in enumerate(zip(py_val, rust_val)):
            compare_values(f"{path}[{i}]", pv, rv, diffs, frame_num)
        return
    
    # Handle dicts
    if isinstance(py_val, dict) and isinstance(rust_val, dict):
        all_keys = set(py_val.keys()) | set(rust_val.keys())
        for key in all_keys:
            compare_values(f"{path}[{key}]", py_val.get(key), rust_val.get(key), diffs, frame_num)
        return
    
    # Handle objects with __dict__
    if hasattr(py_val, '__dict__') and hasattr(rust_val, '__dict__'):
        py_dict = {k: v for k, v in py_val.__dict__.items() if not k.startswith('_')}
        rust_dict = {k: v for k, v in rust_val.__dict__.items() if not k.startswith('_')}
        all_keys = set(py_dict.keys()) | set(rust_dict.keys())
        for key in all_keys:
            compare_values(f"{path}.{key}", py_dict.get(key), rust_dict.get(key), diffs, frame_num)
        return


def compare_gamestates(py_gs, rust_gs, frame_num: int) -> List[FrameDiff]:
    """Compare two GameState objects and return list of differences."""
    diffs = []
    
    # Compare top-level fields
    compare_values("frame", py_gs.frame, rust_gs.frame, diffs, frame_num)
    compare_values("stage", py_gs.stage, rust_gs.stage, diffs, frame_num)
    compare_values("menu_state", py_gs.menu_state, rust_gs.menu_state, diffs, frame_num)
    
    # Compare players
    py_ports = set(py_gs.players.keys())
    rust_ports = set(rust_gs.players().keys()) if hasattr(rust_gs.players, '__call__') else set(rust_gs.players.keys())
    
    for port in py_ports | rust_ports:
        py_player = py_gs.players.get(port)
        rust_player = rust_gs.get_player(port) if hasattr(rust_gs, 'get_player') else rust_gs.players.get(port)
        
        if py_player is None and rust_player is None:
            continue
        if py_player is None or rust_player is None:
            diffs.append(FrameDiff(frame_num, f"player[{port}]", py_player, rust_player))
            continue
        
        # Compare player fields
        compare_values(f"player[{port}].character", py_player.character, rust_player.character, diffs, frame_num)
        compare_values(f"player[{port}].action", py_player.action, rust_player.action, diffs, frame_num)
        compare_values(f"player[{port}].position.x", py_player.position.x, rust_player.position.x, diffs, frame_num)
        compare_values(f"player[{port}].position.y", py_player.position.y, rust_player.position.y, diffs, frame_num)
        compare_values(f"player[{port}].percent", py_player.percent, rust_player.percent, diffs, frame_num)
        compare_values(f"player[{port}].stock", py_player.stock, rust_player.stock, diffs, frame_num)
        compare_values(f"player[{port}].facing", py_player.facing, rust_player.facing, diffs, frame_num)
    
    return diffs


def test_parity(slp_path: str):
    """Test parity between Python and Rust parsers."""
    print("=" * 70)
    print("PARITY TEST: Python vs Rust Parser")
    print("=" * 70)
    print(f"\nTesting with: {slp_path}\n")
    
    # Parse with Python
    print("1. Parsing with Python...")
    os.environ['LIBMELEE_USE_RUST'] = '0'
    
    # Need to reimport to pick up env var change
    import importlib
    import melee.console
    importlib.reload(melee.console)
    from melee.console import Console as PyConsole
    
    py_console = PyConsole(path=slp_path, is_dolphin=False, allow_old_version=True)
    py_console.connect()
    
    py_frames = []
    while True:
        gs = py_console.step()
        if gs is None:
            break
        py_frames.append(gs)
    
    print(f"   Python parsed {len(py_frames)} frames")
    
    # Parse with Rust
    print("\n2. Parsing with Rust...")
    os.environ['LIBMELEE_USE_RUST'] = '1'
    
    # Reload to pick up env var change
    importlib.reload(melee.console)
    from melee.console import Console as RustConsole
    
    rust_console = RustConsole(path=slp_path, is_dolphin=False, allow_old_version=True)
    rust_console.connect()
    
    rust_frames = []
    while True:
        gs = rust_console.step()
        if gs is None:
            break
        rust_frames.append(gs)
    
    print(f"   Rust parsed {len(rust_frames)} frames")
    
    # Compare
    print("\n3. Comparing outputs...")
    
    if len(py_frames) != len(rust_frames):
        print(f"   ❌ FRAME COUNT MISMATCH!")
        print(f"      Python: {len(py_frames)} frames")
        print(f"      Rust: {len(rust_frames)} frames")
        return False
    
    all_diffs = []
    for i, (py_gs, rust_gs) in enumerate(zip(py_frames, rust_frames)):
        diffs = compare_gamestates(py_gs, rust_gs, py_gs.frame)
        all_diffs.extend(diffs)
        
        if i % 1000 == 0 and i > 0:
            print(f"   Checked {i} frames...")
    
    print(f"\n   Checked all {len(py_frames)} frames")
    
    if all_diffs:
        print(f"\n   ❌ FOUND {len(all_diffs)} DIFFERENCES!")
        print("\n   First 20 differences:")
        for diff in all_diffs[:20]:
            print(f"      Frame {diff.frame_num}: {diff.field_path}")
            print(f"         Python: {diff.python_value}")
            print(f"         Rust:   {diff.rust_value}")
        
        if len(all_diffs) > 20:
            print(f"\n   ... and {len(all_diffs) - 20} more differences")
        
        return False
    else:
        print("\n   ✅ NO DIFFERENCES FOUND!")
        print("   Python and Rust parsers produce identical output!")
        return True


def main():
    slp_path = Path(__file__).parent / "test.slp"
    
    if not slp_path.exists():
        print(f"❌ test.slp not found at {slp_path}")
        return 1
    
    try:
        import melee_rust
        print("✓ Rust module available\n")
    except ImportError:
        print("❌ Rust module not available")
        print("   Run: cd melee_rust && maturin develop --release")
        return 1
    
    success = test_parity(str(slp_path))
    
    print("\n" + "=" * 70)
    if success:
        print("✅ PARITY TEST PASSED")
        print("=" * 70)
        return 0
    else:
        print("❌ PARITY TEST FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())

