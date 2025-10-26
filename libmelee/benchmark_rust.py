#!/usr/bin/env python3
"""
Benchmark script to compare Python vs Rust performance for libmelee.

This tests the performance of data structure creation and manipulation,
which are the hot paths in SLP parsing.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def benchmark_data_structures():
    """Benchmark creating and manipulating game state structures."""
    print("=" * 70)
    print("BENCHMARK: Data Structure Creation and Access")
    print("=" * 70)
    
    # Test Python implementation
    print("\n1. Testing Python GameState...")
    from melee.gamestate import GameState as PyGameState, PlayerState as PyPlayerState
    
    iterations = 100000
    start = time.time()
    for i in range(iterations):
        gs = PyGameState()
        gs.frame = i
        for port in range(1, 5):
            player = PyPlayerState()
            player.percent = i % 999
            player.position.x = float(i % 100)
            player.position.y = float(i % 100)
            gs.players[port] = player
    py_time = time.time() - start
    
    print(f"   Python: {iterations:,} iterations in {py_time:.3f}s")
    print(f"   Rate: {iterations/py_time:,.0f} ops/sec")
    
    # Test Rust implementation
    print("\n2. Testing Rust GameState...")
    try:
        import melee_rust
        
        start = time.time()
        for i in range(iterations):
            gs = melee_rust.GameState()
            gs.frame = i
            for port in range(1, 5):
                player = melee_rust.PlayerState()
                player.percent = i % 999
                player.position = melee_rust.Position()
                player.position.x = float(i % 100)
                player.position.y = float(i % 100)
                gs.set_player(port, player)
        rust_time = time.time() - start
        
        print(f"   Rust: {iterations:,} iterations in {rust_time:.3f}s")
        print(f"   Rate: {iterations/rust_time:,.0f} ops/sec")
        
        speedup = py_time / rust_time
        print(f"\n   ⚡ Speedup: {speedup:.2f}x faster")
        
    except ImportError:
        print("   ❌ Rust module not available")
        print("   Run: cd melee_rust && maturin develop --release")


def benchmark_enum_access():
    """Benchmark enum creation and comparison."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Enum Operations")
    print("=" * 70)
    
    # Test Python enums
    print("\n1. Testing Python Enums...")
    from melee import enums as py_enums
    
    iterations = 1000000
    start = time.time()
    for i in range(iterations):
        char = py_enums.Character.FOX
        stage = py_enums.Stage.FINAL_DESTINATION
        action = py_enums.Action.NAIR
        _ = (char == py_enums.Character.FOX)
    py_time = time.time() - start
    
    print(f"   Python: {iterations:,} iterations in {py_time:.3f}s")
    print(f"   Rate: {iterations/py_time:,.0f} ops/sec")
    
    # Test Rust enums
    print("\n2. Testing Rust Enums...")
    try:
        import melee_rust
        
        start = time.time()
        for i in range(iterations):
            char = melee_rust.Character.Fox
            stage = melee_rust.Stage.FinalDestination
            action = melee_rust.Action.Nair
            _ = (char == melee_rust.Character.Fox)
        rust_time = time.time() - start
        
        print(f"   Rust: {iterations:,} iterations in {rust_time:.3f}s")
        print(f"   Rate: {iterations/rust_time:,.0f} ops/sec")
        
        speedup = py_time / rust_time
        print(f"\n   ⚡ Speedup: {speedup:.2f}x faster")
        
    except ImportError:
        print("   ❌ Rust module not available")


def benchmark_slp_parsing():
    """Benchmark parsing an actual SLP file."""
    print("\n" + "=" * 70)
    print("BENCHMARK: SLP File Parsing")
    print("=" * 70)
    
    slp_path = Path(__file__).parent / "test.slp"
    
    if not slp_path.exists():
        print(f"\n   ⚠️  test.slp not found at {slp_path}")
        print("   Skipping SLP parsing benchmark")
        return
    
    print(f"\n   Using: {slp_path.name}")
    
    # Parse with Python
    print("\n1. Parsing with Python implementation...")
    import os
    os.environ['LIBMELEE_USE_RUST'] = '0'
    
    # Reload to pick up env var
    import importlib
    import melee.console
    importlib.reload(melee.console)
    from melee.console import Console
    
    start = time.time()
    console = Console(path=str(slp_path), is_dolphin=False, allow_old_version=True)
    console.connect()
    
    frame_count = 0
    while True:
        gamestate = console.step()
        if gamestate is None:
            break
        frame_count += 1
    
    py_time = time.time() - start
    
    print(f"   Parsed {frame_count:,} frames in {py_time:.3f}s")
    print(f"   Rate: {frame_count/py_time:,.0f} frames/sec")
    
    # Parse with Rust
    print("\n2. Parsing with Rust implementation...")
    try:
        import melee_rust
        
        os.environ['LIBMELEE_USE_RUST'] = '1'
        importlib.reload(melee.console)
        from melee.console import Console as RustConsole
        
        start = time.time()
        console = RustConsole(path=str(slp_path), is_dolphin=False, allow_old_version=True)
        console.connect()
        
        rust_frame_count = 0
        while True:
            gamestate = console.step()
            if gamestate is None:
                break
            rust_frame_count += 1
        
        rust_time = time.time() - start
        
        print(f"   Parsed {rust_frame_count:,} frames in {rust_time:.3f}s")
        print(f"   Rate: {rust_frame_count/rust_time:,.0f} frames/sec")
        
        if frame_count == rust_frame_count:
            speedup = py_time / rust_time
            print(f"\n   ⚡ Speedup: {speedup:.2f}x faster")
        else:
            print(f"\n   ⚠️  Frame count mismatch! Python: {frame_count}, Rust: {rust_frame_count}")
        
    except ImportError:
        print("   ❌ Rust module not available")
    except Exception as e:
        print(f"   ❌ Rust parsing failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print("LIBMELEE RUST ACCELERATION BENCHMARK")
    print("=" * 70)
    print("\nThis benchmark measures the performance improvement from using")
    print("Rust for hot-path operations in libmelee.\n")
    
    try:
        import melee_rust
        print("✓ Rust acceleration module detected\n")
    except ImportError:
        print("⚠️  Rust acceleration module not found")
        print("   To build: cd melee_rust && maturin develop --release\n")
    
    benchmark_data_structures()
    benchmark_enum_access()
    benchmark_slp_parsing()
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print("\nNotes:")
    print("- Run with --release build for accurate performance: maturin develop --release")
    print("- Results may vary based on system load and hardware")
    print("- Larger speedups expected for I/O-bound operations")
    print()


if __name__ == "__main__":
    main()

