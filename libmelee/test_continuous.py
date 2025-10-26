#!/usr/bin/env python3
"""
Continuous parity testing with random replay files (PARALLELIZED).

Continuously picks random .slp files and verifies Python and Rust
parsers produce identical results. Reports any discrepancies immediately.
Uses multiprocessing to test multiple files in parallel.
"""

import sys
import os
import time
import random
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
import traceback

sys.path.insert(0, str(Path(__file__).parent))

from melee import enums


@dataclass
class TestResult:
    file_path: str
    success: bool
    frame_count: int
    python_time: float
    rust_time: float
    differences: int
    error: str = ""


def compare_values(py_val, rust_val) -> bool:
    """Compare two values, handling enums and floats specially."""
    if py_val is None and rust_val is None:
        return True
    if py_val is None or rust_val is None:
        return False
    
    # Handle enums - compare by value
    if isinstance(py_val, (enums.Character, enums.Action, enums.Stage, enums.Menu)):
        py_value = py_val.value if hasattr(py_val, 'value') else py_val
        rust_value = rust_val.value if hasattr(rust_val, 'value') else rust_val
        return py_value == rust_value
    
    # Handle floats with tolerance
    if isinstance(py_val, float) and isinstance(rust_val, float):
        return abs(py_val - rust_val) < 0.001
    
    # Handle tuples
    if isinstance(py_val, tuple) and isinstance(rust_val, tuple):
        if len(py_val) != len(rust_val):
            return False
        return all(compare_values(p, r) for p, r in zip(py_val, rust_val))
    
    # Direct comparison
    return py_val == rust_val


def compare_gamestates(py_gs, rust_gs) -> int:
    """Compare two GameStates. Returns number of differences."""
    diffs = 0
    
    # Compare frame
    if not compare_values(py_gs.frame, rust_gs.frame):
        diffs += 1
    
    # Compare stage
    if not compare_values(py_gs.stage, rust_gs.stage):
        diffs += 1
    
    # Compare players
    py_ports = set(py_gs.players.keys())
    rust_ports = set(rust_gs.players().keys()) if hasattr(rust_gs.players, '__call__') else set(rust_gs.players.keys())
    
    if py_ports != rust_ports:
        diffs += len(py_ports ^ rust_ports)
    
    for port in py_ports & rust_ports:
        py_player = py_gs.players.get(port)
        rust_player = rust_gs.get_player(port) if hasattr(rust_gs, 'get_player') else rust_gs.players.get(port)
        
        if py_player is None or rust_player is None:
            diffs += 1
            continue
        
        # Compare key player fields
        if not compare_values(py_player.character, rust_player.character):
            diffs += 1
        if not compare_values(py_player.action, rust_player.action):
            diffs += 1
        if not compare_values(py_player.percent, rust_player.percent):
            diffs += 1
        if not compare_values(py_player.stock, rust_player.stock):
            diffs += 1
        if not compare_values(py_player.position.x, rust_player.position.x):
            diffs += 1
        if not compare_values(py_player.position.y, rust_player.position.y):
            diffs += 1
    
    return diffs


def test_file_worker(slp_file_str: str) -> TestResult:
    """Worker function to test a single replay file for parity."""
    from melee.console import Console
    import importlib
    import melee.console
    
    slp_file = Path(slp_file_str)
    
    try:
        # Parse with Python
        os.environ['LIBMELEE_USE_RUST'] = '0'
        importlib.reload(melee.console)
        
        start = time.time()
        console = Console(path=str(slp_file), is_dolphin=False, allow_old_version=True)
        
        if not console.connect():
            return TestResult(
                file_path=str(slp_file),
                success=False,
                frame_count=0,
                python_time=0,
                rust_time=0,
                differences=0,
                error="Failed to connect (Python)"
            )
        
        py_frames = []
        while True:
            gs = console.step()
            if gs is None:
                break
            py_frames.append(gs)
        
        python_time = time.time() - start
        
        # Parse with Rust
        os.environ['LIBMELEE_USE_RUST'] = '1'
        importlib.reload(melee.console)
        
        start = time.time()
        console = Console(path=str(slp_file), is_dolphin=False, allow_old_version=True)
        
        if not console.connect():
            return TestResult(
                file_path=str(slp_file),
                success=False,
                frame_count=len(py_frames),
                python_time=python_time,
                rust_time=0,
                differences=0,
                error="Failed to connect (Rust)"
            )
        
        rust_frames = []
        while True:
            gs = console.step()
            if gs is None:
                break
            rust_frames.append(gs)
        
        rust_time = time.time() - start
        
        # Compare frame counts
        if len(py_frames) != len(rust_frames):
            return TestResult(
                file_path=str(slp_file),
                success=False,
                frame_count=len(py_frames),
                python_time=python_time,
                rust_time=rust_time,
                differences=abs(len(py_frames) - len(rust_frames)),
                error=f"Frame count mismatch: Python={len(py_frames)}, Rust={len(rust_frames)}"
            )
        
        # Compare frames
        total_diffs = 0
        for py_gs, rust_gs in zip(py_frames, rust_frames):
            diffs = compare_gamestates(py_gs, rust_gs)
            total_diffs += diffs
        
        return TestResult(
            file_path=str(slp_file),
            success=(total_diffs == 0),
            frame_count=len(py_frames),
            python_time=python_time,
            rust_time=rust_time,
            differences=total_diffs,
            error="" if total_diffs == 0 else f"{total_diffs} differences found"
        )
        
    except Exception as e:
        return TestResult(
            file_path=str(slp_file),
            success=False,
            frame_count=0,
            python_time=0,
            rust_time=0,
            differences=0,
            error=f"Exception: {str(e)}"
        )


class ParallelContinuousTester:
    def __init__(self, replay_dir: str, num_workers: int = None):
        self.replay_dir = Path(replay_dir)
        self.replay_files = list(self.replay_dir.glob("**/*.slp"))
        
        if not self.replay_files:
            raise ValueError(f"No .slp files found in {replay_dir}")
        
        # Default to CPU count - 1 to leave a core for the main process
        self.num_workers = num_workers or max(1, cpu_count() - 1)
        
        print(f"Found {len(self.replay_files)} replay files")
        print(f"Using {self.num_workers} parallel workers")
        
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.total_frames = 0
        self.total_python_time = 0.0
        self.total_rust_time = 0.0
        self.start_time = time.time()
    
    def print_status(self):
        """Print current testing status."""
        elapsed = time.time() - self.start_time
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        avg_speedup = (self.total_python_time / self.total_rust_time) if self.total_rust_time > 0 else 0
        
        print(f"\r[{elapsed:.0f}s] Tests: {self.tests_run} | Passed: {self.tests_passed} | Failed: {self.tests_failed} | "
              f"Success: {success_rate:.1f}% | Frames: {self.total_frames:,} | Speedup: {avg_speedup:.2f}x", end='', flush=True)
    
    def run_continuous(self, max_tests: int = None, verbose: bool = False):
        """Run continuous testing with parallel workers."""
        print("\n" + "=" * 70)
        print("CONTINUOUS PARITY TESTING (PARALLEL)")
        print("=" * 70)
        print(f"Replay directory: {self.replay_dir}")
        print(f"Files available: {len(self.replay_files)}")
        print(f"Workers: {self.num_workers}")
        print(f"Max tests: {max_tests if max_tests else 'unlimited'}")
        print()
        print("Press Ctrl+C to stop")
        print("=" * 70)
        print()
        
        try:
            with Pool(processes=self.num_workers) as pool:
                while True:
                    # Check if we've hit max tests
                    if max_tests and self.tests_run >= max_tests:
                        break
                    
                    # Pick random files for batch testing (one per worker)
                    batch_size = min(self.num_workers, 
                                   max_tests - self.tests_run if max_tests else self.num_workers)
                    slp_files = [str(random.choice(self.replay_files)) for _ in range(batch_size)]
                    
                    # Process batch in parallel
                    results = pool.map(test_file_worker, slp_files)
                    
                    # Process results
                    for result in results:
                        # Update stats
                        self.tests_run += 1
                        self.total_frames += result.frame_count
                        self.total_python_time += result.python_time
                        self.total_rust_time += result.rust_time
                        
                        if result.success:
                            self.tests_passed += 1
                        else:
                            self.tests_failed += 1
                            
                            # Print failure details
                            print()  # New line
                            print()
                            print("❌ FAILURE DETECTED!")
                            print(f"   File: {result.file_path}")
                            print(f"   Frames: {result.frame_count}")
                            print(f"   Error: {result.error}")
                            print()
                        
                        # Print status
                        if verbose or not result.success:
                            print(f"[Test {self.tests_run}] {'✓' if result.success else '✗'} "
                                  f"{Path(result.file_path).name} - {result.frame_count} frames - "
                                  f"{result.python_time:.3f}s/{result.rust_time:.3f}s")
                        else:
                            self.print_status()
        
        except KeyboardInterrupt:
            print("\n\nStopped by user")
        
        # Print final summary
        print("\n")
        print("=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        elapsed = time.time() - self.start_time
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        avg_speedup = (self.total_python_time / self.total_rust_time) if self.total_rust_time > 0 else 0
        
        print(f"Tests run:       {self.tests_run}")
        print(f"Passed:          {self.tests_passed}")
        print(f"Failed:          {self.tests_failed}")
        print(f"Success rate:    {success_rate:.1f}%")
        print(f"Total frames:    {self.total_frames:,}")
        print(f"Time elapsed:    {elapsed:.1f}s")
        print(f"Throughput:      {self.total_frames / elapsed:.0f} frames/sec")
        print(f"Avg speedup:     {avg_speedup:.2f}x")
        print(f"Python time:     {self.total_python_time:.2f}s")
        print(f"Rust time:       {self.total_rust_time:.2f}s")
        print()
        
        if self.tests_failed == 0:
            print("✅ ALL TESTS PASSED!")
        else:
            print(f"⚠️  {self.tests_failed} TEST(S) FAILED")
        
        print("=" * 70)


def main():
    replay_dir = "/Users/eppie/Downloads/ALL_REPLAYS/FOX_vs_FOX"
    
    if not Path(replay_dir).exists():
        print(f"❌ Directory not found: {replay_dir}")
        return 1
    
    try:
        import melee_rust
    except ImportError:
        print("❌ Rust module not available")
        return 1
    
    # Auto-detect number of workers (defaults to cpu_count - 1)
    # Or specify manually: num_workers=4
    tester = ParallelContinuousTester(replay_dir)
    
    # Run continuously (or specify max_tests=100 for limited run)
    # Use verbose=True to see each test, or False for compact display
    tester.run_continuous(max_tests=None, verbose=False)
    
    return 0 if tester.tests_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
