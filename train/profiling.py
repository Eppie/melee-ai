"""Profiling results display and analysis."""

from typing import Dict
from utils import Profiler


def print_profiling_results(profilers: Dict[str, Profiler]) -> None:
    """Print comprehensive profiling results."""

    # Section 1: Overview Table
    print("\n" + "=" * 100)
    print("PROFILING RESULTS - OVERVIEW")
    print("=" * 100)

    # Header
    print(
        f"{'Section':<20} {'Calls':>8} {'Mean (ms)':>12} {'Std (ms)':>12} "
        f"{'Min (ms)':>12} {'Max (ms)':>12} {'Total (s)':>12}"
    )
    print("-" * 100)

    # Collect data
    section_order = [
        "total_step",
        "data_prep",
        "progress_calc",
        "forward",
        "lr_update",
        "backward",
        "stats_update",
        "checkpoint",
        "logging",
    ]

    data = {}
    for name in section_order:
        prof = profilers.get(name)
        if prof:
            summary = prof.summary()
            data[name] = {
                "calls": int(summary["num_calls"]),
                "mean": summary["mean_time"] * 1000,  # Convert to ms
                "std": summary["std_time"] * 1000,
                "min": summary["min_time"] * 1000,
                "max": summary["max_time"] * 1000,
                "total": summary["total_time"],
            }
        else:
            data[name] = {
                "calls": 0,
                "mean": 0,
                "std": 0,
                "min": 0,
                "max": 0,
                "total": 0,
            }

    # Print rows
    for name in section_order:
        d = data[name]
        print(
            f"{name:<20} {d['calls']:>8} {d['mean']:>12.3f} {d['std']:>12.3f} "
            f"{d['min']:>12.3f} {d['max']:>12.3f} {d['total']:>12.2f}"
        )

    # Section 2: Percentage Breakdown
    print("\n" + "=" * 100)
    print("TIME BREAKDOWN (% of total)")
    print("=" * 100)

    total_time = data["total_step"]["total"]
    if total_time > 0:
        print(f"{'Section':<20} {'Time (s)':>12} {'Percentage':>12} {'Calls/sec':>12}")
        print("-" * 100)

        for name in section_order[1:]:  # Skip total_step
            d = data[name]
            pct = (d["total"] / total_time) * 100 if total_time > 0 else 0
            calls_per_sec = d["calls"] / d["total"] if d["total"] > 0 else 0
            print(
                f"{name:<20} {d['total']:>12.2f} {pct:>11.1f}% {calls_per_sec:>12.1f}"
            )

        # Calculate overhead (time not accounted for)
        accounted = sum(data[name]["total"] for name in section_order[1:])
        overhead = total_time - accounted
        overhead_pct = (overhead / total_time) * 100 if total_time > 0 else 0
        print("-" * 100)
        print(f"{'OVERHEAD':<20} {overhead:>12.2f} {overhead_pct:>11.1f}%")

    # Section 3: Key Insights
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)

    if data["forward"]["total"] > 0 and data["backward"]["total"] > 0:
        ratio = data["backward"]["total"] / data["forward"]["total"]
        print(f"• Forward/Backward ratio: 1:{ratio:.2f}")
        print(f"  (Backward takes {ratio:.1f}x as long as forward)")

    if data["checkpoint"]["calls"] > 0:
        print(f"• Checkpointing occurred {data['checkpoint']['calls']} times")
        print(f"  (Average: {data['checkpoint']['mean']:.1f}ms per checkpoint)")
    else:
        print(f"• No checkpoints saved during profiling period")

    if total_time > 0:
        compute_time = data["forward"]["total"] + data["backward"]["total"]
        compute_pct = (compute_time / total_time) * 100
        print(f"• Compute time (forward + backward): {compute_pct:.1f}% of total")

        overhead_time = data["data_prep"]["total"] + data["stats_update"]["total"]
        overhead_pct = (overhead_time / total_time) * 100
        print(f"• Data/stats overhead: {overhead_pct:.1f}% of total")

    # Section 4: Detailed Statistics
    print("\n" + "=" * 100)
    print("DETAILED STATISTICS")
    print("=" * 100)

    for name in section_order:
        prof = profilers.get(name)
        if prof and prof.num_calls > 0:
            summary = prof.summary()
            print(f"\n{name.upper()}:")
            print(f"  Calls: {int(summary['num_calls'])}")
            print(f"  Total time: {summary['total_time']:.3f}s")
            print(f"  Mean: {summary['mean_time']*1000:.3f}ms")
            print(f"  Std: {summary['std_time']*1000:.3f}ms")
            print(f"  Min: {summary['min_time']*1000:.3f}ms")
            print(f"  Max: {summary['max_time']*1000:.3f}ms")
            print(f"  Last: {summary['last_time']*1000:.3f}ms")
            print(f"  EMA: {summary['ema_time']*1000:.3f}ms")
            print(f"  Throughput: {summary['calls_per_second']:.1f} calls/sec")
