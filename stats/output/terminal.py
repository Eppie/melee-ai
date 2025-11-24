"""Terminal output formatter for statistics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from stats.utils.formatting import format_bytes, format_number, format_percent


def print_summary(stats: Dict[str, Any], verbose: bool = True) -> None:
    """Print a formatted summary of statistics to terminal."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS SUMMARY")
    print("=" * 60)

    # Episode stats
    if "episodes" in stats:
        print_episode_summary(stats["episodes"])

    # Data quality
    if "data_quality" in stats:
        print_quality_summary(stats["data_quality"])

    # Action states
    if "action_states" in stats:
        print_action_summary(stats["action_states"], verbose)

    # Controller inputs
    if "controller_inputs" in stats:
        print_controller_summary(stats["controller_inputs"], verbose)

    # Derived metrics
    if "derived_metrics" in stats:
        print_derived_summary(stats["derived_metrics"])

    # Column stats overview
    if "columns" in stats:
        print_column_overview(stats["columns"], verbose)

    print("\n" + "=" * 60)


def print_episode_summary(ep_stats: Dict[str, Any]) -> None:
    """Print episode statistics summary."""
    print("\n--- EPISODE STATISTICS ---")

    print(f"Total episodes:     {format_number(ep_stats.get('total_episodes', 0))}")
    print(f"Total frames:       {format_number(ep_stats.get('total_frames', 0))}")

    length = ep_stats.get("length", {})
    if length:
        print(
            f"Avg episode length: {length.get('mean', 0):.1f} frames ({length.get('mean', 0)/60:.1f}s)"
        )
        print(
            f"Length range:       {length.get('min', 0)} - {length.get('max', 0)} frames"
        )

    # Stage distribution
    stages = ep_stats.get("stages", {})
    if stages:
        print("\nTop stages:")
        sorted_stages = sorted(
            stages.items(), key=lambda x: x[1].get("count", 0), reverse=True
        )[:5]
        for stage, info in sorted_stages:
            count = info.get("count", 0)
            pct = info.get("percent", 0)
            print(f"  {stage}: {count} ({pct:.1f}%)")

    # Outcome distribution
    outcomes = ep_stats.get("outcomes", {})
    if outcomes:
        print("\nOutcomes:")
        p1_wins = outcomes.get("p1_wins", 0)
        p2_wins = outcomes.get("p2_wins", 0)
        draws = outcomes.get("draws", 0)
        print(f"  P1 wins: {p1_wins} ({outcomes.get('p1_win_rate', 0):.1f}%)")
        print(f"  P2 wins: {p2_wins} ({outcomes.get('p2_win_rate', 0):.1f}%)")
        if draws > 0:
            print(f"  Draws: {draws}")


def print_quality_summary(quality: Dict[str, Any]) -> None:
    """Print data quality summary."""
    print("\n--- DATA QUALITY ---")

    summary = quality.get("summary", {})
    score = quality.get("data_quality_score", 0)
    print(f"Quality score: {score:.1f}/100")

    if summary.get("empty_episodes", 0) > 0:
        print(f"  Empty episodes: {summary['empty_episodes']}")
    if summary.get("very_short_episodes", 0) > 0:
        print(f"  Very short (<60 frames): {summary['very_short_episodes']}")
    if summary.get("stock_inconsistencies", 0) > 0:
        print(f"  Stock inconsistencies: {summary['stock_inconsistencies']}")

    # NaN/Inf issues
    nan_inf = quality.get("nan_inf_issues", {})
    if nan_inf:
        total_nan = sum(v.get("nan_count", 0) for v in nan_inf.values())
        total_inf = sum(v.get("inf_count", 0) for v in nan_inf.values())
        if total_nan > 0 or total_inf > 0:
            print(f"  NaN values: {total_nan}, Inf values: {total_inf}")

    # Range violations
    violations = quality.get("range_violations", {})
    if violations:
        total_violations = sum(v.get("count", 0) for v in violations.values())
        print(f"  Range violations: {total_violations}")


def print_action_summary(action_stats: Dict[str, Any], verbose: bool) -> None:
    """Print action state summary."""
    print("\n--- ACTION STATES ---")

    # Top actions from p1
    p1_stats = action_stats.get("p1", {})
    action_dist = p1_stats.get("action_distribution", [])
    if action_dist:
        print("\nTop 10 actions (p1):")
        for action in action_dist[:10]:
            name = action.get("name", f"Action {action.get('action_id')}")
            cat = action.get("category", "unknown")
            pct = action.get("percent", 0)
            print(f"  {name} ({cat}): {pct:.1f}%")

    # Category breakdown
    categories = p1_stats.get("category_distribution", {})
    if categories and verbose:
        print("\nAction categories (p1):")
        sorted_cats = sorted(
            categories.items(), key=lambda x: x[1].get("percent", 0), reverse=True
        )
        for cat_name, info in sorted_cats[:8]:
            print(f"  {cat_name}: {info.get('percent', 0):.1f}%")


def print_controller_summary(ctrl_stats: Dict[str, Any], verbose: bool) -> None:
    """Print controller input summary."""
    print("\n--- CONTROLLER INPUTS ---")

    for player in ["p1", "p2"]:
        player_stats = ctrl_stats.get(player, {})
        if not player_stats:
            continue

        print(f"\n{player.upper()}:")

        # Button usage
        buttons = player_stats.get("buttons", {})
        if buttons:
            print("  Button usage (% frames):")
            sorted_buttons = sorted(
                [(b, info.get("percent_frames", 0)) for b, info in buttons.items()],
                key=lambda x: x[1],
                reverse=True,
            )
            for button, pct in sorted_buttons[:5]:
                if pct > 0.1:
                    print(f"    {button}: {pct:.1f}%")

        # Stick regions
        main_stick = player_stats.get("main_stick", {})
        regions = main_stick.get("regions", {})
        if regions and verbose:
            print("  Main stick regions:")
            sorted_regions = sorted(
                [(r, info.get("percent", 0)) for r, info in regions.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            for region, pct in sorted_regions:
                print(f"    {region}: {pct:.1f}%")


def print_derived_summary(derived: Dict[str, Any]) -> None:
    """Print derived metrics summary."""
    print("\n--- DERIVED METRICS ---")

    # Distance
    dist = derived.get("distance", {})
    if dist:
        print(f"Avg player distance: {dist.get('mean', 0):.1f} units")

    # Advantage
    adv = derived.get("advantage", {})
    if adv:
        print(f"P1 advantage: {adv.get('p1_advantage_percent', 0):.1f}%")
        print(f"P2 advantage: {adv.get('p2_advantage_percent', 0):.1f}%")

    # Combos
    combos = derived.get("combos", {})
    if combos:
        p1_combos = combos.get("p1_received", {})
        p2_combos = combos.get("p2_received", {})
        if p1_combos.get("count", 0) > 0:
            print(
                f"P1 combos received: {p1_combos['count']} (avg {p1_combos.get('mean_length_seconds', 0):.2f}s)"
            )
        if p2_combos.get("count", 0) > 0:
            print(
                f"P2 combos received: {p2_combos['count']} (avg {p2_combos.get('mean_length_seconds', 0):.2f}s)"
            )

    # Kill percents
    kills = derived.get("kill_percents", {})
    if kills:
        p1_kills = kills.get("p1", {})
        p2_kills = kills.get("p2", {})
        if p1_kills.get("count", 0) > 0:
            print(f"P1 avg death percent: {p1_kills.get('mean', 0):.1f}%")
        if p2_kills.get("count", 0) > 0:
            print(f"P2 avg death percent: {p2_kills.get('mean', 0):.1f}%")


def print_column_overview(col_stats: Dict[str, Any], verbose: bool) -> None:
    """Print column statistics overview."""
    if not verbose:
        return

    print("\n--- FEATURE OVERVIEW ---")

    columns = col_stats.get("columns", {})
    if not columns:
        return

    # Group by type
    continuous = []
    categorical = []
    binary = []

    for name, info in columns.items():
        col_type = info.get("type", "unknown")
        if col_type == "continuous":
            continuous.append((name, info))
        elif col_type == "categorical":
            categorical.append((name, info))
        elif col_type == "binary":
            binary.append((name, info))

    print(
        f"Features: {len(continuous)} continuous, {len(categorical)} categorical, {len(binary)} binary"
    )

    # Show a few examples
    if continuous:
        print("\nSample continuous features:")
        for name, info in continuous[:3]:
            stats = info.get("stats", {})
            print(
                f"  {name}: mean={stats.get('mean', 0):.2f}, std={stats.get('std', 0):.2f}"
            )

    if categorical:
        print("\nSample categorical features:")
        for name, info in categorical[:3]:
            stats = info.get("stats", {})
            print(f"  {name}: {stats.get('n_unique', 0)} unique values")
