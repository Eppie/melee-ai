#!/usr/bin/env python3
"""Test slippistats on a replay file to get baseline results."""

import json
from pathlib import Path
from slippistats import Game, StatsComputer

# Use one of the replay files available (must be 1v1 with metadata)
replay_file = "slippi-js/slp/actionEdgeCases.slp"

print(f"Processing {replay_file}...")
print("=" * 80)

# Parse the replay
game = Game(replay_file)
stats_computer = StatsComputer(game)
player_stats_list = stats_computer.stats_compute()

# Convert to dict for easier access
game.player_stats = {ps.port: ps.stats for ps in player_stats_list}

# Get player stats
for ps in player_stats_list:
    port = ps.port
    player_stats = ps.stats
    print(f"\n{'='*80}")
    print(f"PLAYER Port {port}")
    print(f"{'='*80}")

    # Wavedashes
    print(f"\n--- WAVEDASHES ({len(player_stats.wavedashes)}) ---")
    for i, wd in enumerate(player_stats.wavedashes[:5]):  # Show first 5
        print(f"  [{i+1}] Frame {wd.frame_index}: "
              f"angle={wd.angle:.1f}°, direction={wd.direction}, "
              f"trigger_frame={wd.trigger_frame}")
    if len(player_stats.wavedashes) > 5:
        print(f"  ... and {len(player_stats.wavedashes) - 5} more")

    # Dashes
    print(f"\n--- DASHES ({len(player_stats.dashes)}) ---")
    for i, dash in enumerate(player_stats.dashes[:5]):
        print(f"  [{i+1}] Frame {dash.frame_index_start}-{dash.frame_index_end}: "
              f"distance={dash.distance():.2f}, is_dashdance={dash.is_dashdance}")
    if len(player_stats.dashes) > 5:
        print(f"  ... and {len(player_stats.dashes) - 5} more")

    # Techs
    print(f"\n--- TECHS ({len(player_stats.techs)}) ---")
    for i, tech in enumerate(player_stats.techs[:5]):
        print(f"  [{i+1}] Frame {tech.frame_index}: "
              f"type={tech.tech_type}, was_punished={tech.was_punished}, "
              f"towards_center={tech.towards_center}, towards_opponent={tech.towards_opponent}")
    if len(player_stats.techs) > 5:
        print(f"  ... and {len(player_stats.techs) - 5} more")

    # L-Cancels
    print(f"\n--- L-CANCELS ({len(player_stats.l_cancels)}) ---")
    success = sum(1 for lc in player_stats.l_cancels if lc.l_cancel)
    print(f"  Success: {success}/{len(player_stats.l_cancels)} "
          f"({100*success/len(player_stats.l_cancels):.1f}%)" if len(player_stats.l_cancels) > 0 else "  Success: 0/0")
    for i, lc in enumerate(player_stats.l_cancels[:5]):
        print(f"  [{i+1}] Frame {lc.frame_index}: "
              f"l_cancel={lc.l_cancel}, trigger_input_frame={lc.trigger_input_frame}, "
              f"during_hitlag={lc.during_hitlag}, fastfall={lc.fastfall}")
    if len(player_stats.l_cancels) > 5:
        print(f"  ... and {len(player_stats.l_cancels) - 5} more")

    # Shield Drops
    print(f"\n--- SHIELD DROPS ({len(player_stats.shield_drops)}) ---")
    for i, sd in enumerate(player_stats.shield_drops[:5]):
        print(f"  [{i+1}] Frame {sd.frame_index}: "
              f"oo_shieldstun_frame={sd.oo_shieldstun_frame}")
    if len(player_stats.shield_drops) > 5:
        print(f"  ... and {len(player_stats.shield_drops) - 5} more")

    # Take Hits (DI/SDI)
    print(f"\n--- TAKE HITS / DI ({len(player_stats.take_hits)}) ---")
    for i, hit in enumerate(player_stats.take_hits[:5]):
        print(f"  [{i+1}] Frame {hit.frame_index}: "
              f"damage={hit.damage}, hitlag_frames={hit.hitlag_frames}, "
              f"sdi_inputs={hit.sdi_inputs}")
    if len(player_stats.take_hits) > 5:
        print(f"  ... and {len(player_stats.take_hits) - 5} more")

print(f"\n{'='*80}")
print("SUMMARY COUNTS")
print(f"{'='*80}")
for port, player_stats in game.player_stats.items():
    print(f"\nPlayer {port}:")
    print(f"  Wavedashes: {len(player_stats.wavedashes)}")
    print(f"  Dashes: {len(player_stats.dashes)}")
    print(f"  Techs: {len(player_stats.techs)}")
    print(f"  L-Cancels: {len(player_stats.l_cancels)}")
    print(f"  Shield Drops: {len(player_stats.shield_drops)}")
    print(f"  Take Hits: {len(player_stats.take_hits)}")
