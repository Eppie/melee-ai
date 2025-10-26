#!/usr/bin/env python3
"""Test script to parse test.slp and demonstrate Rust module functionality."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from melee.console import Console

def main():
    print("Testing SLP file parsing with libmelee...")
    print("=" * 60)
    
    # Test with the test.slp file
    slp_path = Path(__file__).parent / "test.slp"
    
    if not slp_path.exists():
        print(f"❌ test.slp not found at {slp_path}")
        return 1
    
    print(f"✓ Found test.slp at: {slp_path}")
    
    # Create console from SLP file
    console = Console(path=str(slp_path), is_dolphin=False, allow_old_version=True)
    
    print("✓ Created Console object")
    
    # Connect to the SLP file
    if not console.connect():
        print("❌ Failed to connect to SLP file")
        return 1
    
    print("✓ Connected to SLP file")
    
    # Read frames
    frame_count = 0
    gamestate = None
    
    print("\nParsing frames...")
    while True:
        gamestate = console.step()
        if gamestate is None:
            break
        frame_count += 1
        
        # Print some info every 100 frames
        if frame_count % 100 == 0:
            print(f"  Frame {gamestate.frame}: {len(gamestate.players)} players")
    
    print(f"\n✓ Successfully parsed {frame_count} frames")
    
    if gamestate and gamestate.players:
        print(f"\nLast frame info:")
        print(f"  Frame number: {gamestate.frame}")
        print(f"  Stage: {gamestate.stage}")
        print(f"  Players: {len(gamestate.players)}")
        for port, player in gamestate.players.items():
            print(f"    Port {port}: {player.character} at ({player.position.x:.1f}, {player.position.y:.1f})")
    
    print("\n✅ All tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

