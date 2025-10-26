#!/usr/bin/env python3
"""Debug Rust parser to see what's happening."""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

os.environ['LIBMELEE_USE_RUST'] = '1'

import logging
logging.basicConfig(level=logging.DEBUG)

from melee.console import Console
import melee_rust

def main():
    slp_path = Path(__file__).parent / "test.slp"
    
    print("Creating console with Rust enabled...")
    console = Console(path=str(slp_path), is_dolphin=False, allow_old_version=True)
    
    print("Connecting...")
    connected = console.connect()
    print(f"Connected: {connected}")
    
    print("\nChecking gamestate type...")
    console._temp_gamestate = None  # Force recreation
    
    # Manually step once to see what happens
    print("\nStepping once...")
    gs = console.step()
    
    if gs is None:
        print("❌ step() returned None")
    else:
        print(f"✓ Got gamestate: frame={gs.frame}")
        print(f"  Type: {type(gs)}")
        print(f"  Is Rust? {isinstance(gs, melee_rust.GameState)}")
        
        if hasattr(gs, 'players'):
            if callable(gs.players):
                players = gs.players()
            else:
                players = gs.players
            print(f"  Players: {list(players.keys())}")
    
    print("\nTrying a few more frames...")
    for i in range(5):
        gs = console.step()
        if gs is None:
            print(f"  Frame {i+2}: None")
            break
        print(f"  Frame {i+2}: {gs.frame}")

if __name__ == "__main__":
    main()

