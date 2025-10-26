#!/usr/bin/env python3
"""Detailed debugging to see what events are in the failing file."""

import sys
import os
from pathlib import Path
import ubjson

sys.path.insert(0, str(Path(__file__).parent))

failing_file = "/Users/eppie/Downloads/ALL_REPLAYS/FOX_vs_FOX/Game_20190309T092958_1752517753938.slp"

print("Analyzing file structure:", failing_file)
print()

# Read the raw file
with open(failing_file, 'rb') as f:
    data = ubjson.loadb(f.read())
    raw = data['raw']
    
    print(f"Total size: {len(raw)} bytes")
    print(f"First 100 bytes (hex): {raw[:100].hex()}")
    print()
    
    # Find the first few events
    print("First 10 event types:")
    offset = 0
    event_sizes = [0] * 256
    
    for i in range(20):
        if offset >= len(raw):
            break
            
        event_byte = raw[offset]
        
        # Handle PAYLOADS
        if event_byte == 0x35:
            payload_size = raw[offset + 1]
            cursor = offset + 2
            num_commands = (payload_size - 1) // 3
            
            print(f"  {i}: Offset {offset}: PAYLOADS (0x{event_byte:02x}) - size {payload_size}")
            print(f"      Setting up {num_commands} event sizes:")
            
            for j in range(num_commands):
                if cursor + 3 > len(raw):
                    break
                cmd = raw[cursor]
                cmd_len = int.from_bytes(raw[cursor+1:cursor+3], 'big')
                event_sizes[cmd] = cmd_len + 1
                print(f"        Event 0x{cmd:02x} -> size {cmd_len + 1}")
                cursor += 3
            
            offset += payload_size + 1
            continue
        
        # Other events
        event_size = event_sizes[event_byte]
        if event_size == 0:
            print(f"  {i}: Offset {offset}: UNKNOWN (0x{event_byte:02x}) - size not set!")
            break
        
        event_name = {
            0x10: "GECKO_CODES",
            0x36: "GAME_START",
            0x37: "PRE_FRAME",
            0x38: "POST_FRAME",
            0x39: "GAME_END",
            0x3A: "FRAME_START",
            0x3B: "ITEM_UPDATE",
            0x3C: "FRAME_BOOKEND",
            0x3E: "MENU_EVENT",
        }.get(event_byte, f"UNKNOWN_0x{event_byte:02x}")
        
        print(f"  {i}: Offset {offset}: {event_name} (0x{event_byte:02x}) - size {event_size}")
        
        offset += event_size
    
    print()
    print("Now let's test with Rust parser and see what happens...")
    print()

# Now test with Rust
os.environ['LIBMELEE_USE_RUST'] = '1'
import importlib
import melee.console
importlib.reload(melee.console)

from melee.console import Console

console = Console(path=failing_file, is_dolphin=False, allow_old_version=True)
if not console.connect():
    print("Failed to connect")
    sys.exit(1)

print("Connected, attempting to step...")

try:
    gs = console.step()
    if gs is None:
        print("❌ step() returned None")
    else:
        print(f"✓ Got frame {gs.frame}")
except Exception as e:
    print(f"❌ Exception: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

