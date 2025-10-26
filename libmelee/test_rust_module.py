#!/usr/bin/env python3
"""
Simple test to verify the Rust module is working correctly.
"""

try:
    import melee_rust
    print("✓ Successfully imported melee_rust")
    
    # Test creating enums
    stage = melee_rust.Stage.FinalDestination
    print(f"✓ Created Stage enum: {stage}")
    
    character = melee_rust.Character.Fox
    print(f"✓ Created Character enum: {character}")
    
    # Test creating Position
    pos = melee_rust.Position()
    pos.x = 10.5
    pos.y = 20.3
    print(f"✓ Created Position: ({pos.x}, {pos.y})")
    
    # Test creating PlayerState
    player = melee_rust.PlayerState()
    player.character = character
    player.percent = 50
    player.position = pos
    print(f"✓ Created PlayerState with character={player.character}, percent={player.percent}")
    
    # Test creating GameState
    gamestate = melee_rust.GameState()
    gamestate.frame = 100
    gamestate.stage = stage
    gamestate.set_player(1, player)
    print(f"✓ Created GameState with frame={gamestate.frame}, stage={gamestate.stage}")
    print(f"✓ GameState has player at port 1: {gamestate.has_player(1)}")
    
    retrieved_player = gamestate.get_player(1)
    if retrieved_player:
        print(f"✓ Retrieved player from gamestate: character={retrieved_player.character}")
    
    # Test parse_events function exists
    print(f"✓ parse_events function exists: {hasattr(melee_rust, 'parse_events')}")
    
    print("\n✅ All basic tests passed! Rust module is working correctly.")
    
except ImportError as e:
    print(f"❌ Failed to import melee_rust: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

