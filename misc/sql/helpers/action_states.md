# Action State Reference

This document provides a reference for action state IDs used in Super Smash Bros. Melee. Action states represent the current animation/state of a character.

## Usage in Queries

Action states are stored in the `action_state` column as integers (0-396):

```sql
-- Check if player is in damage state
WHERE action_state BETWEEN 75 AND 92  -- 0x4B to 0x5C in hex

-- Check for specific action
WHERE action_state = 236  -- EscapeAir (air dodge)

-- Check for action category
WHERE action_state IN (65, 66, 67, 68, 69)  -- All aerial attacks
```

## Action State Categories

### Death States (0-11)
| ID (Decimal) | ID (Hex) | Name | Description |
|--------------|----------|------|-------------|
| 0-11         | 0x00-0x0B | Death animations | Various death/respawn states |

Used in death type classification (blast zones, star KOs, etc.).

### Neutral & Wait States (14-23)
| ID (Decimal) | ID (Hex) | Name | Description |
|--------------|----------|------|-------------|
| 14           | 0x0E     | Wait | Standing idle |
| 15-17        | 0x0F-0x11| Walk | Walking animations |
| 18           | 0x12     | Turn | Turning around |
| 20           | 0x14     | Dash | Dashing |
| 21           | 0x15     | Run | Running |
| 24           | 0x18     | KneeBend | Jump squat (pre-jump crouch) |

### Jumping & Airborne (24-43)
| ID (Decimal) | ID (Hex) | Name | Description |
|--------------|----------|------|-------------|
| 24           | 0x18     | KneeBend | Jump squat |
| 25-26        | 0x19-0x1A| JumpF/JumpB | Forward/backward jump |
| 29-31        | 0x1D-0x1F| Fall states | Falling, tumbling |
| 43           | 0x2B     | LandingFallSpecial | Wavedash/waveland landing |

**Wavedash Detection:** KneeBend (24) → EscapeAir (236) → LandingFallSpecial (43)

### Attacks (44-69)

#### Jabs (44-47)
| ID (Decimal) | ID (Hex) | Name | Description |
|--------------|----------|------|-------------|
| 44           | 0x2C     | AttackJab1 | First jab |
| 45           | 0x2D     | AttackJab2 | Second jab |
| 46           | 0x2E     | AttackJab3 | Third jab |
| 47           | 0x2F     | AttackJabM | Rapid jabs |

#### Ground Attacks (50-64)
| ID (Decimal) | ID (Hex) | Name | Description |
|--------------|----------|------|-------------|
| 50           | 0x32     | AttackDash | Dash attack |
| 51-57        | 0x33-0x39| Tilts | F-tilt, U-tilt, D-tilt |
| 58-64        | 0x3A-0x40| Smashes | F-smash, U-smash, D-smash (charge + release) |

#### Aerial Attacks (65-69)
| ID (Decimal) | ID (Hex) | Name | Description |
|--------------|----------|------|-------------|
| 65           | 0x41     | AerialNair | Neutral air |
| 66           | 0x42     | AerialFair | Forward air |
| 67           | 0x43     | AerialBair | Back air |
| 68           | 0x44     | AerialUair | Up air |
| 69           | 0x45     | AerialDair | Down air |

#### Aerial Landings (70-74)
| ID (Decimal) | ID (Hex) | Name | Description |
|--------------|----------|------|-------------|
| 70-74        | 0x46-0x4A| Landing animations | Nair, Fair, Bair, Uair, Dair landings |

**L-Cancel Detection:** Check `lcancel_status` column during these states:
- `lcancel_status = 1`: Success
- `lcancel_status = 2`: Failure

### Damage & Hitstun (75-92)
| ID (Decimal) | ID (Hex) | Name | Description |
|--------------|----------|------|-------------|
| 75-92        | 0x4B-0x5C| Damage animations | Various hitstun/knockback states |
| 38           | 0x26     | DamageFall | Tumbling in hitstun |

**Combo Detection:** Opponent in these states indicates active combo.

### Tech & Down States (183-204)

#### Down (183-198)
| ID (Decimal) | ID (Hex) | Name | Description |
|--------------|----------|------|-------------|
| 183          | 0xB7     | TechMissUp | Missed tech (face up) |
| 191          | 0xBF     | TechMissDown | Missed tech (face down) |

#### Techs (199-204)
| ID (Decimal) | ID (Hex) | Name | Description |
|--------------|----------|------|-------------|
| 199          | 0xC7     | NeutralTech | Tech in place |
| 200          | 0xC8     | ForwardTech | Tech roll forward |
| 201          | 0xC9     | BackwardTech | Tech roll backward |
| 202          | 0xCA     | WallTech | Wall tech |
| 247          | 0xF7     | WallTechFail | Missed wall tech |

**Tech Success Rate:** (Neutral + Forward + Backward + Wall) / (All techs + Misses)

### Grabs & Throws (212-230)

#### Grabbing (212-218)
| ID (Decimal) | ID (Hex) | Name | Description |
|--------------|----------|------|-------------|
| 212          | 0xD4     | Catch | Standing grab |
| 214          | 0xD6     | CatchDash | Dash grab |
| 216          | 0xD8     | CatchWait | Holding opponent |
| 217          | 0xD9     | Pummel | Pummeling |

#### Throws (219-222)
| ID (Decimal) | ID (Hex) | Name | Description |
|--------------|----------|------|-------------|
| 219          | 0xDB     | ThrowForward | Forward throw |
| 220          | 0xDC     | ThrowBack | Back throw |
| 221          | 0xDD     | ThrowUp | Up throw |
| 222          | 0xDE     | ThrowDown | Down throw |

#### Being Grabbed (226-230)
| ID (Decimal) | ID (Hex) | Name | Description |
|--------------|----------|------|-------------|
| 226-230      | 0xE2-0xE6| Capture states | Being grabbed/held |

**Grab Success:** Catch/CatchDash (212/214) → CatchWait (216)

### Defensive Options (233-236)

| ID (Decimal) | ID (Hex) | Name | Description |
|--------------|----------|------|-------------|
| 233          | 0xE9     | RollForward | Forward roll |
| 234          | 0xEA     | RollBackward | Backward roll |
| 235          | 0xEB     | SpotDodge | Spot dodge |
| 236          | 0xEC     | EscapeAir | Air dodge |

### Ledge States (252-263)
| ID (Decimal) | ID (Hex) | Name | Description |
|--------------|----------|------|-------------|
| 252          | 0xFC     | CliffCatch | Grabbing ledge |
| 253-263      | 0xFD-0x107| Ledge options | Ledge hang, getup, roll, jump, attack |

## Character-Specific Action States

Some characters have unique action state IDs for special moves and character-specific attacks.

### Fox Special Moves
| ID (Decimal) | Name | Description |
|--------------|------|-------------|
| 342-346      | Blaster | Laser gun |
| 347-352      | Illusion | Side-B (Fox Illusion) |
| 353-358      | FireFox | Up-B (recovery) |
| 359-363      | Reflector | Down-B (Shine) |

**Example:** Detecting Fox shine usage:
```sql
WHERE action_state BETWEEN 359 AND 363  -- Shine states
```

### Game & Watch Quirks
Game & Watch has non-standard IDs for some attacks:
| ID (Decimal) | ID (Hex) | Name |
|--------------|----------|------|
| 341          | 0x155    | Jab1 |
| 342          | 0x156    | JabM (rapid jabs) |
| 345          | 0x159    | D-tilt |
| 346          | 0x15A    | F-smash |
| 347-349      | 0x15B-0x15D| Nair, Bair, Uair |

### Peach F-Smash Variants
Peach has 3 different F-smash animations (random):
| ID (Decimal) | ID (Hex) | Name |
|--------------|----------|------|
| 349          | 0x15D    | Golf Club |
| 350          | 0x15E    | Frying Pan |
| 351          | 0x15F    | Tennis Racket |

## Common Patterns & Ranges

### Combat States
```sql
-- Player is being comboed
WHERE action_state BETWEEN 75 AND 92  -- Damage
   OR action_state = 38               -- DamageFall
   OR action_state BETWEEN 226 AND 230  -- Grabbed

-- Player is attacking
WHERE action_state BETWEEN 44 AND 69  -- All attacks

-- Player is in control (grounded)
WHERE action_state BETWEEN 14 AND 23  -- Wait/walk/dash/run
   AND action_state != 24             -- Exclude jump squat
```

### State Categories (SQL Helper)
```sql
-- Create a categorization function
CREATE OR REPLACE MACRO get_action_category(state INTEGER) AS (
  CASE
    WHEN state BETWEEN 0 AND 11 THEN 'dead'
    WHEN state BETWEEN 14 AND 23 THEN 'grounded_neutral'
    WHEN state BETWEEN 24 AND 43 THEN 'airborne'
    WHEN state BETWEEN 44 AND 47 THEN 'jab'
    WHEN state = 50 THEN 'dash_attack'
    WHEN state BETWEEN 51 AND 57 THEN 'tilt'
    WHEN state BETWEEN 58 AND 64 THEN 'smash'
    WHEN state BETWEEN 65 AND 69 THEN 'aerial'
    WHEN state BETWEEN 70 AND 74 THEN 'aerial_landing'
    WHEN state BETWEEN 75 AND 92 OR state = 38 THEN 'damage'
    WHEN state BETWEEN 183 AND 204 THEN 'tech_down'
    WHEN state BETWEEN 212 AND 222 THEN 'grab_throw'
    WHEN state BETWEEN 226 AND 230 THEN 'being_grabbed'
    WHEN state BETWEEN 233 AND 236 THEN 'defensive'
    WHEN state BETWEEN 252 AND 263 THEN 'ledge'
    WHEN state BETWEEN 342 AND 396 THEN 'special_move'
    ELSE 'other'
  END
);

-- Usage:
SELECT
  get_action_category(action_state) AS category,
  COUNT(*) AS frame_count
FROM read_parquet('replay_parquet_test/partition_*.parquet')
GROUP BY category;
```

## Action State Frame Counter

The `action_state_frame` column tracks how many frames have been spent in the current action state:

```sql
-- Detect fresh action (just entered state)
WHERE action_state_frame = 0

-- Detect long-duration actions
WHERE action_state_frame > 30  -- More than 0.5 seconds

-- Detect action state transitions
WHERE action_state_frame < LAG(action_state_frame) OVER (
  PARTITION BY replay_file, player_index ORDER BY frame_number
)  -- Frame counter reset indicates new action
```

## Example Queries

### Action State Distribution
```sql
SELECT
  action_state,
  get_action_category(action_state) AS category,
  COUNT(*) AS frame_count,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percent_of_total
FROM read_parquet('replay_parquet_test/partition_*.parquet')
WHERE character_id = 1  -- Fox only
GROUP BY action_state, category
ORDER BY frame_count DESC
LIMIT 20;
```

### Average Time in Each State
```sql
SELECT
  get_action_category(action_state) AS category,
  ROUND(AVG(action_state_frame), 2) AS avg_frames_per_occurrence,
  COUNT(DISTINCT CASE WHEN action_state_frame = 0 THEN frame_number END) AS occurrences
FROM read_parquet('replay_parquet_test/partition_*.parquet')
GROUP BY category
ORDER BY avg_frames_per_occurrence DESC;
```

## References

- Full action state list: `/Users/eppie/PycharmProjects/nano-melee/action_state_mapping.csv`
- Slippi action state constants: [slippi-js common.ts](https://github.com/project-slippi/slippi-js/blob/master/src/common/stats/common.ts)
- Melee frame data: [ikneedata.com](https://ikneedata.com/)
- ActionState enum: [slippi-wiki](https://github.com/project-slippi/slippi-wiki/blob/master/SPEC.md)
