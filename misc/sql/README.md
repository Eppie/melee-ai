# SQL Statistics Queries for Slippi Replay Analysis

This directory contains SQL queries that replicate [slippi-js](https://github.com/project-slippi/slippi-js) statistics computations and [slippistats](https://github.com/ananas-dev/slippistats) enhanced analytics from parquet replay data. These queries analyze Super Smash Bros. Melee replays frame-by-frame to extract competitive gameplay statistics.

## Overview

The queries process frame-by-frame parquet data extracted from `.slp` replay files and compute various statistics:

### Core Statistics (from slippi-js)

| Query File | Purpose | Output Type |
|------------|---------|-------------|
| `stats_combos.sql` | Combo detection with damage tracking | Multiple rows per player (one per combo) |
| `stats_conversions.sql` | Punish sequences and neutral wins | Multiple rows per player (one per conversion) |
| `stats_actions.sql` | Technical skills (wavedash, L-cancel, etc.) | One row per player |
| `stats_inputs.sql` | Input counting and APM metrics | One row per player |
| `stats_stocks.sql` | Stock lifecycle tracking | Multiple rows per player (one per stock) |
| `stats_overall.sql` | Aggregate gameplay summary | One row per player |

### Enhanced Analytics (from slippistats)

| Query File | Purpose | Output Type |
|------------|---------|-------------|
| `stats_wavedash_detailed.sql` | Wavedash/waveland with angle, direction, timing | Multiple rows per player (one per wavedash) |
| `stats_dash_detailed.sql` | Dash distance and dashdance detection | Multiple rows per player (one per dash) |
| `stats_tech_detailed.sql` | Tech with positional analysis and punish tracking | Multiple rows per player (one per tech) |
| `stats_di_sdi.sql` | DI/SDI/ASDI analysis with efficacy metrics | Multiple rows per player (one per hit) |
| `stats_lcancel_detailed.sql` | L-cancel with timing, hitlag, fastfall detection | Multiple rows per player (one per L-cancel) |
| `stats_shield_drops.sql` | Shield drop with OOS frame timing | Multiple rows per player (one per shield drop) |

## Prerequisites

### Install DuckDB

**macOS:**
```bash
brew install duckDB

**Linux:**
```bash
wget https://github.com/duckdb/duckdb/releases/latest/download/duckdb_cli-linux-amd64.zip
unzip duckdb_cli-linux-amd64.zip
sudo mv duckdb /usr/local/bin/
```

**Windows:**
```bash
# Download from https://github.com/duckdb/duckdb/releases
# Add to PATH
```

### Data Requirements

The queries expect parquet data in the following location:
```
replay_parquet_test/partition_*.parquet
```

If your data is elsewhere, edit the `FROM read_parquet('...')` paths in each query file.

## Quick Start

### Running a Single Query

```bash
# Run combo analysis
duckdb < sql/stats_combos.sql

# Save results to CSV
duckdb -c ".mode csv" < sql/stats_combos.sql > combos.csv

# Save results to JSON
duckdb -c ".mode json" < sql/stats_combos.sql > combos.json

# Pretty-print results
duckdb -c ".mode box" < sql/stats_combos.sql
```

### Filtering Results

Each query supports filtering by replay file and character. Edit the query file and uncomment the appropriate WHERE clause:

**Filter by replay file:**
```sql
-- In the base_data CTE or first FROM clause, add:
WHERE replay_file = 'master-diamond-000543d6034bf69893c60a05.slp'

-- Or filter multiple files:
WHERE replay_file IN ('file1.slp', 'file2.slp', 'file3.slp')

-- Or pattern matching:
WHERE replay_file LIKE '%diamond%'
```

**Filter by character:**
```sql
-- Fox only:
WHERE character_id = 1

-- Multiple characters:
WHERE character_id IN (1, 2, 20)  -- Fox, Captain Falcon, Young Link

-- See helpers/character_ids.md for full character ID mapping
```

## Query Details

### 1. Combos (`stats_combos.sql`)

Detects combo sequences where the opponent is in hitstun or damage states.

**Key Metrics:**
- Combo duration (frames and seconds)
- Total damage dealt
- Number of moves landed
- Whether combo resulted in a kill
- Opening type classification

**Reset Logic:** Combo ends after 45 frames without opponent in hitstun/damage state.

**Example Output:**
```
replay_file                              | player_index | combo_id | duration_seconds | total_damage | did_kill
-----------------------------------------|--------------|----------|------------------|--------------|----------
master-diamond-000543d6034bf69893c60a05.slp | 1          | 1        | 2.35             | 42.5         | false
master-diamond-000543d6034bf69893c60a05.slp | 1          | 2        | 3.12             | 68.3         | true
```

### 2. Conversions (`stats_conversions.sql`)

Tracks punish sequences from opening hit until opponent regains neutral control.

**Key Metrics:**
- Conversion duration
- Opening type (neutral-win, counter-attack, trade)
- Total damage dealt
- Kill detection
- Number of distinct moves used

**Reset Logic:** Conversion ends 45 frames after opponent regains control.

**Opening Types:**
- **neutral-win**: Clean hit from neutral game
- **counter-attack**: Punish after being hit yourself
- **trade**: Simultaneous exchanges

### 3. Actions (`stats_actions.sql`)

Detects technical skill executions across all players.

**Detected Techniques:**
- **Movement**: Wavedash, waveland, dash dance
- **Defense**: L-cancel (success/fail %), techs (success/fail %), rolls, spot dodge, air dodge
- **Offense**: Grabs (success/fail %), all attack types (jabs, tilts, smashes, aerials)
- **Ledge**: Ledge grabs
- **Throws**: Up/forward/back/down throws

**Example Output:**
```
player_index | wavedash_count | lcancel_success_rate | grab_success_rate | total_attacks
-------------|----------------|----------------------|-------------------|---------------
1            | 47             | 87.5                 | 65.2              | 123
2            | 23             | 45.8                 | 42.1              | 98
```

### 4. Inputs (`stats_inputs.sql`)

Counts input changes frame-to-frame.

**Metrics:**
- Individual button press counts (A, B, X, Y, Z, L, R)
- Joystick/C-stick movements (9-region detection)
- Trigger activations (0.3 threshold)
- APM (actions per minute)
- Digital APM (buttons only)

**Stick Regions:** Divides analog sticks into 9 regions (8 directions + neutral). Only counts transitions to non-neutral regions.

**Example Output:**
```
player_index | total_inputs | actions_per_minute | digital_inputs_per_minute
-------------|--------------|--------------------|--------------------------
1            | 1247         | 254.3              | 156.7
2            | 983          | 201.5              | 124.3
```

### 5. Stocks (`stats_stocks.sql`)

Tracks individual stock lifecycle from spawn to death.

**Metrics:**
- Stock start/end frames
- Duration
- Percent at start, max reached, and end
- Death animation (action state)
- Death type classification (blast_zone, survived, other)

**Example Output:**
```
player_index | stock_id | duration_seconds | start_percent | max_percent | death_type
-------------|----------|------------------|---------------|-------------|------------
1            | 1        | 87.5             | 0             | 143         | blast_zone
1            | 2        | 62.3             | 0             | 98          | blast_zone
1            | 3        | 45.2             | 0             | 67          | survived
```

### 6. Overall Stats (`stats_overall.sql`)

Aggregate high-level gameplay statistics per player.

**Metrics:**
- Game duration
- Total damage dealt/taken
- Kill/death counts
- Final stocks remaining
- Average/max percent
- Time in hitstun (%)
- Time on ground vs airborne (%)
- Time off-stage (%)

**Example Output:**
```
player_index | total_damage_dealt | kill_count | death_count | time_in_hitstun_percent
-------------|--------------------|-----------|--------------|-----------------------
1            | 287.5              | 4         | 2            | 12.3
2            | 234.8              | 2         | 4            | 18.7
```

---

## Enhanced Analytics Queries

### 7. Wavedash Detailed (`stats_wavedash_detailed.sql`)

Analyzes wavedash and waveland techniques with detailed metrics.

**Metrics:**
- Wavedash type (wavedash vs waveland)
- Angle in degrees below horizontal (0° = horizontal, 90° = straight down)
- Direction (LEFT, RIGHT, DOWN)
- Trigger frame timing (frames from jump squat to trigger press)
- Airdodge duration (frames from trigger to landing)
- Starting position and horizontal distance traveled

**Detection Logic:**
- **Wavedash:** KneeBend (24) → EscapeAir (236) → LandingFallSpecial (43) within 8 frames
- **Waveland:** EscapeAir (236) → LandingFallSpecial (43) without KneeBend

**Example Output:**
```
player_index | frame_number | wavedash_type | angle_degrees | direction | horizontal_distance
-------------|--------------|---------------|---------------|-----------|-------------------
1            | 1234         | wavedash      | 23.5          | RIGHT     | 4.8
1            | 2456         | waveland      | 18.2          | LEFT      | 3.2
```

**Use Cases:**
- Optimize wavedash angles for different situations
- Compare wavedash usage between players/characters
- Identify optimal wavedash directions on different stages

### 8. Dash Detailed (`stats_dash_detailed.sql`)

Tracks dash sequences with positional data and dashdance detection.

**Metrics:**
- Start/end position (X, Y coordinates)
- Distance traveled (Euclidean distance)
- Direction (LEFT, RIGHT, NEUTRAL)
- Dashdance flag (rapid back-and-forth dashing)
- Duration and average velocity

**Detection Logic:**
- **Dash:** Entering Dash (20) or Run (21) action state
- **Dashdance:** Dash → Turn (18) → Dash in opposite direction within 10 frames

**Example Output:**
```
player_index | dash_id | distance | direction | is_dashdance | duration_frames
-------------|---------|----------|-----------|--------------|----------------
1            | 1       | 8.5      | RIGHT     | false        | 15
1            | 2       | 6.2      | LEFT      | true         | 12
1            | 3       | 7.8      | RIGHT     | true         | 14
```

**Use Cases:**
- Analyze neutral game movement patterns
- Identify dashdance frequency and effectiveness
- Compare dash distances between characters

### 9. Tech Detailed (`stats_tech_detailed.sql`)

Enhanced tech analysis with positional strategy and punish tracking.

**Metrics:**
- Tech type (neutral, forward, backward, wall, miss)
- Position (X, Y coordinates)
- Directional analysis:
  - `towards_center`: Tech direction moves toward stage center
  - `towards_opponent`: Tech direction moves toward opponent
- Punish detection (hit within 30 frames after tech)
- Jab reset detection (opponent jabs after missed tech)

**Example Output:**
```
player_index | tech_type      | towards_center | towards_opponent | was_punished | frames_until_punish
-------------|----------------|----------------|------------------|--------------|--------------------
1            | forward_tech   | true           | false            | false        | NULL
1            | tech_miss_down | NULL           | NULL             | true         | 8
```

**Use Cases:**
- Evaluate tech skill and decision-making
- Identify optimal tech directions for different positions
- Analyze tech chase success rates

### 10. DI/SDI Analysis (`stats_di_sdi.sql`)

Analyzes Directional Influence (DI) and Smash DI (SDI) inputs.

**Metrics:**
- Hitlag duration (frames)
- DI stick position and angle
- SDI input count (stick region changes during hitlag)
- Knockback angle before/after DI
- DI efficacy (angle change in degrees)
- Crouch cancel detection (down DI at low percent)

**Detection Logic:**
- **Hitlag:** Frames in hitstun before knockback velocity starts
- **DI:** Joystick angle during hitlag frames
- **SDI:** Count stick region changes during hitlag
- **Crouch Cancel:** Percent < 40% AND joystick_y < 0.35

**Example Output:**
```
player_index | hit_frame | damage_taken | hitlag_frames | di_angle_degrees | kb_angle_after | is_crouch_cancel
-------------|-----------|--------------|---------------|------------------|----------------|------------------
1            | 3456      | 12.5         | 3             | 45.2             | 32.8           | false
1            | 4567      | 8.3          | 2             | -88.5            | -72.1          | true
```

**Use Cases:**
- Evaluate DI quality and consistency
- Identify crouch cancel opportunities
- Analyze survival DI effectiveness

**Note:** True DI efficacy requires hitbox data (attack angles, knockback). Our implementation uses approximations based on velocity vectors.

### 11. L-Cancel Detailed (`stats_lcancel_detailed.sql`)

Enhanced L-cancel analysis with timing and context.

**Metrics:**
- Aerial type (nair, fair, bair, uair, dair)
- L-cancel success/failure
- Trigger input frame (timing relative to landing, e.g., -1 = 1 frame before)
- During hitlag flag (common mistake)
- Fastfall status (velocity_y < -2.0)
- Estimated landing lag frames

**Example Output:**
```
player_index | aerial_type | lcancel_success | trigger_input_frame | during_hitlag | is_fastfall
-------------|-------------|-----------------|---------------------|---------------|-------------
1            | nair        | true            | -2                  | false         | true
1            | dair        | false           | NULL                | true          | true
```

**Use Cases:**
- Identify L-cancel timing patterns
- Detect common mistakes (L-canceling during hitlag)
- Correlate fastfall usage with L-cancel success

### 12. Shield Drops (`stats_shield_drops.sql`)

Tracks shield drop technique with out-of-shieldstun timing.

**Metrics:**
- Shield duration before drop (frames)
- Out-of-shieldstun frame (when shieldstun ended)
- Frames after shieldstun to shield drop
- Position (X, Y)
- Platform detection (approximate based on Y position)

**Detection Logic:**
- **Shield states:** GuardOn/Guard/GuardOff (178-182)
- **Shield drop:** Transition from shield to falling state (29, 30, 31, 358)
- **OOS frame:** Hitstun_remaining transitions from >0 to 0 while in shield

**Example Output:**
```
player_index | shield_drop_frame | shield_duration_frames | oo_shieldstun_frame | frames_after_shieldstun | platform_detected
-------------|-------------------|------------------------|---------------------|-----------------------|-------------------
1            | 5678              | 45                     | 5650                | 28                     | true
```

**Use Cases:**
- Analyze shield pressure escape options
- Identify optimal shield drop timing after shieldstun
- Compare shield drop usage on different platforms

---

## Technical Documentation

For detailed information on detection methods and technical conditions:

- **`helpers/detection_methods.md`** - Complete technical specifications for all detection logic
- **`helpers/action_states.md`** - Action state reference and categorization
- **`helpers/character_ids.md`** - Character ID mapping
- **`helpers/example_usage.sh`** - Example commands and workflows

---

## Advanced Usage

### Combining Multiple Queries

```bash
# Create a comprehensive stats report
duckdb << EOF
-- Create temp tables from each query
CREATE TEMP TABLE combos AS SELECT * FROM 'sql/stats_combos.sql';
CREATE TEMP TABLE inputs AS SELECT * FROM 'sql/stats_inputs.sql';
CREATE TEMP TABLE overall AS SELECT * FROM 'sql/stats_overall.sql';

-- Join and analyze
SELECT
  o.replay_file,
  o.player_index,
  o.total_damage_dealt,
  i.actions_per_minute,
  COUNT(c.combo_id) AS combo_count
FROM overall o
LEFT JOIN inputs i USING (replay_file, player_index)
LEFT JOIN combos c USING (replay_file, player_index)
GROUP BY o.replay_file, o.player_index, o.total_damage_dealt, i.actions_per_minute;
EOF
```

### Filtering by Multiple Criteria

```sql
-- In the first FROM clause of any query, add:
WHERE character_id = 1  -- Fox only
  AND replay_file LIKE '%diamond%'  -- Diamond rank replays only
  AND frame_number >= 0  -- Exclude pre-game frames
```

### Exporting to Different Formats

```bash
# CSV with headers
duckdb -csv -header < sql/stats_actions.sql > actions.csv

# Tab-separated
duckdb -separator '\t' < sql/stats_actions.sql > actions.tsv

# JSON array
duckdb -json < sql/stats_actions.sql > actions.json

# Parquet (for further processing)
duckdb -c "COPY (SELECT * FROM 'sql/stats_actions.sql') TO 'actions_output.parquet' (FORMAT PARQUET);"
```

### Performance Tips

**For large datasets (1000+ games):**

1. **Add indexes** (if using persistent DuckDB database):
```sql
CREATE INDEX idx_replay_player ON parquet_data(replay_file, player_index);
CREATE INDEX idx_frame ON parquet_data(replay_file, frame_number);
```

2. **Filter early** in the query:
```sql
-- Add WHERE clause in first CTE to reduce data processed
WHERE replay_file IN ('file1.slp', 'file2.slp')  -- Specific files only
```

3. **Use DuckDB persistent mode** instead of CLI:
```python
import duckdb
conn = duckdb.connect('melee_stats.db')
conn.execute("CREATE VIEW replays AS SELECT * FROM 'replay_parquet_test/*.parquet'")
result = conn.execute(open('sql/stats_combos.sql').read()).fetchdf()
```

## Reference Tables

See the `helpers/` directory for reference documentation:

- `character_ids.md` - Character ID to name mapping
- `action_states.md` - Action state categories and common IDs
- `example_usage.sh` - Shell script with example commands

## Troubleshooting

### "Table does not exist" error

**Problem:** DuckDB can't find the parquet files.

**Solution:** Edit the `FROM read_parquet('...')` path in the query to match your data location:
```sql
FROM read_parquet('/absolute/path/to/replay_parquet_test/partition_*.parquet')
```

### Incorrect combo/conversion counts

**Problem:** Counts seem too high or too low.

**Solution:** The 45-frame reset timer is based on Melee's 60 FPS. Adjust the reset logic if needed:
```sql
-- Change 45 to different value (in frames)
OR (frame_number - LAG(frame_number) OVER w) > 45  -- Try 30, 60, etc.
```

### Missing wavedash detections

**Problem:** Known wavedashes aren't being detected.

**Solution:** The pattern matching window is 8 frames. Adjust in `stats_actions.sql`:
```sql
AND (frame_number - COALESCE(prev_frame_2, frame_number)) <= 8  -- Try 10, 12
```

### Performance issues

**Problem:** Queries take too long.

**Solution:**
1. Add character/replay filtering to reduce data
2. Use DuckDB persistent mode instead of CLI
3. Process data in batches using LIMIT/OFFSET
4. Ensure parquet files are properly partitioned

## Statistical Notes

### Frame Timing

All timing calculations assume 60 FPS (Melee's native framerate):
- 1 second = 60 frames
- 1 minute = 3600 frames

### Reset Timers

Both combos and conversions use a 45-frame reset timer (0.75 seconds), matching slippi-js behavior:
- **Combos**: Reset when opponent not in hitstun/damage for 45 frames
- **Conversions**: Reset 45 frames after opponent regains neutral control

### Action State Ranges

Key action state ID ranges (see `helpers/action_states.md` for full list):
- Damage: 75-92 (0x4B-0x5C)
- Grabs: 212, 214 (0xD4, 0xD6)
- Grabbed: 226-230 (0xE2-0xE6)
- Techs: 199-204 (0xC7-0xCC)
- Attacks: 44-69 (0x2C-0x45)

## Comparison with slippi-js and slippistats

These SQL queries replicate statistics from both slippi-js and slippistats:

### From slippi-js

| slippi-js Stat | SQL Query | Notes |
|----------------|-----------|-------|
| `combos` | `stats_combos.sql` | Same 45-frame reset logic |
| `conversions` | `stats_conversions.sql` | Includes opening type classification |
| `actionCounts` | `stats_actions.sql` | All technical skills covered |
| `inputCounts` | `stats_inputs.sql` | Matches slippi-js input detection |
| `stocks` | `stats_stocks.sql` | Complete stock lifecycle |
| `overall` | `stats_overall.sql` | High-level aggregates |

### From slippistats

| slippistats Feature | SQL Query | Notes |
|---------------------|-----------|-------|
| `WavedashData` | `stats_wavedash_detailed.sql` | Angle, direction, timing metrics |
| `DashData` | `stats_dash_detailed.sql` | Distance, dashdance detection |
| `TechData` | `stats_tech_detailed.sql` | Positional analysis, punish tracking |
| `TakeHitData` | `stats_di_sdi.sql` | DI/SDI/ASDI analysis, crouch cancel |
| `LCancelData` | `stats_lcancel_detailed.sql` | Timing, hitlag, fastfall detection |
| `ShieldDropData` | `stats_shield_drops.sql` | OOS timing, platform detection |

**Not Included:**
- Target Test mode statistics (per project requirements)
- Home Run Contest mode statistics

## Contributing

To add new statistics or improve existing queries:

1. Follow the existing query structure (CTEs with comments)
2. Use consistent naming conventions
3. Add filtering support (replay_file, character_id)
4. Document the output schema in comments
5. Test with sample data

## License

These queries are part of the nano-melee project. See the main project README for license information.
