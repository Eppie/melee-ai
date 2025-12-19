# Detection Methods and Technical Conditions

This document details the exact technical conditions used to detect techniques and compute statistics in the SQL queries.

## Table of Contents
- [Movement Techniques](#movement-techniques)
- [Combo and Conversion Detection](#combo-and-conversion-detection)
- [Input Techniques](#input-techniques)
- [Defensive Techniques](#defensive-techniques)
- [Advanced Metrics](#advanced-metrics)

---

## Movement Techniques

### Wavedash Detection

**Query:** `stats_wavedash_detailed.sql`

**Definition:** A wavedash is an airdodge into the ground from a jump squat, creating horizontal momentum.

**Technical Conditions:**
```sql
-- Action state sequence within 8 frames:
1. KneeBend (action_state = 24)           -- Jump squat
2. EscapeAir (action_state = 236)         -- Air dodge
3. LandingFallSpecial (action_state = 43) -- Special landing

-- Time constraint:
(landing_frame - jump_squat_frame) <= 8
```

**Key Details:**
- **Maximum duration:** 8 frames from jump squat to landing
- **Distinguishing wavedash vs waveland:** Wavedash has KneeBend before EscapeAir; waveland does not
- **Angle calculation:** `atan2(joystick_y - 0.5, joystick_x - 0.5)` during EscapeAir frame, converted to degrees
- **Direction:** Based on joystick_x during airdodge:
  - `joystick_x < 0.35` → LEFT
  - `joystick_x > 0.65` → RIGHT
  - `0.35 <= joystick_x <= 0.65` → DOWN (spotdodge wavedash)

**Output Metrics:**
- `angle_degrees`: Angle below horizontal (0° = horizontal, 90° = straight down)
- `direction`: LEFT/RIGHT/DOWN
- `trigger_frame`: Frames between KneeBend and EscapeAir
- `airdodge_frames`: Frames between EscapeAir and landing

---

### Waveland Detection

**Query:** `stats_wavedash_detailed.sql`

**Definition:** An airdodge into the ground from an airborne state (not from jump squat).

**Technical Conditions:**
```sql
-- Action state sequence within 4 frames:
1. EscapeAir (action_state = 236)         -- Air dodge (no KneeBend before)
2. LandingFallSpecial (action_state = 43) -- Special landing

-- Time constraint:
(landing_frame - airdodge_frame) <= 4

-- Exclusion:
prev_action_2 != 24  -- NOT from KneeBend
```

**Key Difference from Wavedash:**
- No jump squat (KneeBend) in the 2 frames before airdodge
- Typically happens from full hop, short hop, or platform drop

---

### Dash Detection

**Query:** `stats_dash_detailed.sql`

**Definition:** A dash is entering the dash or run action state.

**Technical Conditions:**
```sql
-- Dash start:
action_state IN (20, 21)  -- Dash or Run
AND prev_action_state NOT IN (20, 21)

-- Dash end:
action_state NOT IN (20, 21)
AND prev_action_state IN (20, 21)
```

**Output Metrics:**
- `distance`: Euclidean distance from start to end position
- `direction`: LEFT/RIGHT/NEUTRAL based on `(end_x - start_x)`
- `duration_frames`: Frames spent in dash/run state

---

### Dashdance Detection

**Query:** `stats_dash_detailed.sql`

**Definition:** Rapid alternating dashes in opposite directions.

**Technical Conditions:**
```sql
-- Current dash must satisfy ALL:
1. Preceded by Turn action (action_state = 18) within last 3 frames
2. Previous dash direction is opposite:
   - prev_direction = 'LEFT' AND current_direction = 'RIGHT'
   - OR prev_direction = 'RIGHT' AND current_direction = 'LEFT'
3. Gap between dashes < 10 frames
4. Both directions are non-neutral
```

**Key Details:**
- Turn (action_state = 18) is the pivot animation
- Gap of < 10 frames ensures rapid back-and-forth movement
- First dash in a sequence is never marked as dashdance (requires previous dash)

---

## Combo and Conversion Detection

### Combo Detection

**Query:** `stats_combos.sql`

**Definition:** A sequence of hits where the opponent is in hitstun or a damage state.

**Technical Conditions:**
```sql
-- Opponent is "in combo" if ANY of these conditions:
1. defender_hitstun > 0
2. defender_in_hitstun = TRUE
3. defender_action_state BETWEEN 75 AND 92   -- Damage states (0x4B-0x5C)
4. defender_action_state BETWEEN 226 AND 230 -- Grabbed (0xE2-0xE6)
5. defender_action_state BETWEEN 219 AND 222 -- Being thrown (0xDB-0xDE)
6. defender_action_state = 38                -- DamageFall (0x26)

-- Combo grouping:
- Consecutive combo frames are grouped together
- 45-frame reset timer: gap > 45 frames starts new combo
- Minimum 2 frames to count as combo
```

**Output Metrics:**
- `total_damage`: Sum of positive percent deltas during combo
- `moves_landed`: Count of distinct attacker action states
- `did_kill`: Defender stock count decreased during combo
- `opening_type`: Classification of how combo started:
  - `neutral-win`: Defender percent < 20 at start
  - `counter-attack`: High hitstun values (> 15 frames)
  - `trade`: Other

---

### Conversion Detection

**Query:** `stats_conversions.sql`

**Definition:** A sequence where the opponent is in any disadvantage state (broader than combo).

**Technical Conditions:**
```sql
-- Opponent is "in disadvantage" if ANY:
1. All combo conditions (hitstun, damage, grabbed, thrown)
2. defender_action_state BETWEEN 183 AND 198  -- Tech/down states (0xB7-0xC6)
3. defender_action_state BETWEEN 199 AND 204  -- Teching (0xC7-0xCC)

-- Conversion grouping:
- 45-frame reset timer (same as combos)
- Includes post-hit advantage (teching, getting up from ground)
- Ends when opponent regains neutral control
```

**Key Difference from Combo:**
- Combos only count true hitstun/damage frames
- Conversions include followup attempts after hitstun ends (tech chases, etc.)
- Conversions are typically longer than combos

---

## Input Techniques

### L-Cancel Detection

**Query:** `stats_lcancel_detailed.sql`

**Definition:** Pressing L/R before landing from an aerial to reduce landing lag.

**Technical Conditions:**
```sql
-- L-cancel opportunity:
action_state IN (70, 71, 72, 73, 74)  -- Aerial landing states (0x46-0x4A)
AND action_state_frame = 0
AND lcancel_status IN (1, 2)  -- 1 = success, 2 = fail

-- Success:
lcancel_status = 1

-- Failure:
lcancel_status = 2
```

**Note:** The `lcancel_status` column in the parquet data is derived from the Slippi replay parser and directly indicates success/failure.

**Enhanced Metrics (stats_lcancel_detailed.sql):**
- `trigger_input_frame`: Which frame (relative to landing) had trigger press
  - Detects trigger press in window of 7 frames before landing
  - `0` = pressed on landing frame, `-1` = 1 frame before, etc.
- `during_hitlag`: Trigger pressed while in hitstun (common mistake)
- `is_fastfall`: Player was fastfalling (velocity_y < -2.0) during aerial
- `landing_lag_frames`: Estimated lag frames (character/aerial specific)

**Aerial Landing State Mapping:**
- 70 (0x46) = Nair landing
- 71 (0x47) = Fair landing
- 72 (0x48) = Bair landing
- 73 (0x49) = Uair landing
- 74 (0x4A) = Dair landing

---

### Grab Detection

**Query:** `stats_actions.sql`

**Definition:** Attempting to grab an opponent.

**Technical Conditions:**
```sql
-- Grab attempt:
action_state IN (212, 214)  -- Catch (0xD4) or CatchDash (0xD6)
AND (action_changed = 1 OR action_frame_reset = 1)

-- Grab success:
action_state = 216  -- CatchWait (0xD8) - holding opponent
AND (action_changed = 1 OR action_frame_reset = 1)
```

**Success Rate Calculation:**
```sql
grab_success_rate = grab_successes / NULLIF(grab_attempts, 0)
```

**Key Details:**
- Standing grab: action_state = 212
- Dash grab: action_state = 214
- Success indicated by reaching CatchWait (216)
- Failure: grab whiffs and returns to neutral

---

### Tech Detection

**Query:** `stats_tech_detailed.sql`

**Definition:** Pressing L/R within 20 frames before hitting the ground/wall to tech.

**Technical Conditions:**
```sql
-- Tech success:
action_state IN (199, 200, 201, 202)
- 199 (0xC7) = NeutralTech (tech in place)
- 200 (0xC8) = ForwardTech (tech roll forward)
- 201 (0xC9) = BackwardTech (tech roll backward)
- 202 (0xCA) = WallTech (tech on wall)

-- Tech miss:
action_state IN (183, 191, 247)
- 183 (0xB7) = TechMissUp (missed tech, face up)
- 191 (0xBF) = TechMissDown (missed tech, face down)
- 247 (0xF7) = WallTechFail (missed wall tech)
```

**Enhanced Metrics (stats_tech_detailed.sql):**
- `towards_center`: Tech direction moves toward stage center (X=0)
  - For forward tech: `(pos_x > 0 AND facing < 0) OR (pos_x < 0 AND facing > 0)`
  - For backward tech: opposite logic
- `towards_opponent`: Tech direction moves toward opponent position
- `was_punished`: Player enters hitstun within 30 frames after tech
- `jab_reset`: Opponent uses jab (action 44-47) within 10 frames after tech miss

**Tech Success Rate:**
```sql
tech_success_rate = (neutral + forward + backward + wall) /
                    (neutral + forward + backward + wall + misses)
```

---

### Attack Detection

**Query:** `stats_actions.sql`

**Definition:** Using any attack move.

**Technical Conditions:**
```sql
-- Detect first frame of attack:
action_state IN [attack_range]
AND (action_changed = 1 OR action_frame_reset = 1)

-- Attack Categories:
Jabs: 44-47 (0x2C-0x2F)
  - 44 = Jab1, 45 = Jab2, 46 = Jab3, 47 = Rapid jabs

Tilts: 51-57 (0x33-0x39)
  - 51-53 = F-tilt (3 angles)
  - 54-56 = U-tilt (3 angles)
  - 57 = D-tilt

Smashes: 58-64 (0x3A-0x40)
  - 58-60 = F-smash (charge, release, charge2)
  - 61-62 = U-smash (charge, release)
  - 63-64 = D-smash (charge, release)

Aerials: 65-69 (0x41-0x45)
  - 65 = Nair, 66 = Fair, 67 = Bair, 68 = Uair, 69 = Dair

Dash Attack: 50 (0x32)
```

---

## Defensive Techniques

### Shield Drop Detection

**Query:** `stats_shield_drops.sql`

**Definition:** Dropping through a platform while in shield state.

**Technical Conditions:**
```sql
-- Shield drop:
1. Previous frame: action_state BETWEEN 178 AND 182  -- Shield states
   - 178 (0xB2) = GuardOn
   - 179 (0xB3) = Guard
   - 180 (0xB4) = GuardOff
   - 181 (0xB5) = GuardSetOff
   - 182 (0xB6) = GuardReflect (powershield)

2. Current frame: action_state IN (29, 30, 31, 358)  -- Fall states
   - 29 = Fall
   - 30 = FallF
   - 31 = FallB
   - 358 (0x166) = PassDown

3. position_y > -20  -- Not at bottom blast zone (indicates on platform)
```

**Enhanced Metrics:**
- `oo_shieldstun_frame`: Frame when shieldstun ended (hitstun_remaining: >0 → 0)
- `frames_after_shieldstun`: Frames between shieldstun end and shield drop
- `platform_detected`: Approximate platform detection (position_y > 10)

**Key Details:**
- Out-of-shieldstun (OOS) frame is important for competitive play
- Fast shield drops are frame-perfect or near-frame-perfect

---

### Roll/Spotdodge Detection

**Query:** `stats_actions.sql`

**Technical Conditions:**
```sql
-- Forward roll:
action_state = 233 (0xE9)

-- Backward roll:
action_state = 234 (0xEA)

-- Spot dodge:
action_state = 235 (0xEB)

-- Air dodge (non-wavedash/waveland):
action_state = 236 (0xEC)
AND next_action != 43  -- NOT leading to LandingFallSpecial
```

---

## Advanced Metrics

### DI (Directional Influence) Detection

**Query:** `stats_di_sdi.sql`

**Definition:** Holding a direction during hitlag to influence knockback trajectory.

**Technical Conditions:**
```sql
-- Hitstun start:
is_in_hitstun = TRUE
AND prev_in_hitstun = FALSE

-- Hitlag duration:
- Count frames until velocity becomes non-zero
- Hitlag = frames where frozen before knockback starts
- Approximation: frames until |velocity_x| > 0.1 OR |velocity_y| > 0.1

-- DI angle:
di_angle_degrees = atan2(joystick_y - 0.5, joystick_x - 0.5)
- Measured during hitlag frames
- Converted to degrees

-- Knockback angle:
kb_angle = atan2(velocity_y, velocity_x)
- Measured at first frame after hitlag ends
```

**DI Efficacy Calculation:**
```sql
-- Simplified (without hitbox data):
di_efficacy = |kb_angle_after - estimated_kb_angle_before|

-- Estimated before angle:
- Upward hit: assume 90° without DI
- Downward hit: assume -90° without DI

-- True efficacy requires hitbox data (attack angle, knockback growth)
```

**Crouch Cancel Detection:**
```sql
-- Crouch cancel:
percent < 40
AND joystick_y < 0.35  -- Stick held down during hit
```

**Key Limitations:**
- True DI efficacy requires hitbox data (not available in frame data)
- Our approximation uses assumed default angles
- More accurate with move database integration

---

### SDI (Smash DI) Detection

**Query:** `stats_di_sdi.sql`

**Definition:** Quarter-circle stick inputs during hitlag to shift position.

**Technical Conditions:**
```sql
-- SDI input counting:
- Divide joystick into 9 regions (8 directions + neutral)
- Count region changes during hitlag frames
- Each region change = potential SDI input

-- Stick regions:
  'neutral': |joystick_x - 0.5| < 0.2 AND |joystick_y - 0.5| < 0.2
  'up': joystick_y > 0.65 AND joystick_x ∈ [0.35, 0.65]
  'down': joystick_y < 0.35 AND joystick_x ∈ [0.35, 0.65]
  'left': joystick_x < 0.35 AND joystick_y ∈ [0.35, 0.65]
  'right': joystick_x > 0.65 AND joystick_y ∈ [0.35, 0.65]
  'up_left': joystick_x < 0.35 AND joystick_y > 0.65
  'up_right': joystick_x > 0.65 AND joystick_y > 0.65
  'down_left': joystick_x < 0.35 AND joystick_y < 0.35
  'down_right': joystick_x > 0.65 AND joystick_y < 0.35

-- SDI count:
sdi_inputs = COUNT(DISTINCT stick_region during hitlag)
```

**Note:** True SDI detection requires frame-by-frame joystick monitoring during hitlag, which is complex with window functions. Our implementation provides a simplified count.

---

### Input Counting (APM)

**Query:** `stats_inputs.sql`

**Definition:** Counting all button presses and stick movements.

**Technical Conditions:**
```sql
-- Button press:
- False → True transition using LAG()
- button_current = TRUE AND button_previous = FALSE

-- Buttons tracked:
  - button_a, button_b, button_x, button_y, button_z
  - trigger_logical_l, trigger_logical_r

-- Joystick input:
- Region change detection (9 regions)
- joystick_region_current != joystick_region_previous
- Excludes re-entering neutral (return to deadzone doesn't count)

-- C-stick input:
- Same 9-region system as joystick
- cstick_region_current != cstick_region_previous

-- Trigger analog:
- Crossing 0.3 threshold
- trigger_analog < 0.3 → trigger_analog >= 0.3

-- APM Calculation:
apm = total_inputs / (total_frames / 60 / 60)
    = total_inputs / duration_minutes

-- Digital APM (buttons only):
digital_apm = total_button_inputs / duration_minutes
```

**Key Details:**
- Joystick/C-stick regions use 9-zone division
- Deadzone threshold: 0.35-0.65 (neutral zone)
- APM includes analog stick movements, digital APM excludes them

---

### Stock Lifecycle Tracking

**Query:** `stats_stocks.sql`

**Definition:** Tracking each stock from spawn to death.

**Technical Conditions:**
```sql
-- Stock change detection:
stocks != prev_stocks

-- Stock loss (death):
stocks < prev_stocks

-- Stock gain (respawn):
stocks > prev_stocks

-- Stock ID assignment:
- Increment on respawn OR first frame
- Each stock gets unique sequential ID per player

-- Death type classification:
CASE
  WHEN death_action_state BETWEEN 0 AND 11 THEN 'blast_zone'
  WHEN death_action_state IS NULL THEN 'survived'
  ELSE 'other'
END
```

**Death Action States:**
- 0-11 (0x00-0x0B): Various death animations (blast zones, star KO, etc.)
- NULL: Stock ended but no death detected (survived to end of game)

**Output Metrics:**
- `duration_frames`: Stock lifetime in frames
- `start_percent`, `max_percent`, `end_percent`: Damage tracking
- `death_action_state`: Animation ID at death
- `death_type`: Classified death cause

---

## Input Detection Details

### Joystick Region System

**9-Region Division:**
```
  up_left (x<0.35, y>0.65)  |  up (x:0.35-0.65, y>0.65)  |  up_right (x>0.65, y>0.65)
  -------------------------------------------------------------------------
  left (x<0.35, y:0.35-0.65) |  neutral (center)          |  right (x>0.65, y:0.35-0.65)
  -------------------------------------------------------------------------
  down_left (x<0.35, y<0.35) |  down (x:0.35-0.65, y<0.35) |  down_right (x>0.65, y<0.35)
```

**Thresholds:**
- Low: < 0.35
- Neutral: 0.35 - 0.65
- High: > 0.65

**Joystick Coordinates:**
- Range: [0.0, 1.0]
- Center (neutral): 0.5
- Stored as floats in parquet data

---

## Time Windows and Reset Timers

### 45-Frame Reset Timer

Used in: `stats_combos.sql`, `stats_conversions.sql`

**Purpose:** Group related hits into sequences while allowing brief neutral resets.

**Logic:**
```sql
-- Gap detection:
frame_gap = current_frame - previous_relevant_frame

-- Start new sequence:
IF frame_gap > 45 OR first_frame THEN
  new_sequence = TRUE
```

**Why 45 frames?**
- 45 frames ≈ 0.75 seconds (at 60 FPS)
- Long enough to allow techchases and followups
- Short enough to exclude unrelated hits
- Standard used by Slippi.js statistics

### Wavedash 8-Frame Window

**Purpose:** Maximum time from jump squat to landing for valid wavedash.

**Logic:**
```sql
(landing_frame - jump_squat_frame) <= 8
```

**Why 8 frames?**
- Jump squat: 3-6 frames (character dependent)
- Airdodge startup: 1 frame
- Landing: 1-2 frames airborne
- Total: ~5-9 frames typical, 8 frame window is lenient

### Tech Punish Window (30 frames)

**Purpose:** Determine if tech was successfully followed up on.

**Logic:**
```sql
-- Punished if hit within 30 frames:
(hit_frame - tech_frame) <= 30
```

**Why 30 frames?**
- Tech roll animation: ~20-25 frames
- Allows time for followup dash/wavedash
- Standard competitive definition

---

## Common Patterns

### Action State Transition Detection

**Pattern used throughout queries:**
```sql
-- Method 1: Action changed flag
action_changed = 1  -- First frame of new action

-- Method 2: Frame counter reset
action_state_frame = 0  -- First frame of action

-- Method 3: LAG comparison
action_state != LAG(action_state) OVER w
```

**When to use each:**
- `action_changed`: Most reliable, pre-computed in parquet extraction
- `action_state_frame = 0`: Good for landing frames, state starts
- LAG comparison: When action_changed not available

### Window Function Pattern

**Standard window for player sequential data:**
```sql
WINDOW w AS (
  PARTITION BY replay_file, player_index
  ORDER BY frame_number
)
```

**Opponent join pattern:**
```sql
FROM frames p1
INNER JOIN frames p2
  ON p1.replay_file = p2.replay_file
  AND p1.frame_number = p2.frame_number
  AND p1.player_index != p2.player_index
```

### Velocity-Based Detection

**When velocity indicates action:**
```sql
-- Fastfall detection:
velocity_y < -2.0

-- Knockback start (end of hitlag):
ABS(velocity_x) > 0.1 OR ABS(velocity_y) > 0.1

-- Stationary (in shield/tech):
ABS(velocity_x) < 0.1 AND ABS(velocity_y) < 0.1
```

---

## References

### Critical Action State Ranges

```sql
-- Deaths: 0-11
-- Neutral/movement: 14-23
-- Jumping: 24-43
-- Attacks: 44-69
-- Damage/hitstun: 75-92, 38
-- Techs: 183-204
-- Grabs/throws: 212-222, 226-230
-- Defensive: 233-236
-- Ledge: 252-263
-- Specials: 342-396 (character specific)
```

### Frame Rate

All frame counts assume 60 FPS (Melee's native framerate):
- 1 second = 60 frames
- 1 frame ≈ 16.67 ms

### Coordinate Systems

**Position:**
- X: Horizontal (left = negative, right = positive)
- Y: Vertical (down = negative, up = positive)
- Stage center typically at (0, 0)

**Velocity:**
- Same coordinate system as position
- Units: position_units per frame

**Joystick/C-stick:**
- Range: [0.0, 1.0]
- Neutral: 0.5
- Deadzone: ~0.35-0.65

### Data Limitations

**What we CAN detect:**
- All action states and transitions
- Position, velocity, percent
- Controller inputs (buttons, sticks, triggers)
- Hitstun, stocks, combo states

**What we CANNOT detect without additional data:**
- Hitbox data (attack angles, damage, knockback)
- True DI efficacy (requires attack properties)
- Frame-perfect tech windows (requires frame data database)
- Move names (only have action state IDs)
- Character-specific special move details

---

## Query Cross-Reference

| Technique | Primary Query | Helper Queries |
|-----------|---------------|----------------|
| Wavedash | stats_wavedash_detailed.sql | stats_actions.sql |
| Waveland | stats_wavedash_detailed.sql | stats_actions.sql |
| Dash/Dashdance | stats_dash_detailed.sql | stats_actions.sql |
| L-cancel | stats_lcancel_detailed.sql | stats_actions.sql |
| Tech | stats_tech_detailed.sql | stats_actions.sql |
| Grab | stats_actions.sql | - |
| Combo | stats_combos.sql | - |
| Conversion | stats_conversions.sql | - |
| DI/SDI | stats_di_sdi.sql | - |
| Shield Drop | stats_shield_drops.sql | - |
| APM/Inputs | stats_inputs.sql | - |
| Stock Lifecycle | stats_stocks.sql | - |
| Overall Stats | stats_overall.sql | All of above |
