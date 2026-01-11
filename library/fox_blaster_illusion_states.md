# Fox: Blaster & Illusion States Documentation

Action states 341-352 covering Fox's Neutral-B (Blaster) and Side-B (Fox Illusion) special moves, both grounded and aerial versions.

## Overview

These 12 states represent two of Fox's four special moves, each with ground and air variants consisting of startup, active/loop, and end phases.

| Move | Version | States | Total Frames | % of Gameplay |
|------|---------|--------|--------------|---------------|
| Blaster | Ground | 341-343 | 93,975 | 0.03% |
| Blaster | Air | 344-346 | 2,531,259 | 0.88% |
| Illusion | Ground | 347-349 | 145,457 | 0.05% |
| Illusion | Air | 350-352 | 3,578,247 | 1.24% |

**Total special move frames**: 6,348,938 (2.20% of gameplay)

### Ground vs Air Usage

| Move | Ground | Air | Air % |
|------|--------|-----|-------|
| Blaster | 93,975 | 2,531,259 | **96.4%** |
| Illusion | 145,457 | 3,578,247 | **96.1%** |

**Key insight**: Both moves are used almost exclusively in the air in competitive play.

---

## Blaster (Neutral-B) States

Fox's Blaster fires rapid red laser projectiles that deal 3% damage but cause no hitstun or flinching. This makes it a zoning/camping tool rather than an approach interrupter.

### State Summary

| State | ID | Internal Name | Instances | Median Dur | Avg Dur |
|-------|-----|---------------|-----------|------------|---------|
| Ground Startup | 341 | BLASTER_GROUND_STARTUP | 2,196 | 6 frames | 5.8 |
| Ground Loop | 342 | BLASTER_GROUND_LOOP | 2,133 | 10 frames | 15.2 |
| Ground End | 343 | BLASTER_GROUND_END | 2,266 | 24 frames | 21.6 |
| Air Startup | 344 | BLASTER_AIR_STARTUP | 173,303 | 4 frames | 3.9 |
| Air Loop | 345 | BLASTER_AIR_LOOP | 173,514 | 7 frames | 9.8 |
| Air End | 346 | BLASTER_AIR_END | 19,181 | 5 frames | 7.6 |

---

### State 341: BLASTER_GROUND_STARTUP

#### Description
The initial windup animation when using Blaster while grounded. Fox draws his blaster before firing.

#### Frame Data
| Metric | Value |
|--------|-------|
| First Loop Frame | 7 |
| Total Animation | ~6 frames median |
| Repeat Window | Frames 4-16 |

#### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| WAIT (14) | 1,125 | 51.2% |
| LANDING (42) | 672 | 30.6% |
| SQUAT (39) | 203 | 9.2% |
| OTTOTTO (245) | 37 | 1.7% |
| TURN (18) | 26 | 1.2% |

**Primary usage**: Standing laser from neutral (51.2%) or immediately after landing (30.6%).

#### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| BLASTER_GROUND_LOOP (342) | 2,069 | 96.2% |
| BLASTER_AIR_STARTUP (344) | 81 | 3.8% |

**Flow**: 96.2% continue to loop phase; 3.8% become airborne (platform drop, getting hit).

---

### State 342: BLASTER_GROUND_LOOP

#### Description
The repeating laser fire animation. Fox can hold B to continue firing multiple shots while grounded.

#### Frame Data
| Metric | Value |
|--------|-------|
| Shot Active | Frames 12-45 |
| Last Loop Frame | 16 |
| Median Duration | 10 frames |

#### Shot Count Distribution
| Shot Count | Sequences | % of Loops |
|------------|-----------|------------|
| 1 shot (1-10 frames) | 1,523 | 71.4% |
| 2 shots (11-20 frames) | 504 | 23.6% |
| 3 shots (21-30 frames) | 40 | 1.9% |
| 4-5 shots (31-50 frames) | 26 | 1.2% |
| 6+ shots (51+ frames) | 40 | 1.9% |

**Key insight**: 95% of ground laser sequences are 1-2 shots.

#### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| BLASTER_GROUND_END (343) | 2,022 | 96.8% |
| BLASTER_AIR_STARTUP (344) | 66 | 3.2% |

---

### State 343: BLASTER_GROUND_END

#### Description
The cooldown animation after releasing B. Fox holsters his blaster.

#### Frame Data
| Metric | Value |
|--------|-------|
| Total Animation | 40 frames |
| Median Duration | 24 frames |

#### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| BLASTER_AIR_STARTUP (344) | 1,791 | 97.5% |
| BLASTER_GROUND_STARTUP (341) | 46 | 2.5% |

**Note**: High transition to air startup suggests players often jump during/after ground laser.

---

### State 344: BLASTER_AIR_STARTUP

#### Description
The initial windup for aerial Blaster. This is the primary entry point for "Short Hop Laser" (SHL), Fox's signature approach/zoning tool.

#### Frame Data
| Metric | Value |
|--------|-------|
| First Loop Frame | 4 |
| Repeat Window | Frames 4-14 |
| Median Duration | 4 frames |

#### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| JUMP_F (25) | 109,477 | 63.0% |
| FALL (29) | 21,069 | 12.1% |
| JUMP_AERIAL_F (27) | 16,773 | 9.6% |
| JUMP_B (26) | 8,812 | 5.1% |
| KNEE_BEND (24) | 6,334 | 3.6% |
| PASS (244) | 6,070 | 3.5% |

**Key insight**: 68.1% from first jump (25, 26) - this is Short Hop Laser. Another 9.6% from double jump.

#### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| BLASTER_AIR_LOOP (345) | 172,589 | 99.99% |

**Flow**: Nearly 100% transition to loop phase.

---

### State 345: BLASTER_AIR_LOOP

#### Description
The repeating laser fire animation in the air. This is where the actual laser projectiles are generated.

#### Frame Data
| Metric | Value |
|--------|-------|
| Shot Active | Frames 10-43 |
| Last Loop Frame | 14 |
| Total Animation | 36 frames |
| Median Duration | 7 frames |

#### Shot Count Distribution (Air)
| Shot Count | Sequences | % of Loops |
|------------|-----------|------------|
| 1 shot (1-10 frames) | 128,034 | 73.8% |
| 2 shots (11-20 frames) | 36,939 | 21.3% |
| 3 shots (21-30 frames) | 6,283 | 3.6% |
| 4-5 shots (31-50 frames) | 2,098 | 1.2% |
| 6+ shots (51+ frames) | 158 | 0.1% |

**Key insight**: 95.1% of aerial laser sequences are 1-2 shots, optimized for short hop timing.

#### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| BLASTER_AIR_STARTUP (344) | 129,779 | 86.3% |
| BLASTER_AIR_END (346) | 19,123 | 12.7% |
| BLASTER_GROUND_STARTUP (341) | 1,522 | 1.0% |

**Flow**: 86.3% loop back for another shot; 12.7% end the sequence.

#### Invulnerability
- Air loop has 16.7% invulnerability rate (283,596 / 1,701,846 frames)
- This is during hitlag/hitstun carry-over, not inherent to the move

---

### State 346: BLASTER_AIR_END

#### Description
The cooldown animation for aerial Blaster. Fox holsters the blaster while airborne.

#### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 5 frames |
| Average Duration | 7.6 frames |

#### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| BLASTER_AIR_STARTUP (344) | 14,861 | 98.6% |
| BLASTER_GROUND_STARTUP (341) | 218 | 1.4% |

**Note**: 98.6% continue to another aerial laser sequence.

---

## Short Hop Laser (SHL) Analysis

SHL is Fox's signature neutral tool, executed by short hopping and pressing B during the jump.

### Execution Pattern
1. KNEE_BEND (24) - Jumpsquat (3 frames)
2. JUMP_F/B (25/26) - Rising jump
3. BLASTER_AIR_STARTUP (344) - Draw blaster
4. BLASTER_AIR_LOOP (345) - Fire laser(s)
5. LANDING (42) or continue aerial sequence

### Usage Statistics
- **63% of aerial lasers** come from forward first jump (state 25)
- **Average 1.3 shots per SHL sequence** (73.8% single shot)
- **96.4% of all lasers are aerial** - ground laser is rarely used

---

## Fox Illusion (Side-B) States

Fox Illusion is a high-speed horizontal dash primarily used for recovery. Fox becomes intangible during the dash but is vulnerable during startup and endlag.

### State Summary

| State | ID | Internal Name | Instances | Median Dur | Avg Dur |
|-------|-----|---------------|-----------|------------|---------|
| Ground Startup | 347 | ILLUSION_GROUND_STARTUP | 2,125 | 19 frames | 15.9 |
| Ground Main | 348 | ILLUSION_GROUND | 3,206 | 2 frames | 2.4 |
| Ground End | 349 | ILLUSION_GROUND_END | 2,781 | 40 frames | 37.4 |
| Air Startup | 350 | ILLUSION_STARTUP_AIR | 101,329 | 19 frames | 19.1 |
| Air Main | 351 | ILLUSION_AIR | 91,455 | 4 frames | 3.3 |
| Air End | 352 | ILLUSION_AIR_END | 81,486 | 15 frames | 16.5 |

---

### State 347: ILLUSION_GROUND_STARTUP

#### Description
The charging animation before ground Illusion. Fox crouches and prepares to dash.

#### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 19 frames |
| Hitbox Active | Frames 22-25 |

#### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| SQUAT (39) | 683 | 32.1% |
| DASH (20) | 462 | 21.7% |
| ILLUSION_STARTUP_AIR (350) | 201 | 9.4% |
| TURN (18) | 177 | 8.3% |
| WALK_SLOW (15) | 149 | 7.0% |

**Key insight**: Ground Illusion is often used from crouch (32.1%) or dash (21.7%).

#### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| ILLUSION_GROUND (348) | 1,601 | 80.3% |
| ILLUSION_STARTUP_AIR (350) | 359 | 18.0% |

---

### State 348: ILLUSION_GROUND

#### Description
The active dash phase of ground Illusion. Fox moves horizontally at high speed with a hitbox.

#### Frame Data
| Metric | Value |
|--------|-------|
| Active Hitbox | Frames 22-25 (4 frames) |
| Damage | 7% |
| Knockback Angle | 68 degrees |
| Median Duration | 2 frames |

#### Properties
- **On Ground**: 100%
- **Very brief**: Only 2 frames median, suggesting players rarely stay grounded during the active phase

#### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| ILLUSION_GROUND_END (349) | 2,747 | 86.4% |
| ILLUSION_AIR (351) | 363 | 11.4% |

---

### State 349: ILLUSION_GROUND_END

#### Description
The landing/recovery animation after ground Illusion completes. Significant endlag.

#### Frame Data
| Metric | Value |
|--------|-------|
| Total Animation | 63 frames
| Median Duration | 40 frames |
| Landing Lag | 23 frames total |

#### Properties
- **On Ground**: 100%
- **Invulnerability**: 0.6% (mostly none)
- **Highly punishable** due to long endlag

#### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| ILLUSION_STARTUP_AIR (350) | 1,988 | 96.9% |

---

### State 350: ILLUSION_STARTUP_AIR

#### Description
The charging animation for aerial Illusion. This is the primary recovery option for Fox, used to cover horizontal distance when offstage.

#### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 19 frames |
| Hitbox Active | Frames 22-25 |
| Total Animation | 63 frames |

#### Startup Duration Distribution
| Duration | Frequency | % | Notes |
|----------|-----------|---|-------|
| 19 frames | 4,382 | 15.0% | Full startup |
| 38 frames | 4,824 | 16.5% | ~2 startups |
| 57 frames | 4,384 | 15.0% | ~3 startups |
| 76 frames | 3,382 | 11.6% | ~4 startups |
| 95 frames | 2,215 | 7.6% | ~5 startups |

**Key insight**: Duration clusters at multiples of 19 frames, indicating repeated Illusion attempts.

#### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| JUMP_AERIAL_F (27) | 59,024 | 56.6% |
| DAMAGE_FALL (38) | 15,693 | 15.0% |
| FALL (29) | 11,629 | 11.1% |
| JUMP_AERIAL_B (28) | 8,818 | 8.5% |
| FALL_AERIAL (32) | 2,191 | 2.1% |
| JUMP_F (25) | 1,700 | 1.6% |
| PASSIVE_WALL_JUMP (203) | 1,269 | 1.2% |

**Primary usage**:
- 65.1% from double jump (27, 28) - standard recovery sequence
- 15.0% from tumble (38) - emergency recovery after being hit

#### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| ILLUSION_AIR (351) | 91,053 | 93.3% |
| ILLUSION_AIR_END (352) | 6,265 | 6.4% |

**Flow**: 93.3% complete startup and enter active phase.

#### Invulnerability
- 3.8% invuln rate (during carry-over from previous states)
- No inherent invulnerability during startup

---

### State 351: ILLUSION_AIR

#### Description
The active dash phase of aerial Illusion. Fox moves horizontally at high speed through the air.

#### Frame Data
| Metric | Value |
|--------|-------|
| Active Hitbox | Frames 22-25 (4 frames) |
| Damage | 7% |
| Knockback Scaling | 60 (aerial) vs 40 (grounded) |
| Median Duration | 4 frames |
| Hitbox Radius | 4.16 units |

#### Properties
- **On Ground**: 0% (always airborne)
- **Invulnerability**: 0.9% (minimal)
- **Cannot clang, unreflectable, unabsorbable**

#### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| ILLUSION_AIR_END (352) | 75,155 | 86.3% |
| ILLUSION_STARTUP_AIR (350) | 10,172 | 11.7% |
| ILLUSION_GROUND (348) | 1,605 | 1.8% |

---

### State 352: ILLUSION_AIR_END

#### Description
The endlag after aerial Illusion. Fox enters a brief freefall-like state before regaining control or landing.

#### Frame Data
| Metric | Value |
|--------|-------|
| Pre-Freefall | 20 frames |
| Post-Freefall | 3 frames |
| Total Lag | 23 frames |
| Ledge Grab Cancel | Frame 29+ |
| Median Duration | 15 frames |

#### Position Analysis
| Metric | Value |
|--------|-------|
| Min Y Position | -145.6 |
| Max Y Position | 217.0 |
| Avg Y Position | -0.95 |

**Note**: Wide Y range indicates use both high above stage and deep offstage.

#### Exit Conditions (Primary Outcomes)
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| LANDING_FALL_SPECIAL (43) | 44,506 | 54.7% |
| CLIFF_CATCH (252) | 22,853 | 28.1% |
| FALL_SPECIAL (35) | 6,576 | 8.1% |
| DAMAGE_FLY_N (88) | 2,311 | 2.8% |
| DAMAGE_FLY_HI (87) | 1,503 | 1.9% |
| DAMAGE_FLY_TOP (90) | 974 | 1.2% |
| DAMAGE_FLY_ROLL (91) | 842 | 1.0% |
| DEAD_DOWN (0) | 191 | 0.2% |
| DEAD_RIGHT (2) | 158 | 0.2% |
| DEAD_LEFT (1) | 143 | 0.2% |

#### Outcome Summary
| Outcome | Count | % |
|---------|-------|---|
| Land on Stage (43) | 44,506 | 54.7% |
| Grab Ledge (252) | 22,853 | 28.1% |
| Continue Falling (35) | 6,576 | 8.1% |
| Got Hit | ~6,900 | 8.5% |
| Death | ~492 | 0.6% |

**Key insight**: 82.8% success rate (land + ledge grab). Only 0.6% result in death directly from Illusion.

---

## Illusion Shortening

Fox Illusion can be shortened by pressing B again during frames 1-5 of forward travel, creating five possible distances useful for edge recovery.

### Shorten Distances
The move has 5 possible shortened lengths, allowing precise horizontal control for recovery. This is critical for:
- Avoiding gimp setups
- Sweetspotting the ledge
- Mixing up recovery timing

### Recovery Statistics
From Air Illusion End (352):
- **28.1% grab ledge** - successful edge recovery
- **54.7% land on stage** - direct stage return
- **8.1% continue falling** - requires additional recovery
- **0.6% die** - failed recovery (SD or punished)

---

## State Flow Diagrams

### Blaster Flow
```
Ground: 341 (Startup) → 342 (Loop) ⟷ 343 (End)
                ↘ 344 ↗
Air:    344 (Startup) → 345 (Loop) ⟷ 346 (End)
         ↑_______________↙   ↑____________↙
```

### Illusion Flow
```
Ground: 347 (Startup) → 348 (Main) → 349 (End)
            ↘ 350 ↗       ↘ 351 ↗      ↘ 350 ↗
Air:    350 (Startup) → 351 (Main) → 352 (End)
         ↑_______________↙   ↑____________↙
```

---

## Competitive Applications

### Blaster
- **Short Hop Laser (SHL)**: Primary neutral tool for Fox
  - 63% of all lasers from forward first jump
  - Used to rack damage safely and force approaches
  - Can be wavelanded for faster follow-up
- **Double Laser**: Pressing B twice during short hop for 6% chip damage
- **Platform Laser**: Lasering from platforms for angle coverage

### Illusion
- **Horizontal Recovery**: Primary horizontal recovery tool
  - Used after double jump for maximum distance
  - Shortening allows precise ledge sweetspots
- **Ledge Cancel**: Landing on a platform during Illusion cancels endlag
- **Combo Tool**: Rarely used in neutral due to high endlag risk
  - Ground Illusion has 40-frame endlag (highly punishable)

---

## Invulnerability Summary

| State | Invuln % | Notes |
|-------|----------|-------|
| BLASTER_GROUND_STARTUP (341) | 8.5% | Hitlag carry-over |
| BLASTER_GROUND_LOOP (342) | 7.4% | Hitlag carry-over |
| BLASTER_GROUND_END (343) | 7.4% | Hitlag carry-over |
| BLASTER_AIR_STARTUP (344) | 19.6% | Higher due to aerial entry |
| BLASTER_AIR_LOOP (345) | 16.7% | Aerial carry-over |
| BLASTER_AIR_END (346) | 7.1% | Minimal |
| ILLUSION_GROUND_STARTUP (347) | 5.0% | Minimal |
| ILLUSION_GROUND (348) | 2.4% | Active phase |
| ILLUSION_GROUND_END (349) | 0.6% | Endlag, very vulnerable |
| ILLUSION_STARTUP_AIR (350) | 3.8% | Startup, vulnerable |
| ILLUSION_AIR (351) | 0.9% | Active phase |
| ILLUSION_AIR_END (352) | 0.2% | Endlag, very vulnerable |

**Note**: Neither move has inherent invulnerability. Percentages reflect carry-over from previous invulnerable states.

---

## Related States

| Relationship | States |
|--------------|--------|
| Pairs with (recovery) | JUMP_AERIAL (27, 28), FALL_SPECIAL (35), CLIFF_CATCH (252) |
| Gets punished by | DAMAGE_FLY (87-91), DEATH (0, 1, 2) |
| Neutral tools | WAIT (14), DASH (20), JUMP (25, 26) |

---

## Data Sources

- Parquet replay data: 288,753,124 total frames (Fox dittos)
- SmashWiki: Fox Blaster and Fox Illusion frame data
- action_state.json: State ID to name mapping
