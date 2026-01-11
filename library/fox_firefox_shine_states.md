# Fox: Fire Fox & Shine States Documentation

Action states 353-369 covering Fox's Up-B (Fire Fox) and Down-B (Reflector/Shine) special moves, both grounded and aerial versions.

## Overview

These 17 states represent Fox's two remaining special moves, including his signature combo tool (Shine) and his vertical recovery option (Fire Fox).

| Move | Version | States | Total Frames | % of Gameplay |
|------|---------|--------|--------------|---------------|
| Fire Fox | Ground | 353, 355, 357 | 161,628 | 0.06% |
| Fire Fox | Air | 354, 356, 358, 359 | 9,347,811 | 3.24% |
| Shine | Ground | 360-364 | 3,662,221 | 1.27% |
| Shine | Air | 365-369 | 2,491,665 | 0.86% |

**Total special move frames**: 15,663,325 (5.42% of gameplay)

### Ground vs Air Usage

| Move | Ground % | Air % | Notes |
|------|----------|-------|-------|
| Fire Fox | 1.7% | **98.3%** | Almost exclusively recovery |
| Shine | **59.5%** | 40.5% | Ground combos + aerial extensions |

---

## Fire Fox (Up-B) States

Fire Fox is Fox's vertical recovery move. It consists of a charging phase where Fox gathers energy, followed by a high-speed launch in a controllable direction.

### State Summary

| State | ID | Internal Name | Instances | Median Dur | Avg Dur |
|-------|-----|---------------|-----------|------------|---------|
| Ground Startup | 353 | FIRE_FOX_GROUND_STARTUP | 687 | 24 frames | 25.2 |
| Air Startup | 354 | FIRE_FOX_AIR_STARTUP | 146,522 | 42 frames | 39.0 |
| Ground Travel | 355 | FIRE_FOX_GROUND | 179 | 30 frames | 30.4 |
| Air Travel | 356 | FIRE_FOX_AIR | 111,332 | 30 frames | 24.3 |
| Ground End | 357 | FIRE_FOX_GROUND_END | 24,371 | 6 frames | 5.7 |
| Air End | 358 | FIRE_FOX_AIR_END | 69,174 | 14 frames | 13.2 |
| Bounce End | 359 | FIRE_FOX_BOUNCE_END | 1,072 | 14 frames | 13.4 |

---

### State 353: FIRE_FOX_GROUND_STARTUP

#### Description
The charging animation for Fire Fox while grounded. Fox crouches and gathers fiery energy around himself. Rarely used since Fire Fox is primarily a recovery tool.

#### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 24 frames |
| Looping Hits | Frames 20, 22, 24, 26, 28, 30, 32 |
| Charge Time | Variable (can release early or hold) |

#### Properties
- **On Ground**: 100%
- **Invulnerability**: 3.2% (minimal)

#### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| FIRE_FOX_AIR_STARTUP (354) | 278 | 40.5% |
| KNEE_BEND (24) | 250 | 36.4% |
| LANDING (42) | 40 | 5.8% |
| WAIT (14) | 32 | 4.7% |

#### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| FIRE_FOX_AIR (356) | 342 | 49.8% |
| FIRE_FOX_GROUND (355) | 179 | 26.1% |
| Got Hit (damage states) | 120 | 17.5% |

---

### State 354: FIRE_FOX_AIR_STARTUP

#### Description
The charging animation for aerial Fire Fox. This is the primary Fire Fox state, used when recovering from offstage. Fox can angle the direction during this phase.

#### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 42 frames |
| Average Duration | 39.0 frames |
| Max Duration | 1,061 frames (held/interrupted) |
| Full Charge | ~42 frames |

#### Properties
- **On Ground**: 0% (always airborne)
- **Invulnerability**: 7.0% (from previous states)

#### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| JUMP_AERIAL_F (27) | 87,877 | 55.8% |
| DAMAGE_FALL (38) | 25,918 | 16.5% |
| FALL (29) | 14,552 | 9.2% |
| JUMP_AERIAL_B (28) | 14,472 | 9.2% |
| FALL_AERIAL (32) | 3,867 | 2.5% |
| PASSIVE_WALL_JUMP (203) | 1,668 | 1.1% |

**Key insight**: 65% from double jump (27, 28) - standard recovery pattern. 16.5% from tumble (38) - emergency recovery.

#### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| FIRE_FOX_AIR (356) | 110,984 | 75.8% |
| CLIFF_CATCH (252) | 36,190 | 24.7% |

**Notable**: 24.7% grab ledge during startup (early ledge sweetspot during charge).

---

### State 355: FIRE_FOX_GROUND

#### Description
The traveling phase of grounded Fire Fox. Fox launches in the aimed direction while remaining on the ground. Very rare.

#### Frame Data
| Metric | Value |
|--------|-------|
| Hitbox Active | Frames 43-72 |
| Total Animation | ~30 frames median |
| Damage | 14% (NTSC) / 12% (PAL) |

#### Properties
- **On Ground**: 100%
- **Invulnerability**: 1.2% (minimal)
- **Instances**: Only 179 in dataset

---

### State 356: FIRE_FOX_AIR

#### Description
The traveling phase of aerial Fire Fox. Fox launches through the air covered in flames with an active hitbox.

#### Frame Data
| Metric | Value |
|--------|-------|
| Hitbox Active | Frames 43-72 (30 frames) |
| Median Duration | 30 frames |
| Damage | 14% (NTSC) / 12% (PAL) |
| Effect | Flame |

#### Properties
- **On Ground**: 0%
- **Invulnerability**: 0.0% (vulnerable during travel)

---

### State 357: FIRE_FOX_GROUND_END

#### Description
The landing/ending animation when Fire Fox ends on the ground.

#### Frame Data
| Metric | Value |
|--------|-------|
| Pre-Freefall Lag | 6 frames |
| Post-Freefall Lag | 3 frames |
| Median Duration | 6 frames |

#### Properties
- **On Ground**: 100%
- **Highly punishable** on whiff

---

### State 358: FIRE_FOX_AIR_END

#### Description
The ending animation when Fire Fox ends in the air. Fox enters a freefall state after this.

#### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 14 frames |
| Average Duration | 13.2 frames |

#### Properties
- **On Ground**: 0%
- **Leads to helpless/freefall**

---

### State 359: FIRE_FOX_BOUNCE_END

#### Description
Special ending animation when Fox bounces off a surface (wall/ceiling) during Fire Fox. The bounce can save Fox from SDs in some situations.

#### Frame Data
| Metric | Value |
|--------|-------|
| Instances | 1,072 |
| Median Duration | 14 frames |

#### Properties
- **On Ground**: 0% (airborne bounce)
- **Rare state** - requires hitting a surface during travel

---

## Reflector/Shine (Down-B) States

Fox's Reflector (commonly called "Shine") is his signature move and one of the most versatile tools in Melee. It's frame 1 intangible, deals set knockback at a semi-spike angle, and is jump-cancellable from frame 4.

### State Summary

| State | ID | Internal Name | Instances | Median Dur | Avg Dur |
|-------|-----|---------------|-----------|------------|---------|
| Ground Startup | 360 | REFLECTOR_GROUND_STARTUP | 395,794 | 3 frames | 4.8 |
| Ground Loop | 361 | REFLECTOR_GROUND_LOOP | 386,848 | 3 frames | 3.5 |
| Ground Reflect | 362 | REFLECTOR_GROUND_REFLECT | 393 | 20 frames | 18.1 |
| Ground End | 363 | REFLECTOR_GROUND_END | 18,752 | 18 frames | 16.1 |
| Ground Turnaround | 364 | REFLECTOR_GROUND_CHANGE_DIRECTION | 30,272 | 3 frames | 2.9 |
| Air Startup | 365 | REFLECTOR_AIR_STARTUP | 232,991 | 3 frames | 3.9 |
| Air Loop | 366 | REFLECTOR_AIR_LOOP | 205,910 | 3 frames | 5.5 |
| Air Reflect | 367 | REFLECTOR_AIR_REFLECT | 170 | 20 frames | 19.1 |
| Air End | 368 | REFLECTOR_AIR_END | 14,791 | 18 frames | 15.8 |
| Air Turnaround | 369 | REFLECTOR_AIR_CHANGE_DIRECTION | 75,296 | 3 frames | 2.9 |

---

### State 360: REFLECTOR_GROUND_STARTUP

#### Description
The initial activation of grounded Shine. The hitbox is active on frame 1 with intangibility. This is Fox's fastest move and the cornerstone of his combo game.

#### Frame Data
| Metric | Value |
|--------|-------|
| Hitbox Active | Frame 1 |
| Intangibility | Frame 1 |
| Jump Cancelable | Frames 4-21 |
| Turnaround Window | Frames 4-21 |
| Median Duration | 3 frames |

#### Hitbox Properties
| Property | Value |
|----------|-------|
| Damage | 5% |
| Angle | 0° (semi-spike) |
| Base Knockback | 100 (set) |
| Knockback Scaling | 80 |
| Effect | Electric |

#### Invulnerability Analysis
| Frame | Invuln % | Notes |
|-------|----------|-------|
| 1 | 97.0% | Active hitbox + intangible |
| 2 | 50.9% | Transitioning |
| 3 | 52.5% | Transitioning |
| 4 | 95.2% | Jump cancel window begins |
| 5-6 | ~25% | Vulnerable |
| 7 | 95.2% | Loop refresh |

**Key insight**: Frame 1 is nearly always intangible (97%). The periodic 95% invuln on frames 4, 7, 10 suggests hitbox refresh timing.

#### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| SQUAT (39) | 328,499 | 75.5% |
| SQUAT_WAIT (40) | 49,248 | 11.3% |
| LANDING_AIR_LW (74) | 10,997 | 2.5% |
| LANDING (42) | 8,959 | 2.1% |
| LANDING_AIR_N (70) | 7,569 | 1.7% |
| LANDING_AIR_B (72) | 6,291 | 1.4% |
| PASSIVE (199) | 3,327 | 0.8% |

**Primary usage patterns**:
- **75.5% from crouch** (39) - crouch cancel into shine, or standing shine buffered from crouch
- **11.3% from crouch wait** (40) - held crouch into shine
- **7.7% from aerial landing lag** (70, 72, 74) - drill-shine, nair-shine, etc.
- **0.8% from tech** (199) - tech-in-place shine

#### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| REFLECTOR_GROUND_LOOP (361) | 352,987 | 91.7% |
| REFLECTOR_AIR_STARTUP (365) | 31,569 | 8.2% |

**Flow**: 91.7% continue to loop (holding shine), 8.2% immediately jump cancel.

---

### State 361: REFLECTOR_GROUND_LOOP

#### Description
The held state when Fox keeps the Reflector active on the ground. Can be jump-cancelled or released.

#### Frame Data
| Metric | Value |
|--------|-------|
| Reflection Window | Frames 4-21+ |
| Max Duration | 131 frames (held) |
| Reflection Radius | 8.5 units |
| Projectile Damage Multiplier | 1.5x |

#### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| REFLECTOR_GROUND_STARTUP (360) | 231,685 | 63.9% |
| REFLECTOR_AIR_STARTUP (365) | 114,071 | 31.5% |
| REFLECTOR_GROUND_END (363) | 15,800 | 4.4% |

**Key insight**:
- **63.9% → 360** - Waveshine pattern! (shine → jump → wavedash → shine)
- **31.5% → 365** - Jump cancel into aerial shine
- **4.4% → 363** - Release shine (endlag)

---

### State 362: REFLECTOR_GROUND_REFLECT

#### Description
Special state when the Reflector successfully deflects a projectile while grounded.

#### Frame Data
| Metric | Value |
|--------|-------|
| Instances | 393 |
| Median Duration | 20 frames |
| Reflection Lag | 19 frames + ending animation |

**Very rare** - only 393 instances in dataset. Fox dittos don't have many projectiles to reflect.

---

### State 363: REFLECTOR_GROUND_END

#### Description
The ending animation when releasing Shine without jump-cancelling. Has significant endlag.

#### Frame Data
| Metric | Value |
|--------|-------|
| End Lag | 19 frames |
| Median Duration | 18 frames |
| Total Animation | 39 frames |

#### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| REFLECTOR_GROUND_STARTUP (360) | 12,457 | 69.6% |
| REFLECTOR_AIR_STARTUP (365) | 5,437 | 30.4% |

---

### State 364: REFLECTOR_GROUND_CHANGE_DIRECTION

#### Description
Turnaround Shine on the ground. Fox reverses direction while in Shine. Disables jump-cancel for 3 additional frames.

#### Frame Data
| Metric | Value |
|--------|-------|
| Instances | 30,272 |
| Median Duration | 3 frames |
| Jump Cancel Delay | +3 frames |

#### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| REFLECTOR_GROUND_LOOP (361) | 14,473 | 47.8% |
| REFLECTOR_GROUND_STARTUP (360) | 14,435 | 47.7% |
| REFLECTOR_AIR_CHANGE_DIRECTION (369) | 1,180 | 3.9% |

---

### State 365: REFLECTOR_AIR_STARTUP

#### Description
The initial activation of aerial Shine. Same frame 1 properties as grounded Shine.

#### Frame Data
| Metric | Value |
|--------|-------|
| Hitbox Active | Frame 1 |
| Intangibility | Frame 1 |
| Median Duration | 3 frames |

#### Invulnerability Analysis
| Frame | Invuln % | Notes |
|-------|----------|-------|
| 1 | 94.5% | Active hitbox + intangible |
| 2 | 37.3% | Transitioning |
| 3 | 39.3% | Transitioning |
| 4 | 94.9% | Refresh |
| 7 | 95.0% | Refresh |
| 10 | 95.3% | Refresh |

#### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| JUMP_F (25) | 85,167 | 35.8% |
| FALL (29) | 82,882 | 34.8% |
| KNEE_BEND (24) | 26,980 | 11.3% |
| DAMAGE_FALL (38) | 15,675 | 6.6% |
| JUMP_B (26) | 5,784 | 2.4% |
| JUMP_AERIAL_F (27) | 5,688 | 2.4% |
| PASS (244) | 4,826 | 2.0% |
| DAMAGE_FLY_TOP (90) | 4,546 | 1.9% |

**Primary usage patterns**:
- **35.8% from first jump** (25) - short hop shine
- **34.8% from fall** (29) - falling shine, combo extension
- **11.3% from jumpsquat** (24) - buffered aerial shine
- **6.6% from tumble** (38) - defensive shine after being hit

#### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| REFLECTOR_AIR_LOOP (366) | 193,583 | 88.6% |
| REFLECTOR_GROUND_STARTUP (360) | 23,986 | 11.0% |

---

### State 366: REFLECTOR_AIR_LOOP

#### Description
The held state for aerial Shine. Can be held, released, or turnaround.

#### Frame Data
| Metric | Value |
|--------|-------|
| Max Duration | 118 frames |
| Median Duration | 3 frames |

#### Properties
- **Invulnerability**: 11.8% (carry-over and periodic refresh)
- **On Ground**: 0%

#### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| REFLECTOR_GROUND_STARTUP (360) | 90,071 | 48.6% |
| REFLECTOR_AIR_STARTUP (365) | 58,442 | 31.5% |
| REFLECTOR_GROUND_LOOP (361) | 22,042 | 11.9% |
| REFLECTOR_AIR_END (368) | 14,699 | 7.9% |

**Key insight**: 48.6% land into ground shine - continuous shine pressure after landing.

---

### State 367: REFLECTOR_AIR_REFLECT

#### Description
Special state when aerial Shine successfully deflects a projectile.

#### Frame Data
| Metric | Value |
|--------|-------|
| Instances | 170 |
| Median Duration | 20 frames |

**Extremely rare** - only 170 instances.

---

### State 368: REFLECTOR_AIR_END

#### Description
The ending animation when releasing aerial Shine. Significant aerial lag.

#### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 18 frames |
| Instances | 14,791 |

#### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| REFLECTOR_GROUND_STARTUP (360) | 6,040 | 44.3% |
| REFLECTOR_AIR_STARTUP (365) | 4,852 | 35.6% |
| REFLECTOR_GROUND_END (363) | 2,739 | 20.1% |

---

### State 369: REFLECTOR_AIR_CHANGE_DIRECTION

#### Description
Turnaround aerial Shine. Fox reverses facing direction while shining in the air.

#### Frame Data
| Metric | Value |
|--------|-------|
| Instances | 75,296 |
| Median Duration | 3 frames |
| Jump Cancel Delay | +3 frames |

#### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| REFLECTOR_AIR_STARTUP (365) | 55,624 | 73.9% |
| REFLECTOR_AIR_LOOP (366) | 19,198 | 25.5% |

#### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| REFLECTOR_AIR_LOOP (366) | 39,916 | 53.0% |
| JUMP_AERIAL_F (27) | 33,393 | 44.4% |

**Key insight**: 44.4% → double jump (27) - turnaround shine often used for repositioning/recovery.

---

## Waveshine Analysis

Waveshine is Fox's signature tech skill: shine → jump cancel → wavedash → shine repeat.

### Detection Pattern
From the shine loop (361) exit data:
- **63.9% → Ground Startup (360)** - This represents the waveshine pattern
- **31.5% → Air Startup (365)** - Jump cancel into aerial shine (for combos)
- **4.4% → Ground End (363)** - Release without cancel

### Drill-Shine Pattern
From ground shine startup (360) entries:
- **2.5% from LANDING_AIR_LW (74)** - Drill (dair) landing into shine
- **1.7% from LANDING_AIR_N (70)** - Nair landing into shine
- **1.4% from LANDING_AIR_B (72)** - Bair landing into shine

---

## Shine Properties Summary

| Property | Value |
|----------|-------|
| Hitbox Active | Frame 1 |
| Intangibility | Frame 1 |
| Jump Cancelable | Frames 4-21 |
| Turnaround Window | Frames 4-21 |
| Damage | 5% |
| Angle | 0° (semi-spike) |
| Knockback | Set (100 base, 80 scaling) |
| Effect | Electric |
| Reflection Multiplier | 1.5x damage, 1.0x speed |
| Max Reflectable Damage | 50% |
| End Lag (if not canceled) | 19 frames |
| Total Animation | 39 frames |

---

## Fire Fox Recovery Statistics

### Entry Context
| Source | % | Notes |
|--------|---|-------|
| Double Jump (27, 28) | 65.0% | Standard recovery |
| Tumble (38) | 16.5% | Emergency recovery |
| Fall (29) | 9.2% | Falling recovery |
| Wall Tech Jump (203) | 1.1% | After wall tech |

### Recovery Success
From Fire Fox Air Startup (354):
- **75.8% → Travel Phase** - Full Fire Fox execution
- **24.7% → Ledge Grab (252)** - Early sweetspot during charge

---

## Invulnerability Summary

| State | % Invuln | Notes |
|-------|----------|-------|
| FIRE_FOX_GROUND_STARTUP (353) | 3.2% | Minimal |
| FIRE_FOX_AIR_STARTUP (354) | 7.0% | Carry-over |
| FIRE_FOX_GROUND (355) | 1.2% | Vulnerable travel |
| FIRE_FOX_AIR (356) | 0.0% | Vulnerable travel |
| FIRE_FOX_GROUND_END (357) | 0.0% | Vulnerable endlag |
| FIRE_FOX_AIR_END (358) | 0.0% | Vulnerable endlag |
| FIRE_FOX_BOUNCE_END (359) | 0.4% | Minimal |
| **REFLECTOR_GROUND_STARTUP (360)** | **56.0%** | Frame 1 intangible |
| REFLECTOR_GROUND_LOOP (361) | 2.3% | Holding |
| REFLECTOR_GROUND_REFLECT (362) | 1.2% | Reflecting |
| REFLECTOR_GROUND_END (363) | 1.3% | Endlag |
| REFLECTOR_GROUND_TURNAROUND (364) | 2.4% | Turnaround |
| **REFLECTOR_AIR_STARTUP (365)** | **54.9%** | Frame 1 intangible |
| REFLECTOR_AIR_LOOP (366) | 11.8% | Holding (aerial) |
| REFLECTOR_AIR_REFLECT (367) | 1.2% | Reflecting |
| REFLECTOR_AIR_END (368) | 1.3% | Endlag |
| REFLECTOR_AIR_TURNAROUND (369) | 2.9% | Turnaround |

**Key finding**: Shine startup states (360, 365) show 55-56% invulnerability because frame 1 is intangible, and the median duration is only 3 frames.

---

## Competitive Applications

### Shine
- **Combo Starter**: Crouch-shine (CC shine) catches opponents' attacks
- **Combo Extender**: Waveshine across stage, drill-shine
- **Edgeguard**: Shine spike sends opponent at 0° angle
- **Defense**: Frame 1 intangible beats many options
- **Pressure**: Jump-cancelled shine is safe on shield

### Fire Fox
- **Vertical Recovery**: Primary recovery tool with directional control
- **Ledge Sweetspot**: 24.7% grab ledge during charge phase
- **Last Resort**: After double jump and Illusion exhausted
- **Kill Move**: 14% damage with flame effect (rarely used offensively)

---

## State Flow Diagrams

### Fire Fox Flow
```
Ground: 353 (Startup) → 355 (Travel) → 357 (End)
                  ↘ 356 ↗
Air:    354 (Startup) → 356 (Travel) → 358 (End)
              ↓              ↓           ↓
            [252]         [359]      [Freefall]
         (Ledge Grab)   (Bounce)
```

### Shine Flow
```
Ground: 360 (Startup) → 361 (Loop) ⟷ 363 (End)
            ↕              ↓ ↑         ↓
           364 ←──────────→ ↓         365
        (Turnaround)        ↓          ↓
                            ↓          ↓
Air:    365 (Startup) → 366 (Loop) ⟷ 368 (End)
            ↕              ↓
           369 ←──────────→
        (Turnaround)

Key Flows:
• 360→361→360 = Waveshine
• 360→361→365 = Jump Cancel
• 361→363 = Release (unsafe)
```

---

## Related States

| Relationship | States |
|--------------|--------|
| Recovery sequence | JUMP_AERIAL (27, 28), ILLUSION (350-352), FIRE_FOX (354-358), CLIFF_CATCH (252) |
| Combo starters | SQUAT (39), LANDING_AIR (70, 72, 74) |
| Shine followups | JUMP (25, 27), KNEE_BEND (24), DASH (20) |

---

## Data Sources

- Parquet replay data: 288,753,124 total frames (Fox dittos)
- SmashWiki: Fire Fox and Reflector frame data
- Smashboards: Shine frame data from SDM's documentation
- action_state.json: State ID to name mapping
