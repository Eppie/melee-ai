# Defensive Options States Documentation

Action states 233-238 covering rolls, spotdodge, airdodge, and attack rebound/clank.

## Overview

Defensive options in Melee provide invulnerability frames to escape pressure and avoid attacks. These are core defensive mechanics that define Melee's shield game and movement options. Airdodge is particularly important as it enables wavedashing, one of Melee's most fundamental advanced techniques.

| State | ID | Internal Name | Instances | % of Gameplay | Median Duration |
|-------|-----|---------------|-----------|---------------|-----------------|
| Roll Forward | 233 | ESCAPE_F | 50,763 | 0.58% | 31 frames |
| Roll Backward | 234 | ESCAPE_B | 38,410 | 0.43% | 31 frames |
| Spotdodge | 235 | ESCAPE | 62,746 | 0.55% | 22 frames |
| Airdodge | 236 | ESCAPE_AIR | 131,838 | 0.79% | 7 frames |
| Rebound Stop | 237 | REBOUND_STOP | 6,472 | 0.01% | 6 frames |
| Rebound | 238 | REBOUND | 6,447 | 0.02% | 11 frames |

**Total defensive option frames**: 6,874,131 (2.38% of gameplay)

---

## Invulnerability Summary

| State | Total Frames | Invuln Frames | Invuln % | Vulnerable Endlag |
|-------|--------------|---------------|----------|-------------------|
| Roll Forward (233) | 31 | 4-19 (16 frames) | 52.6% | 12 frames |
| Roll Backward (234) | 31 | 4-19 (16 frames) | 52.5% | 12 frames |
| Spotdodge (235) | 22 | 2-15 (14 frames) | 66.1% | 7 frames |
| Airdodge (236) | 49 (full) | 4-29 (26 frames) | 58.8% | 20 frames (or 10 landing) |
| Rebound Stop (237) | 6 | ~1 frame | 1.6% | 5 frames |
| Rebound (238) | 11 | ~0 frames | 0.1% | 11 frames |

---

## State 233: ESCAPE_F (Roll Forward)

### Description
Roll forward moves the character in the direction they're facing while providing invulnerability. It's performed by pressing L/R + forward on the control stick while shielding. Rolls are a key escape option from shield pressure but are highly punishable on read.

### Frame Data (Fox)
| Metric | Value |
|--------|-------|
| Total Duration | 31 frames |
| Invulnerability | Frames 4-19 (16 frames) |
| Vulnerable Startup | Frames 1-3 (3 frames) |
| Vulnerable Endlag | Frames 20-31 (12 frames) |
| On Ground | 100% |

### Invulnerability by Frame
| Frame Range | Invuln % | Notes |
|-------------|----------|-------|
| 1-3 | 3-6% | Startup (vulnerable) |
| 4-19 | 100% | Full invincibility |
| 20-31 | 0.4% | Endlag (vulnerable) |

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| GUARD_ON (178) | 31,710 | 54.5% |
| GUARD (179) | 11,959 | 20.6% |
| GUARD_SET_OFF (182) | 7,806 | 13.4% |
| GUARD_REFLECT (181) | 3,762 | 6.5% |
| DASH (20) | 2,801 | 4.8% |

**Key patterns**:
- **Shield states (178, 179, 181, 182)**: 95.0% - Nearly all rolls come from shield
- **Dash (20)**: 4.8% - Dash → shield → roll

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| WAIT (14) | 24,075 | 41.4% |
| GUARD_ON (178) | 6,149 | 10.6% |
| SQUAT (39) | 5,908 | 10.2% |
| TURN (18) | 4,923 | 8.5% |
| DASH (20) | 4,332 | 7.4% |
| WALK_SLOW (15) | 2,013 | 3.5% |
| DAMAGE_FLY_N (88) | 1,954 | 3.4% |
| CAPTURE_WAIT (226) | 1,506 | 2.6% |
| REFLECTOR_GROUND_STARTUP (360) | 924 | 1.6% |

**Key patterns**:
- **Neutral return (14, 15, 18, 20)**: 60.8% - Roll completes safely
- **Defensive followup (178, 39)**: 20.8% - Shield or crouch after roll
- **Punished (88, 87, 90, 226)**: 10.3% - Hit or grabbed during endlag
- **Shine (360)**: 1.6% - Immediate shine after roll

---

## State 234: ESCAPE_B (Roll Backward)

### Description
Roll backward moves the character away from the direction they're facing. Same frame data as forward roll but moves in the opposite direction. Often used to create space from pressure.

### Frame Data (Fox)
| Metric | Value |
|--------|-------|
| Total Duration | 31 frames |
| Invulnerability | Frames 4-19 (16 frames) |
| Vulnerable Startup | Frames 1-3 (3 frames) |
| Vulnerable Endlag | Frames 20-31 (12 frames) |
| On Ground | 100% |

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| GUARD_ON (178) | 22,311 | 52.8% |
| GUARD (179) | 9,545 | 22.6% |
| GUARD_SET_OFF (182) | 6,990 | 16.5% |
| GUARD_REFLECT (181) | 3,382 | 8.0% |

**Note**: Unlike roll forward, roll backward is rarely entered from dash - it's almost exclusively from shield (99.9%).

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| WAIT (14) | 20,519 | 48.4% |
| GUARD_ON (178) | 5,091 | 12.0% |
| DASH (20) | 3,228 | 7.6% |
| SQUAT (39) | 3,019 | 7.1% |
| TURN (18) | 2,485 | 5.9% |
| WALK_SLOW (15) | 1,391 | 3.3% |
| DAMAGE_FLY_N (88) | 1,108 | 2.6% |
| CAPTURE_WAIT (226) | 1,030 | 2.4% |

**Comparison to Forward Roll**:
- Forward roll: 41.4% return to WAIT
- Backward roll: 48.4% return to WAIT (safer escape)
- Forward roll punish rate: 10.3%
- Backward roll punish rate: 8.1% (slightly safer)

---

## State 235: ESCAPE (Spotdodge)

### Description
Spotdodge (sidestep) provides invulnerability while staying in place. Performed by pressing L/R + down while shielding. Faster recovery than rolls but doesn't reposition, making it best for dodging single attacks.

### Frame Data (Fox)
| Metric | Value |
|--------|-------|
| Total Duration | 22 frames |
| Invulnerability | Frames 2-15 (14 frames) |
| Vulnerable Startup | Frame 1 (1 frame) |
| Vulnerable Endlag | Frames 16-22 (7 frames) |
| On Ground | 100% |

### Invulnerability by Frame
| Frame Range | Invuln % | Notes |
|-------------|----------|-------|
| 1 | 4% | Startup |
| 2-15 | 100% | Full invincibility |
| 16-22 | 0.4% | Endlag (vulnerable) |

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| GUARD_ON (178) | 35,088 | 50.3% |
| GUARD (179) | 6,541 | 9.4% |
| GUARD_SET_OFF (182) | 6,349 | 9.1% |
| GUARD_OFF (180) | 4,427 | 6.3% |
| GUARD_REFLECT (181) | 4,167 | 6.0% |
| DOWN_STAND_U (186) | 1,786 | 2.6% |
| LANDING_AIR_LW (74) | 1,535 | 2.2% |
| PASSIVE_STAND_B (201) | 1,466 | 2.1% |

**Key patterns**:
- **Shield states (178-182)**: 81.1% - Most spotdodges from shield
- **Getup options (186, 201, 197)**: 7.1% - Spotdodge after knockdown/tech
- **Landing lag (70-74)**: 5.6% - Spotdodge buffered during landing

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| SQUAT (39) | 23,988 | 32.2% |
| WAIT (14) | 16,085 | 21.6% |
| GUARD_ON (178) | 9,976 | 13.4% |
| REFLECTOR_GROUND_STARTUP (360) | 3,038 | 4.1% |
| DAMAGE_FLY_N (88) | 2,971 | 4.0% |
| TURN (18) | 2,562 | 3.4% |
| WALK_SLOW (15) | 2,134 | 2.9% |
| CAPTURE_WAIT (226) | 2,066 | 2.8% |

**Key patterns**:
- **Crouch (39)**: 32.2% - Spotdodge → crouch is very common (CC followup)
- **Neutral (14, 15, 18)**: 27.9% - Return to standing
- **Shield (178)**: 13.4% - Re-shield after spotdodge
- **Shine (360)**: 4.1% - Spotdodge → shine punish
- **Punished (88, 226)**: 6.8% - Hit or grabbed

### Spotdodge vs Roll Comparison

| Metric | Spotdodge | Roll |
|--------|-----------|------|
| Total Frames | 22 | 31 |
| Invuln Frames | 14 (2-15) | 16 (4-19) |
| Vulnerable Endlag | 7 frames | 12 frames |
| Repositioning | None | Full roll distance |
| Punish Rate | 6.8% | 8-10% |

**Spotdodge is faster** but doesn't reposition. Best for dodging single attacks and punishing with immediate crouch or shine.

---

## State 236: ESCAPE_AIR (Airdodge)

### Description
Airdodge provides aerial invulnerability and is the foundation of **wavedashing** - one of Melee's most important advanced techniques. When airdodge contacts the ground, the character lands with 10 frames of lag (LANDING_FALL_SPECIAL) while retaining momentum, creating the wavedash.

### Frame Data (Fox)
| Metric | Value |
|--------|-------|
| Full Duration | 49 frames (if not landing) |
| Invulnerability | Frames 4-29 (26 frames) |
| Vulnerable Startup | Frames 1-3 (3 frames) |
| Vulnerable Endlag | Frames 30-49 (20 frames, helpless) |
| Landing Lag | 10 frames (LANDING_FALL_SPECIAL) |
| On Ground | 0% (aerial) |

### Invulnerability by Frame
| Frame Range | Invuln % | Notes |
|-------------|----------|-------|
| 1-3 | 17-31% | Startup (some invuln starting) |
| 4-29 | 99-100% | Full invincibility |
| 30+ | 2-5% | Helpless fall (FALL_SPECIAL) |

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| JUMP_F (25) | 214,228 | 46.7% |
| JUMP_AERIAL_F (27) | 132,431 | 28.9% |
| JUMP_B (26) | 76,422 | 16.7% |
| JUMP_AERIAL_B (28) | 17,349 | 3.8% |
| FALL (29) | 9,955 | 2.2% |
| PASS (244) | 5,037 | 1.1% |
| KNEE_BEND (24) | 1,875 | 0.4% |

**Key patterns**:
- **First jump (25, 26)**: 63.4% - Wavedash from ground (jump → airdodge)
- **Double jump (27, 28)**: 32.7% - Waveland or aerial airdodge
- **Fall (29)**: 2.2% - Airdodge from falling state
- **Platform drop (244)**: 1.1% - Shield drop → waveland

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| LANDING_FALL_SPECIAL (43) | 448,767 | 97.9% |
| FALL_SPECIAL (35) | 4,357 | 1.0% |
| DAMAGE_FLY_N (88) | 1,576 | 0.3% |
| DAMAGE_FLY_TOP (90) | 719 | 0.2% |
| DEAD_DOWN (0) | 344 | 0.1% |

**Critical finding**: **97.9% of airdodges result in wavedash landing** (43). This confirms that in competitive Melee, airdodge is almost exclusively used for wavedashing rather than as a defensive aerial option.

### Wavedash Mechanics

**Execution**:
1. Jump (3-frame jumpsquat for Fox)
2. Immediately airdodge diagonally into ground
3. Land with 10 frames of lag (LANDING_FALL_SPECIAL)
4. Retain horizontal momentum from airdodge angle

**Total Wavedash Duration**: Jumpsquat (3) + minimal air time + landing lag (10) = ~14-17 frames

**Wavedash Uses**:
- Ground movement with full action availability
- Approach with shield/attack options
- Retreating while facing opponent
- Platform movement (wavelanding)
- Ledgedash (from ledge)

### Airdodge SD Risk
- 0.1% of airdodges result in DEAD_DOWN (0) - airdodging off stage

---

## State 237: REBOUND_STOP (Attack Clank Stop)

### Description
REBOUND_STOP is the brief freeze when two attacks clank (hitboxes collide within damage range). This is the initial stop before entering the full rebound animation. Very short duration.

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 6 frames |
| Average Duration | 5.6 frames |
| Maximum Duration | 13 frames |
| On Ground | 100% |
| Invulnerability | 1.6% |

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| ATTACK_LW_3 (56) | 2,133 | 33.0% |
| ATTACK_11 (44) | 1,707 | 26.4% |
| ATTACK_DASH (50) | 974 | 15.1% |
| ATTACK_LW_3 (57) | 874 | 13.5% |
| ATTACK_HI_4 (63) | 459 | 7.1% |
| ATTACK_S_3_S (53) | 123 | 1.9% |

**Key insight**: Dtilt (56, 57), jab (44), and dash attack (50) are the most common attacks to clank. These are fast grounded pokes that frequently trade hitboxes.

### Exit Conditions
REBOUND_STOP always transitions to REBOUND (238).

---

## State 238: REBOUND (Attack Clank)

### Description
REBOUND is the full rebound animation after attacks clank. The character is pushed back and cannot act until the animation completes. Duration depends on the damage difference between clanking attacks.

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 11 frames |
| Average Duration | 10.6 frames |
| Maximum Duration | 16 frames |
| On Ground | 100% |
| Invulnerability | 0.1% |

### Rebound Formula
Rebound duration: `R = Roundup(0.559 * (d + 10))`
Where `d` = damage of the weaker attack (rounded down)

The character using the weaker attack exits rebound first, gaining frame advantage.

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| REBOUND_STOP (237) | 6,447 | 100% |

REBOUND is only entered from REBOUND_STOP.

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| WAIT (14) | 3,025 | 46.9% |
| GUARD_ON (178) | 1,177 | 18.3% |
| SQUAT (39) | 672 | 10.4% |
| ATTACK_11 (44) | 343 | 5.3% |
| DASH (20) | 263 | 4.1% |
| DAMAGE_FLY_N (88) | 177 | 2.7% |

**Key patterns**:
- **Neutral (14)**: 46.9% - Return to standing after clank
- **Shield (178)**: 18.3% - Shield after clank (defensive)
- **Crouch (39)**: 10.4% - Crouch after clank
- **Jab (44)**: 5.3% - Immediate counter-attack
- **Punished (88)**: 2.7% - Hit during rebound

### Clank Mechanics

**When Clanks Occur**:
- Two grounded attack hitboxes overlap
- Both attacks within 10% damage of each other
- If one attack deals >10% more damage, only the weaker attack rebounds

**Clank Properties**:
- Creates white "bubble" visual and "ting" sound
- Stronger attack can still connect during clank
- Weaker attack's hitbox is cancelled
- Both characters experience freeze frames

---

## Competitive Applications

### Roll Usage
- **Forward roll**: 50,763 instances (57% of rolls)
- **Backward roll**: 38,410 instances (43% of rolls)
- Forward roll preferred for repositioning
- Backward roll slightly safer (8.1% vs 10.3% punish rate)

### Spotdodge Usage
- **62,746 instances** - More common than either roll type
- 32.2% exit to crouch - spotdodge → CC is a common pattern
- 4.1% exit to shine - punish option
- Fastest grounded defensive option (22 frames total)

### Airdodge/Wavedash Usage
- **131,838 instances** - Most used defensive state
- **97.9% are wavedashes** (exit to LANDING_FALL_SPECIAL)
- 63.4% from first jump (standard wavedash)
- 32.7% from double jump (waveland)
- Only 1.0% enter helpless fall (full airdodge in air)

### Defensive Option Comparison

| Option | Frames | Invuln | Endlag | Best Use |
|--------|--------|--------|--------|----------|
| Spotdodge | 22 | 14 (2-15) | 7 | Single attack dodge, punish setup |
| Roll | 31 | 16 (4-19) | 12 | Escape pressure, reposition |
| Wavedash | ~14-17 | 26 | 10 | Movement, approach, retreat |
| Airdodge (full) | 49 | 26 | 20 | Avoid aerial, emergency |

---

## Roll/Dodge Usage by Situation

### From Shield
| State | Roll F | Roll B | Spotdodge |
|-------|--------|--------|-----------|
| GUARD_ON (178) | 54.5% | 52.8% | 50.3% |
| GUARD (179) | 20.6% | 22.6% | 9.4% |
| GUARD_SET_OFF (182) | 13.4% | 16.5% | 9.1% |
| Total from shield | 95.0% | 99.9% | 81.1% |

**Insight**: Rolls are almost exclusively shield options. Spotdodge has more diverse entry conditions.

### Punish Rates
| Option | Grabbed (226) | Hit (88) | Total Punish |
|--------|---------------|----------|--------------|
| Roll Forward | 2.6% | 3.4% | ~10.3% |
| Roll Backward | 2.4% | 2.6% | ~8.1% |
| Spotdodge | 2.8% | 4.0% | ~6.8% |

**Spotdodge is least punishable** due to shorter duration, but rolls provide repositioning.

---

## Related States

| Relationship | States |
|--------------|--------|
| Entry from | GUARD states (178-182), jumps (25-28), landings |
| Roll/Spotdodge lead to | WAIT (14), SQUAT (39), GUARD_ON (178), attacks |
| Airdodge leads to | LANDING_FALL_SPECIAL (43), FALL_SPECIAL (35) |
| Rebound leads to | WAIT (14), GUARD_ON (178), attacks |

---

## Data Sources

- Parquet replay data: 288,753,124 total frames analyzed
- SmashWiki: Roll frame data, invulnerability timing
- SmashWiki: Spotdodge frame data
- SmashWiki: Airdodge mechanics, wavedash
- SmashWiki: Clank/rebound mechanics
- Smashboards: Wavedash frame data (jumpsquat + 10 landing lag)
