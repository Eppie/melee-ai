# Tech (Successful) States Documentation

Action states 199-204 covering successful techs: tech in place, tech roll forward/back, wall tech, wall tech jump, and ceiling tech.

## Overview

Teching (officially "Ukemi") allows a character to recover from tumble by pressing L/R within 20 frames of hitting a surface. Successful techs provide invulnerability frames and prevent knockdown (missed tech) situations.

| State | ID | Internal Name | Instances | % of Gameplay | Median Duration |
|-------|-----|---------------|-----------|---------------|-----------------|
| Tech in Place | 199 | PASSIVE | 53,427 | 0.74% | 26 frames |
| Tech Roll Forward | 200 | PASSIVE_STAND_F | 39,153 | 0.67% | 40 frames |
| Tech Roll Back | 201 | PASSIVE_STAND_B | 33,125 | 0.54% | 40 frames |
| Wall Tech | 202 | PASSIVE_WALL | 911 | 0.005% | 12 frames |
| Wall Tech Jump | 203 | PASSIVE_WALL_JUMP | 3,461 | 0.02% | 14 frames |
| Ceiling Tech | 204 | PASSIVE_CEIL | 55 | 0.0005% | 26 frames |

**Total tech frames**: 5,672,476 (1.96% of gameplay)

---

## Ground Tech Distribution

When teching on the ground, players choose between three options:

| Tech Option | Instances | % of Ground Techs | Usage Rate |
|-------------|-----------|-------------------|------------|
| Tech in Place | 52,764 | 42.3% | Most common |
| Tech Roll Forward | 38,952 | 31.2% | Second |
| Tech Roll Back | 32,996 | 26.5% | Least common |

**Key insight**: Tech in place is preferred despite being the shortest option, likely because it has the fastest recovery and allows immediate counterplay.

---

## State 199: PASSIVE (Tech in Place)

### Description
Tech in place is executed by pressing L/R without a directional input when hitting a surface during tumble. The character performs a quick recovery animation while staying in the same position.

### Frame Data
| Metric | Value |
|--------|-------|
| Total Duration | 26 frames |
| Invulnerability | Frames 1-20 (20 frames) |
| Vulnerable End Lag | Frames 21-26 (6 frames) |
| Actionable | Frame 27 |

### Duration Distribution (Fox)
| Duration | Frequency | Notes |
|----------|-----------|-------|
| 26 frames | 18,802 (35%) | Standard full animation |
| 52 frames | 5,091 (10%) | Likely buffered action |
| 20-25 frames | ~7,263 (14%) | Interrupted early |
| 1 frame | 608 (1%) | Immediate transition |

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| DAMAGE_FLY_TOP (90) | 40,076 | 42.7% |
| DAMAGE_FLY_N (88) | 34,379 | 36.6% |
| DAMAGE_FLY_HI (87) | 9,840 | 10.5% |
| DAMAGE_FLY_LW (89) | 6,594 | 7.0% |
| DAMAGE_FLY_ROLL (91) | 1,839 | 2.0% |
| DAMAGE_FALL (38) | 530 | 0.6% |
| DOWN_BOUND_U (183) | 216 | 0.2% |

**Primary entry**: 98.9% from tumble/knockback states (87-91)

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| WAIT (14) | 18,466 | 19.7% |
| SQUAT (39) | 12,422 | 13.2% |
| GUARD_ON (178) | 12,185 | 13.0% |
| FALL (29) | 12,158 | 12.9% |
| DAMAGE_FLY_TOP (90) | 9,525 | 10.1% |
| DAMAGE_FLY_N (88) | 6,542 | 7.0% |
| REFLECTOR_GROUND_STARTUP (360) | 3,327 | 3.5% |
| DASH (20) | 2,945 | 3.1% |

**Key patterns**:
- 59.0% escape to neutral (stand, crouch, shield, fall, dash)
- 20.2% get hit immediately after (90, 88 damage states)
- 3.5% shine OoS (Fox-specific tech chase punish)

### Invulnerability Frame Analysis
| Frame | Invulnerable % | Notes |
|-------|----------------|-------|
| 1-18 | 100% | Fully invulnerable |
| 19 | 99.9% | Transition begins |
| 20 | 99.4% | Last invuln frame |
| 21-26 | 16-28% | Vulnerable (some variance) |
| 27+ | 96%+ | Actionable, can shield |

---

## State 200: PASSIVE_STAND_F (Tech Roll Forward)

### Description
Tech roll forward is executed by pressing L/R while holding the control stick toward the direction the character is facing. The character rolls forward with intangibility before becoming vulnerable.

### Frame Data
| Metric | Value |
|--------|-------|
| Total Duration | 40 frames |
| Invulnerability | Frames 1-20 (20 frames) |
| Vulnerable Rolling | Frames 21-40 (20 frames) |
| Actionable | Frame 41 |

### Roll Distance
| Metric | Value |
|--------|-------|
| Average Distance | 48.1 Melee units |
| Median Distance | 38.1 Melee units |
| Maximum Distance | 175.5 Melee units |

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| DAMAGE_FLY_TOP (90) | 29,921 | 57.7% |
| DAMAGE_FLY_N (88) | 14,396 | 27.7% |
| DAMAGE_FLY_HI (87) | 3,761 | 7.3% |
| DAMAGE_FLY_LW (89) | 2,426 | 4.7% |
| DAMAGE_FLY_ROLL (91) | 709 | 1.4% |

**Notable**: Tech roll forward is disproportionately used after upward knockback (DAMAGE_FLY_TOP), likely to escape aerial followups.

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| GUARD_ON (178) | 11,403 | 22.0% |
| WAIT (14) | 7,542 | 14.5% |
| Grabbed (226) | 7,168 | 13.8% |
| DAMAGE_FLY_TOP (90) | 4,412 | 8.5% |
| DAMAGE_FLY_N (88) | 4,079 | 7.9% |
| SQUAT (39) | 3,784 | 7.3% |

**Key insight**: Tech roll forward has a 13.8% grab rate - significantly higher than tech in place, making it the most punishable tech option.

---

## State 201: PASSIVE_STAND_B (Tech Roll Back)

### Description
Tech roll backward is executed by pressing L/R while holding the control stick away from facing direction. Provides the same frame data as forward roll but moves in the opposite direction.

### Frame Data
| Metric | Value |
|--------|-------|
| Total Duration | 40 frames |
| Invulnerability | Frames 1-20 (20 frames) |
| Vulnerable Rolling | Frames 21-40 (20 frames) |
| Actionable | Frame 41 |

### Roll Distance
| Metric | Value |
|--------|-------|
| Average Distance | 54.5 Melee units |
| Median Distance | 42.3 Melee units |
| Maximum Distance | 175.6 Melee units |

**Note**: Tech roll back travels slightly farther than tech roll forward on average.

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| DAMAGE_FLY_TOP (90) | 24,299 | 59.1% |
| DAMAGE_FLY_N (88) | 11,432 | 27.8% |
| DAMAGE_FLY_HI (87) | 2,977 | 7.2% |
| DAMAGE_FLY_LW (89) | 1,579 | 3.8% |
| DAMAGE_FLY_ROLL (91) | 821 | 2.0% |

### Exit Conditions
Similar to tech roll forward, with high grab and damage rates in the vulnerable 20 frames.

---

## State 202: PASSIVE_WALL (Wall Tech)

### Description
Wall tech is executed by pressing L/R within 20 frames of colliding with a wall during tumble/knockback. The character briefly sticks to the wall with intangibility before dropping off.

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 12 frames |
| Average Duration | 15.1 frames |
| Invulnerability | Frames 1-14 (100% invuln) |
| Vulnerable | Frame 15+ (~4% invuln) |

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| DAMAGE_FLY_N (88) | 471 | 49.9% |
| DAMAGE_FLY_TOP (90) | 225 | 23.8% |
| DAMAGE_FLY_ROLL (91) | 97 | 10.3% |
| DAMAGE_FLY_HI (87) | 96 | 10.2% |
| DAMAGE_FLY_LW (89) | 45 | 4.8% |

**Key insight**: DAMAGE_FLY_N (horizontal knockback) leads to wall techs most often - logical since horizontal knockback trajectories are most likely to hit walls.

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| PASSIVE_WALL_JUMP (203) | 392 | 100% |

**Critical finding**: Wall tech (202) ALWAYS transitions to wall tech jump (203) when players hold jump. The states are linked - 202 is the "wall stick" and 203 is the actual jump off.

---

## State 203: PASSIVE_WALL_JUMP (Wall Tech Jump)

### Description
Wall tech jump occurs when the player holds jump during a wall tech. The character pushes off the wall with a jump, enabling recovery and potential counterattacks.

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 14 frames |
| Average Duration | 16.2 frames |
| Invulnerability | Frames 1-14 (~95-100%) |
| Airborne after | Frame 15+ |

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| JUMP_AERIAL_F (27) | 1,063 | 29.0% |
| DAMAGE_FLY_N (88) | 960 | 26.1% |
| JUMP_AERIAL_B (28) | 491 | 13.4% |
| PASSIVE_WALL (202) | 294 | 8.0% |
| FALL (29) | 200 | 5.5% |
| DAMAGE_FLY_TOP (90) | 188 | 5.1% |
| CLIFF_WAIT (253) | 66 | 1.8% |

**Key patterns**:
- 42.4% from double jump states (27, 28) - players wall teching during recovery
- 34.5% from tumble states (88, 90, etc.) - direct wall tech after being hit
- 8.0% from PASSIVE_WALL (202) - buffered wall tech jump

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| FALL (29) | 3,104 | 89.7% |
| ATTACK_AIR_B (67) | 71 | 2.1% |
| JUMP_AERIAL_F/B (27/28) | 65 | 1.9% |
| LANDING (42) | 14 | 0.4% |

**Primary usage**: 89.7% transition to falling - wall tech jump is primarily used for recovery rather than immediate counterattack.

---

## State 204: PASSIVE_CEIL (Ceiling Tech)

### Description
Ceiling tech is executed by pressing L/R within 20 frames of hitting a ceiling during upward knockback. Very rare in Fox dittos since most upward knockback results in death rather than ceiling collision.

### Frame Data
| Metric | Value |
|--------|-------|
| Total Duration | 26 frames (fixed) |
| Invulnerability | Frames 1-14 |
| Vulnerable Falling | Frames 15-26 |

### Statistics
- **Total instances**: 55 (only 0.0005% of gameplay)
- **Average damage %**: 72.1%
- **Position**: avg_y = -45.6 (below main stage, near platform underside)

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| DAMAGE_FLY_N (88) | 38 | 69.1% |
| DAMAGE_FLY_TOP (90) | 7 | 12.7% |
| DAMAGE_FLY_LW (89) | 5 | 9.1% |
| DAMAGE_FLY_HI (87) | 2 | 3.6% |
| DAMAGE_FLY_ROLL (91) | 1 | 1.8% |

**Notable**: Despite being a "ceiling" tech, most entries come from horizontal knockback (88) rather than upward (90). This likely occurs under stages like Battlefield/Dreamland platforms.

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| FALL (29) | 49 | 89.1% |
| FIRE_FOX_AIR_STARTUP (354) | 6 | 10.9% |

---

## Tech Input Window

### Timing
| Phase | Window |
|-------|--------|
| During hitlag | 1 frame if not last hitlag frame, or 20 frames if last hitlag frame |
| After hitlag | Standard 20-frame window before surface contact |
| Tech lockout | 40 frames after failed/missed tech input |

### Input Method
- Press L or R trigger
- For tech roll: hold control stick left/right simultaneously
- For wall tech jump: hold jump button during wall tech

### SDI/ASDI Integration
- **SDI**: Can bump into walls/ceilings during hitlag to enable teching
- **ASDI down**: Can bump into ground after hitlag ends to tech, even if knockback goes upward

---

## Tech Chase Analysis

### Damage % When Teching (Fox)
| Tech Type | Median % | 75th Percentile | Max % |
|-----------|----------|-----------------|-------|
| Tech in Place | 54% | 80% | 264% |
| Tech Roll F | 49% | 75% | 220% |
| Tech Roll B | 47% | 77% | 211% |
| Wall Tech | 86% | 110% | 183% |
| Wall Tech Jump | 66% | 95% | 168% |
| Ceiling Tech | 76% | 96% | 175% |

**Key insight**: Wall techs occur at higher percentages on average, reflecting that wall collision situations require stronger knockback to reach walls.

### Post-Tech Outcomes
From tech exit analysis:
- **Tech in Place**: 20.2% get hit immediately, 59% escape to neutral
- **Tech Roll Forward**: Higher grab rate (13.8%), vulnerable for 20 frames post-invuln
- **Tech Roll Back**: Similar to forward roll, slightly more distance gained

---

## Invulnerability Summary

| State | Invuln Frames | Total Frames | Invuln % |
|-------|---------------|--------------|----------|
| PASSIVE | 1-20 | 26 | 81.3% |
| PASSIVE_STAND_F | 1-20 | 40 | 53.6% |
| PASSIVE_STAND_B | 1-20 | 40 | 53.0% |
| PASSIVE_WALL | 1-14 | ~15 | 73.1% |
| PASSIVE_WALL_JUMP | 1-14 | ~16 | 73.2% |
| PASSIVE_CEIL | 1-14 | 26 | 54.8% |

---

## Competitive Applications

### Tech in Place
- **Fastest recovery** (26 frames vs 40 for rolls)
- **Shine OoS punishable** (3.5% of tech in place → shine)
- **Best for immediate counterplay**

### Tech Rolls
- **Position change** for escaping pressure
- **Predictable** due to fixed distance and long duration
- **Higher grab vulnerability** (13.8% for forward roll)

### Wall Tech Jump
- **Recovery tool** when hit into stage walls
- **"Tech Check"**: Punish moves like Falcon Dive by wall teching → immediate aerial
- **89.7% return to neutral** (fall state)

### Ceiling Tech
- **Extremely rare** (55 instances in dataset)
- **Survival option** under stages on platform maps
- **Peach/Zelda receive no invulnerability** (character-specific)

---

## Character Notes (Fox-Specific)

From the Fox ditto dataset:
- Tech in place → Shine (360) occurs 3.5% of the time, a signature punish option
- Wall tech jump frequently transitions to aerials (bair 2.1%, etc.)
- Fire Fox (354) sometimes used immediately after ceiling tech for recovery

---

## Related States

| Relationship | States |
|--------------|--------|
| Entry from | DAMAGE_FLY (87-91), DAMAGE_FALL (38), DOWN_BOUND (183, 191) |
| Alternative to | DOWN_WAIT (184, 192) - missed tech knockdown |
| Pairs with | GUARD (179) - shield after tech, REFLECTOR (360) - shine after tech |

---

## Data Sources

- Parquet replay data: 288,753,124 total frames analyzed
- SmashWiki: Tech mechanics and frame data
- SmashWiki: Wall tech specifics and SDI integration
