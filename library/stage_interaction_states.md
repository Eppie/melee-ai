# Stage Interaction States Documentation

Action states 244-251 covering platform drops, teetering, wall/ceiling bounces, and missed ledge grabs.

## Overview

Stage interaction states handle character responses to environmental collisions and edge cases. These include dropping through platforms, teetering on edges, bouncing off walls and ceilings (failed techs), and slipping past ledge sweetspots.

| State | ID | Internal Name | Instances | % of Gameplay | Median Duration |
|-------|-----|---------------|-----------|---------------|-----------------|
| Platform Drop | 244 | PASS | 44,791 | 0.47% | 20 frames |
| Teeter Start | 245 | OTTOTTO | 18,363 | 0.03% | 2 frames |
| Teeter Hold | 246 | OTTOTTO_WAIT | 2,188 | 0.01% | 8 frames |
| Wall Bounce | 247 | FLY_REFLECT_WALL | 2,256 | 0.02% | 29 frames |
| Ceiling Bounce | 248 | FLY_REFLECT_CEIL | 43 | 0.0004% | 24 frames |
| Wall Stop | 249 | STOP_WALL | 62 | 0.0004% | 20 frames |
| Ceiling Stop | 250 | STOP_CEIL | 275 | 0.0006% | 6 frames |
| Missed Ledge | 251 | MISS_FOOT | 12,780 | 0.06% | 11 frames |

**Total stage interaction frames**: 1,701,123 (0.59% of gameplay)

---

## State 244: PASS (Platform Drop-Through)

### Description
PASS is the platform drop-through animation. This occurs when a character drops through a soft (pass-through) platform by pressing down while standing, crouching, or shield dropping. The character falls through the platform and enters an aerial state.

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 20 frames |
| Average Duration | 30.3 frames |
| Maximum Duration | 339 frames |
| On Ground | 0% (aerial state) |
| Invulnerability | 3.8% |

### Position Data
| Metric | Value |
|--------|-------|
| Average Y | 18.4 (platform height) |
| Min Y | -90.1 |
| Max Y | 53.7 (Battlefield top platform) |
| Y Std Dev | 12.04 |

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| GUARD_ON (178) | 63,370 | 39.7% |
| SQUAT (39) | 37,092 | 23.2% |
| GUARD (179) | 34,161 | 21.4% |
| GUARD_SET_OFF (182) | 21,760 | 13.6% |
| GUARD_REFLECT (181) | 2,583 | 1.6% |
| DASH (20) | 366 | 0.2% |

**Key patterns**:
- **Shield drop (178, 179, 182, 181)**: 76.3% - The dominant platform drop method
- **Crouch drop (39)**: 23.2% - Pressing down from crouch
- **Other**: 0.5% - Rare edge cases

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| LANDING (42) | 50,553 | 31.6% |
| ATTACK_AIR_B (67) | 49,759 | 31.1% |
| ATTACK_AIR_LW (69) | 15,506 | 9.7% |
| JUMP_AERIAL_F (27) | 7,080 | 4.4% |
| BLASTER_AIR_LOOP (344) | 6,070 | 3.8% |
| ESCAPE_AIR (236) | 5,037 | 3.1% |
| REFLECTOR_AIR_STARTUP (365) | 4,826 | 3.0% |
| ATTACK_AIR_N (65) | 4,655 | 2.9% |
| ATTACK_AIR_F (66) | 3,075 | 1.9% |
| JUMP_AERIAL_B (28) | 2,836 | 1.8% |

**Key patterns**:
- **Immediate attack (67, 69, 65, 66, 68)**: 46.3% - Platform drop into aerial
- **Landing (42, 43)**: 32.6% - Dropped through low platform
- **Double jump (27, 28)**: 6.2% - Repositioning
- **Special moves (344, 365)**: 6.8% - Fox laser/shine from platform

### Shield Drop Technique

Shield dropping is the primary method for dropping through platforms in competitive play:

1. While shielding on a platform, tilt the control stick to a specific angle
2. The shield tilt triggers platform drop without releasing shield
3. Allows immediate aerial attacks with minimal lag

**Advantages over crouch drop**:
- Faster transition to actionable state
- Maintains defensive option until drop initiates
- Can be performed from shield stun

---

## State 245: OTTOTTO (Teeter Start)

### Description
OTTOTTO (Japanese onomatopoeia for wobbling) is the initial teeter animation when a character walks slowly toward an edge. This is a brief transitional state that either leads to full teeter (OTTOTTO_WAIT) or is cancelled by player action.

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 2 frames |
| Average Duration | 4.4 frames |
| Maximum Duration | 42 frames |
| On Ground | 100% |
| Invulnerability | 4.7% |

### Position Data
| Metric | Value |
|--------|-------|
| Average Y | 17.2 |
| Min Y | -35.5 |
| Max Y | 54.4 |

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| LANDING_FALL_SPECIAL (43) | 5,106 | 24.1% |
| LANDING_AIR_LW (70) | 2,780 | 13.1% |
| WAIT (14) | 2,773 | 13.1% |
| RUN_BRAKE (23) | 2,110 | 10.0% |
| LANDING (42) | 2,000 | 9.4% |
| LANDING_AIR_HI (73) | 1,206 | 5.7% |
| LANDING_AIR_B (72) | 1,107 | 5.2% |
| RUN (21) | 933 | 4.4% |
| LANDING_AIR_LW (74) | 836 | 3.9% |
| LANDING_AIR_F (71) | 483 | 2.3% |

**Key patterns**:
- **Wavedash landings (43)**: 24.1% - Wavedashing to edge
- **Aerial landings (70-74)**: 30.2% - Landing near edge after aerials
- **Movement deceleration (23, 21)**: 14.4% - Running/stopping near edge
- **Standing/walking (14)**: 13.1% - Walking slowly toward edge

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| TURN (18) | 5,121 | 24.2% |
| DASH (20) | 4,590 | 21.7% |
| SQUAT (39) | 3,248 | 15.3% |
| GUARD_ON (178) | 2,813 | 13.3% |
| OTTOTTO_WAIT (246) | 2,187 | 10.3% |
| FALL (29) | 1,377 | 6.5% |
| KNEE_BEND (24) | 716 | 3.4% |
| ATTACK_11 (44) | 226 | 1.1% |

**Key patterns**:
- **Movement cancel (18, 20, 24)**: 49.3% - Immediately move away
- **Defensive cancel (39, 178)**: 28.6% - Crouch or shield
- **Full teeter (246)**: 10.3% - Transition to wobble animation
- **Fall off (29)**: 6.5% - Walk off edge

---

## State 246: OTTOTTO_WAIT (Teeter Hold)

### Description
OTTOTTO_WAIT is the sustained teeter animation where the character wobbles on the edge. This occurs when the player doesn't input any action after OTTOTTO initiates. The character can remain in this state until cancelled by any input.

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 8 frames |
| Average Duration | 13.0 frames |
| Maximum Duration | 167 frames |
| On Ground | 100% |
| Invulnerability | 2.9% |

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| OTTOTTO (245) | ~2,187 | ~100% |

OTTOTTO_WAIT is only entered from OTTOTTO (245).

### Exit Conditions
Players can cancel teeter with any action:
- Movement (walk, dash, turn)
- Crouch
- Shield
- Jump
- Attack
- Special move

**Common exits include**: WAIT (14), TURN (18), DASH (20), SQUAT (39), GUARD_ON (178), attacks, etc.

### Teeter Mechanics

**When teeter occurs**:
- Walking slowly toward an edge (not running)
- Landing near an edge during specific animations
- Cannot teeter from running - character will run off

**Teeter cancel**:
- Any action cancels teeter immediately
- Used in some edge setups (e.g., teeter → dtilt for stage spike)

**Hurtbox shift**:
- During teeter animation, the character's body shifts backward
- Can sometimes dodge attacks that would hit a standing character

---

## State 247: FLY_REFLECT_WALL (Wall Bounce)

### Description
FLY_REFLECT_WALL occurs when a character in knockback/tumble hits a wall without teching. The character bounces off the wall and continues in tumble state. This is a "missed wall tech" situation.

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 29 frames |
| Average Duration | 30.4 frames |
| Maximum Duration | 118 frames |
| On Ground | 0% (aerial) |
| Invulnerability | 49.1% overall |

### Invulnerability by Frame
| Frame Range | Invuln % | Notes |
|-------------|----------|-------|
| 1-14 | 100% | Full invincibility during bounce |
| 15 | 99.4% | Transition frame |
| 16+ | 0% | Fully vulnerable |

**Total invuln frames**: 14-15 frames

### Position Data
| Metric | Value |
|--------|-------|
| Average X | 9.4 |
| Average Y | -53.3 (below stage) |
| Min Y | -146.2 |
| Max Y | 43.7 |

### Damage Statistics
| Metric | Value |
|--------|-------|
| Average % | 74.8% |
| Median % | 79% |
| Min % | 3% |
| Max % | 200% |

**Note**: Wall bounces occur at moderate-high percentages, as strong knockback is needed to reach stage walls.

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| DAMAGE_FLY_N (88) | 1,497 | 64.9% |
| DAMAGE_FLY_TOP (90) | 330 | 14.3% |
| DAMAGE_FLY_HI (87) | 179 | 7.8% |
| DAMAGE_FLY_LW (89) | 151 | 6.5% |
| DAMAGE_FLY_ROLL (91) | 145 | 6.3% |

**Key insight**: 64.9% from horizontal knockback (88) - logical since horizontal trajectories are most likely to hit vertical walls.

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| DEAD_DOWN (0) | 1,644 | 72.7% |
| DAMAGE_FALL (38) | 196 | 8.7% |
| FIRE_FOX_AIR_STARTUP (354) | 181 | 8.0% |
| JUMP_AERIAL_F (27) | 84 | 3.7% |
| JUMP_AERIAL_B (28) | 72 | 3.2% |
| DOWN_BOUND_U (183) | 45 | 2.0% |
| PASSIVE_WALL_JUMP (203) | 20 | 0.9% |

**Key patterns**:
- **Death (0, 1, 2)**: 73.0% - Most wall bounces are fatal
- **Recovery attempt (354, 27, 28)**: 14.9% - Fire Fox or double jump
- **Tumble continue (38)**: 8.7% - Still in knockback
- **Wall tech (203)**: 0.9% - Late tech input sometimes works

### Wall Bounce vs Wall Tech

| Situation | State | Invuln | Recovery |
|-----------|-------|--------|----------|
| Successful wall tech | PASSIVE_WALL (202) | 14 frames | Wall tech jump available |
| Failed wall tech | FLY_REFLECT_WALL (247) | 14 frames | Bounce away, tumble continues |

Both states provide ~14 frames of invincibility, but wall tech allows controlled recovery while wall bounce often leads to death.

---

## State 248: FLY_REFLECT_CEIL (Ceiling Bounce)

### Description
FLY_REFLECT_CEIL occurs when a character in upward knockback hits a ceiling without teching. Very rare in Fox dittos since upward knockback at high percentages typically results in star KO rather than ceiling collision.

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 24 frames |
| Average Duration | 24.9 frames |
| Maximum Duration | 56 frames |
| On Ground | 0% (aerial) |
| Invulnerability | 57.9% |
| Total Instances | 43 |

### Position Data
| Metric | Value |
|--------|-------|
| Average Y | -59.5 (below main stage) |
| Min Y | -121.6 |
| Max Y | -21.0 |

**Note**: Negative Y values indicate these occur under stages (Battlefield/Dreamland undersides, Fountain of Dreams structure).

### Damage Statistics
| Metric | Value |
|--------|-------|
| Average % | 81.2% |
| Median % | 85% |
| Min % | 14% |
| Max % | 175% |

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| DAMAGE_FLY_TOP (90) | 28 | 65.1% |
| DAMAGE_FLY_N (88) | 7 | 16.3% |
| FLY_REFLECT_WALL (247) | 6 | 14.0% |
| DAMAGE_FLY_ROLL (91) | 2 | 4.7% |

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| DEAD_DOWN (0) | 31 | 72.1% |
| DAMAGE_FALL (38) | 8 | 18.6% |
| PASSIVE_CEIL (204) | 1 | 2.3% |
| JUMP_AERIAL (27, 28) | 2 | 4.7% |

**Key insight**: 72% result in death - ceiling bounces are typically fatal as they occur at high knockback situations.

---

## State 249: STOP_WALL (Wall Stop)

### Description
STOP_WALL occurs when a grounded character collides with a wall during movement (typically dashing). The character stops against the wall briefly before returning to normal state. Very rare in competitive play.

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 20 frames |
| Average Duration | 17.4 frames |
| Maximum Duration | 20 frames |
| On Ground | 100% |
| Invulnerability | 23.2% |
| Total Instances | 62 |

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| DASH (20) | 59 | 95.2% |
| LANDING (42) | 1 | 1.6% |
| TURN (18) | 1 | 1.6% |
| LANDING_FALL_SPECIAL (43) | 1 | 1.6% |

**Key insight**: 95% from dashing - character ran into a wall.

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| WAIT (14) | 24 | 38.7% |
| FALL (29) | 21 | 33.9% |
| WALK_SLOW (15) | 7 | 11.3% |
| SQUAT (39) | 4 | 6.5% |
| Other | 6 | 9.7% |

---

## State 250: STOP_CEIL (Ceiling Stop)

### Description
STOP_CEIL occurs when an airborne character (typically during double jump) collides with a ceiling. The character's upward momentum stops and they begin falling. Occurs on stages with low ceilings or platforms above.

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 6 frames |
| Average Duration | 6.0 frames |
| Maximum Duration | 12 frames |
| On Ground | 0% (aerial) |
| Invulnerability | 3.7% |
| Total Instances | 275 |

### Position Data
| Metric | Value |
|--------|-------|
| Average Y | -28.6 |
| Min Y | -74.0 |
| Max Y | -19.9 |

**Note**: Negative Y indicates these occur under stage structures (platform undersides, stage ceilings).

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| JUMP_AERIAL_F (27) | 153 | 55.4% |
| JUMP_AERIAL_B (28) | 119 | 43.1% |
| FALL (29) | 2 | 0.7% |
| DAMAGE_FALL (38) | 1 | 0.4% |
| ATTACK_AIR_HI (68) | 1 | 0.4% |

**Key insight**: 98.5% from double jump - jumping under low platforms or stage undersides.

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| FALL (29) | 274 | 99.6% |
| DAMAGE_FLY_N (88) | 1 | 0.4% |

**Key insight**: Nearly always transitions to fall - momentum nullified, character drops.

---

## State 251: MISS_FOOT (Missed Ledge Sweetspot)

### Description
MISS_FOOT occurs when a character attempts to grab the ledge but overshoots or mispositions, slipping past the sweetspot. The character slides along the stage edge and can still grab the ledge in most cases. Also called "slipping" or "missing the sweetspot."

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 11 frames |
| Average Duration | 12.9 frames |
| Maximum Duration | 76 frames |
| On Ground | 0% (aerial) |
| Invulnerability | 0.1% |
| Total Instances | 12,780 |

### Position Data
| Metric | Value |
|--------|-------|
| Average Y | 7.8 (near stage level) |
| Min Y | -89.2 |
| Max Y | 55.5 |

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| GUARD_REFLECT (181) | 6,760 | 52.9% |
| GUARD_ON (178) | 1,396 | 10.9% |
| DAMAGE_N_2 (80) | 1,213 | 9.5% |
| DOWN_FORWARD_U (189) | 909 | 7.1% |
| GUARD (179) | 873 | 6.8% |
| DAMAGE_N_1 (79) | 806 | 6.3% |
| DAMAGE_HI_1 (76) | 355 | 2.8% |
| PASSIVE_STAND_B (201) | 328 | 2.6% |
| LANDING (42) | 307 | 2.4% |

**Key patterns**:
- **Powershield/shield stun (181, 178, 179)**: 70.6% - Shield pressure near edge pushes character off
- **Damage/hitstun (80, 79, 76)**: 18.6% - Knocked toward ledge area
- **Getup options (189, 201)**: 9.7% - Getup roll toward ledge

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| CLIFF_CATCH (252) | 6,688 | 52.3% |
| LANDING (42) | 5,418 | 42.4% |
| DAMAGE_FALL (38) | 460 | 3.6% |
| DAMAGE_AIR_2 (85) | 403 | 3.2% |
| DAMAGE_FLY_N (88) | 342 | 2.7% |
| CAPTURE_PULLED_HI (223) | 185 | 1.4% |

**Key patterns**:
- **Ledge grab (252)**: 52.3% - Successfully recovers to ledge
- **Landing (42)**: 42.4% - Lands on stage (didn't actually go off)
- **Hit during slip (85, 88, 38)**: 9.5% - Punished while vulnerable

### Missed Sweetspot Mechanics

**When MISS_FOOT occurs**:
1. Character's ECB (environment collision box) passes the ledge grab zone
2. Character slides along stage edge
3. Ledge grab is still possible if within range

**Recovery from MISS_FOOT**:
- 52.3% still grab ledge (delayed grab)
- 42.4% land on stage (shallow miss)
- 5.3% punished during slip (vulnerable)

**Causes**:
- Shield pushed off edge (70.6%)
- Knocked toward ledge (18.6%)
- Getup roll overshoots (9.7%)

---

## Competitive Applications

### Shield Dropping (244)
The most important technique using PASS:
- **76.3% of platform drops are shield drops**
- Enables fast platform → aerial attack transitions
- 46.3% of drops lead to immediate aerials

### Teeter Setups (245-246)
Limited competitive use:
- **Teeter dtilt**: Cancel teeter with dtilt for stage spike
- **Hurtbox manipulation**: Teeter shifts hurtbox, can dodge some attacks
- **10.3% enter full teeter** - most are cancelled immediately

### Wall Bounce Survival (247)
Understanding wall bounce for recovery:
- **14 frames of invincibility** - same as successful wall tech
- **73% result in death** - usually fatal
- **Fire Fox recovery (8%)**: Primary escape option after bounce

### Missed Sweetspot (251)
Edge guarding opportunity:
- **Only 0.1% invulnerable** - highly punishable
- **52.3% still grab ledge** - not always fatal
- **Shield pressure causes 70.6%** - forcing opponent off edge

---

## State Comparison

| State | Invuln Frames | Grounded | Common Cause |
|-------|---------------|----------|--------------|
| PASS (244) | ~1-2 | No | Shield drop, crouch drop |
| OTTOTTO (245) | ~1-2 | Yes | Walking to edge |
| OTTOTTO_WAIT (246) | ~0 | Yes | Not cancelling teeter |
| FLY_REFLECT_WALL (247) | 14 | No | Missed wall tech |
| FLY_REFLECT_CEIL (248) | ~14 | No | Missed ceiling tech |
| STOP_WALL (249) | ~4-5 | Yes | Dashing into wall |
| STOP_CEIL (250) | ~0-1 | No | Jumping into ceiling |
| MISS_FOOT (251) | 0 | No | Shield pushed, knockback to edge |

---

## Related States

| Relationship | States |
|--------------|--------|
| PASS leads to | Aerials (65-69), landing (42), jumps (27-28), specials |
| OTTOTTO from | Landings (42-43, 70-74), movement (14, 21, 23) |
| FLY_REFLECT from | Damage/tumble states (87-91) |
| FLY_REFLECT leads to | Death (0-2), recovery specials (354), tumble (38) |
| MISS_FOOT from | Shield states (178-181), damage (79-80), getups |
| MISS_FOOT leads to | CLIFF_CATCH (252), landing (42), damage states |

---

## Data Sources

- Parquet replay data: 288,753,124 total frames analyzed
- SmashWiki: Platform mechanics, teeter description
- SmashWiki: Tech mechanics (wall/ceiling)
- Smashboards: Shield drop technique details
