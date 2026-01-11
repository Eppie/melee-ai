# Ledge States & Taunt Documentation

Action states 252-264 covering all ledge mechanics (grab, hang, getup options) and taunt.

## Overview

Ledge mechanics in Melee provide critical recovery and neutral reset options. Upon grabbing the ledge, characters receive **37 frames of total invincibility** (7 frames during grab + 30 frames while hanging). Ledge getup options are divided into SLOW (≥100% damage) and QUICK (<100% damage) variants with significant frame data differences.

| State | ID | Internal Name | Instances | % of Gameplay | Median Duration |
|-------|-----|---------------|-----------|---------------|-----------------|
| Ledge Catch | 252 | CLIFF_CATCH | 306,510 | 0.74% | 7 frames |
| Ledge Hang | 253 | CLIFF_WAIT | 304,308 | 0.80% | 4 frames |
| Getup Slow | 254 | CLIFF_CLIMB_SLOW | 1,700 | 0.03% | 59 frames |
| Getup Quick | 255 | CLIFF_CLIMB_QUICK | 15,776 | 0.18% | 34 frames |
| Attack Slow | 256 | CLIFF_ATTACK_SLOW | 737 | 0.02% | 69 frames |
| Attack Quick | 257 | CLIFF_ATTACK_QUICK | 5,570 | 0.10% | 54 frames |
| Roll Slow | 258 | CLIFF_ESCAPE_SLOW | 886 | 0.02% | 79 frames |
| Roll Quick | 259 | CLIFF_ESCAPE_QUICK | 7,840 | 0.13% | 49 frames |
| Jump Slow P1 | 260 | CLIFF_JUMP_SLOW_1 | 3,541 | 0.02% | 19 frames |
| Jump Slow P2 | 261 | CLIFF_JUMP_SLOW_2 | 3,537 | 0.03% | 30 frames |
| Jump Quick P1 | 262 | CLIFF_JUMP_QUICK_1 | 17,298 | 0.08% | 14 frames |
| Jump Quick P2 | 263 | CLIFF_JUMP_QUICK_2 | 17,280 | 0.17% | 30 frames |
| Taunt | 264 | APPEAL_R | 998 | 0.03% | 110 frames |

**Total ledge frames**: 4,573,433 (1.58% of gameplay)

---

## Ledge Invincibility Summary

The 37-frame ledge invincibility is one of Melee's most important defensive mechanics.

| State | Invuln % | Duration | Invuln Frames | Notes |
|-------|----------|----------|---------------|-------|
| CLIFF_CATCH (252) | 100% | 7 frames | 1-7 | Fully intangible during grab animation |
| CLIFF_WAIT (253) | 93.5% | Variable | 1-30 | Invuln ends at frame 31 |
| CLIFF_CLIMB_SLOW (254) | 95.2% | 59 frames | 1-56 | Near-full invincibility |
| CLIFF_CLIMB_QUICK (255) | 88.7% | 34 frames | 1-30 | Faster, less coverage |
| CLIFF_ATTACK_SLOW (256) | 82.6% | 69 frames | 1-57 | Hitbox during vulnerable period |
| CLIFF_ATTACK_QUICK (257) | 48.0% | 54 frames | 1-26 | Short invuln, fast attack |
| CLIFF_ESCAPE_SLOW (258) | 81.4% | 79 frames | 1-64 | Long roll, mostly invuln |
| CLIFF_ESCAPE_QUICK (259) | 70.3% | 49 frames | 1-34 | Shorter roll |
| CLIFF_JUMP_SLOW_1 (260) | 100% | 19 frames | 1-19 | Phase 1 fully invuln |
| CLIFF_JUMP_SLOW_2 (261) | 27.7% | 30 frames | 1-8 | Phase 2 mostly vulnerable |
| CLIFF_JUMP_QUICK_1 (262) | 100% | 14 frames | 1-14 | Phase 1 fully invuln |
| CLIFF_JUMP_QUICK_2 (263) | 42.7% | 30 frames | 1-13 | Phase 2 partial invuln |

---

## State 252: CLIFF_CATCH (Ledge Grab)

### Description
CLIFF_CATCH is the initial ledge grab animation. The character's ECB (environment collision box) magnetizes to the ledge and they snap into the hanging position. This state provides **complete invincibility** throughout its duration.

### Frame Data
| Metric | Value |
|--------|-------|
| Duration | 7 frames (fixed for Fox) |
| Invulnerability | Frames 1-7 (100%) |
| Actionable | Frame 8 (transitions to CLIFF_WAIT) |

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| FALL (29) | 103,195 | 33.7% |
| JUMP_AERIAL_F (27) | 48,173 | 15.7% |
| FIRE_FOX_AIR_STARTUP (354) | 36,190 | 11.8% |
| JUMP_F (25) | 31,949 | 10.4% |
| FOX_ILLUSION_AIR_END (352) | 22,853 | 7.5% |
| JUMP_B (26) | 17,035 | 5.6% |
| FIRE_FOX_AIR_MAIN (356) | 14,347 | 4.7% |
| MISS_FOOT (251) | 6,688 | 2.2% |
| JUMP_AERIAL_B (28) | 6,375 | 2.1% |
| FIRE_FOX_AIR_END (358) | 5,611 | 1.8% |

**Key patterns**:
- 67.5% from aerial movement states (falls, jumps)
- 25.8% from recovery specials (Fire Fox, Illusion)
- 2.2% from missed sweetspot (251) - regrabbing after slipping

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| CLIFF_WAIT (253) | ~100% | Always |

CLIFF_CATCH always transitions to CLIFF_WAIT.

---

## State 253: CLIFF_WAIT (Ledge Hang)

### Description
CLIFF_WAIT is the idle hanging state where the character waits on the ledge. Invincibility continues from CLIFF_CATCH for an additional **30 frames**, then expires. Players can remain hanging for up to 11 seconds (<100%) or 8 seconds (≥100%) before automatically dropping.

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 4 frames |
| Average Duration | 7.6 frames |
| Maximum Duration | 383 frames |
| Invulnerability | Frames 1-30 |
| Vulnerable | Frame 31+ |

### Invulnerability by Frame
| Frame | Invuln % | Notes |
|-------|----------|-------|
| 1-30 | 100% | Full invincibility window |
| 31+ | ~1% | Vulnerability begins |
| 38+ | 16-80% | Getup options begin, restore partial invuln |

### Exit Conditions (Ledge Options)
| Target State | Occurrences | % of Exits | Option |
|--------------|-------------|------------|--------|
| FALL (29) | 249,855 | 82.1% | Drop/Ledgedash |
| CLIFF_JUMP_QUICK_1 (262) | 16,236 | 5.3% | Ledge jump |
| CLIFF_CLIMB_QUICK (255) | 15,776 | 5.2% | Normal getup |
| CLIFF_ESCAPE_QUICK (259) | 7,766 | 2.5% | Ledge roll |
| CLIFF_ATTACK_QUICK (257) | 5,253 | 1.7% | Ledge attack |
| CLIFF_JUMP_SLOW_1 (260) | 3,363 | 1.1% | Ledge jump (slow) |
| CLIFF_CLIMB_SLOW (254) | 1,700 | 0.6% | Normal getup (slow) |
| CLIFF_ESCAPE_SLOW (258) | 866 | 0.3% | Ledge roll (slow) |
| WAIT (14) | 836 | 0.3% | Direct stage transition |
| CLIFF_ATTACK_SLOW (256) | 673 | 0.2% | Ledge attack (slow) |

**Critical insight**: 82.1% of ledge situations result in a **drop** rather than a direct getup option. This is because competitive players prefer to:
1. **Ledgedash**: Drop, double jump, wavedash onto stage with invincibility
2. **Refresh invincibility**: Drop and regrab to reset the 37-frame timer
3. **Aerial attack**: Drop, double jump, attack from below

---

## Ledge Getup Options

### SLOW vs QUICK Threshold

The damage threshold for SLOW vs QUICK animations is exactly **100%**:

| Option Type | Damage Range | Median % | Characteristics |
|-------------|--------------|----------|-----------------|
| SLOW | ≥100% | 112% | Longer, clunkier, more invuln |
| QUICK | <100% | 31-51% | Faster, more actionable |

This creates strategic depth: high-percent players have worse options but more invincibility, while low-percent players can act faster but are more vulnerable.

---

## State 254/255: CLIFF_CLIMB (Normal Getup)

### Description
Normal getup (pressing toward stage without attack/jump) climbs from ledge to standing position. The SLOW version is significantly longer and provides more invincibility coverage.

### Frame Data Comparison
| Metric | SLOW (254) | QUICK (255) |
|--------|------------|-------------|
| Duration | 59 frames | 34 frames |
| Invuln % | 95.2% | 88.7% |
| Invuln Frames | 1-56 | 1-30 |
| Instances | 1,700 | 15,776 |
| Median Damage | 112% | 34% |

### Exit Conditions (CLIFF_CLIMB_SLOW - 254)
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| GUARD_ON (178) | 465 | 27.4% |
| WAIT (14) | 348 | 20.5% |
| DAMAGE_FLY_TOP (90) | 210 | 12.4% |
| DAMAGE_FLY_N (88) | 125 | 7.4% |
| DAMAGE_FLY_HI (87) | 101 | 5.9% |

**Key insight**: Players often shield immediately after getup (27.4%), indicating defensive recovery situations.

---

## State 256/257: CLIFF_ATTACK (Ledge Attack)

### Description
Ledge attack (pressing A while hanging) performs an attack while climbing. The hitbox comes out during the rising animation. QUICK version has significantly less invincibility but faster startup.

### Frame Data Comparison
| Metric | SLOW (256) | QUICK (257) |
|--------|------------|-------------|
| Duration | 69 frames | 54 frames |
| Invuln % | 82.6% | 48.0% |
| Invuln Frames | 1-57 | 1-26 |
| Instances | 737 | 5,570 |
| Median Damage | 112% | 44% |

### Exit Conditions (CLIFF_ATTACK_SLOW - 256)
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| GUARD_ON (178) | 231 | 31.3% |
| WAIT (14) | 124 | 16.8% |
| DAMAGE_FLY_TOP (90) | 102 | 13.8% |
| DAMAGE_FLY_N (88) | 101 | 13.7% |

**Key insight**: Ledge attack is the least common option overall. The attack's long endlag makes it highly punishable despite its invincibility.

---

## State 258/259: CLIFF_ESCAPE (Ledge Roll)

### Description
Ledge roll (pressing away from stage or L/R while hanging) rolls onto the stage, traveling a significant distance. This is useful for escaping immediate ledge pressure but is highly punishable due to its long duration.

### Frame Data Comparison
| Metric | SLOW (258) | QUICK (259) |
|--------|------------|-------------|
| Duration | 79 frames | 49 frames |
| Invuln % | 81.4% | 70.3% |
| Invuln Frames | 1-64 | 1-34 |
| Instances | 886 | 7,840 |
| Median Damage | 112% | 31% |

### Exit Conditions (CLIFF_ESCAPE_QUICK - 259)
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| WAIT (14) | 4,973 | 63.4% |
| DASH (20) | 759 | 9.7% |
| WALK_SLOW (15) | 534 | 6.8% |
| SQUAT (39) | 493 | 6.3% |
| GUARD_ON (178) | 254 | 3.2% |

**Key insight**: Unlike other getup options, ledge roll primarily ends in WAIT (standing) rather than immediate shield. Players can immediately act with movement after roll completes.

---

## State 260-263: CLIFF_JUMP (Ledge Jump)

### Description
Ledge jump (pressing jump while hanging) consists of **two phases**:
1. **Phase 1** (260/262): Fully invincible rising portion
2. **Phase 2** (261/263): Mostly vulnerable aerial portion

This two-phase structure is unique to ledge jump and is crucial for advanced techniques like ledgedashing.

### Frame Data (Two-Phase Structure)
| Phase | SLOW (260→261) | QUICK (262→263) |
|-------|----------------|-----------------|
| Phase 1 Duration | 19 frames | 14 frames |
| Phase 1 Invuln | 100% | 100% |
| Phase 2 Duration | 30 frames | 30 frames |
| Phase 2 Invuln | 27.7% | 42.7% |
| Total Duration | 49 frames | 44 frames |

### Phase 1 Transitions
| From | To | Occurrences | % |
|------|-----|-------------|---|
| 260 (SLOW_1) | 261 (SLOW_2) | 3,537 | 99.9% |
| 262 (QUICK_1) | 263 (QUICK_2) | 17,280 | 99.9% |

Phase 1 always transitions to Phase 2 (unless interrupted by damage).

### Phase 2 Exit Conditions (CLIFF_JUMP_QUICK_2 - 263)
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| LANDING (42) | 11,140 | 64.4% |
| DAMAGE_FLY_N (88) | 1,883 | 10.9% |
| DAMAGE_FLY_LW (89) | 852 | 4.9% |
| DAMAGE_AIR_2 (85) | 707 | 4.1% |
| DAMAGE_AIR_3 (86) | 664 | 3.8% |
| FALL (29) | 570 | 3.3% |
| CLIFF_CATCH (252) | 340 | 2.0% |

**Key patterns**:
- 64.4% land on stage (successful ledge jump)
- 23.7% get hit during Phase 2 (vulnerable period)
- 3.3% fall back down (ledge jump miscalculated)
- 2.0% regrab ledge

---

## Ledgedash: Advanced Technique

### Mechanics
Ledgedashing is the optimal ledge option in competitive play, providing **grounded actionable invincibility**:

1. Drop from ledge on frame 9 (earliest possible)
2. Immediately double jump
3. Airdodge diagonally into stage
4. Land with remaining invincibility frames

### Fox Ledgedash Data
| Metric | Value |
|--------|-------|
| Maximum GALINT | 15 frames |
| Jump-canceled Shine (Yoshi's) | 22 semi-actionable frames |
| Jump-canceled Shine (Battlefield) | 20 semi-actionable frames |
| Jump-canceled Shine (other stages) | 18 semi-actionable frames |

**GALINT** = Grounded Actionable Ledge Intangibility

Fox (tied with Pichu) has the best ledgedash in the game, enabling fully invincible grabs, up smashes, shines, and aerials.

### Why Drop Rate is 82%
The data shows 82.1% of ledge situations result in dropping (FALL). This reflects:
1. **Ledgedash prevalence**: The primary competitive option
2. **Invincibility refresh**: Drop and regrab to reset the 37-frame window
3. **Aerial options**: Drop, double jump, aerial attack from below
4. **Ledge hog**: Drop and regrab to deny opponent's recovery

---

## Ledge Grab Sources

Understanding how players reach the ledge:

| Source Category | Occurrences | % |
|-----------------|-------------|---|
| Fall (29) | 103,195 | 33.7% |
| Double Jump (27, 28) | 54,548 | 17.8% |
| Fire Fox (354, 356, 358) | 56,148 | 18.3% |
| Fox Illusion (350, 352) | 26,927 | 8.8% |
| First Jump (25, 26) | 48,984 | 16.0% |
| Miss Foot (251) | 6,688 | 2.2% |
| Special Fall (35, 38, 32) | 6,458 | 2.1% |

**Key insights**:
- 33.7% from falling (ledge drops, edge guards)
- 33.8% from jumps (recovery)
- 27.1% from recovery specials (Fire Fox, Illusion)
- 2.2% from missed sweetspot (slipped ledge, regrabbed)

---

## State 264: APPEAL_R (Taunt)

### Description
Taunt is Fox's signature "C'mon!" animation, initiated by pressing the D-pad. It has no gameplay utility beyond disrespecting the opponent. The animation is very long (110 frames) and provides virtually no invincibility.

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 110 frames |
| Average Duration | 95.1 frames |
| Maximum Duration | 860 frames |
| Invulnerability | 1.9% |
| Total Instances | 998 |

### Properties
| Property | Value |
|----------|-------|
| On Ground | 100% |
| Average Damage % | 43.6% |
| Average Position | x=4.1, y=4.8 (center stage) |

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| WAIT (14) | 722 | 64.4% |
| LANDING (42) | 297 | 26.5% |
| ATTACK_HI_4 (63) | 20 | 1.8% |
| DASH (20) | 19 | 1.7% |
| TURN (18) | 18 | 1.6% |

**Key insight**: 90.9% of taunts come from neutral positions (standing, landing), typically after taking a stock or winning an exchange.

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| WAIT (14) | 490 | 56.4% |
| WALK_SLOW (15) | 55 | 6.3% |
| FALL (29) | 51 | 5.9% |
| TURN (18) | 42 | 4.8% |
| SQUAT (39) | 41 | 4.7% |
| GUARD_ON (178) | 37 | 4.3% |

**Key patterns**:
- 56.4% return to standing (taunt completed)
- 5.9% fall off stage during taunt (disrespect gone wrong)
- 2.6% get hit (DAMAGE states) - punished for taunting

---

## Competitive Applications

### Ledge Pressure
When edgeguarding, opponents have limited options:
- **Getup**: React and punish the 34-59 frame animation
- **Attack**: Shield or space outside hitbox range
- **Roll**: Predict and cover landing position
- **Jump**: Cover aerial space with uair/bair

### Ledgedash Counterplay
Ledgedash is powerful but risky:
- **Mistiming**: Results in airdodge off stage (SD)
- **Prediction**: If opponent expects ledgedash, they can position to punish
- **Stage-dependent**: GALINT frames vary by platform height

### Ledge Stalling
Players can refresh invincibility indefinitely by:
1. Drop from ledge
2. Double jump
3. Regrab ledge
This resets the 37-frame invincibility window.

### Tournament Ledge Grab Limits
Many tournament rulesets limit consecutive ledge grabs (typically 6-8) to prevent excessive stalling.

---

## Quick/Slow Usage Patterns

| Option | SLOW Usage | QUICK Usage | Ratio |
|--------|------------|-------------|-------|
| Getup | 1,700 (9.7%) | 15,776 (90.3%) | 1:9.3 |
| Attack | 737 (11.7%) | 5,570 (88.3%) | 1:7.6 |
| Roll | 886 (10.2%) | 7,840 (89.8%) | 1:8.8 |
| Jump | 3,541 (17.0%) | 17,298 (83.0%) | 1:4.9 |

**Interpretation**: QUICK options are used 5-9x more often than SLOW options. This reflects:
1. Fox dittos rarely reach high percentages before kills
2. Players at ≥100% are often in kill situations, not ledge recovery
3. QUICK options are inherently more useful (faster, more actionable)

---

## Related States

| Relationship | States |
|--------------|--------|
| Entry from | FALL (29), jumps (25-28), specials (350-358), MISS_FOOT (251) |
| Leads to | All getup options (254-263), FALL (29) for ledgedash |
| Pairs with | LANDING_FALL_SPECIAL (43) for ledgedash, aerials for ledge hop |

---

## Data Sources

- Parquet replay data: 288,753,124 total frames analyzed
- SmashWiki: Ledge mechanics, invincibility timing
- SmashWiki: Ledgedash, GALINT calculation
- Smashboards: 37-frame invincibility confirmation
