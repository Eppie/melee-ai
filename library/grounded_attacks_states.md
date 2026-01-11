# Grounded Attacks: Jabs & Tilts (States 44-57)

This document describes the grounded attack action states in Super Smash Bros. Melee, covering jabs (states 44-49), dash attack (state 50), forward tilts (states 51-55), up tilt (state 56), and down tilt (state 57).

## Overview

These states represent Fox's non-smash grounded attacks. They are fundamental neutral tools and combo starters/extenders.

| State ID | Name | Attack | Fox Duration | Instances |
|----------|------|--------|--------------|-----------|
| 44 | ATTACK_11 | Jab 1 | 17 frames | 66,197 |
| 45 | ATTACK_12 | Jab 2 | 25 frames | 12,883 |
| 47 | ATTACK_100_START | Rapid Jab Start | 6 frames | 335 |
| 48 | ATTACK_100_LOOP | Rapid Jab Loop | Variable | 314 |
| 49 | ATTACK_100_END | Rapid Jab End | 9 frames | 197 |
| 50 | ATTACK_DASH | Dash Attack | 39 frames | 58,239 |
| 51 | ATTACK_S_3_HI | Forward Tilt (Up) | 26 frames | 1,457 |
| 52 | ATTACK_S_3_HI_S | Forward Tilt (Up-Neutral) | 26 frames | 18 |
| 53 | ATTACK_S_3_S | Forward Tilt (Neutral) | 26 frames | 8,351 |
| 54 | ATTACK_S_3_LW_S | Forward Tilt (Down-Neutral) | 26 frames | 59 |
| 55 | ATTACK_S_3_LW | Forward Tilt (Down) | 26 frames | 4,257 |
| 56 | ATTACK_HI_3 | Up Tilt | 23 frames | 72,498 |
| 57 | ATTACK_LW_3 | Down Tilt | 29 frames | 37,410 |

**Note**: State 46 (ATTACK_13 / Jab 3) exists in the game data but Fox does not have a third jab hit.

---

## Usage Frequency

| State | Total Frames | % of All Frames | Rank |
|-------|-------------|-----------------|------|
| 56 (Up Tilt) | 3,019,483 | **1.05%** | 1st |
| 50 (Dash Attack) | 2,797,829 | 0.97% | 2nd |
| 44 (Jab 1) | 1,293,175 | 0.45% | 3rd |
| 57 (Down Tilt) | 1,242,790 | 0.43% | 4th |
| 45 (Jab 2) | 252,480 | 0.09% | 5th |
| 53 (Ftilt Neutral) | 227,043 | 0.08% | 6th |

**Key Finding**: Up tilt is Fox's most used grounded attack at over 1% of all frames, followed closely by dash attack. These are Fox's primary combo tools.

---

## Jab States (44-49)

### ATTACK_11 (State 44) - Jab 1

**Internal Name**: `ATTACK_11`

**Description**: Fox's first jab - a quick straight punch. One of the fastest grounded moves in the game, used for interrupts, jab resets, and starting jab sequences.

**Frame Data** (Fox):
| Property | Value |
|----------|-------|
| Startup | Frame 2-3 |
| Total Frames | 17 |
| IASA (Interruptible) | Frame 16 |
| Damage | 4% |

**Duration Statistics** (Fox dittos):
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 17 frames |
| Average | 19.54 frames |
| Max | 252 frames |
| Instances | 66,197 |
| Total Frames | 1,293,175 |

**Duration Distribution** (key durations):
| Duration | Occurrences | Note |
|----------|-------------|------|
| 17 frames | 11,869 | Full animation |
| 20 frames | 8,025 | Late cancel |
| 18 frames | 8,757 | |
| 8 frames | 3,542 | Early interrupt |

**Entry Conditions** (top 10):
| From State | Name | Occurrences |
|------------|------|-------------|
| 14 | WAIT | 55,129 |
| 42 | LANDING | 10,794 |
| 43 | LANDING_FALL_SPECIAL | 2,845 |
| 39 | SQUAT | 2,664 |
| 18 | TURN | 2,487 |
| 41 | SQUAT_RV | 1,602 |
| 70 | LANDING_AIR_N | 1,204 |
| 50 | ATTACK_DASH | 965 |
| 72 | LANDING_AIR_B | 671 |

**Exit Conditions** (top 10):
| To State | Name | Occurrences | Percentage |
|----------|------|-------------|------------|
| 14 | WAIT | 25,733 | 38.9% |
| 45 | ATTACK_12 (Jab 2) | 11,848 | **17.9%** |
| 18 | TURN | 11,251 | 17.0% |
| 20 | DASH | 9,813 | 14.8% |
| 39 | SQUAT | 8,660 | 13.1% |
| 15 | WALK_SLOW | 4,449 | 6.7% |
| 24 | KNEE_BEND | 1,786 | 2.7% |
| 237 | REBOUND (clank) | 1,707 | 2.6% |
| 88 | DAMAGE_FLY_N (hit) | 1,342 | 2.0% |

**Key Properties**:
- **On ground**: 100%
- **Invulnerability**: 6.73% (from ledgedash intangibility)
- **Hitstun**: 0%

**Key Finding**: 17.9% of Jab 1s continue into Jab 2, showing the jab sequence is frequently used. However, the majority (38.9%) return to WAIT, indicating single jabs are common for resets and interrupts.

---

### ATTACK_12 (State 45) - Jab 2

**Internal Name**: `ATTACK_12`

**Description**: Fox's second jab - a quick kick following the first jab. Can be followed by rapid jab or used as a combo reset.

**Frame Data** (Fox):
| Property | Value |
|----------|-------|
| Startup | Frames 8-9 |
| Total Frames | 25 |
| IASA (Interruptible) | Frame 24 |
| Damage | 4% |

**Duration Statistics** (Fox dittos):
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 21 frames |
| Average | 19.6 frames |
| Max | 24 frames |
| Instances | 12,883 |
| Total Frames | 252,480 |

**Entry Conditions**:
| From State | Name | Occurrences | Percentage |
|------------|------|-------------|------------|
| 44 | ATTACK_11 (Jab 1) | 11,848 | **92.0%** |
| 14 | WAIT | 1,028 | 8.0% |
| 15 | WALK_SLOW | 4 | <0.1% |

**Exit Conditions** (top 10):
| To State | Name | Occurrences |
|----------|------|-------------|
| 39 | SQUAT | 3,831 |
| 14 | WAIT | 3,203 |
| 18 | TURN | 1,337 |
| 20 | DASH | 1,056 |
| 15 | WALK_SLOW | 765 |
| 88 | DAMAGE_FLY_N | 401 |
| 24 | KNEE_BEND | 396 |
| 47 | ATTACK_100_START (Rapid Jab) | **333** |

**Key Finding**: Only 2.6% of Jab 2s continue into Rapid Jab (333 occurrences). Most players stop at Jab 2 and crouch (29.7%) or return to neutral (24.9%).

---

### Rapid Jab States (47, 48, 49)

**States**: ATTACK_100_START (47), ATTACK_100_LOOP (48), ATTACK_100_END (49)

**Description**: Fox's rapid jab (also called "gentleman" when stopped early) - a flurry of kicks. Rarely used in competitive play due to its commitment and lack of reward.

**Frame Data** (Fox):
| Property | Value |
|----------|-------|
| Active Frames | Frames 20-21, 27-28, 34-35... (repeating) |
| Total Animation | 62 frames |
| Damage per hit | 1% |

**Usage Statistics**:
| State | Total Frames | Instances |
|-------|-------------|-----------|
| 47 (START) | 1,943 | 335 |
| 48 (LOOP) | 14,509 | 314 |
| 49 (END) | 1,585 | 197 |

**Key Finding**: Rapid jab is extremely rare in competitive Fox play - only 335 instances across millions of frames (0.0007% of all frames). The commitment and low reward make it inferior to other options.

---

## Dash Attack (State 50)

### ATTACK_DASH (State 50) - Dash Attack

**Internal Name**: `ATTACK_DASH`

**Description**: Fox's running kick - a forward-moving attack executed from a dash or run. Excellent whiff punish tool and combo starter against airborne opponents.

**Frame Data** (Fox):
| Property | Value |
|----------|-------|
| Startup | Frame 4 |
| Clean Hit | Frames 4-7 |
| Late Hit | Frames 8-17 |
| Total Frames | 39 |
| IASA (Interruptible) | Frame 36 |
| Clean Damage | 7% |
| Late Damage | 5% |
| Knockback Angle | 72° |

**Duration Statistics** (Fox dittos):
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 39 frames |
| Average | 48.04 frames |
| Max | 370 frames |
| Instances | 58,239 |
| Total Frames | 2,797,829 (0.97% of all) |

**Entry Conditions**:
| From State | Name | Occurrences | Percentage |
|------------|------|-------------|------------|
| 20 | DASH | 63,270 | **79.2%** |
| 21 | RUN | 16,550 | **20.7%** |
| 29 | FALL | 19 | <0.1% |
| Other | - | <100 | <0.1% |

**99.9% of dash attacks come from DASH (79.2%) or RUN (20.7%)** - this is the expected behavior since dash attack can only be performed while dashing/running.

**Exit Conditions** (top 15):
| To State | Name | Occurrences | Percentage |
|----------|------|-------------|------------|
| 18 | TURN | 17,031 | 21.3% |
| 14 | WAIT | 10,749 | 13.5% |
| 20 | DASH | 8,067 | 10.1% |
| 178 | GUARD_ON (shield) | 6,224 | 7.8% |
| 15 | WALK_SLOW | 5,720 | 7.2% |
| 88 | DAMAGE_FLY_N (traded) | 4,917 | 6.2% |
| 39 | SQUAT | 4,170 | 5.2% |
| 24 | KNEE_BEND | 3,599 | 4.5% |
| 226 | CAPTURE_PULLED (grabbed) | 1,986 | 2.5% |
| 29 | FALL (off stage) | 1,770 | 2.2% |

**Key Properties**:
- **On ground**: 100%
- **Invulnerability**: 3.72%
- **Hitstun**: 0%

**Key Findings**:
1. **High shield rate (7.8%)**: Players often shield after dash attack whiffs
2. **Trade rate (6.2%)**: DAMAGE_FLY indicates trading with opponent
3. **Punishable**: 2.5% grabbed after dash attack (high endlag is exploitable)

---

## Forward Tilt States (51-55)

### Forward Tilt Angles

Fox's forward tilt has 5 angle variants based on control stick position:

| State | Name | Angle | Frame Usage | Percentage |
|-------|------|-------|-------------|------------|
| 51 | ATTACK_S_3_HI | Up (↗) | 33,481 | 8.93% |
| 52 | ATTACK_S_3_HI_S | Up-Neutral | 429 | 0.11% |
| 53 | ATTACK_S_3_S | Neutral (→) | 227,043 | **60.58%** |
| 54 | ATTACK_S_3_LW_S | Down-Neutral | 1,509 | 0.40% |
| 55 | ATTACK_S_3_LW | Down (↘) | 112,339 | **29.97%** |

**Key Finding**: Neutral angle (60.6%) and down angle (30%) dominate usage. Up angle is rare (8.9%), and the transitional angles (52, 54) are almost never used (<1%).

### ATTACK_S_3_S (State 53) - Forward Tilt (Neutral)

**Internal Name**: `ATTACK_S_3_S`

**Description**: A quick grounded kick. Not commonly used outside of edgeguarding and specific tech chase situations.

**Frame Data** (Fox):
| Property | Value |
|----------|-------|
| Startup | Frames 5-8 |
| Total Frames | 26 |
| Damage | 9% |
| Angle | Sakurai angle |

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 26 frames |
| Average | 27.19 frames |
| Instances | 8,351 |

**Entry Conditions** (top 5):
| From State | Name | Occurrences |
|------------|------|-------------|
| 18 | TURN | 2,681 |
| 15 | WALK_SLOW | 2,584 |
| 16 | WALK_MIDDLE | 2,405 |
| 17 | WALK_FAST | 756 |
| 42 | LANDING | 109 |

**Key Finding**: Forward tilt is primarily entered from walking states - it's used as a spacing tool while moving, not from neutral stance.

---

## Up Tilt (State 56)

### ATTACK_HI_3 (State 56) - Up Tilt

**Internal Name**: `ATTACK_HI_3`

**Description**: A quick upward kick - Fox's most important grounded combo tool. Excellent for juggling, combos from waveshine, and anti-air.

**Frame Data** (Fox):
| Property | Value |
|----------|-------|
| Startup | Frame 5 |
| Active Frames | Frames 5-11 (7 frames) |
| Total Frames | 23 |
| IASA | Frame 23 |
| Ground Sweetspot Damage | 12% |
| Other Hitboxes | 9% |
| Base Knockback | 18 |
| Knockback Scaling | 140 |

**Duration Statistics** (Fox dittos):
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 28 frames |
| Average | 41.65 frames |
| Max | 502 frames |
| Instances | 72,498 |
| Total Frames | 3,019,483 (**1.05% of all frames**) |

**Entry Conditions** (top 15):
| From State | Name | Occurrences | Percentage |
|------------|------|-------------|------------|
| 14 | WAIT | 69,536 | **53.5%** |
| 18 | TURN | 29,842 | 23.0% |
| 42 | LANDING | 14,654 | 11.3% |
| 15 | WALK_SLOW | 5,036 | 3.9% |
| 43 | LANDING_FALL_SPECIAL | 1,617 | 1.2% |
| 74 | LANDING_AIR_LW (dair) | 1,363 | 1.0% |
| 72 | LANDING_AIR_B (bair) | 1,227 | 0.9% |
| 50 | ATTACK_DASH | 1,157 | 0.9% |

**Exit Conditions** (top 15):
| To State | Name | Occurrences | Percentage |
|----------|------|-------------|------------|
| 14 | WAIT | 64,851 | 49.9% |
| 20 | DASH | 13,198 | 10.2% |
| 18 | TURN | 10,333 | 8.0% |
| 15 | WALK_SLOW | 8,869 | 6.8% |
| 88 | DAMAGE_FLY_N (traded) | 3,748 | 2.9% |
| 178 | GUARD_ON | 3,267 | 2.5% |
| 39 | SQUAT | 3,233 | 2.5% |
| 90 | DAMAGE_FLY_LW | 2,682 | 2.1% |
| 226 | CAPTURE_PULLED | 2,334 | 1.8% |
| 79 | DAMAGE_N_2 | 2,292 | 1.8% |
| 24 | KNEE_BEND | 2,279 | 1.8% |
| 82 | DAMAGE_LW_2 | 2,251 | 1.7% |
| 237 | REBOUND (clank) | 2,133 | 1.6% |

**Key Properties**:
- **On ground**: 100%
- **Invulnerability**: 6.90%
- **Hitstun**: 0%
- **Combo Potential**: High - low base knockback allows followups at low percents
- **KO Potential**: 3rd strongest up tilt in Melee, can KO at ~120%

**Key Findings**:
1. **Most used grounded attack** (1.05% of all frames)
2. **53.5% entered from WAIT** - used as a neutral option
3. **11.3% entered from LANDING** - combo followup after aerial
4. **Landing lag entries (dair 1.0%, bair 0.9%)** - confirms waveshine/aerial → uptilt combos

---

## Down Tilt (State 57)

### ATTACK_LW_3 (State 57) - Down Tilt

**Internal Name**: `ATTACK_LW_3`

**Description**: A quick tail sweep while crouching. Used for pokes, combo extensions, and jab resets on lightweight opponents.

**Frame Data** (Fox):
| Property | Value |
|----------|-------|
| Startup | Frame 7 |
| Active Frames | Frames 7-9 (3 frames) |
| Total Frames | 29 |
| IASA | Frame 28 |
| Damage | 10% |
| Base Knockback | 25 |
| Knockback Scaling | 125 |
| Angle | 70°/80°/90° (varies by hitbox) |

**Duration Statistics** (Fox dittos):
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 32 frames |
| Average | 33.22 frames |
| Max | 223 frames |
| Instances | 37,410 |
| Total Frames | 1,242,790 (0.43% of all frames) |

**Duration Distribution**:
| Duration | Occurrences | Note |
|----------|-------------|------|
| 29 frames | 7,964 | IASA cancel |
| 34 frames | 8,530 | Full animation |
| 32 frames | 2,424 | |

**Entry Conditions** (top 10):
| From State | Name | Occurrences | Percentage |
|------------|------|-------------|------------|
| 39 | SQUAT | 30,183 | **68.2%** |
| 40 | SQUAT_WAIT | 10,555 | **23.9%** |
| 43 | LANDING_FALL_SPECIAL | 1,094 | 2.5% |
| 42 | LANDING | 869 | 2.0% |
| 70 | LANDING_AIR_N | 405 | 0.9% |
| 44 | ATTACK_11 (Jab 1) | 299 | 0.7% |

**Exit Conditions** (top 15):
| To State | Name | Occurrences | Percentage |
|----------|------|-------------|------------|
| 41 | SQUAT_RV (stand from crouch) | 15,195 | **40.6%** |
| 20 | DASH | 7,280 | 19.5% |
| 24 | KNEE_BEND | 3,520 | 9.4% |
| 40 | SQUAT_WAIT | 3,246 | 8.7% |
| 178 | GUARD_ON | 2,564 | 6.9% |
| 15 | WALK_SLOW | 2,185 | 5.8% |
| 88 | DAMAGE_FLY_N | 1,669 | 4.5% |
| 76 | DAMAGE_N_1 | 1,609 | 4.3% |

**Key Properties**:
- **On ground**: 100%
- **Invulnerability**: 1.13%
- **Hitstun**: 0%

**Key Findings**:
1. **92.1% entered from crouch states** (SQUAT 68.2% + SQUAT_WAIT 23.9%)
2. **Crouch → Down Tilt is the standard pattern** for using this move
3. **40.6% exit to SQUAT_RV** - returning to standing after dtilt
4. **4.3% trade rate** (DAMAGE states) - similar to other grounded attacks

---

## Comparative Analysis

### Attack Frequency Ranking

| Rank | Attack | Total Frames | % of All |
|------|--------|-------------|----------|
| 1 | Up Tilt (56) | 3,019,483 | 1.05% |
| 2 | Dash Attack (50) | 2,797,829 | 0.97% |
| 3 | Jab 1 (44) | 1,293,175 | 0.45% |
| 4 | Down Tilt (57) | 1,242,790 | 0.43% |
| 5 | Jab 2 (45) | 252,480 | 0.09% |
| 6 | Forward Tilt (all) | 374,801 | 0.13% |
| 7 | Rapid Jab (all) | 18,037 | 0.006% |

### Entry Pattern Comparison

| Attack | Primary Entry | % | Secondary Entry | % |
|--------|---------------|---|-----------------|---|
| Jab 1 | WAIT | 83% | LANDING | 16% |
| Dash Attack | DASH/RUN | 99.9% | - | - |
| Up Tilt | WAIT | 54% | TURN | 23% |
| Down Tilt | SQUAT | 68% | SQUAT_WAIT | 24% |
| Forward Tilt | Walk states | 70% | TURN | 25% |

### Key Technique Patterns

**Jab Sequence Usage**:
- Jab 1 only: 82.1%
- Jab 1 → Jab 2: 17.9%
- Jab 2 → Rapid Jab: 2.6%

**Down Tilt Crouch Requirement**:
The 92.1% entry rate from crouch states confirms that down tilt requires crouching first - this is a fundamental game mechanic.

---

## State Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GROUNDED NEUTRAL STATES                          │
│                (WAIT, WALK, CROUCH, LANDING)                        │
└───────────────┬───────────────────────────────────────┬─────────────┘
                │                                       │
    ┌───────────┴───────────┐               ┌──────────┴──────────┐
    │                       │               │                     │
    ▼                       ▼               ▼                     ▼
┌─────────┐          ┌──────────┐     ┌──────────┐         ┌───────────┐
│ WAIT/   │          │ DASH/RUN │     │  SQUAT   │         │   WALK    │
│ LANDING │          │ (20/21)  │     │  (39/40) │         │ (15-17)   │
└────┬────┘          └────┬─────┘     └────┬─────┘         └─────┬─────┘
     │                    │                │                     │
     ▼                    ▼                ▼                     ▼
┌─────────┐       ┌────────────┐    ┌───────────┐         ┌───────────┐
│ JAB 1   │       │DASH ATTACK │    │ DOWN TILT │         │FORWARD    │
│  (44)   │       │   (50)     │    │   (57)    │         │ TILT      │
│ 17 frm  │       │  39 frm    │    │  29 frm   │         │ (51-55)   │
└────┬────┘       └────────────┘    └───────────┘         │  26 frm   │
     │                                                     └───────────┘
     ▼
┌─────────┐         ┌─────────┐
│ JAB 2   │ ──────► │ RAPID   │
│  (45)   │  2.6%   │  JAB    │
│ 25 frm  │         │ (47-49) │
└─────────┘         └─────────┘


                    ┌─────────────────────────────────┐
                    │   UP TILT (56) - Most Used      │
                    │   Entry: 54% WAIT, 23% TURN     │
                    │   Frame 5 startup, 23 total     │
                    │   Primary combo/juggle tool     │
                    └─────────────────────────────────┘
```

---

## Advanced Techniques

### Jab Reset
Using Jab 1 (state 44) on a knocked-down opponent to force them into a getup animation. The 17.9% Jab 1 → Jab 2 rate shows players sometimes double jab for resets.

### Waveshine → Up Tilt
Shine (360) → wavedash (43) → Up Tilt (56) is a core Fox combo. The 1.2% entry rate from LANDING_FALL_SPECIAL into uptilt reflects this pattern.

### Crouch Cancel → Down Tilt
From crouch (absorb hit via crouch cancel) → down tilt for punish. The 92% crouch entry rate for dtilt shows this is the standard pattern.

### Running Shine → Dash Attack
Dash attack is used as a followup after running shine when the opponent is launched. 79.2% DASH entry + 20.7% RUN entry confirms exclusive dash/run execution.

---

## Character-Specific Notes (Fox)

Fox's grounded attack attributes:
- **Jab Startup**: Frame 2 (tied for fastest in game)
- **Up Tilt Startup**: Frame 5 (very fast)
- **Dash Attack Range**: Long, with forward movement
- **Down Tilt Profile**: Low-hitting, good for catching landings

Fox's up tilt is notable for:
- 3rd strongest up tilt in Melee
- Low base knockback enables combos at low %
- High knockback scaling enables KOs at high %
- Frame 5 startup makes it excellent out of waveshine

---

## Sources

### Primary Sources (Quantitative)
- **Parquet replay data**: `/Users/eppie/ssb_wiki/fox_vs_fox_parquet/` - 9+ million frames of grounded attack data analyzed

### Documentation Sources (Qualitative)
- [SmashWiki - Fox Neutral Attack](https://www.ssbwiki.com/Fox_(SSBM)/Neutral_attack)
- [SmashWiki - Fox Up Tilt](https://www.ssbwiki.com/Fox_(SSBM)/Up_tilt)
- [SmashWiki - Fox Down Tilt](https://www.ssbwiki.com/Fox_(SSBM)/Down_tilt)
- [SmashWiki - Fox Dash Attack](https://www.ssbwiki.com/Fox_(SSBM)/Dash_attack)
- [SmashWiki - Fox Forward Tilt](https://www.ssbwiki.com/Fox_(SSBM)/Forward_tilt)
- [libmelee Action State Enums](https://libmelee.readthedocs.io/en/latest/enums.html)

---

## Investigation Methodology

1. **Parquet Replay Data (Primary)**: Provided quantitative data on attack frequency, durations, entry/exit patterns, and usage rates.

2. **SmashWiki (High Value)**: Provided frame data (startup, IASA, damage, knockback) for each attack.

3. **action_state.json (Reference)**: Mapped state IDs to internal names.

**Key Findings**:
- Up tilt is Fox's most used grounded attack (1.05% of all frames)
- Down tilt requires crouch entry (92.1% from crouch states)
- Dash attack is exclusively from dash/run states (99.9%)
- Forward tilt neutral angle dominates (60.6%)
- Rapid jab is almost never used (<0.01%)
