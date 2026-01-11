# Grounded Attacks: Smashes (States 60, 63, 64)

This document describes the smash attack action states in Super Smash Bros. Melee, covering forward smash (state 60), up smash (state 63), and down smash (state 64).

## Overview

Smash attacks are Fox's most powerful grounded attacks, used primarily as kill moves and combo finishers. Each can be charged by holding the input, increasing damage and knockback.

| State ID | Name | Attack | Fox Total Frames | Instances |
|----------|------|--------|------------------|-----------|
| 60 | ATTACK_S_4_S | Forward Smash | 39 frames | 15,001 |
| 63 | ATTACK_HI_4 | Up Smash | 41 frames | 40,701 |
| 64 | ATTACK_LW_4 | Down Smash | 49 frames | 9,787 |

**Note**: States 58, 59, 61, 62 are angled forward smash variants (up/down angles), but Fox only uses the neutral angle (state 60) in practice.

---

## Usage Frequency

| State | Total Frames | % of All Frames | Rank Among Smashes |
|-------|-------------|-----------------|-------------------|
| 63 (Up Smash) | 4,780,393 | **1.66%** | 1st |
| 60 (Forward Smash) | 652,909 | 0.23% | 2nd |
| 64 (Down Smash) | 449,780 | 0.16% | 3rd |

**Key Finding**: Up smash accounts for **81.3%** of all smash attack frames. It's Fox's most important kill move and one of the most used moves in his entire toolkit.

---

## Up Smash (State 63) - ATTACK_HI_4

**Internal Name**: `ATTACK_HI_4`

**Description**: Fox's signature kill move - a powerful overhead flip kick. Considered **the best up smash in the game** due to its speed, power, and combo potential. Often performed as a jump-canceled up smash (JC upsmash) from a dash.

### Frame Data (Fox)

| Property | NTSC Value | PAL Value |
|----------|------------|-----------|
| Startup | Frame 7 | Frame 7 |
| Clean Hit | Frames 7-9 | Frames 7-9 |
| Late Hit | Frames 10-17 | Frames 10-17 |
| Total Animation | 41 frames | 41 frames |
| IASA | Frame 49 | Frame 49 |
| Clean Damage | 18% | 17% |
| Late Damage | 13% | 13% |
| Base Knockback (clean) | 112 | 108 |
| Knockback Scaling (clean) | 30 | 26 |
| **Head Intangibility** | Frames 1-9 | Frames 1-9 |

### Duration Statistics (Fox dittos)

| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 97 frames |
| Average | 117.45 frames |
| Max | 806 frames |
| Instances | 40,701 |
| Total Frames | 4,780,393 (**1.66% of all frames**) |

The high median (97 frames) and average (117 frames) compared to the 41-frame animation indicates significant charging is common.

### Entry Conditions

| From State | Name | Occurrences | Percentage |
|------------|------|-------------|------------|
| 24 | KNEE_BEND (jumpsquat) | 83,623 | **74.26%** |
| 14 | WAIT | 11,900 | 10.57% |
| 18 | TURN | 4,803 | 4.27% |
| 39 | SQUAT | 2,435 | 2.16% |
| 16 | WALK_MIDDLE | 2,083 | 1.85% |
| 15 | WALK_SLOW | 1,526 | 1.36% |
| 42 | LANDING | 1,492 | 1.33% |
| 40 | SQUAT_WAIT | 1,301 | 1.16% |
| 50 | ATTACK_DASH | 818 | 0.73% |

**CRITICAL FINDING: 74.26% of up smashes come from KNEE_BEND (jumpsquat)** - this is Jump Cancel Up Smash (JC upsmash), the technique of canceling a dash into jumpsquat and immediately performing up smash. This allows Fox to use up smash out of a run, making it vastly more threatening.

### Exit Conditions

| To State | Name | Occurrences | Percentage |
|----------|------|-------------|------------|
| 14 | WAIT | 48,794 | **43.35%** |
| 178 | GUARD_ON (shield) | 10,290 | 9.14% |
| 18 | TURN | 8,181 | 7.27% |
| 20 | DASH | 8,044 | 7.15% |
| 88 | DAMAGE_FLY_N (traded) | 6,629 | 5.89% |
| 15 | WALK_SLOW | 6,100 | 5.42% |
| 39 | SQUAT | 3,855 | 3.42% |
| 226 | CAPTURE_PULLED (grabbed) | 3,468 | 3.08% |
| 79 | DAMAGE_N_2 | 2,358 | 2.09% |
| 90 | DAMAGE_FLY_LW | 2,157 | 1.92% |

### Key Properties

- **On ground**: 100%
- **Invulnerability**: 1.86%
- **Head Intangibility**: Frames 1-9 (not captured in parquet, but documented)
- **Hitstun**: 0%
- **KO Range**: 75-90% on most characters

### Competitive Uses

1. **JC Upsmash from Dash**: The primary usage (74.26% from KNEE_BEND)
2. **Waveshine → Upsmash**: Combo finisher from shine
3. **Upthrow → Upsmash**: At kill percents on fastfallers
4. **Dash Attack → Upsmash**: On floaty characters
5. **Anti-Air**: Head intangibility makes it safe against aerial approaches

---

## Forward Smash (State 60) - ATTACK_S_4_S

**Internal Name**: `ATTACK_S_4_S`

**Description**: A powerful forward kick. Less commonly used than up smash due to slower speed and weaker knockback, but useful for edgeguarding and reads.

### Frame Data (Fox)

| Property | Value |
|----------|-------|
| Startup | Frame 12 |
| Clean Hit | Frames 12-16 |
| Late Hit | Frames 17-22 |
| Total Animation | 39 frames |
| Clean Damage | 15% |
| Late Damage | 12% |
| Knockback Angle | Sakurai angle |
| Base KB (clean) | 10 |
| KB Scaling (clean) | 105 |

### Duration Statistics (Fox dittos)

| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 39 frames |
| Average | 43.52 frames |
| Max | 327 frames |
| Instances | 15,001 |
| Total Frames | 652,909 (0.23% of all frames) |

The median of 39 frames matches the total animation, indicating most forward smashes are not charged.

### Duration Distribution (key durations)

| Duration | Occurrences | Note |
|----------|-------------|------|
| 46 frames | 1,585 | Most common (slight charge) |
| 45 frames | 485 | |
| 39 frames | ~100 | Full uncharged animation |

### Entry Conditions

| From State | Name | Occurrences | Percentage |
|------------|------|-------------|------------|
| 14 | WAIT | 3,571 | 20.13% |
| 20 | DASH | 3,275 | 18.46% |
| 39 | SQUAT | 2,142 | 12.07% |
| 18 | TURN | 1,946 | 10.97% |
| 42 | LANDING | 1,888 | 10.64% |
| 40 | SQUAT_WAIT | 944 | 5.32% |
| 15 | WALK_SLOW | 942 | 5.31% |
| 16 | WALK_MIDDLE | 689 | 3.88% |

Forward smash has a diverse entry profile - used from neutral (20%), dash (18%), and crouch (12%).

### Exit Conditions

| To State | Name | Occurrences | Percentage |
|----------|------|-------------|------------|
| 14 | WAIT | 6,580 | 43.86% |
| 178 | GUARD_ON | 2,257 | 15.05% |
| 18 | TURN | 1,961 | 13.07% |
| 88 | DAMAGE_FLY_N (traded) | 1,193 | 7.95% |
| 90 | DAMAGE_FLY_LW | 957 | 6.38% |
| 226 | CAPTURE_PULLED (grabbed) | 603 | 4.02% |

### Key Properties

- **On ground**: 100%
- **Invulnerability**: 2.51%
- **Hitstun**: 0%
- **Trade Rate**: 14.33% (DAMAGE_FLY states) - high commitment move

### Competitive Uses

1. **Edgeguarding**: Extended active frames catch recoveries
2. **Hard Reads**: Used on predicted landings/rolls
3. **Tech Chase Punish**: Against missed techs near the ledge
4. **Combo Finisher**: Less common than upsmash, but used at certain angles

---

## Down Smash (State 64) - ATTACK_LW_4

**Internal Name**: `ATTACK_LW_4`

**Description**: A low spinning kick that hits on both sides. Fox's fastest smash attack with a semi-spike angle, excellent for edgeguarding.

### Frame Data (Fox)

| Property | NTSC Value | PAL Value |
|----------|------------|-----------|
| Startup | Frame 6 | Frame 6 |
| Active Frames | Frames 6-10 | Frames 6-10 |
| Total Animation | 49 frames | 49 frames |
| IASA | Frame 46 | Frame 46 |
| Primary Damage | 15% | 13% |
| Late Damage | 12% | 12% |
| Knockback Angle | 25° | 30° |
| Base Knockback | 20 | 20 |

### Duration Statistics (Fox dittos)

| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 48 frames |
| Average | 45.96 frames |
| Max | 255 frames |
| Instances | 9,787 |
| Total Frames | 449,780 (0.16% of all frames) |

### Duration Distribution (key durations)

| Duration | Occurrences | Note |
|----------|-------------|------|
| 45 frames | 1,088 | Most common (IASA cancel) |
| 52 frames | 1,037 | Slight charge |
| 49 frames | 340 | Full animation |
| 56 frames | 571 | Charged |

### Entry Conditions

| From State | Name | Occurrences | Percentage |
|------------|------|-------------|------------|
| 14 | WAIT | 2,171 | 20.29% |
| 39 | SQUAT | 1,978 | **18.48%** |
| 42 | LANDING | 1,103 | 10.31% |
| 40 | SQUAT_WAIT | 1,087 | 10.16% |
| 18 | TURN | 909 | 8.49% |
| 15 | WALK_SLOW | 860 | 8.04% |
| 16 | WALK_MIDDLE | 671 | 6.27% |
| 50 | ATTACK_DASH | 373 | 3.49% |

**28.64% from crouch states** (SQUAT + SQUAT_WAIT) - down smash is often used from crouch for quick punishes.

### Exit Conditions

| To State | Name | Occurrences | Percentage |
|----------|------|-------------|------------|
| 14 | WAIT | 4,201 | 42.9% |
| 178 | GUARD_ON | 1,072 | 10.95% |
| 18 | TURN | 930 | 9.5% |
| 88 | DAMAGE_FLY_N (traded) | 667 | 6.81% |
| 20 | DASH | 657 | 6.71% |
| 15 | WALK_SLOW | 573 | 5.85% |
| 90 | DAMAGE_FLY_LW | 389 | 3.97% |
| 226 | CAPTURE_PULLED (grabbed) | 353 | 3.61% |

### Key Properties

- **On ground**: 100%
- **Invulnerability**: 1.43%
- **Hitstun**: 0%
- **Semi-Spike Angle**: 25° (NTSC) / 30° (PAL)
- **Fastest Smash**: Frame 6 startup

### Competitive Uses

1. **Edgeguarding**: Semi-spike angle sends opponents off-stage at low angles
2. **Waveshine → Down Smash**: At mid-percent near ledge
3. **Crouch Cancel Punish**: From crouch, punish with down smash
4. **Two-Sided Coverage**: Hits both sides simultaneously

---

## Comparative Analysis

### Smash Attack Comparison

| Property | Forward (60) | Up (63) | Down (64) |
|----------|-------------|---------|-----------|
| Startup | Frame 12 | Frame 7 | Frame 6 |
| Total Frames | 39 | 41 | 49 |
| Clean Damage | 15% | 18% | 15% |
| Usage Rate | 0.23% | **1.66%** | 0.16% |
| Instances | 15,001 | 40,701 | 9,787 |
| JC Entry | 0% | **74.26%** | 0% |
| Trade Rate | 14.33% | 7.81% | 10.78% |

### Entry Pattern Comparison

| Source | Forward (60) | Up (63) | Down (64) |
|--------|-------------|---------|-----------|
| KNEE_BEND (JC) | 0% | **74.26%** | 0% |
| WAIT | 20.13% | 10.57% | 20.29% |
| Crouch States | 17.39% | 3.32% | **28.64%** |
| DASH | 18.46% | 0.40% | 0% |
| LANDING | 10.64% | 1.33% | 10.31% |

### Technique Patterns

**Jump Cancel Up Smash (JC Upsmash)**:
The dominant technique for using up smash. From a dash:
1. Jump (KNEE_BEND) - 3 frames for Fox
2. Immediately input up smash
3. The jumpsquat is canceled into up smash

This explains the 74.26% KNEE_BEND entry rate for up smash vs 0% for forward/down smash.

**Crouch → Down Smash**:
28.64% of down smashes come from crouch states, used for:
- Quick punishes from crouch cancel
- Reading opponent approaches

---

## State Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GROUNDED STATES                                  │
│                    (WAIT, DASH, CROUCH, etc.)                            │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│     WAIT      │         │     DASH      │         │    SQUAT      │
│     (14)      │         │    (20/21)    │         │   (39/40)     │
│   20% fsmash  │         │    18% fsmash │         │   28% dsmash  │
└───────┬───────┘         └───────┬───────┘         └───────┬───────┘
        │                         │                         │
        │                         ▼                         │
        │                 ┌───────────────┐                 │
        │                 │  KNEE_BEND    │                 │
        │                 │     (24)      │                 │
        │                 │  Jumpsquat    │                 │
        │                 └───────┬───────┘                 │
        │                         │                         │
        │                         │ 74.26%                  │
        ▼                         ▼                         ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│ FORWARD SMASH │         │   UP SMASH    │         │  DOWN SMASH   │
│     (60)      │         │     (63)      │         │     (64)      │
│   39 frames   │         │   41 frames   │         │   49 frames   │
│   15% damage  │         │   18% damage  │         │   15% damage  │
│   0.23% usage │         │   1.66% usage │         │   0.16% usage │
└───────────────┘         └───────────────┘         └───────────────┘
        │                         │                         │
        │                         │                         │
        ▼                         ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              EXIT STATES                                 │
│        WAIT (43%), SHIELD (9-15%), DAMAGE (8-14%), DASH (7%)            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Advanced Techniques

### Jump Cancel Up Smash (JC Upsmash)

The most important smash attack technique for Fox. Execution:
1. Dash toward opponent
2. Press jump (Y/X) during dash
3. During the 3-frame jumpsquat, input up smash (C-stick up or up + A)
4. Up smash comes out instead of jump

**Why it's powerful**:
- Allows up smash from a running approach
- Covers more ground than standing up smash
- Essential for punish game off of shine and grabs
- 74.26% of all up smashes use this technique

### Waveshine → Up Smash

Core Fox combo:
1. Shine (360) hits opponent
2. Wavedash forward (43)
3. JC Up Smash (63)

This is the quintessential Fox kill confirm at mid-to-high percent.

### Crouch Cancel → Down Smash

Defensive punish:
1. Hold crouch to absorb hit with reduced knockback
2. If not launched, immediately down smash

28.64% crouch entry rate for down smash reflects this pattern.

### Edgeguard Down Smash

The semi-spike angle (25°) makes down smash ideal for edgeguarding:
- Send opponent low off-stage
- Forces difficult recovery angles
- Used off waveshine near ledge

---

## Character-Specific Notes (Fox)

Fox's smash attack characteristics:
- **Up Smash**: Best in the game - frame 7 startup, 18% damage, head intangibility
- **Down Smash**: Fastest smash - frame 6 startup, semi-spike for edgeguards
- **Forward Smash**: Least used - slower startup (frame 12), but longest range

**PAL Nerfs**:
- Up smash: 18% → 17% damage, reduced knockback
- Down smash: 15% → 13% damage, angle 25° → 30°

These nerfs reduce Fox's kill power, particularly for up smash confirms.

---

## Sources

### Primary Sources (Quantitative)
- **Parquet replay data**: `/Users/eppie/ssb_wiki/fox_vs_fox_parquet/` - 5.88 million frames of smash attack data analyzed

### Documentation Sources (Qualitative)
- [SmashWiki - Fox Up Smash](https://www.ssbwiki.com/Fox_(SSBM)/Up_smash)
- [SmashWiki - Fox Forward Smash](https://www.ssbwiki.com/Fox_(SSBM)/Forward_smash)
- [SmashWiki - Fox Down Smash](https://www.ssbwiki.com/Fox_(SSBM)/Down_smash)
- [libmelee Action State Enums](https://libmelee.readthedocs.io/en/latest/enums.html)

---

## Investigation Methodology

1. **Parquet Replay Data (Primary)**: Provided quantitative data on smash usage, durations, entry/exit patterns.

2. **SmashWiki (High Value)**: Provided frame data, damage values, knockback properties, NTSC/PAL differences.

3. **action_state.json (Reference)**: Mapped state IDs to internal names.

**Key Findings**:
- **JC Up Smash dominates**: 74.26% of up smashes from KNEE_BEND
- **Up smash is Fox's most used smash**: 1.66% of all frames (81.3% of smash frames)
- **Down smash from crouch**: 28.64% entry from crouch states
- **Forward smash least used**: Only 0.23% of all frames
- **Trade rates differ**: Forward smash trades most (14.33%), up smash least (7.81%)
