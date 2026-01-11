# Shield & Powershield Action States (178-182, 205-211)

This document describes the shield, powershield, and shield break action states in Super Smash Bros. Melee.

## Overview

Shielding is a fundamental defensive option in Melee. Characters can raise a bubble shield to block attacks, powershield to reflect projectiles or reduce shield stun, and risk shield breaks if their shield takes too much damage.

| Category | States | Description |
|----------|--------|-------------|
| Shield Active | 178, 179, 180 | Shield startup, hold, release |
| Shield Stun | 181 | Hitstun while shielding an attack |
| Powershield | 182 | Perfect shield / projectile reflect |
| Shield Break | 205, 207, 209, 211 | Shield broken → dizzy sequence |

**Shield Properties**:
- **Maximum HP**: 60 (effective 85.71 with 0.7× damage multiplier)
- **Depletion Rate**: 0.28 HP/frame (shield shrinks over time)
- **Regeneration**: 0.07 HP/frame
- **Startup**: Frame 1 (instant)
- **Shield Drop Lag**: 15 frames

---

## Usage Statistics (Fox Dittos)

Total dataset: 288.5 million frames

| State | Name | Total Frames | % of Gameplay | Instances |
|-------|------|--------------|---------------|-----------|
| 179 | GUARD (hold) | 3,392,028 | 1.17% | 372,320 |
| 178 | GUARD_ON (startup) | 2,493,186 | 0.86% | 350,705 |
| 181 | GUARD_SET_OFF (stun) | 2,071,880 | 0.72% | 174,233 |
| 182 | GUARD_REFLECT (powershield) | 1,102,163 | 0.38% | 181,094 |
| 180 | GUARD_OFF (release) | 463,898 | 0.16% | 55,003 |
| 211 | FURA_FURA (dizzy) | 101 | ~0% | 2 |
| 209 | SHIELD_BREAK_STAND_U | 60 | ~0% | 2 |
| 205 | SHIELD_BREAK_FLY | 58 | ~0% | 2 |
| 207 | SHIELD_BREAK_DOWN_U | 52 | ~0% | 2 |

**Key Insight**: Shield-related states account for **3.3%** of all Fox ditto gameplay. Shield breaks are exceptionally rare (only 2 instances in entire dataset).

---

## Duration Statistics

| State | Name | Min | P25 | Median | P75 | Max | Instances |
|-------|------|-----|-----|--------|-----|-----|-----------|
| 178 | GUARD_ON | 1 | 3 | 8 | 8 | 78 | 350,705 |
| 179 | GUARD | 1 | 3 | 6 | 12 | 304 | 372,320 |
| 180 | GUARD_OFF | 1 | 3 | 7 | 15 | 19 | 55,003 |
| 181 | GUARD_SET_OFF | 1 | 8 | 12 | 14 | 59 | 174,233 |
| 182 | GUARD_REFLECT | 1 | 3 | 7 | 8 | 59 | 181,094 |
| 205 | SHIELD_BREAK_FLY | 29 | 29 | 29 | 29 | 29 | 2 |
| 207 | SHIELD_BREAK_DOWN_U | 26 | 26 | 26 | 26 | 26 | 2 |
| 209 | SHIELD_BREAK_STAND_U | 30 | 30 | 30 | 30 | 30 | 2 |
| 211 | FURA_FURA | 30 | 40 | 51 | 61 | 71 | 2 |

---

## Key Properties

| State | On Ground | Invulnerable | In Hitstun |
|-------|-----------|--------------|------------|
| 178 GUARD_ON | 100% | 2.72% | 0% |
| 179 GUARD | 100% | 0.20% | 0% |
| 180 GUARD_OFF | 100% | 2.14% | 0% |
| 181 GUARD_SET_OFF | 100% | 0.37% | 0% |
| 182 GUARD_REFLECT | 100% | 2.60% | 0% |
| 205 SHIELD_BREAK_FLY | 0% | 100% | 0% |
| 207 SHIELD_BREAK_DOWN_U | 100% | 100% | 0% |
| 209 SHIELD_BREAK_STAND_U | 100% | 100% | 0% |
| 211 FURA_FURA | 100% | 0% | 0% |

**Note**: Shield break states (205, 207, 209) are invulnerable. The dizzied character (211) is vulnerable and can be hit with a free punish.

---

## Detailed State Descriptions

### GUARD_ON (State 178) - Shield Startup

**Internal Name**: `GUARD_ON`

**Description**: The shield activation state. Shield comes out on frame 1 but this state represents the initial shield raise animation. Can be canceled into various options or transition to full shield hold.

**Entry Conditions** (top sources):
| From State | Name | Occurrences |
|------------|------|-------------|
| 42 | LANDING | 96,199 |
| 20 | DASH | 88,645 |
| 43 | LANDING_FALL_SPECIAL | 50,201 |
| 70 | LANDING_AIR_N | 38,110 |
| 72 | LANDING_AIR_B | 30,839 |
| 74 | LANDING_AIR_LW | 29,979 |
| 212 | CATCH (whiffed grab) | 13,477 |
| 79 | DAMAGE_N_2 | 12,886 |

**Exit Conditions**:
| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 179 | GUARD (hold shield) | 154,567 | 27.8% |
| 24 | KNEE_BEND (jump OoS) | 81,726 | 14.7% |
| 244 | PASS (drop through) | 63,370 | 11.4% |
| 181 | GUARD_SET_OFF (shield stun) | 56,850 | 10.2% |
| 182 | GUARD_REFLECT (powershield) | 56,671 | 10.2% |
| 235 | ESCAPE (spotdodge) | 35,088 | 6.3% |
| 233 | ESCAPE_F (roll forward) | 31,710 | 5.7% |
| 234 | ESCAPE_B (roll back) | 22,311 | 4.0% |
| 180 | GUARD_OFF (release) | 18,373 | 3.3% |
| 226 | CAPTURE_WAIT (got grabbed) | 13,302 | 2.4% |
| 212 | CATCH (grab) | 12,283 | 2.2% |

**Statistics**:
- Median duration: 8 frames
- Average duration: 7.11 frames

**Key Finding**: 10.2% of GUARD_ON results in GUARD_REFLECT (powershield), indicating the 4-frame powershield window is frequently utilized.

---

### GUARD (State 179) - Shield Hold

**Internal Name**: `GUARD`

**Description**: The main shield holding state. Character maintains their bubble shield and can transition to various out-of-shield options. Shield depletes at 0.28 HP/frame while held.

**Exit Conditions (Out of Shield Options)**:
| Action | Occurrences | % |
|--------|-------------|---|
| KNEE_BEND (Jump OoS) | 183,185 | **49.2%** |
| GUARD_SET_OFF (got hit) | 66,916 | 18.0% |
| PASS (drop through) | 34,161 | 9.2% |
| GUARD_OFF (release) | 25,452 | 6.8% |
| CATCH (shield grab) | 22,393 | 6.0% |
| ROLL (forward + back) | 21,504 | 5.8% |
| SPOT DODGE | 6,541 | 1.8% |
| Got grabbed | 6,071 | 1.6% |
| Got hit (tumble) | 3,035 | 0.8% |

**Statistics**:
- Median duration: 6 frames
- Maximum duration: 304 frames

**Key Finding**: **Jump Out of Shield (49.2%)** is by far the most common OoS option, reflecting Fox's strong aerial game and shine OoS capability.

---

### GUARD_OFF (State 180) - Shield Release

**Internal Name**: `GUARD_OFF`

**Description**: The shield drop animation. Has 15 frames of lag before the character can act, making it the slowest way to exit shield. Players typically avoid this by using OoS options.

**Exit Conditions**:
| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 24 | KNEE_BEND (jump) | 26,596 | 48.4% |
| 14 | WAIT | 11,003 | 20.0% |
| 235 | ESCAPE (spotdodge) | 4,427 | 8.1% |
| 88 | DAMAGE_FLY_N (got hit) | 1,960 | 3.6% |
| 178 | GUARD_ON (re-shield) | 1,939 | 3.5% |

**Statistics**:
- Median duration: 7 frames
- Maximum duration: 19 frames

---

### GUARD_SET_OFF (State 181) - Shield Stun

**Internal Name**: `GUARD_SET_OFF`

**Description**: The shield stun state when an attack connects with the shield. Duration depends on the attack's damage. Character cannot act during shield stun but retains their shield.

**Shield Stun Formula**: `(damage × 0.45 + 2) × (200/201)`

**Entry Conditions**:
| From State | Name | Occurrences |
|------------|------|-------------|
| 179 | GUARD | 66,916 |
| 178 | GUARD_ON | 56,850 |
| 182 | GUARD_REFLECT | 24,162 |

**Exit Conditions**:
| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 179 | GUARD (return to shield) | 133,988 | **76.3%** |
| 251 | MISS_FOOT (pushed off) | 6,760 | 3.9% |
| 24 | KNEE_BEND (jump) | 6,534 | 3.7% |
| 180 | GUARD_OFF (release) | 6,217 | 3.5% |
| 29 | FALL (pushed off stage) | 4,478 | 2.6% |
| 235 | ESCAPE (spotdodge) | 4,167 | 2.4% |

**Duration Distribution** (shield stun frames):
| Frames | Common Attacks |
|--------|----------------|
| 4-6 | Laser, weak nair |
| 7-9 | Jab, weak aerial |
| 10-12 | Standard aerials |
| 13-15 | Strong aerials |
| 16-20 | Smash attacks |
| 20+ | Fully charged smash |

**Statistics**:
- Median duration: 12 frames
- Average duration: 11.89 frames

**Key Finding**: 76.3% of shield stun exits return to GUARD, indicating players successfully hold shield through pressure. 3.9% get pushed off ledge (MISS_FOOT).

---

### GUARD_REFLECT (State 182) - Powershield

**Internal Name**: `GUARD_REFLECT`

**Description**: The powershield state activated within the first 4 frames of shielding (2 frames for projectiles). Powershielding reflects projectiles at half damage and allows powershield canceling for physical attacks.

**Powershield Windows**:
- **Physical attacks**: 4 frames
- **Projectiles**: 2 frames (reflects projectile)

**Entry Conditions** (revealing powershield sources):
| From State | Name | Occurrences | Notes |
|------------|------|-------------|-------|
| 20 | DASH | 102,884 | **Laser powershielding while running** |
| 178 | GUARD_ON | 56,671 | Standard powershield |
| 18 | TURN | 11,635 | Turnaround powershield |
| 21 | RUN | 9,847 | Running powershield |
| 42 | LANDING | 9,362 | Landing powershield |
| 14 | WAIT | 9,187 | Standing powershield |

**Exit Conditions**:
| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 179 | GUARD (hold shield) | 83,702 | 37.6% |
| 24 | KNEE_BEND (jump) | 59,073 | 26.5% |
| 181 | GUARD_SET_OFF (got hit again) | 24,162 | 10.9% |
| 244 | PASS (drop through) | 21,760 | 9.8% |
| 233 | ESCAPE_F (roll forward) | 7,806 | 3.5% |
| 234 | ESCAPE_B (roll back) | 6,990 | 3.1% |
| 235 | ESCAPE (spotdodge) | 6,349 | 2.9% |

**Statistics**:
- Median duration: 7 frames
- Average duration: 6.09 frames
- Instances: 181,094

**Key Finding**: **Dash-powershield (102,884 occurrences)** is the dominant entry method - Fox players dash and powershield incoming lasers while approaching. This is a core Fox neutral game technique.

### Powershield Rate Analysis

| Metric | Value |
|--------|-------|
| GUARD_ON entries | 556,481 |
| GUARD_REFLECT entries | 222,578 |
| **Powershield Rate** | **28.6%** |

28.6% of all shield activations result in a powershield, demonstrating the high technical execution in Fox dittos.

---

## Shield Break States (205, 207, 209, 211)

Shield breaks are **extremely rare** - only 2 occurrences in the entire dataset (288.5 million frames).

### Shield Break Sequence

When a shield breaks, the character goes through this state sequence:

```
179 GUARD (shield breaks)
     │
     ▼
205 SHIELD_BREAK_FLY (29 frames)
    Character launched upward, invulnerable
     │
     ▼
207 SHIELD_BREAK_DOWN_U (26 frames)
    Landing face-up, invulnerable
     │
     ▼
209 SHIELD_BREAK_STAND_U (30 frames)
    Standing up, invulnerable
     │
     ▼
211 FURA_FURA (30-71 frames)
    Dizzy/stunned state, VULNERABLE
    Can be woken by attacks
```

### SHIELD_BREAK_FLY (State 205)

**Description**: Character is launched upward after shield break. Fully invulnerable.

**Statistics**: 29 frames fixed duration

---

### SHIELD_BREAK_DOWN_U (State 207)

**Description**: Character lands face-up after shield break launch. Still invulnerable.

**Statistics**: 26 frames fixed duration

---

### SHIELD_BREAK_STAND_U (State 209)

**Description**: Character stands up from knockdown position. Still invulnerable.

**Statistics**: 30 frames fixed duration

---

### FURA_FURA (State 211) - Dizzy State

**Internal Name**: `FURA_FURA`

**Description**: The stunned/dizzy state after shield break. Character wobbles with stars around their head. **Vulnerable to attack** - opponent gets a free punish (typically fully charged smash).

**Exit Conditions**:
| To State | Occurrences | Notes |
|----------|-------------|-------|
| 90 | DAMAGE_FLY_TOP | 1 | Got hit (punished) |
| 226 | CAPTURE_WAIT | 1 | Got grabbed |

**Statistics**:
- Duration range: 30-71 frames
- Median: 51 frames

**Note**: Dizzy duration can be reduced by mashing buttons/stick.

---

## Out of Shield Options Analysis

From GUARD (179), players exit via:

| Option | Frames to Action | Usage % | Notes |
|--------|------------------|---------|-------|
| **Jump** (→ aerial/shine) | 4 (Fox jumpsquat) | 49.2% | Fastest, most common |
| **Shield Grab** | 7 | 6.0% | Punishes close attacks |
| **Roll** (forward/back) | 4 (startup) | 5.8% | Repositioning |
| **Spotdodge** | 2 (startup) | 1.8% | Invuln frames 2-20 |
| **Drop Through** | 1 | 9.2% | Platform only |
| **Shield Drop** | 15 | 6.8% | Slowest, avoided |

**Optimal OoS Options for Fox**:
1. **Shine OoS**: Jump (4f) → Shine (frame 1) = 5 frame punish
2. **Nair OoS**: Jump (4f) → Nair (frame 4) = 8 frame punish
3. **Up Smash OoS**: Jump cancel up smash = 5 frame punish

---

## State Flow Diagram

```
                    ┌──────────────────────────────────┐
                    │        SHIELDING INITIATED       │
                    │  (from any grounded actionable)  │
                    └────────────────┬─────────────────┘
                                     │
                                     ▼
                    ┌──────────────────────────────────┐
                    │         GUARD_ON (178)           │
                    │      Shield startup (frame 1)    │
                    │         0.86% gameplay           │
                    └──────┬─────────────┬─────────────┘
                           │             │
          ┌────────────────┤             ├────────────────┐
          │ (4 frame       │             │ (no hit)       │
          │ powershield)   │             │                │
          ▼                ▼             ▼                ▼
   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
   │GUARD_REFLECT │ │GUARD_SET_OFF │ │    GUARD     │ │ OoS OPTIONS  │
   │    (182)     │ │    (181)     │ │    (179)     │ │ Jump, Roll,  │
   │ Powershield  │ │ Shield Stun  │ │ Shield Hold  │ │ Grab, Dodge  │
   │   0.38%      │ │   0.72%      │ │   1.17%      │ │              │
   └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────────────┘
          │                │                │
          │    ┌───────────┘                │
          │    │  76.3% return              │
          │    ▼  to GUARD                  │
          │ ┌──────────────┐                │
          └─│              │◄───────────────┘
            │    GUARD     │
            │    (179)     │
            └──────┬───────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
    ▼              ▼              ▼
┌────────┐   ┌──────────┐   ┌──────────┐
│ ATTACK │   │  SHIELD  │   │ SHIELD   │
│  HIT   │   │  GRAB    │   │  BREAK   │
│ (rare) │   │  (6.0%)  │   │ (0.0%)   │
└────────┘   └──────────┘   └────┬─────┘
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │ 205 SHIELD_BREAK_FLY │
                    │     (invulnerable)   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 207 SHIELD_BREAK_    │
                    │ DOWN_U (invuln)      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 209 SHIELD_BREAK_    │
                    │ STAND_U (invuln)     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    211 FURA_FURA     │
                    │   (DIZZY - PUNISH!)  │
                    │   30-71 frames       │
                    └──────────────────────┘
```

---

## Practical Implications

### Shield Pressure

Shield stun formula `(damage × 0.45 + 2)` means:
- **Laser** (1-3%): ~3-4 frames stun
- **Nair** (9-12%): ~6-8 frames stun
- **Shine** (8%): ~6 frames stun
- **Dair** (15%): ~9 frames stun

**True Shield Pressure**: Attacks spaced so shield stun exceeds defender's fastest OoS option (typically 5 frames for Fox shine OoS).

### Powershield Applications

1. **Laser Powershield**: Run at opponent, powershield their laser, continue approach
2. **Attack Powershield**: Powershield melee attack, get 1-frame advantage for punish
3. **Projectile Reflect**: Send projectile back at half damage

### Why Shield Breaks Are Rare

Shield HP management in competitive play:
- Players release shield before it depletes
- Strong attacks are often jumped/rolled rather than blocked
- Multi-hit pressure leads to shield poke (hit through shrunk shield) rather than break

---

## Sources

### Primary Sources (Quantitative)
- **Parquet replay data**: `/Users/eppie/ssb_wiki/fox_vs_fox_parquet/` - 9.5 million frames of shield state data analyzed

### Documentation Sources (Qualitative)
- [SmashWiki - Shield](https://www.ssbwiki.com/Shield)
- [SmashWiki - Powershield](https://www.ssbwiki.com/Powershield)
- Smashboards "Shield Pressure Research Project" thread
- [libmelee Action State Enums](https://libmelee.readthedocs.io/en/latest/enums.html)

### Reference Data
- `action_state.json` - State ID to name mapping
