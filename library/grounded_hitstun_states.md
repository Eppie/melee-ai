# Grounded Hitstun Action States (75-83)

This document describes the grounded hitstun (damage) action states in Super Smash Bros. Melee, covering states 75-83.

## Overview

When a grounded character is hit by an attack, they enter one of 9 grounded hitstun states. These states are organized along two axes:

1. **Knockback Angle**: The direction the character is launched
   - **HI (High)**: Upward knockback angle (states 75-77)
   - **N (Neutral)**: Horizontal/diagonal knockback angle (states 78-80)
   - **LW (Low)**: Downward knockback angle (states 81-83)

2. **Knockback Intensity**: The strength of the hit
   - **Level 1**: Light hits, low hitstun (states 75, 78, 81)
   - **Level 2**: Medium hits, moderate hitstun (states 76, 79, 82)
   - **Level 3**: Heavy hits, high hitstun (states 77, 80, 83)

| Intensity | HI (Upward) | N (Neutral) | LW (Downward) |
|-----------|-------------|-------------|---------------|
| 1 (Light) | 75: DAMAGE_HI_1 | 78: DAMAGE_N_1 | 81: DAMAGE_LW_1 |
| 2 (Medium) | 76: DAMAGE_HI_2 | 79: DAMAGE_N_2 | 82: DAMAGE_LW_2 |
| 3 (Heavy) | 77: DAMAGE_HI_3 | 80: DAMAGE_N_3 | 83: DAMAGE_LW_3 |

---

## Hitstun Mechanics

### Hitstun Calculation

Hitstun duration in Melee is calculated as:

```
Hitstun Frames = Knockback × 0.4
```

For example, a hit dealing 100 units of knockback causes 40 frames of hitstun.

### Key Properties

- **Cannot act during hitstun**: Character is locked in the damage animation
- **Can DI (Directional Influence)**: Alter trajectory using control stick
- **Can SDI (Smash DI)**: Shift position slightly on each hit
- **Grounded hitstun can become airborne**: Heavy hits lift the character off the ground

---

## Usage Statistics (Fox Dittos)

Total dataset: 288.5 million frames

| State | Name | Total Frames | % of Gameplay | Instances |
|-------|------|--------------|---------------|-----------|
| 79 | DAMAGE_N_2 | 1,309,179 | 0.453% | 59,772 |
| 80 | DAMAGE_N_3 | 650,918 | 0.225% | 32,908 |
| 76 | DAMAGE_HI_2 | 639,247 | 0.221% | 57,019 |
| 77 | DAMAGE_HI_3 | 374,894 | 0.130% | 21,582 |
| 82 | DAMAGE_LW_2 | 355,539 | 0.123% | 27,093 |
| 83 | DAMAGE_LW_3 | 103,165 | 0.036% | 4,976 |
| 78 | DAMAGE_N_1 | 24,329 | 0.008% | 2,563 |
| 75 | DAMAGE_HI_1 | 12,677 | 0.004% | 1,220 |
| 81 | DAMAGE_LW_1 | 1,289 | 0.0004% | 108 |

**Key Insight**: DAMAGE_N_2 (state 79) is by far the most common grounded hitstun state, occurring 0.45% of gameplay. Level 2 states (medium intensity) dominate, while Level 1 states (light intensity) are rare in competitive play.

---

## Intensity Level Analysis

### Correlation with Victim Percent

The intensity level strongly correlates with the victim's damage percent when hit:

| Intensity | States | Avg Percent | Avg Hitstun | Description |
|-----------|--------|-------------|-------------|-------------|
| Level 1 | 75, 78, 81 | 7-10% | 5.5 frames | Light jabs, weak hits |
| Level 2 | 76, 79, 82 | 23-27% | 11-13 frames | Standard combo hits |
| Level 3 | 77, 80, 83 | 44-51% | 18-21 frames | Strong hits, combo enders |

### Duration Statistics by State

| State | Name | Min | P25 | Median | P75 | Max | Instances |
|-------|------|-----|-----|--------|-----|-----|-----------|
| 75 | DAMAGE_HI_1 | 1 | 9 | 11 | 13 | 27 | 1,220 |
| 76 | DAMAGE_HI_2 | 1 | 5 | 8 | 16 | 96 | 57,019 |
| 77 | DAMAGE_HI_3 | 1 | 7 | 17 | 21 | 90 | 21,582 |
| 78 | DAMAGE_N_1 | 1 | 6 | 10 | 13 | 30 | 2,563 |
| 79 | DAMAGE_N_2 | 1 | 14 | 19 | 26 | 158 | 59,772 |
| 80 | DAMAGE_N_3 | 1 | 8 | 19 | 25 | 195 | 32,908 |
| 81 | DAMAGE_LW_1 | 1 | 10 | 13 | 14 | 17 | 108 |
| 82 | DAMAGE_LW_2 | 1 | 5 | 15 | 19 | 68 | 27,093 |
| 83 | DAMAGE_LW_3 | 1 | 17 | 20 | 27 | 70 | 4,976 |

---

## Angle Type Analysis

### On-Ground Percentage

A critical finding: higher intensity states often result in the character becoming airborne during the hitstun animation.

| State | Name | % On Ground | Interpretation |
|-------|------|-------------|----------------|
| 75 | DAMAGE_HI_1 | 79.8% | Mostly grounded |
| 76 | DAMAGE_HI_2 | 67.9% | Mixed |
| 77 | DAMAGE_HI_3 | 6.3% | Almost always airborne |
| 78 | DAMAGE_N_1 | 41.1% | Mixed |
| 79 | DAMAGE_N_2 | 81.1% | Mostly grounded |
| 80 | DAMAGE_N_3 | 39.3% | Mixed |
| 81 | DAMAGE_LW_1 | 73.6% | Mostly grounded |
| 82 | DAMAGE_LW_2 | 79.7% | Mostly grounded |
| 83 | DAMAGE_LW_3 | 1.0% | Almost always airborne |

**Key Finding**: The "3" (heavy) variants of HI and LW angles (states 77 and 83) have extremely low on-ground percentages (6.3% and 1.0%), indicating that heavy upward or downward angle hits launch the character significantly into the air.

### Which Moves Cause Which Angles?

Analysis of Fox attack states and their knockback angles:

| Move | State | HI % | N % | LW % | Primary Angle |
|------|-------|------|-----|------|---------------|
| Up Tilt | 56 | 21.5% | 40.4% | 38.1% | N/LW mixed |
| Down Tilt | 57 | 69.4% | 30.2% | 0.5% | HI (sends up!) |
| Up Smash | 63 | 38.9% | 49.4% | 11.8% | N/HI |
| Down Smash | 64 | 44.5% | 45.8% | 9.7% | N/HI |
| Shine (grounded) | 360 | 62.0% | 35.8% | 2.3% | HI |
| Shine (aerial) | 361 | 58.0% | 39.2% | 2.9% | HI |

**Surprising Finding**: Fox's down tilt sends opponents primarily at an upward (HI) angle (69.4%), not downward. Shine also sends primarily upward (60%+), which enables its famous combo potential.

---

## Detailed State Descriptions

### DAMAGE_HI_1 (State 75) - Light Upward Hit

**Internal Name**: `DAMAGE_HI_1`

**Description**: Light upward knockback hitstun. Occurs from weak hits with upward trajectory at low percent.

**Entry Conditions** (top sources):
| From State | Name | Occurrences |
|------------|------|-------------|
| 39 | SQUAT (crouch) | 264 |
| 40 | SQUAT_WAIT | 216 |
| 20 | DASH | 150 |
| 24 | KNEE_BEND | 52 |
| 42 | LANDING | 38 |

**Exit Conditions**:
| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 39 | SQUAT | 387 | 31.4% |
| 42 | LANDING | 213 | 17.3% |
| 178 | GUARD_ON (shield) | 173 | 14.0% |
| 24 | KNEE_BEND (jump) | 123 | 10.0% |
| 14 | WAIT | 104 | 8.4% |

**Statistics**:
- Median duration: 11 frames
- Average percent when hit: 10.08%
- Average hitstun remaining: 5.68 frames
- On ground: 79.8%

---

### DAMAGE_HI_2 (State 76) - Medium Upward Hit

**Internal Name**: `DAMAGE_HI_2`

**Description**: The most common upward hitstun state. Occurs from standard combo hits with upward trajectory. Often chains into other hitstun states or aerial damage states.

**Entry Conditions** (top sources - attacks):
| From State | Name | Occurrences |
|------------|------|-------------|
| 63 | ATTACK_HI_4 (upsmash) | 2,020 |
| 50 | ATTACK_DASH | 1,470 |
| 44 | ATTACK_11 (jab) | 850 |
| 57 | ATTACK_LW_3 (dtilt) | 679 |
| 56 | ATTACK_HI_3 (utilt) | 650 |
| 360 | REFLECTOR (shine) | 491 |

**Exit Conditions**:
| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 82 | DAMAGE_LW_2 | 16,775 | 25.6% |
| 42 | LANDING | 16,657 | 25.5% |
| 79 | DAMAGE_N_2 | 15,046 | 23.0% |
| 178 | GUARD_ON | 4,386 | 6.7% |
| 90 | DAMAGE_FLY_LW | 2,375 | 3.6% |

**Statistics**:
- Median duration: 8 frames
- Average percent when hit: 22.70%
- Average hitstun remaining: 12.74 frames
- On ground: 67.9%

**Key Finding**: State 76 frequently chains into other damage states (25.6% → LW_2, 23.0% → N_2), indicating it's a key combo state where characters receive multiple hits.

---

### DAMAGE_HI_3 (State 77) - Heavy Upward Hit

**Internal Name**: `DAMAGE_HI_3`

**Description**: Heavy upward knockback hitstun. Character is almost always launched into the air (93.7% airborne). Common exit from combos before aerial follow-ups.

**Entry Conditions** (top attack sources):
| From State | Name | Occurrences |
|------------|------|-------------|
| 63 | ATTACK_HI_4 (upsmash) | 721 |
| 50 | ATTACK_DASH | 672 |
| 57 | ATTACK_LW_3 (dtilt) | 412 |
| 44 | ATTACK_11 (jab) | 361 |
| 56 | ATTACK_HI_3 (utilt) | 329 |

**Exit Conditions**:
| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 42 | LANDING | 20,874 | 84.6% |
| 29 | FALL | 836 | 3.4% |
| 27 | JUMP_AERIAL_F (DJ) | 704 | 2.9% |
| 24 | KNEE_BEND | 648 | 2.6% |
| 88 | DAMAGE_FLY_N | 414 | 1.7% |

**Statistics**:
- Median duration: 17 frames
- Average percent when hit: 49.23%
- Average hitstun remaining: 20.53 frames
- On ground: 6.3%

**Key Finding**: 84.6% of DAMAGE_HI_3 exits lead to LANDING, meaning the character was popped into the air by the hit and then landed after hitstun ended.

---

### DAMAGE_N_1 (State 78) - Light Neutral Hit

**Internal Name**: `DAMAGE_N_1`

**Description**: Light neutral/horizontal knockback hitstun. Less common than other states.

**Statistics**:
- Median duration: 10 frames
- Average percent when hit: 8.82%
- Average hitstun remaining: 5.54 frames
- On ground: 41.1%

---

### DAMAGE_N_2 (State 79) - Medium Neutral Hit

**Internal Name**: `DAMAGE_N_2`

**Description**: **The most common grounded hitstun state** (0.45% of all gameplay). Occurs from standard horizontal knockback hits. Central to Melee's combo game.

**Entry Conditions** (top sources):
| From State | Name | Occurrences |
|------------|------|-------------|
| 76 | DAMAGE_HI_2 (chained) | 15,046 |
| 82 | DAMAGE_LW_2 (chained) | 6,809 |
| 20 | DASH | 4,992 |
| 42 | LANDING | 4,585 |
| 212 | CATCH (grab) | 3,426 |
| 43 | LANDING_FALL_SPECIAL | 3,067 |

**Exit Conditions**:
| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 88 | DAMAGE_FLY_N (tumble) | 22,254 | 28.5% |
| 42 | LANDING | 19,154 | 24.5% |
| 178 | GUARD_ON (shield) | 12,886 | 16.5% |
| 90 | DAMAGE_FLY_LW | 4,362 | 5.6% |
| 39 | SQUAT (crouch) | 3,626 | 4.6% |

**Statistics**:
- Median duration: 19 frames
- Average percent when hit: 27.49%
- Average hitstun remaining: 12.03 frames
- On ground: 81.1%

**Key Finding**: 28.5% of exits go to DAMAGE_FLY_N (tumble), meaning hits in this state often lead to getting launched into the air. 16.5% exit to shield, indicating defensive options after hitstun ends.

---

### DAMAGE_N_3 (State 80) - Heavy Neutral Hit

**Internal Name**: `DAMAGE_N_3`

**Description**: Heavy neutral knockback hitstun. Character is often popped into the air (60.7% airborne).

**Exit Conditions**:
| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 42 | LANDING | 26,424 | 65.1% |
| 24 | KNEE_BEND (jump) | 6,591 | 16.2% |
| 251 | MISS_FOOT | 1,213 | 3.0% |
| 86 | DAMAGE_AIR_3 | 984 | 2.4% |

**Statistics**:
- Median duration: 19 frames
- Average percent when hit: 51.11%
- Average hitstun remaining: 17.80 frames
- On ground: 39.3%

---

### DAMAGE_LW_1 (State 81) - Light Downward Hit

**Internal Name**: `DAMAGE_LW_1`

**Description**: Light downward knockback hitstun. **Extremely rare** (only 108 instances in dataset). Downward knockback at low percent is uncommon.

**Statistics**:
- Median duration: 13 frames
- Average percent when hit: 7.45%
- Average hitstun remaining: 5.49 frames
- On ground: 73.6%

---

### DAMAGE_LW_2 (State 82) - Medium Downward Hit

**Internal Name**: `DAMAGE_LW_2`

**Description**: Medium downward knockback hitstun. Often entered from other damage states as the knockback angle shifts.

**Entry Conditions** (top sources):
| From State | Name | Occurrences |
|------------|------|-------------|
| 76 | DAMAGE_HI_2 | 16,775 |
| 56 | ATTACK_HI_3 (utilt) | 2,251 |
| 20 | DASH | 1,640 |
| 79 | DAMAGE_N_2 | 1,116 |

**Exit Conditions**:
| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 79 | DAMAGE_N_2 | 6,809 | 24.4% |
| 88 | DAMAGE_FLY_N | 6,708 | 24.1% |
| 42 | LANDING | 4,059 | 14.6% |
| 178 | GUARD_ON | 3,219 | 11.6% |
| 76 | DAMAGE_HI_2 | 2,310 | 8.3% |

**Statistics**:
- Median duration: 15 frames
- Average percent when hit: 24.77%
- Average hitstun remaining: 11.68 frames
- On ground: 79.7%

**Key Finding**: DAMAGE_LW_2 commonly chains to other damage states (24.4% → N_2, 8.3% → HI_2), indicating it's part of extended combo sequences where angles shift.

---

### DAMAGE_LW_3 (State 83) - Heavy Downward Hit

**Internal Name**: `DAMAGE_LW_3`

**Description**: Heavy downward knockback hitstun. Character is almost always launched (99% airborne). This state is rare and typically occurs at mid-high percent.

**Exit Conditions**:
| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 42 | LANDING | 4,217 | 83.6% |
| 27 | JUMP_AERIAL_F | 256 | 5.1% |
| 29 | FALL | 220 | 4.4% |
| 88 | DAMAGE_FLY_N | 157 | 3.1% |

**Statistics**:
- Median duration: 20 frames
- Average percent when hit: 43.54%
- Average hitstun remaining: 19.47 frames
- On ground: 1.0%

---

## State Chaining Patterns

Grounded hitstun states frequently chain into each other during combos:

| Transition | Occurrences | Pattern |
|------------|-------------|---------|
| 76 → 82 | 16,775 | HI_2 → LW_2 (angle shift) |
| 76 → 79 | 15,046 | HI_2 → N_2 (angle shift) |
| 82 → 79 | 6,809 | LW_2 → N_2 (angle shift) |
| 82 → 76 | 2,310 | LW_2 → HI_2 (multi-hit) |
| 79 → 76 | 1,334 | N_2 → HI_2 (angle shift) |
| 79 → 82 | 1,116 | N_2 → LW_2 (angle shift) |

**Interpretation**: During extended combos, characters transition between different angle states as they receive hits from different directions. The angle classification appears to be based on the current frame's knockback trajectory, not necessarily the original hit.

---

## Relationship to Aerial Hitstun

When knockback is strong enough, grounded hitstun transitions to aerial damage states:

| From State | To Aerial State | Occurrences | Interpretation |
|------------|-----------------|-------------|----------------|
| 79 → 88 | DAMAGE_N_2 → DAMAGE_FLY_N | 22,254 | Combo → Launch |
| 82 → 88 | DAMAGE_LW_2 → DAMAGE_FLY_N | 6,708 | Combo → Launch |
| 76 → 90 | DAMAGE_HI_2 → DAMAGE_FLY_LW | 2,375 | Upward → Downward (gravity) |
| 76 → 88 | DAMAGE_HI_2 → DAMAGE_FLY_N | 1,737 | Combo → Launch |

These transitions represent the moment when a character goes from being hit while grounded to being launched into a tumbling/knockback state.

---

## State Flow Diagram

```
                    ┌─────────────────────────────────┐
                    │         ATTACK CONNECTS         │
                    └────────────────┬────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │   HI (Upward)    │   │   N (Neutral)    │   │   LW (Downward)  │
   │   Knockback      │   │   Knockback      │   │   Knockback      │
   └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
            │                      │                      │
   ┌────────┼────────┐    ┌────────┼────────┐    ┌────────┼────────┐
   │        │        │    │        │        │    │        │        │
   ▼        ▼        ▼    ▼        ▼        ▼    ▼        ▼        ▼
┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
│ 75  │  │ 76  │  │ 77  │  │ 78  │  │ 79  │  │ 80  │  │ 81  │  │ 82  │  │ 83  │
│HI_1 │  │HI_2 │  │HI_3 │  │N_1  │  │N_2  │  │N_3  │  │LW_1 │  │LW_2 │  │LW_3 │
│0.00%│  │0.22%│  │0.13%│  │0.01%│  │0.45%│  │0.23%│  │0.00%│  │0.12%│  │0.04%│
└──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘
   │        │        │        │        │        │        │        │        │
   │        └────────┼────────┴────────┼────────┴────────┼────────┘        │
   │                 │                 │                 │                 │
   │      ┌──────────┴─────────┐       │      ┌──────────┴─────────┐       │
   │      │   Angle Chaining   │       │      │   Angle Chaining   │       │
   │      │   (multi-hit)      │       │      │   (multi-hit)      │       │
   │      └──────────┬─────────┘       │      └──────────┬─────────┘       │
   │                 │                 │                 │                 │
   └─────────────────┼─────────────────┴─────────────────┼─────────────────┘
                     │                                   │
      ┌──────────────┴──────────────┐     ┌──────────────┴──────────────┐
      │                             │     │                             │
      ▼                             ▼     ▼                             ▼
┌──────────────┐              ┌──────────────┐              ┌──────────────┐
│   LANDING    │              │  GUARD_ON    │              │ DAMAGE_FLY   │
│    (42)      │              │   (shield)   │              │  (tumble)    │
│ Hitstun ends │              │  Defensive   │              │   Launch     │
└──────────────┘              └──────────────┘              └──────────────┘
```

---

## Combo Implications

### Optimal Combo Windows

Based on the hitstun remaining data:

| Intensity | Avg Hitstun | Actionable After | Window for Follow-up |
|-----------|-------------|------------------|---------------------|
| Level 1 | 5.5 frames | Frame 6 | Very tight, only fast moves |
| Level 2 | 11-13 frames | Frame 12-14 | Standard combo window |
| Level 3 | 18-21 frames | Frame 19-22 | Extended combo window |

### Landing Override

When characters land while in grounded hitstun with knockback below tumble threshold, hitstun is replaced by 4 frames of normal landing lag. This enables:
- **ASDI down**: Holding down to land faster and escape combos
- **Platform tech chasing**: Using platforms to extend or escape combos

---

## Sources

### Primary Sources (Quantitative)
- **Parquet replay data**: `/Users/eppie/ssb_wiki/fox_vs_fox_parquet/` - 3.47 million frames of grounded hitstun analyzed

### Documentation Sources (Qualitative)
- [SmashWiki - Hitstun](https://www.ssbwiki.com/Hitstun)
- [SmashWiki - Knockback](https://www.ssbwiki.com/Knockback)
- [libmelee Action State Enums](https://libmelee.readthedocs.io/en/latest/enums.html)
- Smashboards Physics Thread - Hitstun formula (KB × 0.4)
- Kadano's knockback/hitstun calculations

### Reference Data
- `action_state.json` - State ID to name mapping
