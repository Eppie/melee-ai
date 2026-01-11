# Aerial Hitstun & Tumble Action States (84-91)

This document describes the aerial hitstun and tumble action states in Super Smash Bros. Melee, covering states 84-91.

## Overview

When an airborne character is hit, they enter one of 8 aerial damage states. These states are divided into two categories:

1. **DAMAGE_AIR (84-86)**: Low-knockback aerial hitstun that doesn't cause tumbling
2. **DAMAGE_FLY (87-91)**: Tumble states from high-knockback hits

| Category | States | Description | Tumble? |
|----------|--------|-------------|---------|
| Aerial Hitstun | 84, 85, 86 | Light hits while airborne | No |
| Tumble (Directional) | 87, 88, 89 | Launched in specific direction | Yes |
| Tumble (Special) | 90, 91 | Strong launch / High-percent roll | Yes |

**Tumble Threshold**: A character enters tumble when receiving an attack that would cause **32+ frames of hitstun** before modifiers.

---

## Usage Statistics (Fox Dittos)

Total dataset: 288.5 million frames

| State | Name | Total Frames | % of Gameplay | Instances |
|-------|------|--------------|---------------|-----------|
| 90 | DAMAGE_FLY_TOP | 15,159,923 | **5.25%** | 214,129 |
| 88 | DAMAGE_FLY_N | 8,801,967 | 3.05% | 224,665 |
| 87 | DAMAGE_FLY_HI | 3,221,311 | 1.12% | 80,449 |
| 89 | DAMAGE_FLY_LW | 2,459,252 | 0.85% | 59,168 |
| 91 | DAMAGE_FLY_ROLL | 1,708,829 | 0.59% | 29,407 |
| 85 | DAMAGE_AIR_2 | 1,364,333 | 0.47% | 78,908 |
| 86 | DAMAGE_AIR_3 | 1,263,677 | 0.44% | 44,066 |
| 84 | DAMAGE_AIR_1 | 14,092 | 0.005% | 1,016 |

**Key Insight**: DAMAGE_FLY_TOP (state 90) is the single most common damage state in the game at 5.25% of all gameplay frames. Combined, aerial hitstun/tumble states account for **11.8%** of all gameplay in Fox dittos.

---

## State Categories

### Category 1: Aerial Hitstun (DAMAGE_AIR)

These states occur when a character is hit while airborne but the knockback is below tumble threshold. The character remains in hitstun but can act afterward without needing to tech.

| State | Name | Intensity | Median % | Avg Hitstun |
|-------|------|-----------|----------|-------------|
| 84 | DAMAGE_AIR_1 | Light | 7% | 5.2 frames |
| 85 | DAMAGE_AIR_2 | Medium | 22% | 12.5 frames |
| 86 | DAMAGE_AIR_3 | Heavy | 44% | 17.2 frames |

### Category 2: Tumble (DAMAGE_FLY)

These states occur when knockback exceeds tumble threshold. Character enters a spinning/flailing animation and must tech when landing or risk getting knocked down.

| State | Name | Direction | Median % | Avg Hitstun |
|-------|------|-----------|----------|-------------|
| 87 | DAMAGE_FLY_HI | Upward | 89% | 31.8 frames |
| 88 | DAMAGE_FLY_N | Neutral/Horizontal | 80% | 30.9 frames |
| 89 | DAMAGE_FLY_LW | Downward | 87% | 31.3 frames |
| 90 | DAMAGE_FLY_TOP | Strong launch | 61% | 33.7 frames |
| 91 | DAMAGE_FLY_ROLL | High-% roll | 116% | 38.5 frames |

---

## Duration Statistics

| State | Name | Min | P25 | Median | P75 | Max | Instances |
|-------|------|-----|-----|--------|-----|-----|-----------|
| 84 | DAMAGE_AIR_1 | 4 | 13 | 14 | 15 | 25 | 1,016 |
| 85 | DAMAGE_AIR_2 | 1 | 6 | 16 | 24 | 169 | 78,908 |
| 86 | DAMAGE_AIR_3 | 1 | 22 | 27 | 33 | 158 | 44,066 |
| 87 | DAMAGE_FLY_HI | 1 | 19 | 38 | 57 | 302 | 80,449 |
| 88 | DAMAGE_FLY_N | 1 | 6 | 32 | 57 | 376 | 224,665 |
| 89 | DAMAGE_FLY_LW | 1 | 20 | 39 | 58 | 254 | 59,168 |
| 90 | DAMAGE_FLY_TOP | 1 | 38 | 57 | 91 | 749 | 214,129 |
| 91 | DAMAGE_FLY_ROLL | 1 | 38 | 62 | 74 | 327 | 29,407 |

**Key Finding**: Tumble states have significantly longer durations (median 32-62 frames) compared to DAMAGE_AIR states (median 14-27 frames).

---

## Key Properties

| State | On Ground | In Hitstun | Avg Height | Avg Jumps |
|-------|-----------|------------|------------|-----------|
| 84 | 2.5% | 86.7% | 7.7 | 0.87 |
| 85 | 1.8% | 93.6% | 13.9 | 0.79 |
| 86 | 0.1% | 97.6% | 9.0 | 0.71 |
| 87 | 0% | 100% | 13.3 | 0.72 |
| 88 | 0% | 100% | 18.6 | 0.79 |
| 89 | 0% | 100% | 26.4 | 0.82 |
| 90 | 0% | 98.9% | 35.9 | 0.93 |
| 91 | 0% | 100% | 35.1 | 0.68 |

**Key Observations**:
- DAMAGE_FLY states are always airborne (0% on ground)
- Average height increases with state: FLY_TOP/ROLL at ~35 units vs DAMAGE_AIR at ~8-14 units
- Most characters retain their double jump (~0.7-0.9 jumps remaining on average)

---

## Detailed State Descriptions

### DAMAGE_AIR_1 (State 84) - Light Aerial Hit

**Internal Name**: `DAMAGE_AIR_1`

**Description**: Light hitstun while airborne. Very rare state (only 1,016 instances in dataset). Occurs from weak attacks hitting an airborne opponent at very low percent.

**Entry Conditions**:
| From State | Name | Occurrences |
|------------|------|-------------|
| 29 | FALL | 232 |
| 25 | JUMP_F | 172 |
| 24 | KNEE_BEND | 116 |
| 67 | ATTACK_AIR_B | 91 |
| 65 | ATTACK_AIR_N | 72 |

**Exit Conditions**:
| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 29 | FALL | 403 | 39.7% |
| 42 | LANDING | 123 | 12.1% |
| 252 | CLIFF_CATCH | 80 | 7.9% |
| 15 | WALK_SLOW | 66 | 6.5% |
| 27 | JUMP_AERIAL_F | 53 | 5.2% |

**Statistics**:
- Median duration: 14 frames
- Average percent: 7%
- On ground: 2.5%

---

### DAMAGE_AIR_2 (State 85) - Medium Aerial Hit

**Internal Name**: `DAMAGE_AIR_2`

**Description**: Medium hitstun while airborne. Common transition state between light hits and tumble. Often used in combo sequences.

**Entry Conditions** (from attacks):
| From State | Name | Occurrences |
|------------|------|-------------|
| 25 | JUMP_F (hit during jump) | 16,006 |
| 65 | ATTACK_AIR_N | 8,764 |
| 67 | ATTACK_AIR_B | 8,471 |
| 69 | ATTACK_AIR_LW (dair) | 6,384 |
| 27 | JUMP_AERIAL_F | 3,951 |
| 24 | KNEE_BEND | 3,662 |

**Exit Conditions**:
| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 42 | LANDING | 41,313 | 47.4% |
| 90 | DAMAGE_FLY_TOP (escalate to tumble) | 18,671 | 21.4% |
| 29 | FALL | 5,596 | 6.4% |
| 27 | JUMP_AERIAL_F | 2,472 | 2.8% |
| 39 | SQUAT | 2,460 | 2.8% |
| 354 | FOX UP-B | 1,985 | 2.3% |

**Statistics**:
- Median duration: 16 frames
- Average percent: 22%
- On ground: 1.8%

**Key Finding**: 21.4% of DAMAGE_AIR_2 exits escalate to DAMAGE_FLY_TOP, indicating multi-hit combos where the second hit causes tumble.

---

### DAMAGE_AIR_3 (State 86) - Heavy Aerial Hit

**Internal Name**: `DAMAGE_AIR_3`

**Description**: Heavy hitstun while airborne, just below tumble threshold. Often transitions into tumble if hit again.

**Exit Conditions**:
| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 42 | LANDING | 26,299 | 53.6% |
| 90 | DAMAGE_FLY_TOP | 9,549 | 19.5% |
| 29 | FALL | 2,330 | 4.7% |
| 27 | JUMP_AERIAL_F | 1,411 | 2.9% |
| 88 | DAMAGE_FLY_N | 844 | 1.7% |

**Statistics**:
- Median duration: 27 frames
- Average percent: 44%
- On ground: 0.1%

---

### DAMAGE_FLY_HI (State 87) - Upward Tumble

**Internal Name**: `DAMAGE_FLY_HI`

**Description**: Tumble state from upward-angle knockback. Character is launched upward and enters spinning animation. Most commonly caused by upward-hitting moves like up-smash, up-tilt, and shine.

**Entry Conditions** (from attacks):
| From State | Name | Occurrences |
|------------|------|-------------|
| 20 | DASH | 7,591 |
| 356 | FOX UP-B (Fire Fox hit) | 6,529 |
| 24 | KNEE_BEND | 4,362 |
| 212 | CATCH (grab release) | 4,299 |
| 25 | JUMP_F | 4,217 |
| 43 | LANDING_FALL_SPECIAL | 3,272 |
| 67 | ATTACK_AIR_B | 2,411 |

**Exit Conditions (Outcomes)**:
| Outcome | Occurrences | % |
|---------|-------------|---|
| Missed Tech (knockdown) | 30,354 | 32.9% |
| DAMAGE_FALL (tumble continues) | 22,941 | 24.8% |
| Successful Tech | 16,609 | 18.0% |
| Double Jump (escaped) | 12,545 | 13.6% |
| DEAD (KO) | 4,628 | 5.0% |
| Hit Again (combo) | 3,117 | 3.4% |

**Statistics**:
- Median duration: 38 frames
- Average percent: 89%
- Avg hitstun remaining: 31.8 frames

**Tech Rate**: 18.0% successful tech, 32.9% missed tech = **35.4% tech rate**

---

### DAMAGE_FLY_N (State 88) - Neutral/Horizontal Tumble

**Internal Name**: `DAMAGE_FLY_N`

**Description**: Tumble state from horizontal/neutral knockback angle. The most common tumble direction state. Character is launched sideways.

**What Attacks Cause This State**:

Analysis of Fox attacks causing tumble:
| Attack | to FLY_N | Total | % of Tumbles |
|--------|----------|-------|--------------|
| Bair (67) | 6,197 | 19,098 | 32.4% |
| Up Smash (63) | 6,629 | 11,532 | 57.5% |
| Nair (65) | 5,203 | 16,763 | 31.0% |
| Dair (69) | 4,214 | 12,300 | 34.3% |
| Up Tilt (56) | 3,748 | 9,309 | 40.3% |

**Exit Conditions (Outcomes)**:
| Outcome | Occurrences | % |
|---------|-------------|---|
| Missed Tech (knockdown) | 148,669 | 45.6% |
| Successful Tech | 62,685 | 19.2% |
| DAMAGE_FALL (continues) | 57,480 | 17.6% |
| Double Jump (escaped) | 33,237 | 10.2% |
| DEAD (KO) | 9,161 | 2.8% |
| Hit Again (combo) | 7,626 | 2.3% |

**Statistics**:
- Median duration: 32 frames
- Average percent: 80%
- Avg hitstun remaining: 30.9 frames

**Tech Rate**: 19.2% successful tech, 45.6% missed tech = **29.6% tech rate**

---

### DAMAGE_FLY_LW (State 89) - Downward Tumble

**Internal Name**: `DAMAGE_FLY_LW`

**Description**: Tumble state from downward-angle knockback. Character is launched diagonally downward. Often leads to stage spikes or tech situations on platforms.

**Exit Conditions (Outcomes)**:
| Outcome | Occurrences | % |
|---------|-------------|---|
| DAMAGE_FALL | 17,462 | 27.5% |
| Missed Tech | 19,063 | 30.0% |
| Double Jump (escaped) | 10,130 | 15.9% |
| Successful Tech | 10,826 | 17.0% |
| DEAD (KO) | 2,059 | 3.2% |

**Statistics**:
- Median duration: 39 frames
- Average percent: 87%
- Avg hitstun remaining: 31.3 frames

**Tech Rate**: 17.0% successful tech, 30.0% missed tech = **36.2% tech rate**

---

### DAMAGE_FLY_TOP (State 90) - Strong Launch

**Internal Name**: `DAMAGE_FLY_TOP`

**Description**: **The most common damage state** (5.25% of gameplay). Occurs from strong knockback hits, typically at the apex of launch trajectory. Used for the main body of tumble before transitioning to directional states or landing.

**Entry Conditions**:
Multiple sources - often from other tumble states or strong attacks:
| From State | Occurrences | Notes |
|------------|-------------|-------|
| 88 → 90 | 2,843 | Combo while tumbling |
| 85 → 90 | 18,671 | Escalation from DAMAGE_AIR |
| 86 → 90 | 9,549 | Escalation from heavy air hit |

**Exit Conditions (Outcomes)**:
| Outcome | Occurrences | % |
|---------|-------------|---|
| Missed Tech | 65,618 | 20.0% |
| Successful Tech | 91,088 | 27.7% |
| Hit Again (tumble) | 51,018 | 15.5% |
| Hit Again (DAMAGE_AIR) | 21,623 | 6.6% |
| Got Grabbed | 23,187 | 7.1% |
| DEAD (KO) | 23,488 | 7.2% |
| DAMAGE_FALL | 19,212 | 5.9% |
| Double Jump (escaped) | 18,869 | 5.7% |

**Statistics**:
- Median duration: 57 frames
- Average percent: 61%
- Avg hitstun remaining: 33.7 frames
- Average height: 35.9 units (highest of all states)

**Key Finding**: Characters in DAMAGE_FLY_TOP get hit again (combo extended) **22.1%** of the time, and grabbed 7.1%. This is the primary combo state in Fox dittos.

**Tech Rate**: 27.7% successful tech, 20.0% missed tech = **58.0% tech rate**

---

### DAMAGE_FLY_ROLL (State 91) - High-Percent Roll

**Internal Name**: `DAMAGE_FLY_ROLL`

**Description**: Special tumble animation that only occurs at **100%+ damage**. Character enters an extended rolling/spinning animation. Has the highest death rate of any tumble state.

**Entry Conditions**:
| From State | Occurrences | Notes |
|------------|-------------|-------|
| 90 | DAMAGE_FLY_TOP | 4,044 | 13.8% |
| 356 | FOX UP-B | 2,789 | Strong vertical hit |
| 358 | FOX UP-B | 1,971 | Fire Fox connection |
| 88 | DAMAGE_FLY_N | 424 | Combo chain |

**Exit Conditions (Outcomes)**:
| Outcome | Occurrences | % |
|---------|-------------|---|
| DAMAGE_FALL | 16,227 | 48.4% |
| Missed Tech | 6,069 | 18.1% |
| **DEAD (KO)** | 5,464 | **16.3%** |
| Successful Tech | 3,369 | 10.1% |
| Double Jump (escaped) | 1,071 | 3.2% |

**Statistics**:
- Median duration: 62 frames (longest of all tumble states)
- **Minimum percent: 100%** (only occurs at high damage)
- Median percent: 116%
- Avg hitstun remaining: 38.5 frames
- Avg jumps remaining: 0.68 (lowest - often used after DJ)

**Key Finding**: 16.3% death rate is more than double any other tumble state. DAMAGE_FLY_ROLL is the "kill tumble" - characters in this state are in serious danger.

**Tech Rate**: 10.1% successful tech, 18.1% missed tech = **35.8% tech rate**

---

## Tumble Direction vs Fox Attacks

Which Fox attacks cause which tumble directions?

| Attack | State | FLY_HI | FLY_N | FLY_LW | FLY_TOP | Total |
|--------|-------|--------|-------|--------|---------|-------|
| Bair | 67 | 12.6% | 32.4% | 17.3% | 37.7% | 19,098 |
| Nair | 65 | 10.9% | 31.0% | 32.3% | 25.7% | 16,763 |
| Dair | 69 | 7.1% | 34.3% | 30.4% | 28.2% | 12,300 |
| Up Smash | 63 | 17.9% | 57.5% | 5.9% | 18.7% | 11,532 |
| Up Tilt | 56 | 13.3% | 40.3% | 17.6% | 28.8% | 9,309 |
| Uair | 68 | 11.1% | 59.8% | 13.6% | 15.5% | 4,667 |
| Down Tilt | 57 | 31.0% | 43.9% | 0.4% | 24.7% | 3,803 |
| Shine (aerial) | 361 | 28.4% | 36.0% | 4.2% | 31.4% | 3,705 |
| Shine (ground) | 360 | 34.5% | 27.4% | 6.3% | 31.8% | 2,292 |

**Observations**:
- **Up Smash & Uair** send primarily to FLY_N (57-60%), despite being "up" attacks
- **Shine** sends primarily to FLY_HI (28-35%), enabling its combo potential
- **Nair & Dair** have significant FLY_LW percentage (30-32%), causing downward tumble

---

## Tumble State Chaining

Characters frequently transition between tumble states during combos:

| Transition | Occurrences | Pattern |
|------------|-------------|---------|
| 90 → 88 | 35,670 | FLY_TOP → FLY_N (hit again) |
| 90 → 89 | 8,900 | FLY_TOP → FLY_LW |
| 90 → 91 | 4,044 | FLY_TOP → FLY_ROLL (high %) |
| 88 → 90 | 2,843 | FLY_N → FLY_TOP |
| 90 → 87 | 2,404 | FLY_TOP → FLY_HI |
| 88 → 89 | 1,458 | FLY_N → FLY_LW |
| 87 → 88 | 1,338 | FLY_HI → FLY_N |

**Key Pattern**: DAMAGE_FLY_TOP (90) is the central combo state. It transitions into directional tumble states when the character is hit again during the combo.

---

## Tumble Outcomes Summary

| State | Tech Rate | Miss Rate | Death Rate | DJ Escape |
|-------|-----------|-----------|------------|-----------|
| 87 (FLY_HI) | 18.0% | 32.9% | 5.0% | 13.6% |
| 88 (FLY_N) | 19.2% | 45.6% | 2.8% | 10.2% |
| 89 (FLY_LW) | 17.0% | 30.0% | 3.2% | 15.9% |
| 90 (FLY_TOP) | 27.7% | 20.0% | 7.2% | 5.7% |
| 91 (FLY_ROLL) | 10.1% | 18.1% | **16.3%** | 3.2% |

**Key Findings**:
- **FLY_N has the highest miss rate** (45.6%) - horizontal knockback makes teching harder
- **FLY_TOP has the best tech rate** (27.7%) - more time to react and position
- **FLY_ROLL has the highest death rate** (16.3%) - kill confirm state
- **FLY_LW has highest DJ escape rate** (15.9%) - downward angle allows easier recovery

---

## State Flow Diagram

```
                    ┌──────────────────────────────────┐
                    │         HIT WHILE AIRBORNE       │
                    └────────────────┬─────────────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                │                    │                    │
      ┌─────────▼─────────┐  ┌───────▼───────┐  ┌────────▼────────┐
      │   Low Knockback   │  │    Medium     │  │ High Knockback  │
      │   (< 32f hitstun) │  │   Knockback   │  │ (≥ 32f hitstun) │
      └─────────┬─────────┘  └───────┬───────┘  └────────┬────────┘
                │                    │                    │
      ┌─────────▼─────────┐  ┌───────▼───────┐  ┌────────▼────────┐
      │    DAMAGE_AIR     │  │  DAMAGE_AIR   │  │   DAMAGE_FLY    │
      │   84 / 85 / 86    │  │  85 / 86      │  │  (TUMBLE)       │
      │   (no tumble)     │  │  (transition) │  │                 │
      └────────┬──────────┘  └───────┬───────┘  └────────┬────────┘
               │                     │                    │
               ▼                     ▼                    │
         ┌──────────┐          ┌──────────┐               │
         │  LANDING │          │  Escalate│               │
         │   (42)   │          │  to 90   │               │
         │  47-54%  │          │  21%     │               │
         └──────────┘          └────┬─────┘               │
                                    │                     │
                    ┌───────────────┼─────────────────────┘
                    │               │
                    ▼               ▼
              ┌───────────────────────────────────────┐
              │           DAMAGE_FLY_TOP (90)          │
              │     Central tumble state - 5.25%       │
              │   Median 57 frames, avg height 35.9    │
              └─────────────────┬─────────────────────┘
                                │
          ┌─────────┬───────────┼───────────┬─────────┐
          │         │           │           │         │
          ▼         ▼           ▼           ▼         ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ FLY_HI   │ │ FLY_N    │ │ FLY_LW   │ │ FLY_ROLL │ │ DAMAGE_  │
    │  (87)    │ │  (88)    │ │  (89)    │ │  (91)    │ │  FALL(38)│
    │ Upward   │ │ Neutral  │ │ Downward │ │ 100%+    │ │ Continue │
    │  1.12%   │ │  3.05%   │ │  0.85%   │ │  0.59%   │ │          │
    └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────────┘
         │            │            │            │
         └────────────┴────────────┴────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │  TECH    │    │  MISSED  │    │   DEAD   │
        │ (199-201)│    │  TECH    │    │  (0-4)   │
        │  ~18-28% │    │ (183/191)│    │   2-16%  │
        └──────────┘    │  ~20-46% │    └──────────┘
                        └──────────┘
```

---

## Practical Implications

### For Combo Game
- **DAMAGE_FLY_TOP (90)** is where combos happen - 22.1% chance of extending
- Characters are grabbed 7.1% of the time during FLY_TOP
- Average 0.93 jumps remaining means double jump is usually available for escape

### For Defense
- FLY_N has highest missed tech rate (45.6%) - focus on DI
- FLY_TOP has best tech rate (27.7%) - more time to prepare
- FLY_ROLL means you're likely dying (16.3% death rate)

### For Edgeguarding
- FLY_LW (downward) has 15.9% DJ escape - high recovery rate
- FLY_ROLL characters often have used their DJ (0.68 jumps remaining)

---

## Sources

### Primary Sources (Quantitative)
- **Parquet replay data**: `/Users/eppie/ssb_wiki/fox_vs_fox_parquet/` - 33.9 million frames of aerial hitstun analyzed

### Documentation Sources (Qualitative)
- [SmashWiki - Tumble](https://www.ssbwiki.com/Tumble)
- [SmashWiki - Hitstun](https://www.ssbwiki.com/Hitstun)
- Smashboards Physics Thread - 32 frame tumble threshold
- [libmelee Action State Enums](https://libmelee.readthedocs.io/en/latest/enums.html)

### Reference Data
- `action_state.json` - State ID to name mapping
