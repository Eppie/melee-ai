# Aerial Attacks & Landing Lag Action States (65-74)

This document describes the aerial attack and landing lag action states in Super Smash Bros. Melee, covering states 65-74.

## Overview

Aerial attacks are fundamental to Melee's combat system, particularly for combo-heavy characters like Fox. Each of the 5 aerial attacks has a corresponding landing lag state that occurs when the attack lands before completing. L-canceling (pressing L, R, or Z within 7 frames before landing) halves this landing lag.

| Attack | Attack State | Landing State | L-Cancel Lag | Normal Lag |
|--------|--------------|---------------|--------------|------------|
| Neutral Air (Nair) | 65 | 70 | 7 frames | 15 frames |
| Forward Air (Fair) | 66 | 71 | 11 frames | 22 frames |
| Back Air (Bair) | 67 | 72 | 9 frames | 18 frames |
| Up Air (Uair) | 68 | 73 | 9 frames | 18 frames |
| Down Air (Dair) | 69 | 74 | 9 frames | 18 frames |

---

## Usage Statistics (Fox Dittos)

Total dataset: 288.5 million frames

| State | Name | Total Frames | % of Gameplay | Instances |
|-------|------|--------------|---------------|-----------|
| 67 | ATTACK_AIR_B (Bair) | 9,447,482 | 3.27% | 427,247 |
| 65 | ATTACK_AIR_N (Nair) | 7,005,965 | 2.43% | 385,857 |
| 69 | ATTACK_AIR_LW (Dair) | 5,444,394 | 1.89% | 315,016 |
| 68 | ATTACK_AIR_HI (Uair) | 4,069,288 | 1.41% | 175,040 |
| 66 | ATTACK_AIR_F (Fair) | 633,099 | 0.22% | 32,434 |
| 72 | LANDING_AIR_B | 3,442,120 | 1.19% | 319,067 |
| 74 | LANDING_AIR_LW | 2,789,857 | 0.97% | 295,125 |
| 70 | LANDING_AIR_N | 2,674,713 | 0.93% | 361,439 |
| 73 | LANDING_AIR_HI | 1,305,235 | 0.45% | 137,000 |
| 71 | LANDING_AIR_F | 333,204 | 0.12% | 28,115 |

**Key Insight**: Back air (bair) is Fox's most used aerial by a significant margin, accounting for 3.27% of all gameplay frames. It's used ~13x more than forward air, reflecting its superior speed, range, and combo utility.

---

## Aerial Attack States (65-69)

### ATTACK_AIR_N (State 65) - Neutral Aerial (Nair)

**Internal Name**: `ATTACK_AIR_N`

**Description**: Fox's sex kick - a quick knee strike that starts strong and weakens over time. One of Fox's fastest aerials with wide utility for combos, approaching, and neutral. The hitbox weakens after frame 7.

**Frame Data** (Fox):

| Property | Value |
|----------|-------|
| Total Frames | 35 |
| Hit Frames | 4-31 |
| Strong Hit | 4-7 (12% damage, 80 BKB) |
| Weak Hit | 8-31 (9% damage, 0 BKB) |
| Autocancel | <3, >32 |
| IASA | Frame 32 |
| Landing Lag | 15 (L-cancel: 7) |

**Entry Conditions**:

| From State | Name | Occurrences | Notes |
|------------|------|-------------|-------|
| 25 | JUMP_F (first jump forward) | 331,907 | 83.2% - SHFFL nairs |
| 27 | JUMP_AERIAL_F (double jump) | 22,889 | 5.7% |
| 24 | KNEE_BEND (jumpsquat) | 17,761 | 4.5% - frame-perfect timing |
| 29 | FALL | 9,248 | 2.3% |
| 26 | JUMP_B (first jump backward) | 7,073 | 1.8% |
| 244 | PASS (platform drop) | 4,655 | 1.2% - platform nairs |

**Exit Conditions**:

| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 70 | LANDING_AIR_N | 361,341 | 90.9% |
| 85 | DAMAGE_AIR_2 (got hit) | 8,764 | 2.2% |
| 89 | DAMAGE_FLY_N (launched) | 5,411 | 1.4% |
| 88 | DAMAGE_FLY_HI (launched up) | 5,203 | 1.3% |
| 90 | DAMAGE_FLY_LW (launched down) | 4,315 | 1.1% |
| 42 | LANDING (autocanceled) | 2,283 | 0.6% |

**Duration Statistics**:

| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 17 frames |
| Average | 18.16 frames |
| Max | 134 frames |

**Key Properties**:
- **On Ground**: 0% (always airborne)
- **Invulnerable**: 5.79% (from ledgedash invincibility)
- **Fastfalling**: 19.88%
- **In Hitstun**: 0%

**Technical Notes**: Nair is Fox's fastest aerial to come out and his best out-of-shield option when combined with shine. The weak hit can set up shine follow-ups at low-mid percent. 90.9% of nairs land in LANDING_AIR_N, with only 0.6% autocanceling (landing after frame 32).

---

### ATTACK_AIR_F (State 66) - Forward Aerial (Fair)

**Internal Name**: `ATTACK_AIR_F`

**Description**: A multi-hit forward kick that covers a large arc in front of Fox. Less commonly used in competitive play due to its high landing lag and situational utility, but can be used for gimping and edgeguards.

**Frame Data** (Fox):

| Property | Value |
|----------|-------|
| Total Frames | 59 |
| Hit Frames | 7-27 (5 hits) |
| Damage | 2%+2%+2%+2%+3% = 11% |
| Autocancel | <4, >39 |
| IASA | Frame 40 |
| Landing Lag | 22 (L-cancel: 11) |

**Entry Conditions**:

| From State | Name | Occurrences | Notes |
|------------|------|-------------|-------|
| 25 | JUMP_F | 14,264 | 43.5% |
| 27 | JUMP_AERIAL_F | 6,115 | 18.7% |
| 29 | FALL | 3,715 | 11.3% |
| 24 | KNEE_BEND | 3,076 | 9.4% |
| 244 | PASS (platform drop) | 3,075 | 9.4% |
| 26 | JUMP_B | 1,198 | 3.7% |

**Exit Conditions**:

| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 71 | LANDING_AIR_F | 28,115 | 86.6% |
| 85 | DAMAGE_AIR_2 | 1,389 | 4.3% |
| 88 | DAMAGE_FLY_HI | 1,028 | 3.2% |
| 90 | DAMAGE_FLY_LW | 847 | 2.6% |
| 89 | DAMAGE_FLY_N | 545 | 1.7% |

**Duration Statistics**:

| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 18 frames |
| Average | 19.52 frames |
| Max | 93 frames |

**Key Properties**:
- **Invulnerable**: 7.78%
- **Fastfalling**: 19.28%

**Technical Notes**: Fair is Fox's least used aerial in high-level play (only 0.22% of gameplay). Its high landing lag (22 frames, 11 L-canceled) and multi-hit nature make it risky. Used primarily for edgeguarding low recoveries or as a mixup.

---

### ATTACK_AIR_B (State 67) - Back Aerial (Bair)

**Internal Name**: `ATTACK_AIR_B`

**Description**: Fox's premier aerial attack - a quick horizontal kick behind him with excellent range, speed, and knockback. The backbone of Fox's neutral game and combo extensions. Can wall out opponents and secure kills at high percent.

**Frame Data** (Fox):

| Property | Value |
|----------|-------|
| Total Frames | 38 |
| Hit Frames | 4-19 |
| Strong Hit | 4-8 (15% damage) |
| Weak Hit | 9-19 (9% damage) |
| Autocancel | <2, >18 |
| IASA | Frame 36 |
| Landing Lag | 18 (L-cancel: 9) |

**Entry Conditions**:

| From State | Name | Occurrences | Notes |
|------------|------|-------------|-------|
| 26 | JUMP_B (first jump backward) | 114,618 | 25.5% |
| 28 | JUMP_AERIAL_B (DJ backward) | 95,720 | 21.3% |
| 27 | JUMP_AERIAL_F (DJ forward) | 71,295 | 15.9% |
| 29 | FALL | 41,296 | 9.2% |
| 25 | JUMP_F (first jump forward) | 40,458 | 9.0% |
| 244 | PASS (platform drop) | 32,199 | 7.2% |
| 24 | KNEE_BEND | 23,162 | 5.2% |

**Exit Conditions**:

| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 72 | LANDING_AIR_B | 319,053 | 62.7% |
| 42 | LANDING (autocanceled) | 143,682 | 28.2% |
| 29 | FALL | 9,581 | 1.9% |
| 85 | DAMAGE_AIR_2 | 8,471 | 1.7% |
| 90 | DAMAGE_FLY_LW | 7,195 | 1.4% |

**Duration Statistics**:

| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 17 frames |
| Average | 22.11 frames |
| Max | 270 frames |

**Key Properties**:

- **Invulnerable**: 8.57%
- **Fastfalling**: 26.34%

**Technical Notes**: Bair has the highest autocancel rate of Fox's aerials (28.2% land in LANDING instead of LANDING_AIR_B). This reflects the ease of timing autocanceled bairs, which have only 4 frames of normal landing lag. The high fastfall percentage (26.34%) indicates aggressive SHFFL usage.

---

### ATTACK_AIR_HI (State 68) - Up Aerial (Uair)

**Internal Name**: `ATTACK_AIR_HI`

**Description**: A flip kick above Fox's head with excellent vertical range. Essential for juggling, combo extensions, and killing off the top. One of Fox's most important combo tools.

**Frame Data** (Fox):

| Property | Value |
|----------|-------|
| Total Frames | 38 |
| Hit Frames | 6-17 |
| Strong Hit | 6-10 (13% damage) |
| Weak Hit | 11-17 (10% damage) |
| Autocancel | <4, >24 |
| IASA | Frame 35 |
| Landing Lag | 18 (L-cancel: 9) |

**Entry Conditions**:

| From State | Name | Occurrences | Notes |
|------------|------|-------------|-------|
| 27 | JUMP_AERIAL_F | 61,091 | 33.0% |
| 25 | JUMP_F | 50,466 | 27.3% |
| 28 | JUMP_AERIAL_B | 28,318 | 15.3% |
| 29 | FALL | 25,099 | 13.6% |
| 26 | JUMP_B | 6,124 | 3.3% |
| 24 | KNEE_BEND | 5,604 | 3.0% |
| 244 | PASS (platform drop) | 4,408 | 2.4% |

**Exit Conditions**:

| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 73 | LANDING_AIR_HI | 136,820 | 78.1% |
| 42 | LANDING (autocanceled) | 37,680 | 21.5% |
| 29 | FALL | 4,768 | 2.7% |
| 88 | DAMAGE_FLY_HI | 2,792 | 1.6% |
| 85 | DAMAGE_AIR_2 | 2,698 | 1.5% |

**Duration Statistics**:

| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 20 frames |
| Average | 23.25 frames |
| Max | 371 frames |

**Key Properties**:
- **Invulnerable**: 13.61% (highest of all aerials)
- **Fastfalling**: 23.78%

**Technical Notes**: Uair has the highest invulnerability rate (13.61%) because it's frequently used as a follow-up after ledgedash or during juggle sequences where invincibility carries over. 33% of uairs come from double jumps, indicating its use in extended aerial combos.

---

### ATTACK_AIR_LW (State 69) - Down Aerial (Dair)

**Internal Name**: `ATTACK_AIR_LW`

**Description**: Fox's "drill" - a multi-hit spinning kick that pops opponents up for follow-ups. One of Fox's best approach tools and combo starters. The final hit's upward knockback leads into shine, up-smash, or grab.

**Frame Data** (Fox):
| Property | Value |
|----------|-------|
| Total Frames | 49 |
| Hit Frames | 5-24 (7 hits) |
| Damage | 2%×6 + 3% = 15% (all hits) |
| Autocancel | <4, >38 |
| IASA | Frame 39 |
| Landing Lag | 18 (L-cancel: 9) |

**Entry Conditions**:
| From State | Name | Occurrences | Notes |
|------------|------|-------------|-------|
| 25 | JUMP_F | 195,478 | 60.9% |
| 27 | JUMP_AERIAL_F | 37,124 | 11.6% |
| 244 | PASS (platform drop) | 36,498 | 11.4% |
| 24 | KNEE_BEND | 30,161 | 9.4% |
| 26 | JUMP_B | 13,461 | 4.2% |
| 28 | JUMP_AERIAL_B | 4,095 | 1.3% |

**Exit Conditions**:
| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 74 | LANDING_AIR_LW | 294,931 | 90.6% |
| 85 | DAMAGE_AIR_2 | 6,384 | 2.0% |
| 42 | LANDING (autocanceled) | 5,753 | 1.8% |
| 88 | DAMAGE_FLY_HI | 4,214 | 1.3% |
| 89 | DAMAGE_FLY_N | 3,742 | 1.1% |

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 16 frames |
| Average | 17.28 frames |
| Max | 142 frames |

**Key Properties**:
- **Invulnerable**: 5.85%
- **Fastfalling**: 25.29%

**Technical Notes**: Dair has the highest entry rate from first jump (60.9% from JUMP_F), confirming its role as Fox's primary approach aerial. 90.6% land in LANDING_AIR_LW with only 1.8% autocanceling, indicating dair is almost always used as an active approach tool rather than retreating/spacing.

---

## Landing Lag States (70-74)

### LANDING_AIR_N (State 70) - Nair Landing Lag

**Internal Name**: `LANDING_AIR_N`

**Description**: The landing lag state after performing a neutral aerial that doesn't autocancel. Duration is 15 frames normally, 7 frames when L-canceled.

**Entry Conditions**:
| From State | Name | Occurrences |
|------------|------|-------------|
| 65 | ATTACK_AIR_N | 361,341 |

**Exit Conditions** (top 10):
| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 14 | WAIT | 164,450 | 45.5% |
| 39 | SQUAT (crouch) | 59,733 | 16.5% |
| 178 | GUARD_ON (shield) | 38,110 | 10.5% |
| 18 | TURN | 30,841 | 8.5% |
| 20 | DASH | 18,296 | 5.1% |
| 360 | Fox SHINE_TURN | 7,569 | 2.1% |
| 15 | WALK_SLOW | 6,558 | 1.8% |
| 88 | DAMAGE_FLY_HI (got hit) | 5,191 | 1.4% |

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 7 frames |
| Average | 7.4 frames |
| Max | 19 frames |

**L-Cancel Analysis**:
| Duration Range | Count | Interpretation |
|----------------|-------|----------------|
| 6-8 frames | 140,251 | L-canceled (expected: 7) |
| 13-16 frames | 50,593 | Not L-canceled (expected: 15) |

**L-Cancel Rate**: **73.5%**

**Key Properties**:
- **On Ground**: 100%
- **Invulnerable**: 5.14%

---

### LANDING_AIR_F (State 71) - Fair Landing Lag

**Internal Name**: `LANDING_AIR_F`

**Description**: Landing lag after forward aerial. The longest landing lag of Fox's aerials at 22 frames (11 L-canceled).

**Entry Conditions**:
| From State | Name | Occurrences |
|------------|------|-------------|
| 66 | ATTACK_AIR_F | 28,115 |

**Exit Conditions** (top 10):
| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 14 | WAIT | 10,553 | 37.5% |
| 178 | GUARD_ON | 4,099 | 14.6% |
| 18 | TURN | 3,451 | 12.3% |
| 39 | SQUAT | 2,574 | 9.2% |
| 88 | DAMAGE_FLY_HI | 1,750 | 6.2% |
| 90 | DAMAGE_FLY_LW | 1,422 | 5.1% |
| 20 | DASH | 1,293 | 4.6% |

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 11 frames |
| Average | 11.85 frames |
| Max | 25 frames |

**L-Cancel Analysis**:
| Duration Range | Count | Interpretation |
|----------------|-------|----------------|
| 10-12 frames | ~18,000 | L-canceled (expected: 11) |
| 20-23 frames | ~4,000 | Not L-canceled (expected: 22) |

**L-Cancel Rate**: ~**82%**

**Key Properties**:
- **On Ground**: 100%
- **Invulnerable**: 3.79%

**Note**: Despite being Fox's least used aerial, fair has a high L-cancel rate, suggesting players who use it know it's risky and focus on execution.

---

### LANDING_AIR_B (State 72) - Bair Landing Lag

**Internal Name**: `LANDING_AIR_B`

**Description**: Landing lag after back aerial. 18 frames normally, 9 frames L-canceled.

**Entry Conditions**:
| From State | Name | Occurrences |
|------------|------|-------------|
| 67 | ATTACK_AIR_B | 319,053 |

**Exit Conditions** (top 10):
| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 14 | WAIT | 144,179 | 45.2% |
| 39 | SQUAT | 42,816 | 13.4% |
| 178 | GUARD_ON | 41,706 | 13.1% |
| 18 | TURN | 27,466 | 8.6% |
| 20 | DASH | 15,076 | 4.7% |
| 88 | DAMAGE_FLY_HI | 9,196 | 2.9% |
| 360 | Fox SHINE_TURN | 5,682 | 1.8% |
| 15 | WALK_SLOW | 5,405 | 1.7% |

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 10 frames |
| Average | 10.79 frames |
| Max | 26 frames |

**L-Cancel Analysis**:
| Duration Range | Count | Interpretation |
|----------------|-------|----------------|
| 8-10 frames | ~150,000 | L-canceled (expected: 9) |
| 16-19 frames | ~45,000 | Not L-canceled (expected: 18) |

**L-Cancel Rate**: ~**77%**

**Key Properties**:
- **On Ground**: 100%
- **Invulnerable**: 4.37%

---

### LANDING_AIR_HI (State 73) - Uair Landing Lag

**Internal Name**: `LANDING_AIR_HI`

**Description**: Landing lag after up aerial. 18 frames normally, 9 frames L-canceled.

**Entry Conditions**:
| From State | Name | Occurrences |
|------------|------|-------------|
| 68 | ATTACK_AIR_HI | 136,820 |

**Exit Conditions** (top 10):
| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 14 | WAIT | 62,115 | 45.4% |
| 39 | SQUAT | 22,620 | 16.5% |
| 178 | GUARD_ON | 14,496 | 10.6% |
| 18 | TURN | 11,169 | 8.2% |
| 20 | DASH | 6,826 | 5.0% |
| 360 | Fox SHINE_TURN | 3,611 | 2.6% |
| 88 | DAMAGE_FLY_HI | 2,914 | 2.1% |

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 9 frames |
| Average | 9.53 frames |
| Max | 22 frames |

**L-Cancel Analysis**:
| Duration Range | Count | Interpretation |
|----------------|-------|----------------|
| 8-10 frames | ~85,000 | L-canceled (expected: 9) |
| 16-19 frames | ~18,000 | Not L-canceled (expected: 18) |

**L-Cancel Rate**: ~**82.5%**

**Key Properties**:
- **On Ground**: 100%
- **Invulnerable**: 13.3% (highest of landing lags)

**Note**: Uair landing lag has the highest invulnerability rate (13.3%) because uairs often follow ledgedashes or occur during juggle combos where ledge invincibility carries over.

---

### LANDING_AIR_LW (State 74) - Dair Landing Lag

**Internal Name**: `LANDING_AIR_LW`

**Description**: Landing lag after down aerial (drill). 18 frames normally, 9 frames L-canceled. Critical state for Fox's drill-shine combos.

**Entry Conditions**:
| From State | Name | Occurrences |
|------------|------|-------------|
| 69 | ATTACK_AIR_LW | 294,931 |

**Exit Conditions** (top 10):
| To State | Name | Occurrences | % |
|----------|------|-------------|---|
| 360 | Fox SHINE_TURN | 101,608 | 34.4% |
| 14 | WAIT | 52,584 | 17.8% |
| 212 | CATCH (grab) | 35,206 | 11.9% |
| 39 | SQUAT | 28,609 | 9.7% |
| 178 | GUARD_ON | 19,177 | 6.5% |
| 18 | TURN | 14,632 | 5.0% |
| 88 | DAMAGE_FLY_HI | 8,614 | 2.9% |
| 20 | DASH | 8,498 | 2.9% |
| 63 | ATTACK_HI_4 (upsmash) | 5,925 | 2.0% |

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 9 frames |
| Average | 9.45 frames |
| Max | 21 frames |

**L-Cancel Analysis**:
| Duration Range | Count | Interpretation |
|----------------|-------|----------------|
| 8-10 frames | ~140,000 | L-canceled (expected: 9) |
| 16-19 frames | ~38,000 | Not L-canceled (expected: 18) |

**L-Cancel Rate**: ~**78.5%**

**Key Properties**:
- **On Ground**: 100%
- **Invulnerable**: 2.56%

**Critical Finding - Drill Follow-ups**:
The dair landing lag exit data reveals Fox's drill combo game:
- **34.4% → SHINE_TURN (360)**: Drill-shine is Fox's most common follow-up
- **11.9% → CATCH (212)**: Drill-grab is a key mixup
- **2.0% → ATTACK_HI_4 (63)**: Drill-upsmash kills

This distribution shows that **46.3% of drill landings lead to an offensive follow-up** (shine, grab, or upsmash), confirming dair's role as Fox's primary combo starter.

---

## L-Cancel Rate Summary

| Aerial | L-Cancel Rate | Notes |
|--------|---------------|-------|
| Nair | 73.5% | High consistency |
| Fair | ~82% | Best rate despite low usage |
| Bair | ~77% | Most used aerial |
| Uair | ~82.5% | Highest L-cancel rate |
| Dair | ~78.5% | Critical for combos |

**Overall L-Cancel Rate**: ~**77%**

This high L-cancel rate (77%) in Fox dittos reflects the technical demands of high-level Melee play.

---

## Autocancel Analysis

| Aerial | Landing in LANDING (42) | Autocancel Rate |
|--------|-------------------------|-----------------|
| Bair | 143,682 | 28.2% |
| Uair | 37,680 | 21.5% |
| Dair | 5,753 | 1.8% |
| Nair | 2,283 | 0.6% |
| Fair | ~300 | <1% |

**Key Insight**: Bair has by far the highest autocancel rate (28.2%) due to its wide autocancel window (<2, >18). Players actively time bair autocancels for minimal landing lag (4 frames vs 9 L-canceled). Dair almost never autocancels because it's used as an approach/combo tool requiring the hitbox to connect.

---

## State Flow Diagram

```
                ┌─────────────────────────────────────────┐
                │           AERIAL INITIATION              │
                │  (from jumps, falls, platform drops)     │
                └──────────────┬──────────────────────────┘
                               │
        ┌──────────┬───────────┼───────────┬──────────┐
        ▼          ▼           ▼           ▼          ▼
   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
   │  NAIR  │ │  FAIR  │ │  BAIR  │ │  UAIR  │ │  DAIR  │
   │  (65)  │ │  (66)  │ │  (67)  │ │  (68)  │ │  (69)  │
   │  2.4%  │ │  0.2%  │ │  3.3%  │ │  1.4%  │ │  1.9%  │
   └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
       │          │          │          │          │
       │          │     ┌────┴─────┐    │          │
       │          │     │  28.2%   │    │          │
       │          │     ▼autocancel│    │          │
       │          │  ┌──────────┐  │    │          │
       │          │  │ LANDING  │  │    │          │
       │          │  │   (42)   │  │    │          │
       │          │  │ 4f lag   │  │    │          │
       │          │  └────┬─────┘  │    │          │
       │          │       │        │    │          │
       ▼          ▼       │        ▼    ▼          ▼
   ┌────────┐ ┌────────┐  │   ┌────────┐ ┌────────┐ ┌────────┐
   │LAND_N  │ │LAND_F  │  │   │LAND_B  │ │LAND_HI │ │LAND_LW │
   │  (70)  │ │  (71)  │  │   │  (72)  │ │  (73)  │ │  (74)  │
   │7/15f   │ │11/22f  │  │   │9/18f   │ │9/18f   │ │9/18f   │
   │ 73.5%  │ │  82%   │  │   │  77%   │ │ 82.5%  │ │ 78.5%  │
   │L-cancel│ │L-cancel│  │   │L-cancel│ │L-cancel│ │L-cancel│
   └───┬────┘ └───┬────┘  │   └───┬────┘ └───┬────┘ └───┬────┘
       │          │       │       │          │          │
       └──────────┴───────┴───────┴──────────┴──────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
           ┌────────┐    ┌────────┐    ┌─────────────┐
           │  WAIT  │    │ SQUAT  │    │ SHINE_TURN  │
           │  (14)  │    │  (39)  │    │    (360)    │
           │  45%   │    │  12%   │    │ 34% (dair)  │
           └────────┘    └────────┘    └─────────────┘
```

---

## SHFFL Patterns

The Short Hop Fast Fall L-cancel (SHFFL) is fundamental to Fox's movement. The data shows:

1. **Jump → Aerial → Fastfall → L-cancel → Follow-up**
   - Average fastfall rate during aerials: 23%
   - Average L-cancel rate: 77%

2. **Most Common SHFFL Sequences** (Fox):
   - SHFFL Nair → Shine: 45.5% wait + crouch indicates combo resets
   - SHFFL Bair → Autocancel or L-cancel: 28% autocancel rate
   - SHFFL Dair → Shine: 34.4% drill-shine execution

---

## Sources

### Primary Sources (Quantitative)
- **Parquet replay data**: `/Users/eppie/ssb_wiki/fox_vs_fox_parquet/` - 37.1 million frames of aerial state data analyzed

### Documentation Sources (Qualitative)
- [SmashWiki - Fox (SSBM) Aerial Attacks](https://www.ssbwiki.com/Fox_(SSBM)#Aerial_attacks)
- [SmashWiki - L-canceling](https://www.ssbwiki.com/L-canceling)
- [SmashWiki - Autocancel](https://www.ssbwiki.com/Autocancel)
- [libmelee Action State Enums](https://libmelee.readthedocs.io/en/latest/enums.html)

### Reference Data
- `action_state.json` - State ID to name mapping
