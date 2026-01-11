# Crouch & Landing Action States (39-43)

This document describes the crouch and landing action states in Super Smash Bros. Melee, covering states 39-43.

## Overview

These states handle two fundamental grounded mechanics: crouching (a defensive/offensive tool) and landing (the transition from airborne to grounded). Understanding these states is crucial for advanced techniques like crouch-canceling, wavedashing, and platform movement.

| State ID | Name | Description | Fox Duration |
|----------|------|-------------|--------------|
| 39 | SQUAT | Crouch entry animation | ~7 frames |
| 40 | SQUAT_WAIT | Holding crouch | Variable |
| 41 | SQUAT_RV | Stand from crouch | ~10 frames |
| 42 | LANDING | Normal landing lag | 4 frames |
| 43 | LANDING_FALL_SPECIAL | Wavedash/helpless landing | 10 frames |

**Note**: States 70-74 (LANDING_AIR_N/F/B/HI/LW) handle aerial attack landing lag separately and are documented in the Aerial Attacks group.

---

## Crouch States

### SQUAT (State 39) - Crouch Entry

**Internal Name**: `SQUAT`

**Description**: The animation when a character begins crouching. This is the transition from standing/moving to the crouch position. During this state, the character's hurtbox begins lowering.

**Duration Statistics** (Fox dittos):
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 3 frames |
| Average | 3.8 frames |
| Most Common | 7 frames (32,233 occurrences) |
| Instances | 486,309 |
| Total Frames | 1,849,824 (0.64% of all frames) |

**Entry Conditions** (top 10):
| From State | Name | Occurrences |
|------------|------|-------------|
| 23 | RUN_BRAKE | 111,854 |
| 74 | LANDING_AIR_LW (dair landing) | 77,099 |
| 14 | WAIT | 70,800 |
| 70 | LANDING_AIR_N (nair landing) | 59,733 |
| 43 | LANDING_FALL_SPECIAL | 51,518 |
| 72 | LANDING_AIR_B (bair landing) | 32,540 |
| 235 | ESCAPE_B (roll back) | 23,988 |
| 199 | PASSIVE (tech) | 12,422 |
| 73 | LANDING_AIR_HI (uair landing) | 11,250 |
| 15 | WALK_SLOW | 9,942 |

**Exit Conditions** (top 15):
| To State | Name | Occurrences | Percentage |
|----------|------|-------------|------------|
| 360 | REFLECTOR_GROUND_STARTUP (Shine) | 328,499 | **58.19%** |
| 40 | SQUAT_WAIT (hold crouch) | 56,316 | 9.98% |
| 24 | KNEE_BEND (jumpsquat) | 42,726 | 7.57% |
| 244 | PASS (platform drop) | 37,092 | 6.57% |
| 57 | ATTACK_LW_3 (dtilt) | 30,183 | 5.35% |
| 41 | SQUAT_RV (stand up) | 22,063 | 3.91% |
| 80 | DAMAGE_N_2 (hit) | 7,800 | 1.38% |
| 226 | CAPTURE_PULLED_HI (grabbed) | 5,756 | 1.02% |
| 178 | GUARD_ON (shield) | 3,559 | 0.63% |
| 182 | GUARD_REFLECT (powershield) | 3,334 | 0.59% |

**Key Properties**:
- **On ground**: 100%
- **Invulnerability**: 3.35% (from ledgedash/tech intangibility carrying over)
- **Hitstun**: 0%
- **Jumps remaining**: 2 (full jump available)

**Key Finding**: **58% of SQUAT exits go directly to Shine (state 360)**. This demonstrates that crouch-to-shine is the dominant use case for crouching in competitive Fox play. Crouch enables the fastest possible shine by canceling other animations.

**Available Actions from SQUAT**:
- Shine (down-B) - Most common
- Jump (leads to jumpsquat)
- Platform drop-through
- Down tilt
- Down smash
- Shield
- Continue to full crouch

---

### SQUAT_WAIT (State 40) - Crouch Hold

**Internal Name**: `SQUAT_WAIT`

**Description**: The sustained crouching state. Character maintains lowered hurtbox. Used for crouch-canceling attacks, waiting in crouch, or setting up crouch-based options.

**Duration Statistics** (Fox dittos):
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 4 frames |
| Average | 5.89 frames |
| Max | 356 frames |
| Instances | 140,415 |
| Total Frames | 827,287 (0.29% of all frames) |

**Duration Distribution** (short durations):
| Duration | Occurrences |
|----------|-------------|
| 1 frame | 21,434 |
| 2 frames | 18,663 |
| 3 frames | 14,701 |
| 4 frames | 11,138 |
| 5 frames | 8,879 |

Most SQUAT_WAIT instances are very short (1-5 frames), indicating players use crouch briefly before transitioning to another action.

**Entry Conditions**:
| From State | Name | Occurrences |
|------------|------|-------------|
| 42 | LANDING | 81,296 |
| 39 | SQUAT | 56,316 |
| 57 | ATTACK_LW_3 (dtilt) | 3,246 |
| 43 | LANDING_FALL_SPECIAL | 263 |

**Exit Conditions** (top 10):
| To State | Name | Occurrences |
|----------|------|-------------|
| 360 | REFLECTOR_GROUND_STARTUP (Shine) | 49,248 |
| 24 | KNEE_BEND (jumpsquat) | 27,429 |
| 41 | SQUAT_RV (stand up) | 26,183 |
| 57 | ATTACK_LW_3 (dtilt) | 10,555 |
| 80 | DAMAGE_N_2 (hit) | 4,834 |
| 20 | DASH | 3,628 |
| 182 | GUARD_REFLECT (powershield) | 2,802 |
| 178 | GUARD_ON (shield) | 2,725 |
| 18 | TURN | 2,319 |
| 77 | DAMAGE_N_3 (hit) | 1,312 |

**Key Properties**:
- **On ground**: 100%
- **Invulnerability**: 1.70%
- **Hitstun**: 0%
- **Crouch Canceling**: When hit during SQUAT_WAIT, knockback and hitlag are reduced to 0.67x

**Crouch Cancel Mechanics**:
Per SmashWiki, crouch canceling in Melee provides:
- Knockback multiplier: 0.67x
- Hitlag multiplier: 0.67x (attacker receives normal hitlag)
- Combined with Melee's higher tumble threshold (80 units vs 64), allows characters to remain in non-tumble states longer
- When landing during non-tumble hitstun, player goes into LANDING (4 frames) instead of full hitstun

---

### SQUAT_RV (State 41) - Stand from Crouch

**Internal Name**: `SQUAT_RV`

**Description**: The animation when standing up from a crouching position. This is the reverse of SQUAT.

**Duration Statistics** (Fox dittos):
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 4 frames |
| Average | 4.79 frames |
| Most Common | 10 frames (14,688 occurrences) |
| Instances | 63,476 |
| Total Frames | 304,324 (0.11% of all frames) |

**Duration Distribution**:
| Duration | Occurrences | Percentage |
|----------|-------------|------------|
| 1 frame | 16,847 | 9.55% |
| 10 frames | 14,688 | 10.50% |
| 2 frames | 7,817 | 6.33% |
| 3 frames | 6,066 | 5.48% |

The bimodal distribution (peaks at 1 frame and 10 frames) suggests:
- 1 frame: Players immediately cancel into another action
- 10 frames: Full stand-up animation completes

**Entry Conditions**:
| From State | Name | Occurrences |
|------------|------|-------------|
| 40 | SQUAT_WAIT | 26,183 |
| 39 | SQUAT | 22,063 |
| 57 | ATTACK_LW_3 (dtilt) | 15,195 |

**Exit Conditions** (top 10):
| To State | Name | Occurrences |
|----------|------|-------------|
| 15 | WALK_SLOW | 22,145 |
| 24 | KNEE_BEND (jumpsquat) | 15,563 |
| 14 | WAIT | 10,093 |
| 16 | WALK_MIDDLE | 4,750 |
| 18 | TURN | 3,071 |
| 44 | ATTACK_11 (jab 1) | 1,602 |
| 20 | DASH | 1,312 |
| 178 | GUARD_ON (shield) | 761 |
| 39 | SQUAT (re-crouch) | 736 |
| 182 | GUARD_REFLECT (powershield) | 659 |

**Key Properties**:
- **On ground**: 100%
- **Invulnerability**: 3.45%
- **Hitstun**: 0%

---

## Landing States

### LANDING (State 42) - Normal Landing

**Internal Name**: `LANDING`

**Description**: The landing lag animation when touching the ground from an airborne state (not during an aerial attack or helpless fall). This is the "clean" landing with minimal lag.

**Frame Data**:
- Fox's normal landing lag: **4 frames**
- Most characters: 4 frames (a few exceptions exist)
- Heavy landing (high downward speed): 2-6 frames depending on character

**Duration Statistics** (Fox dittos):
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 7 frames |
| Average | 10.4 frames |
| Max | 349 frames |
| Instances | 619,026 |
| Total Frames | 6,435,642 (2.23% of all frames) |

**Duration Distribution** (key durations):
| Duration | Occurrences | Note |
|----------|-------------|------|
| 4 frames | **100,998** | **Fox's landing lag - most common** |
| 5 frames | 41,121 | |
| 6 frames | 33,805 | |
| 7 frames | 21,093 | |
| 8 frames | 33,101 | |

The 4-frame duration dominates, confirming Fox's 4-frame landing lag.

**Entry Conditions** (top 15):
| From State | Name | Occurrences |
|------------|------|-------------|
| 29 | FALL | 180,837 |
| 25 | JUMP_F | 173,278 |
| 345 | BLASTER_LOOP (laser) | 148,880 |
| 67 | ATTACK_AIR_B (bair) | 143,682 |
| 27 | JUMP_AERIAL_F (double jump) | 86,339 |
| 244 | PASS (platform drop) | 50,553 |
| 85 | DAMAGE_AIR_2 | 41,313 |
| 68 | ATTACK_AIR_HI (uair) | 37,680 |
| 26 | JUMP_B | 27,153 |
| 80 | DAMAGE_N_2 | 26,424 |

**Exit Conditions** (top 10):
| To State | Name | Occurrences |
|----------|------|-------------|
| 18 | TURN | 267,968 |
| 20 | DASH | 245,272 |
| 24 | KNEE_BEND (jumpsquat) | 138,255 |
| 15 | WALK_SLOW | 116,385 |
| 178 | GUARD_ON (shield) | 96,199 |
| 40 | SQUAT_WAIT | 81,296 |
| 56 | ATTACK_HI_3 (utilt) | 14,654 |
| 16 | WALK_MIDDLE | 12,085 |
| 29 | FALL (off platform) | 10,877 |
| 44 | ATTACK_11 (jab) | 10,794 |

**Key Properties**:
- **On ground**: 100%
- **Invulnerability**: 9.61% (from ledgedash intangibility carrying over)
- **Hitstun**: 0%
- **Jumps remaining**: 2 (refreshed on landing)

**Key Observation**: LANDING from ATTACK_AIR_B (bair) at 143,682 is high - this represents autocanceled bairs. When an aerial autocancels, it uses LANDING instead of LANDING_AIR_*.

---

### LANDING_FALL_SPECIAL (State 43) - Wavedash/Helpless Landing

**Internal Name**: `LANDING_FALL_SPECIAL`

**Description**: The landing lag animation after air dodging (wavedash) or landing from helpless/freefall states (after up-B, side-B, etc.). This has more lag than normal landing.

**Frame Data**:
- Fox's wavedash/special landing lag: **10 frames**
- All characters have 10 frames of airdodge landing lag

**Duration Statistics** (Fox dittos):
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 10 frames |
| Average | 15.27 frames |
| Max | 240 frames |
| Instances | 614,247 |
| Total Frames | 9,382,601 (**3.25% of all frames**) |

**Duration Distribution** (key durations):
| Duration | Occurrences | Note |
|----------|-------------|------|
| 10 frames | **197,813** | **Wavedash landing lag - dominant** |
| 9 frames | 9,715 | Early interrupt |
| 11 frames | 1,484 | |

The 10-frame duration is overwhelmingly dominant, confirming the wavedash landing lag.

**Entry Conditions** (top 10):
| From State | Name | Occurrences | Source Type |
|------------|------|-------------|-------------|
| 236 | ESCAPE_AIR (airdodge) | 448,767 | **Wavedash** |
| 24 | KNEE_BEND (jumpsquat) | 368,723 | **Waveland** |
| 352 | ILLUSION_AIR_END (side-B) | 44,506 | Side-B recovery |
| 25 | JUMP_F | 34,739 | Waveland |
| 27 | JUMP_AERIAL_F | 32,241 | Waveland |
| 26 | JUMP_B | 13,116 | Waveland |
| 28 | JUMP_AERIAL_B | 4,880 | Waveland |
| 35 | FALL_SPECIAL (helpless) | 3,880 | Up-B/side-B landing |
| 244 | PASS (platform drop) | 1,666 | |

**Wavedash vs Waveland Breakdown**:
- **Wavedash** (from ESCAPE_AIR): 448,767 (46.6%)
- **Waveland** (from jumps/KNEE_BEND): ~454,000 (47.1%)
- **Recovery landing** (side-B, up-B): ~48,500 (5.0%)
- **Other**: ~1.3%

The data shows wavedashes and wavelands occur at nearly equal frequency in competitive Fox play!

**Exit Conditions** (top 15):
| To State | Name | Occurrences | Percentage |
|----------|------|-------------|------------|
| 14 | WAIT | 483,213 | **50.58%** |
| 18 | TURN | 82,425 | 8.63% |
| 29 | FALL (off platform) | 81,575 | 8.54% |
| 20 | DASH | 68,306 | 7.15% |
| 16 | WALK_MIDDLE | 56,198 | 5.88% |
| 39 | SQUAT | 51,518 | 5.39% |
| 178 | GUARD_ON (shield) | 50,201 | 5.25% |
| 15 | WALK_SLOW | 14,880 | 1.56% |
| 88 | DAMAGE_FLY_N (hit) | 9,644 | 1.01% |
| 24 | KNEE_BEND (jumpsquat) | 7,401 | 0.77% |
| 226 | CAPTURE_PULLED_HI (grabbed) | 6,601 | 0.69% |
| 245 | OTTOTTO (teeter) | 5,106 | 0.53% |

**Key Properties**:
- **On ground**: 100%
- **Invulnerability**: 11.08% (ledgedash intangibility)
- **Hitstun**: 0%
- **Jumps remaining**: 2 (refreshed on landing)
- **Character is considered standing** - can immediately perform grounded actions after lag ends

**Key Findings**:
1. **50.6% exit to WAIT** - Most wavedashes end in neutral standing position
2. **8.54% exit to FALL** - Wavedashing off platforms is common
3. **5.39% exit to SQUAT** - Wavedash into crouch (for shine setup)
4. **5.25% exit to GUARD_ON** - Wavedash into shield

---

## State Flow Diagram

```
                            ┌─────────────────────────────────────────────┐
                            │           GROUNDED STATES                    │
                            │  (Wait, Walk, Dash, Run, etc.)               │
                            └──────────────┬──────────────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    ▼                      ▼                      ▼
             ┌──────────┐          ┌──────────────┐       ┌────────────┐
             │ Hold Down │          │    Jump      │       │  Airdodge  │
             │ on stick  │          │  (in air)    │       │ to ground  │
             └─────┬─────┘          └──────┬───────┘       └──────┬─────┘
                   │                       │                      │
                   ▼                       │                      │
            ┌─────────────┐                │                      │
            │ SQUAT (39)  │                │                      │
            │ Crouch entry│                │                      │
            │ (~7 frames) │                │                      │
            └──────┬──────┘                │                      │
                   │                       │                      │
    ┌──────────────┼──────────────┐        │                      │
    │              │              │        │                      │
    ▼              ▼              ▼        │                      │
┌────────┐  ┌───────────┐  ┌──────────┐    │                      │
│ SHINE  │  │SQUAT_WAIT │  │ Platform │    │                      │
│ (360)  │  │   (40)    │  │  Drop    │    │                      │
│ 58%!   │  │ Hold crouch│  │ (244)   │    │                      │
└────────┘  └─────┬─────┘  └──────────┘    │                      │
                  │                        │                      │
           ┌──────┴──────┐                 │                      │
           │             │                 │                      │
           ▼             ▼                 │                      │
    ┌───────────┐  ┌──────────┐            │                      │
    │ SQUAT_RV  │  │  Shine   │            │                      │
    │   (41)    │  │  dtilt   │            │                      │
    │Stand from │  │  etc.    │            │                      │
    │  crouch   │  └──────────┘            │                      │
    │(~10 frames│                          │                      │
    └─────┬─────┘                          │                      │
          │                                │                      │
          ▼                                ▼                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         LANDING (42)                                 │
    │              Normal landing lag - 4 frames (Fox)                     │
    │         Entry: From falls, jumps, autocanceled aerials               │
    └──────────────────────────────────┬──────────────────────────────────┘
                                       │
    ┌──────────────────────────────────┼──────────────────────────────────┐
    │                                  │                                  │
    ▼                                  ▼                                  ▼
┌─────────┐                    ┌──────────────┐                   ┌───────────┐
│  WAIT   │                    │    DASH      │                   │  SQUAT    │
│  (14)   │                    │    (20)      │                   │   (39)    │
└─────────┘                    └──────────────┘                   └───────────┘



    ┌─────────────────────────────────────────────────────────────────────┐
    │                    LANDING_FALL_SPECIAL (43)                         │
    │           Wavedash/helpless landing lag - 10 frames                  │
    │    Entry: From airdodge (wavedash), jumps (waveland), up-B/side-B   │
    └──────────────────────────────────┬──────────────────────────────────┘
                                       │
    ┌──────────┬──────────────────────┼─────────────────────┬─────────────┐
    │          │                      │                     │             │
    ▼          ▼                      ▼                     ▼             ▼
┌───────┐  ┌──────┐            ┌──────────┐          ┌──────────┐  ┌──────────┐
│ WAIT  │  │ TURN │            │  DASH    │          │  SQUAT   │  │  FALL    │
│ (14)  │  │ (18) │            │   (20)   │          │  (39)    │  │  (29)    │
│ 50.6% │  │ 8.6% │            │   7.2%   │          │  5.4%    │  │  8.5%    │
└───────┘  └──────┘            └──────────┘          └──────────┘  └──────────┘
```

---

## Key Mechanics

### Wavedash Mechanics

A wavedash is performed by:
1. Jump (enters KNEE_BEND for 3 frames with Fox)
2. Immediately airdodge diagonally into the ground (ESCAPE_AIR)
3. Land in LANDING_FALL_SPECIAL (10 frames lag)

**Total wavedash duration**: 3 + 1 + 10 = **14 frames minimum** (Fox)

During LANDING_FALL_SPECIAL, the character is considered **standing**, enabling:
- Ground attacks (smashes, tilts, jabs)
- Grabs
- Shield
- Another jump

**Waveland**: Same mechanics but performed after already being airborne (from platform, double jump, etc.)

### Crouch Cancel Mechanics

When hit while crouching (SQUAT_WAIT):
- Knockback reduced to **0.67x**
- Hitlag reduced to **0.67x** for the defender
- Combined with Melee's tumble threshold (80 units), allows survival at higher percents
- If not sent into tumble, landing uses LANDING state (4 frames) instead of extended hitstun

### Platform Drop-Through

From SQUAT (39), pressing down on a platform enters PASS (244), allowing the character to fall through. This is the standard platform drop technique.

---

## Comparative Analysis

### State Frequency

| State | Total Frames | % of All Frames | Role |
|-------|-------------|-----------------|------|
| LANDING_FALL_SPECIAL (43) | 9,382,601 | **3.25%** | Most common |
| LANDING (42) | 6,435,642 | 2.23% | |
| SQUAT (39) | 1,849,824 | 0.64% | |
| SQUAT_WAIT (40) | 827,287 | 0.29% | |
| SQUAT_RV (41) | 304,324 | 0.11% | Least common |

LANDING_FALL_SPECIAL is 45% more common than LANDING, demonstrating how frequently wavedashing/wavelanding is used in competitive Fox play.

### Landing Lag Comparison

| Landing Type | State | Fox Frames |
|--------------|-------|------------|
| Normal landing | LANDING (42) | 4 frames |
| Wavedash/airdodge | LANDING_FALL_SPECIAL (43) | 10 frames |
| Nair (L-canceled) | LANDING_AIR_N (70) | 7 frames |
| Nair (no L-cancel) | LANDING_AIR_N (70) | 15 frames |
| Bair (L-canceled) | LANDING_AIR_B (72) | 9 frames |
| Bair (no L-cancel) | LANDING_AIR_B (72) | 18 frames |

### Aerial Landing Lag States (LANDING_AIR_*)

For reference, the aerial-specific landing lag states (not covered in this document):

| State | Name | Fox Median Duration | Instances |
|-------|------|---------------------|-----------|
| 70 | LANDING_AIR_N | 7 frames | 233,873 |
| 71 | LANDING_AIR_F | 11 frames | 26,771 |
| 72 | LANDING_AIR_B | 10 frames | 210,516 |
| 73 | LANDING_AIR_HI | 9 frames | 106,476 |
| 74 | LANDING_AIR_LW | 9 frames | 188,880 |

---

## Character-Specific Notes (Fox)

Fox's attributes relevant to crouch/landing:
- **Jumpsquat**: 3 frames (tied for fastest)
- **Normal landing lag**: 4 frames
- **Wavedash landing lag**: 10 frames (universal)
- **Crouch height**: Very low (effective for dodging high attacks)
- **Wavedash distance**: Medium (affected by high traction)

Fox's crouch-to-shine (58% of SQUAT exits) is a signature technique, enabled by:
1. Fast jumpsquat (3 frames)
2. Frame 1 shine
3. Low crouch profile

---

## Advanced Techniques

### Crouch-Cancel Shine
Enter crouch (SQUAT) and immediately shine. The crouch allows:
- Canceling other animations quickly
- Lower hurtbox while buffering shine
- Can absorb weak hits with crouch cancel before shine activates

### Wavedash Out of Shield (WD OoS)
From shield (GUARD states), jump and immediately airdodge to wavedash. Used for:
- Escaping shield pressure
- Punishing unsafe moves on shield
- Repositioning while maintaining defensive option

### Ledgedash
From ledge, drop → double jump → airdodge into stage. The invincibility from ledge (CLIFF_WAIT) carries into LANDING_FALL_SPECIAL, creating **ledgedash intangibility**. Fox can achieve significant actionable intangibility this way.

Data shows 11.08% of LANDING_FALL_SPECIAL frames are invulnerable, largely from ledgedash.

---

## Sources

### Primary Sources (Quantitative)
- **Parquet replay data**: `/Users/eppie/ssb_wiki/fox_vs_fox_parquet/` - 18.8 million frames of crouch/landing data analyzed

### Documentation Sources (Qualitative)
- [SmashWiki - Crouch](https://www.ssbwiki.com/Crouch)
- [SmashWiki - Crouch Cancel](https://www.ssbwiki.com/Crouch_cancel)
- [SmashWiki - Landing Lag](https://www.ssbwiki.com/Landing_lag)
- [SmashWiki - Wavedash](https://www.ssbwiki.com/Wavedash)
- [SmashWiki - L-canceling](https://www.ssbwiki.com/L-canceling)
- [SmashWiki - Fox Neutral Aerial](https://www.ssbwiki.com/Fox_(SSBM)/Neutral_aerial)
- [libmelee Action State Enums](https://libmelee.readthedocs.io/en/latest/enums.html)

### Community Sources
- Smashboards frame data discussions
- Smashboards wavedash technique threads

---

## Investigation Methodology

This analysis used the following sources, ranked by usefulness:

1. **Parquet Replay Data (Most Valuable)**: Provided definitive quantitative data on state durations, transitions, frequencies, and technique usage patterns. Key finding: 58% of SQUAT exits go to Shine.

2. **SmashWiki (High Value)**: Provided qualitative explanations of crouch cancel mechanics (0.67x knockback/hitlag), wavedash mechanics, and frame data for landing lag.

3. **action_state.json (High Value)**: Definitive mapping of state IDs to internal names.

4. **Smashboards RAG System (Moderate Value)**: Confirmed wavedash has 10 frames of landing lag from community discussions.

5. **Web Search (Supporting)**: Helped verify Fox-specific frame data.

**Key Findings**:
- Crouch-to-shine (58%) dominates SQUAT usage in Fox dittos
- Wavedash and waveland occur at nearly equal frequency (~47% each)
- LANDING_FALL_SPECIAL is the most common of these states (3.25% of all frames)
- 4-frame and 10-frame durations are clearly visible in the data for LANDING and LANDING_FALL_SPECIAL respectively
