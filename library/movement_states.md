# Movement Action States (14-24)

This document describes the grounded movement action states in Super Smash Bros. Melee.

## Overview

Grounded movement in Melee consists of standing, walking, dashing, running, and turning states. These states form the foundation of the game's movement system and enable advanced techniques like dash dancing and wavedashing.

| ID | Internal Name | Description |
|----|---------------|-------------|
| 14 | WAIT | Standing idle |
| 15 | WALK_SLOW | Slow walk (light stick tilt) |
| 16 | WALK_MIDDLE | Medium walk |
| 17 | WALK_FAST | Fast walk (full stick tilt) |
| 18 | TURN | Turnaround during initial dash |
| 19 | TURN_RUN | Turnaround during run |
| 20 | DASH | Initial dash |
| 21 | RUN | Full run (after dash completes) |
| 23 | RUN_BRAKE | Stopping from run |
| 24 | KNEE_BEND | Jumpsquat (pre-jump crouch) |

**Note**: State 22 (RUN_DIRECT) exists in game data but does not appear in the analyzed Fox ditto dataset.

---

## State Descriptions

### WAIT (State 14) - Standing Idle

**Internal Name**: `WAIT`

**Description**: The neutral standing state where the character is idle on the ground. This is the default state characters return to after most grounded actions complete.

**Duration Statistics** (Fox dittos):
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 3 frames |
| Average | 5.47 frames |
| Max | 1,418 frames (~23.6 seconds) |
| Instances | 1,605,589 |
| Total Frames | 8,790,302 |

**Entry Conditions** (top sources):
| From State | Name | Occurrences |
|------------|------|-------------|
| 43 | LANDING_FALL_SPECIAL | 483,213 |
| 70 | LANDING_AIR_N | 164,450 |
| 72 | LANDING_AIR_B | 142,804 |
| 221 | THROW_HI (upthrow) | 111,748 |
| 74 | LANDING_AIR_LW | 82,422 |
| 73 | LANDING_AIR_HI | 73,595 |
| 56 | ATTACK_HI_3 (utilt) | 64,851 |

**Exit Conditions** (top actions):
| To State | Name | Occurrences | % of Exits |
|----------|------|-------------|------------|
| 18 | TURN | 532,545 | 33.2% |
| 20 | DASH | 454,432 | 28.3% |
| 24 | KNEE_BEND | 270,768 | 16.9% |
| 15 | WALK_SLOW | 123,626 | 7.7% |
| 39 | SQUAT | 70,800 | 4.4% |
| 56 | ATTACK_HI_3 | 69,536 | 4.3% |

**Available Actions**: All grounded options - dash, walk, jump, crouch, shield, grab, any grounded attack, spotdodge, roll.

**Key Property**: The median duration of only 3 frames shows that competitive Fox players rarely stand still - they're constantly moving or attacking.

---

### WALK_SLOW (State 15) - Slow Walk

**Internal Name**: `WALK_SLOW`

**Description**: The slowest walking state, entered when lightly tilting the control stick. Walking allows characters to maintain their facing direction while moving.

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 1 frame |
| Average | 2.29 frames |
| Max | 221 frames |
| Instances | 398,501 |
| Total Frames | 912,336 |

**Entry Conditions**:
| From State | Name | Occurrences |
|------------|------|-------------|
| 14 | WAIT | 123,626 |
| 16 | WALK_MIDDLE | 120,555 |

**Exit Conditions**:
| To State | Name | Occurrences |
|----------|------|-------------|
| 20 | DASH | 207,051 |
| 16 | WALK_MIDDLE | 138,189 |
| 14 | WAIT | 55,862 |

**Fox Walk Speed**: 1.6 units/frame (tied with Marth for fastest in game)

**Competitive Note**: Walking is rarely used in high-level play compared to dashing, but allows access to tilts and smashes while moving.

---

### WALK_MIDDLE (State 16) - Medium Walk

**Internal Name**: `WALK_MIDDLE`

**Description**: The middle walking speed, entered with moderate stick tilt. Characters transition through this state when accelerating from slow walk to fast walk.

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 2 frames |
| Average | 3.48 frames |
| Max | 132 frames |
| Instances | 193,058 |
| Total Frames | 670,961 |

**Entry/Exit**: Transitions primarily between WALK_SLOW (15) and WALK_FAST (17) based on stick position.

---

### WALK_FAST (State 17) - Fast Walk

**Internal Name**: `WALK_FAST`

**Description**: The maximum walking speed, achieved with full stick tilt below the dash threshold.

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 5 frames |
| Average | 7.30 frames |
| Max | 93 frames |
| Instances | 24,654 |
| Total Frames | 180,027 |

**Note**: Least used of the walking states - players either walk slowly for precision or dash for speed.

---

### TURN (State 18) - Turnaround (Dash Dance)

**Internal Name**: `TURN`

**Description**: The turnaround animation during the initial dash window. This state is the key to **dash dancing** - rapidly alternating directions during the dash window creates the signature back-and-forth movement.

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 1 frame |
| Average | 1.53 frames |
| Max | 77 frames |
| Instances | 2,732,996 |
| Total Frames | 4,171,586 |

**Dash Dance Pattern**:
The sequence DASH (20) → TURN (18) → DASH (20) occurs **1,568,150 times** in the dataset - this is dash dancing!

**Entry Conditions**:
| From State | Name | Occurrences |
|------------|------|-------------|
| 20 | DASH | 2,195,773 |
| 14 | WAIT | 532,545 |

**Exit Conditions**:
| To State | Name | Occurrences |
|----------|------|-------------|
| 20 | DASH | 2,469,028 |

**Key Mechanics**:
- Can only be performed during the **initial dash window** (first 11 frames for Fox)
- Pressing the opposite direction triggers TURN, which can immediately transition back to DASH
- If dash animation completes and enters RUN, turning triggers TURN_RUN instead (much longer, less useful)

**Competitive Importance**: Dash dancing is fundamental to Melee's movement meta, allowing players to:
- Bait and punish opponent attacks
- Space precisely while remaining mobile
- Threaten multiple approach angles
- React to opponent movement with full options available

---

### TURN_RUN (State 19) - Run Turnaround

**Internal Name**: `TURN_RUN`

**Description**: The turnaround animation when attempting to reverse direction during full run (after initial dash completes). This is a longer, committed animation that leaves the player vulnerable.

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 7 frames |
| Average | 10.15 frames |
| Max | 39 frames |
| Instances | 11,143 |
| Total Frames | 113,118 |

**Entry Conditions**:
| From State | Name | Occurrences |
|------------|------|-------------|
| 23 | RUN_BRAKE | 9,048 |
| 21 | RUN | 2,065 |

**Competitive Note**: This state is generally avoided because:
- It's much longer than TURN (median 7 frames vs 1 frame)
- No attacks can be performed during the animation
- Players prefer to jump out of run or use dash dancing to avoid this state

---

### DASH (State 20) - Initial Dash

**Internal Name**: `DASH`

**Description**: The initial dash animation, the most important movement state in Melee. Dashing provides fast horizontal movement while maintaining access to many options.

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 5 frames |
| Average | 6.69 frames |
| Max | 265 frames |
| Instances | 3,632,815 |
| Total Frames | 24,293,628 |

**Fox's Dash Properties**:
- Initial dash velocity: 1.9 → accelerates to 2.2
- Dash frames before run: **11 frames** (dash dance window)
- Dash acceleration: +0.12 per frame

**Entry Conditions** (top sources):
| From State | Name | Occurrences | % of Entries |
|------------|------|-------------|--------------|
| 18 | TURN | 2,469,028 | 67.0% |
| 14 | WAIT | 454,432 | 12.3% |
| 42 | LANDING | 245,272 | 6.7% |
| 15 | WALK_SLOW | 207,051 | 5.6% |

**Exit Conditions**:
| To State | Name | Occurrences | Action |
|----------|------|-------------|--------|
| 18 | TURN | 2,195,773 | Dash dance |
| 21 | RUN | 363,914 | Dash frames expired |
| 24 | KNEE_BEND | 359,952 | Jump |
| 50 | ATTACK_DASH | 309,119 | Dash attack |
| 212 | CATCH | 225,403 | Dash grab |
| 29 | FALL | 154,743 | Ran off platform |

**Available Actions During Dash**:
- Turn around (dash dance) - within first 11 frames
- Jump / wavedash
- Dash attack (after frame 3)
- Dash grab
- Shield
- Special moves
- Up smash (jump cancel)

**Key Insight**: 67% of DASH entries come from TURN - confirming that dash dancing (DASH↔TURN loop) dominates Fox movement.

---

### RUN (State 21) - Full Run

**Internal Name**: `RUN`

**Description**: The full running state, entered after the initial dash animation completes. Running is faster than dashing but has fewer options.

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 6 frames |
| Average | 7.52 frames |
| Max | 82 frames |
| Instances | 366,159 |
| Total Frames | 2,754,056 |

**Fox's Run Speed**: 2.2 units/frame

**Entry Conditions**: Almost exclusively from DASH (20) when dash frames complete.

**Exit Conditions**:
| To State | Name | Occurrences | Action |
|----------|------|-------------|--------|
| 23 | RUN_BRAKE | 142,663 | Release stick |
| 24 | KNEE_BEND | 141,965 | Jump |
| 50 | ATTACK_DASH | 16,550 | Dash attack |
| 39 | SQUAT | 5,965 | Crouch |
| 19 | TURN_RUN | 2,065 | Turn around |

**Available Actions During Run**:
- Jump (crucial for maintaining movement)
- Dash attack
- Crouch (slide into crouch)
- Turn around (triggers TURN_RUN - generally avoided)
- Shield (rolling stop)

**Competitive Note**: Players avoid staying in RUN because turning around is slow. The preferred pattern is to jump out of run or dash dance instead.

---

### RUN_BRAKE (State 23) - Run Stop

**Internal Name**: `RUN_BRAKE`

**Description**: The stopping animation when releasing the control stick during run. A brief deceleration state before returning to standing.

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 1 frame |
| Average | 1.72 frames |
| Max | 18 frames |
| Instances | 142,756 |
| Total Frames | 245,374 |

**Entry Conditions**: Almost exclusively from RUN (21) - 142,663 occurrences.

**Exit Conditions**: Typically returns to WAIT (14) or enters TURN_RUN (19) if stick is pressed opposite direction.

---

### KNEE_BEND (State 24) - Jumpsquat

**Internal Name**: `KNEE_BEND`

**Description**: The pre-jump crouch animation, commonly called "jumpsquat." This state is critical for all jumping-based techniques including wavedashing and jump-canceled moves.

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 3 frames |
| Average | 3.48 frames |
| Max | 82 frames |
| Instances | 2,239,675 |
| Total Frames | 7,802,881 |

**Fox's Jumpsquat**: 3 frames (among the fastest in the game)

**Entry Conditions** (top sources):
| From State | Name | Occurrences |
|------------|------|-------------|
| 20 | DASH | 359,952 |
| 14 | WAIT | 270,768 |
| 21 | RUN | 141,965 |
| 40 | SQUAT_WAIT | 112,099 |

**Exit Conditions**:
| To State | Name | Occurrences | % | Technique |
|----------|------|-------------|---|-----------|
| 25 | JUMP_F | 1,713,896 | 60.7% | Forward jump |
| 43 | LANDING_FALL_SPECIAL | 368,723 | 13.1% | **Wavedash** |
| 26 | JUMP_B | 271,645 | 9.6% | Backward jump |
| 212 | CATCH | 210,565 | 7.5% | **JC Grab** |
| 63 | ATTACK_HI_4 | 83,623 | 3.0% | **JC Upsmash** |

**Jump Cancel Techniques**:
- **Wavedash**: 13.1% of jumpsquats lead to LANDING_FALL_SPECIAL (air dodge into ground)
- **JC Grab**: 7.5% lead to CATCH (grab during jumpsquat for extended range)
- **JC Upsmash**: 3.0% lead to ATTACK_HI_4 (upsmash during jumpsquat)

**Jumpsquat Frame Data by Character**:
| Frames | Characters |
|--------|------------|
| 3 | Fox, Sheik, Pikachu, Pichu, Samus |
| 4 | Falco, Marth, C. Falcon, Peach, ICs, Doc, Mario, Luigi, Y. Link |
| 5 | Most others |
| 6 | Bowser, Ganondorf, Zelda |

**Competitive Importance**: Fox's 3-frame jumpsquat makes his wavedash, JC grab, and JC upsmash extremely fast and difficult to react to.

---

## Movement State Machine

```
                           ┌─────────────────────────────────┐
                           │         WAIT (14)               │
                           │        Standing idle            │
                           └──────────┬──────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
│  WALK_SLOW (15) │◄────────►│  WALK_MIDDLE(16)│◄────────►│  WALK_FAST (17) │
│   Light tilt    │          │   Medium tilt   │          │   Full tilt     │
└────────┬────────┘          └─────────────────┘          └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│    DASH (20)    │◄───►│    TURN (18)    │ ←── DASH DANCING
│  Initial dash   │     │   Quick turn    │     (1.57M occurrences)
│  (11 frames)    │     │   (1 frame)     │
└────────┬────────┘     └─────────────────┘
         │
         │ (dash frames expire)
         ▼
┌─────────────────┐
│    RUN (21)     │
│   Full run      │
└────────┬────────┘
         │
    ┌────┴────┬──────────────┐
    │         │              │
    ▼         ▼              ▼
┌────────┐ ┌────────────┐ ┌────────────┐
│RUN_    │ │ TURN_RUN   │ │KNEE_BEND   │
│BRAKE   │ │   (19)     │ │   (24)     │
│ (23)   │ │ Slow turn  │ │ Jumpsquat  │
└───┬────┘ │ (7 frames) │ │ (3 frames) │
    │      └────────────┘ └─────┬──────┘
    │                           │
    ▼                           ▼
┌────────┐               ┌──────────────────────────┐
│ WAIT   │               │ JUMP / WAVEDASH /        │
│ (14)   │               │ JC GRAB / JC UPSMASH     │
└────────┘               └──────────────────────────┘
```

---

## Key Techniques Using Movement States

### Dash Dancing
**States**: DASH (20) ↔ TURN (18)
**Occurrences**: 1,568,150 DASH→TURN→DASH sequences

Rapidly alternating stick direction during the initial dash window (11 frames for Fox) to stay mobile while threatening multiple options.

### Wavedashing
**States**: KNEE_BEND (24) → ESCAPE_AIR (236) → LANDING_FALL_SPECIAL (43)
**Occurrences**: 368,723 jumpsquats lead to wavedash landing

Jump, then immediately air dodge diagonally into the ground. Results in a slide while maintaining standing options.

**Total Wavedash Frames**: KNEE_BEND (3) + Air dodge startup + Landing (10) = ~13-14 frames

### Jump-Cancel Grab
**States**: KNEE_BEND (24) → CATCH (212)
**Occurrences**: 210,565

Input grab during jumpsquat for extended grab range compared to standing grab.

### Jump-Cancel Upsmash
**States**: KNEE_BEND (24) → ATTACK_HI_4 (63)
**Occurrences**: 83,623

Input upsmash during jumpsquat to perform upsmash from dash/run without stopping.

---

## Fox Movement Attributes Summary

| Attribute | Value | Rank |
|-----------|-------|------|
| Walk Speed | 1.6 | 1st (tied w/ Marth) |
| Initial Dash | 1.9 | - |
| Run Speed | 2.2 | 2nd |
| Dash Frames | 11 | - |
| Jumpsquat | 3 frames | 1st (tied) |
| Traction | 0.08 | - |

---

## State Frequency Comparison

| State | Total Frames | % of Movement | Median Duration |
|-------|--------------|---------------|-----------------|
| DASH (20) | 24,293,628 | 49.4% | 5 frames |
| WAIT (14) | 8,790,302 | 17.9% | 3 frames |
| KNEE_BEND (24) | 7,802,881 | 15.9% | 3 frames |
| TURN (18) | 4,171,586 | 8.5% | 1 frame |
| RUN (21) | 2,754,056 | 5.6% | 6 frames |
| WALK_SLOW (15) | 912,336 | 1.9% | 1 frame |
| WALK_MIDDLE (16) | 670,961 | 1.4% | 2 frames |
| RUN_BRAKE (23) | 245,374 | 0.5% | 1 frame |
| WALK_FAST (17) | 180,027 | 0.4% | 5 frames |
| TURN_RUN (19) | 113,118 | 0.2% | 7 frames |

**Key Insight**: DASH accounts for nearly half of all movement state frames, confirming that dashing (and dash dancing) is the dominant movement option in competitive Fox play.

---

## Sources

### Primary Sources (Quantitative)
- **Parquet replay data**: `/Users/eppie/ssb_wiki/fox_vs_fox_parquet/` - 49.1 million movement state frames analyzed

### Documentation Sources (Qualitative)
- [SmashWiki - Dash](https://www.ssbwiki.com/Dash)
- [SmashWiki - Dash-dance](https://www.ssbwiki.com/Dash-dance)
- [SmashWiki - Wavedash](https://www.ssbwiki.com/Wavedash)
- [SmashWiki - Fox (SSBM)](https://www.ssbwiki.com/Fox_(SSBM))
- [Smashboards - SSBM Statistics List](https://smashboards.com/threads/ssbm-statistics-list.30064/)
