# Jump & Fall Action States (25-29, 32, 35, 38)

This document describes the jump and fall action states in Super Smash Bros. Melee, covering the complete aerial state machine.

## Overview

When a character leaves the ground, they enter a sequence of jump and fall states. The game tracks different states based on:
- Whether the character used their first jump or double jump
- Whether they're rising or falling
- Whether they're helpless (post-recovery)
- Whether they're tumbling (post-hitstun)

| ID | Internal Name | Description |
|----|---------------|-------------|
| 25 | JUMP_F | First jump (forward facing) |
| 26 | JUMP_B | First jump (backward facing) |
| 27 | JUMP_AERIAL_F | Double jump (forward facing) |
| 28 | JUMP_AERIAL_B | Double jump (backward facing) |
| 29 | FALL | Standard airborne fall |
| 32 | FALL_AERIAL | Fall after double jump |
| 35 | FALL_SPECIAL | Helpless fall (post-recovery) |
| 38 | DAMAGE_FALL | Tumble (post-hitstun) |

**Note**: States 30-31, 33-34, 36-37 are animation variants (_F/_B suffixes) that don't appear in Slippi data - see fall_states.md for details.

---

## Jump States

### JUMP_F (State 25) - First Jump Forward

**Internal Name**: `JUMP_F`

**Description**: The rising animation during a first (grounded) jump while facing or moving forward. This is the most common jump state, accounting for the majority of jumps in competitive play.

**Duration Statistics** (Fox dittos):
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 17 frames |
| Average | 23.69 frames |
| Max | 366 frames |
| Instances | 660,078 |
| Total Frames | 15,635,968 |

**Fox Jump Properties**:
- Jumpsquat: 3 frames
- Full hop height: 31.28 units
- Short hop height: 10.65 units
- Full hop airtime: ~35 frames (rising portion in JUMP_F)
- Short hop airtime: ~21 frames

**Entry Conditions**:
| From State | Name | Occurrences |
|------------|------|-------------|
| 24 | KNEE_BEND (jumpsquat) | 1,713,896 |

**Exit Conditions** (top actions):
| To State | Name | Occurrences | % | Action |
|----------|------|-------------|---|--------|
| 65 | ATTACK_AIR_N | 331,907 | 29.5% | Nair |
| 236 | ESCAPE_AIR | 214,228 | 19.0% | Air dodge |
| 67 | ATTACK_AIR_B | 204,613 | 18.2% | Bair |
| 69 | ATTACK_AIR_LW | 197,830 | 17.6% | Dair |
| 42 | LANDING | 173,278 | 15.4% | Landed (short hop) |
| 68 | ATTACK_AIR_HI | 118,465 | 10.5% | Uair |
| 344 | BLASTER_AIR_STARTUP | 109,477 | 9.7% | Laser |
| 27 | JUMP_AERIAL_F | 108,020 | 9.6% | Double jump |
| 365 | REFLECTOR_AIR_STARTUP | 85,167 | 7.6% | Shine |

**Key Properties**:
- **Jumps remaining**: 100% have 1 jump remaining (double jump available)
- **Average Y position**: 18.76 (rising above stage)
- **Short hop vs Full hop**: ~26% of jumps land directly (short hop to landing)

**Competitive Usage**: The most common aerials from first jump are nair (29.5%), bair (18.2%), and dair (17.6%). Air dodges (19%) are often wavedash attempts.

---

### JUMP_B (State 26) - First Jump Backward

**Internal Name**: `JUMP_B`

**Description**: The rising animation during a first jump while facing backward. Used for retreating aerials and backwards approaches.

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 5 frames |
| Average | 10.71 frames |
| Max | 114 frames |
| Instances | 243,051 |
| Total Frames | 2,604,124 |

**Entry Conditions**: Almost exclusively from KNEE_BEND (24) - 271,645 occurrences.

**Exit Conditions**: Similar to JUMP_F but with different aerial distributions reflecting backward-facing approaches.

**Key Insight**: JUMP_B is much less common than JUMP_F (2.6M vs 15.6M frames), as players typically face their opponent. The shorter median duration (5 vs 17 frames) suggests players often use it for quick retreating options.

---

### JUMP_AERIAL_F (State 27) - Double Jump Forward

**Internal Name**: `JUMP_AERIAL_F`

**Description**: The double jump (aerial jump) animation while facing or moving forward. This is the character's only midair jump and is critical for recovery and combo extensions.

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 20 frames |
| Average | 21.62 frames |
| Max | 209 frames |
| Instances | 537,096 |
| Total Frames | 11,611,808 |

**Fox Double Jump Properties**:
- Height: 40.204 units (higher than first jump!)
- Instant velocity application (not delayed like Ness/Mewtwo)
- Usable from any airborne state (except helpless/hitstun)

**Entry Conditions** (top sources):
| From State | Name | Occurrences | Context |
|------------|------|-------------|---------|
| 29 | FALL | 253,207 | Standard double jump |
| 366 | REFLECTOR_AIR_LOOP | 147,349 | Shine → double jump |
| 25 | JUMP_F | 108,020 | Instant double jump |
| 38 | DAMAGE_FALL | 37,932 | Recovery from tumble |
| 369 | REFLECTOR_AIR_CHANGE | 33,393 | Turnaround shine → DJ |
| 88 | DAMAGE_FLY_N | 30,043 | Jump out of hitstun |
| 365 | REFLECTOR_AIR_STARTUP | 15,994 | Shine startup → DJ |
| 90 | DAMAGE_FLY_TOP | 13,926 | Jump out of launch |

**Exit Conditions**:
| To State | Name | Occurrences |
|----------|------|-------------|
| 32 | FALL_AERIAL | (transitions after apex) |
| 354 | FIRE_FOX_AIR_STARTUP | (up-B recovery) |
| 65-69 | Aerial attacks | (various) |

**Key Properties**:
- **Jumps remaining**: 100% have 0 jumps remaining (double jump expended)
- **Average Y position**: 21.13 (higher than first jump average)
- **Shine → DJ usage**: 147,349 entries from shine - this is waveshine into double jump for combos

**Competitive Note**: The high frequency of entries from REFLECTOR (shine) states shows the importance of "multishines" and waveshine combos in Fox gameplay.

---

### JUMP_AERIAL_B (State 28) - Double Jump Backward

**Internal Name**: `JUMP_AERIAL_B`

**Description**: Double jump while facing backward. Less common than JUMP_AERIAL_F but used for specific recovery angles and retreating double jump aerials.

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 14 frames |
| Average | 16.91 frames |
| Max | 298 frames |
| Instances | 107,264 |
| Total Frames | 1,813,835 |

**Entry Conditions**:
| From State | Name | Occurrences |
|------------|------|-------------|
| 29 | FALL | 45,999 |
| 25 | JUMP_F | 34,319 |
| 26 | JUMP_B | 10,827 |

**Key Properties**:
- **Jumps remaining**: 100% have 0 jumps remaining
- **Average Y position**: 27.44 (highest average of all jump states)
- **Usage**: ~15% as common as JUMP_AERIAL_F

---

## Fall States

### FALL (State 29) - Standard Fall

**Internal Name**: `FALL`

**Description**: The default airborne falling state. Entered after the apex of a first jump, when running off a platform, or after many other aerial transitions.

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 85 frames |
| Average | 105.18 frames |
| Max | 580 frames |
| Instances | 67,076 |
| Total Frames | 7,055,079 |

**Key Properties**:
- **Jumps remaining**: 96.97% have 1 jump (double jump available)
- **Fastfall**: 39.74% of frames are fastfalling
- **Average Y position**: 12.66

**Available Actions**: Double jump, all aerials, air dodge, specials, fastfall, grab ledge.

See `fall_states.md` for complete details.

---

### FALL_AERIAL (State 32) - Post-Double-Jump Fall

**Internal Name**: `FALL_AERIAL`

**Description**: Fall state after using double jump. Indicates no more jumps available.

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 9 frames |
| Average | 12.62 frames |
| Max | 143 frames |
| Instances | 8,927 |
| Total Frames | 112,629 |

**Key Properties**:
- **Jumps remaining**: 100% have 0 jumps
- **Average Y position**: 33.33 (high - entered after double jump apex)
- **Short duration**: Players typically act immediately (up-B, aerial, etc.)

See `fall_states.md` for complete details.

---

### FALL_SPECIAL (State 35) - Helpless Fall

**Internal Name**: `FALL_SPECIAL`

**Description**: The "helpless" or "freefall" state after using recovery moves or air dodging. Very limited options available.

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 17 frames |
| Average | 17.49 frames |
| Max | 119 frames |
| Instances | 34,656 |
| Total Frames | 606,269 |

**Key Properties**:
- **Jumps remaining**: 100% have 0 jumps
- **Average Y position**: -64.38 (far below stage - usually recovering)
- **High death rate**: 25,801 exits to DEAD_DOWN

**Available Actions**: Fastfall, drift, grab ledge only.

See `fall_states.md` for complete details.

---

### DAMAGE_FALL (State 38) - Tumble

**Internal Name**: `DAMAGE_FALL`

**Description**: The tumbling animation entered after being launched with high knockback. Distinct from regular fall states - character spins uncontrollably until they act or land.

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 7 frames |
| Average | 12.20 frames |
| Max | 150 frames |
| Instances | 121,119 |
| Total Frames | 1,478,150 |

**Entry Conditions** (what causes tumble):
| From State | Name | Occurrences |
|------------|------|-------------|
| 88 | DAMAGE_FLY_N | 57,480 |
| 87 | DAMAGE_FLY_HI | 22,941 |
| 90 | DAMAGE_FLY_TOP | 19,212 |
| 89 | DAMAGE_FLY_LW | 17,462 |
| 91 | DAMAGE_FLY_ROLL | 16,227 |

**Exit Conditions** (recovery options):
| To State | Name | Occurrences | Action |
|----------|------|-------------|--------|
| 27 | JUMP_AERIAL_F | 37,932 | Double jump |
| 354 | FIRE_FOX_AIR_STARTUP | 25,918 | Up-B |
| 350 | REFLECTOR_GROUND_STARTUP | 15,693 | Shine |
| 365 | REFLECTOR_AIR_STARTUP | 15,675 | Air shine |
| 29 | FALL | 9,588 | Momentum carried into fall |
| 0 | DEAD_DOWN | 8,572 | Died |
| 28 | JUMP_AERIAL_B | 2,797 | Double jump back |
| 252 | CLIFF_CATCH | 2,867 | Grabbed ledge |

**Key Properties**:
- **Jumps remaining**: 61.86% have 1 jump, 38.14% have 0 (depends on whether DJ was used before being hit)
- **Hitstun**: 99.9% of frames have NO hitstun (tumble is post-hitstun)
- **Average Y position**: 10.70 (near stage level)
- **Can act**: Unlike hitstun, players CAN act during tumble (aerial, jump, special)

**Tumble vs Hitstun**:
- **Hitstun** (DAMAGE_FLY states 87-91): Cannot act, being launched
- **Tumble** (DAMAGE_FALL 38): CAN act, but in spinning animation until input

**Tech Window**: If landing while tumbling, must tech within 20 frames of landing or enter hard knockdown. The 40-frame tech lockout prevents mashing.

---

## State Flow Diagram

```
                              ┌─────────────────────┐
                              │   GROUNDED STATES   │
                              │   (WAIT, DASH, etc) │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │    KNEE_BEND (24)   │
                              │     Jumpsquat       │
                              │     3 frames        │
                              └──────────┬──────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                         │
                    ▼                                         ▼
           ┌────────────────┐                        ┌────────────────┐
           │   JUMP_F (25)  │                        │   JUMP_B (26)  │
           │  First jump    │                        │  First jump    │
           │  (forward)     │                        │  (backward)    │
           │  1 DJ left     │                        │  1 DJ left     │
           └───────┬────────┘                        └───────┬────────┘
                   │                                         │
                   ├──────────────┬──────────────────────────┤
                   │              │                          │
                   ▼              ▼                          ▼
            ┌──────────┐   ┌─────────────┐            ┌──────────┐
            │ LANDING  │   │  FALL (29)  │            │ Aerials  │
            │   (42)   │   │ Standard    │            │ (65-69)  │
            │Short hop │   │ fall        │            │          │
            └──────────┘   │ 1 DJ left   │            └──────────┘
                           └──────┬──────┘
                                  │
               ┌──────────────────┼──────────────────┐
               │                  │                  │
               ▼                  ▼                  ▼
        ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
        │JUMP_AERIAL_F│    │JUMP_AERIAL_B│    │ ESCAPE_AIR  │
        │    (27)     │    │    (28)     │    │   (236)     │
        │ Double jump │    │ Double jump │    │ Air dodge   │
        │  0 DJ left  │    │  0 DJ left  │    │             │
        └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
               │                  │                  │
               └────────┬─────────┘                  │
                        │                            │
                        ▼                            ▼
               ┌────────────────┐           ┌────────────────┐
               │ FALL_AERIAL(32)│           │ FALL_SPECIAL   │
               │ Post-DJ fall   │           │     (35)       │
               │  0 DJ left     │           │  Helpless      │
               │  Can still act │           │  Limited opts  │
               └───────┬────────┘           └───────┬────────┘
                       │                            │
         ┌─────────────┼─────────────┐              │
         │             │             │              │
         ▼             ▼             ▼              ▼
    ┌────────┐   ┌──────────┐  ┌──────────┐  ┌──────────┐
    │LANDING │   │CLIFF_    │  │ UP-B     │  │LAND_FALL │
    │  (42)  │   │CATCH(252)│  │(354-358) │  │SPEC (43) │
    └────────┘   └──────────┘  └────┬─────┘  │ 10 frames│
                                    │        └──────────┘
                                    ▼
                            ┌────────────────┐
                            │ FALL_SPECIAL   │
                            │     (35)       │
                            └────────────────┘


                     ┌─────────────────────────┐
                     │      GOT HIT            │
                     │  (Strong knockback)     │
                     └───────────┬─────────────┘
                                 │
                                 ▼
                     ┌─────────────────────────┐
                     │   DAMAGE_FLY (87-91)    │
                     │   Hitstun - Can't act   │
                     └───────────┬─────────────┘
                                 │ (hitstun ends)
                                 ▼
                     ┌─────────────────────────┐
                     │   DAMAGE_FALL (38)      │
                     │   Tumble - CAN act      │
                     │   61.86% have DJ left   │
                     └───────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
       ┌──────────┐       ┌──────────┐       ┌──────────┐
       │ JUMP_    │       │ UP-B     │       │TECH/LAND │
       │AERIAL(27)│       │(354)     │       │ (199+)   │
       │ Recovery │       │ Recovery │       │          │
       └──────────┘       └──────────┘       └──────────┘
```

---

## Comparative Analysis

### State Frequency

| State | Total Frames | % of Aerial | Instances |
|-------|--------------|-------------|-----------|
| JUMP_F (25) | 15,635,968 | 38.4% | 660,078 |
| JUMP_AERIAL_F (27) | 11,611,808 | 28.5% | 537,096 |
| FALL (29) | 7,055,079 | 17.3% | 67,076 |
| JUMP_B (26) | 2,604,124 | 6.4% | 243,051 |
| JUMP_AERIAL_B (28) | 1,813,835 | 4.5% | 107,264 |
| DAMAGE_FALL (38) | 1,478,150 | 3.6% | 121,119 |
| FALL_SPECIAL (35) | 606,269 | 1.5% | 34,656 |
| FALL_AERIAL (32) | 112,629 | 0.3% | 8,927 |

**Key Insight**: First jump states (25+26) account for 44.8% of aerial frames, while double jump states (27+28) account for 33%. FALL_AERIAL (32) is rare because players typically act immediately after double jump.

### Jump Height Comparison

| Jump Type | Fox Height (units) | Average Y Position |
|-----------|-------------------|-------------------|
| Short Hop | 10.65 | ~10-15 |
| Full Hop | 31.28 | 18.76 (JUMP_F avg) |
| Double Jump | 40.204 | 21.13 (JUMP_AERIAL_F avg) |

### Jumps Remaining by State

| State | 0 Jumps | 1 Jump |
|-------|---------|--------|
| JUMP_F (25) | 0% | 100% |
| JUMP_B (26) | 0% | 100% |
| JUMP_AERIAL_F (27) | 100% | 0% |
| JUMP_AERIAL_B (28) | 100% | 0% |
| FALL (29) | 3% | 97% |
| FALL_AERIAL (32) | 100% | 0% |
| FALL_SPECIAL (35) | 100% | 0% |
| DAMAGE_FALL (38) | 38% | 62% |

---

## Fox-Specific Properties

| Attribute | Value | Rank |
|-----------|-------|------|
| Jumpsquat | 3 frames | 1st (tied) |
| Full Hop Height | 31.28 | 17th |
| Short Hop Height | 10.65 | - |
| Double Jump Height | 40.204 | 8th |
| Fall Speed | 2.8 | 3rd |
| Fast Fall Speed | 3.4 | 2nd |
| Air Speed | 0.83 | 20th |
| Gravity | 0.23 | - |
| Short Hop Window | 2 frames | Strictest |

---

## Key Techniques

### Short Hop Fast Fall L-Cancel (SHFFL)
**Path**: KNEE_BEND → JUMP_F → (aerial) → fastfall → LANDING → L-cancel
**Usage**: Fox's primary approach and pressure tool

### Double Jump Cancel (Not applicable to Fox)
**Affected Characters**: Ness, Yoshi, Mewtwo, Peach
**Mechanic**: These characters can cancel their double jump with an aerial

### Waveshine to Double Jump
**Path**: REFLECTOR_AIR_LOOP (366) → JUMP_AERIAL_F (27)
**Occurrences**: 147,349 (12.7% of all double jumps)
**Usage**: Core Fox combo extension

### Tumble Recovery
**Path**: DAMAGE_FLY → DAMAGE_FALL (38) → (act)
**Recovery Options**:
- Double jump (37,932 occurrences)
- Up-B (25,918)
- Shine (31,368 combined)

---

## Sources

### Primary Sources (Quantitative)
- **Parquet replay data**: `/Users/eppie/ssb_wiki/fox_vs_fox_parquet/` - 40.9 million jump/fall frames analyzed

### Documentation Sources (Qualitative)
- [SmashWiki - Jump](https://www.ssbwiki.com/Jump)
- [SmashWiki - Double jump](https://www.ssbwiki.com/Double_jump)
- [SmashWiki - Short hop](https://www.ssbwiki.com/Short_hop)
- [SmashWiki - Helpless](https://www.ssbwiki.com/Helpless)
- [SmashWiki - Fox (SSBM)](https://www.ssbwiki.com/Fox_(SSBM))

### Related Documentation
- `fall_states.md` - Detailed fall state analysis
- `movement_states.md` - KNEE_BEND (jumpsquat) details
