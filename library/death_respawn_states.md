# Death & Respawn Action States (0-7, 12-13)

This document describes the death and respawn action states in Super Smash Bros. Melee.

## Overview

When a character crosses a blast zone (the invisible boundaries of the stage), they enter a death state. After the death animation completes, they respawn on a revival platform above the stage center. The specific death state depends on which blast zone was crossed.

| ID | Internal Name | Blast Zone | Description |
|----|---------------|------------|-------------|
| 0 | DEAD_DOWN | Bottom | Standard death from falling |
| 1 | DEAD_LEFT | Left | Death from left blast zone |
| 2 | DEAD_RIGHT | Right | Death from right blast zone |
| 4 | DEAD_UP_STAR | Top | "Star KO" - fly into background |
| 6 | DEAD_UP_FALL | Top | Start of "Screen KO" sequence |
| 7 | DEAD_UP_FALL_HIT_CAMERA | Top | "Screen KO" - hit the camera |
| 12 | REBIRTH | N/A | Respawn platform descent |
| 13 | REBIRTH_WAIT | N/A | Waiting on respawn platform |

**Note**: States 3, 5, 8-11 exist in the game data but do not appear in the analyzed Fox ditto dataset:
- 3: DEAD_UP (unused variant)
- 5: DEAD_UP_STAR_ICE (Ice Climbers specific)
- 8: DEAD_UP_FALL_HIT_CAMERA_FLAT (Flat Zone specific)
- 9: DEAD_UP_FALL_ICE (Ice Climbers specific)
- 10: DEAD_UP_FALL_HIT_CAMERA_ICE (Ice Climbers specific)
- 11: SLEEP (different mechanic)

---

## Death States

### DEAD_DOWN (State 0) - Bottom Blast Zone Death

**Internal Name**: `DEAD_DOWN`

**Description**: The death state triggered when a character crosses the bottom blast zone. This is the most common death type, occurring when a character fails to recover or is spiked/meteored off the stage.

**Duration Statistics** (Fox dittos):
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 60 frames |
| Average | 49.87 frames |
| Max | 61 frames |
| Instances | 48,188 |

**Position Data**:
- Average Y: -111.55 (below stage)
- Y Range: -150.66 to -91.0
- Average X: 6.15 (near center, varies by death location)

**Entry Conditions** (top sources):
| From State | Name | Occurrences |
|------------|------|-------------|
| 35 | FALL_SPECIAL (helpless) | 25,801 |
| 38 | DAMAGE_FALL (tumble) | 8,572 |
| 88 | DAMAGE_FLY_N | 2,280 |
| 29 | FALL | 1,827 |
| 247 | FLY_REFLECT_WALL | 1,644 |
| 69 | ATTACK_AIR_LW (dair) | 1,343 |
| 67 | ATTACK_AIR_B (bair) | 1,270 |

**Exit**: Always transitions to REBIRTH (12) if stocks remain.

---

### DEAD_LEFT (State 1) - Left Blast Zone Death

**Internal Name**: `DEAD_LEFT`

**Description**: Death from being launched past the left blast zone. Typically occurs from strong horizontal knockback moves.

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 60 frames |
| Average | 50.33 frames |
| Max | 60 frames |
| Instances | 12,933 |

**Position Data**:
- Average X: -216.46 (far left)
- X Range: -273.85 to -175.70
- Average Y: 44.30 (varies by trajectory)

**Entry Conditions** (top sources):
| From State | Name | Occurrences |
|------------|------|-------------|
| 88 | DAMAGE_FLY_N | 5,053 |
| 87 | DAMAGE_FLY_HI | 2,785 |
| 91 | DAMAGE_FLY_ROLL | 1,816 |
| 89 | DAMAGE_FLY_LW | 1,541 |
| 90 | DAMAGE_FLY_TOP | 1,175 |

**Exit**: Always transitions to REBIRTH (12) if stocks remain.

---

### DEAD_RIGHT (State 2) - Right Blast Zone Death

**Internal Name**: `DEAD_RIGHT`

**Description**: Death from being launched past the right blast zone. Mirror of DEAD_LEFT.

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 60 frames |
| Average | 50.68 frames |
| Max | 60 frames |
| Instances | 14,151 |

**Position Data**:
- Average X: 215.98 (far right)
- X Range: 173.60 to 273.84
- Average Y: 43.93

**Entry Conditions**: Similar to DEAD_LEFT - primarily DAMAGE_FLY states.

**Exit**: Always transitions to REBIRTH (12) if stocks remain.

---

### DEAD_UP_STAR (State 4) - Star KO

**Internal Name**: `DEAD_UP_STAR`

**Description**: The "Star KO" animation where the character flies into the background, screams, and disappears as a twinkling star. This is the longest death animation and occurs randomly when crossing the top blast zone.

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 176 frames |
| Average | 169.17 frames |
| Max | 176 frames |
| Instances | 18,492 |

**Duration**: ~2.93 seconds (176 frames at 60fps)

**Position Data**:
- Average Y: 121.13 (high above stage)
- Y Range: 67.50 to 255.65
- Average X: 3.25 (near center)

**Entry Conditions**:
| From State | Name | Occurrences |
|------------|------|-------------|
| 90 | DAMAGE_FLY_TOP | 18,250 |
| 88 | DAMAGE_FLY_N | 131 |
| 91 | DAMAGE_FLY_ROLL | 64 |

**Probability**: ~81% of top blast zone deaths result in Star KO (vs ~19% Screen KO).

**Exit**: Always transitions to REBIRTH (12) if stocks remain.

**Competitive Note**: The long duration protects the attacker from immediate punishment (e.g., after Jigglypuff's Rest).

---

### DEAD_UP_FALL (State 6) - Screen KO Start

**Internal Name**: `DEAD_UP_FALL`

**Description**: The first part of the "Screen KO" animation sequence. The character begins falling toward the camera after crossing the top blast zone. This state always transitions to DEAD_UP_FALL_HIT_CAMERA (7).

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 51 frames |
| Average | 50.99 frames |
| Max | 51 frames |
| Instances | 3,575 |

**Position Data**:
- Average Y: 191.13 (very high)
- Y Range: 168.0 to 256.18

**Entry Conditions**:
| From State | Name | Occurrences |
|------------|------|-------------|
| 90 | DAMAGE_FLY_TOP | 3,523 |

**Exit**: Always transitions to DEAD_UP_FALL_HIT_CAMERA (7).

---

### DEAD_UP_FALL_HIT_CAMERA (State 7) - Screen KO Impact

**Internal Name**: `DEAD_UP_FALL_HIT_CAMERA`

**Description**: The second part of the "Screen KO" where the character hits the camera, sticks to the screen momentarily with a unique animation, then falls off the bottom. Each character (except Mr. Game & Watch) has a unique splat animation.

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 2 frames |
| Median | 78 frames |
| Average | 72.87 frames |
| Max | 78 frames |
| Instances | 3,574 |

**Total Screen KO Duration**: State 6 (51 frames) + State 7 (78 frames) = **129 frames (~2.15 seconds)**

**Position Data**: Same as State 6 (191.13 avg Y) - character stays in place during splat animation.

**Entry Conditions**: Exclusively from DEAD_UP_FALL (6).

**Exit**: Transitions to REBIRTH (12) if stocks remain.

**Competitive Note**: Screen KOs are ~0.78 seconds faster than Star KOs (129 vs 176 frames), which can allow punishment that wouldn't be possible after a Star KO.

---

## Top Blast Zone KO Distribution

Based on analysis of Fox ditto data:

| KO Type | Instances | Percentage |
|---------|-----------|------------|
| Star KO (State 4) | 14,311 | ~80.7% |
| Screen KO (States 6→7) | 3,410 | ~19.3% |

The game randomly selects between Star KO and Screen KO when a character crosses the top blast zone, with Star KO being approximately 4x more likely.

---

## Respawn States

### REBIRTH (State 12) - Respawn Platform Descent

**Internal Name**: `REBIRTH`

**Description**: The state where the character descends from above on the revival platform (the "angel platform"). The platform carries the character down to a hovering position above stage center.

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 60 frames |
| Average | 59.96 frames |
| Max | 60 frames |
| Instances | 81,229 |

**Duration**: Exactly 60 frames (1 second) for the descent animation.

**Position Data**:
- Average Y: 98.65 (descending)
- Y Range: 45.0 to 188.24 (starts high, ends at hover height)
- Average X: -5.56 (center of stage)

**Entry Conditions**:
| From State | Name | Occurrences |
|------------|------|-------------|
| 0 | DEAD_DOWN | 39,838 |
| 4 | DEAD_UP_STAR | 15,629 |
| 2 | DEAD_RIGHT | 11,914 |
| 1 | DEAD_LEFT | 10,812 |
| 7 | DEAD_UP_FALL_HIT_CAMERA | 3,036 |

**Exit Conditions**:
| To State | Name | Occurrences |
|----------|------|-------------|
| 29 | FALL (dropped immediately) | 55,561 |
| 13 | REBIRTH_WAIT (stayed on platform) | 25,325 |

**Invulnerability**: The character is intangible during the entire descent (while on platform).

**Key Mechanic**: Players can choose to:
1. Stay on the platform (transitions to REBIRTH_WAIT)
2. Drop immediately by pressing down or attacking (transitions to FALL)

---

### REBIRTH_WAIT (State 13) - Waiting on Platform

**Internal Name**: `REBIRTH_WAIT`

**Description**: The state where the character waits on the revival platform. The player can stay here for up to 5 seconds (300 frames) before the platform automatically disappears.

**Duration Statistics**:
| Metric | Value |
|--------|-------|
| Min | 1 frame |
| Median | 5 frames |
| Average | 16.66 frames |
| Max | 240 frames |
| Instances | 25,359 |

**Note**: The median of 5 frames indicates most players drop quickly; the max of 240 frames (4 seconds) shows some waited nearly the full duration.

**Position Data**:
- Average Y: 67.59 (hovering height)
- Y Range: 45.0 to 84.22
- Average X: -4.5 (center)

**Entry Conditions**: Exclusively from REBIRTH (12).

**Exit Conditions**:
| To State | Name | Occurrences | Trigger |
|----------|------|-------------|---------|
| 29 | FALL | 24,008 | Press down or wait timeout |
| 344 | BLASTER_AIR_STARTUP | 1,165 | Pressed B (laser) |
| 65 | ATTACK_AIR_N | 42 | Pressed A (nair) |
| 27 | JUMP_AERIAL_F | 37 | Pressed jump |
| 69 | ATTACK_AIR_LW | 17 | C-stick down (dair) |

**Platform Properties**:
- Duration: Up to 300 frames (5 seconds) before auto-disappear
- Cannot be boarded by other characters
- Character is intangible while on platform

---

## Respawn Invincibility

After leaving the revival platform, the character receives **invincibility frames** to safely return to the stage.

### Invincibility Duration

From empirical data analysis:
- **On-platform invincibility**: Entire duration while on platform (~300 frames max)
- **Post-platform invincibility**: ~120 frames (2 seconds) after dropping

### Invincibility Timeline (Post-Drop)

| Frames After Drop | Invuln % of Players |
|-------------------|---------------------|
| 1-10 | 6% → 34% (ramping up as more players drop) |
| 60 | ~62% |
| 119-120 | ~85% (peak) |
| 121+ | Sharp decline (invuln expiring) |

The data shows invulnerability peaks at frame 119-120 after leaving state 13, then drops sharply - confirming the **120-frame (2-second) invincibility window**.

### Total Respawn Protection

| Phase | Duration | Protection |
|-------|----------|------------|
| REBIRTH (descent) | 60 frames | Intangible |
| REBIRTH_WAIT (on platform) | Up to 300 frames | Intangible |
| Post-drop | 120 frames | Invincible |
| **Maximum total** | **480 frames (8 seconds)** | |

**Practical Note**: The "Angelic" bonus in Melee is awarded if a player wins while still on the revival platform.

---

## State Flow Diagram

```
                    ┌─────────────────────────────────────────┐
                    │           BLAST ZONE CROSSED            │
                    └──────────────────┬──────────────────────┘
                                       │
           ┌───────────┬───────────────┼───────────────┬───────────┐
           ▼           ▼               ▼               ▼           ▼
      ┌─────────┐ ┌─────────┐   ┌───────────┐   ┌─────────┐ ┌─────────┐
      │ DEAD_   │ │ DEAD_   │   │ DEAD_UP_  │   │ DEAD_   │ │ DEAD_   │
      │ DOWN(0) │ │ LEFT(1) │   │  STAR(4)  │   │ RIGHT(2)│ │UP_FALL  │
      │ 60 frms │ │ 60 frms │   │ 176 frms  │   │ 60 frms │ │  (6)    │
      │ Bottom  │ │ Left    │   │ Star KO   │   │ Right   │ │ 51 frms │
      └────┬────┘ └────┬────┘   └─────┬─────┘   └────┬────┘ └────┬────┘
           │           │              │              │           │
           │           │              │              │           ▼
           │           │              │              │    ┌────────────┐
           │           │              │              │    │DEAD_UP_FALL│
           │           │              │              │    │HIT_CAMERA  │
           │           │              │              │    │   (7)      │
           │           │              │              │    │  78 frms   │
           │           │              │              │    │ Screen KO  │
           │           │              │              │    └─────┬──────┘
           │           │              │              │          │
           └───────────┴──────────────┴──────────────┴──────────┘
                                       │
                                       ▼
                              ┌────────────────┐
                              │  REBIRTH (12)  │
                              │   60 frames    │
                              │ Platform desc. │
                              │  Intangible    │
                              └───────┬────────┘
                                      │
                        ┌─────────────┴─────────────┐
                        │                           │
                        ▼                           ▼
               ┌────────────────┐         ┌────────────────┐
               │ REBIRTH_WAIT   │         │   FALL (29)    │
               │     (13)       │         │  Drop immedi-  │
               │ 0-300 frames   │         │  ately         │
               │ Wait on plat   │         │                │
               │  Intangible    │         │                │
               └───────┬────────┘         └───────┬────────┘
                       │                          │
                       ▼                          │
               ┌────────────────┐                 │
               │   FALL (29)    │                 │
               │ (or attack)    │                 │
               └───────┬────────┘                 │
                       │                          │
                       └──────────┬───────────────┘
                                  │
                                  ▼
                         ┌────────────────┐
                         │  120 frames    │
                         │ INVINCIBILITY  │
                         │ (any state)    │
                         └────────────────┘
```

---

## Blast Zone Positions (Battlefield Reference)

Based on position data from deaths:

| Blast Zone | Approximate Position | Death State |
|------------|---------------------|-------------|
| Bottom | Y < -100 | DEAD_DOWN (0) |
| Left | X < -175 | DEAD_LEFT (1) |
| Right | X > 175 | DEAD_RIGHT (2) |
| Top | Y > 180 | DEAD_UP_STAR (4) or DEAD_UP_FALL (6) |

---

## Summary Statistics

### Death State Comparison

| State | Median Duration | Total Frames | Instances |
|-------|-----------------|--------------|-----------|
| DEAD_DOWN (0) | 60 | 2,403,109 | 48,188 |
| DEAD_LEFT (1) | 60 | 650,978 | 12,933 |
| DEAD_RIGHT (2) | 60 | 717,177 | 14,151 |
| DEAD_UP_STAR (4) | 176 | 3,128,240 | 18,492 |
| DEAD_UP_FALL (6) | 51 | 182,274 | 3,575 |
| DEAD_UP_FALL_HIT_CAMERA (7) | 78 | 260,448 | 3,574 |

### Respawn State Comparison

| State | Median Duration | Total Frames | Instances |
|-------|-----------------|--------------|-----------|
| REBIRTH (12) | 60 | 4,870,744 | 81,229 |
| REBIRTH_WAIT (13) | 5 | 422,478 | 25,359 |

---

## Sources

### Primary Sources (Quantitative)
- **Parquet replay data**: `/Users/eppie/ssb_wiki/fox_vs_fox_parquet/` - 12.6 million frames of death/respawn data analyzed

### Documentation Sources (Qualitative)
- [SmashWiki - Star KO](https://www.ssbwiki.com/Star_KO)
- [SmashWiki - Screen KO](https://www.ssbwiki.com/Screen_KO)
- [SmashWiki - Blast line](https://www.ssbwiki.com/Blast_line)
- [SmashWiki - Revival platform](https://www.ssbwiki.com/Revival_platform)
- [SmashWiki - Invincibility](https://www.ssbwiki.com/Invincibility)

---

## Competitive Implications

1. **Star KO vs Screen KO**: The ~47-frame difference (176 vs 129) can determine whether a player gets punished after a risky kill move like Rest.

2. **Respawn Invincibility Management**: Players can maximize protection by:
   - Waiting on platform to observe opponent positioning
   - Dropping with 120 frames of invincibility to safely return
   - Using the full 8 seconds if needed to reset neutral

3. **Platform Camping**: In time-limited matches, a player ahead on stocks can stall on the respawn platform, though this is generally considered unsportsmanlike.

4. **RNG Factor**: The Star/Screen KO randomness (80/20 split) introduces unavoidable variance in competitive matches, though top players account for both possibilities.
