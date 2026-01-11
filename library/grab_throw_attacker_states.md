# Grab, Pummel & Throws (Attacker) - Action States 212-222

This document describes the attacker-side grab and throw mechanics, covering standing grabs, dash grabs, holding opponents, pummeling, and all four throw directions.

## State Overview

| State ID | Internal Name | Description |
|----------|---------------|-------------|
| 212 | CATCH | Standing grab startup |
| 213 | CATCH_PULL | Standing grab connects, pulling opponent |
| 214 | CATCH_DASH | Dash grab startup |
| 215 | CATCH_DASH_PULL | Dash grab connects, pulling opponent |
| 216 | CATCH_WAIT | Holding grabbed opponent |
| 217 | CATCH_ATTACK | Pummel attack |
| 218 | CATCH_CUT | Grab break (opponent mashes out) |
| 219 | THROW_F | Forward throw |
| 220 | THROW_B | Back throw |
| 221 | THROW_HI | Up throw |
| 222 | THROW_LW | Down throw |

---

## State Details

### State 212: CATCH (Standing Grab)

**Description**: The standing grab initiation. Can be performed from standing idle or, more commonly, from jumpsquat via jump-cancel grab (JC grab).

**Frame Data** (Fox):
- Active frames: 7-8 (2-frame window)
- Total animation: 30 frames
- Grab hitbox radius: 3.906 units

**Entry Conditions**:
| From State | % of Entries | Context |
|------------|--------------|---------|
| 24 (KNEE_BEND) | 74.6% | JC grab from jumpsquat |
| 14 (WAIT) | 12.8% | From idle/standing |
| 43 (LANDING_FALL_SPECIAL) | 5.2% | Post-wavedash |
| Other | 7.4% | Various |

**Exit Conditions**:
| To State | % of Exits | Context |
|----------|------------|---------|
| 213 (CATCH_PULL) | 57.4% | Grab connects |
| 14 (WAIT) | 12.2% | Grab whiffs to idle |
| 88 (DAMAGE_FLY_N) | 8.1% | Hit during grab |
| 178 (GUARD_ON) | 4.8% | Trades/interrupt |

**Key Properties**:
- Duration: 6 frames median, 6.0 avg, max 9 (from parquet)
- Ground-only state (on_ground = true)
- No invulnerability

**Technical Notes**:
- JC grab is overwhelmingly preferred (74.6%) because it has less endlag than dash grab while maintaining mobility
- Performed by pressing shield during jumpsquat (frames 1-3 for Fox)
- Input: Y/X → Z within jumpsquat frames

---

### State 213: CATCH_PULL (Standing Grab Connect)

**Description**: Transitional state when standing grab successfully connects. The attacker pulls the opponent into grab hold position.

**Entry Conditions**:
- Only from 212 (CATCH) - successful grab hitbox connection

**Exit Conditions**:
- Always to 216 (CATCH_WAIT) - holding opponent

**Key Properties**:
- Duration: 1-3 frames (transitional)
- Ground-only state

---

### State 214: CATCH_DASH (Dash Grab)

**Description**: Dash grab performed while running. Has more startup and significantly more endlag than standing grab.

**Frame Data** (Fox):
- Active frames: 12-13 (2-frame window)
- Total animation: 40 frames
- 10 more frames endlag than standing grab on whiff

**Entry Conditions**:
| From State | % of Entries | Context |
|------------|--------------|---------|
| 20 (DASH) | 86.7% | Direct from dash |
| 21 (RUN) | 9.8% | From run |
| Other | 3.5% | Various |

**Exit Conditions**:
| To State | % of Exits | Context |
|----------|------------|---------|
| 215 (CATCH_DASH_PULL) | 37.1% | Grab connects |
| 14 (WAIT) | 62.9% | Grab whiffs |

**Key Properties**:
- Duration: 11 frames median, 11.5 avg (from parquet)
- 10,265 instances in dataset
- 37.1% success rate (lower than standing grab due to more reactable startup)
- Ground-only state
- No invulnerability

**Technical Notes**:
- Rarely used in high-level play due to punishable endlag
- Standing grab success rate is higher
- Sometimes used for extended range or catching opponents retreating

---

### State 215: CATCH_DASH_PULL (Dash Grab Connect)

**Description**: Transitional state when dash grab successfully connects.

**Entry Conditions**:
- Only from 214 (CATCH_DASH) - successful grab hitbox connection

**Exit Conditions**:
- Always to 216 (CATCH_WAIT) - holding opponent

**Key Properties**:
- Duration: 1-3 frames (transitional)
- Ground-only state

---

### State 216: CATCH_WAIT (Grab Hold)

**Description**: The state where the attacker holds a grabbed opponent. From here, the attacker can pummel or throw in any direction. Duration depends on opponent's percent and mashing.

**Grab Duration Formula**:
```
Duration = floor(76 + 1.6p - 15h) frames
```
Where:
- p = victim's damage percentage
- h = handicap rank disadvantage (usually 0)

Button mashing by the victim reduces this by 6 frames per input.

**Entry Conditions**:
| From State | % of Entries | Context |
|------------|--------------|---------|
| 213 (CATCH_PULL) | ~75% | From standing grab |
| 215 (CATCH_DASH_PULL) | ~5% | From dash grab |
| 217 (CATCH_ATTACK) | ~20% | After pummel |

**Exit Conditions**:
| To State | % of Exits | Context |
|----------|------------|---------|
| 221 (THROW_HI) | 73.7% | Up throw |
| 217 (CATCH_ATTACK) | 14.3% | Pummel |
| 220 (THROW_B) | 6.8% | Back throw |
| 219 (THROW_F) | 4.7% | Forward throw |
| 222 (THROW_LW) | 0.4% | Down throw |
| 218 (CATCH_CUT) | 0.2% | Opponent mashes out |

**Key Properties**:
- Variable duration based on opponent percent
- Ground-only state
- Can act at any frame with pummel or throw inputs

**Analysis**:
Up throw is by far the most common choice (73.7%), reflecting Fox's powerful up-throw → up-air combo game. Only 14.3% of holds include any pummel before throwing, indicating players prioritize immediate throw follow-ups over damage accumulation.

---

### State 217: CATCH_ATTACK (Pummel)

**Description**: The pummel attack performed while holding an opponent. Deals small damage but resets grab hold timer slightly.

**Entry Conditions**:
- Only from 216 (CATCH_WAIT) via A button press

**Exit Conditions**:
- Always returns to 216 (CATCH_WAIT)

**Key Properties**:
- Duration: ~24-26 frames typically
- Ground-only state
- Low damage (1-3% depending on character)

**Technical Notes**:
- Only 14.3% of grab holds include any pummel
- At low percent, immediate throw is preferred for combo potential
- Pummels become more valuable at high percent to rack up extra damage before kill throw
- Excessive pummeling gives opponent more time to mash out

---

### State 218: CATCH_CUT (Grab Break)

**Description**: The state when an opponent successfully mashes out of a grab before a throw is executed.

**Entry Conditions**:
- Only from 216 (CATCH_WAIT) when opponent mashes enough

**Exit Conditions**:
- Returns to neutral (14 WAIT or similar)

**Key Properties**:
- Only 0.8% of grabs end in grab break
- Both players have slight frame disadvantage
- Rare occurrence due to players throwing before mash-out

---

### State 219: THROW_F (Forward Throw)

**Description**: Forward throw. Launches opponent in the direction the attacker is facing.

**Entry Conditions**:
- Only from 216 (CATCH_WAIT) via forward + A/Z

**Exit Conditions**:
- Returns to neutral after throw animation completes

**Key Properties**:
- Duration: ~30-40 frames total
- Invulnerability: Frames 1-8 (universal for all throws)
- 13.3% of all throws
- Ground-only state

**Usage**:
- Primarily used to gain stage position
- Sets up edgeguard situations
- Less combo potential than up throw for Fox

---

### State 220: THROW_B (Back Throw)

**Description**: Back throw. Turns and launches opponent behind the attacker's original facing direction.

**Entry Conditions**:
- Only from 216 (CATCH_WAIT) via back + A/Z

**Exit Conditions**:
- Returns to neutral after throw animation completes

**Key Properties**:
- Duration: ~35-45 frames total
- Invulnerability: Frames 1-8
- 18.3% of all throws
- Ground-only state

**Usage**:
- Position-dependent throw for stage control
- Good for setting up edgeguards at ledge
- Fox's second most used throw

---

### State 221: THROW_HI (Up Throw)

**Description**: Up throw. Launches opponent upward. Fox's primary throw for combo follow-ups.

**Entry Conditions**:
- Only from 216 (CATCH_WAIT) via up + A/Z

**Exit Conditions**:
| To State | % of Exits | Context |
|----------|------------|---------|
| 14 (WAIT) | ~80% | Standard recovery |
| 25/26 (JUMP) | ~15% | Jump to follow up |
| Other | ~5% | Various |

**Key Properties**:
- Duration: 29 frames median, 29.2 avg, 29-34 range
- Invulnerability: **Frames 1-8** (100% invuln rate in data)
- **67.1% of all throws** (40,009 instances)
- Ground-only state

**Frame-by-Frame Invulnerability**:
| Frame | Invuln Rate |
|-------|-------------|
| 1-8 | 100% |
| 9+ | ~2% (throw animation ending) |

**Usage**:
- Fox's bread-and-butter throw
- Sets up up-air follow-ups at almost all percents
- Enables waveshining regrab at low percent
- The foundation of Fox's punish game

**Technical Notes**:
- The 8 frames of throw invulnerability protect against bystander interference in doubles
- Weight-dependent throw speed: lighter characters are thrown faster
- Knockback calculated using weight = 100 regardless of opponent's actual weight

---

### State 222: THROW_LW (Down Throw)

**Description**: Down throw. Slams opponent into the ground. Has unique properties for tech chasing.

**Entry Conditions**:
- Only from 216 (CATCH_WAIT) via down + A/Z

**Exit Conditions**:
- Returns to neutral after throw animation completes

**Key Properties**:
- Duration: ~35-40 frames total
- Invulnerability: Frames 1-8
- Only 1.3% of all throws (788 instances)
- Ground-only state

**Usage**:
- Creates tech chase situations
- Can lead to regrab if opponent misses tech
- Rarely optimal for Fox compared to up throw

---

## Statistical Summary

### Grab Type Distribution
| Grab Type | Instances | % of Total |
|-----------|-----------|------------|
| Standing Grab (212) | 179,363 | 94.6% |
| Dash Grab (214) | 10,265 | 5.4% |

### Throw Distribution
| Throw | Instances | % of Throws |
|-------|-----------|-------------|
| Up Throw (221) | 40,009 | 67.1% |
| Back Throw (220) | 10,933 | 18.3% |
| Forward Throw (219) | 7,901 | 13.3% |
| Down Throw (222) | 788 | 1.3% |

### JC Grab Dominance

**74.6%** of all standing grabs come from KNEE_BEND (jumpsquat), confirming that JC grab is the dominant grab technique in competitive Melee. This is because:
1. JC grab has less endlag than dash grab (30 vs 40 total frames)
2. JC grab can be performed from run (run → jumpsquat → grab)
3. Standing grab has better frame data than dash grab (7-8 vs 12-13 active)

### Pummel vs Immediate Throw

Only **14.3%** of grab holds include any pummel before throwing. This indicates:
- Immediate throws prioritize combo potential
- At low-mid percent, extending combos is more valuable than ~3% pummel damage
- Players only pummel when grab duration is high (opponent at high %) or throw won't lead to follow-ups

---

## State Transition Diagram

```
                      ┌─────────────┐
                      │ KNEE_BEND   │ (74.6%)
                      │ (Jumpsquat) │
                      └──────┬──────┘
                             │
                      ┌──────▼──────┐        ┌─────────────┐
        ┌─────────────│    CATCH    │◄───────│    WAIT     │ (12.8%)
        │             │    (212)    │        │    (14)     │
        │             └──────┬──────┘        └─────────────┘
        │ Whiff              │ Connect
        │ (~60%)             │ (~40%)
        ▼                    ▼
   ┌─────────┐        ┌─────────────┐
   │  WAIT   │        │ CATCH_PULL  │
   │  (14)   │        │   (213)     │
   └─────────┘        └──────┬──────┘
                             │
                      ┌──────▼──────┐
                      │ CATCH_WAIT  │◄─────────┐
                      │    (216)    │──────────┤
                      └──────┬──────┘          │
                             │                 │
        ┌────────┬───────┬───┴───┬───────┬─────┴────┐
        │        │       │       │       │          │
        ▼        ▼       ▼       ▼       ▼          │
   THROW_HI  THROW_B  THROW_F  THROW_LW  CATCH_CUT  │ (loop)
     (221)    (220)    (219)    (222)     (218)     │
    67.1%    18.3%    13.3%     1.3%      0.2%      │
                                                    │
                      ┌─────────────┐               │
                      │CATCH_ATTACK │───────────────┘
                      │    (217)    │    (14.3%)
                      └─────────────┘
```

---

## Competitive Implications

### The JC Grab Meta
The data shows that JC grab (74.6% of all grabs) is essential to competitive Melee. The technique:
- Combines run momentum with standing grab's superior frame data
- Reduces whiff punishment risk vs dash grab
- Is fundamental to Fox's punish game

### Up Throw Dominance
Fox's up throw comprising 67.1% of all throws reflects:
- Reliable combo starter at all percents
- Up throw → up-air is Fox's most consistent kill confirm
- Weight-independent knockback calculation favors this as universal option
- Back throw (18.3%) used for stage positioning and edgeguards
- Forward throw (13.3%) provides positional advantage

### Low Pummel Rate
The 14.3% pummel rate indicates optimal play prioritizes:
- Combo extensions over raw damage
- Taking guaranteed follow-ups immediately
- Reducing opponent's mash-out opportunity

---

## Sources
- Parquet replay data (Fox vs Fox matches)
- action_state.json (state ID mappings)
- SmashWiki: Grab, Throw articles
- Smashboards RAG: JC grab technique threads
