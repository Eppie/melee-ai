# Knockdown Action States (Missed Tech)

This document describes the action states that occur when a character misses a tech in Super Smash Bros. Melee.

## State IDs

### Face Up (`_U`) States
| ID | Internal Name | Description |
|----|---------------|-------------|
| 183 | `DOWN_BOUND_U` | Knockdown bounce (initial ground impact) |
| 184 | `DOWN_WAIT_U` | Lying on ground, awaiting input |
| 185 | `DOWN_DAMAGE_U` | Hit while lying down |
| 186 | `DOWN_STAND_U` | Neutral getup (stand in place) |
| 187 | `DOWN_ATTACK_U` | Getup attack |
| 188 | `DOWN_FOWARD_U` | Getup roll forward |
| 189 | `DOWN_BACK_U` | Getup roll backward |
| 190 | `DOWN_SPOT_U` | (Possibly unused) |

### Face Down (`_D`) States
| ID | Internal Name | Description |
|----|---------------|-------------|
| 191 | `DOWN_BOUND_D` | Knockdown bounce (face down) |
| 192 | `DOWN_WAIT_D` | Lying on ground (face down) |
| 193 | `DOWN_DAMAGE_D` | Hit while lying down (face down) |
| 194 | `DOWN_STAND_D` | Neutral getup (face down) |
| 195 | `DOWN_ATTACK_D` | Getup attack (face down) |
| 196 | `DOWN_FOWARD_D` | Getup roll forward (face down) |
| 197 | `DOWN_BACK_D` | Getup roll backward (face down) |
| 198 | `DOWN_SPOT_D` | (Possibly unused) |

### Other
| ID | Internal Name | Description |
|----|---------------|-------------|
| 335 | `DOWN_REFLECT` | Reflector-related knockdown |

---

## State Flow

```
Tumbling (DamageFall)
    │
    ▼ (lands without teching)
DOWN_BOUND_U/D (183/191) ─── Bounce animation, ~26 frames minimum
    │
    ▼ (automatic transition)
DOWN_WAIT_U/D (184/192) ─── Lying down, actionable
    │
    ├─► [Up/Shield] ──► DOWN_STAND_U/D (186/194) ─► Standing
    ├─► [A/B/Z] ──────► DOWN_ATTACK_U/D (187/195) ─► Standing
    ├─► [Forward] ────► DOWN_FOWARD_U/D (188/196) ─► Standing
    ├─► [Back] ───────► DOWN_BACK_U/D (189/197) ─► Standing
    └─► [Hit ≥7%] ────► DOWN_DAMAGE_U/D (185/193) ─► Knockback/Reset

If hit with <7% damage while in DOWN_WAIT: Jab reset (stays in DOWN_WAIT)
```

---

## Detailed State Descriptions

### DOWN_BOUND_U/D (183/191) - Knockdown Bounce

**Entry conditions:**
- Character lands on ground while in tumble (DamageFall) state
- No tech input within 20-frame window before landing
- Or tech lockout active (40 frames after failed tech)

**Behavior:**
- Character bounces/flops on ground
- Completely vulnerable to attacks
- No actions possible
- Inputs are buffered for DOWN_WAIT

**Duration:** Minimum ~26 frames (knockdown lag)

**Exit:** Automatic transition to DOWN_WAIT_U/D

---

### DOWN_WAIT_U/D (184/192) - Lying Down

**Entry conditions:**
- After DOWN_BOUND completes
- After jab reset (<7% hit while down)

**Behavior:**
- Character lies motionless
- Vulnerable to attacks
- Can input getup options
- Buffered inputs from DOWN_BOUND execute immediately

**Duration:** Until player input or auto-standup timeout

**Exit:** Based on input → corresponding getup state

---

### DOWN_DAMAGE_U/D (185/193) - Hit While Down

**Entry conditions:**
- Hit by attack dealing ≥7% while in DOWN_BOUND or DOWN_WAIT

**Behavior:**
- Takes damage and knockback
- Standard hitstun applies
- May reenter DOWN_BOUND if grounded, or launch into tumble

**Jab Reset:** Attacks dealing <7% do NOT enter this state; instead cause a "jab reset" where character pops up slightly and returns to DOWN_WAIT

---

### DOWN_STAND_U/D (186/194) - Neutral Getup

**Entry conditions:**
- Press Up or Shield while in DOWN_WAIT
- Or auto-standup after timeout

**Frame data (approximate):**
- Total duration: ~30-40 frames (character dependent)
- Invincibility: Partial, early frames
- Vulnerable: End of animation

---

### DOWN_ATTACK_U/D (187/195) - Getup Attack

**Entry conditions:**
- Press A, B, or Z while in DOWN_WAIT

**Frame data:**
- Total duration: ~50 frames
- Invincibility: ~29 frames
- Hitbox 1: Frame ~19, 2 frames duration
- Hitbox 2: Frame ~28, 2 frames duration
- Damage: ~5-6%
- Hits both sides

**Notes:**
- Long endlag makes it punishable
- Good invincibility but committal

---

### DOWN_FOWARD_U/D (188/196) - Forward Roll

**Entry conditions:**
- Hold forward while in DOWN_WAIT

**Frame data:**
- Total duration: ~36 frames
- Invincibility: ~23 frames
- Horizontal movement starts: ~Frame 8-12

---

### DOWN_BACK_U/D (189/197) - Backward Roll

**Entry conditions:**
- Hold backward while in DOWN_WAIT

**Frame data:**
- Total duration: ~36 frames
- Invincibility: ~19 frames (less than forward roll)
- Horizontal movement starts: ~Frame 8-12

**Notes:**
- Less invincibility than forward roll
- More easily punished

---

## Key Mechanics

| Mechanic | Value |
|----------|-------|
| Tech window | 20 frames before landing |
| Tech lockout | 40 frames after failed tech |
| Minimum knockdown duration | ~26 frames |
| Jab reset threshold | <7% damage |
| Getup attack invincibility | ~29 frames |
| Roll forward invincibility | ~23 frames |
| Roll backward invincibility | ~19 frames |

---

## Sources

- SmashWiki (Tumbling, Floor Recovery)
- Smashboards frame data threads
- Local action_state.json
- Parquet replay data analysis (see below)

---

## Empirical Data from Replays

Data source: `fox_vs_fox_parquet/` (Fox dittos only)

### State Duration Distribution

| State | ID | Min | Median | Avg | Max | Occurrences |
|-------|-----|-----|--------|-----|-----|-------------|
| DOWN_BOUND_U | 183 | 1 | **26** | 27.2 | 135 | 164,191 |
| DOWN_WAIT_U | 184 | 1 | 8 | 12.6 | 190 | 29,628 |
| DOWN_DAMAGE_U | 185 | 4 | 16 | 14.1 | 18 | 173 |
| DOWN_STAND_U | 186 | 1 | **30** | 28.4 | 35 | 33,375 |
| DOWN_ATTACK_U | 187 | 1 | **46** | 44.2 | 67 | 30,722 |
| DOWN_FOWARD_U | 188 | 7 | **35** | 33.7 | 39 | 7,350 |
| DOWN_BACK_U | 189 | 1 | **35** | 34.4 | 38 | 6,512 |
| DOWN_BOUND_D | 191 | 1 | 26 | 20.0 | 85 | 43,397 |
| DOWN_WAIT_D | 192 | 1 | 7 | 11.9 | 220 | 7,285 |
| DOWN_DAMAGE_D | 193 | 1 | 16 | 15.0 | 46 | 22,847 |
| DOWN_STAND_D | 194 | 1 | 28 | 27.0 | 33 | 14,780 |
| DOWN_ATTACK_D | 195 | 19 | 49 | 47.1 | 60 | 5,131 |
| DOWN_FOWARD_D | 196 | 1 | 35 | 33.3 | 38 | 20,647 |
| DOWN_BACK_D | 197 | 1 | 35 | 34.2 | 40 | 35,058 |

**Key findings:**
- DOWN_BOUND median of 26 frames confirms the "~26 frames minimum knockdown lag"
- DOWN_STAND median of 30 frames confirms "~30-40 frames" for neutral getup
- DOWN_ATTACK median of 46 frames matches "~50 frames" (slightly shorter for Fox)
- Rolls consistently 35 frames median

### Invulnerability Analysis

| State | ID | Invuln Frames | Vuln Frames | Invuln % |
|-------|-----|---------------|-------------|----------|
| DOWN_STAND_U | 186 | ~24 | ~6 | 78.8% |
| DOWN_ATTACK_U | 187 | **~29** | ~17 | 63.0% |
| DOWN_FOWARD_U | 188 | ~20 | ~15 | 56.2% |
| DOWN_BACK_U | 189 | ~18 | ~17 | 51.9% |
| DOWN_STAND_D | 194 | ~23 | ~5 | 83.9% |
| DOWN_ATTACK_D | 195 | ~30 | ~19 | 61.6% |
| DOWN_FOWARD_D | 196 | ~20 | ~15 | 57.1% |
| DOWN_BACK_D | 197 | ~25 | ~10 | 70.1% |

**Confirmation:** Getup attack has ~29 invuln frames, matching documentation.

### Entry Conditions (What leads to knockdown)

Top states transitioning INTO DOWN_BOUND:

| From State | Name | → DOWN_BOUND_U | → DOWN_BOUND_D |
|------------|------|----------------|----------------|
| 88 | DAMAGE_FLY_N | 133,461 | 15,208 |
| 90 | DAMAGE_FLY_TOP | 57,920 | 7,698 |
| 87 | DAMAGE_FLY_HI | 18,881 | 11,473 |
| 89 | DAMAGE_FLY_LW | 11,495 | 7,568 |
| 91 | DAMAGE_FLY_ROLL | 1,903 | 4,166 |
| 38 | DAMAGE_FALL | - | 984 |

**Confirms:** Knockdown occurs when landing from any DAMAGE_FLY (tumble/launched) state.

### State Transition Patterns

From DOWN_BOUND_U (183):
- → DOWN_WAIT_U (184): 29,570 (normal flow)
- → DOWN_BACK_D (197): 28,204 (buffered roll)
- → DOWN_ATTACK_U (187): 24,634 (buffered getup attack)
- → DOWN_STAND_U (186): 23,001 (buffered neutral getup)
- → DOWN_DAMAGE_D (193): 19,070 (hit while bouncing)

From DOWN_WAIT_U (184):
- → DOWN_STAND_U (186): 10,363 (neutral getup)
- → DOWN_FOWARD_U (188): 6,160 (forward roll)
- → DOWN_ATTACK_U (187): 6,085 (getup attack)
- → DOWN_BACK_U (189): 5,010 (backward roll)
- → DOWN_DAMAGE_U (185): 137 (hit ≥7% while down)

