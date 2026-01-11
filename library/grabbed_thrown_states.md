# Grabbed & Thrown (Victim) States Documentation

Action states 223-230 and 239-242 covering being grabbed, held, pummeled, escaping, and being thrown.

## Overview

These states represent the victim's perspective during grabs. Melee distinguishes between aerial grabs (HI variants) and grounded grabs (LW variants). The grabbed player can mash to escape faster, but in competitive Fox dittos, grabs almost always result in immediate throws rather than extended pummel sequences.

| State | ID | Internal Name | Instances | % of Gameplay | Median Duration |
|-------|-----|---------------|-----------|---------------|-----------------|
| Aerial Grab Pull | 223 | CAPTURE_PULLED_HI | 27,489 | 0.02% | 2 frames |
| Aerial Grab Hold | 224 | CAPTURE_WAIT_HI | 28,000 | 0.03% | 2 frames |
| Aerial Pummel | 225 | CAPTURE_DAMAGE_HI | 4,366 | 0.03% | 23 frames |
| Ground Grab Pull | 226 | CAPTURE_PULLED_LW | 140,596 | 0.10% | 2 frames |
| Ground Grab Hold | 227 | CAPTURE_WAIT_LW | 136,117 | 0.17% | 2 frames |
| Ground Pummel | 228 | CAPTURE_DAMAGE_LW | 23,602 | 0.19% | 23 frames |
| Grab Escape | 229 | CAPTURE_CUT | 239 | 0.002% | 30 frames |
| Jump Escape | 230 | CAPTURE_JUMP | 120 | 0.001% | 18 frames |
| Forward Thrown | 239 | THROWN_F | 8,522 | 0.04% | 13 frames |
| Back Thrown | 240 | THROWN_B | 12,416 | 0.03% | 6 frames |
| Up Thrown | 241 | THROWN_HI | 143,955 | 0.25% | 5 frames |
| Down Thrown | 242 | THROWN_LW | 825 | 0.01% | 32 frames |

**Total grabbed/thrown frames**: 2,485,161 (0.86% of gameplay)

---

## Aerial vs Ground Grab States

Melee has separate states for aerial and grounded grabs:

| Type | Pull State | Hold State | Pummel State | On Ground |
|------|------------|------------|--------------|-----------|
| Aerial (HI) | 223 | 224 | 225 | 0-0.4% |
| Ground (LW) | 226 | 227 | 228 | 100% |

**Aerial grabs** occur when the grabbed character is airborne (e.g., grabbed out of a jump or during up throw follow-ups).

**Ground grabs** occur when the grabbed character is grounded.

---

## State 223: CAPTURE_PULLED_HI (Aerial Grab Pull)

### Description
The initial frame(s) when an airborne character is grabbed. The victim is pulled toward the grabber before entering the hold state.

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 2 frames |
| Average Duration | 1.9 frames |
| Maximum Duration | 2 frames |
| On Ground | 0% |
| Invulnerability | 0% |

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| DAMAGE_FLY_TOP (90) | 22,351 | 81.3% |
| CAPTURE_PULLED_LW (226) | 1,950 | 7.1% |
| FALL (29) | 662 | 2.4% |
| PASS (244) | 512 | 1.9% |
| ATTACK_AIR_B (67) | 247 | 0.9% |
| DAMAGE_FLY_N (88) | 199 | 0.7% |

**Critical finding**: **81.3% of aerial grabs come from DAMAGE_FLY_TOP (90)** - This is the up throw → regrab pattern, where Fox up throws then immediately regrabs the opponent while they're still in upward knockback.

### Exit Conditions
Always transitions to CAPTURE_WAIT_HI (224).

---

## State 224: CAPTURE_WAIT_HI (Aerial Grab Hold)

### Description
The hold state for aerial grabs. The grabber can pummel or throw from this state. Duration depends on victim's mashing and damage percentage.

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 2 frames |
| Average Duration | 3.5 frames |
| Maximum Duration | 69 frames |
| On Ground | 0.2% |
| Invulnerability | 0% |

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| THROWN_HI (241) | 19,557 | 69.8% |
| CAPTURE_DAMAGE_HI (225) | 4,346 | 15.5% |
| THROWN_B (240) | 2,712 | 9.7% |
| THROWN_F (239) | 1,065 | 3.8% |
| THROWN_LW (242) | 263 | 0.9% |
| CAPTURE_JUMP (230) | 33 | 0.1% |
| CAPTURE_CUT (229) | 23 | 0.1% |

**Key insight**: **69.8% up throw** from aerial holds - consistent with ground grabs. The up throw → regrab → up throw pattern is a core Fox combo.

---

## State 225: CAPTURE_DAMAGE_HI (Aerial Pummel)

### Description
The state when being pummeled during an aerial grab. Each pummel deals 3% damage.

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 23 frames |
| Average Duration | 22.5 frames |
| Maximum Duration | 26 frames |
| On Ground | 0.4% |
| Invulnerability | 0% |

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| CAPTURE_WAIT_HI (224) | 3,909 | 89.5% |
| THROWN_HI (241) | 354 | 8.1% |
| THROWN_B (240) | 70 | 1.6% |
| THROWN_LW (242) | 17 | 0.4% |
| THROWN_F (239) | 16 | 0.4% |

**89.5% return to hold** after pummel, allowing additional pummels or throw.

---

## State 226: CAPTURE_PULLED_LW (Ground Grab Pull)

### Description
The initial frame(s) when a grounded character is grabbed. Most common grab state entry point.

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 2 frames |
| Average Duration | 2.0 frames |
| Maximum Duration | 5 frames |
| On Ground | 100% |
| Invulnerability | 0% |

### Damage When Grabbed
| Metric | Value |
|--------|-------|
| Average % | 43.0% |
| Median % | 34% |
| 25th Percentile | 11% |
| 75th Percentile | 71% |
| Maximum % | 248% |

### Entry Conditions
Ground grabs come from various situations:
- Shield pressure (opponent grabbed out of shield)
- Whiff punishes (opponent grabbed after missing attack)
- Tech chases (opponent grabbed during getup)
- Neutral (raw grab in neutral)

### Exit Conditions
Always transitions to CAPTURE_WAIT_LW (227).

---

## State 227: CAPTURE_WAIT_LW (Ground Grab Hold)

### Description
The hold state for grounded grabs. The most common grab hold state, representing the majority of grab situations.

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 2 frames |
| Average Duration | 3.7 frames |
| Maximum Duration | 160 frames |
| On Ground | 100% |
| Invulnerability | 0% |

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| THROWN_HI (241) | 97,491 | 71.6% |
| CAPTURE_DAMAGE_LW (228) | 23,602 | 17.3% |
| THROWN_B (240) | 8,047 | 5.9% |
| THROWN_F (239) | 6,352 | 4.7% |
| THROWN_LW (242) | 381 | 0.3% |
| CAPTURE_CUT (229) | 129 | 0.1% |
| CAPTURE_JUMP (230) | 87 | 0.1% |

**Key insight**: **71.6% up throw** - Fox's up throw is the dominant throw option due to its combo potential into up air.

---

## State 228: CAPTURE_DAMAGE_LW (Ground Pummel)

### Description
Being pummeled during a grounded grab. Each pummel is 23-26 frames and deals 3% damage.

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 23 frames |
| Average Duration | 22.6 frames |
| Maximum Duration | 26 frames |
| On Ground | 100% |
| Invulnerability | 0% |

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| CAPTURE_WAIT_LW (227) | 21,109 | 89.4% |
| THROWN_HI (241) | 1,868 | 7.9% |
| THROWN_B (240) | 342 | 1.4% |
| THROWN_F (239) | 191 | 0.8% |
| THROWN_LW (242) | 72 | 0.3% |

---

## Pummel Frequency Analysis

| Pummels per Grab | Count | % of Grabs |
|------------------|-------|------------|
| 0 pummels | 302,691 | 92.9% |
| 1 pummel | 18,795 | 5.8% |
| 2 pummels | 3,994 | 1.2% |
| 3 pummels | 359 | 0.1% |
| 4+ pummels | 21 | <0.01% |

**Critical finding**: **92.9% of grabs have zero pummels** - In competitive Fox dittos, players almost always throw immediately without pummeling. This is because:
1. Up throw → up air combo is time-sensitive
2. Extra pummel damage doesn't significantly change combo potential
3. Mashing can allow escape if grabber delays

---

## State 229: CAPTURE_CUT (Grab Escape)

### Description
Successfully mashing out of a grab. Very rare in competitive play - only 239 instances in the entire dataset (0.1% of grabs).

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 30 frames |
| Average Duration | 22.6 frames |
| Maximum Duration | 32 frames |
| On Ground | 80.4% |
| Invulnerability | 0% |

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| CAPTURE_WAIT_LW (227) | 129 | 53.8% |
| CAPTURE_PULLED_HI (223) | 86 | 35.8% |
| CAPTURE_WAIT_HI (224) | 23 | 9.6% |

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| CLIFF_CATCH (252) | 76 | 31.8% |
| WAIT (14) | 61 | 25.5% |
| SQUAT (39) | 31 | 13.0% |
| FALL (29) | 16 | 6.7% |
| ATTACK_11 (44) | 16 | 6.7% |
| GUARD_ON (178) | 14 | 5.9% |

**Key insight**: 31.8% exit to ledge grab - grab escapes often occur near the ledge where the escaped player immediately grabs ledge for safety.

---

## State 230: CAPTURE_JUMP (Jump Escape)

### Description
Escaping a grab by jumping out. Even rarer than standard grab escape - only 120 instances total.

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 18 frames |
| Average Duration | 19.1 frames |
| Maximum Duration | 41 frames |
| On Ground | 0% |
| Invulnerability | 0% |

### Entry Conditions
| Source State | Occurrences | % of Entries |
|--------------|-------------|--------------|
| CAPTURE_WAIT_LW (227) | 87 | 72.5% |
| CAPTURE_WAIT_HI (224) | 33 | 27.5% |

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| LANDING (42) | 111 | 92.5% |
| JUMP_AERIAL_F (27) | 6 | 5.0% |
| JUMP_AERIAL_B (28) | 2 | 1.7% |

**92.5% land immediately** - Jump escapes typically occur at very low height, resulting in immediate landing.

---

## Throw States (239-242)

### Throw Distribution

| Throw | State | Instances | % of Throws |
|-------|-------|-----------|-------------|
| Up Throw | 241 | 143,955 | 86.8% |
| Back Throw | 240 | 12,416 | 7.5% |
| Forward Throw | 239 | 8,522 | 5.1% |
| Down Throw | 242 | 825 | 0.5% |

**Up throw dominance**: 86.8% of all throws are up throws - this reflects Fox's optimal punish game where up throw → up air is the primary combo.

---

## State 239: THROWN_F (Forward Throw)

### Description
Being thrown forward. Fox's forward throw sends opponents at a low angle.

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 13 frames |
| Average Duration | 12.9 frames |
| Maximum Duration | 16 frames |
| On Ground | 86.6% |
| Invulnerability | 0% |

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| DAMAGE_AIR_3 (86) | 4,661 | 54.7% |
| DAMAGE_FLY_N (88) | 3,770 | 44.2% |
| DAMAGE_FLY_ROLL (91) | 91 | 1.1% |

Forward throw sends at a diagonal-horizontal angle.

---

## State 240: THROWN_B (Back Throw)

### Description
Being thrown backward. Fox's back throw has strong horizontal knockback, useful for positioning opponents toward the ledge.

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 6 frames |
| Average Duration | 6.0 frames |
| Maximum Duration | 6 frames |
| On Ground | 74.6% |
| Invulnerability | 0% |

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| DAMAGE_FLY_N (88) | 12,333 | 99.3% |
| DAMAGE_FLY_ROLL (91) | 83 | 0.7% |

**99.3% horizontal knockback** - Back throw consistently sends at a horizontal angle.

---

## State 241: THROWN_HI (Up Throw)

### Description
Being thrown upward. Fox's signature throw - the setup for up throw → up air, one of Melee's most iconic combos. By far the most used throw.

### Frame Data
| Metric | Value |
|--------|-------|
| Median Duration | 5 frames |
| Average Duration | 4.9 frames |
| Maximum Duration | 10 frames |
| On Ground | 83.0% |
| Invulnerability | 0% |

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| DAMAGE_FLY_TOP (90) | 143,955 | 100.0% |

**100% upward knockback** - Up throw always sends directly upward, perfect for up air follow-ups.

### Position Data
| Metric | Value |
|--------|-------|
| Average Y | 3.9 |
| Min Y | -14.9 |
| Max Y | 64.0 |
| Std Dev Y | 9.12 |

---

## State 242: THROWN_LW (Down Throw)

### Description
Being thrown downward. Fox's down throw is a multi-hit move that shoots lasers before releasing. Rarely used competitively due to poor combo potential.

### Frame Data (Fox)
| Metric | Value |
|--------|-------|
| Median Duration | 32 frames |
| Average Duration | 32.0 frames |
| Total Animation | 43 frames |
| On Ground | 63.6% |
| Invulnerability | 0% |

### Damage Breakdown (Fox Down Throw)
| Component | Damage |
|-----------|--------|
| Throw | 1% |
| Release | 3% |
| Laser shots (4) | 2% each |
| **Total** | **12%** |

### Exit Conditions
| Target State | Occurrences | % of Exits |
|--------------|-------------|------------|
| DAMAGE_FLY_TOP (90) | 434 | 52.6% |
| PASSIVE (199) | 145 | 17.6% |
| PASSIVE_STAND_F (200) | 101 | 12.2% |
| DOWN_BOUND_U (183) | 94 | 11.4% |
| PASSIVE_STAND_B (201) | 51 | 6.2% |

**Key patterns**:
- 52.6% upward knockback (at low %)
- 35.9% tech situations (199-201) - victims can tech the bounce
- 11.4% missed tech (183)

---

## Grab Escape Mechanics

### Grab Duration Formula
Base grab duration: **76 + 1.6p – 15h frames**
- p = victim's damage percentage
- h = victim's rank disadvantage (1st vs 4th place = 3)

### Mashing Out
- Each button press reduces grab duration by **6 frames**
- Optimal mashing involves alternating between buttons
- In competitive play, escape rate is only **0.1%** due to immediate throws

### External Hit Escape
- Grabbed opponents take **50% damage** from external attacks
- Escape occurs if hit with **6% or more** (after reduction)
- This enables techniques like "wobbling" (ICs infinite)

---

## Competitive Applications

### Up Throw Dominance
The data shows overwhelming preference for up throw:
- **86.8% of all throws** are up throws
- Up throw → up air is Fox's bread-and-butter combo
- Works at virtually all percentages

### Pummel vs Throw
- **92.9% zero pummel** grabs
- Immediate throw is optimal for combo potential
- Pummeling only recommended when:
  - Opponent at very low % (need extra damage for kill)
  - Near kill % (extra damage for kill confirm)

### Grab Escape Rarity
- Only **0.1% of grabs** result in escape
- Immediate throws prevent mash-out opportunities
- Grab escapes mostly occur when grabber hesitates

### Aerial Regrab Pattern
- **81.3% of aerial grabs** come from up throw knockback
- Up throw → regrab → up throw is a core Fox combo at mid-%
- Creates extended combo sequences

---

## Related States

| Relationship | States |
|--------------|--------|
| Attacker states | CATCH (212), CATCH_WAIT (216), CATCH_ATTACK (217), THROW_F/B/HI/LW (219-222) |
| Entry from | Various (grounded states for 226, DAMAGE_FLY_TOP for 223) |
| Throw exits to | Damage states (86-91), tech states (199-201), knockdown (183) |
| Escape exits to | Neutral (14), ledge (252), defensive options |

---

## Data Sources

- Parquet replay data: 288,753,124 total frames analyzed
- SmashWiki: Grab mechanics, escape formula
- SmashWiki: Fox down throw frame data
- Action state analysis from Fox ditto dataset
