# Master Action State Reference

Complete documentation for 183 action states in Super Smash Bros. Melee, organized into 20 functional groups. Each state includes its ID, internal name, brief description, and a reference to detailed documentation.

---

## Quick Navigation

| Group | States | Documentation File |
|-------|--------|-------------------|
| [1. Death & Respawn](#group-1-death--respawn) | 0-13 | `death_respawn_states.md` |
| [2. Movement](#group-2-movement-stand-walk-dash-run) | 14-24 | `movement_states.md` |
| [3. Jumps & Falls](#group-3-jumps--falls) | 25-38 | `jump_fall_states.md` |
| [4. Crouch & Landing](#group-4-crouch--landing) | 39-43 | `crouch_landing_states.md` |
| [5. Jabs & Tilts](#group-5-grounded-attacks-jabs--tilts) | 44-57 | `grounded_attacks_states.md` |
| [6. Smashes](#group-6-grounded-attacks-smashes) | 58-64 | `smash_attacks_states.md` |
| [7. Aerials & Landing Lag](#group-7-aerial-attacks--landing-lag) | 65-74 | `aerial_attacks_states.md` |
| [8. Grounded Hitstun](#group-8-damage-grounded-hitstun) | 75-83 | `grounded_hitstun_states.md` |
| [9. Aerial Hitstun & Tumble](#group-9-damage-aerial-hitstun--tumble) | 84-91 | `aerial_hitstun_states.md` |
| [10. Shield & Powershield](#group-10-shield--powershield) | 178-182, 205-211 | `shield_states.md` |
| [11. Knockdown & Getup](#group-11-knockdown--getup) | 183-197, 335 | `knockdown_states.md` |
| [12. Tech (Successful)](#group-12-tech-successful) | 199-204 | `tech_states.md` |
| [13. Grab & Throws (Attacker)](#group-13-grab-pummel--throws-attacker) | 212-222 | `grab_throw_attacker_states.md` |
| [14. Grabbed & Thrown (Victim)](#group-14-grabbed--thrown-victim) | 223-230, 239-242 | `grabbed_thrown_states.md` |
| [15. Rolls & Dodges](#group-15-defensive-options-rolls--dodges) | 233-238 | `defensive_options_states.md` |
| [16. Stage Interactions](#group-16-stage-interactions) | 244-251 | `stage_interaction_states.md` |
| [17. Ledge States](#group-17-ledge-states) | 252-263 | `ledge_taunt_states.md` |
| [18. Taunt](#group-18-taunt) | 264 | `ledge_taunt_states.md` |
| [19. Fox: Blaster & Illusion](#group-19-fox-blaster--illusion) | 341-352 | `fox_blaster_illusion_states.md` |
| [20. Fox: Fire Fox & Shine](#group-20-fox-fire-fox--shine) | 353-369 | `fox_firefox_shine_states.md` |

---

## Group 1: Death & Respawn

**Documentation**: [`death_respawn_states.md`](death_respawn_states.md)

Death animations for each blast zone and respawn mechanics.

| ID | Name | Description | Key Properties |
|----|------|-------------|----------------|
| 0 | DEAD_DOWN | Death from bottom blast zone | Standard SD |
| 1 | DEAD_LEFT | Death from left blast zone | Horizontal KO |
| 2 | DEAD_RIGHT | Death from right blast zone | Horizontal KO |
| 4 | DEAD_UP_STAR | Death from top blast zone (star KO) | Long animation, can affect game timer |
| 6 | DEAD_UP_FALL | Fast upward death | Shorter than star KO |
| 7 | DEAD_UP_FALL_HIT_CAMERA | Upward death hitting camera | Screen zoom effect |
| 12 | REBIRTH | Respawn platform descent | Invulnerable |
| 13 | REBIRTH_WAIT | Waiting on respawn platform | Invulnerable until drop |

---

## Group 2: Movement: Stand, Walk, Dash, Run

**Documentation**: [`movement_states.md`](movement_states.md)

Foundation movement states covering idle, walking, dashing, and running.

| ID | Name | Description | Key Properties |
|----|------|-------------|----------------|
| 14 | WAIT | Idle/standing | Fully actionable |
| 15 | WALK_SLOW | Slow walk | Lowest walk speed |
| 16 | WALK_MIDDLE | Medium walk | Mid-tier walk speed |
| 17 | WALK_FAST | Fast walk | Highest walk speed |
| 18 | TURN | Turn around | Reverses facing direction |
| 19 | TURN_RUN | Turn around from run | Creates foxtrot window |
| 20 | DASH | Initial dash | Fox: 14 frames, momentum-building |
| 21 | RUN | Full run | Max ground speed |
| 23 | RUN_BRAKE | Run stop/brake | Deceleration frames |
| 24 | KNEE_BEND | Jumpsquat | Fox: 3 frames; enables JC grab, wavedash |

---

## Group 3: Jumps & Falls

**Documentation**: [`jump_fall_states.md`](jump_fall_states.md)

All aerial movement states including jumps and various fall states.

| ID | Name | Description | Key Properties |
|----|------|-------------|----------------|
| 25 | JUMP_F | First jump (facing forward) | Aerial available frame 4 |
| 26 | JUMP_B | First jump (facing backward) | Same as JUMP_F, opposite facing |
| 27 | JUMP_AERIAL_F | Double jump (facing forward) | Uses DJ resource |
| 28 | JUMP_AERIAL_B | Double jump (facing backward) | Uses DJ resource |
| 29 | FALL | Standard fall | Fully actionable, can fastfall |
| 30 | FALL_F | Fall facing forward | Variant of FALL |
| 31 | FALL_B | Fall facing backward | Variant of FALL |
| 32 | FALL_AERIAL | Fall after double jump | Post-DJ fall |
| 33 | FALL_AERIAL_F | Fall after DJ (forward) | Variant |
| 34 | FALL_AERIAL_B | Fall after DJ (backward) | Variant |
| 35 | FALL_SPECIAL | Helpless/freefall | Cannot act until landing |
| 36 | FALL_SPECIAL_F | Helpless (forward) | Post-special fall |
| 37 | FALL_SPECIAL_B | Helpless (backward) | Post-special fall |
| 38 | DAMAGE_FALL | Tumble | Must tech on landing; can DJ out |

---

## Group 4: Crouch & Landing

**Documentation**: [`crouch_landing_states.md`](crouch_landing_states.md)

Crouching states and landing animations.

| ID | Name | Description | Key Properties |
|----|------|-------------|----------------|
| 39 | SQUAT | Crouch entry | 3-4 frame transition |
| 40 | SQUAT_WAIT | Crouch hold | Crouch cancel available |
| 41 | SQUAT_RV | Stand from crouch | Exit crouch animation |
| 42 | LANDING | Normal landing | 4 frames; from fall |
| 43 | LANDING_FALL_SPECIAL | Special landing (wavedash) | 10 frames; from airdodge |

---

## Group 5: Grounded Attacks: Jabs & Tilts

**Documentation**: [`grounded_attacks_states.md`](grounded_attacks_states.md)

All jab and tilt attack states.

| ID | Name | Description | Key Properties |
|----|------|-------------|----------------|
| 44 | ATTACK_11 | Jab 1 | Frame 2 hitbox |
| 45 | ATTACK_12 | Jab 2 | Follow-up from jab 1 |
| 46 | ATTACK_13 | Jab 3 | Final jab (some characters) |
| 47 | ATTACK_100_START | Rapid jab start | Initiate rapid jab |
| 48 | ATTACK_100_LOOP | Rapid jab loop | Repeating hits |
| 49 | ATTACK_100_END | Rapid jab end | End rapid jab |
| 50 | ATTACK_DASH | Dash attack | Momentum attack from dash |
| 51 | ATTACK_S_3_HI | F-tilt (high angle) | Angled up |
| 52 | ATTACK_S_3_HI_S | F-tilt (high-mid) | Slight up angle |
| 53 | ATTACK_S_3_S | F-tilt (neutral) | Standard f-tilt |
| 54 | ATTACK_S_3_LW_S | F-tilt (low-mid) | Slight down angle |
| 55 | ATTACK_S_3_LW | F-tilt (low angle) | Angled down |
| 56 | ATTACK_HI_3 | Up-tilt | Upward attack |
| 57 | ATTACK_LW_3 | Down-tilt | Low attack |

---

## Group 6: Grounded Attacks: Smashes

**Documentation**: [`smash_attacks_states.md`](smash_attacks_states.md)

All smash attack states including charged variants.

| ID | Name | Description | Key Properties |
|----|------|-------------|----------------|
| 58 | ATTACK_S_4_HI | F-smash (high angle) | Angled up |
| 59 | ATTACK_S_4_HI_S | F-smash (high-mid) | Slight up angle |
| 60 | ATTACK_S_4_S | F-smash (neutral) | Standard f-smash |
| 61 | ATTACK_S_4_LW_S | F-smash (low-mid) | Slight down angle |
| 62 | ATTACK_S_4_LW | F-smash (low angle) | Angled down |
| 63 | ATTACK_HI_4 | Up-smash | Primary kill move for Fox |
| 64 | ATTACK_LW_4 | Down-smash | Two-sided attack |

---

## Group 7: Aerial Attacks & Landing Lag

**Documentation**: [`aerial_attacks_states.md`](aerial_attacks_states.md)

All aerial attacks and their corresponding landing lag states.

| ID | Name | Description | Key Properties |
|----|------|-------------|----------------|
| 65 | ATTACK_AIR_N | Nair | Neutral aerial |
| 66 | ATTACK_AIR_F | Fair | Forward aerial |
| 67 | ATTACK_AIR_B | Bair | Back aerial; Fox's primary kill aerial |
| 68 | ATTACK_AIR_HI | Uair | Up aerial; combo finisher |
| 69 | ATTACK_AIR_LW | Dair | Down aerial (drill for Fox) |
| 70 | LANDING_AIR_N | Nair landing lag | L-cancel: 7 frames |
| 71 | LANDING_AIR_F | Fair landing lag | L-cancel: 7 frames |
| 72 | LANDING_AIR_B | Bair landing lag | L-cancel: 9 frames |
| 73 | LANDING_AIR_HI | Uair landing lag | L-cancel: 7 frames |
| 74 | LANDING_AIR_LW | Dair landing lag | L-cancel: 9 frames; drill-shine combo |

---

## Group 8: Damage: Grounded Hitstun

**Documentation**: [`grounded_hitstun_states.md`](grounded_hitstun_states.md)

Hitstun states when hit while grounded.

| ID | Name | Description | Key Properties |
|----|------|-------------|----------------|
| 75 | DAMAGE_HI_1 | High-angle hit (light) | Low knockback |
| 76 | DAMAGE_HI_2 | High-angle hit (medium) | Medium knockback |
| 77 | DAMAGE_HI_3 | High-angle hit (heavy) | High knockback |
| 78 | DAMAGE_N_1 | Neutral hit (light) | Low horizontal KB |
| 79 | DAMAGE_N_2 | Neutral hit (medium) | Medium horizontal KB |
| 80 | DAMAGE_N_3 | Neutral hit (heavy) | High horizontal KB |
| 81 | DAMAGE_LW_1 | Low-angle hit (light) | Low downward KB |
| 82 | DAMAGE_LW_2 | Low-angle hit (medium) | Medium downward KB |
| 83 | DAMAGE_LW_3 | Low-angle hit (heavy) | High downward KB |

---

## Group 9: Damage: Aerial Hitstun & Tumble

**Documentation**: [`aerial_hitstun_states.md`](aerial_hitstun_states.md)

Hitstun and tumble states when hit while airborne.

| ID | Name | Description | Key Properties |
|----|------|-------------|----------------|
| 84 | DAMAGE_AIR_1 | Aerial hit (light) | No tumble |
| 85 | DAMAGE_AIR_2 | Aerial hit (medium) | No tumble; combo state |
| 86 | DAMAGE_AIR_3 | Aerial hit (heavy) | Near tumble threshold |
| 87 | DAMAGE_FLY_HI | Tumble (upward) | Upward knockback |
| 88 | DAMAGE_FLY_N | Tumble (neutral/horizontal) | Most common tumble; 45.6% missed tech |
| 89 | DAMAGE_FLY_LW | Tumble (downward) | Downward knockback |
| 90 | DAMAGE_FLY_TOP | Tumble (strong launch) | **5.25% of gameplay**; primary combo state |
| 91 | DAMAGE_FLY_ROLL | Tumble (100%+ damage) | 16.3% death rate; kill state |

---

## Group 10: Shield & Powershield

**Documentation**: [`shield_states.md`](shield_states.md)

Shield mechanics including powershield and shield break.

| ID | Name | Description | Key Properties |
|----|------|-------------|----------------|
| 178 | GUARD_ON | Shield startup | Frame 1 active |
| 179 | GUARD | Shield hold | 0.28 HP/frame depletion |
| 180 | GUARD_OFF | Shield release | 15 frames lag |
| 181 | GUARD_SET_OFF | Shield stun | From blocked attack |
| 182 | GUARD_REFLECT | Powershield | 4-frame window; reflects projectiles |
| 205 | SHIELD_BREAK_FLY | Shield break (launch) | 29 frames; invulnerable |
| 206 | SHIELD_BREAK_FALL | Shield break (fall) | Transitional |
| 207 | SHIELD_BREAK_DOWN_U | Shield break (land face-up) | 26 frames; invulnerable |
| 208 | SHIELD_BREAK_DOWN_D | Shield break (land face-down) | Variant |
| 209 | SHIELD_BREAK_STAND_U | Shield break (stand face-up) | 30 frames; invulnerable |
| 210 | SHIELD_BREAK_STAND_D | Shield break (stand face-down) | Variant |
| 211 | FURA_FURA | Dizzy state | **Vulnerable**; 30-71 frames; free punish |

---

## Group 11: Knockdown & Getup

**Documentation**: [`knockdown_states.md`](knockdown_states.md)

Missed tech states and getup options.

| ID | Name | Description | Key Properties |
|----|------|-------------|----------------|
| 183 | DOWN_BOUND_U | Knockdown bounce (face-up) | 26 frame minimum |
| 184 | DOWN_WAIT_U | Lying down (face-up) | Actionable; jab reset target |
| 185 | DOWN_DAMAGE_U | Hit while down (face-up) | From 7%+ attack |
| 186 | DOWN_STAND_U | Neutral getup (face-up) | ~30 frames |
| 187 | DOWN_ATTACK_U | Getup attack (face-up) | ~29 frames invuln |
| 188 | DOWN_FOWARD_U | Getup roll forward (face-up) | ~23 frames invuln |
| 189 | DOWN_BACK_U | Getup roll back (face-up) | ~19 frames invuln |
| 190 | DOWN_SPOT_U | (Unused) | - |
| 191 | DOWN_BOUND_D | Knockdown bounce (face-down) | Face-down variant |
| 192 | DOWN_WAIT_D | Lying down (face-down) | Face-down variant |
| 193 | DOWN_DAMAGE_D | Hit while down (face-down) | Face-down variant |
| 194 | DOWN_STAND_D | Neutral getup (face-down) | Face-down variant |
| 195 | DOWN_ATTACK_D | Getup attack (face-down) | Face-down variant |
| 196 | DOWN_FOWARD_D | Getup roll forward (face-down) | Face-down variant |
| 197 | DOWN_BACK_D | Getup roll back (face-down) | Face-down variant |
| 335 | DOWN_REFLECT | Reflector-related knockdown | Special case |

---

## Group 12: Tech (Successful)

**Documentation**: [`tech_states.md`](tech_states.md)

Successful tech states for ground, wall, and ceiling.

| ID | Name | Description | Key Properties |
|----|------|-------------|----------------|
| 199 | PASSIVE | Tech in place | 26 frames; 20 invuln (1-20) |
| 200 | PASSIVE_STAND_F | Tech roll forward | 40 frames; 20 invuln (1-20) |
| 201 | PASSIVE_STAND_B | Tech roll backward | 40 frames; 20 invuln (1-20) |
| 202 | PASSIVE_WALL | Wall tech | 14 frames invuln |
| 203 | PASSIVE_WALL_JUMP | Wall tech jump | Jump off wall after tech |
| 204 | PASSIVE_CEIL | Ceiling tech | Very rare (55 instances) |

---

## Group 13: Grab, Pummel & Throws (Attacker)

**Documentation**: [`grab_throw_attacker_states.md`](grab_throw_attacker_states.md)

Attacker-side grab and throw states.

| ID | Name | Description | Key Properties |
|----|------|-------------|----------------|
| 212 | CATCH | Standing grab | Frames 7-8 active; **74.6% from JC grab** |
| 213 | CATCH_PULL | Grab connect (standing) | Transitional |
| 214 | CATCH_DASH | Dash grab | Frames 12-13 active; 40 total frames |
| 215 | CATCH_DASH_PULL | Grab connect (dash) | Transitional |
| 216 | CATCH_WAIT | Holding opponent | Variable duration; throw from here |
| 217 | CATCH_ATTACK | Pummel | Only 14.3% of grabs include pummel |
| 218 | CATCH_CUT | Grab break | 0.8% of grabs; opponent escaped |
| 219 | THROW_F | Forward throw | 2.9% of throws; positioning |
| 220 | THROW_B | Back throw | 6.8% of throws; edgeguard setup |
| 221 | THROW_HI | Up throw | **85.6% of throws**; combo starter |
| 222 | THROW_LW | Down throw | 1.5% of throws; tech chase |

---

## Group 14: Grabbed & Thrown (Victim)

**Documentation**: [`grabbed_thrown_states.md`](grabbed_thrown_states.md)

Victim-side grabbed and thrown states.

| ID | Name | Description | Key Properties |
|----|------|-------------|----------------|
| 223 | CAPTURE_PULLED_HI | Aerial grab pull | 81.3% from up throw regrab |
| 224 | CAPTURE_WAIT_HI | Aerial grab hold | Being held (aerial) |
| 225 | CAPTURE_DAMAGE_HI | Aerial pummel | Being pummeled (aerial) |
| 226 | CAPTURE_PULLED_LW | Ground grab pull | Standard grab |
| 227 | CAPTURE_WAIT_LW | Ground grab hold | Being held (grounded) |
| 228 | CAPTURE_DAMAGE_LW | Ground pummel | Being pummeled (grounded) |
| 229 | CAPTURE_CUT | Grab escape | Mashed out successfully |
| 230 | CAPTURE_JUMP | Jump escape | Jumped out of grab |
| 239 | THROWN_F | Being thrown forward | Diagonal knockback |
| 240 | THROWN_B | Being thrown backward | Horizontal knockback |
| 241 | THROWN_HI | Being thrown upward | **86.8% of throws**; vertical KB |
| 242 | THROWN_LW | Being thrown downward | Creates tech situation |

---

## Group 15: Defensive Options: Rolls & Dodges

**Documentation**: [`defensive_options_states.md`](defensive_options_states.md)

Rolls, spotdodge, airdodge, and clank states.

| ID | Name | Description | Key Properties |
|----|------|-------------|----------------|
| 233 | ESCAPE_F | Roll forward | 31 frames; invuln 4-19 |
| 234 | ESCAPE_B | Roll backward | 31 frames; invuln 4-19 |
| 235 | ESCAPE | Spotdodge | 22 frames; invuln 2-15 |
| 236 | ESCAPE_AIR | Airdodge | **97.9% are wavedashes**; 49 frames |
| 237 | REBOUND_STOP | Attack clank (freeze) | ~6 frames |
| 238 | REBOUND | Attack clank (recoil) | ~11 frames |

---

## Group 16: Stage Interactions

**Documentation**: [`stage_interaction_states.md`](stage_interaction_states.md)

Platform drops, teetering, wall/ceiling bounces, and missed ledge.

| ID | Name | Description | Key Properties |
|----|------|-------------|----------------|
| 244 | PASS | Platform drop-through | **76.3% shield drops** |
| 245 | OTTOTTO | Teeter start | Edge wobble initiation |
| 246 | OTTOTTO_WAIT | Teeter hold | Sustained wobble |
| 247 | FLY_REFLECT_WALL | Wall bounce (missed tech) | 14 frames invuln; 73% fatal |
| 248 | FLY_REFLECT_CEIL | Ceiling bounce (missed tech) | 72% fatal |
| 249 | STOP_WALL | Wall stop (grounded) | From dashing into wall |
| 250 | STOP_CEIL | Ceiling stop | From jumping into ceiling |
| 251 | MISS_FOOT | Missed ledge sweetspot | 52.3% still grab ledge |

---

## Group 17: Ledge States

**Documentation**: [`ledge_taunt_states.md`](ledge_taunt_states.md)

All ledge grab, hang, and getup option states.

| ID | Name | Description | Key Properties |
|----|------|-------------|----------------|
| 252 | CLIFF_CATCH | Ledge grab | 7 frames; 100% invuln |
| 253 | CLIFF_WAIT | Ledge hang | **82.1% exit via drop** (ledgedash) |
| 254 | CLIFF_CLIMB_SLOW | Getup (≥100%) | 59 frames; 95.2% invuln |
| 255 | CLIFF_CLIMB_QUICK | Getup (<100%) | 34 frames; 88.7% invuln |
| 256 | CLIFF_ATTACK_SLOW | Ledge attack (≥100%) | 69 frames |
| 257 | CLIFF_ATTACK_QUICK | Ledge attack (<100%) | 54 frames |
| 258 | CLIFF_ESCAPE_SLOW | Ledge roll (≥100%) | 79 frames; 81.4% invuln |
| 259 | CLIFF_ESCAPE_QUICK | Ledge roll (<100%) | 49 frames; 70.3% invuln |
| 260 | CLIFF_JUMP_SLOW_1 | Ledge jump P1 (≥100%) | 19 frames; 100% invuln |
| 261 | CLIFF_JUMP_SLOW_2 | Ledge jump P2 (≥100%) | 30 frames |
| 262 | CLIFF_JUMP_QUICK_1 | Ledge jump P1 (<100%) | 14 frames; 100% invuln |
| 263 | CLIFF_JUMP_QUICK_2 | Ledge jump P2 (<100%) | 30 frames |

---

## Group 18: Taunt

**Documentation**: [`ledge_taunt_states.md`](ledge_taunt_states.md)

| ID | Name | Description | Key Properties |
|----|------|-------------|----------------|
| 264 | APPEAL_R | Taunt (right) | ~110 frames; no utility |
| 265 | APPEAL_L | Taunt (left) | Left-facing variant |

---

## Group 19: Fox: Blaster & Illusion

**Documentation**: [`fox_blaster_illusion_states.md`](fox_blaster_illusion_states.md)

Fox's Neutral-B and Side-B special moves.

| ID | Name | Description | Key Properties |
|----|------|-------------|----------------|
| 341 | BLASTER_GROUND_STARTUP | Blaster startup (ground) | Frame 7 first shot |
| 342 | BLASTER_GROUND_LOOP | Blaster loop (ground) | Repeat shots |
| 343 | BLASTER_GROUND_END | Blaster end (ground) | 24 frames median |
| 344 | BLASTER_AIR_STARTUP | Blaster startup (air) | **63% from short hop** (SHL) |
| 345 | BLASTER_AIR_LOOP | Blaster loop (air) | 73.8% single shot |
| 346 | BLASTER_AIR_END | Blaster end (air) | 5 frames median |
| 347 | ILLUSION_GROUND_STARTUP | Illusion startup (ground) | 19 frames |
| 348 | ILLUSION_GROUND | Illusion active (ground) | Hitbox frames 22-25 |
| 349 | ILLUSION_GROUND_END | Illusion end (ground) | 40 frames; punishable |
| 350 | ILLUSION_STARTUP_AIR | Illusion startup (air) | Primary recovery |
| 351 | ILLUSION_AIR | Illusion active (air) | 7%; can shorten |
| 352 | ILLUSION_AIR_END | Illusion end (air) | **82.8% success** (land/ledge) |

---

## Group 20: Fox: Fire Fox & Shine

**Documentation**: [`fox_firefox_shine_states.md`](fox_firefox_shine_states.md)

Fox's Up-B and Down-B special moves.

| ID | Name | Description | Key Properties |
|----|------|-------------|----------------|
| 353 | FIRE_FOX_GROUND_STARTUP | Fire Fox startup (ground) | 24 frames |
| 354 | FIRE_FOX_AIR_STARTUP | Fire Fox startup (air) | 42 frames; **24.7% early ledge grab** |
| 355 | FIRE_FOX_GROUND | Fire Fox travel (ground) | Hitbox frames 43-72 |
| 356 | FIRE_FOX_AIR | Fire Fox travel (air) | 14% damage |
| 357 | FIRE_FOX_GROUND_END | Fire Fox end (ground) | 6 frames |
| 358 | FIRE_FOX_AIR_END | Fire Fox end (air) | 14 frames; helpless after |
| 359 | FIRE_FOX_BOUNCE_END | Fire Fox bounce | Bounced off surface |
| 360 | REFLECTOR_GROUND_STARTUP | Shine startup (ground) | **Frame 1 intangible**; JC frame 4 |
| 361 | REFLECTOR_GROUND_LOOP | Shine hold (ground) | **63.9% waveshine pattern** |
| 362 | REFLECTOR_GROUND_REFLECT | Shine reflect (ground) | Projectile reflected |
| 363 | REFLECTOR_GROUND_END | Shine end (ground) | 19 frames if not JC'd |
| 364 | REFLECTOR_GROUND_CHANGE_DIRECTION | Shine turnaround (ground) | +3 frames JC delay |
| 365 | REFLECTOR_AIR_STARTUP | Shine startup (air) | Frame 1 intangible |
| 366 | REFLECTOR_AIR_LOOP | Shine hold (air) | Can land or DJ cancel |
| 367 | REFLECTOR_AIR_REFLECT | Shine reflect (air) | Projectile reflected |
| 368 | REFLECTOR_AIR_END | Shine end (air) | 18 frames |
| 369 | REFLECTOR_AIR_CHANGE_DIRECTION | Shine turnaround (air) | 44.4% → DJ (recovery) |

---

## State Categories by Function

### Actionable States
States where the player has full control:
- **14** (WAIT), **29** (FALL), **32** (FALL_AERIAL)
- **39-40** (SQUAT, SQUAT_WAIT)
- **253** (CLIFF_WAIT - first 30 frames)

### Invulnerable States
States with full or partial invulnerability:
- **12-13** (REBIRTH) - Respawn invuln
- **178-182** (GUARD states) - Shield
- **199-204** (PASSIVE/TECH) - Tech invuln (20 frames)
- **252-263** (CLIFF states) - Ledge invuln (37 total)
- **360, 365** (SHINE) - Frame 1 intangible

### High-Usage States (Fox Dittos)
Most frequent states by gameplay percentage:
1. **20** (DASH) - **8.41%**
2. **25** (JUMP_F) - 5.41%
3. **90** (DAMAGE_FLY_TOP) - 5.25% (most common damage state)
4. **27** (JUMP_AERIAL_F) - 4.02%
5. **67** (ATTACK_AIR_B) - 3.27%
6. **43** (LANDING_FALL_SPECIAL) - 3.25%
7. **88** (DAMAGE_FLY_N) - 3.05%
8. **14** (WAIT) - 3.04%

### Kill-Related States
States associated with deaths:
- **0-7** (DEAD_*) - Death animations
- **91** (DAMAGE_FLY_ROLL) - 16.3% death rate
- **247** (FLY_REFLECT_WALL) - 73% fatal
- **248** (FLY_REFLECT_CEIL) - 72% fatal

---

## Key Melee Mechanics by State

### Wavedash
**24** (KNEE_BEND) → **25/26** (JUMP) → **236** (ESCAPE_AIR) → **43** (LANDING_FALL_SPECIAL)

### JC Grab
**24** (KNEE_BEND) → **212** (CATCH)

### L-Cancel
**65-69** (ATTACK_AIR_*) → **70-74** (LANDING_AIR_*) with L/R input

### Waveshine
**360** → **361** → **24** → **236** → **43** → **360** (repeat)

### Ledgedash
**253** (drop) → **29** → **27** → **236** → **43** (with remaining invuln)

### Tech Chase
**87-91** (tumble) → **183/191** (missed tech) or **199-201** (successful tech)

### Up Throw Combo
**212** → **216** → **221** (THROW_HI) → opponent in **90** → **68** (UAIR)

---

## Data Sources

All statistics derived from:
- **Parquet replay data**: 288,753,124 frames of Fox vs Fox matches
- **action_state.json**: Official state ID mappings
- **SmashWiki**: Frame data and mechanic documentation
- **Smashboards RAG**: Community knowledge and advanced techniques

---

## File Index

| File | States Covered | Group(s) |
|------|---------------|----------|
| `death_respawn_states.md` | 0, 1, 2, 4, 6, 7, 12, 13 | 1 |
| `movement_states.md` | 14-21, 23, 24 | 2 |
| `jump_fall_states.md` | 25-38 | 3 |
| `crouch_landing_states.md` | 39-43 | 4 |
| `grounded_attacks_states.md` | 44-57 | 5 |
| `smash_attacks_states.md` | 58-64 | 6 |
| `aerial_attacks_states.md` | 65-74 | 7 |
| `grounded_hitstun_states.md` | 75-83 | 8 |
| `aerial_hitstun_states.md` | 84-91 | 9 |
| `shield_states.md` | 178-182, 205-211 | 10 |
| `knockdown_states.md` | 183-197, 335 | 11 |
| `tech_states.md` | 199-204 | 12 |
| `grab_throw_attacker_states.md` | 212-222 | 13 |
| `grabbed_thrown_states.md` | 223-230, 239-242 | 14 |
| `defensive_options_states.md` | 233-238 | 15 |
| `stage_interaction_states.md` | 244-251 | 16 |
| `ledge_taunt_states.md` | 252-265 | 17, 18 |
| `fox_blaster_illusion_states.md` | 341-352 | 19 |
| `fox_firefox_shine_states.md` | 353-369 | 20 |
