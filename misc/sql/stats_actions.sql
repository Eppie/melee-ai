-- ============================================================
-- TECHNICAL ACTIONS DETECTION QUERY
-- ============================================================
-- Detects all technical skill executions: wavedash, L-cancel, dash dance, grabs, etc.
--
-- Output Schema:
--   replay_file, player_index, wavedash_count, waveland_count, dash_dance_turns,
--   lcancel_success_count, lcancel_fail_count, lcancel_success_rate,
--   grab_attempts, grab_successes, grab_success_rate,
--   throw_forward/back/up/down_count, tech_neutral/forward/backward/wall/miss_count,
--   tech_success_rate, roll_forward/backward_count, spotdodge_count, airdodge_count,
--   ledgegrab_count, jab/dash_attack/tilt/smash/aerial_count, total_attacks
--
-- Filters: replay_file, character_id
-- ============================================================

-- FILTERING CONFIGURATION
-- Uncomment and modify these lines to filter results:
--
-- Filter by specific replay file:
-- WHERE replay_file = 'master-diamond-000543d6034bf69893c60a05.slp'
--
-- Filter by character:
-- WHERE character_id = 1  -- Fox
-- WHERE character_id IN (1, 2, 20)  -- Fox, Captain Falcon, Young Link
--
-- ============================================================

WITH
-- Step 1: Add previous/next action states for pattern detection
action_sequences AS (
  SELECT
    replay_file,
    player_index,
    frame_number,
    action_state,
    action_state_frame,
    lcancel_status,
    on_ground,
    character_id,

    -- Window functions for sequence detection
    LAG(action_state, 1) OVER w AS prev_action_1,
    LAG(action_state, 2) OVER w AS prev_action_2,
    LAG(action_state, 3) OVER w AS prev_action_3,
    LAG(frame_number, 1) OVER w AS prev_frame_1,
    LAG(frame_number, 2) OVER w AS prev_frame_2,
    LAG(on_ground, 1) OVER w AS prev_on_ground,
    LEAD(action_state, 1) OVER w AS next_action_1,

    -- Action state transitions
    CASE WHEN action_state != COALESCE(LAG(action_state, 1) OVER w, action_state)
    THEN 1 ELSE 0 END AS action_changed,

    -- Action state frame counter reset (new action or repeated action)
    CASE WHEN action_state_frame < COALESCE(LAG(action_state_frame, 1) OVER w, action_state_frame)
    THEN 1 ELSE 0 END AS action_frame_reset

  FROM read_parquet('replay_parquet_test/partition_*.parquet')
  WINDOW w AS (PARTITION BY replay_file, player_index ORDER BY frame_number)
  -- Add character filter here if needed:
  -- WHERE character_id = 1  -- Uncomment to filter by character
),

-- Step 2: Detect specific tech patterns
tech_detections AS (
  SELECT
    replay_file,
    player_index,
    frame_number,
    action_state,
    character_id,

    -- WAVEDASH: KneeBend (0x18=24) → EscapeAir (0xEC=236) → LandingFallSpecial (0x2B=43) within 8 frames
    CASE WHEN
      action_state = 43  -- LandingFallSpecial
      AND prev_action_1 = 236  -- EscapeAir (air dodge)
      AND prev_action_2 = 24  -- KneeBend (jump squat)
      AND (frame_number - COALESCE(prev_frame_2, frame_number)) <= 8
    THEN 1 ELSE 0 END AS wavedash,

    -- WAVELAND: EscapeAir → LandingFallSpecial (no jump squat)
    CASE WHEN
      action_state = 43  -- LandingFallSpecial
      AND prev_action_1 = 236  -- EscapeAir
      AND COALESCE(prev_action_2, 0) != 24  -- NOT from jump squat
      AND (frame_number - COALESCE(prev_frame_1, frame_number)) <= 4
    THEN 1 ELSE 0 END AS waveland,

    -- DASH DANCE: Detect Turn during Dash/Run states
    CASE WHEN
      action_state IN (18, 19)  -- 0x12, 0x13: Turn, TurnRun
      AND (action_changed = 1 OR action_frame_reset = 1)
      AND prev_action_1 IN (20, 21)  -- 0x14, 0x15: Dash, Run
    THEN 1 ELSE 0 END AS dash_dance_turn,

    -- L-CANCEL SUCCESS: Landing from aerial with lcancel_status = 1
    CASE WHEN
      action_state IN (70, 71, 72, 73, 74)  -- 0x46-0x4A: Aerial landings
      AND action_state_frame = 0  -- First frame of landing
      AND lcancel_status = 1
    THEN 1 ELSE 0 END AS lcancel_success,

    -- L-CANCEL FAIL: Landing from aerial with lcancel_status = 2
    CASE WHEN
      action_state IN (70, 71, 72, 73, 74)  -- 0x46-0x4A
      AND action_state_frame = 0
      AND lcancel_status = 2
    THEN 1 ELSE 0 END AS lcancel_fail,

    -- GRAB: Entering grab states
    CASE WHEN
      action_state IN (212, 214)  -- 0xD4, 0xD6: Catch, CatchDash
      AND (action_changed = 1 OR action_frame_reset = 1)
    THEN 1 ELSE 0 END AS grab_attempt,

    -- GRAB SUCCESS: Reaching CatchWait (opponent grabbed)
    CASE WHEN
      action_state = 216  -- 0xD8: CatchWait
      AND (action_changed = 1 OR action_frame_reset = 1)
    THEN 1 ELSE 0 END AS grab_success,

    -- THROWS
    CASE WHEN action_state = 219 AND (action_changed = 1 OR action_frame_reset = 1) THEN 1 ELSE 0 END AS throw_forward,
    CASE WHEN action_state = 220 AND (action_changed = 1 OR action_frame_reset = 1) THEN 1 ELSE 0 END AS throw_back,
    CASE WHEN action_state = 221 AND (action_changed = 1 OR action_frame_reset = 1) THEN 1 ELSE 0 END AS throw_up,
    CASE WHEN action_state = 222 AND (action_changed = 1 OR action_frame_reset = 1) THEN 1 ELSE 0 END AS throw_down,

    -- TECHS
    CASE WHEN action_state = 199 AND (action_changed = 1 OR action_frame_reset = 1) THEN 1 ELSE 0 END AS tech_neutral,
    CASE WHEN action_state = 200 AND (action_changed = 1 OR action_frame_reset = 1) THEN 1 ELSE 0 END AS tech_forward,
    CASE WHEN action_state = 201 AND (action_changed = 1 OR action_frame_reset = 1) THEN 1 ELSE 0 END AS tech_backward,
    CASE WHEN action_state = 202 AND (action_changed = 1 OR action_frame_reset = 1) THEN 1 ELSE 0 END AS tech_wall,
    CASE WHEN action_state IN (183, 191, 247) AND (action_changed = 1 OR action_frame_reset = 1) THEN 1 ELSE 0 END AS tech_miss,

    -- ROLLS
    CASE WHEN action_state = 233 AND (action_changed = 1 OR action_frame_reset = 1) THEN 1 ELSE 0 END AS roll_forward,
    CASE WHEN action_state = 234 AND (action_changed = 1 OR action_frame_reset = 1) THEN 1 ELSE 0 END AS roll_backward,

    -- SPOT DODGE
    CASE WHEN action_state = 235 AND (action_changed = 1 OR action_frame_reset = 1) THEN 1 ELSE 0 END AS spotdodge,

    -- AIR DODGE (excluding wavedash/waveland)
    -- Count air dodge only if it doesn't lead to LandingFallSpecial
    CASE WHEN
      action_state = 236  -- EscapeAir
      AND (action_changed = 1 OR action_frame_reset = 1)
      AND COALESCE(next_action_1, 0) != 43  -- NOT leading to LandingFallSpecial
    THEN 1 ELSE 0 END AS airdodge,

    -- LEDGE GRABS
    CASE WHEN action_state = 252 AND (action_changed = 1 OR action_frame_reset = 1) THEN 1 ELSE 0 END AS ledgegrab,

    -- ATTACKS (categorized)
    -- Jabs: 0x2C-0x2F (44-47)
    CASE WHEN action_state BETWEEN 44 AND 47 AND (action_changed = 1 OR action_frame_reset = 1) THEN 1 ELSE 0 END AS jab,

    -- Dash attack: 0x32 (50)
    CASE WHEN action_state = 50 AND (action_changed = 1 OR action_frame_reset = 1) THEN 1 ELSE 0 END AS dash_attack,

    -- Tilts: 0x33-0x39 (51-57)
    CASE WHEN action_state BETWEEN 51 AND 57 AND (action_changed = 1 OR action_frame_reset = 1) THEN 1 ELSE 0 END AS tilt,

    -- Smashes: 0x3A-0x40 (58-64)
    CASE WHEN action_state BETWEEN 58 AND 64 AND (action_changed = 1 OR action_frame_reset = 1) THEN 1 ELSE 0 END AS smash,

    -- Aerials: 0x41-0x45 (65-69)
    CASE WHEN action_state BETWEEN 65 AND 69 AND (action_changed = 1 OR action_frame_reset = 1) THEN 1 ELSE 0 END AS aerial

  FROM action_sequences
),

-- Step 3: Aggregate per player per replay
player_tech_stats AS (
  SELECT
    replay_file,
    player_index,

    -- Wavedash/Waveland
    SUM(wavedash) AS wavedash_count,
    SUM(waveland) AS waveland_count,

    -- Dash dancing
    SUM(dash_dance_turn) AS dash_dance_turns,

    -- L-cancels
    SUM(lcancel_success) AS lcancel_success_count,
    SUM(lcancel_fail) AS lcancel_fail_count,
    ROUND(100.0 * SUM(lcancel_success) / NULLIF(SUM(lcancel_success) + SUM(lcancel_fail), 0), 2) AS lcancel_success_rate,

    -- Grabs
    SUM(grab_attempt) AS grab_attempts,
    SUM(grab_success) AS grab_successes,
    ROUND(100.0 * SUM(grab_success) / NULLIF(SUM(grab_attempt), 0), 2) AS grab_success_rate,

    -- Throws
    SUM(throw_forward) AS throw_forward_count,
    SUM(throw_back) AS throw_back_count,
    SUM(throw_up) AS throw_up_count,
    SUM(throw_down) AS throw_down_count,

    -- Techs
    SUM(tech_neutral) AS tech_neutral_count,
    SUM(tech_forward) AS tech_forward_count,
    SUM(tech_backward) AS tech_backward_count,
    SUM(tech_wall) AS tech_wall_count,
    SUM(tech_miss) AS tech_miss_count,
    ROUND(100.0 * (SUM(tech_neutral) + SUM(tech_forward) + SUM(tech_backward) + SUM(tech_wall)) /
      NULLIF(SUM(tech_neutral) + SUM(tech_forward) + SUM(tech_backward) + SUM(tech_wall) + SUM(tech_miss), 0), 2) AS tech_success_rate,

    -- Defensive options
    SUM(roll_forward) AS roll_forward_count,
    SUM(roll_backward) AS roll_backward_count,
    SUM(spotdodge) AS spotdodge_count,
    SUM(airdodge) AS airdodge_count,

    -- Ledge
    SUM(ledgegrab) AS ledgegrab_count,

    -- Attacks
    SUM(jab) AS jab_count,
    SUM(dash_attack) AS dash_attack_count,
    SUM(tilt) AS tilt_count,
    SUM(smash) AS smash_count,
    SUM(aerial) AS aerial_count,
    SUM(jab + dash_attack + tilt + smash + aerial) AS total_attacks

  FROM tech_detections
  GROUP BY replay_file, player_index
)

-- Final output
SELECT * FROM player_tech_stats
ORDER BY replay_file, player_index;
