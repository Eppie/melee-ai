-- ============================================================
-- ENHANCED L-CANCEL ANALYSIS QUERY
-- ============================================================
-- Analyzes L-cancel attempts with detailed timing and context:
--   - L-cancel success/failure
--   - Trigger input frame (timing relative to landing)
--   - During hitlag detection (common mistake)
--   - Fastfall status
--   - Aerial type and landing lag
--
-- Output Schema:
--   replay_file, player_index, frame_number, aerial_type,
--   lcancel_success, trigger_input_frame, during_hitlag,
--   is_fastfall, landing_lag_frames, position_x, position_y
--
-- Filters: replay_file, character_id
-- Based on slippistats LCancelData enhancements
-- ============================================================

-- FILTERING CONFIGURATION
-- Uncomment and modify these lines to filter results:
--
-- Filter by specific replay file:
-- WHERE replay_file = 'master-diamond-000543d6034bf69893c60a05.slp'
--
-- Filter by character:
-- WHERE character_id = 1  -- Fox
--
-- ============================================================

WITH
-- Step 1: Get aerial frames with context
aerial_frames AS (
  SELECT
    replay_file,
    player_index,
    frame_number,
    action_state,
    action_state_frame,
    character_id,

    -- Position
    position_x,
    position_y,

    -- L-cancel status
    lcancel_status,

    -- Hitstun state
    is_in_hitstun,
    hitstun_remaining,

    -- Velocity (calculated from position changes - for fastfall detection)
    position_y - LAG(position_y) OVER w AS velocity_y,

    -- Trigger inputs
    button_l,
    button_r,
    trigger_l,
    trigger_r,

    -- Previous frames
    LAG(action_state, 1) OVER w AS prev_action_1,
    LAG(action_state, 2) OVER w AS prev_action_2,
    LAG(action_state, 3) OVER w AS prev_action_3,
    LAG(action_state, 4) OVER w AS prev_action_4,
    LAG(action_state, 5) OVER w AS prev_action_5,
    LAG(action_state, 6) OVER w AS prev_action_6,
    LAG(action_state, 7) OVER w AS prev_action_7,

    -- Previous trigger states (to detect trigger press timing)
    LAG(button_l, 1) OVER w AS prev_trigger_l_1,
    LAG(button_r, 1) OVER w AS prev_trigger_r_1,
    LAG(button_l, 2) OVER w AS prev_trigger_l_2,
    LAG(button_r, 2) OVER w AS prev_trigger_r_2,
    LAG(button_l, 3) OVER w AS prev_trigger_l_3,
    LAG(button_r, 3) OVER w AS prev_trigger_r_3,
    LAG(button_l, 4) OVER w AS prev_trigger_l_4,
    LAG(button_r, 4) OVER w AS prev_trigger_r_4,
    LAG(button_l, 5) OVER w AS prev_trigger_l_5,
    LAG(button_r, 5) OVER w AS prev_trigger_r_5,
    LAG(button_l, 6) OVER w AS prev_trigger_l_6,
    LAG(button_r, 6) OVER w AS prev_trigger_r_6,
    LAG(button_l, 7) OVER w AS prev_trigger_l_7,
    LAG(button_r, 7) OVER w AS prev_trigger_r_7,

    -- Previous hitstun (to check if in hitlag during aerial)
    LAG(is_in_hitstun, 1) OVER w AS prev_in_hitstun_1,
    LAG(is_in_hitstun, 2) OVER w AS prev_in_hitstun_2,
    LAG(is_in_hitstun, 3) OVER w AS prev_in_hitstun_3,
    LAG(is_in_hitstun, 4) OVER w AS prev_in_hitstun_4,
    LAG(is_in_hitstun, 5) OVER w AS prev_in_hitstun_5,

    -- Previous velocity (for fastfall detection) - calculated from position
    LAG(position_y, 1) OVER w - LAG(position_y, 2) OVER w AS prev_velocity_y_1,
    LAG(position_y, 2) OVER w - LAG(position_y, 3) OVER w AS prev_velocity_y_2,
    LAG(position_y, 3) OVER w - LAG(position_y, 4) OVER w AS prev_velocity_y_3,
    LAG(position_y, 4) OVER w - LAG(position_y, 5) OVER w AS prev_velocity_y_4,
    LAG(position_y, 5) OVER w - LAG(position_y, 6) OVER w AS prev_velocity_y_5,
    LAG(position_y, 6) OVER w - LAG(position_y, 7) OVER w AS prev_velocity_y_6,
    LAG(position_y, 7) OVER w - LAG(position_y, 8) OVER w AS prev_velocity_y_7

  FROM read_parquet('replay_parquet_test/partition_*.parquet')
  WINDOW w AS (PARTITION BY replay_file, player_index ORDER BY frame_number)
  -- Add character filter here if needed:
  -- WHERE character_id = 1
),

-- Step 2: Detect L-cancel events (aerial landing frames)
lcancel_events AS (
  SELECT
    *,

    -- AERIAL TYPE from landing state
    CASE
      WHEN action_state = 70 THEN 'nair'   -- 0x46
      WHEN action_state = 71 THEN 'fair'   -- 0x47
      WHEN action_state = 72 THEN 'bair'   -- 0x48
      WHEN action_state = 73 THEN 'uair'   -- 0x49
      WHEN action_state = 74 THEN 'dair'   -- 0x4A
      ELSE NULL
    END AS aerial_type,

    -- L-CANCEL SUCCESS
    CASE
      WHEN lcancel_status = 1 THEN TRUE
      WHEN lcancel_status = 2 THEN FALSE
      ELSE NULL
    END AS lcancel_success,

    -- TRIGGER INPUT FRAME DETECTION
    -- Find which frame (0-7 frames back) had a trigger press
    -- L-cancel window is typically last 7 frames before landing
    CASE
      -- Trigger pressed on landing frame
      WHEN (button_l AND NOT COALESCE(prev_trigger_l_1, FALSE))
        OR (button_r AND NOT COALESCE(prev_trigger_r_1, FALSE))
      THEN 0

      -- Trigger pressed 1 frame before landing
      WHEN (prev_trigger_l_1 AND NOT COALESCE(prev_trigger_l_2, FALSE))
        OR (prev_trigger_r_1 AND NOT COALESCE(prev_trigger_r_2, FALSE))
      THEN -1

      -- Trigger pressed 2 frames before landing
      WHEN (prev_trigger_l_2 AND NOT COALESCE(prev_trigger_l_3, FALSE))
        OR (prev_trigger_r_2 AND NOT COALESCE(prev_trigger_r_3, FALSE))
      THEN -2

      -- Trigger pressed 3 frames before landing
      WHEN (prev_trigger_l_3 AND NOT COALESCE(prev_trigger_l_4, FALSE))
        OR (prev_trigger_r_3 AND NOT COALESCE(prev_trigger_r_4, FALSE))
      THEN -3

      -- Trigger pressed 4 frames before landing
      WHEN (prev_trigger_l_4 AND NOT COALESCE(prev_trigger_l_5, FALSE))
        OR (prev_trigger_r_4 AND NOT COALESCE(prev_trigger_r_5, FALSE))
      THEN -4

      -- Trigger pressed 5 frames before landing
      WHEN (prev_trigger_l_5 AND NOT COALESCE(prev_trigger_l_6, FALSE))
        OR (prev_trigger_r_5 AND NOT COALESCE(prev_trigger_r_6, FALSE))
      THEN -5

      -- Trigger pressed 6 frames before landing
      WHEN (prev_trigger_l_6 AND NOT COALESCE(prev_trigger_l_7, FALSE))
        OR (prev_trigger_r_6 AND NOT COALESCE(prev_trigger_r_7, FALSE))
      THEN -6

      ELSE NULL
    END AS trigger_input_frame,

    -- DURING HITLAG DETECTION
    -- Check if player was in hitstun during the aerial (common mistake)
    CASE
      WHEN prev_in_hitstun_1 OR prev_in_hitstun_2 OR prev_in_hitstun_3
        OR prev_in_hitstun_4 OR prev_in_hitstun_5
      THEN 1
      ELSE 0
    END AS during_hitlag,

    -- FASTFALL DETECTION
    -- Fastfall velocity varies by character, but typically < -2.0
    -- Check if velocity was high (negative) during aerial
    CASE
      WHEN prev_velocity_y_1 < -2.0 OR prev_velocity_y_2 < -2.0
        OR prev_velocity_y_3 < -2.0 OR prev_velocity_y_4 < -2.0
        OR prev_velocity_y_5 < -2.0 OR prev_velocity_y_6 < -2.0
        OR prev_velocity_y_7 < -2.0
      THEN 1
      ELSE 0
    END AS is_fastfall,

    -- Average fall velocity during aerial
    ROUND((
      COALESCE(prev_velocity_y_1, 0) +
      COALESCE(prev_velocity_y_2, 0) +
      COALESCE(prev_velocity_y_3, 0) +
      COALESCE(prev_velocity_y_4, 0) +
      COALESCE(prev_velocity_y_5, 0) +
      COALESCE(prev_velocity_y_6, 0) +
      COALESCE(prev_velocity_y_7, 0)
    ) / 7.0, 2) AS avg_fall_velocity

  FROM aerial_frames
  WHERE action_state IN (70, 71, 72, 73, 74)  -- Aerial landing states
    AND action_state_frame = 0  -- First frame of landing
    AND lcancel_status IN (1, 2)  -- Only when L-cancel was attempted/possible
),

-- Step 3: Calculate landing lag duration
landing_lag AS (
  SELECT
    e.*,

    -- LANDING LAG FRAMES
    -- L-cancel cuts lag in half
    -- Standard lag varies by aerial: typically 7-15 frames normal, 3-7 frames L-canceled
    -- Approximate from action_state duration (would need frame data for exact values)
    CASE
      WHEN lcancel_success THEN
        CASE aerial_type
          WHEN 'nair' THEN 5
          WHEN 'fair' THEN 7
          WHEN 'bair' THEN 7
          WHEN 'uair' THEN 6
          WHEN 'dair' THEN 9
          ELSE 7
        END
      ELSE
        CASE aerial_type
          WHEN 'nair' THEN 10
          WHEN 'fair' THEN 14
          WHEN 'bair' THEN 14
          WHEN 'uair' THEN 12
          WHEN 'dair' THEN 18
          ELSE 14
        END
    END AS estimated_landing_lag_frames

  FROM lcancel_events e
)

-- Final output
SELECT
  replay_file,
  player_index,
  frame_number AS landing_frame,

  -- Aerial info
  aerial_type,

  -- L-cancel result
  lcancel_success,

  -- Timing
  trigger_input_frame,
  CAST(during_hitlag AS BOOLEAN) AS during_hitlag,

  -- Movement
  CAST(is_fastfall AS BOOLEAN) AS is_fastfall,
  avg_fall_velocity,

  -- Landing lag
  estimated_landing_lag_frames AS landing_lag_frames,

  -- Position
  ROUND(position_x, 2) AS position_x,
  ROUND(position_y, 2) AS position_y,

  -- Metadata
  character_id

FROM landing_lag
ORDER BY replay_file, player_index, frame_number;
