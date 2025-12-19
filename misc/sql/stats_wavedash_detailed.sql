-- ============================================================
-- ENHANCED WAVEDASH ANALYSIS QUERY
-- ============================================================
-- Detects wavedashes and wavelands with detailed metrics:
--   - Angle (degrees below horizontal)
--   - Direction (LEFT, RIGHT, DOWN)
--   - Trigger timing (frames from jump squat to trigger press)
--   - Airdodge duration (frames from trigger to landing)
--
-- Output Schema:
--   replay_file, player_index, frame_number, wavedash_type,
--   angle_degrees, direction, trigger_frame, airdodge_frames,
--   joystick_x, joystick_y, position_x, position_y
--
-- Filters: replay_file, character_id
-- Based on slippistats WavedashData enhancements
-- ============================================================

-- FILTERING CONFIGURATION
-- Uncomment and modify these lines to filter results:
--
-- Filter by specific replay file:
-- WHERE replay_file = 'master-diamond-000543d6034bf69893c60a05.slp'
--
-- Filter by character:
-- WHERE character_id = 1  -- Fox
-- WHERE character_id IN (1, 22)  -- Spacies only
--
-- ============================================================

WITH
-- Step 1: Get frame data with previous action states
frame_history AS (
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

    -- Joystick inputs (for angle calculation)
    joystick_x,
    joystick_y,

    -- Trigger inputs (for timing)
    button_l,
    button_r,

    -- Previous action states (to detect patterns) - extended to 8 frames
    LAG(action_state, 1) OVER w AS prev_action_1,
    LAG(action_state, 2) OVER w AS prev_action_2,
    LAG(action_state, 3) OVER w AS prev_action_3,
    LAG(action_state, 4) OVER w AS prev_action_4,
    LAG(action_state, 5) OVER w AS prev_action_5,
    LAG(action_state, 6) OVER w AS prev_action_6,
    LAG(action_state, 7) OVER w AS prev_action_7,
    LAG(action_state, 8) OVER w AS prev_action_8,

    -- Previous frame numbers (for timing calculations)
    LAG(frame_number, 1) OVER w AS prev_frame_1,
    LAG(frame_number, 2) OVER w AS prev_frame_2,
    LAG(frame_number, 3) OVER w AS prev_frame_3,
    LAG(frame_number, 4) OVER w AS prev_frame_4,
    LAG(frame_number, 5) OVER w AS prev_frame_5,
    LAG(frame_number, 6) OVER w AS prev_frame_6,
    LAG(frame_number, 7) OVER w AS prev_frame_7,
    LAG(frame_number, 8) OVER w AS prev_frame_8,

    -- Previous joystick positions (for airdodge frame - check multiple frames back)
    LAG(joystick_x, 1) OVER w AS prev_joystick_x_1,
    LAG(joystick_y, 1) OVER w AS prev_joystick_y_1,
    LAG(joystick_x, 2) OVER w AS prev_joystick_x_2,
    LAG(joystick_y, 2) OVER w AS prev_joystick_y_2,
    LAG(joystick_x, 3) OVER w AS prev_joystick_x_3,
    LAG(joystick_y, 3) OVER w AS prev_joystick_y_3,
    LAG(joystick_x, 4) OVER w AS prev_joystick_x_4,
    LAG(joystick_y, 4) OVER w AS prev_joystick_y_4,
    LAG(joystick_x, 5) OVER w AS prev_joystick_x_5,
    LAG(joystick_y, 5) OVER w AS prev_joystick_y_5,

    -- Previous trigger states (for detecting trigger press)
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

    -- Previous positions
    LAG(position_x, 2) OVER w AS prev_position_x_2,
    LAG(position_y, 2) OVER w AS prev_position_y_2

  FROM read_parquet('replay_parquet_test/partition_*.parquet')
  WINDOW w AS (PARTITION BY replay_file, player_index ORDER BY frame_number)
  -- Add character filter here if needed:
  -- WHERE character_id IN (1, 22)  -- Spacies
),

-- Step 2: Detect wavedash and waveland events
wavedash_events AS (
  SELECT
    replay_file,
    player_index,
    frame_number,
    character_id,
    action_state,
    position_x,
    position_y,

    -- Wavedash type detection (FIXED: matches slippistats logic)
    CASE
      -- WAVEDASH: First frame of LandingFallSpecial that had KneeBend within last 8 frames
      -- This handles both fast wavedashes (KneeBend→LandingFallSpecial directly)
      -- and normal wavedashes (KneeBend→Jump→EscapeAir→LandingFallSpecial)
      WHEN action_state = 43  -- LandingFallSpecial
        AND prev_action_1 != 43  -- First frame of LandingFallSpecial
        AND (prev_action_2 = 24 OR prev_action_3 = 24 OR prev_action_4 = 24
             OR prev_action_5 = 24 OR prev_action_6 = 24 OR prev_action_7 = 24
             OR prev_action_8 = 24)  -- Had KneeBend within 8 frames
      THEN 'wavedash'

      -- WAVELAND: First frame of LandingFallSpecial without KneeBend in recent history
      WHEN action_state = 43  -- LandingFallSpecial
        AND prev_action_1 != 43  -- First frame
        AND prev_action_2 != 24 AND prev_action_3 != 24 AND prev_action_4 != 24
        AND prev_action_5 != 24 AND prev_action_6 != 24 AND prev_action_7 != 24
        AND prev_action_8 != 24  -- NO KneeBend
      THEN 'waveland'

      ELSE NULL
    END AS wavedash_type,

    -- Find which frame had the trigger press (check last 5 frames)
    -- Use joystick from the frame where trigger was pressed for angle calculation
    CASE
      WHEN (button_l AND NOT COALESCE(prev_trigger_l_1, FALSE))
        OR (button_r AND NOT COALESCE(prev_trigger_r_1, FALSE))
      THEN 0  -- Trigger pressed on landing frame (impossible but check anyway)
      WHEN (prev_trigger_l_1 AND NOT COALESCE(prev_trigger_l_2, FALSE))
        OR (prev_trigger_r_1 AND NOT COALESCE(prev_trigger_r_2, FALSE))
      THEN 1
      WHEN (prev_trigger_l_2 AND NOT COALESCE(prev_trigger_l_3, FALSE))
        OR (prev_trigger_r_2 AND NOT COALESCE(prev_trigger_r_3, FALSE))
      THEN 2
      WHEN (prev_trigger_l_3 AND NOT COALESCE(prev_trigger_l_4, FALSE))
        OR (prev_trigger_r_3 AND NOT COALESCE(prev_trigger_r_4, FALSE))
      THEN 3
      WHEN (prev_trigger_l_4 AND NOT COALESCE(prev_trigger_l_5, FALSE))
        OR (prev_trigger_r_4 AND NOT COALESCE(prev_trigger_r_5, FALSE))
      THEN 4
      ELSE 5  -- Default if no trigger press found
    END AS trigger_frames_back,

    -- Get joystick position from trigger frame
    CASE
      WHEN (prev_trigger_l_1 AND NOT COALESCE(prev_trigger_l_2, FALSE))
        OR (prev_trigger_r_1 AND NOT COALESCE(prev_trigger_r_2, FALSE))
      THEN prev_joystick_x_1
      WHEN (prev_trigger_l_2 AND NOT COALESCE(prev_trigger_l_3, FALSE))
        OR (prev_trigger_r_2 AND NOT COALESCE(prev_trigger_r_3, FALSE))
      THEN prev_joystick_x_2
      WHEN (prev_trigger_l_3 AND NOT COALESCE(prev_trigger_l_4, FALSE))
        OR (prev_trigger_r_3 AND NOT COALESCE(prev_trigger_r_4, FALSE))
      THEN prev_joystick_x_3
      WHEN (prev_trigger_l_4 AND NOT COALESCE(prev_trigger_l_5, FALSE))
        OR (prev_trigger_r_4 AND NOT COALESCE(prev_trigger_r_5, FALSE))
      THEN prev_joystick_x_4
      ELSE prev_joystick_x_5
    END AS airdodge_joystick_x,

    CASE
      WHEN (prev_trigger_l_1 AND NOT COALESCE(prev_trigger_l_2, FALSE))
        OR (prev_trigger_r_1 AND NOT COALESCE(prev_trigger_r_2, FALSE))
      THEN prev_joystick_y_1
      WHEN (prev_trigger_l_2 AND NOT COALESCE(prev_trigger_l_3, FALSE))
        OR (prev_trigger_r_2 AND NOT COALESCE(prev_trigger_r_3, FALSE))
      THEN prev_joystick_y_2
      WHEN (prev_trigger_l_3 AND NOT COALESCE(prev_trigger_l_4, FALSE))
        OR (prev_trigger_r_3 AND NOT COALESCE(prev_trigger_r_4, FALSE))
      THEN prev_joystick_y_3
      WHEN (prev_trigger_l_4 AND NOT COALESCE(prev_trigger_l_5, FALSE))
        OR (prev_trigger_r_4 AND NOT COALESCE(prev_trigger_r_5, FALSE))
      THEN prev_joystick_y_4
      ELSE prev_joystick_y_5
    END AS airdodge_joystick_y,

    -- Find KneeBend frame for trigger_frame calculation
    CASE
      WHEN prev_action_2 = 24 THEN 2
      WHEN prev_action_3 = 24 THEN 3
      WHEN prev_action_4 = 24 THEN 4
      WHEN prev_action_5 = 24 THEN 5
      WHEN prev_action_6 = 24 THEN 6
      WHEN prev_action_7 = 24 THEN 7
      WHEN prev_action_8 = 24 THEN 8
      ELSE NULL
    END AS kneebend_frames_back,

    -- Starting position (from previous frames)
    prev_position_x_2 AS start_position_x,
    prev_position_y_2 AS start_position_y

  FROM frame_history
  WHERE action_state = 43  -- Only check LandingFallSpecial frames
),

-- Step 3: Calculate angle and direction
wavedash_metrics AS (
  SELECT
    replay_file,
    player_index,
    frame_number,
    character_id,
    wavedash_type,

    -- Position
    position_x,
    position_y,
    start_position_x,
    start_position_y,

    -- Joystick inputs during airdodge
    airdodge_joystick_x,
    airdodge_joystick_y,

    -- Timing
    trigger_frames_back,
    kneebend_frames_back,
    (COALESCE(kneebend_frames_back, 0) - trigger_frames_back) AS trigger_frame,  -- Frames from KneeBend to trigger press
    trigger_frames_back AS airdodge_frames,  -- Frames from airdodge/trigger to landing

    -- ANGLE CALCULATION
    -- atan2(y, x) returns angle in radians
    -- We want degrees below horizontal, so:
    --   - 0° = horizontal (right)
    --   - 90° = straight down
    --   - Negative angles = upward (should be rare for wavedash)
    -- DuckDB degrees() converts radians to degrees
    CASE
      -- Only calculate if joystick is not in deadzone
      WHEN ABS(airdodge_joystick_x - 0.5) > 0.1 OR ABS(airdodge_joystick_y - 0.5) > 0.1
      THEN ROUND(
        DEGREES(
          ATAN2(
            -(airdodge_joystick_y - 0.5),  -- Negative because down = lower Y in game
            (airdodge_joystick_x - 0.5)    -- Right = positive X
          )
        ),
        2
      )
      ELSE NULL
    END AS angle_degrees,

    -- DIRECTION DETERMINATION
    -- Based on horizontal component of joystick during airdodge
    CASE
      WHEN airdodge_joystick_x < 0.35 THEN 'LEFT'     -- Left wavedash
      WHEN airdodge_joystick_x > 0.65 THEN 'RIGHT'    -- Right wavedash
      ELSE 'DOWN'                                      -- Straight down (spotdodge wavedash)
    END AS direction,

    -- Additional metrics
    ROUND(ABS(position_x - start_position_x), 2) AS horizontal_distance

  FROM wavedash_events
  WHERE wavedash_type IS NOT NULL  -- Only keep actual wavedash/waveland events
)

-- Final output
SELECT
  replay_file,
  player_index,
  frame_number,
  wavedash_type,

  -- Angle and direction
  angle_degrees,
  direction,

  -- Timing metrics
  trigger_frame,
  airdodge_frames,

  -- Joystick inputs (for debugging/verification)
  ROUND(airdodge_joystick_x, 3) AS joystick_x,
  ROUND(airdodge_joystick_y, 3) AS joystick_y,

  -- Position
  ROUND(position_x, 2) AS position_x,
  ROUND(position_y, 2) AS position_y,
  ROUND(start_position_x, 2) AS start_position_x,
  ROUND(start_position_y, 2) AS start_position_y,
  horizontal_distance,

  -- Metadata
  character_id

FROM wavedash_metrics
ORDER BY replay_file, player_index, frame_number;
