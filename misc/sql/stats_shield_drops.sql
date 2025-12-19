-- ============================================================
-- SHIELD DROP ANALYSIS QUERY
-- ============================================================
-- Detects shield drop events with timing analysis:
--   - Shield drop execution (shield → fall through platform)
--   - Out-of-shieldstun frame timing
--   - Shield duration before drop
--   - Position (platform detection)
--
-- Output Schema:
--   replay_file, player_index, frame_number, shield_duration_frames,
--   oo_shieldstun_frame, was_in_shieldstun, position_x, position_y,
--   velocity_y, platform_detected
--
-- Filters: replay_file, character_id
-- Based on slippistats ShieldDropData enhancements
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
-- Step 1: Mark shield frames
shield_frames AS (
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

    -- Velocity (calculated from position change)
    position_y - LAG(position_y) OVER w AS velocity_y,

    -- Hitstun (shieldstun)
    hitstun_remaining,
    is_in_hitstun,

    -- Is in shield?
    -- Shield states: 178-182 (0xB2-0xB6)
    CASE
      WHEN action_state BETWEEN 178 AND 182 THEN 1
      ELSE 0
    END AS in_shield,

    -- Previous shield state
    LAG(CASE WHEN action_state BETWEEN 178 AND 182 THEN 1 ELSE 0 END) OVER w AS prev_in_shield,

    -- Previous action
    LAG(action_state, 1) OVER w AS prev_action,
    LAG(action_state, 2) OVER w AS prev_action_2,

    -- Next action (to detect shield drop)
    LEAD(action_state, 1) OVER w AS next_action,

    -- Previous hitstun (for shieldstun detection)
    LAG(hitstun_remaining, 1) OVER w AS prev_hitstun,
    LAG(is_in_hitstun, 1) OVER w AS prev_in_hitstun,

    -- Previous frame number
    LAG(frame_number, 1) OVER w AS prev_frame

  FROM read_parquet('replay_parquet_test/partition_*.parquet')
  WINDOW w AS (PARTITION BY replay_file, player_index ORDER BY frame_number)
  -- Add character filter here if needed:
  -- WHERE character_id = 1
),

-- Step 2: Detect shield drop events
shield_drop_events AS (
  SELECT
    *,

    -- SHIELD DROP DETECTION
    -- Shield drop = transition from shield (178-182) to falling/passdown
    -- Fall states: 29 (Fall), 30 (FallF), 31 (FallB)
    -- PassDown: 358 (0x166) - falling through platform
    CASE
      WHEN in_shield = 0
        AND COALESCE(prev_in_shield, 0) = 1
        AND action_state IN (29, 30, 31, 358)  -- Falling states
        AND position_y > -20  -- Not at bottom blast zone (indicates on platform)
      THEN 1
      ELSE 0
    END AS shield_drop,

    -- OUT OF SHIELDSTUN FRAME
    -- Frame when shieldstun ended (hitstun_remaining went from >0 to 0 while in shield)
    CASE
      WHEN in_shield = 1
        AND hitstun_remaining = 0
        AND COALESCE(prev_hitstun, 0) > 0
      THEN frame_number
      ELSE NULL
    END AS oo_shieldstun_frame_current

  FROM shield_frames
),

-- Step 3: Assign shield IDs (group consecutive shield frames)
shield_sequences AS (
  SELECT
    *,

    -- Mark shield start
    CASE
      WHEN in_shield = 1 AND COALESCE(prev_in_shield, 0) = 0
      THEN 1
      ELSE 0
    END AS shield_start,

    -- Assign shield sequence ID
    SUM(CASE WHEN in_shield = 1 AND COALESCE(prev_in_shield, 0) = 0 THEN 1 ELSE 0 END) OVER (
      PARTITION BY replay_file, player_index
      ORDER BY frame_number
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS shield_id

  FROM shield_drop_events
),

-- Step 4: Get shield sequence stats
shield_stats AS (
  SELECT
    shield_id,
    replay_file,
    player_index,

    -- Shield duration
    MIN(frame_number) AS shield_start_frame,
    MAX(frame_number) AS shield_end_frame,
    COUNT(*) AS shield_duration_frames,

    -- Out of shieldstun frame (first frame after shieldstun ended)
    MIN(CASE WHEN oo_shieldstun_frame_current IS NOT NULL
      THEN oo_shieldstun_frame_current END) AS oo_shieldstun_frame,

    -- Was in shieldstun at any point
    MAX(CASE WHEN hitstun_remaining > 0 AND in_shield = 1 THEN 1 ELSE 0 END) AS was_in_shieldstun

  FROM shield_sequences
  WHERE in_shield = 1
  GROUP BY shield_id, replay_file, player_index
),

-- Step 5: Join shield drops with shield stats
shield_drops_with_context AS (
  SELECT
    e.replay_file,
    e.player_index,
    e.frame_number,
    e.character_id,

    -- Shield drop
    e.shield_drop,

    -- Position
    e.position_x,
    e.position_y,

    -- Velocity
    e.velocity_y,

    -- Shield context
    s.shield_duration_frames,
    s.oo_shieldstun_frame,
    s.was_in_shieldstun,

    -- Calculate frames from end of shieldstun to shield drop
    CASE
      WHEN s.oo_shieldstun_frame IS NOT NULL
      THEN e.frame_number - s.oo_shieldstun_frame
      ELSE NULL
    END AS frames_after_shieldstun,

    -- Platform detection (approximate: Y position above stage floor)
    -- Most stages have platforms around Y=20-50
    CASE
      WHEN e.position_y > 10 THEN 1
      ELSE 0
    END AS platform_detected

  FROM shield_sequences e
  LEFT JOIN shield_stats s
    ON e.replay_file = s.replay_file
    AND e.player_index = s.player_index
    AND e.shield_id = s.shield_id
  WHERE e.shield_drop = 1  -- Only shield drop events
)

-- Final output
SELECT
  replay_file,
  player_index,
  frame_number AS shield_drop_frame,

  -- Timing
  shield_duration_frames,
  oo_shieldstun_frame,
  frames_after_shieldstun,
  CAST(was_in_shieldstun AS BOOLEAN) AS was_in_shieldstun,

  -- Position
  ROUND(position_x, 2) AS position_x,
  ROUND(position_y, 2) AS position_y,
  ROUND(velocity_y, 2) AS velocity_y,

  -- Platform detection
  CAST(platform_detected AS BOOLEAN) AS platform_detected,

  -- Metadata
  character_id

FROM shield_drops_with_context
ORDER BY replay_file, player_index, frame_number;
