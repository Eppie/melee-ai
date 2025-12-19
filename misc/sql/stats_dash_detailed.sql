-- ============================================================
-- ENHANCED DASH ANALYSIS QUERY
-- ============================================================
-- Tracks dash sequences with positional data and dashdance detection:
--   - Start/end position for each dash
--   - Distance traveled
--   - Dashdance pattern detection (dash → turn → dash back)
--   - Dash duration and direction
--
-- Output Schema:
--   replay_file, player_index, dash_id, start_frame, end_frame,
--   duration_frames, start_x, start_y, end_x, end_y, distance,
--   direction, is_dashdance, facing_changed
--
-- Filters: replay_file, character_id
-- Based on slippistats DashData enhancements
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
-- Step 1: Mark dash frames and detect dash start/end
dash_frames AS (
  SELECT
    replay_file,
    player_index,
    frame_number,
    action_state,
    character_id,

    -- Position
    position_x,
    position_y,

    -- Facing direction (1 = right, -1 = left in libmelee)
    facing,

    -- Velocity (calculated from position changes)
    position_x - LAG(position_x) OVER w AS velocity_x,

    -- Is this frame a dash/run?
    CASE WHEN action_state IN (20, 21) THEN 1 ELSE 0 END AS in_dash,

    -- Previous/next dash state
    LAG(CASE WHEN action_state IN (20, 21) THEN 1 ELSE 0 END) OVER w AS prev_in_dash,
    LEAD(CASE WHEN action_state IN (20, 21) THEN 1 ELSE 0 END) OVER w AS next_in_dash,

    -- Previous action (for dashdance detection)
    LAG(action_state, 1) OVER w AS prev_action,
    LAG(action_state, 2) OVER w AS prev_action_2,
    LAG(action_state, 3) OVER w AS prev_action_3,

    -- Previous frame number
    LAG(frame_number, 1) OVER w AS prev_frame,

    -- Previous facing direction (for turn detection)
    LAG(facing) OVER w AS prev_facing

  FROM read_parquet('replay_parquet_test/partition_*.parquet')
  WINDOW w AS (PARTITION BY replay_file, player_index ORDER BY frame_number)
  -- Add character filter here if needed:
  -- WHERE character_id = 1
),

-- Step 2: Mark dash start events
dash_starts AS (
  SELECT
    *,
    -- Dash start: entering dash/run from non-dash state
    CASE WHEN
      in_dash = 1
      AND COALESCE(prev_in_dash, 0) = 0
    THEN 1 ELSE 0 END AS dash_start,

    -- Check if preceded by Turn (18) within last 3 frames (for dashdance detection)
    CASE WHEN
      prev_action = 18  -- Turn
      OR prev_action_2 = 18
      OR prev_action_3 = 18
    THEN 1 ELSE 0 END AS preceded_by_turn

  FROM dash_frames
),

-- Step 3: Assign dash IDs (increment on each dash start)
dash_ids AS (
  SELECT
    *,
    SUM(dash_start) OVER (
      PARTITION BY replay_file, player_index
      ORDER BY frame_number
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS dash_id
  FROM dash_starts
  WHERE in_dash = 1  -- Only keep dash frames
),

-- Step 4: Aggregate per dash sequence
dash_sequences AS (
  SELECT
    replay_file,
    player_index,
    dash_id,
    character_id,

    -- Frames
    MIN(frame_number) AS start_frame,
    MAX(frame_number) AS end_frame,
    COUNT(*) AS duration_frames,

    -- Position (start = first frame, end = last frame)
    FIRST(position_x ORDER BY frame_number) AS start_x,
    FIRST(position_y ORDER BY frame_number) AS start_y,
    LAST(position_x ORDER BY frame_number) AS end_x,
    LAST(position_y ORDER BY frame_number) AS end_y,

    -- Facing direction
    FIRST(facing ORDER BY frame_number) AS start_facing,
    LAST(facing ORDER BY frame_number) AS end_facing,

    -- Turn detection
    MAX(preceded_by_turn) AS was_preceded_by_turn,

    -- Velocity
    AVG(ABS(velocity_x)) AS avg_velocity_x

  FROM dash_ids
  GROUP BY replay_file, player_index, dash_id, character_id
),

-- Step 5: Calculate derived metrics
dash_metrics AS (
  SELECT
    *,

    -- DISTANCE CALCULATION
    ROUND(SQRT(
      POW(end_x - start_x, 2) + POW(end_y - start_y, 2)
    ), 2) AS distance,

    -- DIRECTION (based on horizontal movement)
    CASE
      WHEN (end_x - start_x) > 0.1 THEN 'RIGHT'
      WHEN (end_x - start_x) < -0.1 THEN 'LEFT'
      ELSE 'NEUTRAL'
    END AS direction,

    -- FACING CHANGED (indicates turn during dash)
    CASE WHEN start_facing != end_facing THEN 1 ELSE 0 END AS facing_changed

  FROM dash_sequences
),

-- Step 6: Detect dashdance patterns
-- Dashdance = rapid dash → turn → dash in opposite direction
dash_with_dashdance AS (
  SELECT
    *,

    -- Get previous dash metrics
    LAG(direction) OVER w AS prev_dash_direction,
    LAG(end_frame) OVER w AS prev_dash_end_frame,
    LAG(distance) OVER w AS prev_dash_distance,

    -- IS_DASHDANCE: current dash was preceded by turn AND direction reversed
    CASE WHEN
      was_preceded_by_turn = 1
      AND LAG(direction) OVER w IS NOT NULL
      AND direction != LAG(direction) OVER w
      AND direction IN ('LEFT', 'RIGHT')  -- Exclude neutral
      AND LAG(direction) OVER w IN ('LEFT', 'RIGHT')
      -- Gap between dashes should be small (< 10 frames = turn animation)
      AND (start_frame - LAG(end_frame) OVER w) < 10
    THEN 1 ELSE 0 END AS is_dashdance

  FROM dash_metrics
  WINDOW w AS (PARTITION BY replay_file, player_index ORDER BY start_frame)
)

-- Final output
SELECT
  replay_file,
  player_index,
  dash_id,

  -- Timing
  start_frame,
  end_frame,
  duration_frames,
  ROUND(duration_frames / 60.0, 2) AS duration_seconds,

  -- Position
  ROUND(start_x, 2) AS start_x,
  ROUND(start_y, 2) AS start_y,
  ROUND(end_x, 2) AS end_x,
  ROUND(end_y, 2) AS end_y,

  -- Metrics
  distance,
  direction,
  ROUND(avg_velocity_x, 2) AS avg_velocity_x,

  -- Dashdance detection
  CAST(is_dashdance AS BOOLEAN) AS is_dashdance,
  CAST(facing_changed AS BOOLEAN) AS facing_changed,
  CAST(was_preceded_by_turn AS BOOLEAN) AS was_preceded_by_turn,

  -- Metadata
  character_id

FROM dash_with_dashdance
WHERE dash_id > 0  -- Exclude null dashes
ORDER BY replay_file, player_index, start_frame;
