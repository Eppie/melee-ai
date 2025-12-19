-- ============================================================
-- STOCK LIFECYCLE TRACKING QUERY
-- ============================================================
-- Tracks each stock from spawn to death
--
-- Output Schema:
--   replay_file, player_index, stock_id, start_frame, end_frame,
--   duration_frames, duration_seconds, stocks_at_start, start_percent,
--   max_percent, end_percent, death_action_state, death_type
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
-- Step 1: Detect stock changes
stock_events AS (
  SELECT
    replay_file,
    player_index,
    frame_number,
    stocks,
    percent,
    action_state,
    character_id,

    -- Previous stock count
    LAG(stocks, 1) OVER w AS prev_stocks,

    -- Stock change detection
    CASE WHEN stocks != COALESCE(LAG(stocks, 1) OVER w, stocks)
    THEN 1 ELSE 0 END AS stock_changed,

    -- Stock loss (death) vs stock gain (respawn)
    CASE
      WHEN stocks < COALESCE(LAG(stocks, 1) OVER w, stocks) THEN 'death'
      WHEN stocks > COALESCE(LAG(stocks, 1) OVER w, stocks) THEN 'respawn'
      ELSE NULL
    END AS stock_event_type

  FROM read_parquet('replay_parquet_test/partition_*.parquet')
  WINDOW w AS (PARTITION BY replay_file, player_index ORDER BY frame_number)
  -- Add character filter here if needed:
  -- WHERE character_id = 1  -- Uncomment to filter by character
),

-- Step 2: Mark first frame per player
first_frames AS (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY replay_file, player_index ORDER BY frame_number) AS row_num
  FROM stock_events
),

-- Step 3: Assign stock IDs (increment on respawn or first frame)
stock_ids AS (
  SELECT
    *,
    SUM(CASE WHEN stock_event_type = 'respawn' OR row_num = 1
    THEN 1 ELSE 0 END) OVER (
      PARTITION BY replay_file, player_index
      ORDER BY frame_number
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS stock_id
  FROM first_frames
),

-- Step 4: Aggregate per stock
stock_lifecycle AS (
  SELECT
    replay_file,
    player_index,
    stock_id,

    -- Start/end frames
    MIN(frame_number) AS start_frame,
    MAX(frame_number) AS end_frame,
    MAX(frame_number) - MIN(frame_number) AS duration_frames,

    -- Stock count at start
    FIRST(stocks ORDER BY frame_number) AS stocks_at_start,

    -- Percent tracking
    FIRST(percent ORDER BY frame_number) AS start_percent,
    MAX(percent) AS max_percent,
    LAST(percent ORDER BY frame_number) AS end_percent,

    -- Death animation (action state at death)
    MAX(CASE WHEN stock_event_type = 'death'
      THEN action_state END) AS death_action_state,

    -- Death classification
    CASE
      -- Blast zone deaths: action states 0x00-0x0B (0-11)
      WHEN MAX(CASE WHEN stock_event_type = 'death' THEN action_state END) BETWEEN 0 AND 11
        THEN 'blast_zone'
      -- Stock ended but no death detected (survived to end of game)
      WHEN MAX(CASE WHEN stock_event_type = 'death' THEN action_state END) IS NULL
        THEN 'survived'
      -- Other death types
      ELSE 'other'
    END AS death_type

  FROM stock_ids
  GROUP BY replay_file, player_index, stock_id
)

-- Final output
SELECT
  replay_file,
  player_index,
  stock_id,
  start_frame,
  end_frame,
  duration_frames,
  ROUND(duration_frames / 60.0, 2) AS duration_seconds,
  stocks_at_start,
  start_percent,
  max_percent,
  end_percent,
  death_action_state,
  death_type
FROM stock_lifecycle
ORDER BY replay_file, player_index, stock_id;
