-- ============================================================
-- CONVERSIONS DETECTION QUERY
-- ============================================================
-- Tracks punish sequences from opening to opponent regaining neutral
--
-- Output Schema:
--   replay_file, player_index, conversion_id, start_frame, end_frame,
--   duration_frames, duration_seconds, total_damage, start_percent,
--   end_percent, opening_type, did_kill, moves_used
--
-- Filters: replay_file, character_id
-- Reset Timer: 45 frames after opponent regains control
-- ============================================================

-- FILTERING CONFIGURATION
-- Uncomment and modify these lines to filter results:
--
-- Filter by specific replay file:
-- WHERE replay_file = 'master-diamond-000543d6034bf69893c60a05.slp'
--
-- Filter by character (player performing the conversion):
-- WHERE character_id = 1  -- Fox
-- WHERE character_id IN (1, 2, 20)  -- Fox, Captain Falcon, Young Link
--
-- ============================================================

WITH
-- Step 1: Join with opponent state
frame_pairs AS (
  SELECT
    p1.replay_file,
    p1.frame_number,
    p1.player_index AS attacker_idx,
    p2.player_index AS defender_idx,

    p1.action_state AS attacker_action,
    p1.action_state_frame AS attacker_action_frame,
    p1.percent AS attacker_percent,
    p1.is_in_hitstun AS attacker_in_hitstun,
    p1.character_id AS attacker_character,

    p2.action_state AS defender_action,
    p2.action_state_frame AS defender_action_frame,
    p2.hitstun_remaining AS defender_hitstun,
    p2.percent AS defender_percent,
    p2.stocks AS defender_stocks,
    p2.is_in_hitstun AS defender_in_hitstun,
    p2.is_in_hitlag AS defender_in_hitlag,
    p2.on_ground AS defender_on_ground,

    -- Previous frame states
    LAG(p1.is_in_hitstun) OVER w AS prev_attacker_hitstun,
    LAG(p2.percent) OVER w AS prev_defender_percent,
    LAG(p2.stocks) OVER w AS prev_defender_stocks

  FROM read_parquet('replay_parquet_test/partition_*.parquet') p1
  INNER JOIN read_parquet('replay_parquet_test/partition_*.parquet') p2
    ON p1.replay_file = p2.replay_file
    AND p1.frame_number = p2.frame_number
    AND p1.player_index != p2.player_index
  WINDOW w AS (PARTITION BY p1.replay_file, p1.player_index, p2.player_index ORDER BY p1.frame_number)
  -- Add character filter here if needed:
  -- WHERE p1.character_id = 1  -- Uncomment to filter by attacker character
),

-- Step 2: Classify defender state (neutral vs disadvantage)
classified_frames AS (
  SELECT
    *,

    -- Defender in disadvantage if any of these conditions
    CASE WHEN
      defender_hitstun > 0
      OR defender_in_hitstun
      OR defender_in_hitlag
      OR (defender_action BETWEEN 75 AND 92)  -- 0x4B-0x5C: Damage states
      OR (defender_action BETWEEN 226 AND 230)  -- 0xE2-0xE6: Grabbed
      OR (defender_action BETWEEN 219 AND 222)  -- 0xDB-0xDE: Thrown
      OR (defender_action BETWEEN 183 AND 198)  -- 0xB7-0xC6: Tech/down states
      OR (defender_action = 38)  -- 0x26: DamageFall
      OR (defender_action BETWEEN 35 AND 37)  -- 0x23-0x25: Special fall
      OR (defender_action = 247)  -- 0xF7: Missed wall tech
    THEN 1 ELSE 0 END AS defender_disadvantage,

    -- Damage delta
    defender_percent - COALESCE(prev_defender_percent, defender_percent) AS damage_delta,

    -- Stock loss
    CASE WHEN defender_stocks < COALESCE(prev_defender_stocks, defender_stocks)
    THEN 1 ELSE 0 END AS stock_lost

  FROM frame_pairs
),

-- Step 3: Detect opening types
opening_classified AS (
  SELECT
    *,

    -- Opening frame: transition from neutral to disadvantage
    CASE WHEN
      defender_disadvantage = 1
      AND COALESCE(LAG(defender_disadvantage) OVER (
        PARTITION BY replay_file, attacker_idx, defender_idx
        ORDER BY frame_number
      ), 0) = 0
    THEN 1 ELSE 0 END AS is_opening_frame,

    -- Classify opening type
    CASE
      -- Neutral win: attacker was not in hitstun recently
      WHEN defender_disadvantage = 1
        AND COALESCE(prev_attacker_hitstun, FALSE) = FALSE
        AND damage_delta > 0
      THEN 'neutral-win'

      -- Counter-attack: attacker was in hitstun within last 60 frames
      WHEN defender_disadvantage = 1
        AND attacker_in_hitstun = FALSE
        AND COALESCE(prev_attacker_hitstun, FALSE) = TRUE
      THEN 'counter-attack'

      -- Trade: both players in disadvantage states
      WHEN defender_disadvantage = 1
        AND damage_delta > 0
        AND attacker_in_hitstun
      THEN 'trade'

      ELSE 'unknown'
    END AS opening_type

  FROM classified_frames
  WHERE defender_disadvantage = 1
),

-- Step 4: Group conversions (45-frame reset)
conversion_groups AS (
  SELECT
    *,
    -- Mark conversion start
    CASE WHEN
      is_opening_frame = 1
      OR (frame_number - COALESCE(LAG(frame_number) OVER w, frame_number - 100)) > 45
    THEN 1 ELSE 0 END AS conversion_start

  FROM opening_classified
  WINDOW w AS (PARTITION BY replay_file, attacker_idx, defender_idx ORDER BY frame_number)
),

-- Step 5: Assign conversion IDs
conversion_ids AS (
  SELECT
    *,
    SUM(conversion_start) OVER (
      PARTITION BY replay_file, attacker_idx, defender_idx
      ORDER BY frame_number
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS conversion_id
  FROM conversion_groups
),

-- Step 6: Aggregate per conversion
conversion_stats AS (
  SELECT
    replay_file,
    attacker_idx,
    defender_idx,
    conversion_id,
    attacker_character,

    MIN(frame_number) AS start_frame,
    MAX(frame_number) AS end_frame,
    COUNT(*) AS duration_frames,

    -- Damage dealt
    COALESCE(SUM(CASE WHEN damage_delta > 0 THEN damage_delta ELSE 0 END), 0) AS total_damage,

    -- Opening type (use first non-null, prioritize non-unknown)
    COALESCE(
      MIN(CASE WHEN is_opening_frame = 1 AND opening_type != 'unknown' THEN opening_type END),
      'neutral-win'
    ) AS opening_type,

    -- Start/end percent
    FIRST(defender_percent ORDER BY frame_number) AS start_percent,
    LAST(defender_percent ORDER BY frame_number) AS end_percent,

    -- Did kill
    MAX(stock_lost) AS did_kill,

    -- Moves used
    COUNT(DISTINCT attacker_action) AS moves_used

  FROM conversion_ids
  GROUP BY replay_file, attacker_idx, defender_idx, conversion_id, attacker_character
  HAVING COUNT(*) >= 2
)

-- Final output
SELECT
  replay_file,
  attacker_idx AS player_index,
  conversion_id,
  start_frame,
  end_frame,
  duration_frames,
  ROUND(duration_frames / 60.0, 2) AS duration_seconds,
  total_damage,
  start_percent,
  end_percent,
  opening_type,
  CAST(did_kill AS BOOLEAN) AS did_kill,
  moves_used
FROM conversion_stats
ORDER BY replay_file, player_index, start_frame;
