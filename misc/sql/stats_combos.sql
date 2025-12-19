-- ============================================================
-- COMBOS DETECTION QUERY
-- ============================================================
-- Detects combo sequences where opponent is in hitstun/damage states
--
-- Output Schema:
--   replay_file, player_index, combo_id, start_frame, end_frame,
--   duration_frames, duration_seconds, total_damage, start_percent,
--   end_percent, moves_landed, did_kill, opening_type
--
-- Filters: replay_file, character_id
-- Reset Timer: 45 frames without damage/hitstun
-- ============================================================

-- FILTERING CONFIGURATION
-- Uncomment and modify these lines to filter results:
--
-- Filter by specific replay file:
-- WHERE replay_file = 'master-diamond-000543d6034bf69893c60a05.slp'
--
-- Filter by character (player performing the combo):
-- WHERE character_id = 1  -- Fox
-- WHERE character_id IN (1, 2, 20)  -- Fox, Captain Falcon, Young Link
--
-- ============================================================

WITH
-- Step 1: Create opponent state join (each player sees opponent's state)
frame_pairs AS (
  SELECT
    p1.replay_file,
    p1.frame_number,
    p1.player_index AS attacker_idx,
    p2.player_index AS defender_idx,

    -- Attacker state
    p1.action_state AS attacker_action,
    p1.percent AS attacker_percent,
    p1.character_id AS attacker_character,

    -- Defender state
    p2.action_state AS defender_action,
    p2.hitstun_remaining AS defender_hitstun,
    p2.percent AS defender_percent,
    p2.stocks AS defender_stocks,
    p2.is_in_hitstun AS defender_in_hitstun,

    -- Damage delta (current - previous)
    p2.percent - LAG(p2.percent) OVER (
      PARTITION BY p1.replay_file, p1.player_index, p2.player_index
      ORDER BY p1.frame_number
    ) AS damage_this_frame,

    -- Previous defender stocks for kill detection
    LAG(p2.stocks) OVER (
      PARTITION BY p1.replay_file, p1.player_index, p2.player_index
      ORDER BY p1.frame_number
    ) AS prev_defender_stocks

  FROM read_parquet('replay_parquet_test/partition_*.parquet') p1
  INNER JOIN read_parquet('replay_parquet_test/partition_*.parquet') p2
    ON p1.replay_file = p2.replay_file
    AND p1.frame_number = p2.frame_number
    AND p1.player_index != p2.player_index
  -- Add character filter here if needed:
  -- WHERE p1.character_id = 1  -- Uncomment to filter by attacker character
),

-- Step 2: Mark frames where defender is in combo state
combo_frames AS (
  SELECT
    *,
    -- Defender in combo if: hitstun > 0 OR damage state OR grabbed
    CASE WHEN
      defender_hitstun > 0
      OR defender_in_hitstun
      OR (defender_action BETWEEN 75 AND 92)  -- 0x4B-0x5C: Damage states
      OR (defender_action BETWEEN 226 AND 230)  -- 0xE2-0xE6: Grabbed states
      OR (defender_action BETWEEN 219 AND 222)  -- 0xDB-0xDE: Being thrown
      OR (defender_action = 38)  -- 0x26: DamageFall
    THEN 1 ELSE 0 END AS in_combo_state,

    -- Stock decrease indicates kill
    CASE WHEN
      defender_stocks < COALESCE(prev_defender_stocks, defender_stocks)
    THEN 1 ELSE 0 END AS is_kill_frame

  FROM frame_pairs
),

-- Step 3: Group consecutive combo frames (with 45-frame reset timer)
combo_groups AS (
  SELECT
    *,
    -- Gap from previous combo frame
    frame_number - LAG(frame_number) OVER w AS frame_gap,

    -- Mark combo start: first frame OR gap > 45
    CASE WHEN
      in_combo_state = 1
      AND (
        LAG(in_combo_state) OVER w IS NULL
        OR LAG(in_combo_state) OVER w = 0
        OR (frame_number - LAG(frame_number) OVER w) > 45
      )
    THEN 1 ELSE 0 END AS combo_start

  FROM combo_frames
  WHERE in_combo_state = 1  -- Only combo frames
  WINDOW w AS (PARTITION BY replay_file, attacker_idx, defender_idx ORDER BY frame_number)
),

-- Step 4: Assign combo IDs
combo_ids AS (
  SELECT
    *,
    SUM(combo_start) OVER (
      PARTITION BY replay_file, attacker_idx, defender_idx
      ORDER BY frame_number
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS combo_id
  FROM combo_groups
),

-- Step 5: Aggregate per combo
combo_stats AS (
  SELECT
    replay_file,
    attacker_idx,
    defender_idx,
    combo_id,
    attacker_character,

    -- Combo metadata
    MIN(frame_number) AS start_frame,
    MAX(frame_number) AS end_frame,
    COUNT(*) AS combo_duration_frames,

    -- Damage dealt (sum of positive damage deltas only)
    COALESCE(SUM(CASE WHEN damage_this_frame > 0 THEN damage_this_frame ELSE 0 END), 0) AS total_damage,

    -- Percent at start/end
    FIRST(defender_percent ORDER BY frame_number) AS start_percent,
    LAST(defender_percent ORDER BY frame_number) AS end_percent,

    -- Kill detection
    MAX(is_kill_frame) AS did_kill,

    -- Move count (count distinct attacker action states during combo)
    COUNT(DISTINCT attacker_action) AS moves_landed,

    -- Opening type (simplified classification)
    CASE
      WHEN FIRST(defender_percent ORDER BY frame_number) < 20 THEN 'neutral-win'
      WHEN MAX(defender_hitstun) > 15 THEN 'counter-attack'
      ELSE 'trade'
    END AS opening_type

  FROM combo_ids
  GROUP BY replay_file, attacker_idx, defender_idx, combo_id, attacker_character
  HAVING COUNT(*) >= 2  -- Minimum 2 frames to be a combo
)

-- Final output
SELECT
  replay_file,
  attacker_idx AS player_index,
  combo_id,
  start_frame,
  end_frame,
  combo_duration_frames AS duration_frames,
  ROUND(combo_duration_frames / 60.0, 2) AS duration_seconds,
  total_damage,
  start_percent,
  end_percent,
  moves_landed,
  CAST(did_kill AS BOOLEAN) AS did_kill,
  opening_type
FROM combo_stats
ORDER BY replay_file, player_index, start_frame;
