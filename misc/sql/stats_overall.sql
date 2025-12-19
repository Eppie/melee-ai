-- ============================================================
-- OVERALL PLAYER STATISTICS QUERY
-- ============================================================
-- Aggregate high-level per-player statistics
--
-- Output Schema:
--   replay_file, player_index, character_id, total_frames, game_duration_seconds,
--   total_damage_dealt, total_damage_taken, kill_count, death_count,
--   final_stocks_remaining, avg_percent, max_percent_reached,
--   total_attacks_thrown, time_in_hitstun_frames, time_in_hitstun_percent,
--   time_on_ground_percent, time_off_stage_percent
--
-- Filters: replay_file, character_id
--
-- Note: For APM and detailed input metrics, use stats_inputs.sql
-- Note: For combo/conversion counts, use stats_combos.sql and stats_conversions.sql
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
-- Step 1: Join with opponent for damage/kill tracking
frame_pairs AS (
  SELECT
    p1.replay_file,
    p1.frame_number,
    p1.player_index,
    p1.character_id,
    p1.percent AS my_percent,
    p1.stocks AS my_stocks,
    p1.action_state AS my_action,
    p1.hitstun_remaining AS my_hitstun,
    p1.on_ground AS my_on_ground,
    p1.off_stage AS my_off_stage,

    p2.percent AS opp_percent,
    p2.stocks AS opp_stocks,

    -- Previous frame states
    LAG(p1.percent) OVER w AS prev_my_percent,
    LAG(p1.stocks) OVER w AS prev_my_stocks,
    LAG(p2.percent) OVER w AS prev_opp_percent,
    LAG(p2.stocks) OVER w AS prev_opp_stocks

  FROM read_parquet('replay_parquet_test/partition_*.parquet') p1
  INNER JOIN read_parquet('replay_parquet_test/partition_*.parquet') p2
    ON p1.replay_file = p2.replay_file
    AND p1.frame_number = p2.frame_number
    AND p1.player_index != p2.player_index
  WINDOW w AS (PARTITION BY p1.replay_file, p1.player_index ORDER BY p1.frame_number)
  -- Add character filter here if needed:
  -- WHERE p1.character_id = 1  -- Uncomment to filter by character
),

-- Step 2: Compute damage and kills per frame
frame_stats AS (
  SELECT
    *,

    -- Damage dealt to opponent this frame
    CASE WHEN
      opp_percent > COALESCE(prev_opp_percent, opp_percent)
    THEN opp_percent - prev_opp_percent
    ELSE 0 END AS damage_dealt_this_frame,

    -- Damage taken this frame
    CASE WHEN
      my_percent > COALESCE(prev_my_percent, my_percent)
    THEN my_percent - prev_my_percent
    ELSE 0 END AS damage_taken_this_frame,

    -- Kill (opponent lost stock)
    CASE WHEN
      opp_stocks < COALESCE(prev_opp_stocks, opp_stocks)
    THEN 1 ELSE 0 END AS kill_this_frame,

    -- Death (I lost stock)
    CASE WHEN
      my_stocks < COALESCE(prev_my_stocks, my_stocks)
    THEN 1 ELSE 0 END AS death_this_frame,

    -- Attack thrown (entering attack action states)
    CASE WHEN
      my_action BETWEEN 44 AND 69  -- All attack states (jabs to aerials)
      AND my_action != COALESCE(LAG(my_action) OVER (
        PARTITION BY replay_file, player_index ORDER BY frame_number
      ), my_action)
    THEN 1 ELSE 0 END AS attack_thrown

  FROM frame_pairs
),

-- Step 3: Aggregate per player
player_overall_stats AS (
  SELECT
    replay_file,
    player_index,
    FIRST(character_id ORDER BY frame_number) AS character_id,

    -- Game duration
    COUNT(*) AS total_frames,
    ROUND(COUNT(*) / 60.0, 2) AS game_duration_seconds,
    ROUND(COUNT(*) / 3600.0, 2) AS game_duration_minutes,

    -- Damage metrics
    SUM(damage_dealt_this_frame) AS total_damage_dealt,
    SUM(damage_taken_this_frame) AS total_damage_taken,
    ROUND(SUM(damage_dealt_this_frame) * 60.0 / NULLIF(COUNT(*), 0), 2) AS damage_dealt_per_second,

    -- Kill/death counts
    SUM(kill_this_frame) AS kill_count,
    SUM(death_this_frame) AS death_count,

    -- Final stocks remaining
    LAST(my_stocks ORDER BY frame_number) AS final_stocks_remaining,

    -- Percent statistics
    ROUND(AVG(my_percent), 2) AS avg_percent,
    MAX(my_percent) AS max_percent_reached,
    FIRST(my_percent ORDER BY frame_number) AS start_percent,
    LAST(my_percent ORDER BY frame_number) AS final_percent,

    -- Attack count
    SUM(attack_thrown) AS total_attacks_thrown,
    ROUND(SUM(attack_thrown) * 60.0 / NULLIF(COUNT(*), 0), 2) AS attacks_per_second,

    -- State time analysis
    SUM(CASE WHEN my_hitstun > 0 THEN 1 ELSE 0 END) AS time_in_hitstun_frames,
    ROUND(100.0 * SUM(CASE WHEN my_hitstun > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS time_in_hitstun_percent,

    SUM(CASE WHEN my_on_ground THEN 1 ELSE 0 END) AS time_on_ground_frames,
    ROUND(100.0 * SUM(CASE WHEN my_on_ground THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS time_on_ground_percent,

    SUM(CASE WHEN my_off_stage THEN 1 ELSE 0 END) AS time_off_stage_frames,
    ROUND(100.0 * SUM(CASE WHEN my_off_stage THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS time_off_stage_percent,

    -- Derived metrics
    ROUND(NULLIF(SUM(damage_dealt_this_frame), 0) / NULLIF(SUM(kill_this_frame), 0), 2) AS avg_damage_per_kill,
    ROUND(NULLIF(SUM(damage_taken_this_frame), 0) / NULLIF(SUM(death_this_frame), 0), 2) AS avg_damage_before_death

  FROM frame_stats
  GROUP BY replay_file, player_index
)

-- Final output
SELECT * FROM player_overall_stats
ORDER BY replay_file, player_index;
