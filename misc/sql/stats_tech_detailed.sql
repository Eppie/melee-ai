-- ============================================================
-- ENHANCED TECH ANALYSIS QUERY
-- ============================================================
-- Tracks tech events with positional analysis and punish detection:
--   - Tech type (neutral, forward, backward, wall, miss)
--   - Position and directional analysis
--   - Towards center / towards opponent
--   - Punish detection (opponent hits within window)
--   - Jab reset detection
--
-- Output Schema:
--   replay_file, player_index, frame_number, tech_type,
--   position_x, position_y, towards_center, towards_opponent,
--   was_punished, punish_frame, jab_reset, stage_id
--
-- Filters: replay_file, character_id
-- Based on slippistats TechData enhancements
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
-- Step 1: Join player data with opponent data
frame_pairs AS (
  SELECT
    p1.replay_file,
    p1.frame_number,
    p1.player_index AS tech_player_idx,
    p2.player_index AS opponent_idx,

    -- Tech player state
    p1.action_state AS tech_action,
    p1.action_state_frame AS tech_action_frame,
    p1.character_id AS tech_character,
    p1.position_x AS tech_pos_x,
    p1.position_y AS tech_pos_y,
    p1.facing AS tech_facing,
    p1.stage_id,

    -- Opponent state
    p2.action_state AS opp_action,
    p2.position_x AS opp_pos_x,
    p2.position_y AS opp_pos_y,
    p2.percent AS opp_percent,

    -- Opponent hitstun (for punish detection)
    LEAD(p1.is_in_hitstun, 1) OVER w1 AS tech_player_hit_next_1,
    LEAD(p1.is_in_hitstun, 15) OVER w1 AS tech_player_hit_next_15,
    LEAD(p1.is_in_hitstun, 30) OVER w1 AS tech_player_hit_next_30,

    -- Frame where tech player gets hit (if any)
    CASE
      WHEN LEAD(p1.is_in_hitstun, 1) OVER w1 AND NOT p1.is_in_hitstun THEN p1.frame_number + 1
      WHEN LEAD(p1.is_in_hitstun, 2) OVER w1 AND NOT LEAD(p1.is_in_hitstun, 1) OVER w1 THEN p1.frame_number + 2
      WHEN LEAD(p1.is_in_hitstun, 3) OVER w1 AND NOT LEAD(p1.is_in_hitstun, 2) OVER w1 THEN p1.frame_number + 3
      WHEN LEAD(p1.is_in_hitstun, 4) OVER w1 AND NOT LEAD(p1.is_in_hitstun, 3) OVER w1 THEN p1.frame_number + 4
      WHEN LEAD(p1.is_in_hitstun, 5) OVER w1 AND NOT LEAD(p1.is_in_hitstun, 4) OVER w1 THEN p1.frame_number + 5
      WHEN LEAD(p1.is_in_hitstun, 10) OVER w1 AND NOT LEAD(p1.is_in_hitstun, 5) OVER w1 THEN p1.frame_number + 10
      WHEN LEAD(p1.is_in_hitstun, 15) OVER w1 AND NOT LEAD(p1.is_in_hitstun, 10) OVER w1 THEN p1.frame_number + 15
      WHEN LEAD(p1.is_in_hitstun, 30) OVER w1 AND NOT LEAD(p1.is_in_hitstun, 15) OVER w1 THEN p1.frame_number + 30
      ELSE NULL
    END AS punish_frame,

    -- Next opponent actions (for jab reset detection)
    LEAD(p2.action_state, 1) OVER w2 AS opp_next_action_1,
    LEAD(p2.action_state, 5) OVER w2 AS opp_next_action_5,
    LEAD(p2.action_state, 10) OVER w2 AS opp_next_action_10

  FROM read_parquet('replay_parquet_test/partition_*.parquet') p1
  INNER JOIN read_parquet('replay_parquet_test/partition_*.parquet') p2
    ON p1.replay_file = p2.replay_file
    AND p1.frame_number = p2.frame_number
    AND p1.player_index != p2.player_index
  WINDOW
    w1 AS (PARTITION BY p1.replay_file, p1.player_index ORDER BY p1.frame_number),
    w2 AS (PARTITION BY p2.replay_file, p2.player_index ORDER BY p2.frame_number)
  -- Add character filter here if needed:
  -- WHERE p1.character_id = 1
),

-- Step 2: Detect tech events
tech_events AS (
  SELECT
    *,

    -- TECH TYPE CLASSIFICATION
    CASE
      WHEN tech_action = 199 THEN 'neutral_tech'       -- 0xC7: NeutralTech
      WHEN tech_action = 200 THEN 'forward_tech'       -- 0xC8: ForwardTech
      WHEN tech_action = 201 THEN 'backward_tech'      -- 0xC9: BackwardTech
      WHEN tech_action = 202 THEN 'wall_tech'          -- 0xCA: WallTech
      WHEN tech_action = 183 THEN 'tech_miss_up'       -- 0xB7: TechMissUp
      WHEN tech_action = 191 THEN 'tech_miss_down'     -- 0xBF: TechMissDown
      WHEN tech_action = 247 THEN 'wall_tech_miss'     -- 0xF7: WallTechFail
      ELSE NULL
    END AS tech_type,

    -- Mark first frame of tech (action_state_frame = 0)
    CASE WHEN tech_action_frame = 0 THEN 1 ELSE 0 END AS tech_start

  FROM frame_pairs
  WHERE tech_action IN (199, 200, 201, 202, 183, 191, 247)  -- All tech states
),

-- Step 3: Only keep first frame of each tech
tech_first_frames AS (
  SELECT * FROM tech_events
  WHERE tech_start = 1
),

-- Step 4: Calculate directional metrics
tech_with_direction AS (
  SELECT
    *,

    -- TOWARDS CENTER
    -- Stage center is typically X=0. Moving toward center means:
    --   - If position_x > 0, forward_tech should have facing left (toward center)
    --   - If position_x < 0, forward_tech should have facing right (toward center)
    CASE
      WHEN tech_type = 'forward_tech' THEN
        CASE
          -- Right side of stage, teching left (toward center)
          WHEN tech_pos_x > 0 AND tech_facing < 0 THEN 1
          -- Left side of stage, teching right (toward center)
          WHEN tech_pos_x < 0 AND tech_facing > 0 THEN 1
          ELSE 0
        END
      WHEN tech_type = 'backward_tech' THEN
        CASE
          -- Right side of stage, teching right (away from center)
          WHEN tech_pos_x > 0 AND tech_facing > 0 THEN 0
          -- Left side of stage, teching left (away from center)
          WHEN tech_pos_x < 0 AND tech_facing < 0 THEN 0
          -- Opposite = toward center
          ELSE 1
        END
      ELSE NULL  -- Neutral tech has no direction
    END AS towards_center,

    -- TOWARDS OPPONENT
    -- Calculate if tech direction moves toward opponent
    CASE
      WHEN tech_type = 'forward_tech' THEN
        CASE
          -- Opponent is to the right, teching right
          WHEN opp_pos_x > tech_pos_x AND tech_facing > 0 THEN 1
          -- Opponent is to the left, teching left
          WHEN opp_pos_x < tech_pos_x AND tech_facing < 0 THEN 1
          ELSE 0
        END
      WHEN tech_type = 'backward_tech' THEN
        CASE
          -- Opponent is to the right, teching left (away)
          WHEN opp_pos_x > tech_pos_x AND tech_facing < 0 THEN 0
          -- Opponent is to the left, teching right (away)
          WHEN opp_pos_x < tech_pos_x AND tech_facing > 0 THEN 0
          ELSE 1
        END
      ELSE NULL
    END AS towards_opponent,

    -- Distance to opponent
    ROUND(SQRT(
      POW(opp_pos_x - tech_pos_x, 2) + POW(opp_pos_y - tech_pos_y, 2)
    ), 2) AS distance_to_opponent,

    -- JAB RESET DETECTION
    -- Jab reset = opponent uses jab (action 44-47) within next 10 frames after tech miss
    CASE
      WHEN tech_type IN ('tech_miss_up', 'tech_miss_down', 'wall_tech_miss')
        AND (
          opp_next_action_1 BETWEEN 44 AND 47  -- Jab actions
          OR opp_next_action_5 BETWEEN 44 AND 47
          OR opp_next_action_10 BETWEEN 44 AND 47
        )
      THEN 1
      ELSE 0
    END AS jab_reset,

    -- WAS PUNISHED
    -- Tech player gets hit within 30 frames after tech
    CASE
      WHEN punish_frame IS NOT NULL
        AND (punish_frame - frame_number) <= 30
      THEN 1
      ELSE 0
    END AS was_punished

  FROM tech_first_frames
)

-- Final output
SELECT
  replay_file,
  tech_player_idx AS player_index,
  frame_number,

  -- Tech type
  tech_type,

  -- Position
  ROUND(tech_pos_x, 2) AS position_x,
  ROUND(tech_pos_y, 2) AS position_y,

  -- Directional analysis
  CAST(towards_center AS BOOLEAN) AS towards_center,
  CAST(towards_opponent AS BOOLEAN) AS towards_opponent,
  distance_to_opponent,

  -- Punish detection
  CAST(was_punished AS BOOLEAN) AS was_punished,
  punish_frame,
  CASE
    WHEN punish_frame IS NOT NULL
    THEN punish_frame - frame_number
    ELSE NULL
  END AS frames_until_punish,

  -- Jab reset
  CAST(jab_reset AS BOOLEAN) AS jab_reset,

  -- Opponent position (for reference)
  ROUND(opp_pos_x, 2) AS opponent_x,
  ROUND(opp_pos_y, 2) AS opponent_y,

  -- Metadata
  stage_id,
  tech_character AS character_id

FROM tech_with_direction
ORDER BY replay_file, player_index, frame_number;
