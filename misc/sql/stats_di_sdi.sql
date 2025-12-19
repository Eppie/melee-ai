-- ============================================================
-- DI/SDI/ASDI ANALYSIS QUERY
-- ============================================================
-- Analyzes Directional Influence (DI), Smash DI (SDI), and ASDI:
--   - Hitlag duration (frames in hitstun before knockback)
--   - SDI inputs (stick region changes during hitlag)
--   - DI stick position (average stick during hitlag)
--   - Knockback angle before/after DI
--   - DI efficacy (percentage improvement)
--   - Crouch cancel detection
--
-- Output Schema:
--   replay_file, player_index, frame_number, hitlag_frames,
--   sdi_inputs, di_angle_degrees, kb_angle_before, kb_angle_after,
--   di_efficacy_pct, is_crouch_cancel, damage_taken, percent_before
--
-- Filters: replay_file, character_id
-- Based on slippistats TakeHitData enhancements
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
-- Step 1: Get all frame data with hit detection states
all_frames AS (
  SELECT
    replay_file,
    player_index,
    frame_number,
    character_id,
    action_state,

    -- State
    is_in_hitstun,
    is_in_hitlag,
    hitstun_remaining,
    hitlag_left,
    percent,
    stocks,
    position_x,
    position_y,
    on_ground,

    -- Knockback velocity from parquet (actual game physics)
    speed_x_attack,
    speed_y_attack,

    -- Joystick (for DI/SDI analysis)
    joystick_x,
    joystick_y,

    -- Previous state (for state_before_hit)
    LAG(action_state) OVER w AS prev_action_state,
    LAG(is_in_hitstun) OVER w AS prev_in_hitstun,
    LAG(is_in_hitlag) OVER w AS prev_in_hitlag,
    LAG(percent) OVER w AS prev_percent,

    -- Knockback velocities at different frames after hit (for final KB after hitlag)
    LEAD(speed_x_attack, 1) OVER w AS kb_vel_x_1,
    LEAD(speed_y_attack, 1) OVER w AS kb_vel_y_1,
    LEAD(speed_x_attack, 2) OVER w AS kb_vel_x_2,
    LEAD(speed_y_attack, 2) OVER w AS kb_vel_y_2,
    LEAD(speed_x_attack, 3) OVER w AS kb_vel_x_3,
    LEAD(speed_y_attack, 3) OVER w AS kb_vel_y_3,
    LEAD(speed_x_attack, 4) OVER w AS kb_vel_x_4,
    LEAD(speed_y_attack, 4) OVER w AS kb_vel_y_4,
    LEAD(speed_x_attack, 5) OVER w AS kb_vel_x_5,
    LEAD(speed_y_attack, 5) OVER w AS kb_vel_y_5,
    LEAD(speed_x_attack, 6) OVER w AS kb_vel_x_6,
    LEAD(speed_y_attack, 6) OVER w AS kb_vel_y_6,
    LEAD(speed_x_attack, 7) OVER w AS kb_vel_x_7,
    LEAD(speed_y_attack, 7) OVER w AS kb_vel_y_7,
    LEAD(speed_x_attack, 8) OVER w AS kb_vel_x_8,
    LEAD(speed_y_attack, 8) OVER w AS kb_vel_y_8,

    -- Joystick in next frames (for hitlag duration and SDI detection)
    LEAD(joystick_x, 1) OVER w AS next_joystick_x_1,
    LEAD(joystick_y, 1) OVER w AS next_joystick_y_1,
    LEAD(joystick_x, 2) OVER w AS next_joystick_x_2,
    LEAD(joystick_y, 2) OVER w AS next_joystick_y_2,
    LEAD(joystick_x, 3) OVER w AS next_joystick_x_3,
    LEAD(joystick_y, 3) OVER w AS next_joystick_y_3,
    LEAD(joystick_x, 4) OVER w AS next_joystick_x_4,
    LEAD(joystick_y, 4) OVER w AS next_joystick_y_4,
    LEAD(joystick_x, 5) OVER w AS next_joystick_x_5,
    LEAD(joystick_y, 5) OVER w AS next_joystick_y_5,
    LEAD(joystick_x, 6) OVER w AS next_joystick_x_6,
    LEAD(joystick_y, 6) OVER w AS next_joystick_y_6,
    LEAD(joystick_x, 7) OVER w AS next_joystick_x_7,
    LEAD(joystick_y, 7) OVER w AS next_joystick_y_7,
    LEAD(joystick_x, 8) OVER w AS next_joystick_x_8,
    LEAD(joystick_y, 8) OVER w AS next_joystick_y_8

  FROM read_parquet('replay_parquet_test/partition_*.parquet')
  WINDOW w AS (PARTITION BY replay_file, player_index ORDER BY frame_number)
  -- Add character filter here if needed:
  -- WHERE character_id = 1
),

-- Step 2: Detect hit events (both hitstun and hitlag-only like pummels/throws)
hit_events AS (
  SELECT
    *,

    -- Damage taken this frame
    CASE WHEN prev_percent IS NOT NULL
      THEN percent - prev_percent
      ELSE 0
    END AS damage_taken,

    -- HIT DETECTION: entering hitstun OR entering hitlag (for pummels/throws)
    -- Must also have percent increase to distinguish from attacker's hitlag
    CASE WHEN
      -- Entering hitstun (most hits)
      (is_in_hitstun AND NOT COALESCE(prev_in_hitstun, FALSE))
      -- OR entering hitlag without hitstun (pummels, throws)
      OR (is_in_hitlag AND NOT COALESCE(prev_in_hitlag, FALSE) AND NOT is_in_hitstun)
    THEN 1 ELSE 0 END AS hit_start

  FROM all_frames
  -- Filter to frames where damage was taken (receiving player, not attacker)
  WHERE percent > COALESCE(prev_percent, 0)
),

-- Step 3: Estimate hitlag duration and collect DI inputs
-- Hitlag = frames where player is frozen before knockback
-- Can use hitlag_left from parquet or estimate from velocity
hitlag_analysis AS (
  SELECT
    *,

    -- HITLAG DURATION (directly from parquet - reliable)
    CAST(COALESCE(hitlag_left, 0) AS INT) AS hitlag_frames,

    -- DI ANGLE (average stick position during hitlag)
    -- Calculate from current + next few frames (hitlag window)
    CASE
      WHEN ABS(joystick_x - 0.5) > 0.1 OR ABS(joystick_y - 0.5) > 0.1
      THEN ROUND(
        DEGREES(
          ATAN2(
            joystick_y - 0.5,
            joystick_x - 0.5
          )
        ),
        2
      )
      ELSE NULL
    END AS di_angle_degrees,

    -- SDI INPUT COUNTING
    -- Count distinct stick regions during hitlag (region changes = SDI inputs)
    -- Simplified: count if stick moved to different region in next frames
    CASE
      -- Region at frame 0
      WHEN ABS(joystick_x - 0.5) < 0.2 AND ABS(joystick_y - 0.5) < 0.2 THEN 'neutral'
      WHEN joystick_x < 0.35 AND joystick_y > 0.65 THEN 'up_left'
      WHEN joystick_x > 0.65 AND joystick_y > 0.65 THEN 'up_right'
      WHEN joystick_x < 0.35 AND joystick_y < 0.35 THEN 'down_left'
      WHEN joystick_x > 0.65 AND joystick_y < 0.35 THEN 'down_right'
      WHEN joystick_x < 0.35 THEN 'left'
      WHEN joystick_x > 0.65 THEN 'right'
      WHEN joystick_y > 0.65 THEN 'up'
      WHEN joystick_y < 0.35 THEN 'down'
      ELSE 'neutral'
    END AS stick_region_0,

    -- Region at frame +1
    CASE
      WHEN ABS(next_joystick_x_1 - 0.5) < 0.2 AND ABS(next_joystick_y_1 - 0.5) < 0.2 THEN 'neutral'
      WHEN next_joystick_x_1 < 0.35 AND next_joystick_y_1 > 0.65 THEN 'up_left'
      WHEN next_joystick_x_1 > 0.65 AND next_joystick_y_1 > 0.65 THEN 'up_right'
      WHEN next_joystick_x_1 < 0.35 AND next_joystick_y_1 < 0.35 THEN 'down_left'
      WHEN next_joystick_x_1 > 0.65 AND next_joystick_y_1 < 0.35 THEN 'down_right'
      WHEN next_joystick_x_1 < 0.35 THEN 'left'
      WHEN next_joystick_x_1 > 0.65 THEN 'right'
      WHEN next_joystick_y_1 > 0.65 THEN 'up'
      WHEN next_joystick_y_1 < 0.35 THEN 'down'
      ELSE 'neutral'
    END AS stick_region_1,

    -- Regions at frames +2 through +8 (for full hitlag tracking)
    CASE
      WHEN ABS(next_joystick_x_2 - 0.5) < 0.2 AND ABS(next_joystick_y_2 - 0.5) < 0.2 THEN 'neutral'
      WHEN next_joystick_x_2 < 0.35 AND next_joystick_y_2 > 0.65 THEN 'up_left'
      WHEN next_joystick_x_2 > 0.65 AND next_joystick_y_2 > 0.65 THEN 'up_right'
      WHEN next_joystick_x_2 < 0.35 AND next_joystick_y_2 < 0.35 THEN 'down_left'
      WHEN next_joystick_x_2 > 0.65 AND next_joystick_y_2 < 0.35 THEN 'down_right'
      WHEN next_joystick_x_2 < 0.35 THEN 'left'
      WHEN next_joystick_x_2 > 0.65 THEN 'right'
      WHEN next_joystick_y_2 > 0.65 THEN 'up'
      WHEN next_joystick_y_2 < 0.35 THEN 'down'
      ELSE 'neutral'
    END AS stick_region_2,

    CASE
      WHEN ABS(next_joystick_x_3 - 0.5) < 0.2 AND ABS(next_joystick_y_3 - 0.5) < 0.2 THEN 'neutral'
      WHEN next_joystick_x_3 < 0.35 AND next_joystick_y_3 > 0.65 THEN 'up_left'
      WHEN next_joystick_x_3 > 0.65 AND next_joystick_y_3 > 0.65 THEN 'up_right'
      WHEN next_joystick_x_3 < 0.35 AND next_joystick_y_3 < 0.35 THEN 'down_left'
      WHEN next_joystick_x_3 > 0.65 AND next_joystick_y_3 < 0.35 THEN 'down_right'
      WHEN next_joystick_x_3 < 0.35 THEN 'left'
      WHEN next_joystick_x_3 > 0.65 THEN 'right'
      WHEN next_joystick_y_3 > 0.65 THEN 'up'
      WHEN next_joystick_y_3 < 0.35 THEN 'down'
      ELSE 'neutral'
    END AS stick_region_3,

    CASE
      WHEN ABS(next_joystick_x_4 - 0.5) < 0.2 AND ABS(next_joystick_y_4 - 0.5) < 0.2 THEN 'neutral'
      WHEN next_joystick_x_4 < 0.35 AND next_joystick_y_4 > 0.65 THEN 'up_left'
      WHEN next_joystick_x_4 > 0.65 AND next_joystick_y_4 > 0.65 THEN 'up_right'
      WHEN next_joystick_x_4 < 0.35 AND next_joystick_y_4 < 0.35 THEN 'down_left'
      WHEN next_joystick_x_4 > 0.65 AND next_joystick_y_4 < 0.35 THEN 'down_right'
      WHEN next_joystick_x_4 < 0.35 THEN 'left'
      WHEN next_joystick_x_4 > 0.65 THEN 'right'
      WHEN next_joystick_y_4 > 0.65 THEN 'up'
      WHEN next_joystick_y_4 < 0.35 THEN 'down'
      ELSE 'neutral'
    END AS stick_region_4,

    CASE
      WHEN ABS(next_joystick_x_5 - 0.5) < 0.2 AND ABS(next_joystick_y_5 - 0.5) < 0.2 THEN 'neutral'
      WHEN next_joystick_x_5 < 0.35 AND next_joystick_y_5 > 0.65 THEN 'up_left'
      WHEN next_joystick_x_5 > 0.65 AND next_joystick_y_5 > 0.65 THEN 'up_right'
      WHEN next_joystick_x_5 < 0.35 AND next_joystick_y_5 < 0.35 THEN 'down_left'
      WHEN next_joystick_x_5 > 0.65 AND next_joystick_y_5 < 0.35 THEN 'down_right'
      WHEN next_joystick_x_5 < 0.35 THEN 'left'
      WHEN next_joystick_x_5 > 0.65 THEN 'right'
      WHEN next_joystick_y_5 > 0.65 THEN 'up'
      WHEN next_joystick_y_5 < 0.35 THEN 'down'
      ELSE 'neutral'
    END AS stick_region_5,

    CASE
      WHEN ABS(next_joystick_x_6 - 0.5) < 0.2 AND ABS(next_joystick_y_6 - 0.5) < 0.2 THEN 'neutral'
      WHEN next_joystick_x_6 < 0.35 AND next_joystick_y_6 > 0.65 THEN 'up_left'
      WHEN next_joystick_x_6 > 0.65 AND next_joystick_y_6 > 0.65 THEN 'up_right'
      WHEN next_joystick_x_6 < 0.35 AND next_joystick_y_6 < 0.35 THEN 'down_left'
      WHEN next_joystick_x_6 > 0.65 AND next_joystick_y_6 < 0.35 THEN 'down_right'
      WHEN next_joystick_x_6 < 0.35 THEN 'left'
      WHEN next_joystick_x_6 > 0.65 THEN 'right'
      WHEN next_joystick_y_6 > 0.65 THEN 'up'
      WHEN next_joystick_y_6 < 0.35 THEN 'down'
      ELSE 'neutral'
    END AS stick_region_6,

    CASE
      WHEN ABS(next_joystick_x_7 - 0.5) < 0.2 AND ABS(next_joystick_y_7 - 0.5) < 0.2 THEN 'neutral'
      WHEN next_joystick_x_7 < 0.35 AND next_joystick_y_7 > 0.65 THEN 'up_left'
      WHEN next_joystick_x_7 > 0.65 AND next_joystick_y_7 > 0.65 THEN 'up_right'
      WHEN next_joystick_x_7 < 0.35 AND next_joystick_y_7 < 0.35 THEN 'down_left'
      WHEN next_joystick_x_7 > 0.65 AND next_joystick_y_7 < 0.35 THEN 'down_right'
      WHEN next_joystick_x_7 < 0.35 THEN 'left'
      WHEN next_joystick_x_7 > 0.65 THEN 'right'
      WHEN next_joystick_y_7 > 0.65 THEN 'up'
      WHEN next_joystick_y_7 < 0.35 THEN 'down'
      ELSE 'neutral'
    END AS stick_region_7,

    CASE
      WHEN ABS(next_joystick_x_8 - 0.5) < 0.2 AND ABS(next_joystick_y_8 - 0.5) < 0.2 THEN 'neutral'
      WHEN next_joystick_x_8 < 0.35 AND next_joystick_y_8 > 0.65 THEN 'up_left'
      WHEN next_joystick_x_8 > 0.65 AND next_joystick_y_8 > 0.65 THEN 'up_right'
      WHEN next_joystick_x_8 < 0.35 AND next_joystick_y_8 < 0.35 THEN 'down_left'
      WHEN next_joystick_x_8 > 0.65 AND next_joystick_y_8 < 0.35 THEN 'down_right'
      WHEN next_joystick_x_8 < 0.35 THEN 'left'
      WHEN next_joystick_x_8 > 0.65 THEN 'right'
      WHEN next_joystick_y_8 > 0.65 THEN 'up'
      WHEN next_joystick_y_8 < 0.35 THEN 'down'
      ELSE 'neutral'
    END AS stick_region_8,

    -- CROUCH CANCEL DETECTION
    -- Down DI at low percent (< 40%)
    CASE
      WHEN percent < 40
        AND joystick_y < 0.35  -- Stick held down
      THEN 1
      ELSE 0
    END AS is_crouch_cancel

  FROM hit_events
  WHERE hit_start = 1  -- Only hit start frames
),

-- Step 4: Calculate comprehensive DI/SDI/ASDI metrics (matching slippistats)
di_metrics AS (
  SELECT
    replay_file,
    player_index,
    frame_number,
    character_id,
    prev_action_state,

    -- Basic hit info
    damage_taken,
    percent AS percent_after,
    percent - damage_taken AS percent_before,
    stocks AS stocks_remaining,
    on_ground AS grounded,

    -- Hitlag duration
    hitlag_frames,

    -- STICK REGIONS DURING HITLAG (concatenated list)
    stick_region_0 || ',' ||
    COALESCE(stick_region_1, '') || ',' ||
    COALESCE(stick_region_2, '') || ',' ||
    COALESCE(stick_region_3, '') || ',' ||
    COALESCE(stick_region_4, '') || ',' ||
    COALESCE(stick_region_5, '') || ',' ||
    COALESCE(stick_region_6, '') || ',' ||
    COALESCE(stick_region_7, '') || ',' ||
    COALESCE(stick_region_8, '') AS stick_regions_during_hitlag,

    -- SDI INPUTS (count of stick region changes during hitlag)
    (CASE WHEN stick_region_1 IS NOT NULL AND stick_region_1 != stick_region_0 AND stick_region_1 != 'neutral' THEN 1 ELSE 0 END +
     CASE WHEN stick_region_2 IS NOT NULL AND stick_region_2 != stick_region_1 AND stick_region_2 != 'neutral' THEN 1 ELSE 0 END +
     CASE WHEN stick_region_3 IS NOT NULL AND stick_region_3 != stick_region_2 AND stick_region_3 != 'neutral' THEN 1 ELSE 0 END +
     CASE WHEN stick_region_4 IS NOT NULL AND stick_region_4 != stick_region_3 AND stick_region_4 != 'neutral' THEN 1 ELSE 0 END +
     CASE WHEN stick_region_5 IS NOT NULL AND stick_region_5 != stick_region_4 AND stick_region_5 != 'neutral' THEN 1 ELSE 0 END +
     CASE WHEN stick_region_6 IS NOT NULL AND stick_region_6 != stick_region_5 AND stick_region_6 != 'neutral' THEN 1 ELSE 0 END +
     CASE WHEN stick_region_7 IS NOT NULL AND stick_region_7 != stick_region_6 AND stick_region_7 != 'neutral' THEN 1 ELSE 0 END +
     CASE WHEN stick_region_8 IS NOT NULL AND stick_region_8 != stick_region_7 AND stick_region_8 != 'neutral' THEN 1 ELSE 0 END
    ) AS sdi_input_count,

    -- ASDI (Automatic SDI - stick position on last frame of hitlag)
    CASE
      WHEN hitlag_frames >= 7 THEN stick_region_7
      WHEN hitlag_frames >= 6 THEN stick_region_6
      WHEN hitlag_frames >= 5 THEN stick_region_5
      WHEN hitlag_frames >= 4 THEN stick_region_4
      WHEN hitlag_frames >= 3 THEN stick_region_3
      WHEN hitlag_frames >= 2 THEN stick_region_2
      WHEN hitlag_frames >= 1 THEN stick_region_1
      ELSE stick_region_0
    END AS asdi,

    -- DI STICK POSITION (average stick during hitlag)
    -- Use first few frames of hitlag for DI calculation
    ROUND((joystick_x +
           COALESCE(next_joystick_x_1, joystick_x) +
           COALESCE(next_joystick_x_2, joystick_x)) / 3.0, 3) AS di_stick_x,
    ROUND((joystick_y +
           COALESCE(next_joystick_y_1, joystick_y) +
           COALESCE(next_joystick_y_2, joystick_y)) / 3.0, 3) AS di_stick_y,

    -- DI ANGLE
    di_angle_degrees,

    -- KNOCKBACK VELOCITIES (from game physics, not position deltas)
    -- Initial velocity (on hit frame)
    ROUND(speed_x_attack, 3) AS kb_velocity_x_initial,
    ROUND(speed_y_attack, 3) AS kb_velocity_y_initial,

    -- Final velocity (after hitlag ends, with DI applied)
    ROUND(CASE
      WHEN hitlag_frames >= 8 THEN COALESCE(kb_vel_x_8, speed_x_attack)
      WHEN hitlag_frames >= 7 THEN COALESCE(kb_vel_x_7, speed_x_attack)
      WHEN hitlag_frames >= 6 THEN COALESCE(kb_vel_x_6, speed_x_attack)
      WHEN hitlag_frames >= 5 THEN COALESCE(kb_vel_x_5, speed_x_attack)
      WHEN hitlag_frames >= 4 THEN COALESCE(kb_vel_x_4, speed_x_attack)
      WHEN hitlag_frames >= 3 THEN COALESCE(kb_vel_x_3, speed_x_attack)
      WHEN hitlag_frames >= 2 THEN COALESCE(kb_vel_x_2, speed_x_attack)
      WHEN hitlag_frames >= 1 THEN COALESCE(kb_vel_x_1, speed_x_attack)
      ELSE speed_x_attack
    END, 3) AS kb_velocity_x_final,
    ROUND(CASE
      WHEN hitlag_frames >= 8 THEN COALESCE(kb_vel_y_8, speed_y_attack)
      WHEN hitlag_frames >= 7 THEN COALESCE(kb_vel_y_7, speed_y_attack)
      WHEN hitlag_frames >= 6 THEN COALESCE(kb_vel_y_6, speed_y_attack)
      WHEN hitlag_frames >= 5 THEN COALESCE(kb_vel_y_5, speed_y_attack)
      WHEN hitlag_frames >= 4 THEN COALESCE(kb_vel_y_4, speed_y_attack)
      WHEN hitlag_frames >= 3 THEN COALESCE(kb_vel_y_3, speed_y_attack)
      WHEN hitlag_frames >= 2 THEN COALESCE(kb_vel_y_2, speed_y_attack)
      WHEN hitlag_frames >= 1 THEN COALESCE(kb_vel_y_1, speed_y_attack)
      ELSE speed_y_attack
    END, 3) AS kb_velocity_y_final,

    -- KB ANGLES (from velocities)
    ROUND(DEGREES(ATAN2(speed_y_attack, speed_x_attack)), 2) AS kb_angle_initial,

    -- Final KB angle (after DI)
    ROUND(DEGREES(ATAN2(
      CASE
        WHEN hitlag_frames >= 8 THEN COALESCE(kb_vel_y_8, speed_y_attack)
        WHEN hitlag_frames >= 7 THEN COALESCE(kb_vel_y_7, speed_y_attack)
        WHEN hitlag_frames >= 6 THEN COALESCE(kb_vel_y_6, speed_y_attack)
        WHEN hitlag_frames >= 5 THEN COALESCE(kb_vel_y_5, speed_y_attack)
        WHEN hitlag_frames >= 4 THEN COALESCE(kb_vel_y_4, speed_y_attack)
        WHEN hitlag_frames >= 3 THEN COALESCE(kb_vel_y_3, speed_y_attack)
        WHEN hitlag_frames >= 2 THEN COALESCE(kb_vel_y_2, speed_y_attack)
        WHEN hitlag_frames >= 1 THEN COALESCE(kb_vel_y_1, speed_y_attack)
        ELSE speed_y_attack
      END,
      CASE
        WHEN hitlag_frames >= 8 THEN COALESCE(kb_vel_x_8, speed_x_attack)
        WHEN hitlag_frames >= 7 THEN COALESCE(kb_vel_x_7, speed_x_attack)
        WHEN hitlag_frames >= 6 THEN COALESCE(kb_vel_x_6, speed_x_attack)
        WHEN hitlag_frames >= 5 THEN COALESCE(kb_vel_x_5, speed_x_attack)
        WHEN hitlag_frames >= 4 THEN COALESCE(kb_vel_x_4, speed_x_attack)
        WHEN hitlag_frames >= 3 THEN COALESCE(kb_vel_x_3, speed_x_attack)
        WHEN hitlag_frames >= 2 THEN COALESCE(kb_vel_x_2, speed_x_attack)
        WHEN hitlag_frames >= 1 THEN COALESCE(kb_vel_x_1, speed_x_attack)
        ELSE speed_x_attack
      END
    )), 2) AS kb_angle_final,

    -- DI EFFICACY (angle change from initial to final)
    ROUND(ABS(
      DEGREES(ATAN2(
        CASE
          WHEN hitlag_frames >= 8 THEN COALESCE(kb_vel_y_8, speed_y_attack)
          WHEN hitlag_frames >= 7 THEN COALESCE(kb_vel_y_7, speed_y_attack)
          WHEN hitlag_frames >= 6 THEN COALESCE(kb_vel_y_6, speed_y_attack)
          WHEN hitlag_frames >= 5 THEN COALESCE(kb_vel_y_5, speed_y_attack)
          WHEN hitlag_frames >= 4 THEN COALESCE(kb_vel_y_4, speed_y_attack)
          WHEN hitlag_frames >= 3 THEN COALESCE(kb_vel_y_3, speed_y_attack)
          WHEN hitlag_frames >= 2 THEN COALESCE(kb_vel_y_2, speed_y_attack)
          WHEN hitlag_frames >= 1 THEN COALESCE(kb_vel_y_1, speed_y_attack)
          ELSE speed_y_attack
        END,
        CASE
          WHEN hitlag_frames >= 8 THEN COALESCE(kb_vel_x_8, speed_x_attack)
          WHEN hitlag_frames >= 7 THEN COALESCE(kb_vel_x_7, speed_x_attack)
          WHEN hitlag_frames >= 6 THEN COALESCE(kb_vel_x_6, speed_x_attack)
          WHEN hitlag_frames >= 5 THEN COALESCE(kb_vel_x_5, speed_x_attack)
          WHEN hitlag_frames >= 4 THEN COALESCE(kb_vel_x_4, speed_x_attack)
          WHEN hitlag_frames >= 3 THEN COALESCE(kb_vel_x_3, speed_x_attack)
          WHEN hitlag_frames >= 2 THEN COALESCE(kb_vel_x_2, speed_x_attack)
          WHEN hitlag_frames >= 1 THEN COALESCE(kb_vel_x_1, speed_x_attack)
          ELSE speed_y_attack
        END
      )) - DEGREES(ATAN2(speed_y_attack, speed_x_attack))
    ), 2) AS di_efficacy,

    -- POSITIONS
    ROUND(position_x, 2) AS start_pos_x,
    ROUND(position_y, 2) AS start_pos_y,

    -- End position - for now same as start (will fix with LEAD positions later)
    ROUND(position_x, 2) AS end_pos_x,
    ROUND(position_y, 2) AS end_pos_y,

    -- DISTANCE - placeholder for now
    ROUND(0.0, 2) AS distance,

    -- CROUCH CANCEL
    is_crouch_cancel

  FROM hitlag_analysis
)

-- Final output (matching slippistats TakeHitData schema)
SELECT
  replay_file,
  player_index,
  frame_number AS hit_frame,

  -- Game state
  stocks_remaining,
  prev_action_state AS state_before_hit,
  CAST(grounded AS BOOLEAN) AS grounded,
  CAST(is_crouch_cancel AS BOOLEAN) AS crouch_cancel,

  -- Damage and percent
  ROUND(damage_taken, 2) AS damage,
  ROUND(percent_before, 2) AS percent_before,
  ROUND(percent_after, 2) AS percent_after,

  -- Hitlag
  hitlag_frames,
  stick_regions_during_hitlag,

  -- SDI
  sdi_input_count AS sdi_inputs,

  -- ASDI
  asdi,

  -- DI stick position
  di_stick_x,
  di_stick_y,

  -- DI angle
  di_angle_degrees AS di_angle,

  -- Knockback angles
  kb_angle_initial AS kb_angle,
  kb_angle_final AS final_kb_angle,

  -- Knockback velocities
  kb_velocity_x_initial AS kb_velocity_x,
  kb_velocity_y_initial AS kb_velocity_y,
  kb_velocity_x_final AS final_kb_velocity_x,
  kb_velocity_y_final AS final_kb_velocity_y,

  -- DI efficacy (angle change)
  di_efficacy AS di_efficacy_degrees,

  -- Positions
  start_pos_x,
  start_pos_y,
  end_pos_x,
  end_pos_y,

  -- Distance moved during hit
  distance,

  -- Metadata
  character_id

FROM di_metrics
ORDER BY replay_file, player_index, frame_number;
