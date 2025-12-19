-- ============================================================
-- INPUT COUNTING QUERY
-- ============================================================
-- Counts input changes (button presses, stick movements, trigger presses)
--
-- Output Schema:
--   replay_file, player_index, button_a/b/x/y/z/l/r_count, total_button_inputs,
--   joystick_inputs, cstick_inputs, trigger_l/r_count, total_trigger_inputs,
--   total_inputs, total_frames, duration_seconds, duration_minutes,
--   actions_per_minute, digital_inputs_per_minute
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
-- Step 1: Get previous frame inputs for comparison
input_frames AS (
  SELECT
    replay_file,
    player_index,
    frame_number,
    character_id,

    -- Current frame inputs
    button_a, button_b, button_x, button_y, button_z, button_l, button_r,
    joystick_x, joystick_y,
    cstick_x, cstick_y,
    trigger_l, trigger_r,

    -- Previous frame inputs
    LAG(button_a, 1, FALSE) OVER w AS prev_button_a,
    LAG(button_b, 1, FALSE) OVER w AS prev_button_b,
    LAG(button_x, 1, FALSE) OVER w AS prev_button_x,
    LAG(button_y, 1, FALSE) OVER w AS prev_button_y,
    LAG(button_z, 1, FALSE) OVER w AS prev_button_z,
    LAG(button_l, 1, FALSE) OVER w AS prev_button_l,
    LAG(button_r, 1, FALSE) OVER w AS prev_button_r,
    LAG(joystick_x, 1, 0.5) OVER w AS prev_joystick_x,
    LAG(joystick_y, 1, 0.5) OVER w AS prev_joystick_y,
    LAG(cstick_x, 1, 0.5) OVER w AS prev_cstick_x,
    LAG(cstick_y, 1, 0.5) OVER w AS prev_cstick_y,
    LAG(trigger_l, 1, 0.0) OVER w AS prev_trigger_l,
    LAG(trigger_r, 1, 0.0) OVER w AS prev_trigger_r

  FROM read_parquet('replay_parquet_test/partition_*.parquet')
  WINDOW w AS (PARTITION BY replay_file, player_index ORDER BY frame_number)
  -- Add character filter here if needed:
  -- WHERE character_id = 1  -- Uncomment to filter by character
),

-- Step 2: Helper function to determine stick region
stick_regions AS (
  SELECT
    *,

    -- Joystick region (9 regions: 8 directions + neutral)
    -- Thresholds: <0.35 = low, 0.35-0.65 = neutral, >0.65 = high
    CASE
      WHEN joystick_x BETWEEN 0.35 AND 0.65 AND joystick_y BETWEEN 0.35 AND 0.65 THEN 'neutral'
      WHEN joystick_x BETWEEN 0.35 AND 0.65 AND joystick_y > 0.65 THEN 'up'
      WHEN joystick_x BETWEEN 0.35 AND 0.65 AND joystick_y < 0.35 THEN 'down'
      WHEN joystick_x < 0.35 AND joystick_y BETWEEN 0.35 AND 0.65 THEN 'left'
      WHEN joystick_x > 0.65 AND joystick_y BETWEEN 0.35 AND 0.65 THEN 'right'
      WHEN joystick_x < 0.35 AND joystick_y > 0.65 THEN 'up_left'
      WHEN joystick_x > 0.65 AND joystick_y > 0.65 THEN 'up_right'
      WHEN joystick_x < 0.35 AND joystick_y < 0.35 THEN 'down_left'
      WHEN joystick_x > 0.65 AND joystick_y < 0.35 THEN 'down_right'
      ELSE 'neutral'
    END AS joystick_region,

    CASE
      WHEN prev_joystick_x BETWEEN 0.35 AND 0.65 AND prev_joystick_y BETWEEN 0.35 AND 0.65 THEN 'neutral'
      WHEN prev_joystick_x BETWEEN 0.35 AND 0.65 AND prev_joystick_y > 0.65 THEN 'up'
      WHEN prev_joystick_x BETWEEN 0.35 AND 0.65 AND prev_joystick_y < 0.35 THEN 'down'
      WHEN prev_joystick_x < 0.35 AND prev_joystick_y BETWEEN 0.35 AND 0.65 THEN 'left'
      WHEN prev_joystick_x > 0.65 AND prev_joystick_y BETWEEN 0.35 AND 0.65 THEN 'right'
      WHEN prev_joystick_x < 0.35 AND prev_joystick_y > 0.65 THEN 'up_left'
      WHEN prev_joystick_x > 0.65 AND prev_joystick_y > 0.65 THEN 'up_right'
      WHEN prev_joystick_x < 0.35 AND prev_joystick_y < 0.35 THEN 'down_left'
      WHEN prev_joystick_x > 0.65 AND prev_joystick_y < 0.35 THEN 'down_right'
      ELSE 'neutral'
    END AS prev_joystick_region,

    -- C-stick region
    CASE
      WHEN cstick_x BETWEEN 0.35 AND 0.65 AND cstick_y BETWEEN 0.35 AND 0.65 THEN 'neutral'
      WHEN cstick_x BETWEEN 0.35 AND 0.65 AND cstick_y > 0.65 THEN 'up'
      WHEN cstick_x BETWEEN 0.35 AND 0.65 AND cstick_y < 0.35 THEN 'down'
      WHEN cstick_x < 0.35 AND cstick_y BETWEEN 0.35 AND 0.65 THEN 'left'
      WHEN cstick_x > 0.65 AND cstick_y BETWEEN 0.35 AND 0.65 THEN 'right'
      WHEN cstick_x < 0.35 AND cstick_y > 0.65 THEN 'up_left'
      WHEN cstick_x > 0.65 AND cstick_y > 0.65 THEN 'up_right'
      WHEN cstick_x < 0.35 AND cstick_y < 0.35 THEN 'down_left'
      WHEN cstick_x > 0.65 AND cstick_y < 0.35 THEN 'down_right'
      ELSE 'neutral'
    END AS cstick_region,

    CASE
      WHEN prev_cstick_x BETWEEN 0.35 AND 0.65 AND prev_cstick_y BETWEEN 0.35 AND 0.65 THEN 'neutral'
      WHEN prev_cstick_x BETWEEN 0.35 AND 0.65 AND prev_cstick_y > 0.65 THEN 'up'
      WHEN prev_cstick_x BETWEEN 0.35 AND 0.65 AND prev_cstick_y < 0.35 THEN 'down'
      WHEN prev_cstick_x < 0.35 AND prev_cstick_y BETWEEN 0.35 AND 0.65 THEN 'left'
      WHEN prev_cstick_x > 0.65 AND prev_cstick_y BETWEEN 0.35 AND 0.65 THEN 'right'
      WHEN prev_cstick_x < 0.35 AND prev_cstick_y > 0.65 THEN 'up_left'
      WHEN prev_cstick_x > 0.65 AND prev_cstick_y > 0.65 THEN 'up_right'
      WHEN prev_cstick_x < 0.35 AND prev_cstick_y < 0.35 THEN 'down_left'
      WHEN prev_cstick_x > 0.65 AND prev_cstick_y < 0.35 THEN 'down_right'
      ELSE 'neutral'
    END AS prev_cstick_region

  FROM input_frames
),

-- Step 3: Detect input changes
input_changes AS (
  SELECT
    replay_file,
    player_index,
    frame_number,

    -- Button presses (false → true)
    CASE WHEN button_a AND NOT prev_button_a THEN 1 ELSE 0 END AS button_a_press,
    CASE WHEN button_b AND NOT prev_button_b THEN 1 ELSE 0 END AS button_b_press,
    CASE WHEN button_x AND NOT prev_button_x THEN 1 ELSE 0 END AS button_x_press,
    CASE WHEN button_y AND NOT prev_button_y THEN 1 ELSE 0 END AS button_y_press,
    CASE WHEN button_z AND NOT prev_button_z THEN 1 ELSE 0 END AS button_z_press,
    CASE WHEN button_l AND NOT prev_button_l THEN 1 ELSE 0 END AS button_l_press,
    CASE WHEN button_r AND NOT prev_button_r THEN 1 ELSE 0 END AS button_r_press,

    -- Joystick region change (excluding neutral region entries from neutral)
    CASE WHEN
      joystick_region != prev_joystick_region
      AND joystick_region != 'neutral'
    THEN 1 ELSE 0 END AS joystick_input,

    -- C-stick region change (excluding neutral)
    CASE WHEN
      cstick_region != prev_cstick_region
      AND cstick_region != 'neutral'
    THEN 1 ELSE 0 END AS cstick_input,

    -- Trigger presses (crossing 0.3 threshold)
    CASE WHEN trigger_l >= 0.3 AND prev_trigger_l < 0.3 THEN 1 ELSE 0 END AS trigger_l_press,
    CASE WHEN trigger_r >= 0.3 AND prev_trigger_r < 0.3 THEN 1 ELSE 0 END AS trigger_r_press

  FROM stick_regions
),

-- Step 4: Aggregate per player per replay
player_input_stats AS (
  SELECT
    replay_file,
    player_index,

    -- Button counts
    SUM(button_a_press) AS button_a_count,
    SUM(button_b_press) AS button_b_count,
    SUM(button_x_press) AS button_x_count,
    SUM(button_y_press) AS button_y_count,
    SUM(button_z_press) AS button_z_count,
    SUM(button_l_press) AS button_l_count,
    SUM(button_r_press) AS button_r_count,
    SUM(button_a_press + button_b_press + button_x_press + button_y_press +
        button_z_press + button_l_press + button_r_press) AS total_button_inputs,

    -- Stick inputs
    SUM(joystick_input) AS joystick_inputs,
    SUM(cstick_input) AS cstick_inputs,

    -- Trigger inputs
    SUM(trigger_l_press) AS trigger_l_count,
    SUM(trigger_r_press) AS trigger_r_count,
    SUM(trigger_l_press + trigger_r_press) AS total_trigger_inputs,

    -- Total inputs
    SUM(button_a_press + button_b_press + button_x_press + button_y_press +
        button_z_press + button_l_press + button_r_press +
        joystick_input + cstick_input +
        trigger_l_press + trigger_r_press) AS total_inputs,

    -- Game duration
    COUNT(*) AS total_frames,
    ROUND(COUNT(*) / 60.0, 2) AS duration_seconds,
    ROUND(COUNT(*) / 3600.0, 2) AS duration_minutes,

    -- APM calculations
    ROUND(SUM(button_a_press + button_b_press + button_x_press + button_y_press +
        button_z_press + button_l_press + button_r_press +
        joystick_input + cstick_input +
        trigger_l_press + trigger_r_press) * 60.0 / NULLIF(COUNT(*), 0), 2) AS actions_per_minute,

    ROUND(SUM(button_a_press + button_b_press + button_x_press + button_y_press +
        button_z_press + button_l_press + button_r_press) * 60.0 / NULLIF(COUNT(*), 0), 2) AS digital_inputs_per_minute

  FROM input_changes
  GROUP BY replay_file, player_index
)

-- Final output
SELECT * FROM player_input_stats
ORDER BY replay_file, player_index;
