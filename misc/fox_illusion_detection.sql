-- Fox Side-B (Illusion) Detection Query for DuckDB
-- Detects Fox illusion usage and categorizes by characteristics

WITH illusion_frames AS (
  SELECT 
    replay_file,
    player_index,
    frame_number,
    action_state,
    on_ground,
    button_b,
    position_x,
    position_y,
    LAG(action_state) OVER w AS prev_state,
    LAG(button_b) OVER w AS prev_button_b,
    -- Mark start of illusion (entering state 347 for grounded, 350 for aerial)
    CASE 
      WHEN action_state IN (347, 350) 
       AND COALESCE(prev_state, 0) NOT IN (347, 348, 349, 350, 351, 352)
      THEN 1 ELSE 0 
    END AS illusion_start
  FROM read_parquet('fox_vs_fox_parquet/partition_*.parquet')
  WHERE character_id = 1  -- Fox only
  WINDOW w AS (PARTITION BY replay_file, player_index ORDER BY frame_number)
),
grouped_illusions AS (
  SELECT 
    *,
    -- Group consecutive illusion frames together
    SUM(illusion_start) OVER (
      PARTITION BY replay_file, player_index 
      ORDER BY frame_number
    ) AS illusion_id
  FROM illusion_frames
  WHERE action_state BETWEEN 347 AND 352  -- Illusion states only
),
with_frame_offsets AS (
  SELECT 
    *,
    -- Frames since start of this illusion
    frame_number - MIN(frame_number) OVER (
      PARTITION BY replay_file, player_index, illusion_id
    ) AS frames_from_start,
    -- Detect B press events (false → true transition)
    CASE WHEN button_b AND NOT COALESCE(prev_button_b, FALSE) 
      THEN 1 ELSE 0 
    END AS b_press_event
  FROM grouped_illusions
),
illusion_summary AS (
  SELECT 
    replay_file,
    player_index,
    illusion_id,
    MIN(frame_number) AS start_frame,
    MAX(frame_number) AS end_frame,
    COUNT(*) AS total_duration,
    
    -- Categorize as grounded or aerial (based on starting state)
    CASE 
      WHEN MIN(CASE WHEN frames_from_start = 0 THEN action_state END) = 347 
      THEN 'grounded' 
      ELSE 'aerial' 
    END AS illusion_type,
    
    -- Duration in each state
    SUM(CASE WHEN action_state = 347 THEN 1 ELSE 0 END) AS frames_state_347,
    SUM(CASE WHEN action_state = 348 THEN 1 ELSE 0 END) AS frames_state_348,
    SUM(CASE WHEN action_state = 349 THEN 1 ELSE 0 END) AS frames_state_349,
    SUM(CASE WHEN action_state = 350 THEN 1 ELSE 0 END) AS frames_state_350,
    SUM(CASE WHEN action_state = 351 THEN 1 ELSE 0 END) AS frames_state_351,
    SUM(CASE WHEN action_state = 352 THEN 1 ELSE 0 END) AS frames_state_352,
    
    -- Find B press timing (between frames 15-30 after start)
    MIN(CASE 
      WHEN b_press_event = 1 AND frames_from_start BETWEEN 15 AND 30 
      THEN frames_from_start 
      ELSE NULL 
    END) AS b_press_frame_offset,
    
    -- Starting and ending positions (for distance calculation)
    MIN(CASE WHEN frames_from_start = 0 THEN position_x END) AS start_x,
    MAX(position_x) - MIN(CASE WHEN frames_from_start = 0 THEN position_x END) AS distance_traveled
    
  FROM with_frame_offsets
  GROUP BY replay_file, player_index, illusion_id
  HAVING COUNT(*) >= 15  -- Filter out incomplete/interrupted illusions
)

-- Final output with illusion characteristics
SELECT 
  replay_file,
  player_index,
  illusion_id,
  start_frame,
  end_frame,
  illusion_type,
  total_duration,
  b_press_frame_offset,
  distance_traveled,
  frames_state_347 + frames_state_350 AS startup_frames,
  frames_state_348 + frames_state_351 AS short_travel_frames, 
  frames_state_349 + frames_state_352 AS long_travel_frames
FROM illusion_summary
ORDER BY replay_file, player_index, start_frame;
