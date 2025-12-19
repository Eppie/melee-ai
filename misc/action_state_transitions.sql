-- Action State Transitions Query
-- Counts transitions from one action state to another

WITH state_with_prev AS (
  SELECT 
    replay_file,
    player_index,
    frame_number,
    action_state,
    LAG(action_state) OVER (
      PARTITION BY replay_file, player_index 
      ORDER BY frame_number
    ) AS prev_action_state
  FROM read_parquet('fox_vs_fox_parquet/partition_*.parquet')
  WHERE character_id = 1  -- Fox only
)
SELECT 
  prev_action_state AS from_state,
  action_state AS to_state,
  COUNT(*) as transition_count,
  COUNT(DISTINCT replay_file) as replay_count,
  COUNT(DISTINCT replay_file || '_' || player_index) as player_count,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 4) as pct_of_all_transitions
FROM state_with_prev
WHERE prev_action_state IS NOT NULL  -- Exclude first frame of each player
  AND prev_action_state != action_state  -- Only actual transitions (state changes)
GROUP BY prev_action_state, action_state
ORDER BY transition_count DESC;
