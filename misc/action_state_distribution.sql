-- Action State Distribution Query
-- Counts occurrences of each action state across all replays

SELECT 
  action_state,
  COUNT(*) as frame_count,
  COUNT(DISTINCT replay_file) as replay_count,
  COUNT(DISTINCT replay_file || '_' || player_index) as player_count,
  ROUND(AVG(CASE WHEN on_ground THEN 1.0 ELSE 0.0 END) * 100, 2) as pct_grounded,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 4) as pct_of_total
FROM read_parquet('fox_vs_fox_parquet/partition_*.parquet')
WHERE character_id = 1  -- Fox only
GROUP BY action_state
ORDER BY frame_count DESC;
