#!/bin/bash

# ============================================================
# Example Usage Script for Slippi SQL Statistics Queries
# ============================================================
# This script demonstrates various ways to run the SQL statistics queries
# using DuckDB CLI.
#
# Prerequisites:
# - DuckDB installed (brew install duckdb on macOS)
# - Parquet data in replay_parquet_test/partition_*.parquet
#
# Usage:
#   chmod +x sql/helpers/example_usage.sh
#   ./sql/helpers/example_usage.sh
# ============================================================

# Color output for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Slippi SQL Statistics Examples ===${NC}\n"

# ============================================================
# SECTION 1: Basic Query Execution
# ============================================================
echo -e "${GREEN}1. Running Basic Queries${NC}"
echo "------------------------------------------------------------"

echo "Running combos analysis..."
duckdb -c ".mode box" < sql/stats_combos.sql | head -20

echo -e "\nRunning input statistics..."
duckdb -c ".mode box" < sql/stats_inputs.sql | head -10

# ============================================================
# SECTION 2: Exporting to Different Formats
# ============================================================
echo -e "\n${GREEN}2. Exporting Results to Different Formats${NC}"
echo "------------------------------------------------------------"

echo "Exporting combos to CSV..."
duckdb -csv -header < sql/stats_combos.sql > output/combos.csv
echo "  → Saved to output/combos.csv"

echo "Exporting actions to JSON..."
duckdb -json < sql/stats_actions.sql > output/actions.json
echo "  → Saved to output/actions.json"

echo "Exporting overall stats to Parquet..."
duckdb -c "COPY (SELECT * FROM 'sql/stats_overall.sql') TO 'output/overall.parquet' (FORMAT PARQUET);"
echo "  → Saved to output/overall.parquet"

# ============================================================
# SECTION 3: Filtered Queries
# ============================================================
echo -e "\n${GREEN}3. Running Filtered Queries${NC}"
echo "------------------------------------------------------------"

echo "Fox players only (character_id = 1)..."
duckdb << EOF
SELECT
  replay_file,
  player_index,
  wavedash_count,
  lcancel_success_rate,
  grab_success_rate
FROM (
  SELECT *
  FROM read_parquet('replay_parquet_test/partition_*.parquet')
  WHERE character_id = 1
) base
-- Run stats_actions query logic here (simplified for demo)
LIMIT 5;
EOF

echo -e "\nSpecific replay file..."
duckdb << EOF
SELECT
  player_index,
  SUM(CASE WHEN action_state BETWEEN 65 AND 69 THEN 1 ELSE 0 END) AS aerial_count
FROM read_parquet('replay_parquet_test/partition_*.parquet')
WHERE replay_file = 'master-diamond-000543d6034bf69893c60a05.slp'
GROUP BY player_index;
EOF

# ============================================================
# SECTION 4: Combined Analysis
# ============================================================
echo -e "\n${GREEN}4. Combining Multiple Queries${NC}"
echo "------------------------------------------------------------"

echo "Creating comprehensive player report..."
duckdb -box << 'EOF'
-- Combine stats from multiple queries
WITH
  combos AS (
    -- Simplified combo count
    SELECT
      replay_file,
      player_index,
      COUNT(*) AS combo_count
    FROM (
      SELECT
        replay_file,
        player_index,
        frame_number,
        SUM(CASE WHEN action_state BETWEEN 75 AND 92 THEN 1 ELSE 0 END) OVER (
          PARTITION BY replay_file, player_index
          ORDER BY frame_number
        ) AS combo_id
      FROM read_parquet('replay_parquet_test/partition_*.parquet')
    )
    GROUP BY replay_file, player_index
  ),

  overall AS (
    SELECT
      replay_file,
      player_index,
      COUNT(*) / 60.0 AS duration_seconds,
      SUM(CASE WHEN action_state BETWEEN 65 AND 69 THEN 1 ELSE 0 END) AS aerial_count
    FROM read_parquet('replay_parquet_test/partition_*.parquet')
    GROUP BY replay_file, player_index
  )

SELECT
  o.replay_file,
  o.player_index,
  ROUND(o.duration_seconds, 2) AS game_duration_sec,
  o.aerial_count,
  COALESCE(c.combo_count, 0) AS combo_count,
  ROUND(o.aerial_count / o.duration_seconds, 2) AS aerials_per_second
FROM overall o
LEFT JOIN combos c USING (replay_file, player_index)
LIMIT 10;
EOF

# ============================================================
# SECTION 5: Character-Specific Analysis
# ============================================================
echo -e "\n${GREEN}5. Character-Specific Statistics${NC}"
echo "------------------------------------------------------------"

echo "Comparing top tiers (Fox, Falco, Marth, Sheik)..."
duckdb -box << EOF
SELECT
  CASE character_id
    WHEN 1 THEN 'Fox'
    WHEN 22 THEN 'Falco'
    WHEN 18 THEN 'Marth'
    WHEN 7 THEN 'Sheik'
  END AS character,
  COUNT(DISTINCT replay_file || '_' || player_index) AS player_instances,
  ROUND(AVG(CASE WHEN action_state BETWEEN 65 AND 69 THEN 1 ELSE 0 END), 4) AS avg_aerial_usage_rate
FROM read_parquet('replay_parquet_test/partition_*.parquet')
WHERE character_id IN (1, 22, 18, 7)
GROUP BY character_id
ORDER BY player_instances DESC;
EOF

# ============================================================
# SECTION 6: Advanced Metrics
# ============================================================
echo -e "\n${GREEN}6. Computing Advanced Metrics${NC}"
echo "------------------------------------------------------------"

echo "L-cancel success rate by character..."
duckdb -box << EOF
SELECT
  CASE character_id
    WHEN 1 THEN 'Fox'
    WHEN 22 THEN 'Falco'
    WHEN 2 THEN 'Falcon'
    WHEN 18 THEN 'Marth'
    ELSE 'Other'
  END AS character,
  COUNT(*) AS total_landings,
  SUM(CASE WHEN lcancel_status = 1 THEN 1 ELSE 0 END) AS successful_lcancels,
  ROUND(100.0 * SUM(CASE WHEN lcancel_status = 1 THEN 1 ELSE 0 END) /
    NULLIF(COUNT(*), 0), 2) AS lcancel_success_rate
FROM read_parquet('replay_parquet_test/partition_*.parquet')
WHERE action_state BETWEEN 70 AND 74  -- Aerial landing states
  AND lcancel_status IN (1, 2)  -- Only count when L-cancel was possible
  AND character_id IN (1, 22, 2, 18)
GROUP BY character_id
ORDER BY lcancel_success_rate DESC;
EOF

# ============================================================
# SECTION 7: Time-Series Analysis
# ============================================================
echo -e "\n${GREEN}7. Time-Series Analysis (Damage Over Time)${NC}"
echo "------------------------------------------------------------"

echo "Damage accumulation per 10-second window..."
duckdb -box << EOF
WITH damage_windows AS (
  SELECT
    replay_file,
    player_index,
    FLOOR(frame_number / 600) * 10 AS time_window_seconds,  -- 600 frames = 10 sec
    MAX(percent) AS max_percent_in_window
  FROM read_parquet('replay_parquet_test/partition_*.parquet')
  GROUP BY replay_file, player_index, time_window_seconds
)
SELECT
  replay_file,
  player_index,
  time_window_seconds,
  max_percent_in_window,
  max_percent_in_window - LAG(max_percent_in_window, 1, 0) OVER (
    PARTITION BY replay_file, player_index
    ORDER BY time_window_seconds
  ) AS damage_gain_in_window
FROM damage_windows
ORDER BY replay_file, player_index, time_window_seconds
LIMIT 20;
EOF

# ============================================================
# SECTION 8: Matchup Analysis
# ============================================================
echo -e "\n${GREEN}8. Matchup Analysis${NC}"
echo "------------------------------------------------------------"

echo "Fox vs Marth games analysis..."
duckdb -box << EOF
WITH matchups AS (
  SELECT DISTINCT
    p1.replay_file,
    p1.player_index AS fox_player,
    p2.player_index AS marth_player
  FROM read_parquet('replay_parquet_test/partition_*.parquet') p1
  INNER JOIN read_parquet('replay_parquet_test/partition_*.parquet') p2
    ON p1.replay_file = p2.replay_file
    AND p1.frame_number = p2.frame_number
    AND p1.player_index != p2.player_index
  WHERE p1.character_id = 1  -- Fox
    AND p2.character_id = 18  -- Marth
)
SELECT
  COUNT(*) AS total_fox_vs_marth_games,
  COUNT(DISTINCT replay_file) AS unique_replays
FROM matchups;
EOF

# ============================================================
# SECTION 9: Performance Benchmarking
# ============================================================
echo -e "\n${GREEN}9. Query Performance Benchmarking${NC}"
echo "------------------------------------------------------------"

echo "Timing each query..."
for query in combos conversions actions inputs stocks overall; do
  echo -n "  stats_${query}.sql: "
  time (duckdb < sql/stats_${query}.sql > /dev/null 2>&1)
done

# ============================================================
# SECTION 10: Data Quality Checks
# ============================================================
echo -e "\n${GREEN}10. Data Quality Validation${NC}"
echo "------------------------------------------------------------"

echo "Checking for missing or invalid data..."
duckdb -box << EOF
SELECT
  'Total rows' AS check_type,
  COUNT(*) AS count
FROM read_parquet('replay_parquet_test/partition_*.parquet')

UNION ALL

SELECT
  'Unique replays',
  COUNT(DISTINCT replay_file)
FROM read_parquet('replay_parquet_test/partition_*.parquet')

UNION ALL

SELECT
  'Unique players',
  COUNT(DISTINCT replay_file || '_' || player_index)
FROM read_parquet('replay_parquet_test/partition_*.parquet')

UNION ALL

SELECT
  'Invalid character IDs (> 32)',
  COUNT(*)
FROM read_parquet('replay_parquet_test/partition_*.parquet')
WHERE character_id > 32 AND character_id != 255;
EOF

# ============================================================
# Cleanup
# ============================================================
echo -e "\n${YELLOW}Note: Output files saved to output/ directory${NC}"
echo -e "${BLUE}=== Examples Complete ===${NC}\n"
