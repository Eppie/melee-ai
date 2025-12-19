# Action State Analysis Results

Generated from Fox vs Fox replays dataset.

## Files

1. **action_state_distribution.sql** - Query to get distribution of action states
2. **action_state_distribution.csv** - Results: 183 unique action states
3. **action_state_transitions.sql** - Query to get state transition counts
4. **action_state_transitions.csv** - Results: 4,768 unique transitions

## Summary Statistics

### Distribution
- **Total frames analyzed:** ~138 million
- **Unique action states:** 183
- **Most common state:** State 20 (8.45% of all frames) - appears to be "Dashing"
- **Top 5 states:** 20, 25, 90, 27, 67

### Transitions
- **Total transitions:** 14,645,596
- **Unique transition pairs:** 4,768
- **Most common transition:** 18→20 (8.15% of all transitions) - likely "Run→Dash"
- **Top 5 transitions:** 18→20, 24→25, 20→18, 20→24, 14→18

## Usage

To re-run the analysis:

```bash
# Action state distribution
duckdb -c "$(cat action_state_distribution.sql)" > action_state_distribution_new.csv

# Or with Python
python -c "
import duckdb
con = duckdb.connect()
result = con.execute(open('action_state_distribution.sql').read()).fetchdf()
result.to_csv('action_state_distribution_new.csv', index=False)
"

# Action state transitions
python -c "
import duckdb
con = duckdb.connect()
result = con.execute(open('action_state_transitions.sql').read()).fetchdf()
result.to_csv('action_state_transitions_new.csv', index=False)
"
```

## Notes

- Both queries filter for Fox only (character_id = 1)
- Transitions exclude first frame of each player (no prev_action_state)
- Transitions only count actual state changes (prev != current)
- Distribution includes percentage of frames grounded for each state
- All queries use the full fox_vs_fox_parquet dataset
