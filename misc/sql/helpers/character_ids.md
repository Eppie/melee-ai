# Character ID Reference

This document maps character IDs to character names in Super Smash Bros. Melee.

## Usage in Queries

Filter by character using the `character_id` column:

```sql
-- Filter for Fox only
WHERE character_id = 1

-- Filter for spacies (Fox and Falco)
WHERE character_id IN (1, 22)

-- Filter for top tiers
WHERE character_id IN (1, 22, 9, 18, 2, 7, 16)  -- Fox, Falco, Peach, Marth, Falcon, Sheik, Mewtwo
```

## Complete Character List

| ID | Character Name | Common Abbreviation | Tier (Melee) |
|----|----------------|---------------------|--------------|
| 0  | Mario          | Mario               | Mid          |
| 1  | Fox            | Fox                 | S            |
| 2  | Captain Falcon | Falcon, CF          | S            |
| 3  | Donkey Kong    | DK                  | Low          |
| 4  | Kirby          | Kirby               | Low          |
| 5  | Bowser         | Bowser              | Low          |
| 6  | Link           | Link                | Low          |
| 7  | Sheik          | Sheik               | S            |
| 8  | Ness           | Ness                | Low          |
| 9  | Peach          | Peach               | S            |
| 10 | Popo           | Popo, ICs           | High         |
| 11 | Nana           | Nana (Partner)      | -            |
| 12 | Pikachu        | Pika                | High         |
| 13 | Samus          | Samus               | Mid          |
| 14 | Yoshi          | Yoshi               | Mid          |
| 15 | Jigglypuff     | Puff, Jiggs         | Mid          |
| 16 | Mewtwo         | M2                  | Low          |
| 17 | Luigi          | Luigi, Weegee       | Mid          |
| 18 | Marth          | Marth               | S            |
| 19 | Zelda          | Zelda               | Low          |
| 20 | Young Link     | YL, Young Link      | Mid          |
| 21 | Dr. Mario      | Doc                 | Mid          |
| 22 | Falco          | Falco               | S            |
| 23 | Pichu          | Pichu               | Low          |
| 24 | Game & Watch   | G&W, GnW            | Low          |
| 25 | Ganondorf      | Ganon               | Mid          |
| 26 | Roy            | Roy                 | Low          |
| 29 | Male Wireframe | -                   | Special      |
| 30 | Female Wireframe | -                 | Special      |
| 31 | Giga Bowser    | -                   | Boss         |
| 32 | Sandbag        | -                   | Item         |
| 255| Unknown        | -                   | -            |

## Character Groups

### Top Tier (S Tier)
```sql
WHERE character_id IN (1, 22, 7, 18, 2, 9)  -- Fox, Falco, Sheik, Marth, Falcon, Peach
```

### Spacies (Fox and Falco)
```sql
WHERE character_id IN (1, 22)
```

### Floaties (Light, floaty characters)
```sql
WHERE character_id IN (9, 15, 12)  -- Peach, Puff, Pikachu
```

### Fastfallers (Heavy, fast-falling characters)
```sql
WHERE character_id IN (1, 22, 2)  -- Fox, Falco, Falcon
```

### Ice Climbers Note
- **ID 10 (Popo)**: Main Ice Climbers character (controlled by player)
- **ID 11 (Nana)**: Partner character (CPU-controlled)

**Important:** The parquet extraction script already filters out Nana (ID 11) to avoid duplicate data. You should only see Popo (ID 10) in the dataset.

## Example Queries

### Most Played Characters
```sql
SELECT
  character_id,
  COUNT(DISTINCT replay_file || '_' || player_index) AS player_instances,
  COUNT(DISTINCT replay_file) AS games_played
FROM read_parquet('replay_parquet_test/partition_*.parquet')
GROUP BY character_id
ORDER BY player_instances DESC;
```

### Character-Specific Wavedash Stats
```sql
-- Compare wavedash usage between characters
SELECT
  CASE character_id
    WHEN 1 THEN 'Fox'
    WHEN 22 THEN 'Falco'
    WHEN 2 THEN 'Captain Falcon'
    WHEN 18 THEN 'Marth'
    ELSE 'Other'
  END AS character_name,
  AVG(wavedash_count) AS avg_wavedashes_per_game,
  AVG(waveland_count) AS avg_wavelands_per_game
FROM (SELECT * FROM 'sql/stats_actions.sql')
WHERE character_id IN (1, 22, 2, 18)
GROUP BY character_name
ORDER BY avg_wavedashes_per_game DESC;
```

### Character Matchup Analysis
```sql
-- Analyze Fox vs Marth matchups
WITH matchups AS (
  SELECT
    p1.character_id AS char1,
    p2.character_id AS char2,
    p1.replay_file,
    p1.player_index AS player1,
    p2.player_index AS player2
  FROM read_parquet('replay_parquet_test/partition_*.parquet') p1
  INNER JOIN read_parquet('replay_parquet_test/partition_*.parquet') p2
    ON p1.replay_file = p2.replay_file
    AND p1.frame_number = p2.frame_number
    AND p1.player_index != p2.player_index
  WHERE p1.character_id = 1  -- Fox
    AND p2.character_id = 18  -- Marth
  GROUP BY char1, char2, p1.replay_file, player1, player2
)
SELECT
  COUNT(*) AS total_games,
  COUNT(DISTINCT replay_file) AS unique_replays
FROM matchups;
```

## Notes

- Character IDs are consistent across all Slippi replay files
- IDs 29-32 are special characters (wireframes, Giga Bowser, Sandbag) rarely seen in competitive play
- ID 255 indicates unknown or corrupted character data
- Tier list is based on competitive Melee tier lists (circa 2020s)

## References

- [Melee Tier List](https://www.ssbwiki.com/Tier_list#Super_Smash_Bros._Melee_tier_list)
- [Character Frame Data](https://ikneedata.com/)
- [Slippi Character IDs](https://github.com/project-slippi/slippi-js/blob/master/src/melee/characters.ts)
