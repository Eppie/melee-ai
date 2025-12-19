# Replay Index Scripts

Two scripts for efficiently indexing and querying large collections of Slippi replays stored in zip archives.

## Scripts

### 1. `index_replays.py` - Build the Index

Creates a SQLite database with metadata from all replays in zip archives.

**Features:**
- Extracts metadata without moving files (filename, characters, stage, frame count)
- Validates replays using `filter_bad_replays.py` checks
- Records which checks failed and why
- **Resumable**: Re-running skips already-processed files
- **Parallel processing**: Uses multiple worker processes for speed

**Usage:**
```bash
# Basic usage (uses defaults)
python index_replays.py

# Custom options
python index_replays.py \
  --zip-dir /path/to/zips \
  --db-path my_index.db \
  --workers 16 \
  --batch-size 200
```

**Arguments:**
- `--zip-dir`: Directory containing `.zip` archives (default: `/Users/eppie/Downloads/ALL_REPLAYS`)
- `--db-path`: SQLite database path (default: `replay_index.db`)
- `--workers`: Number of parallel workers (default: 8)
- `--batch-size`: Records per batch write (default: 100)

**Output:**
Creates `replay_index.db` with schema:
```sql
CREATE TABLE replays (
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,           -- replay filename
    zip_file TEXT NOT NULL,            -- source zip file
    num_frames INTEGER,                -- frame count (NULL if unparseable)
    p1_character INTEGER,              -- character ID (peppi_py encoding)
    p2_character INTEGER,              -- character ID (peppi_py encoding)
    stage INTEGER,                     -- stage ID
    passed BOOLEAN NOT NULL,           -- passed all validation checks?
    fail_reason TEXT,                  -- reason code if failed
    fail_message TEXT,                 -- detailed failure message
    processed_at TIMESTAMP,
    UNIQUE(zip_file, filename)
);
```

### 2. `query_replay_index.py` - Query and Extract

Query the database and extract matching replays.

**Usage:**

**Show statistics:**
```bash
python query_replay_index.py --stats
```

**Query examples:**
```bash
# Extract all Fox dittos from master-master-* files
python query_replay_index.py \
  --filename-pattern "master-master-*" \
  --char1 fox \
  --char2 fox \
  --output-dir fox_dittos

# Extract all Marth vs Fox replays (either order)
python query_replay_index.py \
  --char1 marth \
  --char2 fox \
  --output-dir marth_fox

# Extract long matches (>5 minutes at 60fps)
python query_replay_index.py \
  --min-frames 18000 \
  --output-dir long_matches

# Dry run to see what would be extracted
python query_replay_index.py \
  --char1 fox \
  --dry-run

# Extract first 100 matching replays
python query_replay_index.py \
  --char1 falcon \
  --limit 100
```

**Arguments:**
- `--db-path`: Database path (default: `replay_index.db`)
- `--zip-dir`: Directory with zip files (default: `/Users/eppie/Downloads/ALL_REPLAYS`)
- `--output-dir`: Extraction destination (default: `extracted_replays`)
- `--filename-pattern`: Filename glob pattern (use `*` for wildcards)
- `--char1`, `--char2`: Character names (e.g., `fox`, `marth`, `puff`)
- `--stage`: Stage ID filter
- `--min-frames`, `--max-frames`: Frame count filters
- `--include-failed`: Include replays that failed validation
- `--stats`: Show database statistics
- `--dry-run`: Show matches without extracting
- `--limit`: Limit number of extractions

## Example Workflow

```bash
# 1. Build the index (may take a while for large collections)
python index_replays.py --workers 16

# 2. Check statistics
python query_replay_index.py --stats

# 3. Extract specific replays
python query_replay_index.py \
  --filename-pattern "master-diamond-*" \
  --char1 fox \
  --output-dir fox_master_diamond
```

## Character Names

Valid character names (case-insensitive):
`FALCON`, `DK`, `FOX`, `GNW`, `KIRBY`, `BOWSER`, `LINK`, `LUIGI`, `MARIO`, `MARTH`, `MEWTWO`, `NESS`, `PEACH`, `PIKA`, `ICS`, `PUFF`, `SAMUS`, `YOSHI`, `ZELDA`, `SHEIK`, `FALCO`, `YLINK`, `DOC`, `ROY`, `PICHU`, `GANON`

## Validation Checks

The indexer runs the same validation checks as `filter_bad_replays.py`:

**Sanity checks** (fast, no frame parsing):
- Missing metadata
- No contest endings
- Teams mode, items, raining bombs
- Wrong damage ratio, timer, stock count
- PAL version
- Non-human players
- Illegal stages
- Not exactly 2 players

**Quality checks** (requires frame parsing):
- Frame count out of range (3600-21600)
- Low input variance (AFK/controller unplugged detection)

## Performance Tips

- Use `--workers` based on CPU cores (e.g., 16 for 8-core with hyperthreading)
- Increase `--batch-size` for faster writes (200-500 on SSD)
- The script is resumable - it skips already-indexed files automatically
- First run will be slow; subsequent runs only process new files

## Database Queries

You can also query the database directly with SQL:

```bash
sqlite3 replay_index.db

# Find all Fox dittos
SELECT filename, zip_file FROM replays
WHERE p1_character = 2 AND p2_character = 2 AND passed = 1;

# Count replays by character matchup
SELECT p1_character, p2_character, COUNT(*)
FROM replays
WHERE passed = 1
GROUP BY p1_character, p2_character
ORDER BY COUNT(*) DESC;

# Find short matches that still passed
SELECT filename, num_frames FROM replays
WHERE passed = 1 AND num_frames < 5000
ORDER BY num_frames;
```

## Character ID Reference

| ID | Character | ID | Character |
|----|-----------|----|-----------|
| 0 | Falcon | 13 | Sheik |
| 1 | DK | 14 | Falco |
| 2 | Fox | 15 | Young Link |
| 3 | G&W | 16 | Dr. Mario |
| 4 | Kirby | 17 | Roy |
| 5 | Bowser | 18 | Pichu |
| 6 | Link | 19 | Ganondorf |
| 7 | Luigi | 20 | Popo (ICs) |
| 8 | Mario | ... | ... |
| 9 | Marth | | |
| 10 | Mewtwo | | |
| 11 | Ness | | |
| 12 | Peach | | |

(See `filter_bad_replays.py` Character enum for complete list)
