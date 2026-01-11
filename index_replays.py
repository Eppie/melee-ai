from __future__ import annotations

import argparse
import gzip
import shutil
import sqlite3
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import peppi_py

# Import the filter functions from filter_bad_replays
from filter_bad_replays import (
    Character,
    sanity_reason,
    quality_reason,
)

REPLAYS_ZIP_DIR = Path("/Users/eppie/Downloads/ALL_REPLAYS")
DB_PATH = Path("replay_index.db")


class ReplayMetadata(NamedTuple):
    """Metadata extracted from a single replay file."""
    filename: str
    zip_file: str
    num_frames: int | None
    p1_character: int | None
    p2_character: int | None
    stage: int | None
    passed: bool
    fail_reason: str | None
    fail_message: str | None


def create_database(db_path: Path) -> None:
    """Create the database and table if they don't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS replays (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            zip_file TEXT NOT NULL,
            num_frames INTEGER,
            p1_character INTEGER,
            p2_character INTEGER,
            stage INTEGER,
            passed BOOLEAN NOT NULL,
            fail_reason TEXT,
            fail_message TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(zip_file, filename)
        )
    """)

    # Create indexes for faster lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_zip_filename
        ON replays(zip_file, filename)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_passed
        ON replays(passed)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_characters
        ON replays(p1_character, p2_character)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_filename_pattern
        ON replays(filename)
    """)

    conn.commit()
    conn.close()


def get_processed_files(db_path: Path, zip_file: str) -> set[str]:
    """Get the set of already-processed filenames for a given zip."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT filename FROM replays WHERE zip_file = ?",
        (zip_file,)
    )
    processed = {row[0] for row in cursor.fetchall()}
    conn.close()
    return processed


def insert_replay_metadata_batch(db_path: Path, metadata_list: list[ReplayMetadata]) -> None:
    """Insert multiple replay metadata records into the database."""
    if not metadata_list:
        return

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
    cursor = conn.cursor()

    cursor.executemany("""
        INSERT OR REPLACE INTO replays
        (filename, zip_file, num_frames, p1_character, p2_character,
         stage, passed, fail_reason, fail_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        (
            m.filename,
            m.zip_file,
            m.num_frames,
            m.p1_character,
            m.p2_character,
            m.stage,
            m.passed,
            m.fail_reason,
            m.fail_message,
        )
        for m in metadata_list
    ])

    conn.commit()
    conn.close()


def _extract_member(zip_path: Path, member_name: str, dest_dir: Path) -> Path:
    """
    Extract a single member from the archive to dest_dir.
    .gz files are decompressed to their base name.
    """
    target_name = Path(member_name).name
    dest_path = dest_dir / target_name
    if dest_path.suffix == ".gz":
        dest_path = dest_path.with_suffix("")

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member_name) as src:
            if target_name.endswith(".gz"):
                with gzip.open(src, "rb") as gz_in, open(dest_path, "wb") as out:
                    shutil.copyfileobj(gz_in, out, length=1024 * 1024)
            else:
                with open(dest_path, "wb") as out:
                    shutil.copyfileobj(src, out, length=1024 * 1024)

    return dest_path


def process_replay_member(
    zip_path: Path,
    member_name: str,
) -> ReplayMetadata:
    """
    Extract and analyze a single replay from a zip archive.
    Returns metadata including validation results.
    """
    with tempfile.TemporaryDirectory(prefix="slp_index_") as tmpdir:
        try:
            # Extract the file
            extracted_path = _extract_member(zip_path, member_name, Path(tmpdir))

            # First pass: Quick sanity checks without parsing frame data
            game = peppi_py.read_slippi(str(extracted_path), skip_frames=True)

            # Check sanity
            reason = sanity_reason(game)
            if reason is not None:
                code, message = reason
                return ReplayMetadata(
                    filename=Path(member_name).name,
                    zip_file=zip_path.name,
                    num_frames=None,
                    p1_character=None,
                    p2_character=None,
                    stage=game.start.stage if game.start else None,
                    passed=False,
                    fail_reason=code,
                    fail_message=message,
                )

            # Second pass: Parse frames for quality checks
            game = peppi_py.read_slippi(str(extracted_path), skip_frames=False)

            # Get frame count
            num_frames = None
            if game.frames and game.frames.ports:
                try:
                    num_frames = len(game.frames.ports[0].leader.pre.buttons)
                except Exception:
                    pass

            # Check quality
            quality = quality_reason(game)
            if quality is not None:
                code, message = quality
                return ReplayMetadata(
                    filename=Path(member_name).name,
                    zip_file=zip_path.name,
                    num_frames=num_frames,
                    p1_character=game.start.players[0].character if game.start and len(game.start.players) >= 1 else None,
                    p2_character=game.start.players[1].character if game.start and len(game.start.players) >= 2 else None,
                    stage=game.start.stage if game.start else None,
                    passed=False,
                    fail_reason=code,
                    fail_message=message,
                )

            # Passed all checks
            return ReplayMetadata(
                filename=Path(member_name).name,
                zip_file=zip_path.name,
                num_frames=num_frames,
                p1_character=game.start.players[0].character if game.start else None,
                p2_character=game.start.players[1].character if game.start else None,
                stage=game.start.stage if game.start else None,
                passed=True,
                fail_reason=None,
                fail_message=None,
            )

        except Exception as exc:
            # Catch any parsing errors
            return ReplayMetadata(
                filename=Path(member_name).name,
                zip_file=zip_path.name,
                num_frames=None,
                p1_character=None,
                p2_character=None,
                stage=None,
                passed=False,
                fail_reason="corrupt",
                fail_message=str(exc),
            )


def iter_zip_members(zip_path: Path) -> list[str]:
    """Get all .slp and .gz members from a zip file."""
    members = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if name.endswith(".slp") or name.endswith(".gz"):
                members.append(name)
    return members


def process_zip_file(
    zip_path: Path,
    db_path: Path,
    workers: int,
    batch_size: int = 100,
) -> None:
    """Process all replays in a single zip file."""
    print(f"\n{'='*80}")
    print(f"Processing: {zip_path.name}")
    print(f"{'='*80}")

    # Get all members
    all_members = iter_zip_members(zip_path)
    print(f"Found {len(all_members)} replay files in archive")

    # Check which have already been processed
    processed = get_processed_files(db_path, zip_path.name)
    unprocessed = [m for m in all_members if Path(m).name not in processed]

    print(f"Already processed: {len(processed)}")
    print(f"Remaining: {len(unprocessed)}")

    if not unprocessed:
        print("All files already processed, skipping.")
        return

    # Process in parallel
    completed = 0
    batch = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_replay_member, zip_path, member): member
            for member in unprocessed
        }

        for future in as_completed(futures):
            member = futures[future]
            try:
                metadata = future.result()
                batch.append(metadata)

                # Insert in batches for efficiency
                if len(batch) >= batch_size:
                    insert_replay_metadata_batch(db_path, batch)
                    completed += len(batch)
                    batch = []
                    print(f"Progress: {completed}/{len(unprocessed)} ({completed/len(unprocessed)*100:.1f}%)")

            except Exception as exc:
                print(f"Error processing {member}: {exc}")

        # Insert remaining batch
        if batch:
            insert_replay_metadata_batch(db_path, batch)
            completed += len(batch)

    print(f"Completed {zip_path.name}: {completed} files processed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index Slippi replays from zip archives into a SQLite database."
    )
    parser.add_argument(
        "--zip-dir",
        type=Path,
        default=REPLAYS_ZIP_DIR,
        help="Directory containing .zip archives to scan.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DB_PATH,
        help="Path to SQLite database file.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker processes for parallel processing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of records to batch before writing to database.",
    )
    args = parser.parse_args()

    # Create database if needed
    create_database(args.db_path)
    print(f"Database initialized: {args.db_path}")

    # Find all zip files
    zip_files = sorted(args.zip_dir.glob("*.zip"))
    print(f"\nFound {len(zip_files)} zip files to process")

    # Process each zip file
    for i, zip_path in enumerate(zip_files, 1):
        print(f"\n[{i}/{len(zip_files)}]")
        try:
            process_zip_file(zip_path, args.db_path, args.workers, args.batch_size)
        except Exception as exc:
            print(f"Failed to process {zip_path.name}: {exc}")
            continue

    print("\n" + "="*80)
    print("Indexing complete!")
    print("="*80)

    # Print summary statistics
    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM replays")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM replays WHERE passed = 1")
    passed = cursor.fetchone()[0]

    print(f"\nTotal replays indexed: {total:,}")
    print(f"Passed all checks: {passed:,} ({passed/total*100:.1f}%)")
    print(f"Failed checks: {total-passed:,} ({(total-passed)/total*100:.1f}%)")

    conn.close()


if __name__ == "__main__":
    main()
