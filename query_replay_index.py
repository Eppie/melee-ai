from __future__ import annotations

import argparse
import gzip
import shutil
import sqlite3
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from filter_bad_replays import Character

DB_PATH = Path("replay_index.db")
REPLAYS_ZIP_DIR = Path("/Users/eppie/Downloads/ALL_REPLAYS")
OUTPUT_DIR = Path("extracted_replays")


def query_replays(
    db_path: Path,
    filename_pattern: str | None = None,
    p1_char: str | None = None,
    p2_char: str | None = None,
    stage: int | None = None,
    passed_only: bool = True,
    min_frames: int | None = None,
    max_frames: int | None = None,
) -> list[tuple[str, str]]:
    """
    Query the replay database with various filters.

    Returns:
        List of (zip_file, filename) tuples matching the criteria.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = "SELECT zip_file, filename FROM replays WHERE 1=1"
    params = []

    if filename_pattern:
        query += " AND filename LIKE ?"
        params.append(filename_pattern.replace("*", "%"))

    if passed_only:
        query += " AND passed = 1"

    # Handle character filters
    if p1_char or p2_char:
        p1_id = Character[p1_char.upper()].value if p1_char else None
        p2_id = Character[p2_char.upper()].value if p2_char else None

        if p1_id is not None and p2_id is not None:
            # Both characters specified - match either order
            query += " AND ((p1_character = ? AND p2_character = ?) OR (p1_character = ? AND p2_character = ?))"
            params.extend([p1_id, p2_id, p2_id, p1_id])
        elif p1_id is not None:
            # Only p1 specified - match either player
            query += " AND (p1_character = ? OR p2_character = ?)"
            params.extend([p1_id, p1_id])
        elif p2_id is not None:
            # Only p2 specified - match either player
            query += " AND (p1_character = ? OR p2_character = ?)"
            params.extend([p2_id, p2_id])

    if stage is not None:
        query += " AND stage = ?"
        params.append(stage)

    if min_frames is not None:
        query += " AND num_frames >= ?"
        params.append(min_frames)

    if max_frames is not None:
        query += " AND num_frames <= ?"
        params.append(max_frames)

    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()

    return results


def extract_replay(
    zip_path: Path,
    member_name: str,
    output_dir: Path,
) -> Path:
    """Extract a single replay file from a zip archive."""
    output_dir.mkdir(parents=True, exist_ok=True)

    target_name = Path(member_name).name
    dest_path = output_dir / target_name
    if dest_path.suffix == ".gz":
        dest_path = dest_path.with_suffix("")

    with zipfile.ZipFile(zip_path) as zf:
        # Find the member (could be nested in subdirectories)
        matching_members = [m for m in zf.namelist() if m.endswith(member_name) or Path(m).name == member_name]

        if not matching_members:
            raise FileNotFoundError(f"Could not find {member_name} in {zip_path}")

        actual_member = matching_members[0]

        with zf.open(actual_member) as src:
            if target_name.endswith(".gz"):
                with gzip.open(src, "rb") as gz_in, open(dest_path, "wb") as out:
                    shutil.copyfileobj(gz_in, out, length=1024 * 1024)
            else:
                with open(dest_path, "wb") as out:
                    shutil.copyfileobj(src, out, length=1024 * 1024)

    return dest_path


def extract_from_zip(
    zip_path: Path,
    members: list[str],
    output_dir: Path,
) -> tuple[int, int, list[str]]:
    """
    Extract multiple replay files from a single zip archive.
    Opens the zip once and extracts all matching members.

    Returns: (extracted_count, failed_count, error_messages)
    """
    extracted = 0
    failed = 0
    errors = []

    try:
        with zipfile.ZipFile(zip_path) as zf:
            # Build a map of base names to full paths in the zip (scan namelist once)
            name_to_path: dict[str, str] = {}
            for info in zf.infolist():
                if not info.is_dir():
                    base_name = Path(info.filename).name
                    name_to_path[base_name] = info.filename

            for member_name in members:
                target_name = Path(member_name).name
                dest_path = output_dir / target_name
                if dest_path.suffix == ".gz":
                    dest_path = dest_path.with_suffix("")

                # Find the member in our pre-built map
                actual_member = name_to_path.get(member_name)
                if actual_member is None:
                    actual_member = name_to_path.get(target_name)

                if actual_member is None:
                    failed += 1
                    errors.append(f"Could not find {member_name} in {zip_path.name}")
                    continue

                try:
                    with zf.open(actual_member) as src:
                        if target_name.endswith(".gz"):
                            with gzip.open(src, "rb") as gz_in, open(dest_path, "wb") as out:
                                shutil.copyfileobj(gz_in, out, length=1024 * 1024)
                        else:
                            with open(dest_path, "wb") as out:
                                shutil.copyfileobj(src, out, length=1024 * 1024)
                    extracted += 1
                except Exception as exc:
                    failed += 1
                    errors.append(f"Error extracting {member_name}: {exc}")

    except Exception as exc:
        failed += len(members)
        errors.append(f"Could not open {zip_path.name}: {exc}")

    return extracted, failed, errors


# Top-level function for ProcessPoolExecutor (must be picklable)
def _process_zip_task(args: tuple[Path, list[str], Path]) -> tuple[str, int, int, list[str]]:
    """Worker function to process a single zip file."""
    zip_path, members, output_dir = args
    extracted, failed, errors = extract_from_zip(zip_path, members, output_dir)
    return zip_path.name, extracted, failed, errors


def print_statistics(db_path: Path) -> None:
    """Print database statistics."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\n" + "=" * 80)
    print("DATABASE STATISTICS")
    print("=" * 80)

    # Total count
    cursor.execute("SELECT COUNT(*) FROM replays")
    total = cursor.fetchone()[0]
    print(f"\nTotal replays indexed: {total:,}")

    # Pass/fail breakdown
    cursor.execute("SELECT COUNT(*) FROM replays WHERE passed = 1")
    passed = cursor.fetchone()[0]
    print(f"  Passed all checks: {passed:,} ({passed/total*100:.1f}%)")
    print(f"  Failed checks: {total-passed:,} ({(total-passed)/total*100:.1f}%)")

    # Failure reasons
    print("\nTop failure reasons:")
    cursor.execute("""
        SELECT fail_reason, COUNT(*) as count
        FROM replays
        WHERE passed = 0
        GROUP BY fail_reason
        ORDER BY count DESC
        LIMIT 10
    """)
    for reason, count in cursor.fetchall():
        print(f"  {reason}: {count:,} ({count/total*100:.1f}%)")

    # Character distribution
    print("\nTop character matchups (passed replays only):")
    cursor.execute("""
        SELECT p1_character, p2_character, COUNT(*) as count
        FROM replays
        WHERE passed = 1 AND p1_character IS NOT NULL AND p2_character IS NOT NULL
        GROUP BY p1_character, p2_character
        ORDER BY count DESC
        LIMIT 10
    """)
    for p1, p2, count in cursor.fetchall():
        try:
            p1_name = Character(p1).name
            p2_name = Character(p2).name
            print(f"  {p1_name} vs {p2_name}: {count:,}")
        except (ValueError, KeyError):
            print(f"  Unknown({p1}) vs Unknown({p2}): {count:,}")

    conn.close()


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query and extract replays from the indexed database."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DB_PATH,
        help="Path to SQLite database file.",
    )
    parser.add_argument(
        "--zip-dir",
        type=Path,
        default=REPLAYS_ZIP_DIR,
        help="Directory containing .zip archives.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to extract matching replays to.",
    )
    parser.add_argument(
        "--filename-pattern",
        type=str,
        help="Filename pattern to match (use * for wildcards, e.g., 'master-master-*').",
    )
    parser.add_argument(
        "--char1",
        type=str,
        help="First character name (e.g., 'fox', 'marth').",
    )
    parser.add_argument(
        "--char2",
        type=str,
        help="Second character name (e.g., 'fox', 'marth').",
    )
    parser.add_argument(
        "--stage",
        type=int,
        help="Stage ID to filter by.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        help="Minimum frame count.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum frame count.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include replays that failed validation checks.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print database statistics and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show matching files without extracting them.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of files to extract.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for extraction (default: 8).",
    )

    args = parser.parse_args()

    if not args.db_path.exists():
        print(f"Error: Database not found at {args.db_path}")
        print("Run index_replays.py first to create the database.")
        return

    # Print statistics if requested
    if args.stats:
        print_statistics(args.db_path)
        return

    # Query the database
    results = query_replays(
        db_path=args.db_path,
        filename_pattern=args.filename_pattern,
        p1_char=args.char1,
        p2_char=args.char2,
        stage=args.stage,
        passed_only=not args.include_failed,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
    )

    print(f"\nFound {len(results)} matching replays")

    if not results:
        return

    # Apply limit if specified
    if args.limit:
        results = results[: args.limit]
        print(f"Limited to first {len(results)} replays")

    if args.dry_run:
        print("\nMatching files (dry run):")
        for zip_file, filename in results[:20]:  # Show first 20
            print(f"  {zip_file}: {filename}")
        if len(results) > 20:
            print(f"  ... and {len(results) - 20} more")
        return

    # Group results by zip file for efficient extraction
    by_zip: dict[str, list[str]] = defaultdict(list)
    for zip_file, filename in results:
        by_zip[zip_file].append(filename)

    print(f"Grouped into {len(by_zip)} zip files")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare tasks - split large zips into chunks for better parallelism
    # Each chunk becomes a separate task that opens its own zip handle
    chunk_size = max(50, len(results) // (args.workers * 4))  # Aim for ~4 chunks per worker
    tasks = []
    for zip_name, members in by_zip.items():
        zip_path = args.zip_dir / zip_name
        if not zip_path.exists():
            print(f"Warning: Zip file not found: {zip_path}")
            continue
        # Split into chunks for parallelism
        for i in range(0, len(members), chunk_size):
            chunk = members[i : i + chunk_size]
            tasks.append((zip_path, chunk, args.output_dir))

    print(f"\nExtracting {len(results)} replays via {len(tasks)} tasks using {args.workers} workers")

    # Track progress
    total_extracted = 0
    total_failed = 0
    total_files = len(results)
    tasks_completed = 0
    start_time = time.time()

    # Process in parallel using ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_zip_task, task): task[0].name for task in tasks}

        for future in as_completed(futures):
            zip_name, extracted, failed, errors = future.result()
            total_extracted += extracted
            total_failed += failed
            tasks_completed += 1

            # Print errors
            for err in errors:
                print(f"  Error: {err}", file=sys.stderr)

            # Progress update
            elapsed = time.time() - start_time
            files_done = total_extracted + total_failed
            rate = files_done / elapsed if elapsed > 0 else 0
            eta = (total_files - files_done) / rate if rate > 0 else 0

            print(
                f"\rProgress: {files_done:,}/{total_files:,} files "
                f"({files_done / total_files * 100:.1f}%) | "
                f"{tasks_completed}/{len(tasks)} tasks | "
                f"{rate:.0f} files/s | "
                f"ETA: {format_time(eta)}   ",
                end="",
                flush=True,
            )

    print()  # Newline after progress
    elapsed = time.time() - start_time

    print(f"\nExtraction complete in {format_time(elapsed)}")
    print(f"  Extracted: {total_extracted:,}")
    print(f"  Failed: {total_failed:,}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Average rate: {total_extracted / elapsed:.0f} files/s")


if __name__ == "__main__":
    main()
