#!/usr/bin/env python3
"""Create a zip file from compression_test_input for profiling."""

import zipfile
from pathlib import Path

SOURCE_DIR = Path("/Users/eppie/Downloads/ALL_REPLAYS/compression_test_input")
OUTPUT_ZIP = Path("/Users/eppie/PycharmProjects/nano-melee/test_replays.zip")

def create_zip():
    """Create a zip file from all .slp files in compression_test_input."""
    slp_files = sorted(SOURCE_DIR.glob("*.slp"))

    print(f"Found {len(slp_files)} .slp files")
    print(f"Creating zip: {OUTPUT_ZIP}")

    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        for slp_file in slp_files:
            print(f"  Adding: {slp_file.name}")
            zf.write(slp_file, arcname=slp_file.name)

    print(f"✓ Created {OUTPUT_ZIP}")
    print(f"  Zip size: {OUTPUT_ZIP.stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    create_zip()
