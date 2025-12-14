"""Test to check if zarr slices return C-contiguous arrays."""

import numpy as np
import zarr
from pathlib import Path


def test_zarr_contiguity():
    """Load some windows from zarr and check if they're C-contiguous."""

    # Find first shard - try different data directories
    for data_dir_name in [
        "processed_data_5700",
        "processed_data_1000",
        "processed_data_100",
        "processed_data",
    ]:
        data_dir = Path(data_dir_name)
        if data_dir.exists():
            shard_files = sorted(data_dir.glob("shard_*.zarr"))
            if shard_files:
                break
    else:
        print("No shard files found in any processed_data directory")
        return

    print(f"Using data directory: {data_dir}")

    print(f"Testing with {shard_files[0]}")

    # Open zarr (use open_group like the actual code does)
    root = zarr.open_group(str(shard_files[0]), mode="r", path=None)

    # Get first episode
    ep_names = sorted([k for k in root.keys() if k.startswith("ep_")])
    if not ep_names:
        print("No episodes found in shard")
        return

    print(f"\nTesting with {ep_names[0]}")
    ep = root[ep_names[0]]

    X = ep["X"]
    Y = ep["Y"]

    print(f"\nArray info:")
    print(f"  X shape: {X.shape}, dtype: {X.dtype}, chunks: {X.chunks}")
    print(f"  Y shape: {Y.shape}, dtype: {Y.dtype}, chunks: {Y.chunks}")

    # Test different slice positions
    seq_len = 256
    test_positions = [0, 100, 200, 256, 512, 1000]

    print(f"\nTesting {seq_len}-frame windows at different positions:")
    print(
        f"{'Position':>8} | {'X C-contig':>11} | {'Y C-contig':>11} | {'X needs copy':>13} | {'Y needs copy':>13}"
    )
    print("-" * 75)

    for start in test_positions:
        if start + seq_len > X.shape[0]:
            continue

        x_slice = X[start : start + seq_len, :]
        y_slice = Y[start : start + seq_len, :]

        x_contig = x_slice.flags["C_CONTIGUOUS"]
        y_contig = y_slice.flags["C_CONTIGUOUS"]

        print(
            f"{start:>8} | {str(x_contig):>11} | {str(y_contig):>11} | {str(not x_contig):>13} | {str(not y_contig):>13}"
        )

    # Test chunk boundary crossing
    chunk_size_t = X.chunks[0] if X.chunks else None
    if chunk_size_t:
        print(f"\nChunk size (time axis): {chunk_size_t}")
        print(f"\nTesting chunk boundary crossings:")

        # Test at chunk boundary
        boundary_tests = [
            (chunk_size_t - 128, "128 frames before boundary"),
            (chunk_size_t - 10, "10 frames before boundary"),
            (chunk_size_t, "exactly at boundary"),
            (chunk_size_t + 10, "10 frames after boundary"),
        ]

        print(
            f"{'Position':>8} | {'Description':>30} | {'X C-contig':>11} | {'Y C-contig':>11}"
        )
        print("-" * 75)

        for start, desc in boundary_tests:
            if start + seq_len > X.shape[0]:
                continue

            x_slice = X[start : start + seq_len, :]
            y_slice = Y[start : start + seq_len, :]

            x_contig = x_slice.flags["C_CONTIGUOUS"]
            y_contig = y_slice.flags["C_CONTIGUOUS"]

            print(
                f"{start:>8} | {desc:>30} | {str(x_contig):>11} | {str(y_contig):>11}"
            )

    # Test actual memory cost
    print(f"\n\nMemory overhead test:")
    start = 0
    x_slice = X[start : start + seq_len, :]

    print(f"Original slice nbytes: {x_slice.nbytes:,} bytes")
    if not x_slice.flags["C_CONTIGUOUS"]:
        x_contig = np.ascontiguousarray(x_slice)
        print(f"After ascontiguousarray: {x_contig.nbytes:,} bytes")
        print(f"Overhead: makes copy = {x_contig.nbytes:,} bytes per window")
    else:
        print(f"Already contiguous - no copy needed!")

    print(f"\nWith batch_size=128, overhead per batch:")
    print(f"  If all need copy: {128 * x_slice.nbytes / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    test_zarr_contiguity()
