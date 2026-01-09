from __future__ import annotations

import platform
from typing import Tuple

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import SettingsConfigDict
from zarr.codecs import BloscCodec, BloscShuffle


def _get_default_paths() -> Tuple[str, str, str]:
    """
    Automatically determine default paths based on operating system.
    Returns (input_root, out_root, validation_root)
    """
    system = platform.system()

    if system == "Darwin":
        return (
            "/Users/eppie/Downloads/ALL_REPLAYS/FOX_vs_FOX",
            "/Users/eppie/PycharmProjects/nano-melee/processed_data_1000",
            "/Users/eppie/PycharmProjects/nano-melee/validation_set",
        )
    elif system == "Linux":
        return (
            "/home/eppie/hal/fox_dittos",
            "/home/eppie/melee-ai/processed_data_1000",
            "/home/eppie/melee-ai/validation_set",
        )
    raise ValueError(f"Unknown operating system: {system}")


class ZarrConfig(BaseModel):
    model_config = SettingsConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,  # Allow BloscCodec
    )

    input_root: str = Field(
        default_factory=lambda: _get_default_paths()[0],
        description=(
            "Directory containing input Slippi .slp replay files. Auto-detected based on OS. "
            "Override with custom path if replays are in a different location."
        ),
    )
    out_root: str = Field(
        default_factory=lambda: _get_default_paths()[1],
        description=(
            "Directory for output Zarr dataset (processed training data). "
            "Auto-includes episode_count suffix (e.g., processed_data_1000). "
            "Auto-detected based on OS. Override with custom path."
        ),
    )
    validation_root: str = Field(
        default_factory=lambda: _get_default_paths()[2],
        description=(
            "Directory for validation Zarr dataset (held-out test data). "
            "Separate from training data for unbiased evaluation. Auto-detected based on OS."
        ),
    )
    episode_count: int = Field(
        default=6000,
        ge=1,
        description=(
            "Number of episodes (replay files) to process for training dataset. "
            "Effect: More episodes = more diverse training data but longer preprocessing. "
            "Reasonable range: [1000, 10000+]. Note: out_root path is auto-updated to include this count. "
            "Interacts with: shard_size (determines number of shards = episode_count / shard_size)."
        ),
    )
    validation_count: int = Field(
        default=500,
        ge=1,
        description=(
            "Number of episodes to process for validation dataset. Held out from training. "
            "Effect: More episodes = better validation statistics but longer preprocessing. "
            "Reasonable range: [100, 500]. Typical: 5-10% of episode_count."
        ),
    )
    shard_size: int = Field(
        default=100,
        ge=1,
        description=(
            "Number of episodes per Zarr shard. Data is split into multiple shards for parallel loading. "
            "Effect: Smaller shards (50-100) = more parallelism, better for many workers; "
            "larger shards (200-500) = fewer files, simpler management. Reasonable range: [50, 200]. "
            "Interacts with: episode_count (num_shards = episode_count / shard_size), num_workers."
        ),
    )
    target_chunk_mb: float = Field(
        default=8.0,
        gt=0,
        description=(
            "Target chunk size in megabytes for Zarr chunks along the feature axis. "
            "Balances I/O efficiency vs memory usage. Effect: Larger chunks (16-32 MB) = fewer I/O operations; "
            "smaller chunks (4-8 MB) = finer-grained access. Reasonable range: [4.0, 32.0]. "
            "Interacts with: chunk_frames (both determine chunk shape)."
        ),
    )
    chunk_frames: int = Field(
        default=512,
        ge=1,
        description=(
            "Number of frames per Zarr chunk along the time axis. Optimized for typical window sizes. "
            "Effect: Should be >= block_size to minimize chunk reads per window. "
            "Default 512 ensures 256-512 frame windows usually hit 1-2 chunks. "
            "Reasonable range: [256, 1024]. Interacts with: model.block_size (should be >= block_size). "
            "Ignored when sequential_episodes=True (each episode becomes one chunk)."
        ),
    )
    sequential_episodes: bool = Field(
        default=False,
        description=(
            "When True, chunk each episode as a single unit and use sequential sampling within episodes. "
            "Effect: Episodes are shuffled, but windows within each episode are sampled sequentially. "
            "This improves temporal locality for recurrent-style training. "
            "When False (default), windows are sampled globally with random shuffling. "
            "Interacts with: chunk_frames (ignored when True), training sampler selection."
        ),
    )
    seed: int = Field(
        default=42,
        description=(
            "Random seed for dataset splitting and shuffling. "
            "Use consistent seed for reproducible train/validation splits."
        ),
    )
    compressor: BloscCodec = Field(
        default_factory=lambda: BloscCodec(
            cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle
        ),
        description=(
            "Blosc compression codec for Zarr arrays. Controls compression algorithm and level. "
            "Default: zstd (fast, good ratio), level 3 (balanced), bitshuffle (good for game data). "
            "Effect: Higher clevel (5-9) = better compression, slower read/write; "
            "lower clevel (1-3) = faster, larger files. Reasonable clevel range: [1, 7]. "
            "Options: cname=['zstd', 'lz4', 'blosclz'], shuffle=['noshuffle', 'shuffle', 'bitshuffle']. "
            "Recommended: Keep defaults unless storage or I/O speed is critical."
        ),
    )

    @field_validator("compressor", mode="before")
    def validate_compressor(cls, v):
        """Ensure compressor is always a valid BloscCodec."""
        if v is None:
            return BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle)

        # If it's a dict (from JSON), reconstruct the BloscCodec
        if isinstance(v, dict):
            # Handle Zarr v3 codec dict format
            if "configuration" in v:
                config = v["configuration"]
                cname = config.get("cname", "zstd")
                clevel = config.get("clevel", 3)
                shuffle_str = config.get("shuffle", "bitshuffle")
                shuffle = (
                    BloscShuffle[shuffle_str]
                    if isinstance(shuffle_str, str)
                    else shuffle_str
                )
                return BloscCodec(cname=cname, clevel=clevel, shuffle=shuffle)

            # Handle simple dict format
            cname = v.get("cname", "zstd")
            clevel = v.get("clevel", 3)
            shuffle_val = v.get("shuffle", BloscShuffle.bitshuffle)

            # Handle shuffle as string or enum
            if isinstance(shuffle_val, str):
                shuffle = BloscShuffle[shuffle_val]
            elif isinstance(shuffle_val, int):
                shuffle = BloscShuffle(shuffle_val)
            else:
                shuffle = shuffle_val

            return BloscCodec(cname=cname, clevel=clevel, shuffle=shuffle)

        if isinstance(v, BloscCodec):
            return v

        return BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle)

    @model_validator(mode="after")
    def validate_paths_and_sharding(self):
        """Validate paths exist and sharding makes sense."""
        if self.compressor is None:
            # Use object.__setattr__ to bypass frozen config if needed
            object.__setattr__(
                self,
                "compressor",
                BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle),
            )

        # Double-check it's a valid BloscCodec instance
        if not isinstance(self.compressor, BloscCodec):
            object.__setattr__(
                self,
                "compressor",
                BloscCodec(cname="zstd", clevel=7, shuffle=BloscShuffle.bitshuffle),
            )

        self.update_out_root_for_episode_count()
        return self

    def update_out_root_for_episode_count(self) -> None:
        """
        Update out_root to include episode_count in the path.
        Call this after setting episode_count to keep paths in sync.
        """
        # Extract base path without episode count suffix
        base_path = str(self.out_root)
        # Remove any existing _N suffix
        import re

        base_path = re.sub(r"_(\d+)$", "", base_path)
        object.__setattr__(self, "out_root", f"{base_path}_{self.episode_count}")
