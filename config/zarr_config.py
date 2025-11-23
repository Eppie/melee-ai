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
            "/home/eppie/hal/replays",
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

    input_root: str = Field(default_factory=lambda: _get_default_paths()[0])
    out_root: str = Field(default_factory=lambda: _get_default_paths()[1])
    validation_root: str = Field(default_factory=lambda: _get_default_paths()[2])
    episode_count: int = Field(default=100, ge=1)
    validation_count: int = Field(default=100, ge=1)
    shard_size: int = Field(default=100, ge=1)
    target_chunk_mb: float = Field(default=8.0, gt=0)
    chunk_frames: int = Field(
        default=512,
        ge=1,
        description=(
            "Preferred number of frames per Zarr chunk along the time axis. "
            "Defaults to 512 so 256-frame windows typically hit a single chunk."
        ),
    )
    seed: int = Field(default=42)
    compressor: BloscCodec = Field(
        default_factory=lambda: BloscCodec(
            cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle
        )
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
