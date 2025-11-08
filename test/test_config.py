from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
from zarr.codecs import BloscCodec, BloscShuffle

from config import ZarrConfig, init_config, reset_config


@pytest.fixture(autouse=True)
def clear_global_config():
    """Ensure init_config() global state does not leak between tests."""
    reset_config()
    yield
    reset_config()


def test_zarr_out_root_includes_episode_count(tmp_path: Path):
    base_out_root = tmp_path / "processed_data"
    cfg = ZarrConfig(out_root=str(base_out_root), episode_count=32)
    assert cfg.out_root == f"{base_out_root}_32"


def test_model_input_size_computed_from_context():
    cfg = init_config(gamestate_dim=123, controller_dim=45, freeze=False)
    expected_size = (
        cfg.model.num_stages
        + cfg.model.num_characters * 2
        + cfg.model.num_actions * 2
        + 123
        + 45
    )
    assert cfg.model.input_size == expected_size


@pytest.mark.parametrize(
    "compressor_input, expected",
    [
        (None, BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle)),
        (
            {
                "configuration": {
                    "cname": "lz4",
                    "clevel": 1,
                    "shuffle": "noshuffle",
                }
            },
            BloscCodec(cname="lz4", clevel=1, shuffle=BloscShuffle.noshuffle),
        ),
        (
            {"cname": "blosclz", "clevel": 5, "shuffle": "shuffle"},
            BloscCodec(cname="blosclz", clevel=5, shuffle=BloscShuffle.shuffle),
        ),
        (
            BloscCodec(cname="zstd", clevel=9, shuffle=BloscShuffle.bitshuffle),
            BloscCodec(cname="zstd", clevel=9, shuffle=BloscShuffle.bitshuffle),
        ),
        (
            {"cname": "zstd", "clevel": 2, "shuffle": BloscShuffle.noshuffle},
            BloscCodec(cname="zstd", clevel=2, shuffle=BloscShuffle.noshuffle),
        ),
    ],
)
def test_zarr_compressor_supported_inputs(compressor_input: Any, expected: BloscCodec):
    cfg = ZarrConfig(compressor=compressor_input)
    assert isinstance(cfg.compressor, BloscCodec)
    assert cfg.compressor.cname == expected.cname
    assert cfg.compressor.clevel == expected.clevel
    assert cfg.compressor.shuffle == expected.shuffle


def test_zarr_compressor_invalid_type_falls_back_to_default():
    cfg = ZarrConfig(compressor="invalid")
    assert isinstance(cfg.compressor, BloscCodec)
    assert cfg.compressor.cname.value == "zstd"
