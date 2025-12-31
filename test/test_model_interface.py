from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import torch
from tensordict import TensorDict

from model_interface import FrameRecord, GPTInferenceEngine


def _make_outputs() -> TensorDict:
    return TensorDict(
        {
            "main_stick": torch.randn(1, 2, 4),
            "c_stick": torch.randn(1, 2, 4),
            "buttons": torch.randn(1, 2, 5),
            "shoulder": torch.randn(1, 2, 3),
            "value": torch.randn(1, 2, 1),
        },
        batch_size=[1, 2],
    )


def test_capture_logits_returns_tensors() -> None:
    engine = object.__new__(GPTInferenceEngine)
    outputs = _make_outputs()

    logits = engine._capture_logits(outputs)

    for key in ["main_stick", "c_stick", "buttons", "shoulder", "value"]:
        assert key in logits
        assert torch.is_tensor(logits[key])
        assert logits[key].shape[0] == outputs[key].shape[-1]


def test_persist_death_record_serializes_logits(tmp_path: Path) -> None:
    engine = object.__new__(GPTInferenceEngine)
    engine.frame_history = deque(maxlen=2)
    engine.seq_len = 2
    engine.warmup_frames = 0
    engine._frames_seen = 7
    engine._death_log_dir = tmp_path / "death_logs"
    engine._death_counter = 0
    engine.feature_names = ["feat"]
    engine.target_names = ["target"]

    outputs = _make_outputs()
    logits = engine._capture_logits(outputs)
    record = FrameRecord(
        raw_features={"feat": 0.0},
        transformed_features={"feat": 0.0},
        targets={"target": 0.0},
        logits=logits,
    )
    engine.frame_history.append(record)

    assert torch.is_tensor(record.logits["main_stick"])

    engine._persist_death_record()

    files = list((engine._death_log_dir).glob("death_*.json"))
    assert files, "Death log file not created"
    payload = json.loads(files[0].read_text())

    frame_logits = payload["frames"][0]["logits"]
    assert isinstance(frame_logits["main_stick"], list)
    assert isinstance(frame_logits["buttons"], list)
    # Original tensors should remain tensors for potential reuse
    assert torch.is_tensor(record.logits["main_stick"])


def test_maybe_log_death_detects_death_transition() -> None:
    """Test that _maybe_log_death detects transitions into death action states."""
    engine = object.__new__(GPTInferenceEngine)
    engine.frame_history = deque(maxlen=10)
    engine.seq_len = 10
    engine.warmup_frames = 0
    engine._frames_seen = 100
    engine._death_counter = 0
    engine._prev_action = None
    engine.feature_names = ["p1_action"]
    engine.target_names = []

    # Track if _persist_death_record was called
    death_logged = []
    original_persist = GPTInferenceEngine._persist_death_record

    def mock_persist(self):
        death_logged.append(True)

    GPTInferenceEngine._persist_death_record = mock_persist

    try:
        # First call - no previous action, no death logged
        engine._maybe_log_death({"p1_action": 50})  # Normal action state
        assert len(death_logged) == 0

        # Transition to non-death state - no death logged
        engine._maybe_log_death({"p1_action": 100})
        assert len(death_logged) == 0

        # Transition into death state (action 5 is in range 0-10)
        engine._maybe_log_death({"p1_action": 5})
        assert len(death_logged) == 1

        # Stay in death state - no additional log
        engine._maybe_log_death({"p1_action": 3})
        assert len(death_logged) == 1

        # Exit death state
        engine._maybe_log_death({"p1_action": 50})
        assert len(death_logged) == 1

        # Enter death state again
        engine._maybe_log_death({"p1_action": 0})
        assert len(death_logged) == 2

    finally:
        GPTInferenceEngine._persist_death_record = original_persist
