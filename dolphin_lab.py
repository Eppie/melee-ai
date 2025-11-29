#!/usr/bin/env python3
"""
Utility script for launching Dolphin/Slippi in a controlled lab environment.

The script boots into Final Destination with two Fox players (ports 1 & 2)
both driven entirely by code.  It is designed for interactive experimentation:
attach a debugger, grab the global ``LAB`` object, and schedule controller
commands or measurement routines (e.g. measuring L-trigger shield thresholds or
testing custom movement macros).

Example usage:

    $ python dolphin_lab.py --iso /path/to/GALE01.iso --dolphin-executable-path /path/to/dolphin

Inside a debugger / REPL:

    >>> from dolphin_lab import LAB, measure_shield_threshold
    >>> measure_shield_threshold(port=1, tolerance=0.0015)
    >>> LAB.shield_thresholds
    >>> LAB.shield_drain_results
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from config import init_config
from controller_utils import CONTROL_STICK_QUANTIZED
from libmelee.melee.console import Console
from libmelee.melee.controller import Controller, fix_analog_stick, fix_analog_trigger
from libmelee.melee.enums import Action, Button, Character, ControllerType, Menu, Stage
from libmelee.melee import stages
from libmelee.melee.menuhelper import MenuHelper

# Public palette helpers for quick experimentation.
CONTROL_PALETTE: np.ndarray = CONTROL_STICK_QUANTIZED

DEFAULT_CHARGE_FRAMES = 45
DEFAULT_TRAVEL_FRAMES = 120
DEFAULT_COOLDOWN_FRAMES = 30
MIN_MOVEMENT_MAG = 0.35
NEUTRAL_SETTLE_FRAMES = 2
MAX_TRIGGER_RAW = 140
COMMAND_TRIGGER_MAX = 255
SHIELD_FULL_VALUE = 60.0
SHIELD_FULL_TOLERANCE = 0.05
DELTA_STABILITY_FRAMES = 5
DELTA_STABILITY_TOLERANCE = 1e-4
MAX_DRAIN_PRESS_FRAMES = 180
SPAWN_ACTIONS = {Action.ON_HALO_DESCENT, Action.ON_HALO_WAIT}


@dataclass
class ControllerTargets:
    """Desired per-frame state for a controller port (unit domain)."""

    main: Tuple[float, float] = (0.0, 0.0)
    c: Tuple[float, float] = (0.0, 0.0)
    l_shoulder: float = 0.0
    r_shoulder: float = 0.0
    buttons: Dict[Button, bool] = field(default_factory=lambda: defaultdict(bool))


@dataclass
class FirefoxTestResult:
    """Simple telemetry for Firefox angle sweeps."""

    angle: Tuple[float, float]
    start_pos: Tuple[float, float]
    end_pos: Tuple[float, float]
    frames_elapsed: int


@dataclass
class ShieldThresholdState:
    """State machine bookkeeping for the shield activation search."""

    port: int
    low_raw: int = 0
    high_raw: int = MAX_TRIGGER_RAW
    current_raw: int = 0
    current_unit: float = 0.0
    current_value: float = 0.0
    iterations: int = 0
    frames_remaining: int = 0
    settle_frames: int = 12
    tolerance: float = 0.002
    max_iterations: int = 12
    phase: str = "verify_low"
    manual: bool = False
    samples: List[Tuple[int, float, bool]] = field(default_factory=list)
    warmup_frames: int = 0


@dataclass
class ShieldDrainTestState:
    """Tracks per-value shield drain measurements."""

    port: int
    raw_values: List[int]
    current_index: int = 0
    stage: str = "refill"
    current_raw: Optional[int] = None
    current_unit: float = 0.0
    prev_strength: Optional[float] = None
    frames_pressed: int = 0
    recent_deltas: Deque[float] = field(
        default_factory=lambda: deque(maxlen=DELTA_STABILITY_FRAMES)
    )
    manual: bool = False
    wait_frames: int = 0


class GameLab:
    """Manages the Dolphin connection and provides automation helpers."""

    def __init__(
        self,
        *,
        console: Console,
        controllers: Dict[int, Controller],
        stage: Stage = Stage.FINAL_DESTINATION,
        character: Character = Character.FOX,
        verbose: bool = True,
        auto_scan: bool = True,
    ) -> None:
        self.console = console
        self.controllers = controllers
        self.stage = stage
        self.character = character
        self.verbose = verbose
        self.auto_scan = auto_scan

        self.menu_helper = MenuHelper()
        self.current_gamestate = None
        self.targets: Dict[int, ControllerTargets] = {
            port: ControllerTargets() for port in controllers
        }
        for port in controllers:
            self.set_shoulder(port, l=0.0, r=0.0)
        self.macros: Deque[Iterator[None]] = deque()
        self.shield_thresholds: Dict[int, float] = {}
        self.shield_threshold_raw: Dict[int, int] = {}
        self.shield_samples: Dict[int, List[Tuple[int, float, bool]]] = {}
        self._shield_test_state: Optional[ShieldThresholdState] = None
        self.shield_drain_results: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.shield_drain_samples: Dict[int, Dict[int, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._drain_test_state: Optional[ShieldDrainTestState] = None
        self.firefox_results: List[FirefoxTestResult] = []
        self._angle_queue: Deque[Tuple[int, Tuple[float, float], int, int, int]] = (
            deque()
        )
        self._current_params: Optional[
            Tuple[int, Tuple[float, float], int, int, int]
        ] = None
        self._auto_initialized = False
        self._auto_complete_logged = False
        self._current_angle: Optional[Tuple[float, float]] = None
        self._in_game = False
        self.primary_port, *rest = sorted(controllers.keys())
        self.secondary_port = rest[0] if rest else self.primary_port

    # ------------------------------------------------------------------
    # Controller state mutation APIs (callable from debuggers)
    # ------------------------------------------------------------------
    def set_main_stick(self, port: int, x: float, y: float) -> None:
        self.targets[port].main = (self._clip_unit(x), self._clip_unit(y))

    def set_c_stick(self, port: int, x: float, y: float) -> None:
        self.targets[port].c = (self._clip_unit(x), self._clip_unit(y))

    def set_shoulder(
        self, port: int, *, l: Optional[float] = None, r: Optional[float] = None
    ) -> None:
        if l is not None:
            _, amount = self._quantize_trigger(l)
            self.targets[port].l_shoulder = amount
        if r is not None:
            _, amount = self._quantize_trigger(r)
            self.targets[port].r_shoulder = amount

    def press_button(self, port: int, button: Button) -> None:
        self.targets[port].buttons[button] = True

    def release_button(self, port: int, button: Button) -> None:
        self.targets[port].buttons[button] = False

    def set_neutral(self, port: int) -> None:
        self.targets[port] = ControllerTargets()
        self.set_shoulder(port, l=0.0, r=0.0)

    def _quantize_trigger(self, amount: float) -> Tuple[int, float]:
        clipped = self._clip_shoulder(amount)
        raw = int(round(clipped * MAX_TRIGGER_RAW))
        raw = max(0, min(MAX_TRIGGER_RAW, raw))
        return raw, self._trigger_raw_to_amount(raw)

    def _trigger_raw_to_amount(self, raw: int) -> float:
        integer_raw = max(0, min(COMMAND_TRIGGER_MAX, int(raw)))
        if integer_raw <= MAX_TRIGGER_RAW:
            analog = fix_analog_trigger(integer_raw / MAX_TRIGGER_RAW)
        else:
            analog = integer_raw / COMMAND_TRIGGER_MAX
        return min(1.0, max(0.0, analog))

    def _apply_trigger_raw(self, port: int, raw: int, *, shoulder: str = "l") -> float:
        integer_raw = max(0, min(COMMAND_TRIGGER_MAX, int(raw)))
        amount = self._trigger_raw_to_amount(integer_raw)
        if shoulder == "l":
            self.targets[port].l_shoulder = amount
        else:
            self.targets[port].r_shoulder = amount
        return amount

    # ------------------------------------------------------------------
    # Macro helpers
    # ------------------------------------------------------------------
    def enqueue_macro(self, macro: Iterator[None]) -> None:
        """Schedule a coroutine-style macro (yields once per frame)."""
        self.macros.append(macro)

    def clear_macros(self) -> None:
        self.macros.clear()
        if self._current_params is not None:
            self._angle_queue.appendleft(self._current_params)
            self._current_params = None
            self._current_angle = None

    def wait(self, frames: int) -> Iterator[None]:
        for _ in range(max(0, frames)):
            yield

    def perform_firefox(
        self,
        port: int,
        angle: Tuple[float, float],
        *,
        charge_frames: int = DEFAULT_CHARGE_FRAMES,
        travel_frames: int = DEFAULT_TRAVEL_FRAMES,
        cooldown_frames: int = DEFAULT_COOLDOWN_FRAMES,
    ) -> None:
        """Queue a single Firefox test for the provided analog angle (unit domain)."""

        params = (
            port,
            (float(angle[0]), float(angle[1])),
            int(charge_frames),
            int(travel_frames),
            int(cooldown_frames),
        )
        self._angle_queue.appendleft(params)
        self._launch_next_angle()

    def _start_firefox(
        self,
        port: int,
        angle: Tuple[float, float],
        charge_frames: int,
        travel_frames: int,
        cooldown_frames: int,
    ) -> None:
        def _macro() -> Iterator[None]:
            self._ensure_ingame()
            original_main = self.targets[port].main
            original_buttons = dict(self.targets[port].buttons)
            yield from self._prepare_positions(test_port=port, angle=angle)
            start_pos = self.player_position(port)

            self.set_main_stick(port, 0.0, 1.0)
            self.press_button(port, Button.BUTTON_B)
            yield from self.wait(1)
            self.set_main_stick(port, *angle)
            yield from self.wait(charge_frames)
            self.release_button(port, Button.BUTTON_B)
            self.set_main_stick(port, 0.0, 0.0)
            yield from self.wait(travel_frames)
            end_pos = self.player_position(port)

            result = FirefoxTestResult(
                angle=tuple(float(v) for v in angle),
                start_pos=start_pos,
                end_pos=end_pos,
                frames_elapsed=charge_frames + travel_frames,
            )
            self.firefox_results.append(result)
            self._log(
                f"Firefox angle {result.angle}: start={result.start_pos} -> end={result.end_pos} in {result.frames_elapsed}f"
            )

            self.set_main_stick(port, *original_main)
            for button, pressed in original_buttons.items():
                if pressed:
                    self.press_button(port, button)
                else:
                    self.release_button(port, button)

            self._current_params = None
            self._current_angle = None
            yield from self.wait(cooldown_frames)
            self._launch_next_angle()

        self._current_params = (
            port,
            angle,
            charge_frames,
            travel_frames,
            cooldown_frames,
        )
        self._current_angle = angle
        self.enqueue_macro(_macro())

    def _launch_next_angle(self) -> None:
        if not self._in_game:
            return
        if self._current_params is not None or self.macros:
            return
        if not self._angle_queue:
            if (
                self.auto_scan
                and self._auto_initialized
                and not self._auto_complete_logged
            ):
                self._log("Automatic Firefox scan complete.")
                self._auto_complete_logged = True
            return
        params = self._angle_queue.popleft()
        port, angle, charge, travel, cooldown = params
        self._start_firefox(port, angle, charge, travel, cooldown)

    def scan_firefox_angles(
        self,
        port: int,
        angles: Iterable[Tuple[float, float]],
        *,
        charge_frames: int = 45,
        travel_frames: int = 120,
        cooldown_frames: int = 60,
    ) -> None:
        """Queue a batch of Firefox tests over the supplied stick angles."""
        self.clear_macros()
        self.firefox_results.clear()
        self._angle_queue.clear()
        self._current_params = None
        self._current_angle = None
        self._auto_initialized = True
        skipped = 0
        for angle in angles:
            if float(angle[1]) <= 0.0:
                skipped += 1
                continue
            self._angle_queue.append(
                (
                    port,
                    (float(angle[0]), float(angle[1])),
                    int(charge_frames),
                    int(travel_frames),
                    int(cooldown_frames),
                )
            )
        if not self._angle_queue:
            self._log("scan_firefox_angles: no upward angles supplied; nothing to do.")
            if skipped:
                self._log(f"scan_firefox_angles: skipped {skipped} downward angles.")
            return
        if skipped:
            self._log(f"scan_firefox_angles: skipped {skipped} downward angles.")
        self._launch_next_angle()

    def measure_shield_threshold(
        self,
        port: int,
        *,
        tolerance: float = 0.0000001,
        max_iterations: int = 12,
        settle_frames: int = 12,
        warmup_frames: int = 0,
    ) -> None:
        """Begin (or restart) a shield activation threshold search for the given port."""
        if port not in self.controllers:
            raise ValueError(f"Unknown controller port: {port}")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if settle_frames <= 0:
            raise ValueError("settle_frames must be positive")
        self._start_shield_test(
            port,
            tolerance=tolerance,
            max_iterations=max_iterations,
            settle_frames=settle_frames,
            manual=True,
            warmup_frames=warmup_frames,
        )

    def measure_shield_drain(
        self,
        port: int,
        *,
        start_raw: Optional[int] = None,
        end_raw: Optional[int] = None,
    ) -> None:
        self._start_shield_drain_test(
            port,
            start_raw=start_raw,
            end_raw=end_raw,
            manual=True,
        )

    def _start_shield_test(
        self,
        port: int,
        *,
        tolerance: float,
        max_iterations: int,
        settle_frames: int,
        manual: bool,
        warmup_frames: int = 0,
    ) -> None:
        self._ensure_ingame()
        tolerance = float(tolerance)
        max_iterations = max(1, int(max_iterations))
        settle_frames = max(1, int(settle_frames))
        warmup_frames = max(0, int(warmup_frames))
        previous_state = self._shield_test_state
        if previous_state is not None:
            if previous_state.port != port:
                self._log(
                    f"Interrupting shield threshold test on port {previous_state.port} to start a new test on port {port}."
                )
                self._apply_trigger_raw(previous_state.port, 0)
            else:
                self._log(f"Restarting shield threshold test on port {port}.")
        state = ShieldThresholdState(
            port=port,
            low_raw=0,
            high_raw=MAX_TRIGGER_RAW,
            current_raw=0,
            current_value=0.0,
            iterations=0,
            frames_remaining=settle_frames + warmup_frames,
            settle_frames=settle_frames,
            tolerance=tolerance,
            max_iterations=max_iterations,
            phase="verify_low",
            manual=manual,
            warmup_frames=warmup_frames,
        )
        self._shield_test_state = state
        self.shield_samples[port] = state.samples
        self.shield_thresholds.pop(port, None)
        self.shield_threshold_raw.pop(port, None)
        state.current_value = self._apply_trigger_raw(port, state.current_raw)
        state.current_unit = state.current_raw / MAX_TRIGGER_RAW

    def _advance_shield_test(self) -> None:
        state = self._shield_test_state
        if state is None:
            return
        if (
            not self.current_gamestate
            or state.port not in self.current_gamestate.players
        ):
            return
        player = self.current_gamestate.players[state.port]
        if state.frames_remaining > 0:
            state.frames_remaining -= 1
            return

        current_action = getattr(player, "action", None)
        if (
            getattr(player, "is_inactive", False)
            or getattr(player, "is_dead", False)
            or current_action in SPAWN_ACTIONS
        ):
            state.frames_remaining = max(state.frames_remaining, state.settle_frames)
            return

        shield_active = bool(player.is_shield_active)
        state.samples.append((state.current_raw, state.current_unit, shield_active))

        if state.phase == "verify_low":
            if shield_active:
                prefix = "Manual" if state.manual else "Automatic"
                result_unit = 0.0
                result_analog = self._trigger_raw_to_amount(0)
                self._finalize_shield_test(
                    state,
                    result_raw=0,
                    result_unit=result_unit,
                    message=(
                        f"{prefix} shield threshold for port {state.port}: raw=0, unit={result_unit:.6f}, "
                        f"analog={result_analog:.6f}; shield already active at zero L press."
                    ),
                )
                return
            state.low_raw = state.current_raw
            state.phase = "verify_high"
            state.current_raw = MAX_TRIGGER_RAW
            state.current_value = self._apply_trigger_raw(state.port, state.current_raw)
            state.current_unit = state.current_raw / MAX_TRIGGER_RAW
            state.frames_remaining = state.settle_frames
            return

        if state.phase == "verify_high":
            if not shield_active:
                self._finalize_shield_test(
                    state,
                    result_raw=None,
                    result_unit=None,
                    message=(
                        f"Shield threshold test failed on port {state.port}: shield did not activate at full L press."
                    ),
                )
                return
            state.high_raw = state.current_raw
            if state.high_raw - state.low_raw <= 1:
                result_raw = state.high_raw
                result_unit = result_raw / MAX_TRIGGER_RAW
                result_analog = self._trigger_raw_to_amount(result_raw)
                prefix = "Manual" if state.manual else "Automatic"
                self._finalize_shield_test(
                    state,
                    result_raw=result_raw,
                    result_unit=result_unit,
                    message=(
                        f"{prefix} shield threshold for port {state.port}: raw={result_raw}, "
                        f"unit={result_unit:.6f}, analog={result_analog:.6f} (verified at extremes)."
                    ),
                )
                return
            state.phase = "search"
            state.current_raw = (state.low_raw + state.high_raw) // 2
            if state.current_raw <= state.low_raw:
                state.current_raw = min(state.high_raw - 1, state.low_raw + 1)
            state.current_value = self._apply_trigger_raw(state.port, state.current_raw)
            state.current_unit = state.current_raw / MAX_TRIGGER_RAW
            state.frames_remaining = state.settle_frames
            return

        if shield_active:
            state.high_raw = min(state.high_raw, state.current_raw)
        else:
            state.low_raw = max(state.low_raw, state.current_raw)
        state.iterations += 1

        high_unit = state.high_raw / MAX_TRIGGER_RAW
        low_unit = state.low_raw / MAX_TRIGGER_RAW
        converged = (
            state.high_raw - state.low_raw <= 1
            or (high_unit - low_unit) <= state.tolerance
            or state.iterations >= state.max_iterations
        )
        if converged:
            result_raw = state.high_raw
            result_unit = high_unit
            result_analog = self._trigger_raw_to_amount(result_raw)
            prefix = "Manual" if state.manual else "Automatic"
            message = (
                f"{prefix} shield threshold for port {state.port}: raw={result_raw}, unit={result_unit:.6f}, "
                f"analog={result_analog:.6f} (iterations={state.iterations}, samples={len(state.samples)})."
            )
            self._finalize_shield_test(
                state,
                result_raw=result_raw,
                result_unit=result_unit,
                message=message,
            )
            return

        next_raw = (state.low_raw + state.high_raw) // 2
        if next_raw <= state.low_raw:
            next_raw = min(state.high_raw - 1, state.low_raw + 1)
        state.current_raw = next_raw
        state.current_value = self._apply_trigger_raw(state.port, state.current_raw)
        state.current_unit = state.current_raw / MAX_TRIGGER_RAW
        state.frames_remaining = state.settle_frames

    def _finalize_shield_test(
        self,
        state: ShieldThresholdState,
        *,
        result_raw: Optional[int],
        result_unit: Optional[float],
        message: str,
    ) -> None:
        port = state.port
        if result_raw is not None and result_unit is not None:
            self.shield_threshold_raw[port] = result_raw
            self.shield_thresholds[port] = result_unit
        else:
            self.shield_threshold_raw.pop(port, None)
            self.shield_thresholds.pop(port, None)
        self._shield_test_state = None
        self._apply_trigger_raw(port, 0)
        self._log(message)
        if not state.manual and result_raw is None and not self._auto_complete_logged:
            self._auto_complete_logged = True

    def _start_shield_drain_test(
        self,
        port: int,
        *,
        start_raw: Optional[int] = None,
        end_raw: Optional[int] = None,
        manual: bool = False,
    ) -> None:
        if port not in self.controllers:
            raise ValueError(f"Unknown controller port: {port}")
        if self._drain_test_state is not None:
            if manual:
                self._log(
                    "Shield drain test already in progress; ignoring manual request."
                )
            return

        def _coerce_raw(value: Optional[float]) -> Optional[int]:
            if value is None:
                return None
            val = float(value)
            if 0.0 <= val <= 1.0:
                return int(round(val * MAX_TRIGGER_RAW))
            return int(round(val))

        raw_start = _coerce_raw(start_raw)
        raw_end = _coerce_raw(end_raw)
        if raw_start is None:
            raw_start = 0
        if raw_end is None:
            raw_end = COMMAND_TRIGGER_MAX
        raw_start = max(0, min(COMMAND_TRIGGER_MAX, raw_start))
        raw_end = max(0, min(COMMAND_TRIGGER_MAX, raw_end))
        if raw_start > raw_end:
            raw_start, raw_end = raw_end, raw_start
        raw_end = 255
        raw_values = [
            val
            for val in range(raw_start, raw_end + 1)
            if 0 <= val <= COMMAND_TRIGGER_MAX
        ]
        if not raw_values:
            self._log(
                "Shield drain test: selected raw range has no values within [0, COMMAND_TRIGGER_MAX]; skipping."
            )
            return
        print(f"Shield drain test start: {raw_start}, end: {raw_end}")
        if not raw_values:
            self._log("Shield drain test: no raw values to evaluate; skipping.")
            return
        mode = "Manual" if manual else "Automatic"
        self._log(
            f"{mode} shield drain test starting on port {port} for raw values {raw_values[0]}-{raw_values[-1]}."
        )
        state = ShieldDrainTestState(port=port, raw_values=raw_values, manual=manual)
        state.stage = "refill"
        state.wait_frames = 5
        self._drain_test_state = state
        self.shield_drain_results[port] = {}
        self.shield_drain_samples[port] = {}
        self._apply_trigger_raw(port, 0)

    def _begin_shield_drain_value(
        self,
        state: ShieldDrainTestState,
        *,
        initial_strength: Optional[float] = None,
    ) -> None:
        if state.current_index >= len(state.raw_values):
            self._complete_shield_drain_test(state)
            return
        raw = state.raw_values[state.current_index]
        state.current_raw = raw
        state.current_unit = raw / MAX_TRIGGER_RAW if MAX_TRIGGER_RAW else 0.0
        state.prev_strength = initial_strength
        state.frames_pressed = 0
        state.recent_deltas = deque(maxlen=DELTA_STABILITY_FRAMES)
        self.shield_drain_samples[state.port][raw] = []
        state.stage = "press"
        analog = self._trigger_raw_to_amount(raw)
        self._log(
            f"Shield drain test: port {state.port} testing raw={raw} (unit={state.current_unit:.6f}, analog={analog:.6f})."
        )
        self._apply_trigger_raw(state.port, raw)

    def _advance_shield_drain(self) -> None:
        state = self._drain_test_state
        if state is None:
            return
        if (
            not self.current_gamestate
            or state.port not in self.current_gamestate.players
        ):
            return
        player = self.current_gamestate.players[state.port]
        strength = float(getattr(player, "shield_strength", SHIELD_FULL_VALUE))
        shield_active = bool(getattr(player, "is_shield_active", False))

        if state.stage == "refill":
            self._apply_trigger_raw(state.port, 0)
            if state.wait_frames > 0:
                state.wait_frames -= 1
                state.prev_strength = strength
                return
            if strength >= SHIELD_FULL_VALUE - SHIELD_FULL_TOLERANCE:
                state.prev_strength = strength
                self._begin_shield_drain_value(state, initial_strength=strength)
            else:
                state.prev_strength = strength
            return

        if state.stage != "press" or state.current_raw is None:
            return

        self._apply_trigger_raw(state.port, state.current_raw)
        if not shield_active:
            state.prev_strength = strength
            return

        if state.prev_strength is None:
            state.prev_strength = strength
            return

        delta = state.prev_strength - strength
        state.prev_strength = strength
        if delta < 0.0:
            delta = 0.0
        state.frames_pressed += 1

        samples = self.shield_drain_samples[state.port].setdefault(
            state.current_raw, []
        )
        samples.append(delta)
        state.recent_deltas.append(delta)

        stable = (
            len(state.recent_deltas) == DELTA_STABILITY_FRAMES
            and (max(state.recent_deltas) - min(state.recent_deltas))
            <= DELTA_STABILITY_TOLERANCE
        )
        limit_reached = (
            state.frames_pressed >= MAX_DRAIN_PRESS_FRAMES or strength <= 0.0
        )

        if not stable and not limit_reached:
            return

        rate = (
            sum(state.recent_deltas) / len(state.recent_deltas)
            if state.recent_deltas
            else 0.0
        )
        analog = self._trigger_raw_to_amount(state.current_raw)
        mode = "Manual" if state.manual else "Automatic"
        self.shield_drain_results[state.port][state.current_raw] = rate
        self._log(
            f"{mode} shield drain rate for port {state.port}: raw={state.current_raw}, "
            f"unit={state.current_unit:.6f}, analog={analog:.6f}, rate={rate:.6f} (frames={state.frames_pressed})."
        )

        state.current_index += 1
        state.stage = "refill"
        state.prev_strength = strength
        state.recent_deltas = deque(maxlen=DELTA_STABILITY_FRAMES)
        state.frames_pressed = 0
        state.wait_frames = 5
        self._apply_trigger_raw(state.port, 0)

        if state.current_index >= len(state.raw_values):
            self._complete_shield_drain_test(state)
            return

    def _complete_shield_drain_test(self, state: ShieldDrainTestState) -> None:
        port = state.port
        mode = "Manual" if state.manual else "Automatic"
        self._drain_test_state = None
        self._apply_trigger_raw(port, 0)
        tested = len(self.shield_drain_results.get(port, {}))
        self._log(
            f"{mode} shield drain test complete for port {port} ({tested} raw values)."
        )
        if not state.manual:
            self._auto_complete_logged = True

    # ------------------------------------------------------------------
    # Runtime loop
    # ------------------------------------------------------------------
    def spin(self) -> None:
        """Main loop. Blocks forever until interrupted."""
        while True:
            gamestate = self.console.step()
            if gamestate is None:
                continue
            self.current_gamestate = gamestate
            if gamestate.menu_state in (Menu.IN_GAME, Menu.SUDDEN_DEATH):
                if not self._in_game:
                    self._in_game = True
                    self._log("Entered match; automation enabled.")
                self._maybe_start_auto_scan()
                self._process_macros()
                self._apply_targets()
            else:
                if self._in_game:
                    self._log(f"Exited match (state={gamestate.menu_state.name}).")
                    self._in_game = False
                    self.clear_macros()
                    if self._shield_test_state is not None:
                        self._log(
                            f"Shield threshold test aborted early (match ended) for port {self._shield_test_state.port}."
                        )
                        self.set_shoulder(self._shield_test_state.port, l=0.0)
                        self._shield_test_state = None
                    if self._drain_test_state is not None:
                        self._log(
                            f"Shield drain test aborted early (match ended) for port {self._drain_test_state.port}."
                        )
                        self._apply_trigger_raw(self._drain_test_state.port, 0)
                        self._drain_test_state = None
                self._drive_menus(gamestate)

    def shutdown(self) -> None:
        """Release controllers and stop Dolphin."""
        for controller in self.controllers.values():
            try:
                controller.release_all()
                controller.disconnect()
            except Exception:
                pass
        try:
            self.console.stop()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _drive_menus(self, gamestate) -> None:
        self.menu_helper.menu_helper_simple(
            gamestate,
            self.controllers[self.primary_port],
            self.character,
            self.stage,
            costume=1,
            autostart=False,
            swag=False,
        )
        if self.secondary_port in self.controllers:
            self.menu_helper.choose_character(
                character=self.character,
                gamestate=gamestate,
                controller=self.controllers[self.secondary_port],
                cpu_level=0,
                costume=1,
                swag=False,
                start=True,
            )

    def _process_macros(self) -> None:
        if not self.macros:
            return
        macro = self.macros[0]
        try:
            next(macro)
        except StopIteration:
            self.macros.popleft()

    def _apply_targets(self) -> None:
        for port, controller in self.controllers.items():
            target = self.targets[port]
            main_x = fix_analog_stick((self._clip_unit(target.main[0]) + 1.0) / 2.0)
            main_y = fix_analog_stick((self._clip_unit(target.main[1]) + 1.0) / 2.0)
            controller.tilt_analog(Button.BUTTON_MAIN, main_x, main_y)
            c_x = fix_analog_stick((self._clip_unit(target.c[0]) + 1.0) / 2.0)
            c_y = fix_analog_stick((self._clip_unit(target.c[1]) + 1.0) / 2.0)
            controller.tilt_analog(Button.BUTTON_C, c_x, c_y)
            controller.press_shoulder(Button.BUTTON_L, target.l_shoulder)
            controller.press_shoulder(Button.BUTTON_R, target.r_shoulder)
            for button, pressed in target.buttons.items():
                if pressed:
                    controller.press_button(button)
                else:
                    controller.release_button(button)

    def _clip_unit(self, value: float) -> float:
        return float(max(-1.0, min(1.0, value)))

    def _clip_shoulder(self, value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    def player_position(self, port: int) -> Tuple[float, float]:
        if not self.current_gamestate or port not in self.current_gamestate.players:
            return (float("nan"), float("nan"))
        pos = self.current_gamestate.players[port].position
        return (float(pos.x), float(pos.y))

    def _log(self, message: str) -> None:
        if not self.verbose:
            return
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{ts}][lab] {message}")

    def _ensure_ingame(self) -> None:
        if not self._in_game:
            raise RuntimeError("Requested action requires an active match.")

    def _maybe_start_auto_scan(self) -> None:
        if not self._in_game:
            return
        if self._shield_test_state is not None:
            if self._current_params is not None or self.macros:
                return
            self._advance_shield_test()
            return
        if self._drain_test_state is not None:
            if self._current_params is not None or self.macros:
                return
            self._advance_shield_drain()
            return
        if not self.auto_scan:
            return
        if self._current_params is not None or self.macros:
            return
        if not self._auto_initialized:
            self._auto_initialized = True
            self._auto_complete_logged = False
            self._log(
                f"Starting automatic shield threshold test on port {self.primary_port}."
            )
            self._start_shield_test(
                self.primary_port,
                tolerance=0.0000001,
                max_iterations=12,
                settle_frames=12,
                manual=False,
                warmup_frames=180,
            )
            self._advance_shield_test()
            return
        threshold_raw = self.shield_threshold_raw.get(self.primary_port)
        if (
            threshold_raw is not None
            and not self._auto_complete_logged
            and self._drain_test_state is None
        ):
            self._start_shield_drain_test(
                self.primary_port,
                start_raw=threshold_raw,
                end_raw=COMMAND_TRIGGER_MAX,
                manual=False,
            )
            self._advance_shield_drain()

    def _prepare_positions(
        self,
        *,
        test_port: int,
        angle: Tuple[float, float],
        epsilon: float = 0.002,
        max_frames: int = 240,
    ) -> Iterator[None]:
        """Move the test character to center and other characters to the stage edge."""
        edge = float(stages.EDGE_POSITION[self.stage])
        targets: Dict[int, float] = {test_port: 0.0}
        other_ports = [p for p in self.controllers.keys() if p != test_port]
        if other_ports:
            base_sign = -1.0 if angle[0] >= 0 else 1.0
            for idx, port in enumerate(other_ports):
                sign = base_sign if idx % 2 == 0 else -base_sign
                targets[port] = sign * (edge - epsilon)

        settle_counts: Dict[int, int] = {port: 0 for port in targets}
        neutral_cooldown: Dict[int, int] = {port: 0 for port in targets}
        last_direction: Dict[int, float] = {port: 0.0 for port in targets}
        phase: Dict[int, str] = {port: "drive" for port in targets}
        frames = 0
        aligned = False
        while frames < max_frames:
            frames += 1
            all_done = True
            for port, target_x in targets.items():
                pos = self.player_position(port)
                if math.isnan(pos[0]):
                    settle_counts[port] = 0
                    phase[port] = "drive"
                    all_done = False
                    continue
                delta = target_x - pos[0]
                if phase[port] == "coast":
                    self.set_main_stick(port, 0.0, 0.0)
                    if abs(delta) <= epsilon:
                        settle_counts[port] = min(
                            settle_counts[port] + 1, NEUTRAL_SETTLE_FRAMES
                        )
                    else:
                        settle_counts[port] = 0
                    if abs(delta) > 1.0:
                        phase[port] = "drive"
                        neutral_cooldown[port] = NEUTRAL_SETTLE_FRAMES
                        last_direction[port] = 0.0
                        all_done = False
                        continue
                    if settle_counts[port] >= NEUTRAL_SETTLE_FRAMES:
                        continue
                    all_done = False
                    continue

                if abs(delta) <= epsilon:
                    self.set_main_stick(port, 0.0, 0.0)
                    settle_counts[port] = min(
                        settle_counts[port] + 1, NEUTRAL_SETTLE_FRAMES
                    )
                    last_direction[port] = 0.0
                    neutral_cooldown[port] = max(neutral_cooldown[port] - 1, 0)
                    phase[port] = "coast"
                    continue
                settle_counts[port] = 0
                all_done = False
                if abs(delta) <= 1.0:
                    phase[port] = "coast"
                    self.set_main_stick(port, 0.0, 0.0)
                    last_direction[port] = 0.0
                    neutral_cooldown[port] = NEUTRAL_SETTLE_FRAMES
                    continue
                direction = 1.0 if delta > 0 else -1.0
                if last_direction[port] != 0.0 and direction != last_direction[port]:
                    self.set_main_stick(port, 0.0, 0.0)
                    neutral_cooldown[port] = NEUTRAL_SETTLE_FRAMES
                    last_direction[port] = 0.0
                    continue
                if neutral_cooldown[port] > 0:
                    self.set_main_stick(port, 0.0, 0.0)
                    neutral_cooldown[port] -= 1
                    continue
                magnitude = min(1.0, max(MIN_MOVEMENT_MAG, abs(delta) / 12.0))
                if abs(delta) > 20.0:
                    magnitude = 1.0
                self.set_main_stick(port, direction * magnitude, 0.0)
                last_direction[port] = direction
            if all_done:
                aligned = True
                break
            yield

        for port in targets:
            self.set_main_stick(port, 0.0, 0.0)
        if not aligned:
            positions = {port: self.player_position(port)[0] for port in targets}
            self._log(
                "Warning: failed to fully align characters before test; continuing anyway. "
                + ", ".join(f"p{port}={pos:.3f}" for port, pos in positions.items())
            )
        yield from self.wait(6)


# ----------------------------------------------------------------------
# Convenience wrappers exposed at module level for debugger ergonomics
# ----------------------------------------------------------------------
LAB: Optional[GameLab] = None


def ensure_lab() -> GameLab:
    if LAB is None:
        raise RuntimeError("LAB has not been initialised. Run dolphin_lab.py first.")
    return LAB


def set_main_stick(port: int, x: float, y: float) -> None:
    ensure_lab().set_main_stick(port, x, y)


def set_c_stick(port: int, x: float, y: float) -> None:
    ensure_lab().set_c_stick(port, x, y)


def set_shoulder(
    port: int, *, l: Optional[float] = None, r: Optional[float] = None
) -> None:
    ensure_lab().set_shoulder(port, l=l, r=r)


def press_button(port: int, button: Button) -> None:
    ensure_lab().press_button(port, button)


def release_button(port: int, button: Button) -> None:
    ensure_lab().release_button(port, button)


def scan_firefox_angles(
    port: int,
    angles: Iterable[Tuple[float, float]],
    *,
    charge_frames: int = DEFAULT_CHARGE_FRAMES,
    travel_frames: int = DEFAULT_TRAVEL_FRAMES,
    cooldown_frames: int = DEFAULT_COOLDOWN_FRAMES,
) -> None:
    ensure_lab().scan_firefox_angles(
        port,
        angles,
        charge_frames=charge_frames,
        travel_frames=travel_frames,
        cooldown_frames=cooldown_frames,
    )


def measure_shield_threshold(
    port: int,
    *,
    tolerance: float = 0.0000001,
    max_iterations: int = 12,
    settle_frames: int = 12,
    warmup_frames: int = 0,
) -> None:
    ensure_lab().measure_shield_threshold(
        port,
        tolerance=tolerance,
        max_iterations=max_iterations,
        settle_frames=settle_frames,
        warmup_frames=warmup_frames,
    )


def measure_shield_drain(
    port: int,
    *,
    start_raw: Optional[int] = None,
    end_raw: Optional[int] = None,
) -> None:
    ensure_lab().measure_shield_drain(
        port,
        start_raw=start_raw,
        end_raw=end_raw,
    )


def perform_firefox(
    port: int,
    angle: Tuple[float, float],
    *,
    charge_frames: int = DEFAULT_CHARGE_FRAMES,
    travel_frames: int = DEFAULT_TRAVEL_FRAMES,
    cooldown_frames: int = DEFAULT_COOLDOWN_FRAMES,
) -> None:
    ensure_lab().perform_firefox(
        port,
        angle,
        charge_frames=charge_frames,
        travel_frames=travel_frames,
        cooldown_frames=cooldown_frames,
    )


def run_wait(frames: int) -> None:
    """Convenience helper to enqueue a simple wait."""
    ensure_lab().enqueue_macro(ensure_lab().wait(frames))


def player_position(port: int) -> Tuple[float, float]:
    return ensure_lab().player_position(port)


__all__ = [
    "LAB",
    "CONTROL_PALETTE",
    "FirefoxTestResult",
    "GameLab",
    "measure_shield_threshold",
    "measure_shield_drain",
    "perform_firefox",
    "player_position",
    "press_button",
    "release_button",
    "run_wait",
    "scan_firefox_angles",
    "set_c_stick",
    "set_main_stick",
    "set_shoulder",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Dolphin lab harness.")
    parser.add_argument(
        "--address",
        "-a",
        default="127.0.0.1",
        help="IP address of the Slippi/Wii instance.",
    )
    parser.add_argument(
        "--dolphin-executable-path",
        "-e",
        default=None,
        help="Path to the Dolphin executable.",
    )
    parser.add_argument("--iso", default=None, help="Path to the GALE01 Melee ISO.")
    parser.add_argument(
        "--save-replays",
        action="store_true",
        help="Persist Slippi replays (disabled by default).",
    )
    parser.add_argument(
        "--no-verbose", action="store_true", help="Silence status logs."
    )
    parser.add_argument(
        "--no-auto-scan",
        action="store_true",
        help="Disable automatic shield threshold test when the match starts.",
    )
    return parser.parse_args()


def main() -> None:
    init_config()
    args = _parse_args()

    console = Console(
        path=args.dolphin_executable_path,
        slippi_address=args.address,
        save_replays=args.save_replays,
        copy_home_directory=False,
        tmp_home_directory=False,
        blocking_input=True,
    )

    controllers = {
        1: Controller(
            console=console,
            port=1,
            type=ControllerType.STANDARD,
            fix_analog_inputs=False,
        ),
        2: Controller(
            console=console,
            port=2,
            type=ControllerType.STANDARD,
            fix_analog_inputs=False,
        ),
    }

    def _cleanup(*_args: object) -> None:
        if "LAB" in globals() and LAB is not None:
            LAB.shutdown()
        else:
            for controller in controllers.values():
                try:
                    controller.release_all()
                    controller.disconnect()
                except Exception:
                    pass
            try:
                console.stop()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)

    console.run(iso_path=args.iso)
    if not console.connect():
        print("ERROR: Failed to connect to console.")
        sys.exit(-1)
    for controller in controllers.values():
        if not controller.connect():
            print(f"ERROR: Failed to connect controller {controller.port}.")
            sys.exit(-1)
        controller.release_all()

    global LAB
    LAB = GameLab(
        console=console,
        controllers=controllers,
        stage=Stage.FINAL_DESTINATION,
        character=Character.FOX,
        verbose=not args.no_verbose,
        auto_scan=not args.no_auto_scan,
    )
    LAB._log(
        "Lab initialised. Attach a debugger and use helper functions (e.g. measure_shield_threshold)."
    )
    if LAB.auto_scan:
        LAB._log("Automatic shield threshold test will begin once the match loads.")
    else:
        LAB._log(
            "Automatic shield threshold test disabled (call measure_shield_threshold manually)."
        )
    try:
        LAB.spin()
    finally:
        LAB.shutdown()


if __name__ == "__main__":
    main()
