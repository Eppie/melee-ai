#!/usr/bin/env python3
"""
Utility script for launching Dolphin/Slippi in a controlled lab environment.

The script boots into Final Destination with two Fox players (ports 1 & 2)
both driven entirely by code.  It is designed for interactive experimentation:
attach a debugger, grab the global ``LAB`` object, and schedule controller
commands or measurement routines (e.g. testing custom movement macros).

Example usage:

    $ python dolphin_lab.py --iso /path/to/GALE01.iso --dolphin-executable-path /path/to/dolphin

Inside a debugger / REPL:

    >>> from dolphin_lab import LAB
"""

from __future__ import annotations

import argparse
import csv
import math
import signal
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Deque, Dict, Iterable, Iterator, List, Optional, Set, Tuple

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
NEUTRAL_SETTLE_FRAMES = 4
MAX_TRIGGER_RAW = 140
COMMAND_TRIGGER_MAX = 255
DELTA_STABILITY_FRAMES = 5
DELTA_STABILITY_TOLERANCE = 1e-4
MAX_DRAIN_PRESS_FRAMES = 180
SPAWN_ACTIONS = {Action.ON_HALO_DESCENT, Action.ON_HALO_WAIT}
EXCLUDED_ACTION_LOG_STATES: Set[Action] = {Action.STANDING, *SPAWN_ACTIONS}
NEUTRAL_COMPLETE_ACTIONS: Set[Action] = {
    Action.STANDING,
    Action.CROUCHING,
    Action.CROUCH_END,
    Action.LANDING,
    Action.LANDING_SPECIAL,
}
MAX_MOVE_CAPTURE_FRAMES = 1200


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
        action_log_path: Optional[Path] = None,
        enable_action_sweep: bool = True,
        iso_path: Optional[Path] = None,
    ) -> None:
        self.console = console
        self.controllers = controllers
        self.stage = stage
        self.character = character
        self.verbose = verbose
        self.action_log_path = action_log_path
        self.enable_action_sweep = enable_action_sweep
        self.iso_path = iso_path

        self.menu_helper = MenuHelper()
        self.current_gamestate = None
        self.targets: Dict[int, ControllerTargets] = {
            port: ControllerTargets() for port in controllers
        }
        for port in controllers:
            self.set_shoulder(port, l=0.0, r=0.0)
        self.macros: Deque[Iterator[None]] = deque()
        self.firefox_results: List[FirefoxTestResult] = []
        self._angle_queue: Deque[
            Tuple[int, Tuple[float, float], int, int, int]
        ] = deque()
        self._current_params: Optional[
            Tuple[int, Tuple[float, float], int, int, int]
        ] = None
        self._auto_initialized = False
        self._auto_complete_logged = False
        self._current_angle: Optional[Tuple[float, float]] = None
        self._in_game = False
        self.primary_port, *rest = sorted(controllers.keys())
        self.secondary_port = rest[0] if rest else self.primary_port
        self.action_state_logs: List[Tuple[str, List[str]]] = []
        self._action_log_file = None
        self._action_log_writer: Optional[csv.writer] = None
        self._action_sweep_characters: List[Character] = []
        self._action_sweep_index = 0
        self._action_sweep_run_in_match = False

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
            if self._auto_initialized and not self._auto_complete_logged:
                self._log("Automatic Firefox scan complete.")
            self._auto_complete_logged = True
            return
        params = self._angle_queue.popleft()
        port, angle, charge, travel, cooldown = params
        self._start_firefox(port, angle, charge, travel, cooldown)

    def enqueue_action_state_capture(
        self,
        *,
        name: str,
        move_inputs: Callable[[int], Iterator[None]],
        character: Character,
        aerial_or_grounded: str,
        port: Optional[int] = None,
        exclude: Optional[Set[Action]] = None,
        settle_frames: int = 6,
        cooldown_frames: int = 20,
    ) -> None:
        """Log action transitions while performing a move macro."""

        target_port = self.primary_port if port is None else port
        excluded_actions = exclude

        def _macro() -> Iterator[None]:
            self._ensure_ingame()
            yield from self._wait_until_player_active(target_port)
            self.set_neutral(target_port)
            yield from self.wait(settle_frames)

            # Store initial position and facing
            initial_pos = self.player_position(target_port)
            initial_facing = (
                self.current_gamestate.players[target_port].facing
                if self._player_present(target_port)
                else True
            )

            frame = 0
            move_iter = move_inputs(target_port)
            move_iter_exhausted = False
            started_move = False
            seen_nonneutral = False
            while True:
                if not self._player_present(target_port):
                    self.set_neutral(target_port)
                    yield
                    continue
                player = self.current_gamestate.players[target_port]
                if not self._player_is_active(target_port):
                    self.set_neutral(target_port)
                    yield
                    continue
                if not started_move:
                    try:
                        next(move_iter)
                    except StopIteration:
                        move_iter_exhausted = True
                    started_move = True
                elif not move_iter_exhausted:
                    try:
                        next(move_iter)
                    except StopIteration:
                        move_iter_exhausted = True
                action = player.action
                if excluded_actions is None or action not in excluded_actions:
                    self._log_action_frame(
                        character=character,
                        aerial_or_grounded=aerial_or_grounded,
                        move=name,
                        frame=frame,
                        action=action,
                    )
                if action not in EXCLUDED_ACTION_LOG_STATES:
                    seen_nonneutral = True
                if seen_nonneutral and self._move_is_complete(player):
                    break
                if move_iter_exhausted and frame >= MAX_MOVE_CAPTURE_FRAMES:
                    self._log(
                        f"Move capture timeout after {MAX_MOVE_CAPTURE_FRAMES} frames for {name} ({aerial_or_grounded}); breaking."
                    )
                    break
                frame += 1
                yield
            self.set_neutral(target_port)
            yield from self.wait(cooldown_frames)

            # Ensure character has returned to initial position and facing
            yield from self._return_to_position(target_port, initial_pos)
            yield from self._return_to_facing(target_port, initial_facing)

        self.enqueue_macro(_macro())

    def queue_default_action_state_demo(self, port: Optional[int] = None) -> None:
        """Queue Fox B-move action-state logger macros."""

        target_port = self.primary_port if port is None else port
        self.clear_macros()
        self.enqueue_macro(self.wait(180))  # allow spawn/intro frames to finish
        moves: List[Tuple[str, str, Callable[[int], Iterator[None]]]] = [
            ("neutral B", "grounded", self._fox_neutral_b_macro),
            ("up B", "grounded", self._fox_up_b_macro),
            ("side B", "grounded", self._fox_side_b_macro),
            ("down B", "grounded", self._fox_down_b_macro),
            ("neutral B", "aerial", self._fox_neutral_b_air_macro),
            ("up B", "aerial", self._fox_up_b_air_macro),
            ("side B", "aerial", self._fox_side_b_air_macro),
            ("down B", "aerial", self._fox_down_b_air_macro),
        ]
        for move_name, variant, macro in moves:
            self.enqueue_action_state_capture(
                name=move_name,
                character=self.character,
                aerial_or_grounded=variant,
                move_inputs=macro,
                port=target_port,
            )

    def _fox_up_b_macro(self, port: int) -> Iterator[None]:
        yield from self.wait(1)
        self.set_main_stick(port, 0.0, 1.0)
        self.press_button(port, Button.BUTTON_B)
        yield
        self.release_button(port, Button.BUTTON_B)
        yield from self.wait(DEFAULT_CHARGE_FRAMES)
        self.set_main_stick(port, 0.0, 1.0)
        yield from self.wait(DEFAULT_TRAVEL_FRAMES)
        self.set_main_stick(port, 0.0, 0.0)

    def _fox_side_b_macro(self, port: int) -> Iterator[None]:
        yield from self.wait(1)
        self.set_main_stick(port, 1.0, 0.0)
        self.press_button(port, Button.BUTTON_B)
        yield
        self.release_button(port, Button.BUTTON_B)
        yield from self.wait(DEFAULT_TRAVEL_FRAMES // 2)
        self.set_main_stick(port, 0.0, 0.0)

    def _fox_down_b_macro(self, port: int) -> Iterator[None]:
        yield from self.wait(1)
        self.set_main_stick(port, 0.0, -1.0)
        self.press_button(port, Button.BUTTON_B)
        yield
        self.release_button(port, Button.BUTTON_B)
        yield from self.wait(DEFAULT_CHARGE_FRAMES)
        self.set_main_stick(port, 0.0, 0.0)

    def _fox_neutral_b_macro(self, port: int) -> Iterator[None]:
        yield from self.wait(1)
        self.set_main_stick(port, 0.0, 0.0)
        self.press_button(port, Button.BUTTON_B)
        yield
        self.release_button(port, Button.BUTTON_B)
        yield from self.wait(20)

    def _short_hop(self, port: int, airborne_wait: int = 4) -> Iterator[None]:
        """Light jump to reach airborne state before a move."""

        self.press_button(port, Button.BUTTON_Y)
        yield
        self.release_button(port, Button.BUTTON_Y)
        yield from self.wait(airborne_wait)

    def _full_hop(self, port: int, airborne_wait: int = 4) -> Iterator[None]:
        """Full height jump to reach airborne state before a move."""

        self.press_button(port, Button.BUTTON_Y)
        yield from self.wait(4)
        self.release_button(port, Button.BUTTON_Y)
        yield from self.wait(airborne_wait)

    def _fox_up_b_air_macro(self, port: int) -> Iterator[None]:
        yield from self.wait(1)
        yield from self._full_hop(port, airborne_wait=12)
        self.set_main_stick(port, 0.0, 1.0)
        self.press_button(port, Button.BUTTON_B)
        yield
        self.release_button(port, Button.BUTTON_B)
        yield from self.wait(DEFAULT_CHARGE_FRAMES)
        self.set_main_stick(port, 0.0, 1.0)
        yield from self.wait(DEFAULT_TRAVEL_FRAMES)
        self.set_main_stick(port, 0.0, 0.0)

    def _fox_side_b_air_macro(self, port: int) -> Iterator[None]:
        yield from self.wait(1)
        yield from self._full_hop(port, airborne_wait=12)
        self.set_main_stick(port, 1.0, 0.0)
        self.press_button(port, Button.BUTTON_B)
        yield
        self.release_button(port, Button.BUTTON_B)
        yield from self.wait(DEFAULT_TRAVEL_FRAMES // 2)
        self.set_main_stick(port, 0.0, 0.0)

    def _fox_down_b_air_macro(self, port: int) -> Iterator[None]:
        yield from self.wait(1)
        yield from self._full_hop(port, airborne_wait=12)
        self.set_main_stick(port, 0.0, -1.0)
        self.press_button(port, Button.BUTTON_B)
        yield
        self.release_button(port, Button.BUTTON_B)
        yield from self.wait(DEFAULT_CHARGE_FRAMES)
        self.set_main_stick(port, 0.0, 0.0)

    def _fox_neutral_b_air_macro(self, port: int) -> Iterator[None]:
        yield from self.wait(1)
        yield from self._full_hop(port, airborne_wait=12)
        self.set_main_stick(port, 0.0, 0.0)
        self.press_button(port, Button.BUTTON_B)
        yield
        self.release_button(port, Button.BUTTON_B)
        yield from self.wait(20)

    def _all_playable_characters(self) -> List[Character]:
        blocked = {
            Character.WIREFRAME_MALE,
            Character.WIREFRAME_FEMALE,
            Character.GIGA_BOWSER,
            Character.SANDBAG,
            Character.UNKNOWN_CHARACTER,
            Character.NANA,
        }
        return [char for char in Character if char not in blocked]

    def start_action_state_sweep(self) -> None:
        if not self.enable_action_sweep:
            return
        self._action_sweep_characters = self._all_playable_characters()
        self._action_sweep_index = 0
        self._action_sweep_run_in_match = False
        if not self._action_sweep_characters:
            self._log("No playable characters found for action sweep; disabling.")
            self.enable_action_sweep = False
            return
        self.character = self._action_sweep_characters[0]
        self._log(
            f"Action state sweep initialised; starting with {self._human_character_name(self.character)}."
        )

    def _enqueue_moves_for_current_character(self) -> None:
        if self._action_sweep_index >= len(self._action_sweep_characters):
            self.enable_action_sweep = False
            return
        character = self._action_sweep_characters[self._action_sweep_index]
        moves: List[Tuple[str, str, Callable[[int], Iterator[None]]]] = [
            ("neutral B", "grounded", self._fox_neutral_b_macro),
            ("up B", "grounded", self._fox_up_b_macro),
            ("side B", "grounded", self._fox_side_b_macro),
            ("down B", "grounded", self._fox_down_b_macro),
            ("neutral B", "aerial", self._fox_neutral_b_air_macro),
            ("up B", "aerial", self._fox_up_b_air_macro),
            ("side B", "aerial", self._fox_side_b_air_macro),
            ("down B", "aerial", self._fox_down_b_air_macro),
        ]
        self.enqueue_macro(self.wait(180))
        for move_name, variant, macro in moves:
            self.enqueue_action_state_capture(
                name=move_name,
                move_inputs=macro,
                character=character,
                aerial_or_grounded=variant,
                port=self.primary_port,
            )
        self.enqueue_macro(self._complete_character_and_reset_macro(self.primary_port))

    def _complete_character_and_reset_macro(self, port: int) -> Iterator[None]:
        self._advance_action_sweep_character()
        if not self.enable_action_sweep:
            self._log("Action state sweep complete for all characters.")
            yield from self.wait(60)
            return
        success = self._restart_emulator()
        if not success:
            self._log("Restart failed; aborting action sweep.")
            self.enable_action_sweep = False
            yield from self.wait(60)
            return
        yield from self.wait(300)

    def _advance_action_sweep_character(self) -> None:
        self._action_sweep_index += 1
        if self._action_sweep_index < len(self._action_sweep_characters):
            next_char = self._action_sweep_characters[self._action_sweep_index]
            self.character = next_char
            self._action_sweep_run_in_match = False
            self._log(
                f"Advancing to next character: {self._human_character_name(next_char)}."
            )
        else:
            self.enable_action_sweep = False

    def _reset_to_css_macro(self, port: int) -> Iterator[None]:
        self.set_neutral(port)
        yield from self.wait(10)
        self._log("Reset: pausing (Start).")
        # Pause first.
        self.press_button(port, Button.BUTTON_START)
        yield from self.wait(10)
        self.release_button(port, Button.BUTTON_START)
        yield from self.wait(12)
        # Hold L+R+A+Start while paused to return to CSS.
        combo_buttons = (
            Button.BUTTON_L,
            Button.BUTTON_R,
            Button.BUTTON_A,
            Button.BUTTON_START,
        )
        self._log("Reset: holding L+R+A+Start.")
        for button in combo_buttons:
            self.press_button(port, button)
        yield from self.wait(120)
        self._log("Reset: releasing L+R+A+Start.")
        for button in combo_buttons:
            self.release_button(port, button)
        self.set_neutral(port)
        # Wait until we leave the match (or timeout), then allow menus to settle.
        max_wait = 900
        waited = 0
        while (
            self.current_gamestate
            and self.current_gamestate.menu_state in (Menu.IN_GAME, Menu.SUDDEN_DEATH)
            and waited < max_wait
        ):
            waited += 1
            yield
        if waited >= max_wait:
            self._log("Reset: timeout waiting to exit match; continuing anyway.")
        else:
            self._log(
                f"Reset: detected exit to menu ({self.current_gamestate.menu_state.name if self.current_gamestate else 'unknown'})."
            )
        yield from self.wait(180)

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

    # ------------------------------------------------------------------
    # Runtime loop
    # ------------------------------------------------------------------
    def spin(self) -> None:
        """Main loop. Blocks forever until interrupted."""
        while True:
            gamestate = self.console.step()
            if gamestate is not None:
                self.current_gamestate = gamestate
            current_menu = (
                self.current_gamestate.menu_state
                if self.current_gamestate
                else Menu.UNKNOWN_MENU
            )
            in_match = current_menu in (Menu.IN_GAME, Menu.SUDDEN_DEATH)
            if gamestate is not None:
                if in_match and not self._in_game:
                    self._in_game = True
                    self._log("Entered match; automation enabled.")
                    self._action_sweep_run_in_match = False
                if not in_match and self._in_game:
                    self._log(f"Exited match (state={current_menu.name}).")
                    self._in_game = False
                    self._action_sweep_run_in_match = False
                    for port in self.controllers:
                        self.set_neutral(port)
                    if not self.enable_action_sweep:
                        self.clear_macros()

            if in_match:
                self._maybe_start_action_sweep()
                self._maybe_start_auto_scan()
            self._process_macros()
            self._apply_targets()
            if not in_match and self.current_gamestate and not self.macros:
                self._drive_menus(self.current_gamestate)

    def shutdown(self) -> None:
        """Release controllers and stop Dolphin."""
        for controller in self.controllers.values():
            try:
                controller.release_all()
                controller.disconnect()
            except Exception:
                pass
        if self._action_log_file is not None:
            try:
                self._action_log_file.close()
            except Exception:
                pass
        try:
            self.console.stop()
        except Exception:
            pass

    def _restart_emulator(self) -> bool:
        """Stop Dolphin and start a fresh instance."""

        self._log("Restarting Dolphin for next character.")
        old_console_path = self.console.path
        old_console_address = self.console.slippi_address
        old_console_save_replays = getattr(self.console, 'save_replays', False)

        try:
            self.console.stop()
        except Exception:
            pass
        for controller in self.controllers.values():
            try:
                controller.release_all()
                controller.disconnect()
            except Exception:
                pass
        self.current_gamestate = None
        self._in_game = False
        self._action_sweep_run_in_match = False

        # Create a fresh Console object instead of reusing the old one
        self.console = Console(
            path=old_console_path,
            slippi_address=old_console_address,
            save_replays=old_console_save_replays,
            copy_home_directory=False,
            tmp_home_directory=False,
            blocking_input=True,
        )

        # Reconnect controllers to the new console
        for port, controller in self.controllers.items():
            self.controllers[port] = Controller(
                console=self.console,
                port=port,
                type=ControllerType.STANDARD,
                fix_analog_inputs=False,
            )

        try:
            self.console.run(iso_path=str(self.iso_path) if self.iso_path else None)
        except Exception as exc:
            self._log(f"Restart failed to launch Dolphin: {exc}")
            return False
        if not self.console.connect():
            self._log("Restart failed: could not connect to Dolphin.")
            return False
        for controller in self.controllers.values():
            if not controller.connect():
                self._log(
                    f"Restart warning: failed to connect controller {controller.port}."
                )
            try:
                controller.release_all()
            except Exception:
                pass
        self._log("Restart complete; waiting for menus to load.")
        return True

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

    def _player_present(self, port: int) -> bool:
        return bool(self.current_gamestate and port in self.current_gamestate.players)

    def _player_is_active(self, port: int) -> bool:
        if not self._player_present(port):
            return False
        player = self.current_gamestate.players[port]
        if player.is_dead or getattr(player, "is_inactive", False):
            return False
        return player.action not in SPAWN_ACTIONS

    def _move_is_complete(self, player) -> bool:
        action = player.action
        if action is None:
            return False
        if player.is_dead or getattr(player, "is_inactive", False):
            return True
        if action in NEUTRAL_COMPLETE_ACTIONS:
            return True
        return False

    def _wait_until_player_active(self, port: int) -> Iterator[None]:
        """Yield until the player is alive, active, and past spawn actions."""

        while not self._player_is_active(port):
            yield

    def _return_to_position(
        self,
        port: int,
        target_pos: Tuple[float, float],
        *,
        tolerance: float = 0.05,
        max_frames: int = 300,
    ) -> Iterator[None]:
        """Move character back to target position within tolerance."""

        target_x, target_y = target_pos
        if math.isnan(target_x) or math.isnan(target_y):
            # Invalid initial position, skip position reset
            return

        frames = 0
        settle_count = 0
        neutral_cooldown = 0
        last_direction = 0.0

        while frames < max_frames:
            frames += 1

            if not self._player_is_active(port):
                self.set_neutral(port)
                yield
                continue

            pos = self.player_position(port)
            if math.isnan(pos[0]) or math.isnan(pos[1]):
                self.set_neutral(port)
                yield
                continue

            delta_x = target_x - pos[0]
            delta_y = abs(target_y - pos[1])

            # Check if we're close enough (within tolerance on X, and grounded/close on Y)
            if abs(delta_x) <= tolerance and delta_y <= 5.0:
                self.set_neutral(port)
                settle_count = min(settle_count + 1, NEUTRAL_SETTLE_FRAMES)
                if settle_count >= NEUTRAL_SETTLE_FRAMES:
                    break
                yield
                continue

            settle_count = 0

            # Need to move back
            if abs(delta_x) <= tolerance:
                # Close enough on X, just wait
                self.set_neutral(port)
                yield
                continue

            # Determine direction to move
            direction = 1.0 if delta_x > 0 else -1.0

            # If we changed direction, insert neutral cooldown
            if last_direction != 0.0 and direction != last_direction:
                self.set_neutral(port)
                neutral_cooldown = NEUTRAL_SETTLE_FRAMES
                last_direction = 0.0
                yield
                continue

            if neutral_cooldown > 0:
                self.set_neutral(port)
                neutral_cooldown -= 1
                yield
                continue

            # Move toward target
            magnitude = min(1.0, max(MIN_MOVEMENT_MAG, abs(delta_x) / 12.0))
            if abs(delta_x) > 20.0:
                magnitude = 1.0

            self.set_main_stick(port, direction * magnitude, 0.0)
            last_direction = direction
            yield

        # Final settle
        self.set_neutral(port)
        yield from self.wait(NEUTRAL_SETTLE_FRAMES)

    def _return_to_facing(
        self,
        port: int,
        target_facing: bool,
        *,
        max_attempts: int = 3,
    ) -> Iterator[None]:
        """Ensure character is facing the target direction (True = right, False = left)."""

        for attempt in range(max_attempts):
            if not self._player_is_active(port):
                yield
                continue

            player = self.current_gamestate.players[port]
            current_facing = player.facing

            if current_facing == target_facing:
                # Already facing the correct direction
                break

            # Need to turn around - tap the opposite direction briefly
            turn_direction = -1.0 if current_facing else 1.0
            self.set_main_stick(port, turn_direction, 0.0)
            yield from self.wait(2)
            self.set_neutral(port)
            yield from self.wait(8)

        # Final settle
        self.set_neutral(port)
        yield from self.wait(NEUTRAL_SETTLE_FRAMES)

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

    def _human_character_name(self, character: Character) -> str:
        return character.name.replace("_", " ").title()

    def _ensure_action_log_writer(self) -> None:
        if self.action_log_path is None:
            return
        if self._action_log_writer is not None and self._action_log_file is not None:
            return
        path = self.action_log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = path.exists() and path.stat().st_size > 0
        self._action_log_file = path.open("a", newline="")
        self._action_log_writer = csv.writer(self._action_log_file)
        if not file_exists:
            self._action_log_writer.writerow(
                [
                    "character",
                    "character_id",
                    "aerial_or_grounded",
                    "move",
                    "frame",
                    "action_state",
                    "action_state_id",
                    "action_state_id_hex",
                ]
            )
            self._action_log_file.flush()

    def _log_action_frame(
        self,
        *,
        character: Character,
        aerial_or_grounded: str,
        move: str,
        frame: int,
        action: Action,
    ) -> None:
        if self.action_log_path is None:
            return
        self._ensure_action_log_writer()
        if self._action_log_writer is None or self._action_log_file is None:
            return
        row = [
            self._human_character_name(character),
            character.value,
            aerial_or_grounded,
            move,
            frame,
            action.name,
            action.value,
            hex(action.value),
        ]
        self._action_log_writer.writerow(row)
        self._action_log_file.flush()
        print("[action-csv] " + ",".join(str(item) for item in row))

    def _ensure_ingame(self) -> None:
        if not self._in_game:
            raise RuntimeError("Requested action requires an active match.")

    def _maybe_start_action_sweep(self) -> None:
        if not self.enable_action_sweep:
            return
        if not self._in_game:
            return
        if self._action_sweep_index >= len(self._action_sweep_characters):
            return
        if self.macros or self._current_params:
            return
        if self._action_sweep_run_in_match:
            return
        self._action_sweep_run_in_match = True
        self._enqueue_moves_for_current_character()

    def _maybe_start_auto_scan(self) -> None:
        if self.enable_action_sweep:
            return
        if not self._in_game:
            return
        if self._current_params is not None or self.macros:
            return
        if not self._auto_initialized:
            self._auto_initialized = True
            self._auto_complete_logged = False
            return

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


def queue_default_action_state_demo(port: Optional[int] = None) -> None:
    """Enqueue the default Fox B-move action logging demo."""
    ensure_lab().queue_default_action_state_demo(port)


def start_action_state_sweep() -> None:
    """Begin (or restart) the full roster B-move sweep."""
    ensure_lab().start_action_state_sweep()


__all__ = [
    "LAB",
    "CONTROL_PALETTE",
    "FirefoxTestResult",
    "GameLab",
    "queue_default_action_state_demo",
    "start_action_state_sweep",
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
        "--action-log-path",
        default="action_state_log.csv",
        help="CSV path for action-state logging (appends; includes header when empty).",
    )
    parser.add_argument(
        "--no-action-state-sweep",
        action="store_true",
        help="Disable the automatic per-character B-move action-state sweep.",
    )
    return parser.parse_args()


def main() -> None:
    init_config()
    args = _parse_args()
    action_log_path = (
        None
        if args.no_action_state_sweep
        else Path(args.action_log_path).expanduser().resolve()
    )

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
        action_log_path=action_log_path,
        enable_action_sweep=not args.no_action_state_sweep,
        iso_path=Path(args.iso).expanduser().resolve() if args.iso else None,
    )
    LAB._log("Lab initialised. Attach a debugger and use helper functions.")
    if LAB.enable_action_sweep:
        LAB._log(
            f"Action-state sweep enabled. Logging to {LAB.action_log_path} (tail -f to watch)."
        )
        LAB.start_action_state_sweep()
    else:
        LAB._log("Action-state sweep disabled (--no-action-state-sweep supplied).")
    try:
        LAB.spin()
    finally:
        LAB.shutdown()


if __name__ == "__main__":
    main()
