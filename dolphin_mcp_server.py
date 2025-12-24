#!/usr/bin/env python3
"""
MCP server exposing Dolphin/Melee lab functionality to Claude.

This server runs Dolphin in a background thread and exposes tools for:
- Setting up matches (stage, characters)
- Controlling the game (inputs, frame advance)
- Reading game state
- Running Python experiments with helper functions

Usage:
    python dolphin_mcp_server.py --iso /path/to/melee.iso --dolphin /path/to/dolphin

Then configure Claude Code to use this as an MCP server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import signal
import sys
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# MCP imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Add repo to path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libmelee.melee.console import Console
from libmelee.melee.controller import Controller
from libmelee.melee.enums import Action, Button, Character, ControllerType, Menu, Stage
from libmelee.melee.gamestate import GameState, PlayerState
from libmelee.melee import stages
from libmelee.melee.menuhelper import MenuHelper


# ---------------------------------------------------------------------------
# Data structures for commands and responses
# ---------------------------------------------------------------------------

@dataclass
class Command:
    """A command to be executed in the game thread."""
    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    response_event: threading.Event = field(default_factory=threading.Event)
    response: Any = None
    error: Optional[str] = None


@dataclass
class FrameInputs:
    """Controller inputs for a single frame."""
    main_stick: Tuple[float, float] = (0.0, 0.0)
    c_stick: Tuple[float, float] = (0.0, 0.0)
    l_trigger: float = 0.0
    r_trigger: float = 0.0
    buttons: List[str] = field(default_factory=list)  # Button names to hold


@dataclass
class PlayerSnapshot:
    """Serializable snapshot of player state."""
    port: int
    character: str
    position_x: float
    position_y: float
    percent: float
    stock: int
    action: str
    action_frame: int
    facing: bool  # True = right
    on_ground: bool
    jumps_left: int
    shield_strength: float
    hitstun_frames_left: int
    hitlag_left: int
    invulnerability_left: int
    is_dead: bool
    speed_x: float
    speed_y: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "port": self.port,
            "character": self.character,
            "position": {"x": self.position_x, "y": self.position_y},
            "percent": self.percent,
            "stock": self.stock,
            "action": self.action,
            "action_frame": self.action_frame,
            "facing": "right" if self.facing else "left",
            "on_ground": self.on_ground,
            "jumps_left": self.jumps_left,
            "shield_strength": self.shield_strength,
            "hitstun_frames_left": self.hitstun_frames_left,
            "hitlag_left": self.hitlag_left,
            "invulnerability_left": self.invulnerability_left,
            "is_dead": self.is_dead,
            "speed": {"x": self.speed_x, "y": self.speed_y},
        }


@dataclass
class GameSnapshot:
    """Serializable snapshot of full game state."""
    frame: int
    stage: str
    menu_state: str
    players: Dict[int, PlayerSnapshot]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.frame,
            "stage": self.stage,
            "menu_state": self.menu_state,
            "players": {p: s.to_dict() for p, s in self.players.items()},
        }


def snapshot_player(port: int, player: PlayerState) -> PlayerSnapshot:
    """Create a serializable snapshot from PlayerState."""
    return PlayerSnapshot(
        port=port,
        character=player.character.name if player.character else "UNKNOWN",
        position_x=float(player.position.x),
        position_y=float(player.position.y),
        percent=float(player.percent),
        stock=int(player.stock),
        action=player.action.name if player.action else "UNKNOWN",
        action_frame=int(player.action_frame),
        facing=bool(player.facing),
        on_ground=bool(player.on_ground),
        jumps_left=int(player.jumps_left),
        shield_strength=float(player.shield_strength),
        hitstun_frames_left=int(player.hitstun_frames_left),
        hitlag_left=int(player.hitlag_left),
        invulnerability_left=int(player.invulnerability_left),
        is_dead=bool(player.is_dead),
        speed_x=float(player.speed_air_x_self + player.speed_ground_x_self),
        speed_y=float(player.speed_y_self),
    )


def snapshot_gamestate(gs: GameState) -> GameSnapshot:
    """Create a serializable snapshot from GameState."""
    return GameSnapshot(
        frame=gs.frame,
        stage=gs.stage.name if gs.stage else "UNKNOWN",
        menu_state=gs.menu_state.name if gs.menu_state else "UNKNOWN",
        players={port: snapshot_player(port, p) for port, p in gs.players.items()},
    )


# ---------------------------------------------------------------------------
# Game Lab with MCP command handling
# ---------------------------------------------------------------------------

class MCPGameLab:
    """
    Game lab that processes MCP commands.

    Runs in a background thread, processing commands from the MCP server
    while running the Dolphin game loop.
    """

    def __init__(
        self,
        console: Console,
        controllers: Dict[int, Controller],
        command_queue: queue.Queue,
        stage: Stage = Stage.FINAL_DESTINATION,
        p1_character: Character = Character.FOX,
        p2_character: Character = Character.FOX,
        verbose: bool = True,
    ):
        self.console = console
        self.controllers = controllers
        self.command_queue = command_queue
        self.stage = stage
        self.p1_character = p1_character
        self.p2_character = p2_character
        self.verbose = verbose

        self.menu_helper = MenuHelper()
        self.current_gamestate: Optional[GameState] = None
        self._in_game = False
        self._running = True
        self._match_started = False

        # Controller state
        self._controller_state: Dict[int, FrameInputs] = {
            port: FrameInputs() for port in controllers
        }

        # Frame recording
        self._recording: List[GameSnapshot] = []
        self._is_recording = False

        # Ports
        self.primary_port = min(controllers.keys())
        self.secondary_port = max(controllers.keys()) if len(controllers) > 1 else self.primary_port

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[mcp-lab] {msg}", file=sys.stderr)

    # -----------------------------------------------------------------------
    # Command handlers
    # -----------------------------------------------------------------------

    def handle_command(self, cmd: Command) -> None:
        """Process a command and set its response."""
        try:
            handler = getattr(self, f"_cmd_{cmd.name}", None)
            if handler is None:
                cmd.error = f"Unknown command: {cmd.name}"
            else:
                cmd.response = handler(**cmd.args)
        except Exception as e:
            cmd.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        finally:
            cmd.response_event.set()

    def _cmd_get_state(self) -> Dict[str, Any]:
        """Get current game state."""
        if not self.current_gamestate:
            return {"status": "no_gamestate", "in_game": False}

        snapshot = snapshot_gamestate(self.current_gamestate)
        result = snapshot.to_dict()
        result["in_game"] = self._in_game
        return result

    def _cmd_configure_match(
        self,
        stage: str,
        p1_character: str,
        p2_character: str,
    ) -> Dict[str, Any]:
        """Configure match settings."""
        try:
            self.stage = Stage[stage.upper()]
        except KeyError:
            return {"error": f"Unknown stage: {stage}"}

        try:
            self.p1_character = Character[p1_character.upper()]
        except KeyError:
            return {"error": f"Unknown character: {p1_character}"}

        try:
            self.p2_character = Character[p2_character.upper()]
        except KeyError:
            return {"error": f"Unknown character: {p2_character}"}

        self._match_started = False
        return {
            "stage": self.stage.name,
            "p1_character": self.p1_character.name,
            "p2_character": self.p2_character.name,
        }

    def _cmd_set_controller(
        self,
        port: int,
        main_stick: Optional[Tuple[float, float]] = None,
        c_stick: Optional[Tuple[float, float]] = None,
        l_trigger: Optional[float] = None,
        r_trigger: Optional[float] = None,
        buttons: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Set controller state for a port."""
        if port not in self._controller_state:
            return {"error": f"Invalid port: {port}"}

        state = self._controller_state[port]
        if main_stick is not None:
            state.main_stick = (
                max(-1.0, min(1.0, main_stick[0])),
                max(-1.0, min(1.0, main_stick[1])),
            )
        if c_stick is not None:
            state.c_stick = (
                max(-1.0, min(1.0, c_stick[0])),
                max(-1.0, min(1.0, c_stick[1])),
            )
        if l_trigger is not None:
            state.l_trigger = max(0.0, min(1.0, l_trigger))
        if r_trigger is not None:
            state.r_trigger = max(0.0, min(1.0, r_trigger))
        if buttons is not None:
            state.buttons = buttons

        return {"ok": True, "port": port}

    def _cmd_neutral(self, port: int) -> Dict[str, Any]:
        """Reset controller to neutral."""
        if port not in self._controller_state:
            return {"error": f"Invalid port: {port}"}
        self._controller_state[port] = FrameInputs()
        return {"ok": True, "port": port}

    def _cmd_advance_frames(
        self,
        frames: int,
        p1_inputs: Optional[List[Dict]] = None,
        p2_inputs: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Advance N frames with optional per-frame inputs.
        Returns list of game states for each frame.

        Each input dict can have: main_stick, c_stick, l_trigger, r_trigger, buttons
        """
        if frames < 1 or frames > 600:
            return {"error": "frames must be 1-600"}

        states = []
        for i in range(frames):
            # Apply per-frame inputs if provided
            if p1_inputs and i < len(p1_inputs):
                self._apply_input_dict(self.primary_port, p1_inputs[i])
            if p2_inputs and i < len(p2_inputs):
                self._apply_input_dict(self.secondary_port, p2_inputs[i])

            # Run one frame
            self._apply_controller_states()
            gs = self.console.step()
            if gs:
                self.current_gamestate = gs
                self._update_in_game_status()
                states.append(snapshot_gamestate(gs).to_dict())

        return {"frames_advanced": len(states), "states": states}

    def _apply_input_dict(self, port: int, inputs: Dict) -> None:
        """Apply a single frame's inputs from a dict."""
        state = self._controller_state[port]
        if "main_stick" in inputs:
            state.main_stick = tuple(inputs["main_stick"])
        if "c_stick" in inputs:
            state.c_stick = tuple(inputs["c_stick"])
        if "l_trigger" in inputs:
            state.l_trigger = inputs["l_trigger"]
        if "r_trigger" in inputs:
            state.r_trigger = inputs["r_trigger"]
        if "buttons" in inputs:
            state.buttons = inputs["buttons"]

    def _cmd_wait_for_game(self, max_frames: int = 600) -> Dict[str, Any]:
        """Wait until we're in-game, with timeout."""
        for _ in range(max_frames):
            self._apply_controller_states()
            gs = self.console.step()
            if gs:
                self.current_gamestate = gs
                self._update_in_game_status()
                if self._in_game:
                    return {"ok": True, "frame": gs.frame}
                # Drive menus while waiting
                self._drive_menus(gs)
        return {"error": "Timeout waiting for game to start"}

    def _cmd_reset_match(self) -> Dict[str, Any]:
        """
        Reset the match using L+R+A+Start.
        Returns when back in-game or at character select.
        """
        if not self._in_game:
            return {"error": "Not in game"}

        port = self.primary_port

        # Pause first
        self._press_button(port, Button.BUTTON_START)
        for _ in range(12):
            self._apply_controller_states()
            self.console.step()
        self._release_button(port, Button.BUTTON_START)

        # Hold L+R+A+Start
        for btn in [Button.BUTTON_L, Button.BUTTON_R, Button.BUTTON_A, Button.BUTTON_START]:
            self._press_button(port, btn)

        for _ in range(120):
            self._apply_controller_states()
            gs = self.console.step()
            if gs:
                self.current_gamestate = gs

        # Release
        for port in self.controllers:
            self._controller_state[port] = FrameInputs()

        # Wait for menu or new game
        for _ in range(300):
            self._apply_controller_states()
            gs = self.console.step()
            if gs:
                self.current_gamestate = gs
                self._update_in_game_status()
                if gs.menu_state not in (Menu.IN_GAME, Menu.SUDDEN_DEATH):
                    return {"ok": True, "menu_state": gs.menu_state.name}

        return {"ok": True, "note": "Reset completed"}

    def _cmd_run_experiment(self, code: str, timeout: float = 30.0) -> Dict[str, Any]:
        """
        Execute Python code with access to helper functions.

        Available in the code:
        - gamestate: current GameState object
        - lab: this MCPGameLab instance
        - advance(n, p1_input=None, p2_input=None): advance n frames, return states
        - set_stick(port, x, y): set main stick
        - set_c_stick(port, x, y): set C-stick
        - press(port, button_name): press a button
        - release(port, button_name): release a button
        - neutral(port): reset to neutral
        - get_position(port): get (x, y) position
        - get_action(port): get current action name
        - get_percent(port): get damage percent
        - get_stocks(port): get remaining stocks
        - wait_until(condition, max_frames=300): wait until condition(gamestate) is True
        - record_until(condition, max_frames=300, inputs=None): record states until condition
        - Button, Action, Character, Stage: enums
        - stages: stage geometry module
        """
        # Build execution environment
        env = self._build_experiment_env()

        # Execute with timeout
        result = {"output": None, "error": None}

        def run_code():
            try:
                exec(code, env)
                result["output"] = env.get("result", None)
            except Exception as e:
                result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        thread = threading.Thread(target=run_code)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            return {"error": f"Experiment timed out after {timeout}s"}

        if result["error"]:
            return {"error": result["error"]}

        # Serialize the result
        output = result["output"]
        if output is None:
            return {"result": None}

        try:
            # Try to JSON serialize
            json.dumps(output)
            return {"result": output}
        except (TypeError, ValueError):
            # Return string representation
            return {"result": str(output)}

    def _build_experiment_env(self) -> Dict[str, Any]:
        """Build the execution environment for experiments."""

        def advance(n: int, p1_input: Optional[Dict] = None, p2_input: Optional[Dict] = None) -> List[Dict]:
            """Advance n frames, optionally with constant inputs."""
            states = []
            for _ in range(n):
                if p1_input:
                    self._apply_input_dict(self.primary_port, p1_input)
                if p2_input:
                    self._apply_input_dict(self.secondary_port, p2_input)
                self._apply_controller_states()
                gs = self.console.step()
                if gs:
                    self.current_gamestate = gs
                    self._update_in_game_status()
                    states.append(snapshot_gamestate(gs).to_dict())
            return states

        def set_stick(port: int, x: float, y: float) -> None:
            self._controller_state[port].main_stick = (
                max(-1.0, min(1.0, x)),
                max(-1.0, min(1.0, y)),
            )

        def set_c_stick(port: int, x: float, y: float) -> None:
            self._controller_state[port].c_stick = (
                max(-1.0, min(1.0, x)),
                max(-1.0, min(1.0, y)),
            )

        def press(port: int, button_name: str) -> None:
            state = self._controller_state[port]
            if button_name not in state.buttons:
                state.buttons.append(button_name)

        def release(port: int, button_name: str) -> None:
            state = self._controller_state[port]
            if button_name in state.buttons:
                state.buttons.remove(button_name)

        def neutral(port: int) -> None:
            self._controller_state[port] = FrameInputs()

        def get_position(port: int) -> Tuple[float, float]:
            if not self.current_gamestate or port not in self.current_gamestate.players:
                return (float("nan"), float("nan"))
            p = self.current_gamestate.players[port]
            return (float(p.position.x), float(p.position.y))

        def get_action(port: int) -> str:
            if not self.current_gamestate or port not in self.current_gamestate.players:
                return "UNKNOWN"
            return self.current_gamestate.players[port].action.name

        def get_percent(port: int) -> float:
            if not self.current_gamestate or port not in self.current_gamestate.players:
                return 0.0
            return float(self.current_gamestate.players[port].percent)

        def get_stocks(port: int) -> int:
            if not self.current_gamestate or port not in self.current_gamestate.players:
                return 0
            return int(self.current_gamestate.players[port].stock)

        def wait_until(
            condition: Callable[[GameState], bool],
            max_frames: int = 300,
        ) -> List[Dict]:
            """Wait until condition returns True, return states."""
            states = []
            for _ in range(max_frames):
                self._apply_controller_states()
                gs = self.console.step()
                if gs:
                    self.current_gamestate = gs
                    self._update_in_game_status()
                    states.append(snapshot_gamestate(gs).to_dict())
                    if condition(gs):
                        break
            return states

        def record_until(
            condition: Callable[[GameState], bool],
            max_frames: int = 300,
            p1_input: Optional[Dict] = None,
            p2_input: Optional[Dict] = None,
        ) -> List[Dict]:
            """Record states while applying inputs until condition is True."""
            states = []
            for _ in range(max_frames):
                if p1_input:
                    self._apply_input_dict(self.primary_port, p1_input)
                if p2_input:
                    self._apply_input_dict(self.secondary_port, p2_input)
                self._apply_controller_states()
                gs = self.console.step()
                if gs:
                    self.current_gamestate = gs
                    self._update_in_game_status()
                    states.append(snapshot_gamestate(gs).to_dict())
                    if condition(gs):
                        break
            return states

        return {
            "gamestate": self.current_gamestate,
            "lab": self,
            "advance": advance,
            "set_stick": set_stick,
            "set_c_stick": set_c_stick,
            "press": press,
            "release": release,
            "neutral": neutral,
            "get_position": get_position,
            "get_action": get_action,
            "get_percent": get_percent,
            "get_stocks": get_stocks,
            "wait_until": wait_until,
            "record_until": record_until,
            "Button": Button,
            "Action": Action,
            "Character": Character,
            "Stage": Stage,
            "stages": stages,
            "result": None,  # Set this to return a value
        }

    def _cmd_list_characters(self) -> Dict[str, Any]:
        """List available characters."""
        blocked = {"WIREFRAME_MALE", "WIREFRAME_FEMALE", "GIGA_BOWSER", "SANDBAG", "UNKNOWN_CHARACTER", "NANA"}
        chars = [c.name for c in Character if c.name not in blocked]
        return {"characters": chars}

    def _cmd_list_stages(self) -> Dict[str, Any]:
        """List available stages."""
        valid = {"FINAL_DESTINATION", "BATTLEFIELD", "DREAMLAND", "FOUNTAIN_OF_DREAMS", "YOSHIS_STORY", "POKEMON_STADIUM"}
        return {"stages": list(valid)}

    def _cmd_get_stage_info(self, stage: str) -> Dict[str, Any]:
        """Get stage geometry information."""
        try:
            stg = Stage[stage.upper()]
        except KeyError:
            return {"error": f"Unknown stage: {stage}"}

        info = {
            "name": stg.name,
            "blastzones": stages.BLASTZONES.get(stg),
            "edge_position": stages.EDGE_POSITION.get(stg),
            "edge_ground_position": stages.EDGE_GROUND_POSITION.get(stg),
            "left_platform": stages.left_platform_position(stg),
            "right_platform": stages.right_platform_position(stg),
            "top_platform": stages.top_platform_position(stg),
        }
        return info

    # -----------------------------------------------------------------------
    # Controller helpers
    # -----------------------------------------------------------------------

    def _press_button(self, port: int, button: Button) -> None:
        """Add button to pressed list."""
        state = self._controller_state[port]
        btn_name = button.name
        if btn_name not in state.buttons:
            state.buttons.append(btn_name)

    def _release_button(self, port: int, button: Button) -> None:
        """Remove button from pressed list."""
        state = self._controller_state[port]
        btn_name = button.name
        if btn_name in state.buttons:
            state.buttons.remove(btn_name)

    def _apply_controller_states(self) -> None:
        """Apply current controller states to the actual controllers."""
        from libmelee.melee.controller import fix_analog_stick

        for port, controller in self.controllers.items():
            state = self._controller_state[port]

            # Main stick (convert from -1..1 to 0..1)
            main_x = fix_analog_stick((state.main_stick[0] + 1.0) / 2.0)
            main_y = fix_analog_stick((state.main_stick[1] + 1.0) / 2.0)
            controller.tilt_analog(Button.BUTTON_MAIN, main_x, main_y)

            # C-stick
            c_x = fix_analog_stick((state.c_stick[0] + 1.0) / 2.0)
            c_y = fix_analog_stick((state.c_stick[1] + 1.0) / 2.0)
            controller.tilt_analog(Button.BUTTON_C, c_x, c_y)

            # Triggers
            controller.press_shoulder(Button.BUTTON_L, state.l_trigger)
            controller.press_shoulder(Button.BUTTON_R, state.r_trigger)

            # Buttons
            all_buttons = [
                Button.BUTTON_A, Button.BUTTON_B, Button.BUTTON_X, Button.BUTTON_Y,
                Button.BUTTON_Z, Button.BUTTON_L, Button.BUTTON_R, Button.BUTTON_START,
                Button.BUTTON_D_UP, Button.BUTTON_D_DOWN, Button.BUTTON_D_LEFT, Button.BUTTON_D_RIGHT,
            ]
            for btn in all_buttons:
                if btn.name in state.buttons:
                    controller.press_button(btn)
                else:
                    controller.release_button(btn)

            controller.flush()

    def _update_in_game_status(self) -> None:
        """Update whether we're currently in a match."""
        if not self.current_gamestate:
            self._in_game = False
            return
        menu = self.current_gamestate.menu_state
        self._in_game = menu in (Menu.IN_GAME, Menu.SUDDEN_DEATH)

    def _drive_menus(self, gs: GameState) -> None:
        """Navigate menus to start a match."""
        self.menu_helper.menu_helper_simple(
            gs,
            self.controllers[self.primary_port],
            self.p1_character,
            self.stage,
            costume=1,
            autostart=False,
            swag=False,
        )
        if self.secondary_port in self.controllers and self.secondary_port != self.primary_port:
            self.menu_helper.choose_character(
                character=self.p2_character,
                gamestate=gs,
                controller=self.controllers[self.secondary_port],
                cpu_level=0,
                costume=1,
                swag=False,
                start=True,
            )

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def spin(self) -> None:
        """Main game loop. Processes commands and runs Dolphin."""
        self._log("Game loop started")

        while self._running:
            # Check for commands (non-blocking)
            try:
                cmd = self.command_queue.get_nowait()
                self.handle_command(cmd)
            except queue.Empty:
                pass

            # Run one frame
            self._apply_controller_states()
            gs = self.console.step()

            if gs:
                self.current_gamestate = gs
                self._update_in_game_status()

                # Drive menus when not in-game
                if not self._in_game:
                    self._drive_menus(gs)

        self._log("Game loop ended")

    def shutdown(self) -> None:
        """Stop the game loop and clean up."""
        self._running = False
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


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

class DolphinMCPServer:
    """MCP server that exposes Dolphin lab as tools."""

    def __init__(self, command_queue: queue.Queue):
        self.command_queue = command_queue
        self.server = Server("dolphin-lab")
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Register MCP tool handlers."""

        @self.server.list_tools()
        async def list_tools():
            return [
                Tool(
                    name="get_game_state",
                    description="Get current game state including player positions, actions, percents, stocks, and more.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                Tool(
                    name="configure_match",
                    description="Configure the match: stage and characters for both players. Call this before starting a match.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "stage": {
                                "type": "string",
                                "description": "Stage name (e.g., BATTLEFIELD, FINAL_DESTINATION, YOSHIS_STORY)",
                            },
                            "p1_character": {
                                "type": "string",
                                "description": "Player 1 character (e.g., FOX, MARTH, FALCO)",
                            },
                            "p2_character": {
                                "type": "string",
                                "description": "Player 2 character (e.g., FOX, MARTH, CPTFALCON)",
                            },
                        },
                        "required": ["stage", "p1_character", "p2_character"],
                    },
                ),
                Tool(
                    name="set_controller",
                    description="Set controller inputs for a port. Inputs persist until changed.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "port": {"type": "integer", "description": "Controller port (1 or 2)"},
                            "main_stick": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "[x, y] main stick position, each -1.0 to 1.0",
                            },
                            "c_stick": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "[x, y] C-stick position, each -1.0 to 1.0",
                            },
                            "l_trigger": {"type": "number", "description": "L trigger 0.0 to 1.0"},
                            "r_trigger": {"type": "number", "description": "R trigger 0.0 to 1.0"},
                            "buttons": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of buttons to hold: BUTTON_A, BUTTON_B, BUTTON_X, BUTTON_Y, BUTTON_Z, BUTTON_L, BUTTON_R, BUTTON_START",
                            },
                        },
                        "required": ["port"],
                    },
                ),
                Tool(
                    name="neutral",
                    description="Reset a controller to neutral (all inputs released).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "port": {"type": "integer", "description": "Controller port (1 or 2)"},
                        },
                        "required": ["port"],
                    },
                ),
                Tool(
                    name="advance_frames",
                    description="Advance the game by N frames, optionally with per-frame inputs. Returns game state for each frame.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "frames": {
                                "type": "integer",
                                "description": "Number of frames to advance (1-600)",
                            },
                            "p1_inputs": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "main_stick": {"type": "array", "items": {"type": "number"}},
                                        "c_stick": {"type": "array", "items": {"type": "number"}},
                                        "l_trigger": {"type": "number"},
                                        "r_trigger": {"type": "number"},
                                        "buttons": {"type": "array", "items": {"type": "string"}},
                                    },
                                },
                                "description": "Per-frame inputs for player 1. Array length should match frames.",
                            },
                            "p2_inputs": {
                                "type": "array",
                                "items": {"type": "object"},
                                "description": "Per-frame inputs for player 2.",
                            },
                        },
                        "required": ["frames"],
                    },
                ),
                Tool(
                    name="wait_for_game",
                    description="Wait until the game has started (navigating menus automatically). Use after configure_match.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "max_frames": {
                                "type": "integer",
                                "description": "Maximum frames to wait (default 600)",
                            },
                        },
                    },
                ),
                Tool(
                    name="reset_match",
                    description="Reset the current match using L+R+A+Start. Returns to character select.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                Tool(
                    name="run_experiment",
                    description="""Execute Python code with access to game control helpers.

Available functions:
- advance(n, p1_input=None, p2_input=None): Advance n frames with optional inputs, returns list of states
- set_stick(port, x, y): Set main stick (-1 to 1)
- set_c_stick(port, x, y): Set C-stick
- press(port, button): Press button (e.g., 'BUTTON_A', 'BUTTON_B')
- release(port, button): Release button
- neutral(port): Reset controller to neutral
- get_position(port): Get (x, y) position
- get_action(port): Get current action name
- get_percent(port): Get damage percent
- get_stocks(port): Get remaining stocks
- wait_until(condition, max_frames): Wait until condition(gamestate) is True
- record_until(condition, max_frames, p1_input, p2_input): Record states until condition

Set 'result = <value>' to return data from the experiment.""",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute",
                            },
                            "timeout": {
                                "type": "number",
                                "description": "Timeout in seconds (default 30)",
                            },
                        },
                        "required": ["code"],
                    },
                ),
                Tool(
                    name="list_characters",
                    description="List all playable characters.",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="list_stages",
                    description="List all legal stages.",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="get_stage_info",
                    description="Get stage geometry: blastzones, edge positions, platforms.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "stage": {"type": "string", "description": "Stage name"},
                        },
                        "required": ["stage"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict):
            result = await self._execute_command(name, arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _execute_command(self, name: str, args: Dict) -> Dict[str, Any]:
        """Execute a command in the game thread and wait for response."""
        cmd = Command(name=name, args=args)
        self.command_queue.put(cmd)

        # Wait for response (with timeout)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: cmd.response_event.wait(timeout=60))

        if cmd.error:
            return {"error": cmd.error}
        return cmd.response if cmd.response is not None else {"ok": True}

    async def run(self) -> None:
        """Run the MCP server."""
        async with stdio_server() as (read, write):
            await self.server.run(read, write, self.server.create_initialization_options())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dolphin MCP Server")
    parser.add_argument("--iso", required=True, help="Path to Melee ISO")
    parser.add_argument("--dolphin", default=None, help="Path to Dolphin executable")
    parser.add_argument("--stage", default="FINAL_DESTINATION", help="Initial stage")
    parser.add_argument("--p1", default="FOX", help="Player 1 character")
    parser.add_argument("--p2", default="FOX", help="Player 2 character")
    parser.add_argument("--no-verbose", action="store_true", help="Disable verbose logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Create command queue for communication
    command_queue: queue.Queue = queue.Queue()

    # Create console
    console = Console(
        path=args.dolphin,
        slippi_address="127.0.0.1",
        save_replays=False,
        copy_home_directory=False,
        tmp_home_directory=False,
        blocking_input=True,
    )

    # Create controllers
    controllers = {
        1: Controller(console=console, port=1, type=ControllerType.STANDARD, fix_analog_inputs=False),
        2: Controller(console=console, port=2, type=ControllerType.STANDARD, fix_analog_inputs=False),
    }

    # Create game lab
    try:
        stage = Stage[args.stage.upper()]
    except KeyError:
        stage = Stage.FINAL_DESTINATION

    try:
        p1_char = Character[args.p1.upper()]
    except KeyError:
        p1_char = Character.FOX

    try:
        p2_char = Character[args.p2.upper()]
    except KeyError:
        p2_char = Character.FOX

    lab = MCPGameLab(
        console=console,
        controllers=controllers,
        command_queue=command_queue,
        stage=stage,
        p1_character=p1_char,
        p2_character=p2_char,
        verbose=not args.no_verbose,
    )

    # Cleanup handler
    def cleanup(*_):
        lab.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Start Dolphin
    print(f"[mcp] Starting Dolphin with ISO: {args.iso}", file=sys.stderr)
    console.run(iso_path=args.iso)

    if not console.connect():
        print("[mcp] ERROR: Failed to connect to console", file=sys.stderr)
        sys.exit(1)

    for controller in controllers.values():
        if not controller.connect():
            print(f"[mcp] ERROR: Failed to connect controller {controller.port}", file=sys.stderr)
            sys.exit(1)
        controller.release_all()

    print("[mcp] Dolphin connected, starting game thread", file=sys.stderr)

    # Start game loop in background thread
    game_thread = threading.Thread(target=lab.spin, daemon=True)
    game_thread.start()

    # Run MCP server in main thread
    mcp_server = DolphinMCPServer(command_queue)
    try:
        asyncio.run(mcp_server.run())
    finally:
        lab.shutdown()


if __name__ == "__main__":
    main()
