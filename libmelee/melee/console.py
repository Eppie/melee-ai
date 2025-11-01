"""The Console represents the engine running the game.

This can be Dolphin (Slippi's Ishiiruka) or an SLP file. The Console object
is your method to start and stop Dolphin, set configs, and get the latest GameState.
"""

import base64
import configparser
import csv
import dataclasses
import enum
import logging
import math
import os
import platform
import shutil
import stat
import struct
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Final

import numpy as np
from packaging import version

from libmelee.melee import enums
from libmelee.melee import stages
from libmelee.melee.enums import Action
from libmelee.melee.gamestate import GameState, Projectile, PlayerState
from libmelee.melee.slippstream import (
    SlippstreamClient,
    EventType,
    EVENT_TYPE_BY_BYTE,
    EVENT_TO_STAGE,
)
from libmelee.melee.slpfilestreamer import SLPFileStreamer


class SlippiVersionTooLow(Exception):
    """Raised when the Slippi version is not recent enough"""

    def __init__(self, message):
        self.message = message


class InvalidDolphinPath(Exception):
    """Raised when given path to Dolphin is invalid"""

    def __init__(self, message):
        self.message = message


def _ignore_fifos(src, names):
    fifos = []
    for name in names:
        path = os.path.join(src, name)
        if stat.S_ISFIFO(os.stat(path).st_mode):
            fifos.append(name)
    return fifos


def _copytree_safe(src, dst):
    shutil.copytree(src, dst, ignore=_ignore_fifos)


def _default_home_path(path: str) -> str:
    return "/Users/eppie/Library/Application Support/com.project-slippi.dolphin/netplay/User/"
    if platform.system() == "Darwin":
        return path + "/Contents/Resources/User/"

    # Next check if the home path is in the same dir as the exe
    user_path = path + "/User/"
    if os.path.isdir(user_path):
        return user_path

    # Otherwise, this must be an appimage install. Use the .config
    if platform.system() == "Linux":
        return str(Path.home()) + "/.config/SlippiOnline/"

    raise FileNotFoundError("Could not find dolphin home directory.")


def read_byte(event_bytes: bytes, offset: int):
    return np.ndarray((1,), ">B", event_bytes, offset)[0]


def read_shift_jis(event_bytes: bytes, offset: int) -> str:
    # Same behavior, but fast scan with memoryview
    mv = memoryview(event_bytes)
    end = offset
    # Find NUL terminator
    while mv[end] != 0:
        end += 1
    return bytes(mv[offset:end]).decode("shift-jis")


def get_exe_path(path: str) -> str:
    """Return the path to the dolphin executable"""
    if os.path.isfile(path):
        return path

    exe_path = [path]
    if platform.system() == "Darwin":
        exe_path.append("Contents/MacOS")

    if platform.system() == "Windows":
        exe_name = "Slippi Dolphin.exe"
    elif platform.system() == "Darwin":
        exe_name = "Slippi Dolphin"
    else:  # Linux
        exe_name = "dolphin-emu"

    return os.path.join(*exe_path, exe_name)


# TODO: custom mainline builds can have headless support -- we should include
# that in the version output and parse it in get_dolphin_version
class DolphinBuild(enum.Enum):
    NETPLAY = enum.auto()
    PLAYBACK = enum.auto()
    EXI_AI = enum.auto()


_STRING_TO_BUILD = {
    "Playback": DolphinBuild.PLAYBACK,
    "ExiAI": DolphinBuild.EXI_AI,
}
# Precompiled Structs (big-endian)
_S_I: Final = struct.Struct(">i")
_S_H: Final = struct.Struct(">H")
_S_B: Final = struct.Struct(">B")
_S_b: Final = struct.Struct(">b")
_S_F: Final = struct.Struct(">f")
_S_F4: Final = struct.Struct(">4f")
_S_2F: Final = struct.Struct(">2f")
_S_6F: Final = struct.Struct(">6f")


def _i32(mv: memoryview, off: int) -> int:
    return _S_I.unpack_from(mv, off)[0]


def _u16(mv: memoryview, off: int) -> int:
    return _S_H.unpack_from(mv, off)[0]


def _u8(mv: memoryview, off: int) -> int:
    return _S_B.unpack_from(mv, off)[0]


def _i8(mv: memoryview, off: int) -> int:
    return _S_b.unpack_from(mv, off)[0]


def _f32(mv: memoryview, off: int) -> float:
    return _S_F.unpack_from(mv, off)[0]


@dataclasses.dataclass
class DolphinVersion:
    mainline: bool
    version: str
    build: DolphinBuild


def get_dolphin_version(path: str) -> DolphinVersion:
    exe_path = get_exe_path(path)
    result = subprocess.run([exe_path, "--version"], capture_output=True)

    # Ishiiruka actually gives returncode 1 and puts
    # "Faster Melee - Slippi (3.4.0)" in stderr!
    output = result.stdout if result.returncode == 0 else result.stderr
    output = output.decode().strip()

    # Mainline versions look like "4.0.0-mainline-beta.4"
    if output.find("mainline") != -1:
        version = output.split("-")[0]
        return DolphinVersion(
            mainline=True,
            version=version,
            # Mainline only supports netplay for now.
            build=DolphinBuild.NETPLAY,
        )

    # Ishiiruka on MacOS behaves a bit differently.
    if platform.system() == "Darwin":
        # Sadly playback dolphin doesn't output anything differently, so we
        # just assume it's a netplay build.
        assert result.returncode == 255
        return DolphinVersion(
            mainline=False,
            version=result.stdout.decode().strip(),
            build=DolphinBuild.NETPLAY,
        )

    # Ishiiruka
    contents = output.split(" - ")
    if contents[0] != "Faster Melee":
        raise ValueError(f"Unexpected dolphin version {output}")

    # "Slippi (VERSION)"
    begin = contents[1].find("(") + 1
    end = contents[1].find(")")
    version = contents[1][begin:end]

    if len(contents) == 2:
        build = DolphinBuild.NETPLAY
    else:
        build_str = contents[2]
        if build_str not in _STRING_TO_BUILD:
            raise ValueError(f"Unexpected dolphin version {output}")
        build = _STRING_TO_BUILD[build_str]

    return DolphinVersion(False, version, build)


@dataclasses.dataclass
class DumpConfig:
    dump: bool = False
    format: Optional[str] = None
    codec: Optional[str] = None
    encoder: Optional[str] = None
    path: Optional[str] = None

    def update_gfx_ini(self, gfx_ini: configparser.ConfigParser):
        section = "Settings"
        if not gfx_ini.has_section(section):
            gfx_ini.add_section(section)
        if self.format:
            gfx_ini.set(section, "DumpFormat", self.format)
        if self.codec:
            gfx_ini.set(section, "DumpCodec", self.codec)
        if self.encoder:
            gfx_ini.set(section, "DumpEncoder", self.encoder)
        if self.path:
            gfx_ini.set(section, "DumpPath", self.path)

        gfx_ini.set(section, "BitrateKbps", "3000")
        gfx_ini.set(section, "InternalResolutionFrameDumps", "True")


# pylint: disable=too-many-instance-attributes
class Console:
    """The console object that represents your Dolphin / Wii / SLP file"""

    def __init__(
        self,
        path: Optional[str] = None,
        is_dolphin: bool = True,
        dolphin_home_path: Optional[str] = None,
        tmp_home_directory: bool = True,
        copy_home_directory: bool = False,
        slippi_address: str = "127.0.0.1",
        slippi_port: int = 51441,
        online_delay: int = 0,
        blocking_input: bool = False,
        polling_mode: bool = False,
        polling_timeout: float = 0,
        skip_rollback_frames: bool = True,
        allow_old_version: bool = False,
        logger=None,
        setup_gecko_codes: bool = True,
        fullscreen: bool = True,
        gfx_backend: str = "",
        disable_audio: bool = False,
        overclock: Optional[float] = None,
        emulation_speed: float = 1.0,
        save_replays: bool = True,
        replay_dir: Optional[str] = None,
        user_json_path: Optional[str] = None,
        log_level: int = 3,  # WARN, see Source/Core/Common/Logging/Log.h
        log_types: list[str] = ["SLIPPI"],
        infinite_time: bool = False,
        use_exi_inputs=False,
        enable_ffw=False,
        dump_config: Optional[DumpConfig] = None,
    ):
        """Create a Console object

        Args:
            path (str): Path to the directory where your dolphin executable is located.
                If None, will assume the dolphin is remote and won't try to configure it.
            dolphin_home_path (str): Path to dolphin user directory. Optional.
            is_dolphin (bool): Is this console a dophin instance, or SLP file?
            is_mainline (bool): Is this mainline dolphin or Ishiiruka slippi?
            tmp_home_directory (bool): Use a temporary directory for the dolphin User path
                This is useful so instances don't interfere with each other.
            copy_home_directory (bool): Copy an existing home directory on the system.
                Unset to get a fresh directory that doesn't depend on system state.
            slippi_address (str): IP address of the Dolphin / Wii to connect to.
            slippi_port (int): UDP port that slippi will listen on
            online_delay (int): How many frames of delay to apply in online matches
            blocking_input (bool): Should dolphin block waiting for bot input
                This is only really useful if you're doing ML training.
            polling_mode (bool): Polls input to console rather than blocking for it
                When set, step() will always return immediately, but may be None if no
                gamestate is available yet.
            polling_timeout (float): In polling_mode, how long to wait for.
            allow_old_version (bool): Allow SLP versions older than 3.0.0 (rollback era)
                Only enable if you know what you're doing. You probably don't want this.
                Gamestates will be missing key information, come in really late, or possibly not work at all
            logger (logger.Logger): Logger instance to use. None for no logger.
            setup_gecko_codes (bool): Overwrites the user's GALE01r2.ini with libmelee's
                custom gecko codes. Should be used with tmp_home_directory.
            fullscreen (bool): Run melee fullscreen.
            gfx_backend (str): Graphics backend. Leave blank to use default.
            disable_audio (bool): Turn off sound.
            overclock (bool): Overclock the dolphin CPU. I haven't seen any benefit to
                this in my experiments.
            emulation_speed (float): Speed the game runs at. Set to 0 for unlimited speed.
                Only works with mainline dolphin, Ishiiruka ignores this option.
            save_replays (bool): Save slippi replays.
            replay_dir (str): Directory to save replays to. Defaults to "~/Slippi".
            user_json_path (str): Path to custom user.json for netplay. Doesn't work on
                Mac as the path is hardcoded.
            log_level (int): Dolphin log level.
            infinite_time (bool): Set the game to infinite time mode.
            use_exi_inputs (bool): Enable gecko code for exi dolphin inputs. This is
                necessary for fast-forward mode which ignores dolphin's normal polling.
                Must be used with a compatible Ishiiruka branch such as
                https://github.com/vladfi1/slippi-Ishiiruka/tree/exi-ai-rebase. Note that
                this will likely be incompatible with netplay.
            enable_ffw (bool): Enable fast-forward mode. Useful for bot training. Must
                have use_exi_inputs=True.
            dump_config (DumpConfig): Settings for video dumps.
        """
        self.logger = logger
        self.is_dolphin = is_dolphin
        self.path = path
        self.dolphin_home_path = dolphin_home_path
        self.temp_dir = None
        if tmp_home_directory and self.is_dolphin:
            self.temp_dir = tempfile.mkdtemp(prefix="libmelee_")
            home_dir = self.temp_dir + "/User/"
            if copy_home_directory:
                _copytree_safe(self._get_dolphin_home_path(), home_dir)
            self.dolphin_home_path = home_dir

        self.processingtime = 0
        self._frametimestamp = time.time()
        self.slippi_address = slippi_address
        """(str): IP address of the Dolphin / Wii to connect to."""
        self.slippi_port = slippi_port
        """(int): UDP port of slippi server. Default 51441"""
        self.eventsize = [0] * 0x100
        self.connected = False
        self.nick = ""
        """(str): The nickname the console has given itself."""
        self.version = ""
        """(str): The Slippi version of the console"""
        self.cursor = 0
        from libmelee.melee.controller import Controller  # avoid circular import

        self.controllers: list[Controller] = []
        self._current_stage = enums.Stage.NO_STAGE
        self._frame = 0
        self._polling_mode = polling_mode
        self._polling_timeout = polling_timeout
        self.skip_rollback_frames = skip_rollback_frames
        self.slp_version = "unknown"
        """(str): The SLP version this stream/file currently is."""
        self._allow_old_version = allow_old_version
        self._use_manual_bookends = False
        self._costumes = {0: 0, 1: 0, 2: 0, 3: 0}
        self._cpu_level = {0: 0, 1: 0, 2: 0, 3: 0}
        self._team_id = {0: 0, 1: 0, 2: 0, 3: 0}
        self._is_teams = False
        self._display_names: dict[int, str] = {}
        self._connect_codes: dict[int, str] = {}

        self.setup_gecko_codes = setup_gecko_codes
        self.online_delay = online_delay
        self.blocking_input = blocking_input
        self.fullscreen = fullscreen
        self.gfx_backend = gfx_backend
        self.disable_audio = disable_audio
        self.overclock = overclock
        self.emulation_speed = emulation_speed
        self.save_replays = save_replays
        self.replay_dir = replay_dir
        self.user_json_path = user_json_path
        self.log_level = log_level
        self.log_types = log_types
        self.infinite_time = infinite_time
        self.use_exi_inputs = use_exi_inputs
        if enable_ffw and not use_exi_inputs:
            raise ValueError("Must use exi inputs to enable ffw mode.")
        self.enable_ffw = enable_ffw
        self.dump_config = dump_config

        # Keep a running copy of the last gamestate produced
        self._prev_gamestate = GameState()
        # Half-completed gamestate not yet ready to add to the list
        self._temp_gamestate = None
        self._process = None
        if self.is_dolphin:
            self._slippstream = SlippstreamClient(self.slippi_address, self.slippi_port)
            if self.path:
                self.dolphin_version = get_dolphin_version(path)
                self.is_mainline = self.dolphin_version.mainline

                if gfx_backend == "Null":
                    if not (
                        self.is_mainline
                        or self.dolphin_version.build == DolphinBuild.EXI_AI
                    ):
                        raise ValueError(
                            "Null video requires mainline or ExiAI Ishiiruka."
                        )

                if self.dolphin_version.build is DolphinBuild.EXI_AI:
                    if not gfx_backend:
                        self.gfx_backend = "Null"
                        logging.info(
                            "ExiAI dolphin detected, setting Null video backend."
                        )
                        # TODO: modify ExiAI dolphin to use Null by default
                    elif gfx_backend != "Null":
                        # In principle ExiAI could support something like EGL...
                        raise ValueError("ExiAI dolphin requires Null video backend.")

                if (
                    self.use_exi_inputs
                    and self.dolphin_version.build != DolphinBuild.EXI_AI
                ):
                    raise ValueError(
                        "EXI inputs require a custom dolphin build. "
                        "See https://github.com/vladfi1/libmelee?tab=readme-ov-file#setup-instructions"
                    )

                self._setup_home_directory()
        else:
            self._slippstream = SLPFileStreamer(self.path)

        # Prepare some structures for fixing melee data
        path = os.path.dirname(os.path.realpath(__file__))
        with open(path + "/actiondata.csv") as csvfile:
            # A list of dicts containing the frame data
            actiondata = list(csv.DictReader(csvfile))
            # Dict of sets
            self.zero_indices = defaultdict(set)
            for line in actiondata:
                if line["zeroindex"] == "True":
                    self.zero_indices[int(line["character"])].add(int(line["action"]))

        # Read the character data csv
        self.characterdata = dict()
        with open(path + "/characterdata.csv") as csvfile:
            reader = csv.DictReader(csvfile)
            for line in reader:
                del line["Character"]
                # Convert all fields to numbers
                for key, value in line.items():
                    line[key] = float(value)
                self.characterdata[enums.Character(line["CharacterIndex"])] = line

    def connect(self):
        """Connects to the Slippi server (dolphin or wii).

        Returns:
            True is successful, False otherwise
        """
        return self._slippstream.connect()

    def _get_dolphin_home_path(self):
        """Return the path to dolphin's home directory"""
        if self.dolphin_home_path:
            return self.dolphin_home_path

        assert self.path, "Must specify a dolphin path."

        return _default_home_path(self.path)

    def _get_dolphin_config_path(self):
        """Return the path to dolphin's config directory."""
        return os.path.join(self._get_dolphin_home_path(), "Config")

    def get_dolphin_pipes_path(self, port):
        """Get the path of the named pipe input file for the given controller port"""
        if platform.system() == "Windows":
            return "\\\\.\\pipe\\slippibot" + str(port)
        pipes_path = self._get_dolphin_home_path() + "/Pipes/"
        if not os.path.isdir(pipes_path):
            os.makedirs(pipes_path, exist_ok=True)
        return pipes_path + f"slippibot{port}"

    def run(
        self,
        iso_path: Optional[str] = None,
        dolphin_user_path: Optional[str] = None,
        environment_vars: Optional[dict] = None,
        platform: Optional[str] = None,
    ):
        """Run the Dolphin emulator.

        This starts the Dolphin process, so don't run this if you're connecting to an
        already running Dolphin instance.

        Args:
            iso_path (str, optional): Path to Melee ISO for dolphin to read
            dolphin_user_path (str, optional): Alternative user path for dolphin
                if not using the default
            environment_vars (dict, optional): Dict (string->string) of environment variables to set
            exe_name (str, optional): Name of the dolphin executable.
            platform (str, optional): Set to "headless" to run dolphin in
              headless mode. Default is typically gui, depending on how
              dolphin was built. Only applies to mainline dolphin; Ishiiruka
              bakes the platform into the executable at compilation time.
        """
        assert self.is_dolphin and self.path

        exe_path = get_exe_path(self.path)
        command = [exe_path]

        if iso_path is not None:
            command.append("-e")
            command.append(iso_path)

        dolphin_user_path = dolphin_user_path or self._get_dolphin_home_path()
        command.append("-u")
        command.append(dolphin_user_path)

        if platform is not None:
            if not self.is_mainline:
                raise ValueError("Can only set platform for mainline dolphin.")
            command.append("--platform")
            command.append(platform)

        env = os.environ.copy()
        if environment_vars is not None:
            env.update(environment_vars)

        self._process = subprocess.Popen(command, env=env)

    def stop(self):
        """Stop the console.

        For Dolphin instances, this will kill the dolphin process.
        For Wiis and SLP files, it just shuts down our connection
        """
        if self.path:
            self.connected = False
            self._slippstream.shutdown()
            # If dolphin, kill the process
            if self._process is not None:
                # Sadly dolphin doesn't respect terminate
                self._process.kill()
                self._process.wait()
                self._process = None

        if self.temp_dir:
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None

    def _setup_home_directory(
        self,
    ):
        self._setup_dolphin_ini()

        if self.user_json_path:
            home_path = self._get_dolphin_home_path()
            slippi_path = os.path.join(home_path, "Slippi")
            os.makedirs(slippi_path, exist_ok=True)
            user_json_path = os.path.join(slippi_path, "user.json")
            shutil.copyfile(self.user_json_path, user_json_path)

        if self.setup_gecko_codes:
            self._setup_gecko_codes()

    def _setup_dolphin_ini(self):
        # Setup some dolphin config options
        config_path = self._get_dolphin_config_path()
        os.makedirs(config_path, exist_ok=True)
        dolphin_ini_path = os.path.join(config_path, "Dolphin.ini")

        config = configparser.ConfigParser()
        if os.path.isfile(dolphin_ini_path):
            config.read(dolphin_ini_path)

        for section in ["Core", "Input", "Display", "DSP", "Slippi", "Movie"]:
            if not config.has_section(section):
                config.add_section(section)

        if self.is_mainline:
            config.set("Slippi", "EnableSpectator", "True")
            config.set("Slippi", "SpectatorLocalPort", str(self.slippi_port))
            config.set("Slippi", "OnlineDelay", str(self.online_delay))
            config.set("Slippi", "BlockingPipes", str(self.blocking_input))

            config.set("Slippi", "SaveReplays", str(self.save_replays))
            if self.replay_dir:
                config.set("Slippi", "ReplayDir", self.replay_dir)
        else:
            config.set("Core", "SlippiEnableSpectator", "True")
            config.set("Core", "SlippiSpectatorLocalPort", str(self.slippi_port))
            config.set("Core", "SlippiOnlineDelay", str(self.online_delay))
            config.set("Core", "BlockingPipes", str(self.blocking_input))

            config.set("Core", "SlippiSaveReplays", str(self.save_replays))
            if self.replay_dir:
                config.set("Core", "SlippiReplayDir", self.replay_dir)

        # Turn on background input so we don't need to have window focus on dolphin
        config.set("Input", "backgroundinput", "True")
        config.set("Core", "GFXBackend", self.gfx_backend)
        config.set("Display", "Fullscreen", str(self.fullscreen))
        if self.disable_audio:
            disable_str = "No Audio Output" if self.is_mainline else "No audio output"
            config.set("DSP", "Backend", disable_str)

        if self.overclock:
            config.set("Core", "Overclock", str(self.overclock))
            config.set("Core", "OverclockEnable", "True")

        config.set("Core", "EmulationSpeed", str(self.emulation_speed))

        if self.dump_config:
            config.set("Movie", "DumpFrames", str(self.dump_config.dump))

        with open(dolphin_ini_path, "w") as dolphinfile:
            config.write(dolphinfile)

        # Set up logger config
        logger_ini_path = os.path.join(config_path, "Logger.ini")
        logger_config = configparser.ConfigParser()
        if os.path.isfile(logger_ini_path):
            logger_config.read(logger_ini_path)

        for section in ["Options", "Logs"]:
            if not logger_config.has_section(section):
                logger_config.add_section(section)

        logger_config.set("Options", "WriteToFile", "True")
        logger_config.set("Options", "Verbosity", str(self.log_level))

        for log_type in self.log_types:
            logger_config.set("Logs", log_type, "True")

        with open(logger_ini_path, "w") as f:
            logger_config.write(f)

        # Set up graphics config
        gfx_ini_path = os.path.join(config_path, "GFX.ini")

        gfx_config = configparser.ConfigParser()
        if os.path.isfile(gfx_ini_path):
            gfx_config.read(gfx_ini_path)

        if self.dump_config:
            self.dump_config.update_gfx_ini(gfx_config)

        with open(gfx_ini_path, "w") as f:
            gfx_config.write(f)

    def _setup_gecko_codes(self):
        ini_name = "GALE01r2.ini"

        game_settings_path = os.path.join(self._get_dolphin_home_path(), "GameSettings")
        os.makedirs(game_settings_path, exist_ok=True)
        dst_ini_path = os.path.join(game_settings_path, ini_name)

        libmelee_path = os.path.dirname(os.path.realpath(__file__))
        src_ini_path = os.path.join(libmelee_path, ini_name)
        with open(src_ini_path) as f:
            ini_text = f.read()

        extra_codes = []
        if self.infinite_time:
            extra_codes.append("$Optional: Infinite Time Mode")
        if self.use_exi_inputs:
            extra_codes.append("$Optional: Allow Bot Input Overrides")
        if self.enable_ffw:
            extra_codes.append("$Optional: FFW VS Mode")

        extra_codes = "\n".join(extra_codes)
        ini_text = ini_text.format(extra_codes=extra_codes)

        with open(dst_ini_path, "w") as f:
            f.write(ini_text)

    def setup_dolphin_controller(
        self, port, controllertype=enums.ControllerType.STANDARD
    ):
        """Setup the necessary files for dolphin to recognize the player at the given
        controller port and type"""

        pipes_path = self.get_dolphin_pipes_path(port)
        if (
            platform.system() != "Windows"
            and controllertype == enums.ControllerType.STANDARD
        ):
            if not os.path.exists(pipes_path):
                os.mkfifo(pipes_path)

        # Read in dolphin's controller config file
        controller_config_path = os.path.join(
            self._get_dolphin_config_path(), "GCPadNew.ini"
        )
        config = configparser.ConfigParser()
        config.read(controller_config_path)

        # Add a bot standard controller config to the given port
        section = "GCPad" + str(port)
        if not config.has_section(section):
            config.add_section(section)

        if controllertype == enums.ControllerType.STANDARD:
            config.set(section, "Device", "Pipe/0/slippibot" + str(port))
            config.set(section, "Buttons/A", "Button A")
            config.set(section, "Buttons/B", "Button B")
            config.set(section, "Buttons/X", "Button X")
            config.set(section, "Buttons/Y", "Button Y")
            config.set(section, "Buttons/Z", "Button Z")
            config.set(section, "Buttons/L", "Button L")
            config.set(section, "Buttons/R", "Button R")
            config.set(section, "Buttons/Threshold", "50")
            config.set(section, "Main Stick/Up", "Axis MAIN Y +")
            config.set(section, "Main Stick/Down", "Axis MAIN Y -")
            config.set(section, "Main Stick/Left", "Axis MAIN X -")
            config.set(section, "Main Stick/Right", "Axis MAIN X +")
            config.set(section, "Triggers/L", "Button L")
            config.set(section, "Triggers/R", "Button R")
            config.set(section, "Main Stick/Radius", "100")
            config.set(section, "D-Pad/Up", "Button D_UP")
            config.set(section, "D-Pad/Down", "Button D_DOWN")
            config.set(section, "D-Pad/Left", "Button D_LEFT")
            config.set(section, "D-Pad/Right", "Button D_RIGHT")
            config.set(section, "Buttons/Start", "Button START")
            config.set(section, "Buttons/A", "Button A")
            config.set(section, "C-Stick/Up", "Axis C Y +")
            config.set(section, "C-Stick/Down", "Axis C Y -")
            config.set(section, "C-Stick/Left", "Axis C X -")
            config.set(section, "C-Stick/Right", "Axis C X +")
            config.set(section, "C-Stick/Radius", "100")
            config.set(section, "Triggers/L-Analog", "Axis L +")
            config.set(section, "Triggers/R-Analog", "Axis R +")
            # Note: this actually applies to digital presses. If set to 100,
            # digital presses no longer work because the comparison is strict.
            config.set(section, "Triggers/Threshold", "90")
        # This section is unused if it's not a standard input (I think...)
        else:
            config.set(section, "Device", "XInput2/0/Virtual core pointer")

        with open(controller_config_path, "w") as configfile:
            config.write(configfile)

        dolphin_config_path = os.path.join(
            self._get_dolphin_config_path(), "Dolphin.ini"
        )
        config = configparser.ConfigParser()
        config.read(dolphin_config_path)
        # Indexed at 0. "6" means standard controller, "12" means GCN Adapter
        #  The enum is scoped to the proper value, here
        config.set("Core", "SIDevice" + str(port - 1), controllertype.value)
        with open(dolphin_config_path, "w") as dolphinfile:
            config.write(dolphinfile)

    def step(self):
        """'step' to the next state of the game and flushes all controllers

        Returns:
            GameState object that represents new current state of the game"""
        self.processingtime = time.time() - self._frametimestamp

        # Flush the controllers
        for controller in self.controllers:
            controller.flush()

        if self._temp_gamestate is None:
            self._temp_gamestate = GameState()

        frame_ended = False
        while not frame_ended:
            message = self._slippstream.dispatch(
                self._polling_mode, timeout=self._polling_timeout
            )
            if message is None:
                return None

            if message["type"] == "connect_reply":
                self.connected = True
                self.nick = message["nick"]
                self.version = message["version"]
                self.cursor = message["cursor"]

            elif message["type"] == "game_event":
                if len(message["payload"]) > 0:
                    if self.is_dolphin:
                        frame_ended = self.__handle_slippstream_events(
                            base64.b64decode(message["payload"]), self._temp_gamestate
                        )
                    else:
                        frame_ended = self.__handle_slippstream_events(
                            message["payload"], self._temp_gamestate
                        )

            elif message["type"] == "menu_event":
                if len(message["payload"]) > 0:
                    self.__handle_slippstream_menu_event(
                        base64.b64decode(message["payload"]), self._temp_gamestate
                    )
                    frame_ended = True

            elif (
                self._use_manual_bookends
                and message["type"] == "frame_end"
                and self._frame != -10000
            ):
                frame_ended = True

        gamestate = self._temp_gamestate
        self._temp_gamestate = None
        self.__fixframeindexing(gamestate)
        self.__fixiasa(gamestate)
        # Insert some metadata into the gamestate
        gamestate.playedOn = self._slippstream.playedOn
        gamestate.startAt = self._slippstream.timestamp
        gamestate.consoleNick = self._slippstream.consoleNick
        for i, names in self._slippstream.players.items():
            try:
                gamestate.players[int(i) + 1].nickName = names["names"]["netplay"]
            except KeyError:
                pass
            try:
                gamestate.players[int(i) + 1].connectCode = names["names"]["code"]
            except KeyError:
                pass

        for port, player in gamestate.players.items():
            i = port - 1
            if i in self._display_names:
                player.displayName = self._display_names[i]
            if i in self._connect_codes:
                player.connectCode = self._connect_codes[i]

        # Start the processing timer now that we're done reading messages
        self._frametimestamp = time.time()
        return gamestate

    def __handle_slippstream_events(
        self, event_bytes: bytes, gamestate: GameState
    ) -> bool:
        """Handle a series of events, provided sequentially in a byte array"""
        gamestate.menu_state = enums.Menu.IN_GAME
        mv = memoryview(event_bytes)
        while len(mv) > 0:
            command_byte = mv[0]
            event_type = EVENT_TYPE_BY_BYTE[command_byte]
            if event_type is None:
                logging.error("Got invalid event type: %s", command_byte)
                return False

            if event_type == EventType.MENU_EVENT:
                logging.error(
                    "Got a menu event in the middle of a frame. Continuing anyway."
                )
                self.__handle_slippstream_menu_event(mv.tobytes(), gamestate)
                return True

            if event_type == EventType.PAYLOADS:
                # Build payload size table once from the payload blob
                payload_size = mv[1]
                cursor = 0x2
                num_commands = (payload_size - 1) // 3
                for _ in range(num_commands):
                    cmd = _u8(mv, cursor)
                    cmd_len = _u16(mv, cursor + 0x1)
                    self.eventsize[cmd] = cmd_len + 1
                    cursor += 3
                mv = mv[payload_size + 1 :]
                continue

            event_size = self.eventsize[command_byte]
            if len(mv) < event_size:
                logging.warning(
                    "Something went wrong unpacking events. Data is probably missing"
                )
                return False

            chunk = mv[:event_size]
            mv = mv[event_size:]

            if event_type == EventType.FRAME_START:
                continue

            elif event_type == EventType.GAME_START:
                self.__game_start(gamestate, chunk)
                # First frame: flush controllers
                for controller in self.controllers:
                    controller.release_all()
                    controller.flush()

            elif event_type == EventType.GAME_END:
                return self._use_manual_bookends

            elif event_type == EventType.PRE_FRAME:
                self.__pre_frame(gamestate, chunk)

            elif event_type == EventType.POST_FRAME:
                self.__post_frame(gamestate, chunk)

            elif event_type == EventType.GECKO_CODES:
                pass

            elif event_type == EventType.FRAME_BOOKEND:
                self.__frame_bookend(gamestate, chunk)
                if gamestate.frame <= self._frame and self.skip_rollback_frames:
                    if self.blocking_input:
                        for controller in self.controllers:
                            controller.flush()
                    return False
                self._frame = gamestate.frame
                return True

            elif event_type == EventType.ITEM_UPDATE:
                self.__item_update(gamestate, chunk)

            elif event_type in [
                EventType.FOD_INFO,
                EventType.DL_INFO,
                EventType.PS_INFO,
            ]:
                expected_stage = EVENT_TO_STAGE[event_type]
                if self._current_stage is not expected_stage:
                    logging.warning(
                        "Got stage info for %s, but gamestate says %s",
                        expected_stage,
                        gamestate.stage,
                    )

            else:
                logging.error("Got an unhandled event type: %s", event_type)
                return False
        return False

    def __game_start(self, gamestate: GameState, event_bytes: bytes):
        self._frame = -10000
        major = np.ndarray((1,), ">B", event_bytes, 0x1)[0]
        minor = np.ndarray((1,), ">B", event_bytes, 0x2)[0]
        version_num = np.ndarray((1,), ">B", event_bytes, 0x3)[0]
        self.slp_version = str(major) + "." + str(minor) + "." + str(version_num)
        self._use_manual_bookends = self._allow_old_version and (
            version.parse(self.slp_version) < version.parse("3.0.0")
        )
        if major < 3 and not self._allow_old_version:
            raise SlippiVersionTooLow(self.slp_version)
        try:
            self._current_stage = enums.to_internal_stage(
                np.ndarray((1,), ">H", event_bytes, 0x13)[0]
            )
        except ValueError:
            self._current_stage = enums.Stage.NO_STAGE

        self._is_teams = not (np.ndarray((1,), ">H", event_bytes, 0xD)[0] == 0)

        for i in range(4):
            self._costumes[i] = np.ndarray((1,), ">B", event_bytes, 0x68 + (0x24 * i))[
                0
            ]

        for i in range(4):
            self._cpu_level[i] = np.ndarray((1,), ">B", event_bytes, 0x74 + (0x24 * i))[
                0
            ]

        for i in range(4):
            self._team_id[i] = np.ndarray((1,), ">B", event_bytes, 0x6E + (0x24 * i))[0]

        for i in range(4):
            if np.ndarray((1,), ">B", event_bytes, 0x66 + (0x24 * i))[0] != 1:
                self._cpu_level[i] = 0

        slp_version = (major, minor, version_num)

        if slp_version >= (3, 9, 0):
            shift_jis_hash = b"\x81\x94".decode("shift-jis")

            for i in range(4):
                self._display_names[i] = read_shift_jis(event_bytes, 0x1A5 + 0x1F * i)

                connect_code = read_shift_jis(event_bytes, 0x221 + 0xA * i)
                self._connect_codes[i] = connect_code.replace(shift_jis_hash, "#")

    def __pre_frame(self, gamestate: GameState, event_bytes: bytes) -> None:
        mv = memoryview(event_bytes)
        controller_port = mv[0x5] + 1

        if controller_port not in gamestate.players:
            gamestate.players[controller_port] = PlayerState()
        ps = gamestate.players[controller_port]

        # Nana?
        if mv[0x6] == 1:
            ps.nana = PlayerState()
            ps = ps.nana

        stick_vals = _S_F4.unpack_from(event_bytes, 0x19)
        ps.controller_state.main_stick = (
            (stick_vals[0] / 2.0) + 0.5,
            (stick_vals[1] / 2.0) + 0.5,
        )
        ps.controller_state.c_stick = (
            (stick_vals[2] / 2.0) + 0.5,
            (stick_vals[3] / 2.0) + 0.5,
        )

        # Raw main stick (added in later versions)
        raw_main_x = _S_b.unpack_from(event_bytes, 0x3B)[0] if len(mv) > 0x3B else 0
        raw_main_y = _S_b.unpack_from(event_bytes, 0x40)[0] if len(mv) > 0x40 else 0
        ps.controller_state.raw_main_stick = (raw_main_x, raw_main_y)

        trigger = _S_F.unpack_from(event_bytes, 0x29)[0]
        ps.controller_state.l_shoulder = trigger
        ps.controller_state.r_shoulder = trigger

        buttonbits = int.from_bytes(event_bytes[0x31:0x33], "big")
        btn = ps.controller_state.button
        btn[enums.Button.BUTTON_A] = bool(buttonbits & 0x0100)
        btn[enums.Button.BUTTON_B] = bool(buttonbits & 0x0200)
        btn[enums.Button.BUTTON_X] = bool(buttonbits & 0x0400)
        btn[enums.Button.BUTTON_Y] = bool(buttonbits & 0x0800)
        btn[enums.Button.BUTTON_START] = bool(buttonbits & 0x1000)
        btn[enums.Button.BUTTON_Z] = bool(buttonbits & 0x0010)
        btn[enums.Button.BUTTON_R] = bool(buttonbits & 0x0020)
        btn[enums.Button.BUTTON_L] = bool(buttonbits & 0x0040)

        if self._use_manual_bookends:
            self._frame = gamestate.frame

    def __post_frame(self, gamestate: GameState, event_bytes: bytes) -> None:
        mv = memoryview(event_bytes)
        gs = gamestate
        gs.stage = self._current_stage
        gs.is_teams = self._is_teams
        gs.frame = _S_I.unpack_from(event_bytes, 0x1)[0]
        controller_port = mv[0x5] + 1

        if controller_port not in gs.players:
            gs.players[controller_port] = PlayerState()
        ps = gs.players[controller_port]

        # Nana?
        if mv[0x6] == 1:
            ps.nana = PlayerState()
            ps = ps.nana

        # Position
        pos_x, pos_y = _S_2F.unpack_from(event_bytes, 0x0A)
        ps.position.x = pos_x
        ps.position.y = pos_y

        # Enums via value map (avoid ValueError)
        ch_val = mv[0x7]
        ps.character = enums.Character._value2member_map_.get(
            ch_val, enums.Character.UNKNOWN_CHARACTER
        )  # type: ignore[attr-defined]

        act_val = _S_H.unpack_from(event_bytes, 0x8)[0]
        ps.action = enums.Action._value2member_map_.get(
            act_val, enums.Action.UNKNOWN_ANIMATION
        )  # type: ignore[attr-defined]

        # Facing from float sign
        ps.facing = _S_F.unpack_from(event_bytes, 0x12)[0] > 0.0

        percent_val, shield_strength = _S_2F.unpack_from(event_bytes, 0x16)
        ps.percent = int(percent_val)
        ps.shield_strength = shield_strength
        ps.stock = mv[0x21]
        ps.action_frame = int(_S_F.unpack_from(event_bytes, 0x22)[0])

        # Status bytes (read once)
        if len(mv) > 0x2A:
            sb1, sb2, sb3, sb4, sb5 = event_bytes[0x26:0x2B]
        else:
            sb1 = sb2 = sb3 = sb4 = sb5 = 0

        ps.is_reflect_active = (sb1 & 0x10) != 0
        ps.is_subaction_invulnerable = (sb2 & 0x04) != 0
        ps.is_fastfalling = (sb2 & 0x08) != 0
        ps.is_defender_in_hitlag = (sb2 & 0x10) != 0
        ps.is_in_hitlag = (sb2 & 0x20) != 0
        ps.is_holding_character = (sb3 & 0x04) != 0
        ps.is_shield_active = (sb3 & 0x80) != 0
        ps.is_in_hitstun = (sb4 & 0x02) != 0
        ps.is_powershield = (sb4 & 0x20) != 0
        ps.is_dead = (sb5 & 0x40) != 0
        ps.is_offscreen = (sb5 & 0x80) != 0

        # Scalars with length guards (avoid try/except in hot path)
        blen = len(mv)
        ps.hitstun_frames_left = (
            int(_S_F.unpack_from(event_bytes, 0x2B)[0]) if blen > 0x2E else 0
        )
        ps.on_ground = (mv[0x2F] == 0) if blen > 0x2F else True
        if blen > 0x32:
            ps.jumps_left = mv[0x32]
        else:
            ps.jumps_left = 1
        if blen > 0x33:
            ps.l_cancel_status = mv[0x33]
        else:
            ps.l_cancel_status = 0
        if blen > 0x34:
            ps.invulnerable = mv[0x34] != 0
        else:
            ps.invulnerable = False

        # TODO: Verify this processing is correct
        if blen > 0x4C:
            air_x, y_self, x_attack, y_attack, ground_x, hitlag = _S_6F.unpack_from(
                event_bytes, 0x35
            )
            ps.speed_air_x_self = air_x
            ps.speed_y_self = y_self
            ps.speed_x_attack = x_attack
            ps.speed_y_attack = y_attack
            ps.speed_ground_x_self = ground_x
            ps.hitlag_left = int(hitlag)
        else:
            ps.speed_air_x_self = (
                _S_F.unpack_from(event_bytes, 0x35)[0] if blen > 0x38 else 0.0
            )
            ps.speed_y_self = (
                _S_F.unpack_from(event_bytes, 0x39)[0] if blen > 0x3C else 0.0
            )
            ps.speed_x_attack = (
                _S_F.unpack_from(event_bytes, 0x3D)[0] if blen > 0x40 else 0.0
            )
            ps.speed_y_attack = (
                _S_F.unpack_from(event_bytes, 0x41)[0] if blen > 0x44 else 0.0
            )
            ps.speed_ground_x_self = (
                _S_F.unpack_from(event_bytes, 0x45)[0] if blen > 0x48 else 0.0
            )
            ps.hitlag_left = (
                int(_S_F.unpack_from(event_bytes, 0x49)[0]) if blen > 0x4C else 0
            )

        # TODO: Maybe calculate this ourselves in a vectorized way, later?
        edge = stages.EDGE_GROUND_POSITION.get(gs.stage)
        if (
            edge is not None
            and (abs(ps.position.x) > edge or ps.position.y < -6)
            and not ps.on_ground
        ):
            ps.off_stage = True
        else:
            ps.off_stage = False

        # ECB (8 floats: top/bottom/left/right x,y) 0x4D..0x6C
        if blen >= 0x6D:
            dt = np.dtype(">f4")
            ecb = np.frombuffer(mv, dtype=dt, count=8, offset=0x4D)
            tx, ty, bx, by, lx, ly, rx, ry = map(float, ecb)
        else:
            tx = ty = bx = by = lx = ly = rx = ry = 0.0

        ps.ecb.top.x = tx
        ps.ecb.top.y = ty
        ps.ecb_top = (tx, ty)
        ps.ecb.bottom.x = bx
        ps.ecb.bottom.y = by
        ps.ecb_bottom = (bx, by)
        ps.ecb.left.x = lx
        ps.ecb.left.y = ly
        ps.ecb_left = (lx, ly)
        ps.ecb.right.x = rx
        ps.ecb.right.y = ry
        ps.ecb_right = (rx, ry)

        if self._use_manual_bookends:
            self._frame = gs.frame

    def __frame_bookend(self, gamestate, event_bytes):
        self._prev_gamestate = gamestate
        # Calculate helper distance variable
        #   This is a bit kludgey.... :/
        i = 0
        player_one_x, player_one_y, player_two_x, player_two_y = 0, 0, 0, 0
        for _, player_state in gamestate.players.items():
            if i == 0:
                player_one_x, player_one_y = (
                    player_state.position.x,
                    player_state.position.y,
                )
            if i == 1:
                player_two_x, player_two_y = (
                    player_state.position.x,
                    player_state.position.y,
                )
            i += 1
        xdist = player_one_x - player_two_x
        ydist = player_one_y - player_two_y
        gamestate.distance = math.sqrt((xdist**2) + (ydist**2))

    def __item_update(self, gamestate, event_bytes):
        projectile = Projectile()
        projectile.position.x = np.ndarray((1,), ">f", event_bytes, 0x14)[0]
        projectile.position.y = np.ndarray((1,), ">f", event_bytes, 0x18)[0]
        projectile.x = projectile.position.x
        projectile.y = projectile.position.y
        projectile.speed.x = np.ndarray((1,), ">f", event_bytes, 0xC)[0]
        projectile.speed.y = np.ndarray((1,), ">f", event_bytes, 0x10)[0]
        projectile.x_speed = projectile.speed.x
        projectile.y_speed = projectile.speed.y
        try:
            projectile.owner = np.ndarray((1,), ">B", event_bytes, 0x2A)[0] + 1
            if projectile.owner > 4:
                projectile.owner = -1
        except TypeError:
            projectile.owner = -1
        try:
            projectile.type = enums.ProjectileType(
                np.ndarray((1,), ">H", event_bytes, 0x5)[0]
            )
        except ValueError:
            projectile.type = enums.ProjectileType.UNKNOWN_PROJECTILE

        try:
            projectile.frame = int(np.ndarray((1,), ">f", event_bytes, 0x1E)[0])
        except ValueError:
            projectile.frame = -1

        projectile.subtype = np.ndarray((1,), ">B", event_bytes, 0x7)[0]

        # Ignore exploded Samus bombs. They are subtype 3
        if (
            projectile.type == enums.ProjectileType.SAMUS_BOMB
            and projectile.subtype == 3
        ):
            return
        # Ignore exploded Samus missles
        if (
            projectile.type == enums.ProjectileType.SAMUS_MISSLE
            and projectile.subtype in [2, 3]
        ):
            return
        # Ignore Samus charge beam while charging (not firing)
        if (
            projectile.type == enums.ProjectileType.SAMUS_CHARGE_BEAM
            and projectile.subtype == 0
        ):
            return

        # Add the projectile to the gamestate list
        gamestate.projectiles.append(projectile)

    def __handle_slippstream_menu_event(
        self, event_bytes: bytes, gamestate: GameState
    ) -> None:
        mv = memoryview(event_bytes)
        scene = _u16(mv, 0x1)
        if scene == 0x02:
            gamestate.menu_state = enums.Menu.CHARACTER_SELECT
            # All ports active
            gamestate.players[1] = PlayerState()
            gamestate.players[2] = PlayerState()
            gamestate.players[3] = PlayerState()
            gamestate.players[4] = PlayerState()
        elif scene in [0x0102, 0x0108]:
            gamestate.menu_state = enums.Menu.STAGE_SELECT
            gamestate.players[1] = PlayerState()
            gamestate.players[2] = PlayerState()
            gamestate.players[3] = PlayerState()
            gamestate.players[4] = PlayerState()
        elif scene == 0x0202:
            gamestate.menu_state = enums.Menu.IN_GAME
        elif scene == 0x0001:
            gamestate.menu_state = enums.Menu.MAIN_MENU
        elif scene == 0x0008:
            gamestate.menu_state = enums.Menu.SLIPPI_ONLINE_CSS
            gamestate.players[1] = PlayerState()
            gamestate.players[2] = PlayerState()
            gamestate.players[3] = PlayerState()
            gamestate.players[4] = PlayerState()
        elif scene == 0x0000:
            gamestate.menu_state = enums.Menu.PRESS_START
        else:
            gamestate.menu_state = enums.Menu.UNKNOWN_MENU

        # controller port statuses at CSS
        if gamestate.menu_state in [
            enums.Menu.CHARACTER_SELECT,
            enums.Menu.SLIPPI_ONLINE_CSS,
        ]:
            gamestate.players[1].controller_status = enums.ControllerStatus(
                _u8(mv, 0x25)
            )
            gamestate.players[2].controller_status = enums.ControllerStatus(
                _u8(mv, 0x26)
            )
            gamestate.players[3].controller_status = enums.ControllerStatus(
                _u8(mv, 0x27)
            )
            gamestate.players[4].controller_status = enums.ControllerStatus(
                _u8(mv, 0x28)
            )

            # CSS Cursors
            gamestate.players[1].cursor_x = np.ndarray((1,), ">f", event_bytes, 0x3)[0]
            gamestate.players[1].cursor_y = np.ndarray((1,), ">f", event_bytes, 0x7)[0]
            gamestate.players[2].cursor_x = np.ndarray((1,), ">f", event_bytes, 0xB)[0]
            gamestate.players[2].cursor_y = np.ndarray((1,), ">f", event_bytes, 0xF)[0]
            gamestate.players[3].cursor_x = np.ndarray((1,), ">f", event_bytes, 0x13)[0]
            gamestate.players[3].cursor_y = np.ndarray((1,), ">f", event_bytes, 0x17)[0]
            gamestate.players[4].cursor_x = np.ndarray((1,), ">f", event_bytes, 0x1B)[0]
            gamestate.players[4].cursor_y = np.ndarray((1,), ">f", event_bytes, 0x1F)[0]

            gamestate.ready_to_start = _u8(mv, 0x23)

            # Coin down
            try:
                gamestate.players[1].coin_down = (
                    np.ndarray((1,), ">B", event_bytes, 0x2D)[0] == 2
                )
            except TypeError:
                gamestate.players[1].coin_down = False
            try:
                gamestate.players[2].coin_down = (
                    np.ndarray((1,), ">B", event_bytes, 0x2E)[0] == 2
                )
            except TypeError:
                gamestate.players[2].coin_down = False
            try:
                gamestate.players[3].coin_down = (
                    np.ndarray((1,), ">B", event_bytes, 0x2F)[0] == 2
                )
            except TypeError:
                gamestate.players[3].coin_down = False
            try:
                gamestate.players[4].coin_down = (
                    np.ndarray((1,), ">B", event_bytes, 0x30)[0] == 2
                )
            except TypeError:
                gamestate.players[4].coin_down = False

            # Character selected
            try:
                gamestate.players[1].character = enums.to_internal(_u8(mv, 0x29))
                gamestate.players[1].character_selected = gamestate.players[1].character
            except TypeError:
                gamestate.players[1].character = enums.Character.UNKNOWN_CHARACTER
                gamestate.players[
                    1
                ].character_selected = enums.Character.UNKNOWN_CHARACTER
            try:
                gamestate.players[2].character = enums.to_internal(_u8(mv, 0x2A))
                gamestate.players[2].character_selected = gamestate.players[2].character
            except TypeError:
                gamestate.players[2].character = enums.Character.UNKNOWN_CHARACTER
                gamestate.players[
                    2
                ].character_selected = enums.Character.UNKNOWN_CHARACTER
            try:
                gamestate.players[3].character = enums.to_internal(_u8(mv, 0x2B))
                gamestate.players[3].character_selected = gamestate.players[3].character
            except TypeError:
                gamestate.players[3].character = enums.Character.UNKNOWN_CHARACTER
                gamestate.players[
                    3
                ].character_selected = enums.Character.UNKNOWN_CHARACTER
            try:
                gamestate.players[4].character = enums.to_internal(_u8(mv, 0x2C))
                gamestate.players[4].character_selected = gamestate.players[4].character
            except TypeError:
                gamestate.players[4].character = enums.Character.UNKNOWN_CHARACTER
                gamestate.players[
                    4
                ].character_selected = enums.Character.UNKNOWN_CHARACTER

        if gamestate.menu_state == enums.Menu.STAGE_SELECT:
            try:
                gamestate.stage = enums.Stage(_u8(mv, 0x24))
            except ValueError:
                gamestate.stage = enums.Stage.NO_STAGE

            # Stage Select Cursor X, Y
            for player in gamestate.players.values():
                player.cursor.x = np.ndarray((1,), ">f", event_bytes, 0x31)[0]
                player.cursor.y = np.ndarray((1,), ">f", event_bytes, 0x35)[0]
                gamestate.stage_select_cursor_x = player.cursor.x
                gamestate.stage_select_cursor_y = player.cursor.y

        # Frame count
        gamestate.frame = _i32(mv, 0x39)

        # Sub-menu
        if len(mv) > 0x3D:
            try:
                gamestate.submenu = enums.SubMenu(_u8(mv, 0x3D))
            except (TypeError, ValueError, struct.error):
                gamestate.submenu = enums.SubMenu.UNKNOWN_SUBMENU
        else:
            gamestate.submenu = enums.SubMenu.UNKNOWN_SUBMENU

        # Selected menu
        gamestate.menu_selection = _u8(mv, 0x3E) if len(mv) > 0x3E else 0

        # Nametag / submenu switching
        if gamestate.menu_state == enums.Menu.SLIPPI_ONLINE_CSS and len(mv) > 0x40:
            nametag = _u8(mv, 0x40)
            if nametag == 0x05:
                gamestate.submenu = enums.SubMenu.NAME_ENTRY_SUBMENU
            elif nametag == 0x00:
                gamestate.submenu = enums.SubMenu.ONLINE_CSS

        # Is Holding CPU Slider
        try:
            for i in range(4):
                gamestate.players[i + 1].is_holding_cpu_slider = np.ndarray(
                    (1,), ">B", event_bytes, 0x45 + i
                )[0]
        except TypeError:
            pass
        except KeyError:
            pass

        try:
            for i in range(4):
                gamestate.players[i + 1].cpu_level = np.ndarray(
                    (1,), ">B", event_bytes, 0x41 + i
                )[0]
        except TypeError:
            pass
        except KeyError:
            pass

    def __fixframeindexing(self, gamestate: GameState):
        """Melee's indexing of action frames is wildly inconsistent.
        Here we adjust all of the frames to be indexed at 1 (so math is easier)"""
        for _, player in gamestate.players.items():
            if player.action.value in self.zero_indices[player.character.value]:
                player.action_frame = player.action_frame + 1

    def __fixiasa(self, gamestate: GameState):
        """The IASA flag doesn't set or reset for special attacks.
        So let's just set IASA to False for all non-A attacks.
        """
        for _, player in gamestate.players.items():
            # Luckily for us, all the A-attacks are in a contiguous place in the enums!
            #   So we don't need to call them out one by one
            if (
                player.action.value < Action.NEUTRAL_ATTACK_1.value
                or player.action.value > Action.DAIR.value
            ):
                player.iasa = False
