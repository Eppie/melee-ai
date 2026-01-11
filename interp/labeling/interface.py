"""
Terminal interface for frame labeling.

Provides an interactive terminal UI for labeling game frames with
probe categories. Displays frame information, model predictions,
and handles keyboard input for efficient labeling.

Usage:
    from interp.labeling import TerminalLabeler, FrameSampler, LabelStorage
    from interp.probes import MELEE_PROBES

    # Sample frames
    sampler = FrameSampler(model, dataset, colmap, device)
    frames = sampler.sample(100, strategy="diversity")

    # Create labeler
    storage = LabelStorage("labels/game_phase.json", probe_name="game_phase")
    config = MELEE_PROBES["game_phase"]
    labeler = TerminalLabeler(config, storage, frames)

    # Start labeling
    labeler.run()
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from interp.labeling.sampler import SampledFrame, SamplingResult
    from interp.labeling.storage import LabelStorage
    from interp.probes import ProbeConfig


# Character name mapping (subset)
CHARACTER_NAMES = {
    0: "Falcon",
    1: "DK",
    2: "Fox",
    3: "GnW",
    4: "Kirby",
    5: "Bowser",
    6: "Link",
    7: "Luigi",
    8: "Mario",
    9: "Marth",
    10: "Mewtwo",
    11: "Ness",
    12: "Peach",
    13: "Pikachu",
    14: "ICs",
    15: "Puff",
    16: "Samus",
    17: "Yoshi",
    18: "Zelda",
    19: "Sheik",
    20: "Falco",
    21: "YLink",
    22: "Doc",
    23: "Roy",
    24: "Pichu",
    25: "Ganon",
}

# Stage name mapping (subset)
STAGE_NAMES = {
    0: "Fountain",
    1: "Stadium",
    2: "Peach's",
    3: "Kongo",
    4: "Brinstar",
    5: "Corneria",
    6: "Yoshi's",
    7: "Onett",
    8: "Mute City",
    9: "Rainbow",
    10: "Jungle",
    11: "Temple",
    12: "Yoshi's Island",
    13: "Green Greens",
    14: "Fourside",
    15: "Mushroom I",
    16: "Mushroom II",
    17: "Akaneia",
    18: "Venom",
    19: "PokeFloats",
    20: "Big Blue",
    21: "Icicle",
    22: "Icetop",
    23: "Flat Zone",
    24: "Dream Land",
    25: "Yoshis Story",
    26: "Battlefield",
    27: "FD",
}

# Action state names (common ones)
ACTION_NAMES = {
    0: "DEAD_DOWN",
    1: "DEAD_LEFT",
    2: "DEAD_RIGHT",
    3: "DEAD_UP",
    4: "DEAD_FLY_STAR",
    5: "DEAD_FLY",
    6: "DEAD_FLY_SPLATTER",
    7: "DEAD_FLY_HIT",
    10: "SLEEP",
    11: "REBIRTH",
    12: "REBIRTH_WAIT",
    13: "WAIT",
    14: "WALK_SLOW",
    15: "WALK_MIDDLE",
    16: "WALK_FAST",
    17: "TURN",
    18: "TURN_RUN",
    19: "DASH",
    20: "RUN",
    21: "RUN_DIRECT",
    22: "RUN_BRAKE",
    23: "KNEE_BEND",
    24: "JUMP_F",
    25: "JUMP_B",
    26: "JUMP_AERIAL_F",
    27: "JUMP_AERIAL_B",
    28: "FALL",
    29: "FALL_F",
    30: "FALL_B",
    31: "FALL_AERIAL",
    32: "FALL_AERIAL_F",
    33: "FALL_AERIAL_B",
    34: "FALL_SPECIAL",
    35: "FALL_SPECIAL_F",
    36: "FALL_SPECIAL_B",
    37: "DAMAGE_FALL",
    38: "SQUAT",
    39: "SQUAT_WAIT",
    40: "SQUAT_RV",
    41: "LANDING",
    42: "LANDING_FALL_SPECIAL",
    43: "ATTACK_11",
    44: "ATTACK_12",
    45: "ATTACK_13",
    46: "ATTACK_100_START",
    47: "ATTACK_100_LOOP",
    48: "ATTACK_100_END",
    49: "ATTACK_DASH",
    50: "ATTACK_S3_HI",
    51: "ATTACK_S3_HI_S",
    52: "ATTACK_S3_S",
    53: "ATTACK_S3_LW_S",
    54: "ATTACK_S3_LW",
    55: "ATTACK_HI3",
    56: "ATTACK_LW3",
    57: "ATTACK_S4_HI",
    58: "ATTACK_S4_HI_S",
    59: "ATTACK_S4_S",
    60: "ATTACK_S4_LW_S",
    61: "ATTACK_S4_LW",
    62: "ATTACK_HI4",
    63: "ATTACK_LW4",
    64: "ATTACK_AIR_N",
    65: "ATTACK_AIR_F",
    66: "ATTACK_AIR_B",
    67: "ATTACK_AIR_HI",
    68: "ATTACK_AIR_LW",
    69: "LANDING_AIR_N",
    70: "LANDING_AIR_F",
    71: "LANDING_AIR_B",
    72: "LANDING_AIR_HI",
    73: "LANDING_AIR_LW",
    74: "DAMAGE_HI1",
    178: "GUARD_ON",
    179: "GUARD",
    180: "GUARD_OFF",
    181: "GUARD_SET_OFF",
    182: "GUARD_REFLECT",
    183: "DOWN_BOUND_U",
    184: "DOWN_WAIT_U",
    185: "DOWN_DAMAGE_U",
    186: "DOWN_STAND_U",
    187: "DOWN_ATTACK_U",
    188: "DOWN_FOWARD_U",
    189: "DOWN_BACK_U",
    190: "DOWN_SPOT_U",
    191: "DOWN_BOUND_D",
    192: "DOWN_WAIT_D",
    212: "ESCAPE",
    213: "ESCAPE_F",
    214: "ESCAPE_B",
    215: "ESCAPE_AIR",
    219: "CATCH",
    220: "CATCH_PULL",
    221: "CATCH_DASH",
    222: "CATCH_DASH_PULL",
    223: "CATCH_WAIT",
    224: "CATCH_ATTACK",
    225: "CATCH_CUT",
    226: "THROW_F",
    227: "THROW_B",
    228: "THROW_HI",
    229: "THROW_LW",
    233: "CLIFF_CATCH",
    234: "CLIFF_WAIT",
    235: "CLIFF_CLIMB_SLOW",
    236: "CLIFF_CLIMB_QUICK",
    237: "CLIFF_ATTACK_SLOW",
    238: "CLIFF_ATTACK_QUICK",
    239: "CLIFF_ESCAPE_SLOW",
    240: "CLIFF_ESCAPE_QUICK",
    241: "CLIFF_JUMP_SLOW_1",
    242: "CLIFF_JUMP_SLOW_2",
    243: "CLIFF_JUMP_QUICK_1",
    244: "CLIFF_JUMP_QUICK_2",
}


def get_character_name(char_id: int) -> str:
    """Get character name from ID."""
    return CHARACTER_NAMES.get(char_id, f"Char_{char_id}")


def get_stage_name(stage_id: int) -> str:
    """Get stage name from ID."""
    return STAGE_NAMES.get(stage_id, f"Stage_{stage_id}")


def get_action_name(action_id: int) -> str:
    """Get action name from ID."""
    return ACTION_NAMES.get(action_id, f"Action_{action_id}")


def clear_screen():
    """Clear terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def get_single_char() -> str:
    """Get a single character from stdin without waiting for enter."""
    import tty
    import termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


@dataclass
class LabelingStats:
    """Statistics for a labeling session."""

    total_frames: int = 0
    frames_labeled: int = 0
    frames_skipped: int = 0
    label_distribution: Dict[str, int] = None

    def __post_init__(self):
        if self.label_distribution is None:
            self.label_distribution = {}

    @property
    def progress_pct(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return (self.frames_labeled + self.frames_skipped) / self.total_frames * 100


class TerminalLabeler:
    """
    Interactive terminal labeler for game frames.

    Displays frame information and accepts keyboard input for
    fast labeling of probe categories.
    """

    def __init__(
        self,
        config: "ProbeConfig",
        storage: "LabelStorage",
        frames: List["SampledFrame"],
        show_model_predictions: bool = True,
    ):
        """
        Initialize labeler.

        Args:
            config: Probe configuration (defines classes)
            storage: Label storage for persistence
            frames: Frames to label (from FrameSampler)
            show_model_predictions: Show model predictions if available
        """
        self.config = config
        self.storage = storage
        self.frames = frames
        self.show_predictions = show_model_predictions

        self.current_idx = 0
        self.stats = LabelingStats(total_frames=len(frames))

        # Skip already labeled frames
        self._skip_labeled()

    def _skip_labeled(self) -> None:
        """Skip to first unlabeled frame."""
        while self.current_idx < len(self.frames):
            frame = self.frames[self.current_idx]
            if not self.storage.has_label(frame.frame_id):
                break
            self.stats.frames_labeled += 1
            self.current_idx += 1

    @property
    def current_frame(self) -> Optional["SampledFrame"]:
        """Get current frame to label."""
        if self.current_idx < len(self.frames):
            return self.frames[self.current_idx]
        return None

    def _format_frame_display(self, frame: "SampledFrame") -> str:
        """Format frame information for display."""
        lines = []

        # Header
        lines.append("=" * 60)
        lines.append(
            f"FRAME {self.current_idx + 1} of {len(self.frames)} | "
            f"{frame.frame_id}"
        )
        lines.append("=" * 60)

        # Stage
        stage = frame.stage or "Unknown"
        if stage.startswith("stage_"):
            stage_id = int(stage.split("_")[1])
            stage = get_stage_name(stage_id)
        lines.append(f"\nStage: {stage}")

        # Player 1 info
        p1_char = frame.p1_character or "Unknown"
        if p1_char.startswith("char_"):
            char_id = int(p1_char.split("_")[1])
            p1_char = get_character_name(char_id)

        p1_action = frame.p1_action or "Unknown"
        if p1_action.startswith("action_"):
            action_id = int(p1_action.split("_")[1])
            p1_action = get_action_name(action_id)

        p1_pos = frame.p1_position
        lines.append(
            f"\nP1 ({p1_char}): {p1_action}"
        )
        lines.append(
            f"   Position: ({p1_pos[0]:.1f}, {p1_pos[1]:.1f}) | "
            f"{frame.p1_percent:.0f}% | {frame.p1_stocks} stocks"
        )

        # Player 2 info
        p2_char = frame.p2_character or "Unknown"
        if p2_char.startswith("char_"):
            char_id = int(p2_char.split("_")[1])
            p2_char = get_character_name(char_id)

        p2_action = frame.p2_action or "Unknown"
        if p2_action.startswith("action_"):
            action_id = int(p2_action.split("_")[1])
            p2_action = get_action_name(action_id)

        p2_pos = frame.p2_position
        lines.append(
            f"\nP2 ({p2_char}): {p2_action}"
        )
        lines.append(
            f"   Position: ({p2_pos[0]:.1f}, {p2_pos[1]:.1f}) | "
            f"{frame.p2_percent:.0f}% | {frame.p2_stocks} stocks"
        )

        # Model predictions (if available)
        if self.show_predictions and frame.model_prediction:
            lines.append("\nModel predictions:")
            for head, pred in frame.model_prediction.items():
                conf = frame.model_confidence.get(head, 0) if frame.model_confidence else 0
                lines.append(f"   {head}: {pred} ({conf:.1%})")

        # Sampling info
        if frame.sampling_score != 0:
            lines.append(f"\nSampling score: {frame.sampling_score:.3f}")
        if frame.cluster_id is not None:
            lines.append(f"Cluster: {frame.cluster_id}")

        return "\n".join(lines)

    def _format_prompt(self) -> str:
        """Format the labeling prompt."""
        lines = ["\n" + "-" * 60]
        lines.append(f"Label for: {self.config.name}")
        lines.append(f"Description: {self.config.description}")
        lines.append("")

        for i, class_name in enumerate(self.config.class_names):
            lines.append(f"  [{i + 1}] {class_name}")

        lines.append("")
        lines.append("  [s] Skip  [b] Back  [q] Quit  [?] Help")
        lines.append("-" * 60)
        lines.append(f"\nProgress: {self.stats.frames_labeled}/{len(self.frames)} "
                     f"({self.stats.progress_pct:.1f}%)")

        return "\n".join(lines)

    def _format_help(self) -> str:
        """Format help text."""
        return """
LABELING CONTROLS
=================

Number keys (1-9): Select class label
s: Skip this frame
b: Go back to previous frame
q: Quit and save
?: Show this help

The label is saved automatically when you select a class.
Progress is saved periodically.

Press any key to continue...
"""

    def _format_stats(self) -> str:
        """Format session statistics."""
        lines = ["\n" + "=" * 60, "LABELING SESSION COMPLETE", "=" * 60]
        lines.append(f"\nTotal frames: {len(self.frames)}")
        lines.append(f"Labeled: {self.stats.frames_labeled}")
        lines.append(f"Skipped: {self.stats.frames_skipped}")

        if self.stats.label_distribution:
            lines.append("\nLabel distribution:")
            for label_name, count in sorted(self.stats.label_distribution.items()):
                pct = count / self.stats.frames_labeled * 100 if self.stats.frames_labeled > 0 else 0
                lines.append(f"  {label_name}: {count} ({pct:.1f}%)")

        return "\n".join(lines)

    def label_frame(self, frame: "SampledFrame", label_idx: int) -> None:
        """Apply label to current frame."""
        label_name = self.config.class_names[label_idx]

        self.storage.add_label(
            frame_id=frame.frame_id,
            label=label_idx,
            label_name=label_name,
            probe_name=self.config.name,
            activations=frame.activations,
            episode_idx=frame.episode_idx,
            frame_idx=frame.frame_idx,
            p1_character=frame.p1_character,
            p2_character=frame.p2_character,
            p1_action=frame.p1_action,
            p2_action=frame.p2_action,
            p1_percent=frame.p1_percent,
            p2_percent=frame.p2_percent,
            p1_stocks=frame.p1_stocks,
            p2_stocks=frame.p2_stocks,
            stage=frame.stage,
        )

        self.stats.frames_labeled += 1
        self.stats.label_distribution[label_name] = (
            self.stats.label_distribution.get(label_name, 0) + 1
        )

    def run(self) -> LabelingStats:
        """
        Run the interactive labeling session.

        Returns:
            LabelingStats with session statistics
        """
        # Start session
        self.storage.start_session()

        try:
            while self.current_idx < len(self.frames):
                frame = self.current_frame
                if frame is None:
                    break

                # Skip if already labeled
                if self.storage.has_label(frame.frame_id):
                    self.current_idx += 1
                    continue

                # Display frame
                clear_screen()
                print(self._format_frame_display(frame))
                print(self._format_prompt())

                # Get input
                key = get_single_char()

                # Handle input
                if key == "q":
                    break
                elif key == "?":
                    clear_screen()
                    print(self._format_help())
                    get_single_char()
                elif key == "s":
                    self.stats.frames_skipped += 1
                    self.current_idx += 1
                elif key == "b":
                    if self.current_idx > 0:
                        self.current_idx -= 1
                        # Remove previous label if exists
                        prev_frame = self.frames[self.current_idx]
                        if self.storage.has_label(prev_frame.frame_id):
                            label = self.storage.get_label(prev_frame.frame_id)
                            if label:
                                self.stats.label_distribution[label.label_name] = (
                                    self.stats.label_distribution.get(label.label_name, 1) - 1
                                )
                            self.storage.remove_label(prev_frame.frame_id)
                            self.stats.frames_labeled -= 1
                elif key.isdigit():
                    label_idx = int(key) - 1
                    if 0 <= label_idx < self.config.num_classes:
                        self.label_frame(frame, label_idx)
                        self.current_idx += 1

        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")

        finally:
            # Save and show stats
            self.storage.end_session(self.stats.frames_labeled)
            self.storage.save()

            clear_screen()
            print(self._format_stats())

        return self.stats


def quick_label(
    config: "ProbeConfig",
    frames: List["SampledFrame"],
    output_path: str,
    labeler_name: str = "unknown",
) -> "LabelStorage":
    """
    Convenience function for quick labeling session.

    Args:
        config: Probe configuration
        frames: Frames to label
        output_path: Path for label storage
        labeler_name: Name of labeler

    Returns:
        LabelStorage with labels
    """
    from interp.labeling.storage import LabelStorage

    storage = LabelStorage(
        output_path,
        probe_name=config.name,
        labeler=labeler_name,
    )

    labeler = TerminalLabeler(config, storage, frames)
    labeler.run()

    return storage


__all__ = [
    "TerminalLabeler",
    "LabelingStats",
    "quick_label",
    "get_character_name",
    "get_stage_name",
    "get_action_name",
]
