#!/usr/bin/env python3
"""Visualize future position prediction probabilities on stage canvas.

This script loads a replay file, runs model inference frame-by-frame, and generates
a visualization showing the probability distributions for future X/Y positions at
multiple time horizons (10, 20, 30, 40, 50, 60 frames ahead).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F_torch
from matplotlib.patches import Rectangle

from column_map import ColumnMap
from config import init_config
from controller_quantization import FUTURE_X_BUCKETS, FUTURE_Y_BUCKETS
from libmelee.melee import enums
from libmelee.melee.console import Console
from libmelee.melee.stages import BLASTZONES, top_platform_position, right_platform_position, left_platform_position
from model_interface import GPTInferenceEngine, collect_raw_inputs_from_gamestate
from schema import get_feature_names, get_target_names
from train.batch_utils import build_model_inputs


# Stage names for display
STAGE_NAMES = {
    enums.Stage.BATTLEFIELD: "Battlefield",
    enums.Stage.FINAL_DESTINATION: "Final Destination",
    enums.Stage.DREAMLAND: "Dreamland",
    enums.Stage.FOUNTAIN_OF_DREAMS: "Fountain of Dreams",
    enums.Stage.POKEMON_STADIUM: "Pokemon Stadium",
    enums.Stage.YOSHIS_STORY: "Yoshi's Story",
}



def stage_edge_positions(stage: enums.Stage) -> tuple[Optional[float], Optional[float]]:
    """Gets the left and right edge positions of the main stage platform (ledges).

    Args:
        stage: The current stage

    Returns:
        Tuple of (left edge, right edge). (None, None) if unknown
    """
    # Stage ledge positions (where characters grab ledges)
    STAGE_EDGES = {
        enums.Stage.BATTLEFIELD: (-68.4, 68.4),
        enums.Stage.FINAL_DESTINATION: (-85.5656, 85.5656),
        enums.Stage.DREAMLAND: (-77.2713, 77.2713),
        enums.Stage.FOUNTAIN_OF_DREAMS: (-63.3499, 63.3499),
        enums.Stage.POKEMON_STADIUM: (-87.7501, 87.7501),
        enums.Stage.YOSHIS_STORY: (-56.0, 56.0),
    }

    return STAGE_EDGES.get(stage, (None, None))


def get_stage_bounds(stage: enums.Stage) -> tuple[float, float, float, float]:
    """Get blastzone boundaries for a stage."""
    if stage not in BLASTZONES:
        # Default to FD if unknown
        return BLASTZONES[enums.Stage.FINAL_DESTINATION]
    return BLASTZONES[stage]


def get_gameplay_bounds(stage: enums.Stage) -> tuple[float, float, float, float]:
    """Get focused gameplay area (stage + 10% margin on each side).

    Returns: (left, right, top, bottom) in game coordinates
    """
    # Focus on main stage area instead of full blast zones
    # Most stages have platforms from ~-70 to 70 in X, and 0 to 80 in Y
    # With 10% margin on each side

    # Stage-specific gameplay bounds (main platform + common aerial space)
    GAMEPLAY_BOUNDS = {
        enums.Stage.BATTLEFIELD: (-80, 80, 90, -60),
        enums.Stage.FINAL_DESTINATION: (-90, 90, 90, -60),
        enums.Stage.DREAMLAND: (-85, 85, 95, -60),
        enums.Stage.FOUNTAIN_OF_DREAMS: (-75, 75, 85, -60),
        enums.Stage.POKEMON_STADIUM: (-85, 85, 85, -60),
        enums.Stage.YOSHIS_STORY: (-70, 70, 75, -55),
    }

    if stage not in GAMEPLAY_BOUNDS:
        # Default to reasonable bounds
        stage_bounds = (-85, 85, 90, -60)
    else:
        stage_bounds = GAMEPLAY_BOUNDS[stage]

    left, right, top, bottom = stage_bounds

    # Add 10% margin on each side
    width = right - left
    height = top - bottom

    margin_x = width * 0.10
    margin_y = height * 0.10

    return (
        left - margin_x,
        right + margin_x,
        top + margin_y,
        bottom - margin_y
    )


def create_stage_canvas(
    stage: enums.Stage, dpi: int = 100
) -> tuple[plt.Figure, plt.Axes]:
    """Create matplotlib figure with single stage canvas.

    Args:
        stage: Stage enum
        dpi: DPI for the figure

    Returns:
        Figure and single axis
    """
    # Use focused gameplay bounds instead of full blast zones
    left, right, top, bottom = get_gameplay_bounds(stage)
    stage_name = STAGE_NAMES.get(stage, "Unknown Stage")

    # Create figure with single subplot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=dpi)

    fig.suptitle(f"{stage_name} - Future Position Prediction", fontsize=18, y=0.96)

    # Configure the subplot with focused view
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title('Predicted future position (~30-40 frames ahead)', fontsize=14, pad=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Draw stage box (top at y=0)
    stage_left, stage_right, _, _ = get_stage_bounds(stage)
    stage_top = 0
    stage_bottom = -10
    stage_rect = Rectangle(
        (stage_left, stage_bottom),
        stage_right - stage_left,
        stage_top - stage_bottom,
        fill=False,
        edgecolor='white',
        linewidth=2,
        linestyle='--',
        alpha=0.7
    )
    ax.add_patch(stage_rect)

    plt.tight_layout()
    return fig, ax


class VisualizationState:
    """Holds persistent matplotlib artists for fast updates."""
    def __init__(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        stage: enums.Stage,
        x_edges: np.ndarray,
        y_edges: np.ndarray,
    ):
        self.fig = fig
        self.ax = ax
        self.stage = stage
        self.x_edges = x_edges
        self.y_edges = y_edges

        # Get blast zone bounds for stage box
        stage_left, stage_right, _, _ = get_stage_bounds(stage)

        # Create initial heatmap with zeros
        initial_data = np.zeros((32, 32))
        self.heatmap = ax.pcolormesh(
            x_edges, y_edges, initial_data,
            cmap='hot',
            shading='flat',
            alpha=0.8,
            vmin=0,
            vmax=1.0,
            zorder=1
        )

        # Create stage box (drawn above heatmap)
        stage_top = 0
        stage_bottom = -50
        self.stage_rect = Rectangle(
            (stage_left, stage_bottom),
            stage_right - stage_left,
            stage_top - stage_bottom,
            fill=False,
            edgecolor='cyan',
            linewidth=2,
            linestyle='--',
            alpha=0.8,
            zorder=5
        )
        ax.add_patch(self.stage_rect)

        # Draw vertical lines at stage edges (ledges)
        left_edge, right_edge = stage_edge_positions(stage)
        self.stage_edge_lines = []
        if left_edge is not None and right_edge is not None:
            # Draw from stage bottom to slightly above stage top
            edge_bottom = stage_bottom
            edge_top = stage_top + 5

            # Left edge line
            left_line, = ax.plot(
                [left_edge, left_edge],
                [edge_bottom, edge_top],
                'y-',  # Yellow color
                linewidth=2,
                alpha=0.7,
                zorder=7
            )
            self.stage_edge_lines.append(left_line)

            # Right edge line
            right_line, = ax.plot(
                [right_edge, right_edge],
                [edge_bottom, edge_top],
                'y-',  # Yellow color
                linewidth=2,
                alpha=0.7,
                zorder=7
            )
            self.stage_edge_lines.append(right_line)

        self.platform_rects = []
        platform_thickness = 4.0  # Visual thickness of platform

        # Top platform
        top_height, top_left, top_right = top_platform_position(stage)
        if top_height is not None:
            top_platform = Rectangle(
                (top_left, top_height - platform_thickness / 2),
                top_right - top_left,
                platform_thickness,
                fill=True,
                facecolor='gray',
                edgecolor='white',
                linewidth=1,
                alpha=0.6,
                zorder=6
            )
            ax.add_patch(top_platform)
            self.platform_rects.append(top_platform)

        # Left platform
        left_height, left_left, left_right = left_platform_position(stage)
        if left_height is not None:
            left_platform = Rectangle(
                (left_left, left_height - platform_thickness / 2),
                left_right - left_left,
                platform_thickness,
                fill=True,
                facecolor='gray',
                edgecolor='white',
                linewidth=1,
                alpha=0.6,
                zorder=6
            )
            ax.add_patch(left_platform)
            self.platform_rects.append(left_platform)

        # Right platform
        right_height, right_left, right_right = right_platform_position(stage)
        if right_height is not None:
            right_platform = Rectangle(
                (right_left, right_height - platform_thickness / 2),
                right_right - right_left,
                platform_thickness,
                fill=True,
                facecolor='gray',
                edgecolor='white',
                linewidth=1,
                alpha=0.6,
                zorder=6
            )
            ax.add_patch(right_platform)
            self.platform_rects.append(right_platform)

        # Create ground truth trajectory line (current pos -> 31 frames ahead)
        self.ground_truth_line, = ax.plot(
            [], [], '-',
            color='lime',  # Brighter green
            linewidth=4,  # Thicker for better visibility
            alpha=0.9,
            zorder=9,
            label='Ground truth (31f ahead)'
        )

        # Create ground truth endpoint marker
        self.ground_truth_endpoint, = ax.plot(
            [], [], '*',
            color='lime',  # Brighter green
            markersize=15,
            markeredgecolor='white',
            markeredgewidth=1.5,
            zorder=11
        )

        # Create player markers (initially at origin)
        self.p1_marker, = ax.plot(
            [0], [0], 'go',
            markersize=12,
            markeredgecolor='white',
            markeredgewidth=2,
            zorder=10
        )
        self.p2_marker, = ax.plot(
            [0], [0], 'ro',
            markersize=12,
            markeredgecolor='white',
            markeredgewidth=2,
            zorder=10
        )


def compute_bucket_edges(boundaries: torch.Tensor) -> np.ndarray:
    """Compute pcolormesh edges from bucket boundaries.

    28 boundaries define 32 buckets:
    - Bucket 0: extrapolated left
    - Buckets 1-28: between boundaries
    - Buckets 29-31: extrapolated right

    Args:
        boundaries: Tensor of 28 boundary values

    Returns:
        Array of 33 edge values for pcolormesh
    """
    boundaries = boundaries.cpu().numpy()
    edges = np.zeros(33)

    # Edge 0: extrapolate left of first bucket
    edges[0] = boundaries[0] - (boundaries[1] - boundaries[0])

    # Edges 1-28: use the 28 boundaries
    edges[1:29] = boundaries

    # Edges 29-32: extrapolate right of last bucket
    step = boundaries[-1] - boundaries[-2]
    for i in range(29, 33):
        edges[i] = boundaries[-1] + (i - 28) * step

    return edges


def initialize_visualization(
    fig: plt.Figure,
    ax: plt.Axes,
    stage: enums.Stage,
) -> VisualizationState:
    """Initialize persistent visualization elements.

    Args:
        fig: Matplotlib figure
        ax: Axis to draw on
        stage: Stage enum

    Returns:
        VisualizationState with persistent artists
    """
    # Use actual non-uniform bucket boundaries for X and Y
    x_edges = compute_bucket_edges(FUTURE_X_BUCKETS)
    y_edges = compute_bucket_edges(FUTURE_Y_BUCKETS)

    return VisualizationState(fig, ax, stage, x_edges, y_edges)


def visualize_frame(
    viz_state: VisualizationState,
    frame_num: int,
    p1_x: float,
    p1_y: float,
    p2_x: float,
    p2_y: float,
    future_x_probs: torch.Tensor,  # [32]
    future_y_probs: torch.Tensor,  # [32]
    p1_future_x: Optional[list[float]] = None,
    p1_future_y: Optional[list[float]] = None,
) -> np.ndarray:
    """Visualize future position probabilities for one frame.

    Args:
        viz_state: VisualizationState with persistent artists
        frame_num: Current frame number
        p1_x, p1_y: Player 1 current position
        p2_x, p2_y: Player 2 position (opponent)
        future_x_probs: Future X probabilities [32 buckets]
        future_y_probs: Future Y probabilities [32 buckets]
        p1_future_x: List of Player 1 actual X positions for next 60 frames (optional)
        p1_future_y: List of Player 1 actual Y positions for next 60 frames (optional)

    Returns:
        RGB image array (H, W, 3) for video encoding
    """
    # Get probabilities
    px = future_x_probs.cpu().numpy()  # [32]
    py = future_y_probs.cpu().numpy()  # [32]

    # Compute joint probability (assuming independence)
    joint_prob = np.outer(py, px)  # [32, 32] - outer product gives P(y, x)

    # Update heatmap data (much faster than recreating)
    viz_state.heatmap.set_array(joint_prob.ravel())
    vmax = joint_prob.max() if joint_prob.max() > 0 else 1.0
    viz_state.heatmap.set_clim(vmin=0, vmax=vmax)

    # Update player marker positions
    viz_state.p1_marker.set_data([p1_x], [p1_y])
    viz_state.p2_marker.set_data([p2_x], [p2_y])

    # Update ground truth trajectory if available
    # p1_future_x and p1_future_y are now lists of positions for all frames ahead
    if p1_future_x is not None and p1_future_y is not None and len(p1_future_x) > 0:
        # Draw path through all positions from current to future frames
        viz_state.ground_truth_line.set_data(p1_future_x, p1_future_y)
        # Mark the final position (end of trajectory)
        viz_state.ground_truth_endpoint.set_data([p1_future_x[-1]], [p1_future_y[-1]])
    else:
        # Hide the line if no ground truth available
        viz_state.ground_truth_line.set_data([], [])
        viz_state.ground_truth_endpoint.set_data([], [])

    # Update figure title with frame number
    stage_name = STAGE_NAMES.get(viz_state.stage, "Unknown Stage")
    viz_state.fig.suptitle(
        f"{stage_name} - Future Position Prediction (Frame {frame_num})",
        fontsize=18, y=0.96
    )

    # Convert figure to numpy array
    viz_state.fig.canvas.draw()
    # Use buffer_rgba() which is the modern API, then convert to RGB
    buf = np.frombuffer(viz_state.fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = viz_state.fig.canvas.get_width_height()
    # Calculate actual dimensions (may be scaled for retina display)
    expected_size = w * h * 4
    actual_size = len(buf)
    scale = int(np.sqrt(actual_size / expected_size))
    img = buf.reshape(h * scale, w * scale, 4)
    # Convert RGBA to RGB
    img = img[..., :3]

    return img


def process_replay(
    replay_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    max_frames: Optional[int] = None,
) -> None:
    """Process replay and generate visualization video.

    Args:
        replay_path: Path to .slp replay file
        checkpoint_path: Path to model checkpoint
        output_path: Path for output video
        max_frames: Optional maximum number of frames to process
    """
    print(f"Loading replay: {replay_path}")
    console = Console(is_dolphin=False, allow_old_version=True, path=str(replay_path))

    if not console.connect():
        print(f"Error: Failed to connect to replay file", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {checkpoint_path}")
    init_config()  # Initialize global config
    engine = GPTInferenceEngine(checkpoint_path)

    # Verify we're using GPU acceleration
    print(f"Device: {engine.device}")
    if engine.device.type == "mps":
        print("✓ Using Metal Performance Shaders (MPS) for GPU acceleration")
    elif engine.device.type == "cuda":
        print("✓ Using CUDA for GPU acceleration")
    else:
        print("⚠ Warning: Using CPU (this will be slow)")

    # Determine player ports (assume ports 1 and 2)
    bot_port = 1
    opp_port = 2

    # Setup for video
    context_window = 256
    future_horizon = 31  # frames ahead for ground truth visualization
    visualization_delay = future_horizon  # Delay visualization to have future trajectory available
    frame_num = 0
    processed_frames = 0
    stage = None
    fig = None
    ax = None
    viz_state = None
    video_writer = None

    # List to store all positions for ground truth visualization
    # Each entry: (p1_x, p1_y) indexed by frame number
    all_positions = []

    # Buffer to store predictions and positions for delayed visualization
    # Each entry: (frame_num, p1_x, p1_y, p2_x, p2_y, future_x_probs, future_y_probs)
    viz_buffer = deque()

    # Performance tracking
    inference_times = []
    viz_times = []
    total_start = time.time()

    print(f"Processing frames (skipping first {context_window} for warmup, then buffering {visualization_delay} for trajectory)...")

    try:
        while True:
            gamestate = console.step()

            if gamestate is None:
                break

            # Skip invalid frames
            if gamestate.menu_state not in [enums.Menu.IN_GAME, enums.Menu.SUDDEN_DEATH]:
                continue

            # Check if both players exist
            if bot_port not in gamestate.players or opp_port not in gamestate.players:
                continue

            p1 = gamestate.players[bot_port]
            p2 = gamestate.players[opp_port]

            # Get stage on first valid frame
            if stage is None:
                stage = gamestate.stage
                print(f"Stage: {STAGE_NAMES.get(stage, 'Unknown')}")

                # Create figure now that we know the stage
                fig, ax = create_stage_canvas(stage)

                # Initialize visualization state with persistent artists
                viz_state = initialize_visualization(fig, ax, stage)

                # Setup video writer
                fps = 60
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                # Get frame size from first render
                # Create dummy tensors on the correct device
                dummy_probs = torch.zeros(32, device=engine.device)
                dummy_img = visualize_frame(
                    viz_state, 0,
                    0.0, 0.0, 0.0, 0.0,
                    dummy_probs, dummy_probs
                )
                height, width = dummy_img.shape[:2]

                video_writer = cv2.VideoWriter(
                    str(output_path), fourcc, fps, (width, height)
                )

                print(f"Output video: {output_path} ({width}x{height} @ {fps} FPS)")

            # Extract positions
            p1_x = p1.position.x
            p1_y = p1.position.y
            p2_x = p2.position.x
            p2_y = p2.position.y

            # Store positions for ground truth trajectory
            all_positions.append((p1_x, p1_y))

            # Collect features and run inference
            raw_inputs = collect_raw_inputs_from_gamestate(gamestate, bot_port, opp_port)

            # Prepare inputs for the model
            inputs_td, _, _, _ = engine.prepare_only(raw_inputs)

            frame_num += 1

            # Skip frames until we have enough context
            if frame_num <= context_window:
                continue

            # Run inference if we have enough context
            if inputs_td is not None and len(engine.buffer) >= engine.warmup_frames:
                # Time the inference
                inf_start = time.time()

                # Get the stacked frames from the buffer (already on CPU)
                stacked_frames = torch.stack(list(engine.buffer), dim=0).unsqueeze(0).to(engine.device)  # [B=1, L, F]

                # Add horizon feature (fixed at 30 frames, normalized by 60.0)
                B, L, F = stacked_frames.shape
                horizon = 30
                horizon_norm = torch.full((B, L, 1), horizon / 60.0, device=engine.device)
                stacked_frames = torch.cat([stacked_frames, horizon_norm], dim=-1)  # [B=1, L, F+1]

                # Build model inputs with horizon feature
                model_inputs = build_model_inputs(stacked_frames, engine.colmap)

                # Run inference once - stays on MPS
                with torch.inference_mode():
                    outputs = engine.model(model_inputs)

                    # Extract future position logits for last timestep (still on MPS)
                    future_x_logits = outputs["future_x"][0, -1]  # [32]
                    future_y_logits = outputs["future_y"][0, -1]  # [32]

                    # Convert to probabilities (on MPS)
                    future_x_probs = F_torch.softmax(future_x_logits, dim=0)  # [32]
                    future_y_probs = F_torch.softmax(future_y_logits, dim=0)  # [32]

                inference_times.append(time.time() - inf_start)

                # Buffer this frame's data for delayed visualization
                viz_buffer.append({
                    'frame_num': frame_num,
                    'p1_x': p1_x,
                    'p1_y': p1_y,
                    'p2_x': p2_x,
                    'p2_y': p2_y,
                    'future_x_probs': future_x_probs.cpu(),  # Move to CPU for storage
                    'future_y_probs': future_y_probs.cpu(),
                })

                # Once we have enough buffer, visualize delayed frames with trajectories
                if len(viz_buffer) > visualization_delay:
                    # Time the visualization
                    viz_start = time.time()

                    # Get the oldest frame from buffer (60 frames ago)
                    delayed_frame = viz_buffer.popleft()

                    # Extract ground truth trajectory from delayed frame to current frame
                    delayed_idx = delayed_frame['frame_num'] - 1
                    current_idx = frame_num - 1

                    # Trajectory from delayed frame to current frame
                    trajectory_positions = all_positions[delayed_idx:current_idx + 1]
                    p1_future_x_list = [pos[0] for pos in trajectory_positions]
                    p1_future_y_list = [pos[1] for pos in trajectory_positions]

                    # Generate visualization frame (transfers to CPU inside visualize_frame)
                    img = visualize_frame(
                        viz_state,
                        delayed_frame['frame_num'],
                        delayed_frame['p1_x'],
                        delayed_frame['p1_y'],
                        delayed_frame['p2_x'],
                        delayed_frame['p2_y'],
                        delayed_frame['future_x_probs'],
                        delayed_frame['future_y_probs'],
                        p1_future_x_list,
                        p1_future_y_list
                    )

                    # Write frame to video
                    # Convert RGB to BGR for OpenCV
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    video_writer.write(img_bgr)

                    viz_times.append(time.time() - viz_start)

            processed_frames += 1

            if max_frames and processed_frames >= max_frames:
                break

            if processed_frames % 60 == 0:
                print(f"Processed {processed_frames} frames ({processed_frames/60:.1f} seconds of gameplay)")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if video_writer:
            video_writer.release()
        if fig:
            plt.close(fig)

    total_time = time.time() - total_start

    print(f"\nDone! Processed {processed_frames} frames in {total_time:.2f}s")
    if output_path.exists():
        print(f"Output saved to: {output_path}")

    # Performance summary
    if inference_times:
        avg_inference = np.mean(inference_times) * 1000  # Convert to ms
        avg_viz = np.mean(viz_times) * 1000
        fps = processed_frames / total_time

        print(f"\n=== Performance Summary ===")
        print(f"Device: {engine.device}")
        print(f"Average inference time: {avg_inference:.2f}ms/frame")
        print(f"Average visualization time: {avg_viz:.2f}ms/frame")
        print(f"Total time per frame: {(avg_inference + avg_viz):.2f}ms")
        print(f"Processing speed: {fps:.1f} FPS")
        print(f"Realtime ratio: {fps/60:.2f}x (60 FPS = 1.0x)")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize future position predictions from a Melee replay"
    )
    parser.add_argument(
        "replay",
        type=Path,
        help="Path to .slp replay file"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("future_predictions.mp4"),
        help="Output video path (default: future_predictions.mp4)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all frames)"
    )

    args = parser.parse_args()

    if not args.replay.exists():
        print(f"Error: Replay file not found: {args.replay}", file=sys.stderr)
        sys.exit(1)

    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    process_replay(args.replay, args.checkpoint, args.output, args.max_frames)


if __name__ == "__main__":
    main()
