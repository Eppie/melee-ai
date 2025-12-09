"""Visualization tools for future position predictions."""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import sys

# Add the parent directory to sys.path to import the local future_position package
sys.path.insert(0, str(Path(__file__).parents[2]))

from future_position.inference.engine import FuturePredictor
from future_position.data.parse_slp import parse_slp_to_features
from future_position.config import VisualizationConfig
from future_position.constants import STAGE_HALF_WIDTH, STAGE_HALF_HEIGHT


def visualize_replay(
    replay_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    config: Optional[VisualizationConfig] = None,
) -> None:
    """Visualize predictions on a replay.

    Args:
        replay_path: Path to .slp replay file
        checkpoint_path: Path to model checkpoint
        output_path: Path to save output video
        config: Visualization configuration

    Notes:
        - Parses replay
        - Runs inference per frame
        - Renders predictions as heatmaps overlaid on stage
        - Outputs video
    """
    if config is None:
        config = VisualizationConfig()

    # 1. Parse replay to features
    print(f"Parsing replay: {replay_path}")
    data = parse_slp_to_features(replay_path)
    features = data['features']
    # Extract positions assuming same indices as build_dataset (temp fix)
    # P1: [0, 1], P2: [37, 38] approx.
    # Better: re-use extract_player_features knowledge or constants
    # P1_X, P1_Y is 0, 1
    # P2_X, P2_Y is at feature_dim // 2, feature_dim // 2 + 1
    feature_dim = features.shape[1]
    p2_offset = 37 + 3 # 37 features per player + 3 relational features in the middle? No.
    # Check parse_slp.py:
    # [p1_features, p2_features, relational_features, global_ids]
    # p1_features len = 37
    # p2_features len = 37
    # relational len = 3
    # global len = 3
    # So P1 pos: 0, 1
    # P2 pos: 37, 38
    
    p1_idx = [0, 1]
    p2_idx = [37, 38]
    
    stage_id = int(data['stage'])

    # 2. Initialize predictor
    print("Initializing predictor...")
    predictor = FuturePredictor(checkpoint_path=checkpoint_path)

    # 3. Initialize video writer
    video_writer = VideoWriter(
        output_path=output_path,
        fps=config.fps,
        resolution=config.resolution,
    )

    # 4. For each frame
    print("Rendering video...")
    for i in tqdm(range(len(features)), desc="Frames"):
        frame = features[i]
        
        # Run prediction
        predictions = predictor.predict(frame)
        
        # Extract current positions for rendering
        p1_pos = (frame[p1_idx[0]], frame[p1_idx[1]])
        p2_pos = (frame[p2_idx[0]], frame[p2_idx[1]])
        positions = np.array([p1_pos[0], p1_pos[1], p2_pos[0], p2_pos[1]])

        if predictions is not None:
            # Render frame with predictions
            img = render_frame(
                frame_idx=i,
                positions=positions,
                predictions=predictions,
                stage_id=stage_id,
                config=config,
            )
            video_writer.write_frame(img)
        else:
            # Buffer filling up, maybe render just the game state?
            # For now, skip or render empty
            pass

    # 5. Close video writer
    video_writer.close()
    print(f"Video saved to {output_path}")


def render_frame(
    frame_idx: int,
    positions: np.ndarray,
    predictions: Dict,
    stage_id: int,
    config: VisualizationConfig,
) -> np.ndarray:
    """Render a single frame with predictions.

    Args:
        frame_idx: Current frame index
        positions: [4] current positions (p1_x, p1_y, p2_x, p2_y)
        predictions: Prediction dict from FuturePredictor
        stage_id: Stage ID for rendering
        config: Visualization config

    Returns:
        [H, W, 3] RGB image (uint8)
    """
    fig = plt.figure(figsize=(config.resolution[0]/config.dpi, config.resolution[1]/config.dpi), dpi=config.dpi)
    ax = fig.add_subplot(111)
    
    # Set bounds roughly around stage
    ax.set_xlim(-STAGE_HALF_WIDTH * 1.5, STAGE_HALF_WIDTH * 1.5)
    ax.set_ylim(-STAGE_HALF_HEIGHT * 1.0, STAGE_HALF_HEIGHT * 2.0)
    ax.set_aspect('equal')

    draw_stage(ax, stage_id)
    
    p1_pos = positions[:2]
    p2_pos = positions[2:]
    draw_player_positions(ax, p1_pos, p2_pos)

    # Draw predictions
    if config.show_prediction:
        # Iterate through horizons we want to display
        # predictions['p1'] is a list of dicts, one for each trained horizon
        # We need to map config.display_horizons to indices if possible, or just display what's available
        # The predictor returns all trained horizons in order.
        # Assuming predictor.horizons matches trained order.
        
        from ..constants import HORIZONS
        
        for h_idx, h_val in enumerate(HORIZONS):
            if h_val in config.display_horizons:
                # Render P1
                p1_mix = predictions['p1'][h_idx]
                # Convert to heatmap and contour/imshow
                # Note: Mixture params are normalized deltas.
                # Need to denormalize and add to current position to get absolute target distribution.
                
                # This is complex to do efficiently with full grid evaluation for every frame.
                # Approximating by plotting samples or simplified Gaussian ellipses might be faster.
                # For now, let's plot the mean of the most probable component or just samples.
                
                # Simplest for prototype: plot means of components weighted by alpha
                # Or just plot the main Gaussian components as ellipses.
                pass

    # Cleanup
    ax.axis('off')
    fig.tight_layout(pad=0)
    
    img = figure_to_array(fig)
    plt.close(fig)
    return img


def mixture_to_heatmap(
    mixture_params: Dict,
    x_range: tuple,
    y_range: tuple,
    resolution: int = 64,
) -> np.ndarray:
    """Convert mixture of Gaussians to heatmap.
    """
    # TODO: Implement grid evaluation
    return np.zeros((resolution, resolution))


def draw_stage(ax: plt.Axes, stage_id: int) -> None:
    """Draw stage background."""
    # Placeholder: just a floor line
    ax.add_patch(plt.Rectangle((-50, -5), 100, 5, color='gray'))


def draw_player_positions(
    ax: plt.Axes,
    p1_pos: tuple,
    p2_pos: tuple,
) -> None:
    """Draw current player positions."""
    ax.plot(p1_pos[0], p1_pos[1], 'ro', markersize=10, label='P1')
    ax.plot(p2_pos[0], p2_pos[1], 'bo', markersize=10, label='P2')


def figure_to_array(fig: plt.Figure) -> np.ndarray:
    """Convert matplotlib figure to numpy array."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    # Convert to numpy array
    X = np.asarray(buf)
    # Convert RGBA to BGR for OpenCV
    return cv2.cvtColor(X, cv2.COLOR_RGBA2BGR)


class VideoWriter:
    """Wrapper for cv2.VideoWriter."""

    def __init__(
        self,
        output_path: Path,
        fps: int = 60,
        resolution: tuple = (1200, 1000),
    ):
        self.output_path = str(output_path)
        self.fps = fps
        self.resolution = resolution
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, self.resolution)

    def write_frame(self, frame: np.ndarray) -> None:
        if frame.shape[0] != self.resolution[1] or frame.shape[1] != self.resolution[0]:
            frame = cv2.resize(frame, self.resolution)
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()


if __name__ == '__main__':
    """Command-line interface for visualization."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize future position predictions.")
    parser.add_argument("--replay", type=Path, required=True, help="Path to .slp replay file")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Path to save output video")
    
    args = parser.parse_args()

    visualize_replay(
        replay_path=args.replay,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )
