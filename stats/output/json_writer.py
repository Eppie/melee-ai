"""JSON output writer for statistics."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def write_json_stats(
    stats: Dict[str, Any],
    output_path: Path,
    config: Any = None,
    pretty: bool = True,
) -> None:
    """Write statistics to JSON file.

    Args:
        stats: Dictionary of statistics from all collectors
        output_path: Path to write JSON file
        config: Optional config to include in output
        pretty: Whether to pretty-print JSON
    """
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "version": "2.0",
        },
        "statistics": stats,
    }

    if config is not None:
        output["metadata"]["config"] = {
            "zarr_dir": str(config.zarr_dir),
            "max_episodes": config.max_episodes,
            "collectors": list(config.collectors) if config.collectors else "all",
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        if pretty:
            json.dump(output, f, indent=2, cls=NumpyEncoder)
        else:
            json.dump(output, f, cls=NumpyEncoder)


def write_summary_json(
    stats: Dict[str, Any],
    output_path: Path,
) -> None:
    """Write a compact summary JSON without histograms and large arrays."""
    summary = extract_summary(stats)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)


def extract_summary(stats: Dict[str, Any], max_items: int = 10) -> Dict[str, Any]:
    """Extract key summary statistics without large arrays."""
    summary = {}

    for key, value in stats.items():
        if isinstance(value, dict):
            # Recursively process nested dicts
            nested = extract_summary(value, max_items)
            if nested:
                summary[key] = nested
        elif isinstance(value, list):
            # Truncate long lists
            if len(value) > max_items:
                if all(isinstance(v, (int, float, str)) for v in value[:max_items]):
                    summary[key] = value[:max_items]
                    summary[f"{key}_truncated"] = True
            else:
                # Skip histogram-like 2D arrays
                if value and isinstance(value[0], list):
                    continue
                summary[key] = value
        elif isinstance(value, (int, float, str, bool)) or value is None:
            summary[key] = value

    return summary
