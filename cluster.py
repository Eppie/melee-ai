from __future__ import annotations

from pathlib import Path

import duckdb
from matplotlib.colors import LogNorm
from sklearn.cluster import KMeans
# from sklearn_extra.cluster import KMedoids

import random
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np

SHARDS_DIR = Path("/Users/eppie/melee-ai/shards")
DEFAULT_GLOB = str(SHARDS_DIR / "frames_chunk_*.parquet")


def plot_representatives(
        coords: np.ndarray,
        weights: np.ndarray,
        representatives: np.ndarray,
        *,
        title: str = "Representative points",
        bins: int | tuple[int, int] = 200,
        log_norm: bool = True,
        vmax_percentile: float = 99.9,  # ← clip extreme bright bins
        vmin_percentile: float = 1.0,  # ← lift very dark bins (ignore zeros)
        cmap: str = "viridis",
        rep_marker: str = "X",
        rep_size: int = 120,
) -> None:
    H, xedges, yedges = np.histogram2d(
        coords[:, 0], coords[:, 1], bins=bins, weights=weights
    )
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    Hpos = H[H > 0]
    vmin = vmax = None
    norm = None
    if Hpos.size:
        if log_norm:
            vmin = max(1.0, float(np.percentile(Hpos, vmin_percentile)))
            vmax = float(np.percentile(Hpos, vmax_percentile))
            if vmax <= vmin:  # guard for tiny ranges
                vmax = float(Hpos.max())
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            vmin, vmax = np.percentile(Hpos, [vmin_percentile, vmax_percentile])

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        H.T,
        extent=extent,
        origin="lower",
        cmap=cmap,
        norm=norm,
        vmin=None if norm else vmin,
        vmax=None if norm else vmax,
        interpolation="nearest",
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85, label="Weighted count")

    ax.scatter(
        representatives[:, 0],
        representatives[:, 1],
        s=rep_size,
        marker=rep_marker,
        edgecolor="black",
        facecolor="none",
        linewidths=1.5,
        zorder=10,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def load_xy_with_weights(
        parquet_glob: str = DEFAULT_GLOB,
        x_col: str = "p2_pre_joystick_x",
        y_col: str = "p2_pre_joystick_y",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read shards → Arrow Table → numpy, keeping one copy in memory.

    Returns
    -------
    coords  : (N, 2) float32  – unique (x, y) pairs
    weights : (N,)   int64    – occurrence counts
    """
    con = duckdb.connect()
    tbl = con.execute(f"""
        SELECT {x_col} AS x, {y_col} AS y, COUNT(*) AS w
        FROM parquet_scan('{parquet_glob}')
        GROUP BY x, y
    """).arrow()

    # Arrow → numpy (zero-copy where possible)
    x_arr = tbl["x"].combine_chunks().to_numpy(zero_copy_only=False)
    y_arr = tbl["y"].combine_chunks().to_numpy(zero_copy_only=False)
    coords = np.column_stack((x_arr, y_arr)).astype(np.float32, copy=False)

    weights = tbl["w"].combine_chunks().to_numpy(zero_copy_only=False) \
        .astype(np.int64, copy=False)

    return coords, weights


def kmeans_representatives(coords: np.ndarray,
                           weights: np.ndarray,
                           k: int,
                           random_seed: int = 0) -> np.ndarray:
    """
    Returns k synthetic centroids, shape (k, 2).
    """
    km = KMeans(n_clusters=k,
                init="k-means++",
                n_init="auto",
                random_state=random_seed)
    km.fit(coords, sample_weight=weights)
    return km.cluster_centers_


if __name__ == "__main__":
    xy, w = load_xy_with_weights()

    k = 21
    centroids = kmeans_representatives(xy, w, k)
    plot_representatives(xy, w, centroids,
                         title=f"{k} k-means centroids")
    centroids.sort()
    print(f"centroids: {centroids}")