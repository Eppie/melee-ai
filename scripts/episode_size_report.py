import time
from pathlib import Path

import numpy as np
import psutil

from window_dataset import WindowDataset


def main() -> None:
    root = Path("processed_data_5700")
    start = time.time()
    print(f"[step] creating dataset at {start:.2f}")
    ds = WindowDataset(
        str(root), in_memory=False, in_memory_shared=False, episode_cache_size=0
    )
    print(f"[step] dataset created at {time.time():.2f}")
    max_eps = 16
    sizes = []
    shapes = None
    for idx, ep in enumerate(ds.index.episodes[:max_eps]):
        t0 = time.time()
        print(f"[step] ep {idx} open arrays start {t0:.2f}", flush=True)
        feats, targs = ds.index.open_episode_arrays(ep)
        t1 = time.time()
        print(f"[step] ep {idx} open arrays done {t1:.2f} (dt={t1-t0:.2f}s)", flush=True)
        # Use metadata to compute logical in-memory size without full materialization
        size_bytes = feats.nbytes + targs.nbytes
        sizes.append(size_bytes)
        shapes = (feats.shape, targs.shape)
        rss = psutil.Process().memory_info().rss / 1024**2
        print(
            f"[step] ep {idx} logical size {size_bytes/1024**2:.2f} MiB rss {rss:.2f} MiB",
            flush=True,
        )
        if time.time() - start > 60:
            print("[step] time budget reached, stopping early", flush=True)
            break

    sizes = np.array(sizes, dtype=np.int64)
    print(f"measured episodes: {len(sizes)}")
    print(f"sample shapes (features, targets): {shapes}")
    print(f"avg size: {sizes.mean()/1024**2:.2f} MiB")
    print(f"median size: {np.median(sizes)/1024**2:.2f} MiB")
    print(f"p95 size (of sample): {np.percentile(sizes,95)/1024**2:.2f} MiB")
    print(f"max size: {sizes.max()/1024**2:.2f} MiB")
    print(f"RSS after sampling: {psutil.Process().memory_info().rss/1024**2:.2f} MiB")
    print(f"elapsed: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
