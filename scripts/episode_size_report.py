import time
from pathlib import Path

import numpy as np
import psutil

# Suppress CUDA init for this inspection script
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from window_dataset import WindowDataset, worker_init_fn


def main() -> None:
    # Trigger worker_init_fn memory report by instantiating a DataLoader with 1 worker.
    root = Path("processed_data_5700")
    start = time.time()
    print(f"[step] creating dataset at {start:.2f}")
    rss0 = psutil.Process().memory_info().rss / 1024**2
    print(f"[rss] before dataset init: {rss0:.2f} MiB")
    ds = WindowDataset(
        str(root), in_memory=False
    )
    print(f"[step] dataset created at {time.time():.2f}")
    rss1 = psutil.Process().memory_info().rss / 1024**2
    print(f"[rss] after dataset init: {rss1:.2f} MiB (delta {rss1-rss0:.2f} MiB)")
    max_eps = 16
    sizes = []
    shapes = None
    # Explicitly materialize with a timeout guard
    for idx, ep in enumerate(ds.index.episodes[:max_eps]):
        t0 = time.time()
        print(f"[step] ep {idx} open arrays start {t0:.2f}", flush=True)
        feats, targs = ds.index.open_episode_arrays(ep)
        t1 = time.time()
        print(
            f"[step] ep {idx} open arrays done {t1:.2f} (dt={t1-t0:.2f}s)", flush=True
        )
        # Force materialization to measure true in-memory footprint
        f_np = np.asarray(feats[:], dtype=np.float32, order="C")
        t_np = np.asarray(targs[:], dtype=np.float32, order="C")
        t2 = time.time()
        print(
            f"[step] ep {idx} materialize done {t2:.2f} (dt={t2-t1:.2f}s)", flush=True
        )
        size_bytes = f_np.nbytes + t_np.nbytes
        sizes.append(size_bytes)
        shapes = (f_np.shape, t_np.shape)
        rss = psutil.Process().memory_info().rss / 1024**2
        print(
            f"[step] ep {idx} logical size {size_bytes/1024**2:.2f} MiB rss {rss:.2f} MiB",
            flush=True,
        )
        if time.time() - start > 60:
            print("[step] time budget reached, stopping early", flush=True)
            break

    sizes = np.array(sizes, dtype=np.int64)
    rss_final = psutil.Process().memory_info().rss / 1024**2
    print(
        f"[rss] after sampling: {rss_final:.2f} MiB (delta {rss_final-rss1:.2f} MiB from init)"
    )
    print(f"measured episodes: {len(sizes)}")
    print(f"sample shapes (features, targets): {shapes}")
    print(f"avg size: {sizes.mean()/1024**2:.2f} MiB")
    print(f"median size: {np.median(sizes)/1024**2:.2f} MiB")
    print(f"p95 size (of sample): {np.percentile(sizes,95)/1024**2:.2f} MiB")
    print(f"max size: {sizes.max()/1024**2:.2f} MiB")
    print(f"RSS after sampling: {psutil.Process().memory_info().rss/1024**2:.2f} MiB")
    print(f"elapsed: {time.time()-start:.1f}s")

    # Spin up a tiny DataLoader to invoke worker_init_fn (worker 0 will log mem)
    try:
        from torch.utils.data import DataLoader, RandomSampler

        sampler = RandomSampler(ds, replacement=False)
        loader = DataLoader(
            ds,
            batch_size=1,
            sampler=sampler,
            num_workers=1,
            worker_init_fn=worker_init_fn,
            pin_memory=False,
        )
        print("[step] iterating one batch to trigger worker init")
        next(iter(loader))
    except Exception as exc:
        print(f"[warn] failed to run worker mem test: {exc}")


if __name__ == "__main__":
    main()
