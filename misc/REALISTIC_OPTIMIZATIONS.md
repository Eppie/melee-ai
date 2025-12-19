# Realistic Optimization Roadmap (Revised)

After profiling and API analysis, here are the **actually achievable** optimizations:

---

## ❌ What WON'T Work

### Cache Game Object
**Why not**: peppi_py doesn't support incremental parsing. With `skip_frames=True`, you get metadata only. With `skip_frames=False`, you must re-parse the entire file. No way to "resume" parsing.

**Tested**: Two-pass (0.174s) is already faster than single-pass (0.214s) because early rejection of bad files saves frame parsing time.

---

## ✅ What WILL Work

### 🥇 **Priority 1: Optimize `_active_stick_ratio` (15-25% speedup)**

**Current bottleneck**: Called 74 times (2x per file), uses multiple numpy operations

**Effort**: 2-3 hours | **Impact**: 15-25% speedup | **ROI**: ⭐⭐⭐⭐⭐

**Three approaches** (pick one):

#### Option A: Numba JIT (FASTEST - 20-25% speedup)
```python
import numba
import numpy as np

@numba.jit(nopython=True, cache=True, fastmath=True)
def _compute_active_count(x: np.ndarray, y: np.ndarray, deadzone: float) -> tuple[int, int]:
    """JIT-compiled hot loop for stick ratio calculation."""
    active_count = 0
    total_count = 0

    for i in range(len(x)):
        if not (np.isnan(x[i]) or np.isnan(y[i]) or
                np.isinf(x[i]) or np.isinf(y[i])):
            total_count += 1
            if abs(x[i]) > deadzone or abs(y[i]) > deadzone:
                active_count += 1

    return active_count, total_count

def _active_stick_ratio(joystick) -> float:
    x_arr = getattr(joystick, "x", None)
    y_arr = getattr(joystick, "y", None)
    if x_arr is None or y_arr is None:
        return 0.0

    x = x_arr.to_numpy(zero_copy_only=False)
    y = y_arr.to_numpy(zero_copy_only=False)
    if x.size == 0 or y.size == 0:
        return 0.0

    active, total = _compute_active_count(x, y, MAIN_STICK_DEADZONE)
    return active / total if total > 0 else 0.0
```

**Pros**: 5-10x faster, minimal code change
**Cons**: Adds numba dependency (~50MB install)

---

#### Option B: Pure NumPy Optimization (GOOD - 10-15% speedup)
```python
def _active_stick_ratio(joystick) -> float:
    """Optimized version with reduced allocations."""
    x_arr = getattr(joystick, "x", None)
    y_arr = getattr(joystick, "y", None)
    if x_arr is None or y_arr is None:
        return 0.0

    try:
        x = x_arr.to_numpy(zero_copy_only=False)
        y = y_arr.to_numpy(zero_copy_only=False)
    except Exception:
        return 0.0

    if x.size == 0 or y.size == 0:
        return 0.0

    # Single pass: compute finite mask and active mask together
    finite = np.isfinite(x) & np.isfinite(y)
    total = np.count_nonzero(finite)

    if total == 0:
        return 0.0

    # Only compute abs() on finite values (lazy evaluation helps)
    active = np.count_nonzero(
        (np.abs(x) > MAIN_STICK_DEADZONE) | (np.abs(y) > MAIN_STICK_DEADZONE) & finite
    )

    return active / total
```

**Pros**: No new dependencies, cleaner code
**Cons**: Only ~2x faster than current

---

#### Option C: Early Exit Optimization (EASIEST - 5-10% speedup)
```python
def _active_stick_ratio(joystick) -> float:
    """Early exit optimization for bad data."""
    # Early validation with cheap checks first
    if joystick is None:
        return 0.0

    x_arr = getattr(joystick, "x", None)
    y_arr = getattr(joystick, "y", None)

    # Fail fast on None
    if x_arr is None or y_arr is None:
        return 0.0

    # Check length before expensive to_numpy() conversion
    if len(x_arr) == 0 or len(y_arr) == 0:
        return 0.0

    x = x_arr.to_numpy(zero_copy_only=False)
    y = y_arr.to_numpy(zero_copy_only=False)

    # Rest of function unchanged...
```

**Pros**: Zero risk, no dependencies
**Cons**: Minimal impact

---

### 🥈 **Priority 2: Optimize Worker Chunking (10-20% speedup on large batches)**

**Current**: `chunksize=20` for directory processing, `max(1, extract_workers)` for zip

**Effort**: 30 minutes | **Impact**: 10-20% on batches >1000 files | **ROI**: ⭐⭐⭐⭐

```python
def _process_zip_archive(...):
    # Dynamic chunking based on file count
    members = list(_iter_zip_members(zip_path, filename_filter))
    optimal_chunksize = max(1, min(100, len(members) // (process_workers * 4)))

    with ProcessPoolExecutor(max_workers=process_workers) as processor:
        for _ in processor.map(
            _extract_and_process_member,
            repeat(zip_path),
            members,
            repeat(allowed_chars),
            chunksize=optimal_chunksize,  # Was: max(1, extract_workers)
        ):
            pass
```

**Testing needed**:
```bash
# Benchmark different chunksizes
for chunk in 10 20 50 100 200; do
    time python filter_bad_replays.py --zip-file test_replays.zip --workers 8 --extract-workers $chunk
done
```

---

### 🥉 **Priority 3: Worker Pre-Initialization (10-20% on cold starts)**

**Current**: Each worker imports peppi_py/pandas on first file (0.081s overhead)

**Effort**: 30 minutes | **Impact**: 10-20% on batches <100 files | **ROI**: ⭐⭐⭐

```python
def _worker_init():
    """Pre-load heavy imports in worker process."""
    import peppi_py
    import pandas as pd
    import pyarrow as pa
    import numpy as np
    # Force module initialization
    _ = peppi_py.__version__

def _process_zip_archive(...):
    with ProcessPoolExecutor(
        max_workers=process_workers,
        initializer=_worker_init  # NEW
    ) as processor:
        # ...
```

Also update directory processing:
```python
def main():
    # ...
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init  # NEW
    ) as pool:
        # ...
```

---

### 🏅 **Priority 4: Reduce to_numpy() Calls (5-10% speedup)**

**Current**: `to_numpy(zero_copy_only=False)` always copies data

**Effort**: 1 hour | **Impact**: 5-10% | **ROI**: ⭐⭐⭐

Try zero-copy first, fallback to copy:
```python
def _active_stick_ratio(joystick) -> float:
    x_arr = getattr(joystick, "x", None)
    y_arr = getattr(joystick, "y", None)
    if x_arr is None or y_arr is None:
        return 0.0

    # Try zero-copy first (faster if possible)
    try:
        x = x_arr.to_numpy(zero_copy_only=True)
        y = y_arr.to_numpy(zero_copy_only=True)
    except (ValueError, TypeError):
        # Fallback to copy if zero-copy fails
        x = x_arr.to_numpy(zero_copy_only=False)
        y = y_arr.to_numpy(zero_copy_only=False)

    # Rest unchanged...
```

---

## 📊 Expected Combined Impact

Implementing all 4 priorities:

| Optimization | Individual | Cumulative |
|--------------|------------|------------|
| **Baseline** | - | 100% (0.174s / 37 files) |
| + Active stick ratio (numba) | -20% | **80%** (0.139s) |
| + Optimal chunking | -12% | **71%** (0.123s) |
| + Worker init | -10% | **64%** (0.111s) |
| + Zero-copy numpy | -5% | **61%** (0.106s) |

**Total speedup**: **1.64x faster** (39% time reduction)
**Effort**: 4-5 hours
**New throughput**: ~350 files/second (up from 212/sec)

---

## 🎯 My Recommendation

**Start with this order:**

1. **Numba-optimize `_active_stick_ratio`** (2h, 20% gain) ← **DO THIS FIRST**
2. **Optimize chunking** (30min, 12% gain) ← **Quick win**
3. **Add worker initialization** (30min, 10% gain) ← **Quick win**
4. **Try zero-copy numpy** (1h, 5% gain) ← **If you have time**

**Total**: 4 hours for ~40% speedup

---

## 🚫 What to Skip (Low ROI)

- ❌ **Custom .slp parser**: 20+ hours for 40% gain on sanity failures only
- ❌ **Stream from zip**: Requires peppi_py API changes (impossible without fork)
- ❌ **Cython compilation**: 10+ hours for 10-15% gain (numba is easier)
- ❌ **Parallel decompression**: Not supported by zipfile format
- ❌ **Memory-mapped I/O**: Minimal impact, high complexity

---

## 📈 When to Stop Optimizing

**Stop when**:
- Processing 10,000 files takes <30 seconds (currently ~47 seconds, projected ~29 seconds after optimizations)
- The bottleneck shifts to network I/O (downloading replays)
- You're spending more time optimizing than you'd save in processing time

**Rule of thumb**: If you process <1 million files/month, current performance is probably fine.
