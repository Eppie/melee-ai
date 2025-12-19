# Filter Bad Replays - Optimization Roadmap

## Current Performance Baseline
- **37 files in 0.174s** (two-pass optimized)
- **4.7 ms/file average**
- **Bottlenecks**:
  - peppi_py parsing: 30.2% (0.176s / 74 parses)
  - zlib decompression: 15.3% (0.089s)
  - Active stick ratio: ~10% (numpy operations)
  - Import overhead: 14% (one-time per worker)

---

## Optimization Opportunities

### 🔥 HIGH IMPACT, LOW EFFORT

#### 1. **Cache Parsed Game Object** (Est. 25-30% speedup)
**Current Issue**: We parse each file **twice**:
- First pass: `skip_frames=True` for sanity checks
- Second pass: `skip_frames=False` for quality checks

**Solution**: Cache the first parse and reuse it
```python
# First pass
game_metadata = peppi_py.read_slippi(str(path), skip_frames=True)
if sanity_reason(game_metadata):
    return None

# Instead of re-parsing, just parse frames separately
game_full = peppi_py.read_slippi(str(path), skip_frames=False)
# Or keep metadata and just load frames
```

**Impact**:
- Eliminates ~37 peppi_py calls (saves ~0.088s / 37 files = 2.4ms/file)
- **25-30% speedup** on files that pass sanity checks
- **50% speedup** on files that fail sanity checks (already optimized)

**Effort**: 1-2 hours
- Modify `process_file()` to cache game object
- Test that behavior is identical

**Priority**: ⭐⭐⭐⭐⭐ **DO THIS FIRST**

---

#### 2. **Optimize Chunking Strategy** (Est. 10-15% speedup for large batches)
**Current Issue**: `chunksize = max(1, extract_workers)` is too conservative
```python
for _ in pool.map(func, files, repeat(allowed_chars), chunksize=20):
```

**Solution**: Dynamic chunking based on file count
```python
chunksize = max(1, min(100, len(files) // (workers * 4)))
```

**Impact**:
- Reduces process pool overhead for large batches
- Amortizes worker startup costs
- **10-15% speedup** for batches > 1000 files
- **Minimal impact** for small batches (<100 files)

**Effort**: 30 minutes
- Test different chunksizes: 20, 50, 100, 200
- Pick optimal based on throughput

**Priority**: ⭐⭐⭐⭐ **Quick win for production workloads**

---

### 🔶 MEDIUM IMPACT, LOW-MEDIUM EFFORT

#### 3. **Vectorize Active Stick Ratio** (Est. 15-20% speedup)
**Current Issue**: `_active_stick_ratio()` processes arrays element-wise
```python
def _active_stick_ratio(joystick) -> float:
    x = x_arr.to_numpy(zero_copy_only=False)  # Copy overhead
    y = y_arr.to_numpy(zero_copy_only=False)  # Copy overhead
    finite_mask = np.isfinite(x) & np.isfinite(y)
    active = np.logical_or(
        np.abs(x) > MAIN_STICK_DEADZONE,
        np.abs(y) > MAIN_STICK_DEADZONE
    )
    # ...
```

**Solution**: Optimize with numba or pre-compiled operations
```python
import numba

@numba.jit(nopython=True, cache=True)
def _compute_active_ratio(x: np.ndarray, y: np.ndarray, deadzone: float) -> float:
    active_count = 0
    valid_count = 0
    for i in range(len(x)):
        if np.isfinite(x[i]) and np.isfinite(y[i]):
            valid_count += 1
            if abs(x[i]) > deadzone or abs(y[i]) > deadzone:
                active_count += 1
    return active_count / valid_count if valid_count > 0 else 0.0
```

**Impact**:
- Called 74 times (2x per file)
- Current: ~10% of total time
- **15-20% speedup** with numba
- **5-10% speedup** with just better vectorization

**Effort**: 2-3 hours
- Add numba dependency
- Write JIT-compiled version
- Benchmark against current
- Fallback if numba not available

**Priority**: ⭐⭐⭐ **Good ROI if processing millions of files**

---

#### 4. **Batch Worker Initialization** (Est. 20% speedup on cold starts)
**Current Issue**: Each worker imports pandas/pyarrow on first file (0.081s overhead)

**Solution**: Use `initializer` in ProcessPoolExecutor
```python
def _worker_init():
    """Pre-import heavy dependencies in worker process."""
    import peppi_py
    import pandas
    import pyarrow
    import numpy

with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
    # ...
```

**Impact**:
- Amortizes import cost across all workers
- First file in each worker is 20% faster
- **Negligible impact** after first file
- **20% speedup** on total time for small batches (<50 files)

**Effort**: 30 minutes
- Add initializer function
- Test that imports work correctly

**Priority**: ⭐⭐⭐ **Nice QOL improvement**

---

### 🔷 MEDIUM IMPACT, HIGH EFFORT

#### 5. **Stream from Zip Without Temp Files** (Est. 30-40% speedup)
**Current Issue**: Extract → Write temp file → Read temp file → Parse
```python
# Current flow
zip → decompress → write temp → read temp → peppi_py
      (0.089s)    (0.012s)     (included)   (0.176s)
```

**Solution**: Parse directly from bytes if peppi_py supports it
```python
with zipfile.ZipFile(zip_path) as zf:
    with zf.open(member) as src:
        slp_bytes = gzip.decompress(src.read()) if is_gz else src.read()
        game = peppi_py.read_slippi_bytes(slp_bytes)  # Need this API
```

**Impact**:
- Eliminates temp file I/O (saves ~0.012s write + disk ops)
- Reduces memory pressure
- **30-40% speedup** on zip processing
- **No impact** on directory processing

**Effort**: 8-16 hours
- **Check if peppi_py supports bytes** (likely NOT - Rust bindings usually require paths)
- Options:
  1. Write to in-memory file-like object with temp path (hacky)
  2. Fork peppi_py and add bytes support (major effort)
  3. Use NamedTemporaryFile with `delete=False` (minor improvement)

**Priority**: ⭐⭐ **Only worth it if peppi_py supports bytes natively**

---

#### 6. **Custom Metadata Parser** (Est. 40-50% speedup on sanity checks)
**Current Issue**: `peppi_py.read_slippi(..., skip_frames=True)` still parses full metadata
- Parses all player data, stage info, UCF settings, etc.
- We only need: `is_teams`, `is_pal`, `stage`, `timer`, `player_count`, `player_types`

**Solution**: Write minimal .slp parser for sanity checks only
```python
def fast_sanity_check(slp_path: Path) -> FilterFailure | None:
    """Parse only metadata fields needed for sanity checks."""
    with open(slp_path, 'rb') as f:
        # Read .slp header
        # Parse only required UBJSON fields
        # Return early on first failure
        pass
```

**Impact**:
- First pass becomes ~5-10x faster (from ~2.4ms to ~0.3ms)
- Files failing sanity checks processed **80% faster**
- Files passing sanity checks: **20% faster overall**

**Effort**: 16-24 hours
- Learn .slp file format (UBJSON)
- Implement minimal parser
- Handle edge cases
- Test against peppi_py for correctness

**Priority**: ⭐⭐ **High effort, only worth it for massive datasets**

---

### 🔵 LOW IMPACT, VARIES EFFORT

#### 7. **Parallel Zip Decompression** (Est. 5% speedup)
**Current Issue**: zlib decompression is serial (0.089s)

**Solution**: Use `indexed_gzip` or `parallel-gzip`
- Requires seekable compressed streams
- Most zip implementations don't support this

**Impact**:
- **5% speedup** best case
- May increase memory usage

**Effort**: 4-8 hours
- Test if zip format supports parallel decompression
- Likely not worth it

**Priority**: ⭐ **Skip unless decompression becomes >30% of time**

---

#### 8. **Cythonize Hot Paths** (Est. 10-15% speedup)
**Solution**: Rewrite `_active_stick_ratio`, `sanity_reason`, `quality_reason` in Cython

**Impact**:
- Pure Python → Cython: ~3-5x speedup on those functions
- Overall: **10-15% speedup**

**Effort**: 8-12 hours
- Write .pyx files
- Setup build system
- Maintain C compilation

**Priority**: ⭐ **Only if you can't use numba**

---

## Recommended Roadmap

### Phase 1: Quick Wins (2-4 hours total)
1. ✅ **Cache parsed game object** → 25-30% speedup
2. ✅ **Optimize chunking** → 10-15% speedup
3. ✅ **Add worker initializer** → 20% cold start improvement

**Expected Result**: **50-60% total speedup** for 6-8 hours work

---

### Phase 2: If Still Bottlenecked (6-10 hours)
4. **Vectorize/JIT active stick ratio** → 15-20% speedup
5. **NamedTemporaryFile optimization** → 5-10% speedup

**Expected Result**: Additional **20-25% speedup**

---

### Phase 3: Diminishing Returns (20+ hours)
6. Custom metadata parser → 40% on sanity failures only
7. Cython compilation → 10-15% overall

**Expected Result**: Additional **15-30%** depending on dataset

---

## ROI Analysis

| Optimization | Effort | Impact | ROI | Priority |
|--------------|--------|--------|-----|----------|
| Cache game object | 2h | 25-30% | ⭐⭐⭐⭐⭐ | **DO FIRST** |
| Optimize chunking | 0.5h | 10-15% | ⭐⭐⭐⭐⭐ | **DO FIRST** |
| Worker init | 0.5h | 20% cold | ⭐⭐⭐⭐ | Quick win |
| Vectorize stick ratio | 3h | 15-20% | ⭐⭐⭐ | Good |
| Stream from zip | 12h | 30-40% | ⭐⭐ | If bytes API exists |
| Custom parser | 20h | 40% sanity | ⭐⭐ | Large datasets only |
| Cythonize | 10h | 10-15% | ⭐ | Last resort |

---

## Next Steps

**I recommend focusing on Phase 1 first:**

1. **Cache game object** (biggest win, minimal risk)
2. **Chunking optimization** (5 minute change)
3. **Worker initialization** (nice QOL)

This gets you **~60% speedup** for **3-4 hours of work**.

After that, profile again and decide if further optimization is worth it based on your actual workload size.
