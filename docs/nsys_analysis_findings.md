# Nsight Systems Profile Analysis - Performance Optimization Report

**Date:** 2025-12-13
**Profile:** `x.nsys-rep` / `x.sqlite`
**Trace Duration:** 4.36 seconds
**GPU Utilization:** 61.2% (should be >90%)

---

## Executive Summary

Analyzed training pipeline using Nsight Systems and identified **three critical bottlenecks** that are causing ~40% performance loss. The primary issues are:

1. **Pageable memory in async transfers** - Causing 130ms blocking overhead per 4-byte transfer
2. **Excessive CPU-GPU synchronizations** - 1,253 sync calls totaling 719ms
3. **Misleading NVTX attribution** - CPU timeline doesn't reflect actual GPU work

**Expected speedup from fixes:** 40-60% overall throughput improvement.

---

## Problem 1: Pageable Memory Blocking "Async" Transfers

### Evidence

```sql
-- Query from CUPTI_ACTIVITY_KIND_MEMCPY
SELECT bytes, copyKind, api_duration_ms, gpu_duration_ms
FROM memcpy_analysis
WHERE api_duration_ms > 100;
```

**Results:**
```
Bytes  Direction  API Time (CPU)  GPU Time  Overhead
-----  ---------  --------------  --------  --------
4      DtoH       136 ms          1.6 µs    81,000x
4      DtoH       134 ms          1.6 µs    81,000x
4      DtoH       133 ms          1.6 µs    81,000x
```

**Expert System Detection:**
```
** CUDA Async Memcpy with Pageable Memory (cuda_memcpy_async):

The following APIs use PAGEABLE memory which causes asynchronous CUDA memcpy
operations to block and be executed synchronously.

Suggestion: If applicable, use PINNED memory instead.
```

**Key Statistics:**
- Total `cudaMemcpyAsync` time: **1,458ms (40.7% of total CPU time)**
- Number of transfers: 1,574
- Average overhead: 926µs per call
- Most expensive: 136ms for 4-byte transfer

### Root Cause

The code is using `torch.empty()` or standard Python allocations for host-side memory. When CUDA tries to copy from GPU to this memory:
1. OS must page-lock the memory (synchronous operation)
2. Forces GPU to wait for CPU memory management
3. Blocks the CUDA stream despite "Async" in the API name

**Code Location:** Found in multiple locations via grep:
```
train/logging.py:    all_stats = _compute_tensor_stats_batch(tensors).cpu().tolist()
train/logging.py:    loss_values = loss_tensor.cpu().tolist()
train/loop.py:334:   loss_value = float(forward_result.loss.detach().cpu().item())
train/metrics.py:    tp: np.ndarray = self.btn_true_positives.cpu().numpy()
```

### Recommended Fix

**Option A: Use pinned memory** (Immediate fix)
```python
# Create pinned buffer at initialization
self.loss_buffer = torch.empty(1, dtype=torch.float32, pin_memory=True)

# Transfer with non-blocking
loss_value_tensor.cpu(out=self.loss_buffer, non_blocking=True)
```

**Option B: Defer all `.cpu()` calls** (Better long-term)
```python
# Current (BAD):
loss_value = float(forward_result.loss.detach().cpu().item())  # Every step
components.loss_variance_tracker.add(loss_value)

# Proposed (GOOD):
components.loss_variance_tracker.add_tensor(forward_result.loss.detach())  # Keep on GPU

# Only sync when logging:
if _should_log():
    loss_values = loss_variance_tracker.get_and_reset().cpu().tolist()
```

### Verification in Future Profile

**Check these metrics:**
```bash
# 1. Verify pageable memory warnings are gone
nsys analyze x.nsys-rep | grep -A 10 "Pageable Memory"
# Should show: "no problems detected"

# 2. Check cudaMemcpyAsync overhead reduction
sqlite3 x.sqlite "
SELECT AVG(end - start)/1e6 as avg_overhead_ms
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE s.value = 'cudaMemcpyAsync_v3020';"
# Should be: <0.05ms (was 0.926ms)

# 3. Verify total memcpy time reduction
nsys stats --report cuda_api_sum x.nsys-rep | grep cudaMemcpyAsync
# Should be: <5% of total time (was 40.7%)
```

---

## Problem 2: Excessive Stream Synchronizations

### Evidence

**API Summary:**
```
API Name                     Time (%)  Total Time (ms)  Num Calls
---------------------------  --------  ---------------  ---------
cudaMemcpyAsync              40.7%     1,458 ms         1,574
cudaLaunchKernel             24.7%       886 ms        19,175
cudaStreamSynchronize        20.1%       719 ms         1,253  ← Problem
```

**Expert System Detection:**
```
** CUDA Synchronization APIs (cuda_api_sync):

The following are synchronization APIs that block the host until all issued
CUDA calls are complete.

Top 10 syncs:
- 72.8ms, 72.3ms, 72.3ms, 71.9ms, 71.5ms...
```

**SQL Analysis:**
```sql
SELECT
    COUNT(*) as total_syncs,
    SUM(end - start)/1e9 as total_time_sec,
    AVG(end - start)/1e6 as avg_ms,
    MAX(end - start)/1e6 as max_ms
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
WHERE nameId = (SELECT id FROM StringIds WHERE value = 'cudaStreamSynchronize_v3020');
```

**Results:**
```
total_syncs: 1,253
total_time: 0.719 seconds
avg_ms: 0.574 ms
max_ms: 72.8 ms
```

### Root Cause

Every `.cpu()`, `.item()`, and `.numpy()` call implicitly synchronizes:
```python
# These all trigger cudaStreamSynchronize:
x = tensor.cpu()          # Waits for all GPU work
y = tensor.item()         # Waits for all GPU work
z = tensor.numpy()        # Waits for all GPU work
print(f"loss: {loss}")    # If loss is on GPU, syncs!
```

**Found in training loop:**
```python
# train/loop.py:334 (executed EVERY step)
loss_value = float(forward_result.loss.detach().cpu().item())
components.loss_variance_tracker.add(loss_value)
```

This single line causes:
- 1 sync per training step
- Waits for all forward pass kernels to complete
- Creates GPU bubble while CPU does minimal work

### Recommended Fix

**High Priority Locations** (most frequent syncs):

1. **train/loop.py:334** - Defer loss tracking
   ```python
   # Before: Syncs every step
   loss_value = float(forward_result.loss.detach().cpu().item())

   # After: Accumulate on GPU
   self.loss_accumulator.append(forward_result.loss.detach())

   # Sync only when logging (every 100 steps)
   if _should_log():
       loss_values = torch.stack(self.loss_accumulator).cpu().tolist()
       self.loss_accumulator.clear()
   ```

2. **train/logging.py** - Batch all transfers
   ```python
   # Before: Multiple syncs
   stat1 = tensor1.cpu().item()
   stat2 = tensor2.cpu().item()

   # After: Single sync
   all_stats = torch.stack([tensor1, tensor2])
   stat1, stat2 = all_stats.cpu().tolist()
   ```

3. **train/metrics.py** - Defer metric computation
   ```python
   # Keep metrics on GPU, only sync at validation/logging time
   # Use torch.compile to fuse metric computations
   ```

### Verification in Future Profile

```bash
# 1. Check total sync count reduction
sqlite3 x.sqlite "
SELECT COUNT(*) as sync_count
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE s.value = 'cudaStreamSynchronize_v3020';"
# Should be: <100 (was 1,253)

# 2. Verify sync time reduction
nsys stats --report cuda_api_sum x.nsys-rep | grep Synchronize
# Should be: <5% of total time (was 20.1%)

# 3. Check GPU utilization improvement
sqlite3 x.sqlite "
WITH timeline AS (
    SELECT MIN(start) as min_start, MAX(end) as max_end
    FROM CUPTI_ACTIVITY_KIND_KERNEL
),
kernel_time AS (
    SELECT SUM(end - start) as total_kernel_ns
    FROM CUPTI_ACTIVITY_KIND_KERNEL
)
SELECT 100.0 * total_kernel_ns / (max_end - min_start) as gpu_util_pct
FROM timeline, kernel_time;"
# Should be: >85% (was 61.2%)
```

---

## Problem 3: NVTX Time Attribution Misleading

### Evidence

**Case Study: cudaStreamSynchronize at 25.4112s**

**What NVTX Shows:**
```
├─ model_inference:        14.7ms
├─ compute_sample_weights: 72.8ms  ← Appears slow!
```

**What Actually Happened:**
```sql
-- NVTX ranges
SELECT text, start/1e9, end/1e9, (end-start)/1e6 as duration_ms
FROM NVTX_EVENTS
WHERE start >= 25.395e9 AND start <= 25.485e9 AND eventType = 59
ORDER BY start;

-- Result:
model_inference:         25.397s - 25.411s (14.7ms)
compute_sample_weights:  25.411s - 25.484s (72.8ms)
```

**GPU Timeline:**
```sql
-- Kernels executing during "compute_sample_weights" NVTX range
SELECT COUNT(*), MIN(start/1e9), MAX(end/1e9)
FROM CUPTI_ACTIVITY_KIND_KERNEL
WHERE start >= 25411195108 AND end <= 25484044453;

-- Result: 212 kernels from 25.411s to 25.480s
```

**Correlation Analysis:**
```sql
-- What was queued before the sync?
SELECT s.value, r.start/1e9, r.correlationId
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE r.start >= 25.410e9 AND r.start <= 25.411227e9
ORDER BY r.start DESC LIMIT 5;

-- Result:
cudaStreamSynchronize_v3020  25.411227517  (starts here)
cudaMemcpyAsync_v3020        25.411216817  (4 bytes DtoH)
cudaLaunchKernel_v7000       25.411021341
...
```

**Memory Transfer Evidence:**
```sql
SELECT bytes, copyKind,
       r.start/1e9 as api_queued,
       m.start/1e9 as gpu_executed,
       (m.start - r.start)/1e6 as queue_delay_ms
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN CUPTI_ACTIVITY_KIND_MEMCPY m ON r.correlationId = m.correlationId
WHERE r.correlationId = 53538;

-- Result:
bytes: 4
copyKind: 1 (DtoH)
api_queued: 25.4112s
gpu_executed: 25.4805s
queue_delay: 69.3ms
```

### Root Cause

**NVTX shows CPU timeline, not GPU execution:**

1. CPU launches 200+ kernels async in "model_inference" (14ms CPU time)
2. CPU returns immediately, enters "compute_sample_weights" range
3. Something calls `.cpu()` on 4-byte value → triggers sync
4. Sync waits 69ms for GPU to finish those 200 kernels
5. NVTX attributes the 69ms GPU work to "compute_sample_weights"

**The Confusion:**
- High SM warp occupancy during "compute_sample_weights" → GPU doing work
- But it's doing *forward pass* work, not sample weight computation
- Sample weights only start AFTER the 69ms sync completes

### Recommended Fix

**1. Add Fine-Grained NVTX Ranges**

```python
with nvtx_range("forward_pass"):
    with nvtx_range("model_inference"):
        pred = model(inputs)

    # Add this to catch sync points:
    with nvtx_range("post_inference_sync"):
        # Any .cpu() calls here will be clearly attributed
        pass

    with nvtx_range("compute_sample_weights_actual"):
        weights = compute_component_sample_weights(...)
```

**2. Use CUDA Events for Accurate Timing**

```python
# Create events
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Measure GPU time
start_event.record()
pred = model(inputs)
end_event.record()

# Later (when logging):
torch.cuda.synchronize()  # Controlled sync
gpu_time_ms = start_event.elapsed_time(end_event)
```

**3. Enable CUDA API Backtraces**

For finding hidden syncs:
```bash
nsys profile \
  --trace=cuda,nvtx \
  --sample=cpu \
  --cudabacktrace=sync \  # Add this
  --gpu-metrics-devices=all \
  -o x_with_backtraces \
  python train.py
```

Then query:
```sql
-- Find sync calls with backtraces
SELECT
    s.value as api_name,
    r.start/1e9 as time_sec,
    (r.end - r.start)/1e6 as duration_ms,
    r.callchainId
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE s.value LIKE '%Sync%' AND callchainId IS NOT NULL
ORDER BY (r.end - r.start) DESC
LIMIT 10;
```

### Verification in Future Profile

**1. Check NVTX range correlation with GPU work:**
```sql
-- For each NVTX range, calculate actual GPU occupancy
WITH nvtx_ranges AS (
    SELECT text, start, end, (end - start) as duration
    FROM NVTX_EVENTS
    WHERE eventType = 59 AND text IN ('model_inference', 'compute_sample_weights')
),
kernel_overlap AS (
    SELECT
        n.text,
        n.duration as nvtx_duration,
        SUM(
            CASE
                WHEN k.end <= n.start OR k.start >= n.end THEN 0
                ELSE LEAST(k.end, n.end) - GREATEST(k.start, n.start)
            END
        ) as actual_gpu_time
    FROM nvtx_ranges n
    LEFT JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON 1=1
    GROUP BY n.text, n.duration
)
SELECT
    text,
    nvtx_duration/1e6 as nvtx_ms,
    actual_gpu_time/1e6 as gpu_ms,
    100.0 * actual_gpu_time / nvtx_duration as gpu_utilization_pct
FROM kernel_overlap;
```

**Expected result after fix:**
```
text                      nvtx_ms  gpu_ms  gpu_util_pct
------------------------  -------  ------  ------------
model_inference           15       14      93%  (was 100% due to async)
compute_sample_weights    3        2       67%  (was 95% but wrong kernels)
```

**2. Verify sync points are properly attributed:**
```bash
# Check that explicit sync ranges show up
nsys stats --report nvtx_sum x.nsys-rep | grep -i sync
# Should show new "post_inference_sync" range with the 69ms
```

---

## Problem 4: GPU Idle Gaps

### Evidence

**Expert System Detection:**
```
** GPU Gaps (gpu_gaps):

The following are ranges where a GPU is idle for more than 500ms.

Row#  Duration (s)  Start (s)  Device ID
----  ------------  ---------  ---------
1     5.96          17.639     0
```

**GPU Utilization by Time Period:**
```
** GPU Time Utilization (gpu_time_util):

Row#  In-Use (%)  Duration (s)  Start (s)
----  ----------  ------------  ---------
1     0.3%        6.35          17.619
2     19.8%       1.06          24.317
3     39.7%       0.35          27.843
```

### Root Cause

**Likely causes** (requires further profiling):
1. Data loading bottleneck (CPU-side preprocessing)
2. Epoch transitions / checkpoint saving
3. Dataset initialization / chunk loading
4. Initial compilation overhead (first batch)

**Note:** The trace doesn't have sufficient NVTX coverage in data loading pipeline to pinpoint exact cause.

### Recommended Fix

**1. Add NVTX ranges to data loading:**

```python
# In window_dataset.py or wherever data loading happens
with nvtx_range("data_loading"):
    for batch in dataloader:
        with nvtx_range("to_device"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with nvtx_range("train_step"):
            # Training step
            pass
```

**2. Profile data loader separately:**
```bash
# Add this to identify CPU bottlenecks
nsys profile \
  --trace=cuda,nvtx,osrt \  # Add OS runtime
  --sample=cpu \
  --cpuctxsw=process-tree \  # Add context switch tracking
  --gpu-metrics-devices=all \
  -o x_with_cpu_details \
  python train.py
```

**3. Check prefetch settings:**

Current training code should verify:
- `num_workers` for DataLoader
- `prefetch_factor`
- `persistent_workers=True`
- `pin_memory=True` for DataLoader

### Verification in Future Profile

```bash
# 1. Check GPU idle time reduction
nsys stats x.nsys-rep | grep -A 5 "GPU Gaps"
# Should show: no gaps >500ms

# 2. Verify overall GPU utilization
sqlite3 x.sqlite "
WITH timeline AS (
    SELECT MIN(start) as min_start, MAX(end) as max_end
    FROM CUPTI_ACTIVITY_KIND_KERNEL
),
kernel_time AS (
    SELECT SUM(end - start) as total_kernel_ns
    FROM CUPTI_ACTIVITY_KIND_KERNEL
)
SELECT
    100.0 * total_kernel_ns / (max_end - min_start) as gpu_util_pct
FROM timeline, kernel_time;"
# Should be: >85% (was 61.2%)

# 3. Check time regions
nsys stats x.nsys-rep | grep -A 10 "GPU Time Utilization"
# Should show: all regions >70% utilization
```

---

## Problem 5: High Kernel Launch Overhead

### Evidence

**API Summary:**
```
API Name             Num Calls  Total Time (ms)  Avg (µs)
-------------------  ---------  ---------------  --------
cudaLaunchKernel     19,175     886              46
```

**Kernel Statistics:**
```
Total kernels: 19,629
Total kernel time: 2.67 seconds
Average kernel duration: 136 µs
```

**Analysis:**
- 19,175 kernel launches in 4.36 seconds = 4,400 launches/sec
- Each launch averages 46µs of CPU overhead
- Many tiny kernels (min 0.768µs, median ~11µs)

### Root Cause

**Typical causes:**
1. Element-wise operations not fused (e.g., separate `add`, `mul`, `relu` kernels)
2. Broadcasting operations creating small kernels
3. Lack of kernel fusion in backward pass
4. Inefficient optimizer step (separate kernel per parameter)

**Example from kernel summary:**
```
vectorized_elementwise_kernel: 2,184 instances, avg 116µs
  - Should be fused with neighboring operations
```

### Recommended Fix

**1. Enable torch.compile for model:**

```python
# Current
model = GPT(config)

# Proposed
model = GPT(config)
model = torch.compile(model, mode="reduce-overhead")  # or "max-autotune"
```

**2. Fuse optimizer operations:**

```python
# Use fused optimizer
from torch.optim import AdamW
optimizer = AdamW(
    model.parameters(),
    lr=config.lr,
    fused=True  # Fuses all parameter updates into single kernel
)
```

**3. Use TorchScript for data preprocessing:**

```python
# If doing preprocessing on GPU
@torch.jit.script
def preprocess_batch(x: torch.Tensor) -> torch.Tensor:
    # This will fuse all operations
    return (x - mean) / std
```

### Verification in Future Profile

```bash
# 1. Check kernel count reduction
sqlite3 x.sqlite "
SELECT COUNT(*) as total_kernels,
       SUM(end - start)/1e9 as total_kernel_time_sec
FROM CUPTI_ACTIVITY_KIND_KERNEL;"
# Should be: <5,000 kernels (was 19,629)

# 2. Check launch overhead reduction
nsys stats --report cuda_api_sum x.nsys-rep | grep cudaLaunchKernel
# Should be: <10% of total time (was 24.7%)

# 3. Verify kernel fusion
nsys stats --report cuda_gpu_kern_sum x.nsys-rep | head -20
# Should see: fewer unique kernel names, larger average duration
```

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 hours, 30-40% speedup)

1. **Fix train/loop.py:334** - Defer loss.cpu().item()
   ```python
   # Change single line
   - loss_value = float(forward_result.loss.detach().cpu().item())
   + self.loss_buffer.append(forward_result.loss.detach())
   ```

2. **Batch logging transfers** in train/logging.py
   - Combine all `.cpu()` calls into single transfer

3. **Enable fused optimizer**
   ```python
   optimizer = AdamW(..., fused=True)
   ```

### Phase 2: Medium Effort (4-6 hours, additional 10-15% speedup)

1. **Refactor variance tracking** to accumulate on GPU
2. **Add pinned memory buffers** for all host transfers
3. **Defer all metric computations** to logging time
4. **Add fine-grained NVTX ranges** for better attribution

### Phase 3: Long-term (1-2 days, additional 5-10% speedup)

1. **Enable torch.compile** for model
2. **Profile data loading** with detailed NVTX
3. **Optimize data loader** prefetch settings
4. **Review and fuse custom CUDA kernels** if any

---

## Verification Checklist

After implementing fixes, run new profile and check:

### Automated Checks (SQL Queries)

```bash
# Run all verification queries
./scripts/verify_optimizations.sh x_before.sqlite x_after.sqlite
```

**Script contents:**
```sql
-- 1. GPU Utilization
WITH timeline AS (
    SELECT MIN(start) as min_start, MAX(end) as max_end
    FROM CUPTI_ACTIVITY_KIND_KERNEL
),
kernel_time AS (
    SELECT SUM(end - start) as total_kernel_ns
    FROM CUPTI_ACTIVITY_KIND_KERNEL
)
SELECT
    'GPU Utilization' as metric,
    ROUND(100.0 * total_kernel_ns / (max_end - min_start), 1) as value,
    '%' as unit,
    CASE
        WHEN 100.0 * total_kernel_ns / (max_end - min_start) > 85 THEN '✅ PASS'
        ELSE '❌ FAIL'
    END as status
FROM timeline, kernel_time;

-- 2. Sync Count
SELECT
    'Stream Sync Count' as metric,
    COUNT(*) as value,
    'calls' as unit,
    CASE
        WHEN COUNT(*) < 100 THEN '✅ PASS'
        ELSE '❌ FAIL'
    END as status
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE s.value = 'cudaStreamSynchronize_v3020';

-- 3. Kernel Launch Count
SELECT
    'Total Kernels' as metric,
    COUNT(*) as value,
    'kernels' as unit,
    CASE
        WHEN COUNT(*) < 8000 THEN '✅ PASS'
        ELSE '❌ FAIL'
    END as status
FROM CUPTI_ACTIVITY_KIND_KERNEL;

-- 4. Pageable Memory Issues
SELECT
    'Pageable Memcpy Issues' as metric,
    COUNT(*) as value,
    'instances' as unit,
    CASE
        WHEN COUNT(*) = 0 THEN '✅ PASS'
        ELSE '❌ FAIL'
    END as status
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN CUPTI_ACTIVITY_KIND_MEMCPY m ON r.correlationId = m.correlationId
WHERE (r.end - r.start) > 100000;  -- >100µs overhead indicates pageable
```

### Manual Verification

1. **Run Expert System:**
   ```bash
   nsys analyze x_after.nsys-rep
   ```
   - Should show no pageable memory warnings
   - Should show no high-duration sync warnings

2. **Compare API Summary:**
   ```bash
   nsys stats --report cuda_api_sum --format column x_after.nsys-rep
   ```
   - cudaMemcpyAsync: <5% (was 40.7%)
   - cudaStreamSynchronize: <5% (was 20.1%)
   - cudaLaunchKernel: <15% (was 24.7%)

3. **Check GPU Idle Time:**
   ```bash
   nsys stats x_after.nsys-rep | grep -A 5 "GPU Gaps"
   ```
   - Should show no gaps >500ms

4. **Verify Throughput:**
   ```bash
   # Compare samples/second in training logs
   # Should see 40-60% improvement
   ```

---

## Appendix: SQL Queries Used in Analysis

### Query 1: GPU Utilization
```sql
WITH timeline AS (
    SELECT MIN(start) as min_start, MAX(end) as max_end
    FROM CUPTI_ACTIVITY_KIND_KERNEL
),
kernel_time AS (
    SELECT SUM(end - start) as total_kernel_ns
    FROM CUPTI_ACTIVITY_KIND_KERNEL
)
SELECT
    (max_end - min_start)/1e9 as trace_duration_sec,
    total_kernel_ns/1e9 as kernel_time_sec,
    100.0 * total_kernel_ns / (max_end - min_start) as gpu_utilization_pct
FROM timeline, kernel_time;
```

### Query 2: Top Synchronization Calls
```sql
SELECT
    start/1e9 as start_time_sec,
    (end - start)/1e6 as duration_ms,
    correlationId
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE s.value = 'cudaStreamSynchronize_v3020'
ORDER BY (end - start) DESC
LIMIT 10;
```

### Query 3: NVTX Range Timing
```sql
SELECT
    text as nvtx_name,
    start/1e9 as start_sec,
    end/1e9 as end_sec,
    (end - start)/1e6 as duration_ms
FROM NVTX_EVENTS
WHERE eventType = 59
ORDER BY (end - start) DESC
LIMIT 20;
```

### Query 4: Memory Transfer Analysis
```sql
SELECT
    m.copyKind,
    COUNT(*) as copies,
    SUM(m.bytes)/(1024.0*1024.0) as total_MB,
    AVG(m.bytes)/(1024.0*1024.0) as avg_MB,
    SUM(m.end - m.start)/1e9 as total_time_sec,
    AVG(m.end - m.start)/1e6 as avg_time_ms
FROM CUPTI_ACTIVITY_KIND_MEMCPY m
GROUP BY m.copyKind
ORDER BY total_time_sec DESC;
```

### Query 5: Kernel Launch Overhead
```sql
SELECT
    s.value as api_name,
    COUNT(*) as calls,
    SUM(end - start)/1e9 as total_time_sec,
    AVG(end - start)/1e6 as avg_time_ms
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE s.value = 'cudaLaunchKernel_v7000'
GROUP BY r.nameId;
```

---

## References

- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [NVIDIA Nsight Systems Analysis Guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
- [SQLite Export Schema](https://docs.nvidia.com/nsight-systems/nsys-exporter/exported_data.html)
- [PyTorch CUDA Best Practices](https://pytorch.org/docs/stable/notes/cuda.html)

---

**Next Steps:**
1. Implement Phase 1 fixes
2. Run new profile: `nsys profile --trace=cuda,nvtx --sample=cpu --gpu-metrics-devices=all -o x_phase1 python train.py`
3. Verify improvements using checklist above
4. Iterate on Phase 2 and 3 as needed
