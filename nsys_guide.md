Below is a practical, “start broad → drill down” workflow for interpreting an **Nsight Systems (`nsys`)** capture like:

`nsys profile --trace=cuda,nvtx --sample=cpu --gpu-metrics-devices=all -o x python train.py`

This assumes you already have `x.nsys-rep` (and maybe `x.sqlite`).

---

## 1) What `nsys` is (and isn’t)

**Nsight Systems is a timeline profiler.** It answers questions like:

* *Is the GPU idle or busy over time?*
* *Where are the gaps (CPU launch overhead / sync / input pipeline stalls)?*
* *Which kernels / memcopies dominate wall time?*
* *How do those map to my code regions (NVTX)?*

It can also collect **device-level GPU metrics** (your `--gpu-metrics-devices=all`) to help identify *compute vs memory vs IO limitation*, but those metrics are **device-level** and don’t inherently know which process/context caused them. ([NVIDIA Docs][1])

When you get to “**why is *this kernel* slow**?” you typically pivot to **Nsight Compute (`ncu`)** for per-kernel hardware counters; `nsys` is the scout that finds *where* to look.

---

## 2) First commands to run after you captured `x.nsys-rep`

### 2.1 Get an overview fast (CLI)

Run the default stats set:

```bash
nsys stats x.nsys-rep
```

`nsys stats` will **export a sibling SQLite file** (e.g. `x.sqlite`) if it doesn’t already exist, then print default summary reports. ([NVIDIA Docs][1])

### 2.2 List what reports you can generate (important)

On your machine:

```bash
nsys stats --help-reports
# or for everything:
nsys stats --help-reports ALL
```

Different Nsight Systems versions ship slightly different report scripts; don’t memorize—**discover**.

### 2.3 The “triage set” of reports I run first

This is a solid baseline (names shown below are documented examples in NVIDIA’s guide):

```bash
nsys stats \
  --report cuda_api_sum \
  --report cuda_gpu_kern_sum \
  --report cuda_gpu_trace \
  --format csv,column \
  --output .,- \
  x.nsys-rep
```

This exact *pattern* (multiple reports, mixed outputs/formats) is in the Nsight Systems user guide, including `cuda_gpu_trace`, `cuda_gpu_kern_sum`, and `cuda_api_sum`. ([NVIDIA Docs][1])

What you get from each:

* **`cuda_api_sum`**: CPU-side CUDA API time (launch overhead, sync calls, allocator calls, etc.)
* **`cuda_gpu_kern_sum`**: GPU kernel time aggregated by kernel name (hot kernels)
* **`cuda_gpu_trace`**: event-level kernel/memcpy timeline in tabular form (good for “what happened when” without opening the GUI)

### 2.4 Run the Expert System (“tell me what looks fishy”)

Nsight Systems has an **Expert Systems** mode that runs rule scripts on the SQLite DB to detect common bottlenecks (sync memcpys, pageable-memory “async” copies that become sync, etc.). ([NVIDIA Docs][2])

```bash
nsys analyze x.nsys-rep
# and:
nsys analyze --help-rules
```

If you’re using the GUI, it will also show you the equivalent CLI invocation. ([NVIDIA Docs][2])

---

## 3) How to interpret common bottleneck patterns

### Pattern A: GPU is frequently idle (bubbles/gaps)

What it looks like:

* GPU kernel track has gaps
* GPU metrics show low utilization during those gaps
* CPU threads may be busy or blocked

How to debug:

1. **Check `cuda_api_sum`** for lots of time in blocking/sync calls (e.g., synchronizations, synchronous memcpys).
2. **Look for launch starvation**: small kernels with lots of CPU overhead per launch.
3. **Correlate with NVTX**: do the gaps align with dataloader / Python overhead / step transitions?

Next actions:

* Add more NVTX around *input pipeline vs forward vs backward vs optimizer*.
* Consider capturing **more CUDA APIs** if you suspect “missing” calls:

  * `--cuda-trace=all-apis=true` (but note overhead) ([NVIDIA Docs][3])
* If sync calls dominate, consider collecting CUDA API backtraces (below).

### Pattern B: GPU is busy, but it’s “the wrong kind of busy”

Use GPU Metrics to classify the limiter. GPU metrics are meant to answer things like:

* “Is my GPU idle?”
* “Are my SMs full / warp slots full?”
* “Am I using Tensor Cores?”
* “Am I blocked on IO (PCIe/NVLink/DRAM bandwidth)?” ([NVIDIA Docs][1])

Heuristics:

* **High SM active + low DRAM throughput** → likely compute-bound (then optimize math / use tensor cores / kernel efficiency)
* **High DRAM throughput + low SM active** → likely memory-bound (optimize memory access patterns, fusion, layout)
* **Both low** → you’re not feeding the GPU (CPU bottleneck, sync, input stalls, tiny kernels, serialization)

Important caveat: GPU Metrics is **device-level** and doesn’t know which process/context caused activity. If you need attribution between processes/contexts, use **GPU context switch tracing** (less precise counters, more attribution). ([NVIDIA Docs][1])

### Pattern C: Data transfer / Unified Memory issues

If you see lots of memcpy time, or “async” copies blocking:

* Expert System has rules specifically for **synchronous memcpy** and **pageable-memory async copies that become sync**, with suggestions like using pinned memory where appropriate. ([NVIDIA Docs][2])
* If you suspect UM faults, you can enable UM page fault tracking next time (overhead can be significant). ([NVIDIA Docs][3])

---

## 4) Getting more “granular” (without drowning in data)

Your capture already has CUDA + NVTX + CPU sampling + GPU metrics. Common “next-run” toggles (use selectively):

### 4.1 CUDA API backtraces (pinpoint *who* is calling sync/alloc/etc.)

Enable backtraces for expensive CUDA APIs:

```bash
nsys profile \
  --trace=cuda,nvtx \
  --sample=cpu \
  --cudabacktrace=sync,kernel,memory \
  -o x_bt \
  python train.py
```

`--cudabacktrace` can be filtered by class and threshold, but it has **significant overhead** and requires CPU sampling. ([NVIDIA Docs][3])

### 4.2 Track GPU memory usage by kernels (use sparingly)

```bash
nsys profile --trace=cuda,nvtx --cuda-memory-usage=true ...
```

This is explicitly warned as potentially **significant runtime overhead**. ([NVIDIA Docs][3])

### 4.3 If you suspect “missing” CUDA calls in the trace

```bash
nsys profile --trace=cuda,nvtx --cuda-trace-all-apis=true ...
```

Also warned as potentially significant overhead. ([NVIDIA Docs][3])

### 4.4 Control GPU metrics sets/frequency (and permissions)

GPU Metrics is controlled by:

* `--gpu-metrics-devices=[all|cuda-visible|none|<index>]`
* `--gpu-metrics-set=[<alias>|file:<file>]`
* `--gpu-metrics-frequency=[10..200000]` (default 10 kHz) ([NVIDIA Docs][1])

It **requires elevated permissions** (admin on Windows / sudo on Linux). ([NVIDIA Docs][1])

Also: device counter sampling infrastructure can conflict with other tools/services (commonly **DCGM**). If GPU metrics aren’t showing up or you get “already under profiling” style failures, you may need to stop/pause DCGM. ([NVIDIA Developer Forums][4])

---

## 5) When to use the GUI vs CLI

### CLI is best for:

* Quick “top N kernels / top N APIs” summaries
* Automated regression checks (CI)
* Exporting CSVs for your own dashboards/scripts
* Running Expert System rules

### GUI is best for:

* Understanding overlap (CPU/GPU/streams) visually
* Zooming into a single iteration/step and seeing causality
* Exploring NVTX range nesting vs kernel bursts
* Viewing GPU Metrics time-series alongside kernels

Open the report with Nsight Systems UI (varies by install, often `nsys-ui x.nsys-rep`) and focus on:

* **Where the critical path is** (what determines iteration wall time)
* **CUDA API track** (sync calls, allocator spikes, launch burst patterns)
* **Stream concurrency** (are you unintentionally serializing?)
* **NVTX ranges** (do they bracket “real” work or are they too coarse?)

---

## 6) Does it ever make sense to query the `.sqlite` directly?

Yes—often. Typical reasons:

* You want custom aggregations beyond shipped `nsys stats` scripts
* You want to join NVTX ↔ CUDA runtime calls ↔ kernel executions programmatically
* You want to extract per-iteration metrics automatically (e.g., in a perf pipeline)

### 6.1 How the SQLite is produced

You can export explicitly:

```bash
nsys export --type sqlite x.nsys-rep
```

`nsys export` supports output types like `sqlite, hdf, text, arrow, json, info`. ([NVIDIA Docs][5])

Or you can let `nsys stats` / `nsys analyze` auto-generate the `.sqlite` if it isn’t present. ([NVIDIA Docs][1])

You can also point `nsys stats` at a specific sqlite filename; if it doesn’t exist it will be created from the `.nsys-rep` (unless you’re already passing a sqlite input). ([NVIDIA Docs][1])

### 6.2 Schema reality check

* Tables are created **lazily** (so not every table exists in every export). ([NVIDIA Docs][6])
* The schema can evolve across versions (the exporter docs explicitly warn it may change). ([NVIDIA Docs][6])

So: treat the exporter docs as guidance, and use `sqlite3 x.sqlite` + `.schema` to confirm.

### 6.3 The most relevant tables (CUDA + NVTX + callchains)

From NVIDIA’s schema reference:

**Strings**

* `StringIds(id INTEGER PRIMARY KEY, value TEXT)` for resolving names. ([NVIDIA Docs][6])

**CUDA runtime API calls**

* `CUPTI_ACTIVITY_KIND_RUNTIME`
  Key fields include `start`, `end`, `globalTid`, `nameId` (→ `StringIds`), and crucially `correlationId` to link to kernels/memcpys. ([NVIDIA Docs][6])

**CUDA kernels**

* `CUPTI_ACTIVITY_KIND_KERNEL`
  Includes `start/end`, `deviceId/contextId/streamId`, `correlationId`, and kernel name fields like `demangledName`, `shortName`, etc. (often as StringIds). ([NVIDIA Docs][6])

**Memcpy / Memset**

* `CUPTI_ACTIVITY_KIND_MEMCPY`, `CUPTI_ACTIVITY_KIND_MEMSET` with sizes, kinds, correlationId, etc. ([NVIDIA Docs][6])

**Synchronization events**

* `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION` (useful for seeing sync activity as events). ([NVIDIA Docs][6])

**NVTX**

* `NVTX_EVENTS` (ranges/marks/categories; used heavily for correlation). ([NVIDIA Docs][7])

**CUDA callchains (if you enabled backtraces)**

* `CUDA_CALLCHAINS` (joinable via `CUPTI_ACTIVITY_KIND_RUNTIME.callchainId`). ([NVIDIA Docs][6])

Also, there are tables for cuBLAS/cuDNN events (`CUBLAS_EVENTS`, `CUDNN_EVENTS`) if you trace those. ([NVIDIA Docs][6])

### 6.4 Practical SQL “recipes” you’ll actually use

**A) Link CUDA runtime calls to kernels via `correlationId`**
NVIDIA’s examples show populating helper columns by joining `CUPTI_ACTIVITY_KIND_RUNTIME` ↔ `CUPTI_ACTIVITY_KIND_KERNEL` and resolving strings through `StringIds`. ([NVIDIA Docs][8])

**B) Rename / attribute kernels using innermost NVTX range**
NVIDIA provides a worked example joining `NVTX_EVENTS` + `CUPTI_ACTIVITY_KIND_RUNTIME` + `CUPTI_ACTIVITY_KIND_KERNEL` (via correlationId and time containment) to label kernels with the relevant NVTX range text. ([NVIDIA Docs][7])

**C) Backtrace inspection**
There’s also an example query pattern for joining `CUDA_CALLCHAINS` with runtime calls and resolving symbols via `StringIds`. ([NVIDIA Docs][7])

### 6.5 A good way to explore the DB (quickly)

In `sqlite3 x.sqlite`:

```sql
.headers on
.mode column
.schema
.tables
```

Those `.headers`/`.mode` helper commands are also suggested in NVIDIA’s SQLite examples doc. ([NVIDIA Docs][8])

---

## 7) A realistic “full workflow” for bottlenecks

1. **`nsys stats` triage**

   * Identify top CUDA APIs (sync/alloc/launch overhead)
   * Identify top kernels and whether there are too many small kernels
2. **Expert System (`nsys analyze`)**

   * Let the rules flag obvious patterns (sync copies, pageable-memory async, etc.) ([NVIDIA Docs][2])
3. **GUI timeline**

   * Confirm whether iteration wall time is GPU-bound or “GPU waiting on CPU”
   * Use NVTX to align compute bursts with code phases
4. **GPU metrics overlay**

   * Classify compute vs memory vs IO limitation at a high level ([NVIDIA Docs][1])
5. **SQL for custom attribution**

   * Join NVTX ↔ runtime ↔ kernels to get “per phase top kernels” programmatically ([NVIDIA Docs][7])
6. **Nsight Compute (`ncu`) on the top kernels**

   * Use `nsys` to choose kernels/regions; use `ncu` for per-kernel counters (occupancy, memory transactions, tensor usage, etc.)

[1]: https://docs.nvidia.com/nsight-systems/2025.1/UserGuide/index.html "User Guide — nsight-systems 2025.1 documentation"
[2]: https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html "Post-Collection Analysis Guide — Nsight Systems"
[3]: https://docs.nvidia.com/nsight-systems/UserGuide/index.html "User Guide — Nsight Systems"
[4]: https://forums.developer.nvidia.com/t/issue-with-gpu-metrics-collection-for-nvidia-a100-on-nsight-systems/294781?utm_source=chatgpt.com "Issue with GPU Metrics Collection for NVIDIA A100 on ..."
[5]: https://docs.nvidia.com/nsight-systems/2022.4/nsys-exporter/overview.html "Overview — NVIDIA Nsight Systems export  documentation"
[6]: https://docs.nvidia.com/nsight-systems/2022.4/nsys-exporter/exported_data.html "SQLite Export Schema Reference — NVIDIA Nsight Systems export  documentation"
[7]: https://docs.nvidia.com/nsight-systems/2021.5/nsys-exporter/examples.html "Common SQLite examples — NVIDIA Nsight Systems export  documentation"
[8]: https://docs.nvidia.com/nsight-systems/2022.4/nsys-exporter/examples.html "Common SQLite examples — NVIDIA Nsight Systems export  documentation"

