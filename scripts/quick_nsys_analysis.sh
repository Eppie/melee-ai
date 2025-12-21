#!/bin/bash
# Quick analysis script for Nsight Systems SQLite exports
# Usage: ./scripts/quick_nsys_analysis.sh x.sqlite

set -euo pipefail

DB="${1:-x.sqlite}"

if [ ! -f "$DB" ]; then
    echo "Error: Database not found: $DB"
    exit 1
fi

echo "=================================================="
echo "Quick Nsight Systems Analysis"
echo "Database: $DB"
echo "=================================================="
echo ""

# Helper function for queries
run_query() {
    local title="$1"
    local query="$2"
    echo "----------------------------------------"
    echo "$title"
    echo "----------------------------------------"
    sqlite3 -header -column "$DB" "$query"
    echo ""
}

# 1. Overall GPU Utilization
run_query "1. GPU Utilization" \
"WITH timeline AS (
    SELECT MIN(start) as min_start, MAX(end) as max_end
    FROM CUPTI_ACTIVITY_KIND_KERNEL
),
kernel_time AS (
    SELECT SUM(end - start) as total_kernel_ns
    FROM CUPTI_ACTIVITY_KIND_KERNEL
)
SELECT
    ROUND((max_end - min_start)/1e9, 2) as trace_duration_sec,
    ROUND(total_kernel_ns/1e9, 2) as kernel_time_sec,
    ROUND(100.0 * total_kernel_ns / (max_end - min_start), 1) as gpu_utilization_pct
FROM timeline, kernel_time;"

# 2. Top 10 CUDA API calls by time
run_query "2. Top 10 CUDA API Calls by Time" \
"SELECT
    s.value as api_name,
    COUNT(*) as calls,
    ROUND(SUM(end - start)/1e6, 1) as total_ms,
    ROUND(AVG(end - start)/1e3, 1) as avg_us,
    ROUND(100.0 * SUM(end - start) / (SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_RUNTIME), 1) as pct_total
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
GROUP BY r.nameId
ORDER BY SUM(end - start) DESC
LIMIT 10;"

# 3. Top 10 GPU kernels by time
run_query "3. Top 10 GPU Kernels by Total Time" \
"SELECT
    s.value as kernel_name,
    COUNT(*) as instances,
    ROUND(SUM(end - start)/1e6, 1) as total_ms,
    ROUND(AVG(end - start)/1e3, 1) as avg_us,
    ROUND(100.0 * SUM(end - start) / (SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL), 1) as pct_total
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.shortName = s.id
GROUP BY k.shortName
ORDER BY SUM(end - start) DESC
LIMIT 10;"

# 4. Memory transfer summary
run_query "4. Memory Transfer Summary by Type" \
"SELECT
    CASE copyKind
        WHEN 1 THEN 'Host->Device'
        WHEN 2 THEN 'Device->Host'
        WHEN 8 THEN 'Device->Device'
        ELSE 'Other'
    END as direction,
    COUNT(*) as transfers,
    ROUND(SUM(bytes)/(1024.0*1024.0), 1) as total_MB,
    ROUND(AVG(bytes)/(1024.0), 1) as avg_KB,
    ROUND(SUM(end - start)/1e6, 1) as total_ms,
    ROUND(AVG(end - start)/1e3, 1) as avg_us
FROM CUPTI_ACTIVITY_KIND_MEMCPY
GROUP BY copyKind
ORDER BY SUM(end - start) DESC;"

# 5. Stream synchronization analysis
run_query "5. Stream Synchronization Analysis" \
"SELECT
    COUNT(*) as total_syncs,
    ROUND(SUM(end - start)/1e6, 1) as total_ms,
    ROUND(AVG(end - start)/1e3, 1) as avg_us,
    ROUND(MIN(end - start)/1e3, 1) as min_us,
    ROUND(MAX(end - start)/1e3, 1) as max_us
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE s.value = 'cudaStreamSynchronize_v3020';"

# 6. NVTX ranges by duration (if available)
echo "----------------------------------------"
echo "6. Top 10 NVTX Ranges by Duration"
echo "----------------------------------------"
nvtx_count=$(sqlite3 "$DB" "SELECT COUNT(*) FROM NVTX_EVENTS WHERE eventType = 59;" 2>/dev/null || echo "0")
if [ "$nvtx_count" -gt "0" ]; then
    sqlite3 -header -column "$DB" \
    "SELECT
        text as nvtx_range,
        COUNT(*) as occurrences,
        ROUND(SUM(end - start)/1e6, 1) as total_ms,
        ROUND(AVG(end - start)/1e3, 1) as avg_us,
        ROUND(MAX(end - start)/1e3, 1) as max_us
    FROM NVTX_EVENTS
    WHERE eventType = 59
    GROUP BY textId
    ORDER BY SUM(end - start) DESC
    LIMIT 10;"
else
    echo "No NVTX events found in trace."
fi
echo ""

# 7. Potential pageable memory issues
run_query "7. Potential Pageable Memory Issues (cudaMemcpyAsync >100µs)" \
"SELECT
    COUNT(*) as suspect_transfers,
    ROUND(AVG(r.end - r.start)/1e3, 1) as avg_overhead_us,
    ROUND(MAX(r.end - r.start)/1e3, 1) as max_overhead_us,
    ROUND(AVG(m.bytes), 1) as avg_bytes
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN CUPTI_ACTIVITY_KIND_MEMCPY m ON r.correlationId = m.correlationId
JOIN StringIds s ON r.nameId = s.id
WHERE s.value = 'cudaMemcpyAsync_v3020'
  AND (r.end - r.start) > 100000;"

# 8. Kernel count and size distribution
run_query "8. Kernel Duration Distribution" \
"WITH kernel_buckets AS (
    SELECT
        CASE
            WHEN (end - start) < 1000 THEN '<1µs (tiny)'
            WHEN (end - start) < 10000 THEN '1-10µs (small)'
            WHEN (end - start) < 100000 THEN '10-100µs (medium)'
            WHEN (end - start) < 1000000 THEN '100µs-1ms (large)'
            ELSE '>1ms (very large)'
        END as bucket,
        (end - start) as duration
    FROM CUPTI_ACTIVITY_KIND_KERNEL
)
SELECT
    bucket,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL), 1) as pct_of_kernels,
    ROUND(SUM(duration)/1e6, 1) as total_ms,
    ROUND(100.0 * SUM(duration) / (SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL), 1) as pct_of_time
FROM kernel_buckets
GROUP BY bucket
ORDER BY
    CASE bucket
        WHEN '<1µs (tiny)' THEN 1
        WHEN '1-10µs (small)' THEN 2
        WHEN '10-100µs (medium)' THEN 3
        WHEN '100µs-1ms (large)' THEN 4
        ELSE 5
    END;"

echo "=================================================="
echo "Analysis Complete"
echo "=================================================="
echo ""
echo "Recommended next steps:"
echo "  • Run expert system: nsys analyze $DB"
echo "  • Check for issues: nsys analyze --help-rules"
echo "  • View full reports: nsys stats --help-reports"
echo ""
echo "For detailed optimization guide, see:"
echo "  docs/nsys_analysis_findings.md"
echo "=================================================="
