#!/bin/bash
# Verification script for Nsight Systems optimizations
# Usage: ./scripts/verify_optimizations.sh x_before.sqlite x_after.sqlite

set -euo pipefail

BEFORE_DB="${1:-x.sqlite}"
AFTER_DB="${2:-x_after.sqlite}"

if [ ! -f "$BEFORE_DB" ]; then
    echo "Error: Before database not found: $BEFORE_DB"
    exit 1
fi

if [ ! -f "$AFTER_DB" ]; then
    echo "Warning: After database not found: $AFTER_DB"
    echo "Run new profile first: nsys profile --trace=cuda,nvtx --sample=cpu --gpu-metrics-devices=all -o x_after python train.py"
    exit 1
fi

echo "=================================================="
echo "Nsight Systems Optimization Verification"
echo "=================================================="
echo "Before: $BEFORE_DB"
echo "After:  $AFTER_DB"
echo ""

# Function to run query on both DBs and compare
run_comparison() {
    local metric_name="$1"
    local query="$2"
    local target_op="$3"  # gt (>), lt (<), eq (=)
    local target_value="$4"
    local unit="$5"

    echo "----------------------------------------"
    echo "Metric: $metric_name"
    echo "----------------------------------------"

    before_value=$(sqlite3 "$BEFORE_DB" "$query" 2>/dev/null || echo "N/A")
    after_value=$(sqlite3 "$AFTER_DB" "$query" 2>/dev/null || echo "N/A")

    printf "Before: %s %s\n" "$before_value" "$unit"
    printf "After:  %s %s\n" "$after_value" "$unit"

    if [ "$before_value" = "N/A" ] || [ "$after_value" = "N/A" ]; then
        echo "Status: ⚠️  INCOMPLETE (missing data)"
        return
    fi

    # Calculate improvement
    if [ "$before_value" != "0" ] 2>/dev/null; then
        improvement=$(echo "scale=1; (($before_value - $after_value) / $before_value) * 100" | bc 2>/dev/null || echo "N/A")
        if [ "$improvement" != "N/A" ]; then
            printf "Change: %+.1f%%\n" "$improvement"
        fi
    fi

    # Check against target
    case "$target_op" in
        "gt")
            if [ "$after_value" != "N/A" ] && (( $(echo "$after_value > $target_value" | bc -l) )); then
                echo "Status: ✅ PASS (> $target_value $unit)"
            else
                echo "Status: ❌ FAIL (target: > $target_value $unit)"
            fi
            ;;
        "lt")
            if [ "$after_value" != "N/A" ] && (( $(echo "$after_value < $target_value" | bc -l) )); then
                echo "Status: ✅ PASS (< $target_value $unit)"
            else
                echo "Status: ❌ FAIL (target: < $target_value $unit)"
            fi
            ;;
        "eq")
            if [ "$after_value" = "$target_value" ]; then
                echo "Status: ✅ PASS (= $target_value $unit)"
            else
                echo "Status: ❌ FAIL (target: = $target_value $unit)"
            fi
            ;;
    esac
    echo ""
}

# Test 1: GPU Utilization
run_comparison \
    "GPU Utilization" \
    "WITH timeline AS (SELECT MIN(start) as min_start, MAX(end) as max_end FROM CUPTI_ACTIVITY_KIND_KERNEL), kernel_time AS (SELECT SUM(end - start) as total_kernel_ns FROM CUPTI_ACTIVITY_KIND_KERNEL) SELECT ROUND(100.0 * total_kernel_ns / (max_end - min_start), 1) FROM timeline, kernel_time;" \
    "gt" \
    "85" \
    "%"

# Test 2: Stream Synchronization Count
run_comparison \
    "Stream Sync Count" \
    "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId = s.id WHERE s.value = 'cudaStreamSynchronize_v3020';" \
    "lt" \
    "100" \
    "calls"

# Test 3: Total Kernel Count
run_comparison \
    "Total Kernel Launches" \
    "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL;" \
    "lt" \
    "8000" \
    "kernels"

# Test 4: cudaMemcpyAsync overhead
run_comparison \
    "Avg cudaMemcpyAsync Overhead" \
    "SELECT ROUND(AVG(end - start)/1e3, 1) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId = s.id WHERE s.value = 'cudaMemcpyAsync_v3020';" \
    "lt" \
    "50" \
    "µs"

# Test 5: Pageable memory issues
run_comparison \
    "Pageable Memcpy Issues (>100µs overhead)" \
    "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId = s.id WHERE s.value = 'cudaMemcpyAsync_v3020' AND (r.end - r.start) > 100000;" \
    "eq" \
    "0" \
    "instances"

# Test 6: Average kernel duration (should increase with fusion)
run_comparison \
    "Average Kernel Duration" \
    "SELECT ROUND(AVG(end - start)/1e3, 1) FROM CUPTI_ACTIVITY_KIND_KERNEL;" \
    "gt" \
    "200" \
    "µs"

# Test 7: Total trace duration (for throughput comparison)
echo "----------------------------------------"
echo "Additional Metrics"
echo "----------------------------------------"
before_duration=$(sqlite3 "$BEFORE_DB" "SELECT ROUND((MAX(end) - MIN(start))/1e9, 2) FROM CUPTI_ACTIVITY_KIND_KERNEL;" 2>/dev/null || echo "N/A")
after_duration=$(sqlite3 "$AFTER_DB" "SELECT ROUND((MAX(end) - MIN(start))/1e9, 2) FROM CUPTI_ACTIVITY_KIND_KERNEL;" 2>/dev/null || echo "N/A")
echo "Trace Duration:"
echo "  Before: $before_duration seconds"
echo "  After:  $after_duration seconds"
echo ""

# Test 8: GPU idle gaps
before_gaps=$(sqlite3 "$BEFORE_DB" "SELECT COUNT(*) FROM (SELECT k1.end, k2.start, (k2.start - k1.end)/1e9 as gap_sec FROM CUPTI_ACTIVITY_KIND_KERNEL k1 JOIN CUPTI_ACTIVITY_KIND_KERNEL k2 ON k2.start > k1.end WHERE (k2.start - k1.end) > 500e6 ORDER BY k1.end LIMIT 100) WHERE gap_sec > 0.5;" 2>/dev/null || echo "N/A")
after_gaps=$(sqlite3 "$AFTER_DB" "SELECT COUNT(*) FROM (SELECT k1.end, k2.start, (k2.start - k1.end)/1e9 as gap_sec FROM CUPTI_ACTIVITY_KIND_KERNEL k1 JOIN CUPTI_ACTIVITY_KIND_KERNEL k2 ON k2.start > k1.end WHERE (k2.start - k1.end) > 500e6 ORDER BY k1.end LIMIT 100) WHERE gap_sec > 0.5;" 2>/dev/null || echo "N/A")
echo "GPU Idle Gaps (>500ms):"
echo "  Before: $before_gaps gaps"
echo "  After:  $after_gaps gaps"
if [ "$after_gaps" = "0" ]; then
    echo "  Status: ✅ PASS (no large gaps)"
else
    echo "  Status: ⚠️  WARNING (still has gaps)"
fi
echo ""

echo "=================================================="
echo "Summary"
echo "=================================================="
echo "Review the results above. Key targets:"
echo "  • GPU Utilization: >85% (was 61.2%)"
echo "  • Stream Syncs: <100 calls (was 1,253)"
echo "  • Kernel Count: <8,000 (was 19,629)"
echo "  • Pageable Issues: 0 instances"
echo ""
echo "Next steps:"
echo "  1. If tests fail, review optimization implementation"
echo "  2. Run: nsys analyze $AFTER_DB"
echo "  3. Compare: nsys stats --report cuda_api_sum $AFTER_DB"
echo "=================================================="
