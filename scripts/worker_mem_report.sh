#!/usr/bin/env bash
set -euo pipefail

# Report memory breakdown for pt_data_worker processes using smaps_rollup.
# Shows RSS, PSS, and private (approx USS) per worker.

echo "Collecting pt_data_worker memory..."
PIDS=($(ps -C pt_data_worker -o pid= | awk '{print $1}'))

if [[ ${#PIDS[@]} -eq 0 ]]; then
  echo "No pt_data_worker processes found."
  exit 0
fi

printf "%-8s %-10s %-10s %-10s %-10s\n" "PID" "RSS(MiB)" "PSS(MiB)" "PRIV(MiB)" "CMD"

for pid in "${PIDS[@]}"; do
  if [[ ! -r "/proc/$pid/smaps_rollup" ]]; then
    echo "PID $pid: smaps_rollup not readable (process may have exited)."
    continue
  fi
  rss_kb=$(grep -m1 "^Rss:" /proc/$pid/smaps_rollup | awk '{print $2}')
  pss_kb=$(grep -m1 "^Pss:" /proc/$pid/smaps_rollup | awk '{print $2}')
  priv_clean_kb=$(grep -m1 "^Private_Clean:" /proc/$pid/smaps_rollup | awk '{print $2}')
  priv_dirty_kb=$(grep -m1 "^Private_Dirty:" /proc/$pid/smaps_rollup | awk '{print $2}')
  priv_kb=$((priv_clean_kb + priv_dirty_kb))
  cmd=$(ps -p "$pid" -o cmd=)
  printf "%-8s %-10.2f %-10.2f %-10.2f %-10s\n" \
    "$pid" \
    "$(bc <<<"scale=2;$rss_kb/1024")" \
    "$(bc <<<"scale=2;$pss_kb/1024")" \
    "$(bc <<<"scale=2;$priv_kb/1024")" \
    "$cmd"
done
