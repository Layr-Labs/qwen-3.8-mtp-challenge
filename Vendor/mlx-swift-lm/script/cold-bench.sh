#!/usr/bin/env bash
# Cold-cache benchmark harness.
#
# Strategy: read ~80 GB of unrelated large files to /dev/null, applying
# LRU page-cache pressure to evict the target model. Then run BenchLoad
# with (warmup=0, runs=2) — run 1 is post-eviction (cold), run 2 is
# page-cache-warm. We also cycle which displacement files are used on
# each iteration so they too get evicted on the next round.
#
# Usage: cold-bench.sh <model-dir> <iterations>

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <model-dir> <iterations>" >&2
    exit 64
fi
MODEL_DIR="$1"
ITERS="$2"

BIN="/Users/gaj/Documents/Builds/MLX-Swift/mlx-swift-lm-opt/.build/arm64-apple-macosx/release/BenchLoad"

# Build a long list of large unrelated files. We rotate through groups of
# ~16 (~80 GB) so each iteration's displacement reads are fresh (not still
# page-cache-warm from the previous iteration's displacement).
ALL_DISPLACE=()
while IFS= read -r f; do ALL_DISPLACE+=("$f"); done < <(
    find /Users/gaj/.cache/huggingface/hub -type f -name "*.safetensors" \
        -size +1000M ! -path "*$MODEL_DIR*" 2>/dev/null
)
echo "available displacement files: ${#ALL_DISPLACE[@]}"
echo "model:                        $MODEL_DIR"
echo "iterations:                   $ITERS"
echo ""

cold_times=()
warm_times=()

GROUP_SIZE=16
for i in $(seq 1 "$ITERS"); do
    echo "=== iteration $i ==="
    # Pick a fresh group of files for this iteration.
    start=$(( ((i - 1) * GROUP_SIZE) % ${#ALL_DISPLACE[@]} ))
    bytes=0
    files_used=0
    t0=$(python3 -c "import time; print(time.time())")
    for ((j=0; j<GROUP_SIZE; j++)); do
        idx=$(( (start + j) % ${#ALL_DISPLACE[@]} ))
        f="${ALL_DISPLACE[$idx]}"
        dd if="$f" of=/dev/null bs=1m 2>/dev/null || true
        sz=$(stat -f "%z" "$f" 2>/dev/null || echo 0)
        bytes=$(( bytes + sz ))
        files_used=$(( files_used + 1 ))
    done
    t1=$(python3 -c "import time; print(time.time())")
    gb=$(python3 -c "print(f'{$bytes/1024/1024/1024:.1f}')")
    secs=$(python3 -c "print(f'{($t1-$t0):.1f}')")
    echo "  displaced $files_used files, ${gb} GiB in ${secs}s"

    out=$("$BIN" "$MODEL_DIR" 2 0 2>&1)
    cold=$(echo "$out" | grep -E "^  \[run 1\]" | awk '{print $3}')
    warm=$(echo "$out" | grep -E "^  \[run 2\]" | awk '{print $3}')
    echo "  cold: ${cold} ms"
    echo "  warm: ${warm} ms"
    cold_times+=("$cold")
    warm_times+=("$warm")
done

echo ""
echo "=== summary ==="
echo "cold runs (ms): ${cold_times[*]}"
echo "warm runs (ms): ${warm_times[*]}"
