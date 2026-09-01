#!/usr/bin/env bash
# Sweep the raw DSA <-> GPU transfer ceiling over 1..8 GPUs x 1..8 DSA work queues.
#
# Needs root: mapping a DSA portal requires CAP_SYS_RAWIO on kernels carrying the
# INTEL-SA-01084 fix.
#
#   sudo -n benchmark/kvstore/dsa_xfer_sweep.sh [mib_per_gpu] [iters] [inflight] [chunk]
#
# Emits TSV on stdout: gpus, wqs, dir, best_GBps, avg_GBps, us_per_chunk.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$ROOT/iaxl/csrc/dsa_xfer_test"

MIB=${1:-256}
ITERS=${2:-5}
INFLIGHT=${3:-4}
CHUNK=${4:-32768}

[[ -x "$BIN" ]] || { echo "missing $BIN (run: make -C $ROOT/iaxl/csrc dsa_xfer_test)" >&2; exit 1; }

# GPUs are paired behind four PCIe switches (measured via /sys/bus/pci/devices/*):
#   36:00.0 -> gpu0,gpu1 | 46:00.0 -> gpu2,gpu3   (NUMA 0)
#   b6:00.0 -> gpu4,gpu5 | c6:00.0 -> gpu6,gpu7   (NUMA 1)
# A switch uplink saturates at ~40 GB/s D2H, so pick one GPU per switch first
# (and alternate sockets) before doubling up.
GPU_SET=([1]="0" [2]="0,4" [3]="0,2,4" [4]="0,2,4,6" [5]="0,2,4,6,1" \
         [6]="0,2,4,6,1,5" [7]="0,2,4,6,1,3,5" [8]="0,1,2,3,4,5,6,7")
# dsa0/2/4/6 are on NUMA 0, dsa8/10/12/14 on NUMA 1; same balancing rule.
WQ_SET=([1]="wq0.0" [2]="wq0.0,wq8.0" [3]="wq0.0,wq2.0,wq8.0" \
        [4]="wq0.0,wq2.0,wq8.0,wq10.0" [5]="wq0.0,wq2.0,wq4.0,wq8.0,wq10.0" \
        [6]="wq0.0,wq2.0,wq4.0,wq8.0,wq10.0,wq12.0" \
        [7]="wq0.0,wq2.0,wq4.0,wq6.0,wq8.0,wq10.0,wq12.0" \
        [8]="wq0.0,wq2.0,wq4.0,wq6.0,wq8.0,wq10.0,wq12.0,wq14.0")

printf 'gpus\twqs\tdir\tbest_GBps\tavg_GBps\tus_per_chunk\n'

for n in 1 2 3 4 5 6 7 8; do
  for m in 1 2 3 4 5 6 7 8; do
    out=$(CUDA_VISIBLE_DEVICES="${GPU_SET[$n]}" "$BIN" \
            --gpus "$n" --wqs "${WQ_SET[$m]}" --mib-per-gpu "$MIB" \
            --iters "$ITERS" --inflight "$INFLIGHT" --chunk "$CHUNK" 2>/dev/null) || {
      printf '%d\t%d\tFAILED\t\t\t\n' "$n" "$m"; continue; }
    echo "$out" | awk -F'\t' '$1 == "RESULT" { printf "%s\t%s\t%s\t%s\t%s\t%s\n", $3, $4, $2, $5, $6, $7 }'
  done
done
