#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Tuned KVStore accelerator profile. Source this before running a benchmark or vLLM:
#   source benchmark/kvstore/tuned_env.sh
#
# Measured on 2x Xeon 6767P (8x QAT 4xxx, 8x IAA, 8x DSA) + RTX 6000D, bf16,
# 32 layers, process pinned to NUMA node 0, block = 32 KiB, 512 MiB payload:
#
#   compression on   PUT  8.5 -> 25.2 GiB/s   GET  8.9 -> 26.6 GiB/s  (ratio 1.25x)
#   compression off  PUT  8.5 -> 28.6 GiB/s   GET  8.9 -> 36.3 GiB/s
#
# DSA requires CAP_SYS_RAWIO on this kernel; without it the profile still works but
# loses roughly half of the gain. Payload verification passes in every case.

# CPU DEFLATE only reaches ~50-115 MB/s on KV data. Because all backends pull from
# one shared task pool, a CPU worker that claims a block near the end of a batch
# stalls the whole layer, and the stall grows with block size. Turning it off is
# the single biggest win at large block sizes.
export IAXL_CPU_ZIP_ENABLE=${IAXL_CPU_ZIP_ENABLE:-0}

# IAA is off by default but compress and decompress scale almost linearly with
# instance count, and QAT and IAA run concurrently, so their compress throughput adds:
# QAT32 alone 23.5 GB/s, IAA16 alone 12.9 GB/s, together 32.9 GB/s.
export IAXL_IAA_ZIP_ENABLE=${IAXL_IAA_ZIP_ENABLE:-1}
export IAXL_IAA_INSTANCE_NUM=${IAXL_IAA_INSTANCE_NUM:-16}

# The default IAXL_QAT_DEVICES=0 puts all QAT instances on a single device.
# QAT compress is device-bound at ~3 GB/s per device, so use all eight. For one rank
# per socket use the four local devices instead (0,1,2,3 / 4,5,6,7) and IAA8.
export IAXL_QAT_ZIP_ENABLE=${IAXL_QAT_ZIP_ENABLE:-1}
export IAXL_QAT_DEVICES=${IAXL_QAT_DEVICES:-0,1,2,3,4,5,6,7}
export IAXL_QAT_INSTANCE_NUM=${IAXL_QAT_INSTANCE_NUM:-32}

# DSA+GDRCopy needs CAP_SYS_RAWIO on this kernel (6.8 idxd_cdev_mmap, INTEL-SA-01084).
# Without it the portal mmap returns EPERM and kv_xfer silently falls back to CUDA,
# which is *slower* than leaving DSA off because the failed mmap is retried per call.
# When it does work it is worth 2.1x on D2H and 2.8x on H2D at 32 KiB blocks.
if [[ -r /dev/dsa/wq0.0 ]] && \
   python3 -c "import mmap,os,sys
fd=os.open('/dev/dsa/wq0.0',os.O_RDWR)
try: mmap.mmap(fd,4096,flags=mmap.MAP_SHARED,prot=mmap.PROT_WRITE)
except OSError: sys.exit(1)
finally: os.close(fd)" 2>/dev/null; then
    export IAXL_DSA_GD_ENABLE=${IAXL_DSA_GD_ENABLE:-1}
    export IAXL_DSA_WQS=${IAXL_DSA_WQS:-wq0.0,wq2.0,wq4.0,wq6.0}
else
    export IAXL_DSA_GD_ENABLE=0
    echo "[tuned_env] DSA portal not mappable (needs CAP_SYS_RAWIO) -> DSA disabled"
fi

# kv_zip asserts OMP_NUM_THREADS == qat + iaa + cpu workers, so it is not free to
# tune while compression is on. With IAXL_KV_COMPRESSION=0 the same variable gates
# the per-chunk host copy into the DDR pool, where 24 measured best.
if [[ "${IAXL_KV_COMPRESSION:-1}" == "0" ]]; then
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-24}
else
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$((IAXL_QAT_INSTANCE_NUM + IAXL_IAA_INSTANCE_NUM))}
fi

# The pool stores every chunk with its own malloc, so the copy stage is dominated by
# first-touch page faults. Growing the heap in big steps and never trimming lifts that
# stage from ~18 GB/s to ~39 GB/s. Do NOT set MALLOC_MMAP_THRESHOLD_ here, it regresses.
export MALLOC_TRIM_THRESHOLD_=${MALLOC_TRIM_THRESHOLD_:--1}
export MALLOC_TOP_PAD_=${MALLOC_TOP_PAD_:-1073741824}

echo "[tuned_env] qat=${IAXL_QAT_INSTANCE_NUM}inst on dev ${IAXL_QAT_DEVICES}," \
     "iaa=${IAXL_IAA_INSTANCE_NUM}inst, cpu_zip=${IAXL_CPU_ZIP_ENABLE}," \
     "dsa=${IAXL_DSA_GD_ENABLE}, omp=${OMP_NUM_THREADS}"
