#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Does a contiguous pinned slab destination beat one pinned buffer per chunk?

The current ScratchPool hands out an independent pinned tensor per chunk, so
copy_chunks_batch must emit one descriptor per chunk. If the destination were
one slab with chunks at consecutive offsets, runs of consecutive block indices
could collapse into a single descriptor.
"""

import time

import torch

KV, KV_HEADS, HEAD_DIM = 2, 4, 128
DTYPE = torch.bfloat16
CACHE_BLOCKS = 8192


def bench(fn, iters=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> int:
    print(f"{'blk_tok':>7} {'KiB/blk':>8} {'#chunks':>8} {'MiB':>7} "
          f"{'per-chunk bufs':>15} {'slab+coalesced':>15} {'speedup':>8}")
    print("-" * 76)

    for bt in (1, 2, 4, 8, 16, 32, 64):
        src = torch.zeros((KV, CACHE_BLOCKS, bt, KV_HEADS, HEAD_DIM),
                          dtype=DTYPE, device="cuda")
        n = CACHE_BLOCKS // 2
        idx = list(range(n))  # one contiguous run of block indices
        bytes_per_block = KV * bt * KV_HEADS * HEAD_DIM * DTYPE.itemsize
        total = bytes_per_block * n

        # (a) status quo: an independent pinned buffer per chunk, one copy each
        bufs = [torch.empty((KV, bt, KV_HEADS, HEAD_DIM), dtype=DTYPE,
                            device="cpu", pin_memory=True) for _ in range(n)]

        def per_chunk():
            for i, b in zip(idx, bufs):
                b.copy_(src[:, i], non_blocking=True)

        # (b) proposal: one pinned slab, the whole run as a single strided copy
        slab = torch.empty((KV, n, bt, KV_HEADS, HEAD_DIM), dtype=DTYPE,
                           device="cpu", pin_memory=True)

        def slab_coalesced():
            slab.copy_(src[:, idx[0]:idx[-1] + 1], non_blocking=True)

        t_a = bench(per_chunk)
        t_b = bench(slab_coalesced)
        print(f"{bt:>7} {bytes_per_block/1024:>8.0f} {n:>8} {total/1024**2:>7.1f} "
              f"{total/t_a/1024**3:>11.2f} GiB/s {total/t_b/1024**3:>11.2f} GiB/s "
              f"{t_a/t_b:>7.1f}x", flush=True)

        del src, bufs, slab
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
