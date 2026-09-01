#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Isolate the GPU<->host chunk transfer stage of KVFlow.

Runs Context.xfer_chunks_batch / xfer_finish / xfer_wait with no compression and
no memory pool, so the measured time is purely descriptor overhead + PCIe time.
Fits t(n) = a + b*n over the chunk count to separate the fixed per-chunk cost
from the achieved bandwidth.
"""

import argparse
import time

import torch

from iaxl.torch_ext import Context, GpuTransferDirection
from iaxl.kvflow.scratch_pool import ScratchPool

KV, KV_HEADS, HEAD_DIM = 2, 4, 128
BLOCK_DIM = 1


def time_xfer(tensor, pool, indices, direction, iters):
    chunk_shape = list(tensor.shape)
    del chunk_shape[BLOCK_DIM]
    cpu_tensors = pool.allocate(len(indices), tuple(chunk_shape), tensor.dtype)
    stream = torch.cuda.Stream()

    best = float("inf")
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ctx = Context.create(tensor, BLOCK_DIM, direction, "bench", work_stream=stream)
        ctx.xfer_chunks_batch(indices, cpu_tensors)
        ctx.xfer_finish()
        ctx.xfer_wait()
        best = min(best, time.perf_counter() - t0)
    pool.release(cpu_tensors)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-tokens", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--cache-blocks", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    dtype = torch.bfloat16
    pool = ScratchPool(cache_size_gb=8.0)

    print(f"{'blk_tok':>7} {'KiB/blk':>8} {'pattern':>10} {'#chunks':>8} "
          f"{'MiB':>8} {'D2H ms':>8} {'D2H GiB/s':>10} {'us/chunk':>9} "
          f"{'H2D ms':>8} {'H2D GiB/s':>10}")
    print("-" * 100)

    for bt in args.block_tokens:
        cb = args.cache_blocks
        shape = (KV, cb, bt, KV_HEADS, HEAD_DIM)
        tensor = torch.zeros(shape, dtype=dtype, device="cuda")
        bytes_per_block = KV * bt * KV_HEADS * HEAD_DIM * dtype.itemsize

        patterns = {
            "strided2": list(range(0, cb, 2)),        # every other block
            "contig": list(range(0, cb // 2)),        # one contiguous run
            "half-strided": list(range(0, cb, 4)),    # sparser
        }
        for name, idx in patterns.items():
            n = len(idx)
            total = bytes_per_block * n
            d2h = time_xfer(tensor, pool, idx, GpuTransferDirection.D2H, args.iters)
            h2d = time_xfer(tensor, pool, idx, GpuTransferDirection.H2D, args.iters)
            print(
                f"{bt:>7} {bytes_per_block/1024:>8.0f} {name:>10} {n:>8} "
                f"{total/1024**2:>8.1f} {d2h*1e3:>8.2f} {total/d2h/1024**3:>10.3f} "
                f"{d2h/n*1e6:>9.3f} {h2d*1e3:>8.2f} {total/h2d/1024**3:>10.3f}",
                flush=True,
            )
        del tensor
        torch.cuda.empty_cache()

    # Reference ceiling: one flat contiguous D2H copy of the same volume.
    flat_mib = 256
    src = torch.zeros(flat_mib * 1024 * 1024 // 2, dtype=dtype, device="cuda")
    dst = torch.empty_like(src, device="cpu", pin_memory=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    el = time.perf_counter() - t0
    print("-" * 100)
    print(f"reference: single contiguous {flat_mib} MiB D2H = "
          f"{flat_mib/1024/el:.3f} GiB/s ({el*1e3:.2f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
