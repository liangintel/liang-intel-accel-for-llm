#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sweep KV block size at constant total payload and attribute the cost per stage.

Every case transfers the same number of bytes, so a throughput change can only come
from per-block fixed overhead, not from the amount of data moved.
"""

import argparse
import json
import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(SCRIPT_DIR, "kvstore_benchmark.py")

# shape = (2, cache_blocks, block_tokens, kv_heads, head_dim)
KV_HEADS = 4
HEAD_DIM = 128
NUM_LAYERS = 32
ITEMSIZE = 2  # bf16
# bytes per block = 2 * block_tokens * KV_HEADS * HEAD_DIM * ITEMSIZE
BYTES_PER_TOKEN = 2 * KV_HEADS * HEAD_DIM * ITEMSIZE  # 2048 B

PAT = {
    "put_s": r"^\s*PUT:\s+([\d.]+) s",
    "put_gibs": r"^\s*PUT:.*?([\d.]+) GiB/s",
    "get_s": r"^\s*GET:\s+([\d.]+) s",
    "get_gibs": r"^\s*GET:.*?([\d.]+) GiB/s",
    "ratio": r"Compression ratio:\s+([\d.]+)x",
    "comp_gbps": r"Compress throughput:\s+([\d.]+) GB/s",
    "comp_ms": r"Compress throughput:.*?in ([\d.]+) ms",
    "decomp_gbps": r"Decompress throughput:\s+([\d.]+) GB/s",
    "decomp_ms": r"Decompress throughput:.*?in ([\d.]+) ms",
}


def parse(out: str) -> dict:
    res = {}
    for key, pat in PAT.items():
        m = re.search(pat, out, re.M)
        res[key] = float(m.group(1)) if m else None
    res["verified"] = "Verification: passed" in out
    return res


def run_case(block_tokens: int, total_mib: int, env_over: dict, timeout: int) -> dict:
    """One benchmark run. cache_blocks is derived so total bytes stay constant."""
    # total = bytes_per_block * (cache_blocks/2) * NUM_LAYERS
    total_bytes = total_mib * 1024 * 1024
    bytes_per_block = BYTES_PER_TOKEN * block_tokens
    cache_blocks = 2 * total_bytes // (bytes_per_block * NUM_LAYERS)
    if cache_blocks < 2:
        raise ValueError(f"block_tokens={block_tokens} too large for {total_mib} MiB")

    env = os.environ.copy()
    env.setdefault("IAXL_CACHE_CACHEGROUP_NUM", "1000000")
    env.update({k: str(v) for k, v in env_over.items()})

    cmd = [
        "numactl", "--cpunodebind=0", "--membind=0",
        sys.executable, BENCH,
        "--kv-cache-shape", "2", str(cache_blocks), str(block_tokens),
        str(KV_HEADS), str(HEAD_DIM),
        "--num-layers", str(NUM_LAYERS),
    ]
    proc = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=timeout
    )
    res = parse(proc.stdout)
    res.update(
        block_tokens=block_tokens,
        cache_blocks=cache_blocks,
        kib_per_block=bytes_per_block / 1024,
        num_chunks=(cache_blocks // 2) * NUM_LAYERS,
        rc=proc.returncode,
    )
    if proc.returncode != 0 or res["put_s"] is None:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
        res["error"] = " | ".join(tail)
    return res


HDR = (
    f"{'blk_tok':>7} {'KiB/blk':>8} {'#chunks':>8} "
    f"{'PUT s':>7} {'PUT GiB/s':>10} {'GET s':>7} {'GET GiB/s':>10} "
    f"{'zip GB/s':>9} {'zip ms':>8} {'unzip GB/s':>11} {'unzip ms':>9} {'ratio':>6} {'ok':>3}"
)


def fmt_row(r: dict) -> str:
    def f(key, spec):
        v = r.get(key)
        return format(v, spec) if isinstance(v, (int, float)) else "-"

    if r.get("error"):
        return (
            f"{r['block_tokens']:>7} {r['kib_per_block']:>8.0f} {r['num_chunks']:>8} "
            f"  FAILED: {r['error'][:110]}"
        )
    return (
        f"{r['block_tokens']:>7} {r['kib_per_block']:>8.0f} {r['num_chunks']:>8} "
        f"{f('put_s', '>7.3f')} {f('put_gibs', '>10.3f')} "
        f"{f('get_s', '>7.3f')} {f('get_gibs', '>10.3f')} "
        f"{f('comp_gbps', '>9.3f')} {f('comp_ms', '>8.1f')} "
        f"{f('decomp_gbps', '>11.3f')} {f('decomp_ms', '>9.1f')} "
        f"{f('ratio', '>6.3f')} {'Y' if r['verified'] else 'N':>3}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--block-tokens", type=int, nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128],
        help="block_tokens values to sweep (dim 2 of the KV shape).",
    )
    ap.add_argument("--total-mib", type=int, default=512,
                    help="Total payload per run, held constant across block sizes.")
    ap.add_argument("--env", action="append", default=[], metavar="K=V",
                    help="Extra env var applied to every run. Repeatable.")
    ap.add_argument("--label", default="default", help="Name of this configuration.")
    ap.add_argument("--json-out", default=None, help="Append results as JSON lines.")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    env_over = {}
    for item in args.env:
        k, _, v = item.partition("=")
        env_over[k] = v

    print("=" * len(HDR))
    print(f"CONFIG: {args.label}   total={args.total_mib} MiB, layers={NUM_LAYERS}, "
          f"env={env_over or '{}'}")
    print("=" * len(HDR))
    print(HDR)
    print("-" * len(HDR))

    rows = []
    for bt in args.block_tokens:
        try:
            r = run_case(bt, args.total_mib, env_over, args.timeout)
        except Exception as exc:  # keep sweeping when one point blows up
            r = {"block_tokens": bt, "kib_per_block": BYTES_PER_TOKEN * bt / 1024,
                 "num_chunks": 0, "verified": False, "error": str(exc)}
        r["label"] = args.label
        r["env"] = env_over
        rows.append(r)
        print(fmt_row(r), flush=True)

    if args.json_out:
        with open(args.json_out, "a") as fp:
            for r in rows:
                fp.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
