#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Scale the KVStore benchmark across GPUs and report aggregate PUT/GET throughput.

One process per GPU, each with the same payload, all timing the same phase at the
same time (see --sync-dir in kvstore_benchmark.py). QAT devices, IAA NUMA nodes and
DSA work queues are partitioned so every process only drives accelerators that are
local to its GPU, which is what a real one-rank-per-GPU deployment would do.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(SCRIPT_DIR, "kvstore_benchmark.py")

KV_HEADS = 4
HEAD_DIM = 128
NUM_LAYERS = 32
BYTES_PER_TOKEN = 2 * KV_HEADS * HEAD_DIM * 2  # bf16

GPU_NODE = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}
QAT_DEVS = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
DSA_WQS = {0: ["wq0.0", "wq2.0", "wq4.0", "wq6.0"],
           1: ["wq8.0", "wq10.0", "wq12.0", "wq14.0"]}
IAA_INST_PER_NODE = 16  # 4 IAA devices x IAXL_IAA_ZIP_INSTANCES_PER_DEVICE(4)

# Balance across sockets first; GPUs 0-3 are on NUMA 0 and 4-7 on NUMA 1.
GPU_SETS = {1: [0], 2: [0, 4], 3: [0, 1, 4], 4: [0, 1, 4, 5],
            5: [0, 1, 2, 4, 5], 6: [0, 1, 2, 4, 5, 6],
            7: [0, 1, 2, 3, 4, 5, 6], 8: [0, 1, 2, 3, 4, 5, 6, 7]}

PAT = {
    "put_s": r"^\s*PUT:\s+([\d.]+) s",
    "put_gibs": r"^\s*PUT:.*?([\d.]+) GiB/s",
    "get_s": r"^\s*GET:\s+([\d.]+) s",
    "get_gibs": r"^\s*GET:.*?([\d.]+) GiB/s",
    "ratio": r"Compression ratio:\s+([\d.]+)x",
    "comp_gbps": r"Compress throughput:\s+([\d.]+) GB/s",
    "decomp_gbps": r"Decompress throughput:\s+([\d.]+) GB/s",
}


def worker_port(gpu: int) -> int:
    return 18800 + gpu


def split(items: list, nparts: int, part: int) -> list:
    """Contiguous balanced split; the remainder goes to the low parts so nothing is dropped."""
    if nparts >= len(items):
        return [items[part % len(items)]]
    per, rem = divmod(len(items), nparts)
    start = part * per + min(part, rem)
    return items[start:start + per + (1 if part < rem else 0)]


def share(total: int, nparts: int, part: int) -> int:
    """Split a scalar budget the same way, keeping the sum exactly equal to total."""
    per, rem = divmod(total, nparts)
    return max(1, per + (1 if part < rem else 0))


def build_env(gpu: int, peers_on_node: int, rank_on_node: int, single_node: bool,
              compression: bool, backends: str, iaa_total: int,
              cache_root: str, extra: dict) -> dict:
    node = GPU_NODE[gpu]
    # With GPUs on one socket only, the far socket's accelerators would otherwise idle,
    # so the pool stays at 8 QAT devices for every GPU count (at the cost of locality).
    qat_pool = QAT_DEVS[0] + QAT_DEVS[1] if single_node else QAT_DEVS[node]
    iaa_budget = iaa_total if single_node else iaa_total // 2
    qat = split(qat_pool, peers_on_node, rank_on_node)
    wqs = split(DSA_WQS[node], peers_on_node, rank_on_node)
    # With compression off nothing is submitted to an accelerator, but OMP_NUM_THREADS
    # still drives the memcpy loop, so keep the full thread budget for comparability.
    use_qat = backends in ("qat", "both") or not compression
    use_iaa = backends in ("iaa", "both") or not compression
    qat_inst = 4 * len(qat) if use_qat else 0
    iaa_inst = share(iaa_budget, peers_on_node, rank_on_node) if use_iaa else 0
    # IAXL_IAA_DEVICES lists NUMA nodes; one node caps out at IAA_INST_PER_NODE.
    iaa_nodes = str(node) if iaa_inst <= IAA_INST_PER_NODE else "0,1"

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "IAXL_CACHE_DIR": os.path.join(cache_root, f"gpu{gpu}"),
        "IAXL_CACHE_CACHEGROUP_NUM": "1000000",
        "IAXL_KV_COMPRESSION": "1" if compression else "0",
        "IAXL_CPU_ZIP_ENABLE": "0",
        "IAXL_QAT_ZIP_ENABLE": "1" if use_qat else "0",
        "IAXL_QAT_DEVICES": ",".join(str(d) for d in qat),
        "IAXL_QAT_INSTANCE_NUM": str(qat_inst),
        "IAXL_IAA_ZIP_ENABLE": "1" if use_iaa else "0",
        "IAXL_IAA_DEVICES": iaa_nodes,
        "IAXL_IAA_INSTANCE_NUM": str(iaa_inst),
        "IAXL_DSA_GD_ENABLE": "1",
        "IAXL_DSA_WQS": ",".join(wqs),
        "MALLOC_TRIM_THRESHOLD_": "-1",
        "MALLOC_TOP_PAD_": "1073741824",
        # kv_zip asserts this equality while compression is on.
        "OMP_NUM_THREADS": str(qat_inst + iaa_inst),
        # Every process runs its own management server; without this they all
        # try to bind 18800 and every process after the first one dies.
        "IAXL_API_WORKER_BASE_PORT": str(worker_port(gpu)),
    })
    env.update({k: str(v) for k, v in extra.items()})
    return env


def run_group(gpus: list, block_tokens: int, total_mib: int, compression: bool,
              backends: str, iaa_total: int, extra: dict, timeout: int) -> list:
    """Launch one process per GPU; they rendezvous so the timed phases overlap."""
    bytes_per_block = BYTES_PER_TOKEN * block_tokens
    cache_blocks = 2 * total_mib * 1024 * 1024 // (bytes_per_block * NUM_LAYERS)

    cache_root = tempfile.mkdtemp(prefix="kvscale_")
    sync_dir = os.path.join(cache_root, "sync")
    os.makedirs(sync_dir, exist_ok=True)

    node_ranks = {}
    peers = {n: sum(1 for g in gpus if GPU_NODE[g] == n) for n in (0, 1)}
    single_node = min(peers.values()) == 0

    procs = []
    try:
        for gpu in gpus:
            node = GPU_NODE[gpu]
            rank = node_ranks.get(node, 0)
            node_ranks[node] = rank + 1
            env = build_env(gpu, peers[node], rank, single_node, compression,
                            backends, iaa_total, cache_root, extra)
            cmd = [
                "numactl", f"--cpunodebind={node}", f"--membind={node}",
                sys.executable, BENCH,
                "--kv-cache-shape", "2", str(cache_blocks), str(block_tokens),
                str(KV_HEADS), str(HEAD_DIM),
                "--num-layers", str(NUM_LAYERS),
                "--sync-dir", sync_dir, "--sync-count", str(len(gpus)),
                "--metrics-url",
                f"http://127.0.0.1:{worker_port(gpu)}/v1/cache/metrics",
            ]
            log = open(os.path.join(cache_root, f"gpu{gpu}.log"), "w+")
            procs.append((gpu, subprocess.Popen(
                cmd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True), log))

        # A crashed peer would leave the survivors blocked in the barrier, so bail
        # out as soon as any process exits non-zero.
        deadline = time.perf_counter() + timeout
        while any(p.poll() is None for _, p, _ in procs):
            if any(p.returncode not in (None, 0) for _, p, _ in procs):
                break
            if time.perf_counter() > deadline:
                break
            time.sleep(0.05)
        for _, p, _ in procs:
            if p.poll() is None:
                p.kill()
                p.wait()

        out = []
        for gpu, p, log in procs:
            log.seek(0)
            so = log.read()
            log.close()
            res = {k: (float(m.group(1)) if (m := re.search(v, so, re.M)) else None)
                   for k, v in PAT.items()}
            res.update(gpu=gpu, rc=p.returncode,
                       verified="Verification: passed" in so)
            if p.returncode != 0 or res["put_s"] is None:
                res["error"] = " | ".join(so.strip().splitlines()[-5:])
            out.append(res)
        return out
    finally:
        for _, p, _ in procs:
            if p.poll() is None:
                p.kill()
        shutil.rmtree(cache_root, ignore_errors=True)


HDR = (f"{'GPUs':>5} {'payload MiB':>12} "
       f"{'PUT/GPU':>9} {'PUT total':>10} {'GET/GPU':>9} {'GET total':>10} "
       f"{'zip/GPU':>9} {'unzip/GPU':>10} {'ratio':>6} {'ok':>3}")


def summarize(n: int, total_mib: int, rows: list) -> str:
    bad = [r for r in rows if r.get("error")]
    if bad:
        return f"{n:>5} {total_mib * n:>12}   FAILED: {bad[0]['error'][:100]}"

    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    # Aggregate = n x payload divided by the slowest process, since the barrier
    # makes every process time the same wall-clock window.
    put_tot = (total_mib * n / 1024) / max(r["put_s"] for r in rows)
    get_tot = (total_mib * n / 1024) / max(r["get_s"] for r in rows)
    ok = all(r["verified"] for r in rows)
    return (f"{n:>5} {total_mib * n:>12} "
            f"{mean('put_gibs'):>9.2f} {put_tot:>10.2f} "
            f"{mean('get_gibs'):>9.2f} {get_tot:>10.2f} "
            f"{mean('comp_gbps'):>9.2f} {mean('decomp_gbps'):>10.2f} "
            f"{mean('ratio'):>6.3f} {'Y' if ok else 'N':>3}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--block-tokens", type=int, default=16)
    ap.add_argument("--total-mib", type=int, default=512, help="Payload per GPU.")
    ap.add_argument("--compression", type=int, choices=(0, 1), default=1)
    ap.add_argument("--backends", choices=("qat", "iaa", "both"), default="both",
                    help="Which zip backends to enable (ignored when --compression 0).")
    ap.add_argument("--iaa-total", type=int, default=32,
                    help="Total IAA instances across all processes.")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    extra = dict(item.partition("=")[::2] for item in args.env)

    print("=" * len(HDR))
    print(f"block_tokens={args.block_tokens} "
          f"({BYTES_PER_TOKEN * args.block_tokens // 1024} KiB/block), "
          f"payload={args.total_mib} MiB/GPU, compression={args.compression}, "
          f"backends={args.backends}, iaa_total={args.iaa_total}, env={extra or '{}'}")
    print(HDR)
    print("-" * len(HDR))

    all_rows = []
    for n in args.gpus:
        gpus = GPU_SETS.get(n) or list(range(n))
        for _ in range(args.reps):
            rows = run_group(gpus, args.block_tokens, args.total_mib,
                             bool(args.compression), args.backends,
                             args.iaa_total, extra, args.timeout)
            print(summarize(n, args.total_mib, rows), flush=True)
            all_rows.append({"gpus": gpus, "rows": rows,
                             "compression": args.compression,
                             "backends": args.backends,
                             "iaa_total": args.iaa_total})

    if args.json_out:
        with open(args.json_out, "a") as fp:
            for r in all_rows:
                fp.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
