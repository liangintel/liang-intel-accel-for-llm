#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""KVStore PUT/GET sweep: N GPUs x compression backend configs.

One benchmark process per GPU, all rendezvousing before every timed iteration so
the phases overlap and the accelerators are contended the way they would be in a
real one-rank-per-GPU deployment. Accelerators are partitioned by multigpu_scale's
build_env, which is why it is imported rather than duplicated.

Unlike multigpu_scale.py this repeats the timed PUT and GET --iters times and keeps
every per-iteration latency, which is what makes P50/P95/P99 available. Each
iteration writes its own set of block hashes, so a PUT never overwrites an earlier
one and every GET is a cache hit.

Run as root: DSA and IAA portals need it.
"""

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)
from multigpu_scale import (  # noqa: E402
    BENCH, BYTES_PER_TOKEN, GPU_NODE, GPU_SETS, HEAD_DIM, KV_HEADS, NUM_LAYERS,
    build_env, worker_port,
)

VENV = "/home/pese/miniforge3/envs/kvcache"

CONFIGS = [
    ("no-zip",    dict(compression=False, backends="qat",  iaa_total=32)),
    ("qat8",      dict(compression=True,  backends="qat",  iaa_total=32)),
    ("iaa16",     dict(compression=True,  backends="iaa",  iaa_total=16)),
    ("iaa32",     dict(compression=True,  backends="iaa",  iaa_total=32)),
    ("qat+iaa16", dict(compression=True,  backends="both", iaa_total=16)),
    ("qat+iaa32", dict(compression=True,  backends="both", iaa_total=32)),
]


def pctl(samples: list, q: float) -> float:
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def phase_summary(phase: dict, bytes_per_iter: int) -> dict:
    t = phase["times_s"]
    mean = sum(t) / len(t)
    return {
        "n": len(t),
        "mean_ms": mean * 1e3,
        "p50_ms": pctl(t, 50) * 1e3,
        "p90_ms": pctl(t, 90) * 1e3,
        "p95_ms": pctl(t, 95) * 1e3,
        "p99_ms": pctl(t, 99) * 1e3,
        "min_ms": min(t) * 1e3,
        "max_ms": max(t) * 1e3,
        "gibs_mean": bytes_per_iter / mean / 1024**3,
        "cpu_cores": phase["cpu_cores"],
        "host_cpu_cores": phase["host_cpu_cores"],
        "cpu_ms_per_op": phase["cpu_core_s"] / len(t) * 1e3,
        "times_s": t,
    }


def run_group(gpus: list, cfg: dict, args: argparse.Namespace) -> list:
    """Launch one process per GPU and collect their per-iteration latencies."""
    bytes_per_block = BYTES_PER_TOKEN * args.block_tokens
    cache_blocks = 2 * args.payload_mib * 1024 * 1024 // (bytes_per_block * NUM_LAYERS)

    cache_root = tempfile.mkdtemp(prefix="kvsweep_")
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
            # An explicit pool bounds host memory: the default is a tenth of free
            # RAM per process, which 8 processes would badly oversubscribe. It must
            # still exceed iters x payload or entries get evicted and GET misses.
            extra = {"IAXL_DDR_POOL_SIZE_GB": args.pool_gb,
                     "IAXL_LOG_LEVEL": "WARNING",
                     "PYTHONPATH": REPO_ROOT,
                     "LD_LIBRARY_PATH": f"{VENV}/lib"}
            env = build_env(gpu, peers[node], rank, single_node, cfg["compression"],
                            cfg["backends"], cfg["iaa_total"], cache_root, extra)
            raw = os.path.join(cache_root, f"gpu{gpu}.json")
            cmd = [
                "numactl", f"--cpunodebind={node}", f"--membind={node}",
                os.path.join(VENV, "bin", "python3"), BENCH,
                "--kv-cache-shape", "2", str(cache_blocks), str(args.block_tokens),
                str(KV_HEADS), str(HEAD_DIM),
                "--num-layers", str(NUM_LAYERS),
                "--iters", str(args.iters),
                "--sync-dir", sync_dir, "--sync-count", str(len(gpus)),
                "--json-out", raw,
                "--metrics-url",
                f"http://127.0.0.1:{worker_port(gpu)}/v1/cache/metrics",
            ]
            log = open(f"/tmp/kvs_gpu{gpu}.log", "w+")
            procs.append((gpu, subprocess.Popen(
                cmd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True,
                cwd=REPO_ROOT, start_new_session=True), log, raw))

        # A crashed peer leaves the survivors stuck in the barrier, so stop as soon
        # as any process exits non-zero rather than waiting out the timeout.
        deadline = time.perf_counter() + args.timeout
        while any(p.poll() is None for _, p, _, _ in procs):
            if any(p.returncode not in (None, 0) for _, p, _, _ in procs):
                break
            if time.perf_counter() > deadline:
                break
            time.sleep(0.1)
        for _, p, _, _ in procs:
            if p.poll() is None:
                p.kill()
                p.wait()

        rows = []
        for gpu, p, log, raw in procs:
            log.seek(0)
            text = log.read()
            log.close()
            row = {"gpu": gpu, "rc": p.returncode}
            if p.returncode != 0 or not os.path.exists(raw):
                tail = " | ".join(text.strip().splitlines()[-5:])
                row["error"] = tail or f"exited {p.returncode} with no output"
                rows.append(row)
                continue
            with open(raw) as fp:
                data = json.load(fp)
            per_iter = data["bytes_per_iter"]
            row.update(
                put=phase_summary(data["put"], per_iter),
                get=phase_summary(data["get"], per_iter),
                bytes_per_iter=per_iter,
                verified=data["verified"],
                compression_ratio=data["compression_ratio"],
                misses=data["misses"],
                compress_gbps=data["compress_gbps"],
                decompress_gbps=data["decompress_gbps"],
            )
            if not data["verified"]:
                row["error"] = "verification FAILED"
            if data["misses"]:
                row["error"] = f"{data['misses']} cache misses (pool too small?)"
            rows.append(row)
        return rows
    finally:
        for _, p, _, _ in procs:
            if p.poll() is None:
                p.kill()
        shutil.rmtree(cache_root, ignore_errors=True)


HDR = (f"{'config':>10} {'GPUs':>5} {'PUT mean':>9} {'PUT p50':>8} {'PUT p95':>8} "
       f"{'PUT p99':>8} {'GET mean':>9} {'GET p50':>8} {'GET p95':>8} {'GET p99':>8} "
       f"{'PUT GiB/s':>10} {'GET GiB/s':>10} {'CPU cores':>10} {'ratio':>6} {'ok':>3}")


def summarize(name: str, n: int, rows: list) -> str:
    bad = [r for r in rows if r.get("error")]
    if bad:
        return f"{name:>10} {n:>5}   FAILED: {bad[0]['error'][:100]}"

    def mean(phase, key):
        return sum(r[phase][key] for r in rows) / len(rows)

    def total(phase, key):
        return sum(r[phase][key] for r in rows)

    return (f"{name:>10} {n:>5} "
            f"{mean('put', 'mean_ms'):>9.2f} {mean('put', 'p50_ms'):>8.2f} "
            f"{mean('put', 'p95_ms'):>8.2f} {mean('put', 'p99_ms'):>8.2f} "
            f"{mean('get', 'mean_ms'):>9.2f} {mean('get', 'p50_ms'):>8.2f} "
            f"{mean('get', 'p95_ms'):>8.2f} {mean('get', 'p99_ms'):>8.2f} "
            f"{total('put', 'gibs_mean'):>10.2f} {total('get', 'gibs_mean'):>10.2f} "
            f"{total('put', 'cpu_cores'):>10.1f} "
            f"{sum(r['compression_ratio'] for r in rows) / len(rows):>6.3f} {'Y':>3}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--configs", nargs="+", default=[c[0] for c in CONFIGS])
    ap.add_argument("--block-tokens", type=int, default=16)
    ap.add_argument("--payload-mib", type=int, default=256,
                    help="KV bytes moved by one PUT or GET, per GPU.")
    ap.add_argument("--iters", type=int, default=60,
                    help="Timed repetitions per phase; also the percentile sample count.")
    ap.add_argument("--pool-gb", type=float, default=24.0,
                    help="Host KV pool per process; must exceed iters x payload.")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    need_gb = args.iters * args.payload_mib / 1024
    if args.pool_gb < need_gb * 1.2:
        print(f"warning: --pool-gb {args.pool_gb} is tight for "
              f"{need_gb:.1f} GiB of writes; expect evictions and GET misses",
              file=sys.stderr)
    # The children mlock DMA buffers, and the default soft limit is far below what
    # 8 processes need; without this iaa_zip silently falls back to unpinned memory.
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft != resource.RLIM_INFINITY:
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))

    by_name = dict(CONFIGS)
    print("=" * len(HDR))
    print(f"payload={args.payload_mib} MiB/GPU/op, iters={args.iters}, "
          f"block_tokens={args.block_tokens}, pool={args.pool_gb} GiB/process")
    print(HDR)
    print("-" * len(HDR))

    for name in args.configs:
        cfg = by_name[name]
        for n in args.gpus:
            gpus = GPU_SETS.get(n) or list(range(n))
            rows = run_group(gpus, cfg, args)
            print(summarize(name, n, rows), flush=True)
            if args.json_out:
                # Appended per group so a crash only loses the group in flight.
                with open(args.json_out, "a") as fp:
                    fp.write(json.dumps({"config": name, "gpus": gpus, "cfg": cfg,
                                         "payload_mib": args.payload_mib,
                                         "iters": args.iters, "rows": rows}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
