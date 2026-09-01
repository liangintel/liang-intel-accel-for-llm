#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end TTFT sweep: N vLLM servers (one per GPU) x compression backend configs.

Each GPU runs an independent tp=1 vLLM server behind the KVShrink connector, so the
GPU count sweep is 1..8 (a tensor-parallel sweep could only do 1/2/4/8, since vLLM
requires the KV head count to be divisible by tp). Accelerators are partitioned the
same way multigpu_scale.py does it, which is why build_env is imported from there
rather than duplicated.

Every request reuses one long shared prefix, so after the warmup the prefill is served
from the offloaded KV cache and TTFT is dominated by the decompress + host-to-device
path being measured.

Run as root: DSA and IAA portals need it.
"""

import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)

from multigpu_scale import GPU_NODE, GPU_SETS, build_env  # noqa: E402

VENV = "/home/pese/miniforge3/envs/kvcache"
CUDA_HOME = os.path.join(VENV, "lib/python3.12/site-packages/nvidia/cu13")

CONFIGS = [
    ("no-zip", dict(compression=False, backends="qat", iaa_total=32)),
    ("qat8", dict(compression=True, backends="qat", iaa_total=32)),
    ("iaa16", dict(compression=True, backends="iaa", iaa_total=16)),
    ("iaa32", dict(compression=True, backends="iaa", iaa_total=32)),
    ("qat+iaa16", dict(compression=True, backends="both", iaa_total=16)),
    ("qat+iaa32", dict(compression=True, backends="both", iaa_total=32)),
]

TTFT_PAT = {
    "ttft_mean_ms": r"Mean TTFT \(ms\):\s+([\d.]+)",
    "ttft_p50_ms": r"P50 TTFT \(ms\):\s+([\d.]+)",
    "ttft_p90_ms": r"P90 TTFT \(ms\):\s+([\d.]+)",
    "ttft_p95_ms": r"P95 TTFT \(ms\):\s+([\d.]+)",
    "ttft_p99_ms": r"P99 TTFT \(ms\):\s+([\d.]+)",
    "tpot_mean_ms": r"Mean TPOT \(ms\):\s+([\d.]+)",
    "tpot_p99_ms": r"P99 TPOT \(ms\):\s+([\d.]+)",
    "itl_mean_ms": r"Mean ITL \(ms\):\s+([\d.]+)",
    "itl_p99_ms": r"P99 ITL \(ms\):\s+([\d.]+)",
    "e2el_mean_ms": r"Mean E2EL \(ms\):\s+([\d.]+)",
    "e2el_p50_ms": r"P50 E2EL \(ms\):\s+([\d.]+)",
    "e2el_p99_ms": r"P99 E2EL \(ms\):\s+([\d.]+)",
    "req_throughput": r"Request throughput \(req/s\):\s+([\d.]+)",
    "successful": r"Successful requests:\s+(\d+)",
}


def node_cpus(node: int) -> list:
    with open(f"/sys/devices/system/node/node{node}/cpulist") as fp:
        spec = fp.read().strip()
    cpus = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-"))
            cpus.extend(range(lo, hi + 1))
        else:
            cpus.append(int(part))
    return cpus


def cpu_slice(node: int, peers: int, rank: int) -> str:
    """Contiguous share of the node's CPUs, as a vLLM VLLM_CPU_OMP_THREADS_BIND spec."""
    cpus = node_cpus(node)
    per = len(cpus) // peers
    mine = cpus[rank * per:(rank + 1) * per]
    return ",".join(str(c) for c in mine)


def server_env(gpu: int, peers: int, rank: int, single_node: bool, cfg: dict,
               cache_root: str) -> dict:
    env = build_env(gpu, peers, rank, single_node, cfg["compression"],
                    cfg["backends"], cfg["iaa_total"], cache_root, {})
    env.update({
        "CUDA_HOME": CUDA_HOME,
        # System gcc is 12 but only g++-11 ships cc1plus, so nvcc needs pointing at it.
        "NVCC_PREPEND_FLAGS": "-ccbin /usr/bin/g++-11",
        "PATH": f"{CUDA_HOME}/bin:{VENV}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LD_LIBRARY_PATH": f"{VENV}/lib",
        "PYTHONPATH": REPO_ROOT,
        "HF_HUB_OFFLINE": "1",
        # build_env only separates the worker port, so the KVStore mgmt server would
        # otherwise collide across the per-GPU servers.
        "IAXL_API_CONTROLLER_PORT": str(18700 + gpu),
        # NHD makes vLLM hand the connector a permuted (non-contiguous) view; HND is
        # the identity stride order, which is what the DSA/zip path requires.
        "VLLM_KV_CACHE_LAYOUT": "HND",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "VLLM_CPU_OMP_THREADS_BIND": cpu_slice(GPU_NODE[gpu], peers, rank),
        "VLLM_LOGGING_LEVEL": "WARNING",
    })
    return env


def wait_ready(ports: list, procs: list, timeout: int) -> bool:
    import urllib.error
    import urllib.request
    deadline = time.time() + timeout
    pending = set(ports)
    while pending and time.time() < deadline:
        if any(p.poll() not in (None, 0) for p in procs):
            return False
        for port in list(pending):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        pending.discard(port)
            except (urllib.error.URLError, OSError):
                pass
        if pending:
            time.sleep(5)
    return not pending


def wait_gpu_idle(timeout: int = 180) -> bool:
    """Block until no compute process holds GPU memory, so a group never starts dirty."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                              "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        if not out:
            return True
        time.sleep(5)
    print(f"WARNING: GPUs still busy after {timeout}s: {out.splitlines()}", flush=True)
    return False


def run_bench(port: int, model: str, args, log_path: str) -> dict:
    prefix_len = args.input_len * args.hit_rate // 100
    cmd = [
        os.path.join(VENV, "bin", "vllm"), "bench", "serve",
        "--backend", "vllm", "--model", model, "--tokenizer", model,
        "--host", "127.0.0.1", "--port", str(port),
        "--dataset-name", "random",
        "--random-input-len", str(args.input_len - prefix_len),
        "--random-prefix-len", str(prefix_len),
        "--random-output-len", str(args.output_len),
        "--ignore-eos", "--trust-remote-code", "--request-rate", "inf",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,90,95,99",
        "--seed", "1234",
        "--num-warmups", str(args.warmups),
        "--num-prompts", str(args.prompts),
        "--max-concurrency", str(args.concurrency),
        # per-request latencies, so any percentile can be recomputed without re-running
        "--save-result", "--save-detailed",
        "--result-filename", f"/tmp/ttft_raw_p{port}.json",
    ]
    env = os.environ.copy()
    env.update(PATH=f"{VENV}/bin:" + env.get("PATH", ""), HF_HUB_OFFLINE="1",
               LD_LIBRARY_PATH=f"{VENV}/lib")
    with open(log_path, "w+") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                             timeout=args.bench_timeout)
        log.seek(0)
        out = log.read()
    res = {k: (float(m.group(1)) if (m := re.search(v, out)) else None)
           for k, v in TTFT_PAT.items()}
    res["port"] = port
    res["rc"] = rc
    if not res["successful"]:
        res["error"] = " | ".join(out.strip().splitlines()[-4:])
    return res


def run_group(gpus: list, cfg: dict, args) -> list:
    cache_root = "/tmp/ttft_cache"
    shutil.rmtree(cache_root, ignore_errors=True)
    os.makedirs(cache_root, exist_ok=True)
    if not wait_gpu_idle():
        return [{"error": "GPUs still occupied by a previous run"}]

    peers = {n: sum(1 for g in gpus if GPU_NODE[g] == n) for n in (0, 1)}
    single_node = min(peers.values()) == 0
    ranks = {}

    procs, logs, ports = [], [], []
    try:
        for gpu in gpus:
            node = GPU_NODE[gpu]
            rank = ranks.get(node, 0)
            ranks[node] = rank + 1
            env = server_env(gpu, peers[node], rank, single_node, cfg, cache_root)
            port = 8000 + gpu
            cmd = [
                os.path.join(VENV, "bin", "vllm"), "serve", args.model,
                "--port", str(port), "--trust-remote-code",
                "--kv-transfer-config", json.dumps({
                    "kv_connector": "KVShrinkConnector",
                    "kv_connector_module_path": "kvshrink.kvshrink_connector",
                    "kv_role": "kv_both"}),
                "--gpu-memory-utilization", str(args.gpu_mem_util),
                "-tp", "1", "--max-model-len", args.max_model_len,
                "--no-enable-prefix-caching",
            ]
            log = open(f"/tmp/ttft_server_gpu{gpu}.log", "w")
            procs.append(subprocess.Popen(cmd, env=env, stdout=log,
                                          stderr=subprocess.STDOUT, cwd=REPO_ROOT,
                                          start_new_session=True))
            logs.append(log)
            ports.append(port)

        if not wait_ready(ports, procs, args.startup_timeout):
            return [{"error": "server startup failed/timed out (see /tmp/ttft_server_gpu*.log)"}]

        with ThreadPoolExecutor(max_workers=len(ports)) as pool:
            futs = [pool.submit(run_bench, p, args.model, args,
                                f"/tmp/ttft_bench_p{p}.log") for p in ports]
            return [f.result() for f in futs]
    finally:
        pgids = []
        for p in procs:
            with contextlib.suppress(ProcessLookupError):
                pgids.append(os.getpgid(p.pid))
        for pgid in pgids:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGTERM)
        for p in procs:
            with contextlib.suppress(subprocess.TimeoutExpired):
                p.wait(timeout=90)
        for pgid in pgids:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
        for log in logs:
            log.close()
        # EngineCore gets reparented to init when the API server dies, so a plain
        # terminate() leaves it alive holding the whole GPU allocation.
        subprocess.call(["pkill", "-9", "-f", "VLLM::EngineCore"])
        subprocess.call(["pkill", "-9", "-f", "vllm serve"])
        shutil.rmtree(cache_root, ignore_errors=True)
        wait_gpu_idle()


HDR = (f"{'config':>10} {'GPUs':>5} {'TTFT mean':>10} {'TTFT p50':>9} {'TTFT p90':>9} "
       f"{'TTFT p95':>9} {'TTFT p99':>9} {'TTFT worst':>11} {'TPOT mean':>10} "
       f"{'ITL p99':>8} {'E2EL p50':>9} {'E2EL p99':>9} {'req/s tot':>10} {'ok':>3}")


def summarize(name: str, n: int, rows: list) -> str:
    bad = [r for r in rows if r.get("error")]
    if bad:
        return f"{name:>10} {n:>5}   FAILED: {bad[0]['error'][:90]}"

    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    worst = max(r["ttft_mean_ms"] for r in rows)
    rps = sum(r["req_throughput"] for r in rows if r.get("req_throughput") is not None)
    return (f"{name:>10} {n:>5} {mean('ttft_mean_ms'):>10.1f} {mean('ttft_p50_ms'):>9.1f} "
            f"{mean('ttft_p90_ms'):>9.1f} {mean('ttft_p95_ms'):>9.1f} "
            f"{mean('ttft_p99_ms'):>9.1f} {worst:>11.1f} {mean('tpot_mean_ms'):>10.2f} "
            f"{mean('itl_p99_ms'):>8.2f} {mean('e2el_p50_ms'):>9.1f} "
            f"{mean('e2el_p99_ms'):>9.1f} {rps:>10.2f} {'Y':>3}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/pese/qiuyu/Qwen2.5-7B-Instruct")
    ap.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--configs", nargs="+", default=[c[0] for c in CONFIGS])
    ap.add_argument("--input-len", type=int, default=8000)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--hit-rate", type=int, default=80)
    ap.add_argument("--prompts", type=int, default=20)
    ap.add_argument("--warmups", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-model-len", default="18k")
    ap.add_argument("--gpu-mem-util", type=float, default=0.8)
    ap.add_argument("--startup-timeout", type=int, default=1500)
    ap.add_argument("--bench-timeout", type=int, default=1800)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    selected = [(n, c) for n, c in CONFIGS if n in args.configs]
    print(f"model={args.model} input_len={args.input_len} (prefix {args.hit_rate}%) "
          f"output_len={args.output_len} prompts={args.prompts} "
          f"concurrency={args.concurrency}/server warmups={args.warmups}")
    print(HDR)
    print("-" * len(HDR))

    for name, cfg in selected:
        for n in args.gpus:
            gpus = GPU_SETS.get(n) or list(range(n))
            rows = run_group(gpus, cfg, args)
            print(summarize(name, n, rows), flush=True)
            if args.json_out:
                with open(args.json_out, "a") as fp:
                    fp.write(json.dumps({"config": name, "gpus": gpus,
                                         "cfg": cfg, "rows": rows}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
