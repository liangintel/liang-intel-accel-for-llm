#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
import os
import time
import urllib.request
from typing import List, Tuple

import torch

from iaxl.utils.logger import setup_root_logger

setup_root_logger(show_pid_tid=False)


_env_parser = argparse.ArgumentParser(add_help=False)
_env_parser.add_argument(
    "--quant", type=str, default=os.environ.get("IAXL_KV_LOSSY_TRUNC", "0")
)
_env_parser.add_argument(
    "--data-shuffle",
    type=int,
    choices=(0, 1),
    default=int(os.environ.get("IAXL_KV_DATA_SHUFFLE", "0")),
)
_env_args, _ = _env_parser.parse_known_args()
os.environ["IAXL_KV_LOSSY_TRUNC"] = str(_env_args.quant)
os.environ["IAXL_KV_DATA_SHUFFLE"] = str(_env_args.data_shuffle)

from iaxl.kvstore import KVStore, get_accelerator_device

DEFAULT_KV_CACHE_SHAPE = (2, 1024, 16, 4, 128)


DEFAULT_METRICS_URL = "http://127.0.0.1:18800/v1/cache/metrics"
SEED = 42
BLOCK_DIM = 1

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp8_e4m3": torch.float8_e4m3fn,
}


def add_env_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--quant",
        type=str,
        default=os.environ.get("IAXL_KV_LOSSY_TRUNC", "0"),
        help="Quantization: 'auto', 0 (off), or N LSB bits to truncate "
        "(env IAXL_KV_LOSSY_TRUNC).",
    )
    parser.add_argument(
        "--data-shuffle",
        type=int,
        choices=(0, 1),
        default=int(os.environ.get("IAXL_KV_DATA_SHUFFLE", "0")),
        help="Enable data shuffle before compression (env IAXL_KV_DATA_SHUFFLE).",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark KVStore PUT/GET with generated or file-backed data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--kv-cache-shape",
        "--shape",
        type=int,
        nargs=5,
        default=DEFAULT_KV_CACHE_SHAPE,
        metavar=("KV", "CACHE_BLOCKS", "TOKENS", "KV_HEADS", "HEAD_DIM"),
        help="Shape of one KV cache layer. KV_HEADS is 8 for TP=1, 1 for TP=8.",
    )
    parser.add_argument(
        "--sync-dir",
        default=None,
        help="Rendezvous directory shared by concurrent benchmark processes, so "
        "their timed PUT and GET phases overlap instead of drifting apart.",
    )
    parser.add_argument(
        "--sync-count", type=int, default=1,
        help="Number of processes meeting at --sync-dir.",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPE_MAP),
        default="bf16",
        help="KV cache tensor dtype.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=32,
        help="Number of KV cache layers to allocate, mimicking a real model. "
        "PUT/GET and their waits are issued layer by layer like vLLM. "
        "Each layer holds a full --kv-cache-shape tensor, so GPU memory "
        "scales with this value.",
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Optional test-data file whose bytes fill the KV cache (cycled "
        "to fit). If omitted, deterministic mock data is generated.",
    )
    parser.add_argument(
        "--metrics-url",
        default=DEFAULT_METRICS_URL,
        help="KVStore REST endpoint for compression throughput metrics.",
    )
    add_env_arguments(parser)
    args = parser.parse_args()
    shape = tuple(args.kv_cache_shape)
    if shape[0] != 2:
        parser.error("the first --kv-cache-shape dimension must be 2 (key/value)")
    if any(dim <= 0 for dim in shape):
        parser.error("all --kv-cache-shape dimensions must be greater than 0")
    if args.num_layers <= 0:
        parser.error("--num-layers must be greater than 0")
    args.kv_cache_shape = shape
    return args


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def make_block_hashes(num_blocks: int) -> List[str]:
    return [str(i) for i in range(num_blocks)]


def generate_mock_kv_cache(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    kv_count, cache_blocks, block_tokens, kv_heads, head_dim = shape
    latent_dim = min(32, max(8, head_dim // 8))
    generator = torch.Generator(device="cuda").manual_seed(SEED)

    block_context = torch.randn(
        (cache_blocks, 1, kv_heads, latent_dim),
        device="cuda",
        generator=generator,
    )
    token_innovations = torch.randn(
        (cache_blocks, block_tokens, kv_heads, latent_dim),
        device="cuda",
        generator=generator,
    )
    block_drift = torch.randn(
        (cache_blocks, 1, kv_heads, latent_dim),
        device="cuda",
        generator=generator,
    )
    token_positions = torch.linspace(-1.0, 1.0, block_tokens, device="cuda").view(
        1, block_tokens, 1, 1
    )
    hidden_states = (
        0.65 * block_context
        + 0.70 * token_innovations
        + 0.15 * token_positions * block_drift
    )

    projections = torch.randn(
        (kv_count, kv_heads, latent_dim, head_dim),
        device="cuda",
        generator=generator,
    ) / math.sqrt(latent_dim)
    kv_cache = torch.einsum("bthr,khrd->kbthd", hidden_states, projections)

    channel_scales = (
        torch.randn(
            (kv_count, 1, 1, kv_heads, head_dim),
            device="cuda",
            generator=generator,
        )
        .mul_(0.25)
        .exp_()
    )
    kv_cache.mul_(channel_scales)
    return kv_cache.to(dtype).contiguous()


def load_kv_cache_from_file(
    path: str, shape: Tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    with open(path, "rb") as fp:
        data = fp.read()
    if not data:
        raise ValueError(f"test-data file {path!r} is empty")

    total_bytes = math.prod(shape) * dtype.itemsize
    repeats = -(-total_bytes // len(data))
    raw = bytearray(data) * repeats
    del raw[total_bytes:]

    flat = torch.frombuffer(raw, dtype=torch.uint8).view(dtype)
    print(f"[data] loaded {path} ({len(data)} bytes, cycled to fill cache)")
    return flat.reshape(shape).to("cuda")


def print_result(name: str, elapsed: float, num_blocks: int, num_bytes: int) -> None:
    print(
        f"{name:>4}: {elapsed:.3f} s, "
        f"{num_blocks / elapsed:,.2f} blocks/s, "
        f"{num_bytes / elapsed / (1024**3):.3f} GiB/s"
    )


def _rest_get_json(url: str, query: str = "") -> dict:
    if query:
        url = f"{url}?{query}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def cache_metrics_rest(url: str, query: str = "") -> dict:
    return _rest_get_json(url, query)


def cache_status_rest(metrics_url: str) -> dict:
    status_url = metrics_url.rsplit("/", 1)[0] + "/status"
    return _rest_get_json(status_url)


def barrier(sync_dir: str, count: int, tag: str, timeout: float = 600.0) -> None:
    if not sync_dir or count <= 1:
        return
    slot = os.path.join(sync_dir, tag)
    os.makedirs(slot, exist_ok=True)
    open(os.path.join(slot, str(os.getpid())), "w").close()
    deadline = time.perf_counter() + timeout
    while len(os.listdir(slot)) < count:
        if time.perf_counter() > deadline:
            raise RuntimeError(f"barrier '{tag}' timed out waiting for {count} processes")
        time.sleep(0.002)


def run_benchmark(args: argparse.Namespace) -> bool:
    if get_accelerator_device() != "cuda":
        print("No CUDA device available; KVStore benchmark cannot run.")
        return False

    shape: Tuple[int, ...] = args.kv_cache_shape
    dtype = DTYPE_MAP[args.dtype]
    num_layers = args.num_layers
    cache_blocks = shape[BLOCK_DIM]

    block_indices = list(range(0, cache_blocks, 2))
    num_blocks = len(block_indices)
    bytes_per_block = math.prod(shape) // cache_blocks * dtype.itemsize

    total_blocks = num_blocks * num_layers
    total_bytes = bytes_per_block * num_blocks * num_layers

    layer_names = [str(i) for i in range(num_layers)]

    print("=" * 80)
    print("KVStore Benchmark")
    print("=" * 80)
    print(f"KV cache shape: {list(shape)} (one layer)")
    print(f"Layers:         {num_layers}")
    print(f"Blocks tested:  {num_blocks} (even indices, strided) x {num_layers} layers")
    print(f"Dtype:          {dtype}")
    print(f"Transfer size:  {format_bytes(total_bytes)}")

    if args.data_file is None or args.data_file == "random":
        print("\nGenerating mock KV cache data...")
        base = generate_mock_kv_cache(shape, dtype)
    else:
        print(f"\nLoading KV cache from {args.data_file}...")
        base = load_kv_cache_from_file(args.data_file, shape, dtype)

    kv_caches = {name: base.clone() for name in layer_names}
    del base

    index = torch.tensor(
        [block_indices[0], block_indices[num_blocks // 2], block_indices[-1]],
        device="cuda",
    )
    expected = kv_caches[layer_names[0]].index_select(BLOCK_DIM, index).clone()

    block_hashes = make_block_hashes(num_blocks)

    kvstore = KVStore(
        model_name="kvstore_benchmark",
        kv_caches=kv_caches,
        block_dim=BLOCK_DIM,
    )

    cache_metrics_rest(args.metrics_url, "enable=1&reset=1")

    print("\nWarming up...")
    warmup_hashes = [f"warmup_{h}" for h in block_hashes]
    warmup_put = kvstore.put(block_indices, warmup_hashes, description="warmup PUT")
    if not kvstore.put_wait(warmup_put):
        raise RuntimeError("warm-up PUT did not complete")
    warmup_get = kvstore.get(block_indices, warmup_hashes, description="warmup GET")
    if not kvstore.get_wait(warmup_get):
        raise RuntimeError("warm-up GET did not complete")
    torch.cuda.synchronize()

    cache_metrics_rest(args.metrics_url, "reset=1")

    torch.cuda.synchronize()
    barrier(args.sync_dir, args.sync_count, "put")
    start = time.perf_counter()
    with torch.cuda.nvtx.range("benchmark PUT"):
        put_tasks = {}
        for name in layer_names:
            put_tasks.update(
                kvstore.put(
                    block_indices,
                    block_hashes,
                    layer_names=[name],
                    description="benchmark PUT",
                )
            )
        if not kvstore.put_wait(put_tasks):
            raise RuntimeError("PUT did not complete")
    put_time = time.perf_counter() - start

    for tensor in kv_caches.values():
        tensor.zero_()
    torch.cuda.synchronize()
    barrier(args.sync_dir, args.sync_count, "get")

    start = time.perf_counter()
    with torch.cuda.nvtx.range("benchmark GET"):
        get_tasks = kvstore.get(
            block_indices, block_hashes, layer_names=None, description="benchmark GET"
        )
        for name in layer_names:
            if not kvstore.get_wait(get_tasks, layer_names=[name]):
                raise RuntimeError("GET did not complete")
    get_time = time.perf_counter() - start

    verified = all(
        torch.equal(kv_caches[name].index_select(BLOCK_DIM, index), expected)
        for name in layer_names
    )

    print("\nResults")
    print("-" * 80)
    print_result("PUT", put_time, total_blocks, total_bytes)
    print_result("GET", get_time, total_blocks, total_bytes)
    print(f"Verification: {'passed' if verified else 'FAILED'}")

    status = cache_status_rest(args.metrics_url)
    print(
        f"Compression ratio: {status['compression_ratio (unzip/zip, higher=better)']:.3f}x"
    )
    print(
        f"Native cache size: {format_bytes(status['current_bytes'])} "
        f"({status['cache_entries']} chunks in {status['group_count']} groups)"
    )
    print(
        f"Cache hits/misses/puts: {status['hits']}/{status['misses']}/{status['puts']}"
    )
    print(
        f"Compressed/uncompressed bytes: {format_bytes(status['total_zip_bytes'])} / "
        f"{format_bytes(status['total_unzip_bytes'])}"
    )

    metrics = cache_metrics_rest(args.metrics_url)
    print(
        f"Compress throughput:   {metrics['compress_gbps']:.3f} GB/s "
        f"({format_bytes(metrics['compress_bytes'])} in {metrics['compress_ns'] / 1e6:.1f} ms)"
    )
    print(
        f"Decompress throughput: {metrics['decompress_gbps']:.3f} GB/s "
        f"({format_bytes(metrics['decompress_bytes'])} in {metrics['decompress_ns'] / 1e6:.1f} ms)"
    )
    return verified


def main() -> int:
    args = parse_args()
    try:
        return 0 if run_benchmark(args) else 1
    except Exception as error:
        print(f"\nBenchmark FAILED: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
