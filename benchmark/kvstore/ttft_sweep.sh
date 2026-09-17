#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Reproduces the end-to-end TTFT sweep: 6 compression backend configurations x
# 1..8 GPUs, reporting TTFT P50/P95/P99 and CPU consumption.
#
#   ./ttft_sweep.sh check     preflight: hardware, accelerators, python env
#   ./ttft_sweep.sh build     rebuild the iaxl torch extension
#   ./ttft_sweep.sh run       run the 48-group sweep (auto-resumes failed groups)
#   ./ttft_sweep.sh verify    post-run sanity checks
#   ./ttft_sweep.sh report    print the result tables and the CPU regression
#   ./ttft_sweep.sh html      write a standalone HTML report
#   ./ttft_sweep.sh all       check + build + run + verify + report + html
#
# Must run as root: the DSA and IAA portals are mmap'd, which needs privileges.
# Every tunable below can be overridden from the environment, e.g.
#   PROMPTS=50 GPU_LIST="1 4 8" ./ttft_sweep.sh run

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- What to run -------------------------------------------------------------
VENV="${VENV:-/home/pese/miniforge3/envs/kvcache}"
MODEL="${MODEL:-/home/pese/qiuyu/Qwen2.5-7B-Instruct}"
GPU_LIST="${GPU_LIST:-1 2 3 4 5 6 7 8}"
CONFIG_LIST="${CONFIG_LIST:-no-zip qat8 iaa16 iaa32 qat+iaa16 qat+iaa32}"

# 16000 with a 95% shared prefix means 15200 tokens per request are served from the
# offloaded KV cache, so TTFT is dominated by the decompress + host-to-device path.
# An earlier sweep at 8000/80% left only a 1.6% spread across all six backends: the
# externally-loaded portion was too small to show through.
INPUT_LEN="${INPUT_LEN:-16000}"
HIT_RATE="${HIT_RATE:-95}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"

# Concurrency 1 on purpose. At 4, queueing dominates TTFT and the backends become
# indistinguishable; the per-request serial path is what this benchmark is about.
CONCURRENCY="${CONCURRENCY:-1}"

# P99 of n samples is roughly the 2nd-worst observation at n=200. Below ~100 the
# P99 column degenerates into "the single worst request" and stops being useful.
PROMPTS="${PROMPTS:-200}"

# The first requests populate the shared prefix in the external cache; excluded.
WARMUPS="${WARMUPS:-4}"

# Must exceed INPUT_LEN + OUTPUT_LEN with headroom.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-20k}"

OUT_LOG="${OUT_LOG:-/tmp/ttft_e2e.log}"
OUT_JSONL="${OUT_JSONL:-/tmp/ttft_e2e.jsonl}"
# Under the repo rather than /tmp: editors and browsers refuse file:// outside the
# workspace, and _data is already gitignored.
OUT_HTML="${OUT_HTML:-$REPO_ROOT/_data/ttft_report.html}"

# ---- Toolchain ---------------------------------------------------------------
SITE_PACKAGES="$VENV/lib/python3.12/site-packages"
export CUDA_HOME="${CUDA_HOME:-$SITE_PACKAGES/nvidia/cu13}"
export LD_LIBRARY_PATH="$VENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# System gcc is 12 but only g++-11 ships cc1plus, so nvcc has to be pointed at it.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--ccbin /usr/bin/g++-11}"

PY="$VENV/bin/python3"

log() { echo -e "\n\033[1m== $* ==\033[0m"; }
die() { echo "ERROR: $*" >&2; exit 1; }

###############################################################################
# check
###############################################################################
cmd_check() {
    log "NUMA topology"
    # build_env splits QAT devices, DSA work queues and IAA instances per NUMA node,
    # so the GPU-to-node mapping in multigpu_scale.py must match this machine.
    lscpu | grep -E "^NUMA node[0-9]+ CPU" || die "no NUMA information"

    log "GPUs"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
        || die "nvidia-smi failed"

    log "Enabled accelerator work queues"
    # iax* drive IAA, dsa* drive the DSA transfer path. block_on_fault matters: with
    # bof=0 a page fault fails the descriptor outright instead of stalling the engine.
    local n_iax=0 n_dsa=0
    for d in /sys/bus/dsa/devices/wq*; do
        [[ "$(cat "$d/state" 2>/dev/null)" == enabled ]] || continue
        local parent
        parent="$(basename "$(readlink -f "$d/..")")"
        printf '  %-8s %-8s mode=%-10s bof=%s\n' \
            "$(basename "$d")" "$parent" "$(cat "$d/mode")" "$(cat "$d/block_on_fault")"
        [[ "$parent" == iax* ]] && n_iax=$((n_iax + 1))
        [[ "$parent" == dsa* ]] && n_dsa=$((n_dsa + 1))
    done
    echo "  -> $n_iax IAA wq, $n_dsa DSA wq"
    (( n_iax > 0 )) || die "no IAA work queue enabled; run tools/auto_config.sh"

    log "QAT devices"
    ls -d /sys/bus/pci/drivers/4xxx/0000:* 2>/dev/null | wc -l

    log "Kernel settings that affect the accelerators"
    # numa_balancing periodically makes anonymous PTEs PROT_NONE to sample access.
    # IAA reaches its DMA buffers through the IOMMU, so those samples turn into IO
    # page faults. iaa_zip.c defends itself with mbind+mlock; this is informational.
    echo "  numa_balancing = $(cat /proc/sys/kernel/numa_balancing 2>/dev/null)"
    echo "  RLIMIT_MEMLOCK = $(ulimit -l) KiB   (iaa_zip mlocks its DMA buffers)"

    log "Python environment"
    [[ -x "$PY" ]] || die "$PY not found; create the conda env first"
    "$PY" - <<'EOF' || die "torch/vllm import failed"
import torch, vllm
print(f"  torch {torch.__version__}")
print(f"  vllm  {vllm.__version__}")
# vLLM <= 0.26 pins torch 2.11, which would force a full iaxl rebuild against a
# different ABI. 0.27.x and 0.28.x leave torch 2.13 alone.
major, minor = (int(x) for x in vllm.__version__.split(".")[:2])
assert (major, minor) >= (0, 27), "vLLM >= 0.27 required for the 4-D KV cache layout"
EOF

    log "iaxl resolves to the working tree"
    PYTHONPATH="$REPO_ROOT" "$PY" -c "import iaxl; print(' ', iaxl.__file__)" \
        || die "cannot import iaxl"

    log "Model"
    [[ -d "$MODEL" ]] || die "model directory $MODEL not found"
    echo "  $MODEL"
    echo "preflight OK"
}

###############################################################################
# build
###############################################################################
cmd_build() {
    log "Rebuilding iaxl torch extension"
    # iaa_zip.c / cpu_zip.c / kv_zip.cpp are compiled straight into torch_ext (see the
    # file(GLOB ...) calls in CMakeLists.txt). Rebuilding _lib/lib*_zip.so alone has no
    # effect on the vLLM path -- this step is mandatory after touching those sources.
    cd "$REPO_ROOT" || die "cannot cd $REPO_ROOT"

    # CMake's find_package(Python) may pick the base interpreter, which has no torch,
    # and then fails with "Failed to locate the active PyTorch installation".
    export IAXL_CMAKE_ARGS="${IAXL_CMAKE_ARGS:--DPython_EXECUTABLE=$PY}"
    export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"

    "$PY" setup.py build_ext --inplace || die "build_ext failed"
    ls -l "$REPO_ROOT"/iaxl/torch_ext.cpython-*.so
}

###############################################################################
# run
###############################################################################
cmd_run() {
    [[ $EUID -eq 0 ]] || die "must run as root (DSA/IAA portal mmap)"

    log "Sweep: $(echo "$CONFIG_LIST" | wc -w) configs x $(echo "$GPU_LIST" | wc -w) GPU counts"
    echo "  input_len=$INPUT_LEN hit_rate=${HIT_RATE}% concurrency=$CONCURRENCY prompts=$PROMPTS"
    echo "  log=$OUT_LOG  jsonl=$OUT_JSONL"

    rm -rf /tmp/ttft_cache /tmp/ttft_raw_p*.json "$OUT_LOG" "$OUT_JSONL"

    # The sweep appends one JSON line per finished group, so a crash only loses the
    # group in flight. run_sweep() below is called again until nothing is missing.
    local attempt
    for attempt in 0 1 2 3; do
        local missing
        missing="$(missing_groups)"
        [[ -z "$missing" ]] && { echo "all groups complete"; return 0; }

        if (( attempt > 0 )); then
            echo "attempt $attempt: refilling ->"
            echo "$missing" | sed 's/^/  /'
            # A crashed API server leaves EngineCore reparented to init, still holding
            # the full GPU allocation, which makes every later group fail to start.
            pkill -9 -f "VLLM::EngineCore"
            pkill -9 -f "vllm serve"
            rm -rf /tmp/ttft_cache
            sleep 10
        fi

        # One invocation per config so a partial config only re-runs its missing counts.
        while read -r cfg gpus; do
            [[ -n "$cfg" ]] || continue
            run_sweep "$cfg" "$gpus"
        done <<< "$missing"
    done

    [[ -z "$(missing_groups)" ]] || die "groups still missing after 4 attempts"
}

# Emits "<config> <gpu counts>" per line for every group without a clean result yet.
missing_groups() {
    CONFIG_LIST="$CONFIG_LIST" GPU_LIST="$GPU_LIST" OUT_JSONL="$OUT_JSONL" "$PY" - <<'EOF'
import json, os
have = set()
path = os.environ["OUT_JSONL"]
if os.path.exists(path):
    with open(path) as fp:
        for line in fp:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            # A group that reported an error is treated as missing so it gets re-run.
            if d.get("rows") and not any(r.get("error") for r in d["rows"]):
                have.add((d["config"], len(d["gpus"])))
for cfg in os.environ["CONFIG_LIST"].split():
    todo = [n for n in os.environ["GPU_LIST"].split() if (cfg, int(n)) not in have]
    if todo:
        print(cfg, " ".join(todo))
EOF
}

run_sweep() {
    local cfg="$1" gpus="$2"
    echo "  running $cfg gpus=[$gpus]"
    "$PY" -u "$SCRIPT_DIR/ttft_sweep.py" \
        --model "$MODEL" \
        --configs "$cfg" \
        --gpus $gpus \
        --input-len "$INPUT_LEN" \
        --hit-rate "$HIT_RATE" \
        --output-len "$OUTPUT_LEN" \
        --concurrency "$CONCURRENCY" \
        --prompts "$PROMPTS" \
        --warmups "$WARMUPS" \
        --max-model-len "$MAX_MODEL_LEN" \
        --json-out "$OUT_JSONL" >> "$OUT_LOG" 2>&1
}

###############################################################################
# verify
###############################################################################
cmd_verify() {
    log "Groups completed"
    local want gone
    want=$(( $(echo "$CONFIG_LIST" | wc -w) * $(echo "$GPU_LIST" | wc -w) ))
    # Each missing_groups line is "<config> <n> <n> ...", so the group count is the
    # number of fields past the config name.
    gone=$(missing_groups | awk '{s += NF - 1} END {print s + 0}')
    echo "  expected $want groups, $gone still missing"
    (( gone == 0 )) || missing_groups | sed 's/^/    /'

    log "Failed groups in the log"
    # A stale FAILED line is fine if the group was refilled later; cross-check with
    # missing_groups, which reads the structured results rather than the text log.
    grep -c FAILED "$OUT_LOG" 2>/dev/null || echo 0

    log "GPU memory (must be 0 MiB everywhere)"
    # Non-zero here means an orphaned EngineCore survived teardown and poisoned the run.
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

    log "External KV cache hits on GPU 0"
    # Proves the prefill really came from the offloaded cache instead of being
    # recomputed; the expected value is INPUT_LEN * HIT_RATE / 100.
    echo "  expected $(( INPUT_LEN * HIT_RATE / 100 )) tokens per request"
    grep -oE "externally-cached tokens: [0-9]+" /tmp/ttft_server_gpu0.log 2>/dev/null \
        | sort | uniq -c | sort -rn | head -3
}

###############################################################################
# report
###############################################################################
# The config order given here is the order of every table, chart and legend, and
# the first one is the baseline the deltas are measured against.
cmd_report() {
    "$PY" "$SCRIPT_DIR/ttft_report.py" "$OUT_JSONL" --configs $CONFIG_LIST
}

cmd_html() {
    mkdir -p "$(dirname "$OUT_HTML")"
    "$PY" "$SCRIPT_DIR/ttft_report.py" "$OUT_JSONL" --configs $CONFIG_LIST --html "$OUT_HTML"
}

###############################################################################
case "${1:-all}" in
    check)  cmd_check ;;
    build)  cmd_build ;;
    run)    cmd_run ;;
    verify) cmd_verify ;;
    report) cmd_report ;;
    html)   cmd_html ;;
    all)    cmd_check && cmd_build && cmd_run && cmd_verify && cmd_report && cmd_html ;;
    *)      die "usage: $0 {check|build|run|verify|report|html|all}" ;;
esac
