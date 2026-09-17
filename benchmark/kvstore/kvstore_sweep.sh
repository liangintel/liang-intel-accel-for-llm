#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# ==============================================================================
# KVStore PUT/GET 性能扫描 —— 可复现测试脚本
# ==============================================================================
#
# 【测的是什么】
#   KVStore 是 iaxl 把 GPU 显存里的 KV cache 卸载到主机内存的那一层。
#     PUT = GPU 显存 -> (可选压缩) -> 主机 KV 池       （卸载 / offload）
#     GET = 主机 KV 池 -> (可选解压) -> GPU 显存       （回填 / reload）
#   本脚本直接压这两个原语，不起 vLLM、不加载模型。相比端到端 TTFT 测试，它把
#   「压缩后端 + 数据搬运」这条路径单独隔离出来，噪声小得多。
#
# 【扫描的两个维度】
#   1) GPU 卡数 1..8：每张卡一个独立进程，进程之间用文件屏障(barrier)对齐，
#      保证同一次迭代里所有卡同时发起 PUT/GET —— 这样加速器才是被真实争抢的。
#      卡数越多，QAT/IAA 的实例被摊得越薄，扩展性问题才会暴露出来。
#   2) 压缩后端 6 种配置：
#        no-zip     关压缩，KV 原样搬运（性能上界 / CPU 下界的参照系）
#        qat8       只用 8 张 QAT 卡（每卡 4 实例 = 32 实例）
#        iaa16      只用 IAA，全机 16 实例
#        iaa32      只用 IAA，全机 32 实例
#        qat+iaa16  QAT + IAA(16 实例) 同时启用
#        qat+iaa32  QAT + IAA(32 实例) 同时启用
#      6 x 8 = 48 组。
#
# 【为什么每次迭代要换一套 block hash】
#   iaxl 的 put/get 没有去重：同一个 hash 再 PUT 一次仍然会走完整的搬运+压缩。
#   但如果所有迭代复用同一套 hash，主机侧会不断覆盖同一批条目，缓存行为、内存
#   分配路径都跟真实场景不同。所以第 i 次迭代用前缀 "i<i>_" 的独立 hash 集：
#   PUT 永远是新写入，GET 永远 100% 命中，两个阶段做的功完全对等。
#   代价是主机内存占用 = iters x payload，见下面 POOL_GB 的约束。
#
# 【为什么必须 root】
#   DSA / IAA 的 portal 需要 mmap 设备文件，非 root 会直接退化或失败。
#
# ------------------------------------------------------------------------------
# 用法：
#   ./kvstore_sweep.sh check     # 环境自检（NUMA / GPU / QAT / IAA / 内存 / 版本）
#   ./kvstore_sweep.sh build     # 重新编译 iaxl C++ 扩展（改过 C++ 才需要）
#   ./kvstore_sweep.sh run       # 跑完整 48 组扫描，失败自动补跑
#   ./kvstore_sweep.sh verify    # 检查结果文件是否 48 组齐全
#   ./kvstore_sweep.sh report    # 打印控制台表格
#   ./kvstore_sweep.sh html      # 生成网页报告
#   ./kvstore_sweep.sh all       # check + run + verify + report + html
#
# 所有参数都能用环境变量覆盖，例如：
#   ITERS=200 GPU_LIST="1 8" ./kvstore_sweep.sh run
# ==============================================================================

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ------------------------------------------------------------------------------
# 可调参数
# ------------------------------------------------------------------------------
VENV="${VENV:-/home/pese/miniforge3/envs/kvcache}"   # conda kvcache 环境
PY="$VENV/bin/python3"

GPU_LIST="${GPU_LIST:-1 2 3 4 5 6 7 8}"              # 扫描的卡数
CONFIG_LIST="${CONFIG_LIST:-no-zip qat8 iaa16 iaa32 qat+iaa16 qat+iaa32}"

# 每次 PUT/GET 搬运的 KV 字节数（每卡）。256 MiB 对应约 4000 token 的 KV，
# 量级和一次长上下文 prefill 的回填相当；再小的话 Python 调用开销会污染读数。
PAYLOAD_MIB="${PAYLOAD_MIB:-256}"

# 计时迭代次数 = 分位数的样本量。100 个样本时 P99 约等于「第 2 差」的那次，
# 已经能稳定反映长尾；再往上加主要是被主机内存卡住（见 POOL_GB）。
ITERS="${ITERS:-100}"

# KV block 的 token 数。iaxl 的压缩粒度按 block 走，16 是 vLLM 的默认值。
BLOCK_TOKENS="${BLOCK_TOKENS:-16}"

# 每个进程的主机侧 KV 池上限(GiB)。必须 > ITERS x PAYLOAD_MIB / 1024，否则老条目
# 会被淘汰，GET 就不再是 100% 命中，测出来的是「miss 路径」而不是解压路径。
# 默认 100 x 256MiB = 25 GiB，留 28% 余量 -> 32 GiB；8 进程峰值实测约 188 GB，
# 本机 481 GB 可用，安全。不设这个值的话 iaxl 默认按「可用内存/10」每进程各拿一份，
# 8 个进程会严重超订。
POOL_GB="${POOL_GB:-32}"

TIMEOUT="${TIMEOUT:-1800}"                            # 单组超时(秒)
OUT_LOG="${OUT_LOG:-/tmp/kvs_e2e.log}"                # 控制台日志
OUT_JSONL="${OUT_JSONL:-/tmp/kvs_e2e.jsonl}"          # 每组一行 JSON
OUT_HTML="${OUT_HTML:-$REPO_ROOT/_data/kvstore_report.html}"

SWEEP_PY="$SCRIPT_DIR/kvstore_sweep.py"
REPORT_PY="$SCRIPT_DIR/kvstore_report.py"

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ==============================================================================
# check —— 环境自检。跑长测试之前先确认前提都成立，省得 20 分钟后才发现少个 WQ。
# ==============================================================================
cmd_check() {
    log "===== 环境自检 ====="

    [[ $EUID -eq 0 ]] || log "警告：当前不是 root。DSA/IAA portal 需要 root，run 时请用 sudo。"

    echo "--- NUMA ---"; numactl --hardware | head -6
    echo "--- 内存 ---"; free -g | head -2
    echo "--- GPU ---"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

    local ngpu qat wq_dsa wq_iaa
    ngpu=$(nvidia-smi -L | wc -l)
    qat=$(ls -d /sys/bus/pci/drivers/4xxx/0000:* 2>/dev/null | wc -l)
    wq_dsa=$(grep -l . /sys/bus/dsa/devices/wq*/state 2>/dev/null | xargs -r grep -l enabled | wc -l)
    echo "GPU 数=$ngpu  QAT 4xxx 设备数=$qat  已启用 WQ 数=$wq_dsa"
    [[ $ngpu -ge 1 ]] || die "没找到 GPU"
    [[ $qat  -ge 1 ]] || die "没找到 QAT 设备（qat8 / qat+iaa* 配置会失败）"

    echo "--- 加速器工作队列（name / mode / block_on_fault）---"
    for wq in /sys/bus/dsa/devices/wq*; do
        [[ -e "$wq/state" ]] || continue
        printf "  %-8s %-10s dev=%-6s mode=%-10s bof=%s\n" \
            "$(basename "$wq")" "$(cat "$wq/state")" \
            "$(basename "$(readlink -f "$wq/..")")" \
            "$(cat "$wq/mode" 2>/dev/null)" "$(cat "$wq/block_on_fault" 2>/dev/null)"
    done

    echo "--- Python 环境 ---"
    LD_LIBRARY_PATH="$VENV/lib" PYTHONPATH="$REPO_ROOT" "$PY" - <<'EOF' || die "Python 环境自检失败"
import sys, torch, psutil
print("python  ", sys.version.split()[0])
print("torch   ", torch.__version__, "cuda", torch.cuda.is_available())
import iaxl; print("iaxl     import OK")
from iaxl.kvstore.kvstore import KVStore; print("KVStore  import OK")
print("available RAM %.0f GB" % (psutil.virtual_memory().available / 1024**3))
EOF

    # RLIMIT_MEMLOCK：子进程要 pin DMA 缓冲区，软限制太低会静默退化成非 pinned 内存。
    echo "--- RLIMIT_MEMLOCK ---"; ulimit -l

    # 内存充足性检查：ITERS x PAYLOAD 是每进程真实写入量，乘最大卡数即峰值。
    local need_gb peak_gb avail_gb max_gpu
    max_gpu=$(echo "$GPU_LIST" | tr ' ' '\n' | sort -n | tail -1)
    need_gb=$(( ITERS * PAYLOAD_MIB / 1024 ))
    peak_gb=$(( need_gb * max_gpu ))
    avail_gb=$(free -g | awk 'NR==2{print $7}')
    echo "每进程写入 ${need_gb} GiB，${max_gpu} 卡峰值约 ${peak_gb} GiB，当前可用 ${avail_gb} GiB"
    [[ $POOL_GB -ge $need_gb ]] || die "POOL_GB=$POOL_GB 小于每进程写入量 ${need_gb} GiB，GET 会 miss"
    [[ $avail_gb -gt $peak_gb ]] || die "可用内存 ${avail_gb} GiB 不足以承载峰值 ${peak_gb} GiB"

    log "自检通过"
}

# ==============================================================================
# build —— 只在改过 iaxl 的 C++ 源码之后才需要。
#   g++-12 在本机缺 cc1plus，所以强制 nvcc 用 g++-11。
# ==============================================================================
cmd_build() {
    log "===== 编译 iaxl 扩展 ====="
    cd "$REPO_ROOT" || die "cd $REPO_ROOT 失败"
    CUDA_HOME="$VENV/lib/python3.12/site-packages/nvidia/cu13" \
    NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11" \
    IAXL_CMAKE_ARGS="-DPython_EXECUTABLE=$PY" \
    LD_LIBRARY_PATH="$VENV/lib" \
        "$PY" setup.py build_ext --inplace 2>&1 | tail -20
    [[ ${PIPESTATUS[0]} -eq 0 ]] || die "编译失败"
    log "编译完成"
}

# ------------------------------------------------------------------------------
# missing_groups —— 读结果文件，输出还缺哪些组，格式：「<config> <卡数> <卡数> ...」
# 用于 run 的自动补跑：一次崩溃只需要重跑缺的那几组，不用从头再来。
# ------------------------------------------------------------------------------
missing_groups() {
    OUT_JSONL="$OUT_JSONL" GPU_LIST="$GPU_LIST" CONFIG_LIST="$CONFIG_LIST" "$PY" - <<'EOF'
import json, os
path = os.environ["OUT_JSONL"]
want_n = [int(x) for x in os.environ["GPU_LIST"].split()]
want_c = os.environ["CONFIG_LIST"].split()
done = set()
if os.path.exists(path):
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        # 只有全部进程都干净返回的组才算完成，有 error 的要重跑。
        if r.get("rows") and not any(x.get("error") for x in r["rows"]):
            done.add((r["config"], len(r["gpus"])))
for c in want_c:
    miss = [str(n) for n in want_n if (c, n) not in done]
    if miss:
        print(c, " ".join(miss))
EOF
}

# ==============================================================================
# run —— 执行扫描。最多 4 轮：每轮只跑「还缺的组」，直到补齐或轮次用尽。
# ==============================================================================
cmd_run() {
    [[ $EUID -eq 0 ]] || die "run 必须以 root 执行（DSA/IAA portal 需要）：sudo -E $0 run"
    log "===== 开始扫描 ====="
    log "卡数=[$GPU_LIST]  配置=[$CONFIG_LIST]"
    log "payload=${PAYLOAD_MIB} MiB/卡/次  iters=$ITERS  pool=${POOL_GB} GiB/进程"
    log "结果 -> $OUT_JSONL"

    for attempt in 1 2 3 4; do
        local missing; missing=$(missing_groups)
        if [[ -z "$missing" ]]; then
            log "所有组已完成"
            return 0
        fi
        log "--- 第 $attempt 轮，待跑：---"
        echo "$missing" | sed 's/^/      /'

        # 上一轮可能留下僵尸子进程，占着 GPU 显存和加速器实例，先清干净。
        pkill -9 -f kvstore_benchmark.py 2>/dev/null
        sleep 3

        while read -r cfg gpus; do
            [[ -n "$cfg" ]] || continue
            log "跑 $cfg （卡数 $gpus）"
            LD_LIBRARY_PATH="$VENV/lib" PYTHONPATH="$REPO_ROOT" \
            "$PY" -u "$SWEEP_PY" \
                --configs "$cfg" \
                --gpus $gpus \
                --block-tokens "$BLOCK_TOKENS" \
                --payload-mib "$PAYLOAD_MIB" \
                --iters "$ITERS" \
                --pool-gb "$POOL_GB" \
                --timeout "$TIMEOUT" \
                --json-out "$OUT_JSONL" 2>&1 | tee -a "$OUT_LOG"
        done <<< "$missing"
    done

    local left; left=$(missing_groups)
    [[ -z "$left" ]] || { log "仍有组未完成："; echo "$left"; return 1; }
    log "所有组已完成"
}

# ==============================================================================
# verify —— 结果完整性检查。
# ==============================================================================
cmd_verify() {
    log "===== 结果校验 ====="
    [[ -f "$OUT_JSONL" ]] || die "结果文件不存在：$OUT_JSONL"

    local n_cfg n_gpu expect missing n_missing
    n_cfg=$(echo "$CONFIG_LIST" | wc -w)
    n_gpu=$(echo "$GPU_LIST" | wc -w)
    expect=$(( n_cfg * n_gpu ))
    missing=$(missing_groups)
    # missing 每行是「config n1 n2 ...」，缺的组数 = 每行字段数 - 1 的总和。
    n_missing=$(echo "$missing" | awk 'NF{s += NF - 1} END {print s + 0}')
    log "应有 $expect 组，仍缺 $n_missing 组"
    [[ -n "$missing" ]] && echo "$missing" | sed 's/^/      /'

    log "结果行数：$(wc -l < "$OUT_JSONL")"
    log "含 FAILED 的行数：$(grep -c '"error"' "$OUT_JSONL")"
    log "GPU 残留显存："
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/      /'
    [[ $n_missing -eq 0 ]]
}

cmd_report() {
    [[ -f "$OUT_JSONL" ]] || die "结果文件不存在：$OUT_JSONL"
    "$PY" "$REPORT_PY" "$OUT_JSONL" --configs $CONFIG_LIST
}

cmd_html() {
    [[ -f "$OUT_JSONL" ]] || die "结果文件不存在：$OUT_JSONL"
    mkdir -p "$(dirname "$OUT_HTML")"
    "$PY" "$REPORT_PY" "$OUT_JSONL" --configs $CONFIG_LIST --html "$OUT_HTML"
}

case "${1:-all}" in
    check)  cmd_check ;;
    build)  cmd_build ;;
    run)    cmd_run ;;
    verify) cmd_verify ;;
    report) cmd_report ;;
    html)   cmd_html ;;
    all)    cmd_check && cmd_run && cmd_verify && cmd_report && cmd_html ;;
    *)      die "未知子命令 '$1'（可用：check build run verify report html all）" ;;
esac
