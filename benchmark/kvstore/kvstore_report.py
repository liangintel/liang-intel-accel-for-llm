#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Turn a kvstore_sweep.py jsonl into console tables or a standalone HTML report.

    python3 kvstore_report.py /tmp/kvs_e2e.jsonl --configs no-zip qat8 ...
    python3 kvstore_report.py /tmp/kvs_e2e.jsonl --configs ... --html out.html
"""

import argparse
import html
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_html import CONFIG_ZH, heat, matrix, page, svg_lines  # noqa: E402

STATS = ("mean", "p50", "p90", "p95", "p99")
STAT_ZH = {"mean": "均值", "p50": "P50", "p90": "P90", "p95": "P95", "p99": "P99"}
GIB = 1024 ** 3


class Sweep:
    """Per-(config, gpu-count) aggregates over the per-GPU rows of one group."""

    def __init__(self, path, wanted):
        # lat[phase][config][n][stat] -> ms, averaged over the GPUs in the group.
        self.lat = {p: defaultdict(dict) for p in ("put", "get")}
        # bw[phase][config][n] -> aggregate GiB/s for the whole group.
        self.bw = {p: defaultdict(dict) for p in ("put", "get")}
        # cpu[phase][config][n] -> cores summed over the group's processes.
        self.cpu = {p: defaultdict(dict) for p in ("put", "get")}
        # host[phase][config][n] -> host-wide cores. /proc/stat is machine-wide,
        # so every rank reports the same thing; averaging (not summing) is correct.
        self.host = {p: defaultdict(dict) for p in ("put", "get")}
        # cpu_gib[phase][config][n] -> CPU-milliseconds spent per GiB moved.
        self.cpu_gib = {p: defaultdict(dict) for p in ("put", "get")}
        self.ratio, self.zip_gbps, self.unzip_gbps = (defaultdict(dict) for _ in range(3))
        self.cfg, self.groups, self.payload_mib, self.iters = {}, 0, None, None
        counts = set()

        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows = rec.get("rows") or []
            if not rows or any(r.get("error") for r in rows):
                continue
            c, n = rec["config"], len(rec["gpus"])
            if c not in wanted:
                continue
            self.groups += 1
            counts.add(n)
            self.cfg[c] = rec["cfg"]
            self.payload_mib = rec["payload_mib"]
            self.iters = rec["iters"]
            k = len(rows)
            gib_per_op = rows[0]["bytes_per_iter"] / GIB

            for phase in ("put", "get"):
                self.lat[phase][c][n] = {
                    s: sum(r[phase][f"{s}_ms"] for r in rows) / k for s in STATS}
                self.bw[phase][c][n] = sum(r[phase]["gibs_mean"] for r in rows)
                self.cpu[phase][c][n] = sum(r[phase]["cpu_cores"] for r in rows)
                self.host[phase][c][n] = sum(r[phase]["host_cpu_cores"] for r in rows) / k
                self.cpu_gib[phase][c][n] = (
                    sum(r[phase]["cpu_ms_per_op"] for r in rows) / k / gib_per_op)

            self.ratio[c][n] = sum(r["compression_ratio"] for r in rows) / k
            self.zip_gbps[c][n] = sum(r["compress_gbps"] for r in rows)
            self.unzip_gbps[c][n] = sum(r["decompress_gbps"] for r in rows)

        # Keep the caller's ordering so the baseline stays in row 1.
        self.configs = [c for c in wanted if c in self.cfg]
        # A partially finished sweep has different counts per config; restrict to the
        # ones every config completed so the tables stay rectangular and comparable.
        self.counts = sorted(n for n in counts
                             if all(n in self.lat["put"][c] for c in self.configs))

    def scaling(self, phase, c):
        """Aggregate GiB/s at max cards divided by (1-card GiB/s x cards) -> 1.0 is linear."""
        lo, hi = self.counts[0], self.counts[-1]
        ideal = self.bw[phase][c][lo] / lo * hi
        return self.bw[phase][c][hi] / ideal


def table(sw, title, get, fmt="{:8.2f}"):
    head = f"{'config':>10}" + "".join(f"{n:>9}" for n in sw.counts)
    body = "\n".join(
        f"{c:>10}" + "".join((" " + fmt).format(get(c, n)) for n in sw.counts)
        for c in sw.configs)
    return f"\n### {title}\n{head}\n{body}\n"


def render_text(sw):
    out = [f"\n# KVStore PUT/GET sweep: {sw.groups} groups, "
           f"{sw.payload_mib} MiB/GPU/op, {sw.iters} iterations per phase",
           "# rows = backend config, columns = GPU count (one process per GPU)"]
    for phase in ("put", "get"):
        for s in STATS:
            out.append(table(sw, f"{phase.upper()} {s} latency (ms), averaged over ranks",
                             lambda c, n, p=phase, k=s: sw.lat[p][c][n][k]))
    for phase in ("put", "get"):
        out.append(table(sw, f"{phase.upper()} aggregate bandwidth (GiB/s, summed over ranks)",
                         lambda c, n, p=phase: sw.bw[p][c][n]))
    for phase in ("put", "get"):
        out.append(table(sw, f"{phase.upper()} CPU busy (cores, summed over ranks)",
                         lambda c, n, p=phase: sw.cpu[p][c][n]))
        out.append(table(sw, f"{phase.upper()} CPU cost (CPU-ms per GiB moved)",
                         lambda c, n, p=phase: sw.cpu_gib[p][c][n]))
    out.append(table(sw, "Compression ratio (unzip/zip, higher = better)",
                     lambda c, n: sw.ratio[c][n], fmt="{:8.3f}"))
    out.append(table(sw, "Compress throughput (GB/s, summed over ranks)",
                     lambda c, n: sw.zip_gbps[c][n]))
    out.append(table(sw, "Decompress throughput (GB/s, summed over ranks)",
                     lambda c, n: sw.unzip_gbps[c][n]))

    out.append("\n### Scaling efficiency: aggregate GiB/s at "
               f"{sw.counts[-1]} cards / (1-card x {sw.counts[-1]}); 1.00 = linear")
    out.append(f"{'config':>10} {'PUT':>8} {'GET':>8}")
    for c in sw.configs:
        out.append(f"{c:>10} {sw.scaling('put', c):>8.2f} {sw.scaling('get', c):>8.2f}")
    return "\n".join(out)


def render_html(sw, jsonl_path):
    counts, base = sw.counts, sw.configs[0]
    lat = lambda p: (lambda c, n, k="p50": sw.lat[p][c][n][k])

    cfg_rows = "".join(
        f"<tr><th>{html.escape(c)}</th><td>{CONFIG_ZH.get(c, '')}</td>"
        f"<td>{'开' if sw.cfg[c]['compression'] else '关'}</td>"
        f"<td>{sw.cfg[c]['backends']}</td>"
        f"<td>{sw.cfg[c]['iaa_total']}</td></tr>" for c in sw.configs)

    def lat_tables(phase, first_idx):
        zh = "PUT（显存 → 主机）" if phase == "put" else "GET（主机 → 显存）"
        return "".join(
            matrix(sw.configs, counts, f"表 {first_idx + i}. {zh} 延迟 {STAT_ZH[s]}（毫秒）",
                   f"单元格 = 一次 {phase.upper()} 搬运 {sw.payload_mib} MiB KV 所花的时间，"
                   f"单位毫秒，<b>越小越好</b>。每张卡跑 {sw.iters} 次取该分位数，再对组内各卡"
                   f"求算术平均。小字为相对基线 <code>{base}</code> 的百分比变化，"
                   f"绿=更快、红=更慢，|Δ|&lt;1% 不着色。",
                   lambda c, n, k=s, p=phase: sw.lat[p][c][n][k],
                   fmt="{:.2f}", baseline=base, pct=True)
            for i, s in enumerate(STATS))

    scale_rows = "".join(
        f"<tr><th>{html.escape(c)}</th>"
        f"<td>{sw.bw['put'][c][counts[0]]:.2f}</td>"
        f"<td>{sw.bw['put'][c][counts[-1]]:.2f}</td>"
        f"<td>{sw.scaling('put', c):.2f}</td>"
        f"<td>{sw.bw['get'][c][counts[0]]:.2f}</td>"
        f"<td>{sw.bw['get'][c][counts[-1]]:.2f}</td>"
        f"<td>{sw.scaling('get', c):.2f}</td></tr>" for c in sw.configs)

    ch = lambda p, k, unit: svg_lines(
        counts, [(c, [sw.lat[p][c][n][k] for n in counts]) for c in sw.configs],
        "GPU 卡数（每卡一个独立进程，同步发起）", unit, ytick_fmt="{:.1f}")

    chart_put_p50 = ch("put", "p50", "PUT P50 延迟（毫秒，越低越好）")
    chart_get_p50 = ch("get", "p50", "GET P50 延迟（毫秒，越低越好）")
    chart_put_p99 = ch("put", "p99", "PUT P99 延迟（毫秒，越低越好）")
    chart_get_p99 = ch("get", "p99", "GET P99 延迟（毫秒，越低越好）")
    chart_bw = svg_lines(
        counts, [(c, [sw.bw["get"][c][n] for n in counts]) for c in sw.configs],
        "GPU 卡数", "GET 整机聚合带宽（GiB/s，越高越好）", ytick_fmt="{:.0f}")
    chart_cpu = svg_lines(
        counts, [(c, [sw.cpu["put"][c][n] for n in counts]) for c in sw.configs],
        "GPU 卡数", "PUT 阶段整组 CPU 占用（核，越低越好）", ytick_fmt="{:.0f}")

    body = f"""
<h2>1. 测试设计</h2>
<p>本报告测的是 <b>KVStore 的两个原语</b>，不经过 vLLM、不加载模型，所以读数只反映
「数据搬运 + 压缩/解压」这条路径本身：</p>
<ul>
<li><b>PUT（卸载 offload）</b>：GPU 显存 → 压缩 → 主机 KV 池。</li>
<li><b>GET（回填 reload）</b>：主机 KV 池 → 解压 → GPU 显存。</li>
</ul>
<p>每张 GPU 起一个独立进程，进程之间用文件屏障对齐：<b>同一次迭代里所有卡同时发起
PUT/GET</b>，加速器因此被真实争抢。屏障等待在计时器之外，不计入延迟。</p>

<h3>1.1 六种后端配置</h3>
<table><caption>表 1. 配置图例（取自结果文件中每组记录的 <code>cfg</code> 字段）</caption>
<thead><tr><th>标识</th><th>含义</th><th>压缩</th><th>后端</th><th>IAA 实例总数</th></tr></thead>
<tbody>{cfg_rows}</tbody></table>
<p class="note"><b>标识</b>：报告中所有表格和曲线使用的配置名。
<b>压缩</b>：<code>IAXL_KV_COMPRESSION</code>，关闭时 KV 原样搬运。
<b>后端</b>：<code>qat</code> 只用 QAT 卡、<code>iaa</code> 只用 IAA、<code>both</code> 两者同时启用。
<b>IAA 实例总数</b>：整机分配的 IAA 实例数，按 GPU 数均分到各进程。</p>

<h3>1.2 关键参数与取值理由</h3>
<table><caption>表 2. 参数取值</caption>
<thead><tr><th>参数</th><th>取值</th><th>为什么这么取</th></tr></thead><tbody>
<tr><th>每次搬运量</th><td>{sw.payload_mib} MiB / 卡 / 次</td>
<td style="text-align:left">约等于一次长上下文 prefill 需要回填的 KV 量级；再小的话
Python 调用开销会污染读数。</td></tr>
<tr><th>计时迭代数</th><td>{sw.iters} 次 / 阶段 / 卡</td>
<td style="text-align:left">分位数的样本量。{sw.iters} 个样本时 P99 约等于「第二差」的那次，
足以反映长尾；再往上加会被主机内存容量卡住。</td></tr>
<tr><th>block hash</th><td>每次迭代一套独立的</td>
<td style="text-align:left">iaxl 的 put/get 不做去重。用独立 hash 集可保证 PUT 永远是
新写入、GET 永远 100% 命中，两个阶段做的功完全对等。</td></tr>
<tr><th>主机 KV 池</th><td>32 GiB / 进程</td>
<td style="text-align:left">必须大于「迭代数 × 每次搬运量」= {sw.iters * sw.payload_mib / 1024:.0f} GiB，
否则老条目被淘汰、GET 变成 miss，量到的就不是解压路径了。</td></tr>
<tr><th>NUMA 绑定</th><td>按 GPU 所在 node</td>
<td style="text-align:left">GPU 0-3 在 node0、4-7 在 node1；进程用
<code>numactl --cpunodebind --membind</code> 绑到对应 node，避免跨 socket 访存干扰读数。</td></tr>
<tr><th>正确性校验</th><td>每组都做</td>
<td style="text-align:left">GET 之前把显存清零，GET 之后逐层比对张量；同时要求
cache miss = 0。任一不满足则该组标记为失败并重跑。</td></tr>
</tbody></table>

<h2>2. PUT 延迟（显存 → 主机，卸载方向）</h2>
{lat_tables("put", 3)}
{chart_put_p50}
<p class="note">图 1. 横轴 = GPU 卡数（1~{counts[-1]}）；纵轴 = PUT 的 P50 延迟，单位毫秒，越低越好。
每条曲线是一种后端配置，颜色见图例。注意每卡的搬运量固定为 {sw.payload_mib} MiB，
所以横轴右移代表<b>总负载在同比例增长</b>，曲线上扬即扩展性损失。</p>
{chart_put_p99}
<p class="note">图 2. 同图 1，但纵轴换成 P99 延迟（毫秒）。P99 与 P50 的差距就是长尾的幅度。</p>

<h2>3. GET 延迟（主机 → 显存，回填方向）</h2>
{lat_tables("get", 8)}
{chart_get_p50}
<p class="note">图 3. 横轴 = GPU 卡数；纵轴 = GET 的 P50 延迟（毫秒，越低越好）。</p>
{chart_get_p99}
<p class="note">图 4. 同图 3，纵轴换成 P99 延迟（毫秒）。</p>

<h2>4. 带宽与扩展性</h2>
{matrix(sw.configs, counts, "表 13. PUT 整机聚合带宽（GiB/s）",
        "组内各卡带宽之和，<b>越大越好</b>（本表着色方向与延迟表相反：绿=更快）。"
        "单卡带宽 = 每次搬运量 / 该卡的平均延迟。",
        lambda c, n: sw.bw["put"][c][n], fmt="{:.2f}",
        baseline=base, pct=True, higher_better=True)}
{matrix(sw.configs, counts, "表 14. GET 整机聚合带宽（GiB/s）",
        "同表 13，方向为主机 → 显存。",
        lambda c, n: sw.bw["get"][c][n], fmt="{:.2f}",
        baseline=base, pct=True, higher_better=True)}
{chart_bw}
<p class="note">图 5. 横轴 = GPU 卡数；纵轴 = GET 的整机聚合带宽（GiB/s，越高越好）。
理想的线性扩展应是一条过原点的直线，曲线变平即达到瓶颈。</p>

<table><caption>表 15. 扩展效率（{counts[0]} 卡 → {counts[-1]} 卡）</caption>
<thead><tr><th rowspan="2">配置</th><th colspan="3">PUT</th><th colspan="3">GET</th></tr>
<tr><th>{counts[0]} 卡 GiB/s</th><th>{counts[-1]} 卡 GiB/s</th><th>效率</th>
<th>{counts[0]} 卡 GiB/s</th><th>{counts[-1]} 卡 GiB/s</th><th>效率</th></tr></thead>
<tbody>{scale_rows}</tbody></table>
<p class="note"><b>效率</b> = {counts[-1]} 卡的聚合带宽 ÷ ({counts[0]} 卡带宽 × {counts[-1]})。
1.00 表示完美线性扩展，0.50 表示加到 {counts[-1]} 卡只拿到理想值的一半。</p>

<h2>5. CPU 消耗</h2>
{matrix(sw.configs, counts, "表 16. PUT 阶段整组 CPU 占用（核）",
        "对组内每个进程读 <code>getrusage(RUSAGE_SELF)</code> 的 utime+stime，除以该阶段"
        "墙钟时间得到「核」，再对组内所有进程求和。<b>越小越好</b>。"
        "注意各配置的 <code>OMP_NUM_THREADS</code> 不同"
        "（no-zip 64 / qat8 32 / iaa16 16 / iaa32 32 / qat+iaa16 48 / qat+iaa32 64），"
        "所以这一列不能脱离线程数裸比，要配合表 18 的归一化指标一起看。",
        lambda c, n: sw.cpu["put"][c][n], baseline=base, pct=True)}
{matrix(sw.configs, counts, "表 17. GET 阶段整组 CPU 占用（核）",
        "同表 16，统计的是 GET 阶段。",
        lambda c, n: sw.cpu["get"][c][n], baseline=base, pct=True)}
{matrix(sw.configs, counts, "表 18. PUT 的 CPU 单位成本（CPU-毫秒 / GiB）",
        "把表 16 除以实际搬运的数据量：<b>每搬运 1 GiB KV 需要花多少 CPU 毫秒</b>。"
        "这是跨配置、跨卡数唯一可直接比较的 CPU 指标，<b>越小越好</b>。",
        lambda c, n: sw.cpu_gib["put"][c][n], fmt="{:.1f}", baseline=base, pct=True)}
{matrix(sw.configs, counts, "表 19. GET 的 CPU 单位成本（CPU-毫秒 / GiB）",
        "同表 18，统计的是 GET 阶段。",
        lambda c, n: sw.cpu_gib["get"][c][n], fmt="{:.1f}", baseline=base, pct=True)}
{matrix(sw.configs, counts, "表 20. PUT 阶段整机 CPU 忙碌度（核）",
        "读 <code>/proc/stat</code> 的全机忙碌时间，能把内核侧的加速器相关开销也算进来。"
        "由于是全机口径，各进程读到的是同一个数，这里取平均而非求和；"
        "它比表 16 高出的部分即为测试进程之外的系统开销。",
        lambda c, n: sw.host["put"][c][n], baseline=base, pct=True)}
{chart_cpu}
<p class="note">图 6. 横轴 = GPU 卡数；纵轴 = PUT 阶段整组 CPU 占用（核，越低越好）。
每条曲线一种配置。曲线的斜率 = 每多接一张卡的边际 CPU 成本，截距 = 与卡数无关的固定开销
（主要由 <code>OMP_NUM_THREADS</code> 决定）。</p>

<h2>6. 压缩效果</h2>
{matrix(sw.configs, counts, "表 21. 压缩比（原始 / 压缩后，越大越好）",
        "由 iaxl 的 <code>compression_ratio</code> 指标直接给出。"
        "<code>no-zip</code> 恒为 1.000（不压缩）。本测试用的是随机 KV 张量，"
        "真实模型的 KV 分布不同，压缩比会有出入。",
        lambda c, n: sw.ratio[c][n], fmt="{:.3f}", higher_better=True)}
{matrix(sw.configs, counts, "表 22. 压缩吞吐（GB/s，组内求和，越大越好）",
        "iaxl 内部计时：压缩引擎处理原始数据的速率。仅统计压缩路径，不含 D2H 搬运。",
        lambda c, n: sw.zip_gbps[c][n], fmt="{:.1f}", higher_better=True)}
{matrix(sw.configs, counts, "表 23. 解压吞吐（GB/s，组内求和，越大越好）",
        "同表 22，方向为解压。解压通常显著快于压缩，这也是 GET 比 PUT 快的主因之一。",
        lambda c, n: sw.unzip_gbps[c][n], fmt="{:.1f}", higher_better=True)}
<p class="note"><b>注意</b>：<code>no-zip</code> 行的表 22/23 并不是压缩吞吐——关压缩时
iaxl 仍然对这条路径计时，量到的是纯 memcpy 的速率（压缩比恒为 1.000 可以佐证）。
它只能当作「不压缩时同一段搬运有多快」的参照，不能和其它行比压缩引擎强弱。</p>

<h2>7. 结论</h2>
<ol>
<li><span class="tag m">实测</span><b>GET 始终快于 PUT。</b>8 卡 P50：
<code>no-zip</code> 13.76 vs 27.45 ms，<code>qat+iaa32</code> 34.32 vs 57.32 ms。
<span class="tag i">推断</span>压缩比解压慢（表 22/23：8 卡 <code>qat+iaa32</code>
压缩 39.7 GB/s、解压 75.7 GB/s），且 PUT 端还要在主机侧做分配。</li>

<li><span class="tag m">实测</span><b>在这个微基准里关压缩最快。</b>8 卡 PUT P50
27.45 ms vs 最好的压缩配置 57.32 ms（慢 109%）；GET 13.76 vs 26.15 ms（慢 90%）。
<span class="tag i">推断 + 重要限定</span>本测试用的是<b>随机张量</b>，压缩比只有
1.25~1.26（表 21），压缩省下的搬运量抵不过压缩本身的开销。真实模型的 KV 冗余更多、
压缩比更高；同一套硬件上的<b>端到端 TTFT 测试结论相反</b>（8 卡 P50：<code>no-zip</code>
206 ms vs <code>iaa32</code> 203 ms、<code>iaa16</code> 210 ms，各配置基本持平），
因为那里的瓶颈在 PCIe 和主机带宽。<b>本报告只刻画原语路径，不能直接外推到端到端结论。</b></li>

<li><span class="tag m">实测</span><b>PUT 方向：QAT+IAA 混合 &gt; 单 QAT &gt; 单 IAA。</b>
8 卡 PUT P50：<code>qat+iaa32</code> 57.32 / <code>qat+iaa16</code> 58.44 /
<code>qat8</code> 87.36 / <code>iaa32</code> 145.78 / <code>iaa16</code> 157.54 ms。
<span class="tag i">推断</span>PUT 走压缩路径，而 IAA 的压缩能力最弱（8 卡压缩吞吐
<code>iaa32</code> 15.2 GB/s、<code>qat8</code> 25.6 GB/s、<code>qat+iaa32</code>
39.7 GB/s）。混合配置的吞吐≈两者之和，说明 QAT 与 IAA 的压缩引擎互不争抢、可叠加。</li>

<li><span class="tag m">实测</span><b>GET 方向反过来：纯 IAA（32 实例）最好。</b>
8 卡 GET P50：<code>iaa32</code> 26.15 &lt; <code>qat+iaa32</code> 34.32 &lt;
<code>qat+iaa16</code> 35.71 ≈ <code>iaa16</code> 35.91 &lt;&lt; <code>qat8</code> 50.38 ms。
<span class="tag i">推断</span>GET 走解压路径，IAA 解压很强（8 卡 <code>iaa32</code>
106.9 GB/s vs <code>qat8</code> 48.0 GB/s）。混合配置反而比纯 <code>iaa32</code> 慢，
是因为 PUT 时被路由到 QAT 压缩的那部分数据，GET 时只能由较慢的 QAT 解压，木桶效应。</li>

<li><span class="tag m">实测</span><b>GET 的 CPU 单位成本上，<code>iaa16</code> 甚至比关压缩还省。</b>
8 卡 CPU-毫秒/GiB：<code>iaa16</code> 344 &lt; <code>iaa32</code> 501 &lt;
<code>no-zip</code> 546 &lt; <code>qat8</code> 876 &lt; <code>qat+iaa16</code> 946 &lt;
<code>qat+iaa32</code> 1206。<span class="tag i">推断</span>IAA 是纯卸载式的，CPU 只提交
描述符；而关压缩时主机-显存之间的搬运本身就要占 CPU，且搬运量最大。</li>

<li><span class="tag m">实测</span><b>六种配置的多卡扩展性都不理想</b>（表 15）：
8 卡聚合带宽只有理想线性值的 PUT 0.15~0.28、GET 0.25~0.55。
<span class="tag i">推断</span>加速器实例总数是固定的（QAT 32 实例、IAA 16 或 32 实例），
卡数增加时每个进程分到的实例被摊薄，于是单卡延迟近似线性上升、聚合带宽趋于平台。
要继续扩展需要增加实例数或提高单实例吞吐，而不是加卡。</li>

<li><span class="tag m">实测</span><b>延迟分布极紧</b>：除下一条的异常外，所有组的
P99/P50 ≤ 1.02。<span class="tag i">推断</span>文件屏障让每次迭代由确定性的最慢路径主导，
随机抖动被吸收；这也意味着这些 P95/P99 反映的是<b>稳态成本</b>而非排队长尾。</li>

<li><span class="tag m">实测</span><b>唯一的异常点：<code>no-zip</code> 在 8 卡 PUT 上出现长尾。</b>
P50 27.45 ms 但 P90 83.34、P99 93.59 ms，聚合带宽从 7 卡的 73.67 掉到 54.73 GiB/s。
其余 47 组都没有这种现象。<span class="tag i">推断（未验证）</span>关压缩时主机侧写入量最大
（8 进程 × 25 GiB），最可能是撞到内存带宽或页分配的瓶颈。需要单独复现才能定论。</li>
</ol>

<h2>8. 复现步骤</h2>
<p>全部封装在 <code>benchmark/kvstore/kvstore_sweep.sh</code> 里，需要 root
（DSA/IAA portal 要 mmap 设备文件）：</p>
<pre>cd /home/pese/liang/1/liang-intel-accel-for-llm

# 1) 环境自检：NUMA / GPU / QAT / 工作队列 / 内存充足性 / Python 版本
sudo -n ./benchmark/kvstore/kvstore_sweep.sh check

# 2) 跑完整扫描（{sw.groups} 组），失败的组会自动补跑
sudo -n ./benchmark/kvstore/kvstore_sweep.sh run

# 3) 校验结果完整性
sudo -n ./benchmark/kvstore/kvstore_sweep.sh verify

# 4) 控制台表格 / 网页报告
./benchmark/kvstore/kvstore_sweep.sh report
./benchmark/kvstore/kvstore_sweep.sh html

# 只跑其中几组：
ITERS=200 GPU_LIST="1 8" CONFIG_LIST="no-zip iaa32" \\
    sudo -nE ./benchmark/kvstore/kvstore_sweep.sh run</pre>
<p class="note">原始逐次延迟保存在结果文件的 <code>times_s</code> 数组里，
任何其它分位数都可以事后重算，不必重跑。</p>
"""

    sub = (f"8× RTX 6000D · 8× Intel QAT · IAA/DSA 工作队列 · "
           f"共 {sw.groups} 组、每组每卡 {sw.iters} 次 PUT + {sw.iters} 次 GET · "
           f"单次搬运 {sw.payload_mib} MiB/卡 · "
           f"数据源 <code>{html.escape(jsonl_path)}</code>")
    return page("KVStore PUT/GET 多卡压缩后端评测报告", sub, body)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl")
    ap.add_argument("--configs", nargs="+", required=True,
                    help="Config names in report order; the first one is the baseline.")
    ap.add_argument("--html", default=None)
    args = ap.parse_args()

    sw = Sweep(args.jsonl, args.configs)
    if not sw.configs:
        raise SystemExit(f"no clean results in {args.jsonl}")
    if args.html:
        with open(args.html, "w") as fp:
            fp.write(render_html(sw, os.path.abspath(args.jsonl)))
        print(f"wrote {args.html}  ({sw.groups} groups)")
    else:
        print(render_text(sw))


if __name__ == "__main__":
    main()
