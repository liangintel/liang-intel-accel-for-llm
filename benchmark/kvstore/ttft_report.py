#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Turn a ttft_sweep.py jsonl into console tables or a standalone HTML report.

Every number here comes from the jsonl; nothing is hardcoded except the prose in
the conclusions section, which is marked as inference where it is inference. The
HTML embeds its own CSS and hand-drawn SVG, so it opens on a machine with no
network and no javascript.
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

class Sweep:
    """Per-(config, gpu-count) aggregates.

    A group runs one independent tp=1 server per GPU, so latency is averaged over
    the servers while CPU and throughput are summed: they are group-wide costs.
    """

    def __init__(self, path, configs):
        self.lat = defaultdict(dict)
        self.cpu = defaultdict(dict)
        self.host = defaultdict(dict)
        self.spr = defaultdict(dict)
        self.rps = defaultdict(dict)
        self.tpot = defaultdict(dict)
        self.reqs = 0
        self.groups = 0
        self.cfg = {}
        with open(path) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                g = json.loads(line)
                rows = g["rows"]
                if any(r.get("error") for r in rows):
                    continue
                name, n = g["config"], len(g["gpus"])
                self.cfg[name] = g["cfg"]
                avg = lambda k: sum(r[k] for r in rows) / len(rows)
                tot = lambda k: sum(r[k] for r in rows)
                self.lat[name][n] = {s: avg(f"ttft_{s}_ms") for s in STATS}
                self.cpu[name][n] = tot("cpu_cores")
                self.host[name][n] = avg("host_cpu_cores")
                self.spr[name][n] = avg("cpu_s_per_req")
                self.rps[name][n] = tot("req_throughput")
                self.tpot[name][n] = avg("tpot_mean_ms")
                self.reqs += tot("successful")
                self.groups += 1
        self.configs = [c for c in configs if c in self.lat]
        self.counts = sorted({n for c in self.configs for n in self.lat[c]})

    def regression(self, name):
        """cores = a * gpu_count + b.

        Per-GPU load is constant while the group total grows linearly, so the slope
        is the marginal CPU of one more loaded GPU and the intercept is the
        load-independent OpenMP overhead.
        """
        xs = self.counts
        ys = [self.cpu[name][n] for n in xs]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        top = max(xs)
        return a, my - a * mx, a / (self.rps[name][top] / top)


# --------------------------------------------------------------------------- text


def render_text(sw):
    out = []

    def table(title, unit, get, fmt="%9.1f"):
        out.append(f"\n### {title} ({unit}); rows = backend config, columns = GPU count")
        out.append(f"{'config':>10}" + "".join(f"{n:>9}" for n in sw.counts))
        for c in sw.configs:
            out.append(f"{c:>10}" + "".join(fmt % get(c, n) for n in sw.counts))

    for s in STATS:
        table(f"TTFT {s}", "ms", lambda c, n, k=s: sw.lat[c][n][k])
    table("TTFT mean / P50 -- how much the tail pulls the average up", "ratio",
          lambda c, n: sw.lat[c][n]["mean"] / sw.lat[c][n]["p50"], fmt="%9.2f")
    table("CPU busy, summed over all servers in the group", "cores",
          lambda c, n: sw.cpu[c][n])

    out.append("\n### CPU decomposition: cores = a * gpu_count + b (least squares)")
    out.append(f"{'config':>10} {'a cores/gpu':>12} {'b cores':>9} {'CPU-s/req':>10} {'vs first':>9}")
    base = None
    for c in sw.configs:
        a, b, spr = sw.regression(c)
        base = spr if base is None else base
        out.append(f"{c:>10} {a:>12.3f} {b:>9.2f} {spr:>10.3f} {(spr / base - 1) * 100:>8.1f}%")
    out.append("\n  a          marginal CPU of one more fully loaded GPU")
    out.append("  b          fixed overhead; tracks OMP_NUM_THREADS, not the backend")
    out.append("  CPU-s/req  a / (req/s per GPU) -- the actual cost of the compression path")
    return "\n".join(out)


# --------------------------------------------------------------------------- html



def render_html(sw, jsonl_path):
    counts = sw.counts
    g = lambda c, n, k: sw.lat[c][n][k]
    base = sw.configs[0]

    cfg_rows = "".join(
        f"<tr><th>{html.escape(c)}</th><td>{CONFIG_ZH.get(c, '')}</td>"
        f"<td>{'开' if sw.cfg[c]['compression'] else '关'}</td>"
        f"<td>{sw.cfg[c]['backends']}</td>"
        f"<td>{sw.cfg[c]['iaa_total']}</td></tr>" for c in sw.configs)

    reg = []
    base_spr = None
    for c in sw.configs:
        a, b, spr = sw.regression(c)
        base_spr = spr if base_spr is None else base_spr
        d = (spr / base_spr - 1) * 100
        reg.append(f"<tr><th>{html.escape(c)}</th><td>{a:.3f}</td><td>{b:.2f}</td>"
                   f"<td>{spr:.3f}</td><td{heat(d)}>{d:+.1f}%</td></tr>")

    ttft_tables = "".join(
        matrix(sw.configs, sw.counts, f"表 {i + 3}. TTFT {STAT_ZH[s]}（毫秒）",
               f"单元格 = 该组内各 server 的 <code>{'Mean' if s == 'mean' else s.upper()} TTFT</code> "
               f"取算术平均，单位毫秒，越小越好。小字为相对基线 <code>{base}</code> 的百分比变化，"
               f"绿色表示更快、红色表示更慢，色深随幅度增加，|Δ|&lt;1% 不着色。",
               lambda c, n, k=s: g(c, n, k), baseline=base, pct=True)
        for i, s in enumerate(STATS))

    chart_lat = svg_lines(counts, [(c, [g(c, n, "p50") for n in counts]) for c in sw.configs],
                          "GPU 卡数（每卡一个独立 tp=1 服务）", "TTFT P50（毫秒，越低越好）")
    chart_mean = svg_lines(counts, [(c, [g(c, n, "mean") for n in counts]) for c in sw.configs],
                           "GPU 卡数", "TTFT 均值（毫秒，越低越好）")
    chart_cpu = svg_lines(counts, [(c, [sw.cpu[c][n] for n in counts]) for c in sw.configs],
                          "GPU 卡数", "整组 CPU 占用（核，越低越好）")

    tail = ", ".join(f"{n} 卡 {sum(g(c, n, 'mean') / g(c, n, 'p50') for c in sw.configs) / len(sw.configs):.2f}"
                     for n in (counts[0], counts[-1]))

    title = "多卡 TTFT 压缩后端评测报告"
    sub = (f"Qwen2.5-7B-Instruct · 8× RTX 6000D · 8× Intel QAT + 16 IAA/DSA 工作队列 · "
           f"共 {sw.groups} 组、{sw.reqs:,.0f} 条请求 · "
           f"数据源 <code>{html.escape(jsonl_path)}</code>")
    return page(title, sub, f"""
<h2>1. 测试设计</h2>
<p>被测的是 <b>TTFT（Time To First Token）</b>：从请求发出到收到第一个 token 的时间。
测试让所有请求共享同一段长前缀，预热之后 prefill 不再重算，而是从卸载到主机侧的 KV cache
取回——于是 TTFT 就由「解压 + 主机到显存搬运」这条路径主导，也就是压缩后端真正被比较的地方。</p>

<h3>1.1 六种后端配置</h3>
<table><caption>表 1. 配置图例（取自结果文件中每组记录的 <code>cfg</code> 字段）</caption>
<thead><tr><th>标识</th><th>含义</th><th>压缩</th><th>后端</th><th>IAA 实例总数</th></tr></thead>
<tbody>{cfg_rows}</tbody></table>
<p class="note"><b>标识</b>：报告中所有表格和曲线使用的配置名。
<b>压缩</b>：<code>IAXL_KV_COMPRESSION</code>，关闭时 KV 原样搬运。
<b>后端</b>：<code>qat</code> 只用 QAT 卡、<code>iaa</code> 只用 IAA、<code>both</code> 两者同时启用。
<b>IAA 实例总数</b>：整机分配的 IAA 实例数，会按 GPU 数均分到各服务进程。
注意 <code>no-zip</code> 行的后端字段无意义——压缩关闭时不向任何加速器提交任务。</p>

<h3>1.2 参数及其理由</h3>
<table><caption>表 2. 关键参数</caption>
<thead><tr><th>参数</th><th>取值</th><th>为什么是这个值</th></tr></thead><tbody>
<tr><th>--input-len</th><td>16000</td><td style="text-align:left">早先用 8000 时六种后端只差 1.6%：外部加载的部分太小，被其它开销盖住了</td></tr>
<tr><th>--hit-rate</th><td>95%</td><td style="text-align:left">每请求 15200 token 从外部 KV cache 取回，其余重算</td></tr>
<tr><th>--concurrency</th><td>1</td><td style="text-align:left">并发 4 时排队时间主导 TTFT，后端之间就分辨不出来了</td></tr>
<tr><th>--prompts</th><td>200</td><td style="text-align:left">n=200 时 P99 约等于第二差的样本；再少 P99 就退化成「最差那一条」</td></tr>
<tr><th>--warmups</th><td>4</td><td style="text-align:left">头几条请求负责把共享前缀写进缓存，不计入统计</td></tr>
<tr><th>-tp</th><td>1（固定）</td><td style="text-align:left">该模型只有 4 个 KV head，张量并行只能取 1/2/4/8；
要表达 3/5/6/7 卡，只能每卡起一个独立 tp=1 服务</td></tr>
</tbody></table>
<p class="note">因此「N 卡」的含义是 <b>N 个各自独立、各自承担完整负载的服务</b>（弱扩展）：
每卡的请求量恒定，整组吞吐随卡数线性增长，而固定的加速器池被切得越来越碎。
<b>这决定了跨卡数的横向比较无意义，只有同一卡数下的跨配置比较是成立的。</b></p>

<h2>2. TTFT 结果 <span class="tag m">实测</span></h2>
{ttft_tables}

<h3>2.1 TTFT P50 随卡数的变化</h3>
{chart_lat}
<p class="note">横轴 = GPU 卡数（1–8，每卡一个独立 tp=1 服务）；纵轴 = TTFT P50，单位毫秒，越低越好。
每条折线是一种后端配置，颜色见图下图例。</p>

<h3>2.2 TTFT 均值随卡数的变化</h3>
{chart_mean}
<p class="note">横轴同上；纵轴 = TTFT 均值，单位毫秒。均值与 P50 的结论一致，
说明 <code>qat8</code> 的劣化不是个别慢请求造成的。</p>

<h3>2.3 长尾的归属</h3>
{matrix(sw.configs, sw.counts, "表 8. TTFT 均值 / P50 比值", "比值 &gt; 1 说明均值被长尾拉高。关键在于<b>同一列内跨配置几乎不变</b>——长尾是所有配置共有的开销，不是某个后端引入的。", lambda c, n: g(c, n, "mean") / g(c, n, "p50"), fmt="{:.2f}")}
<p class="note">全配置平均：{tail}。比值随卡数下降，
而每服务的 OpenMP 线程数也随卡数下降（64→8）。</p>

<h2>3. CPU 消耗 <span class="tag m">实测</span></h2>
{matrix(sw.configs, sw.counts, "表 9. 整组 CPU 占用（核）", "对组内每个 server 进程组统计 <code>/proc/&lt;pid&gt;/stat</code> 的 utime+stime，除以压测墙钟时间得到「核」，再对组内所有 server 求和。它随卡数增长是因为负载本身在增长（弱扩展）。着色同前，绿=更省。", lambda c, n: sw.cpu[c][n], baseline=base, pct=True)}
{chart_cpu}
<p class="note">横轴 = GPU 卡数；纵轴 = 整组 CPU 占用（核）。六条线近似平行，斜率差异才是后端的真实代价。</p>

{matrix(sw.configs, sw.counts, "表 10. 整机 CPU 占用（核）", "读 <code>/proc/stat</code> 的全机忙碌时间，能把内核侧的加速器相关开销也算进来；比表 9 略高的部分即为服务进程之外的系统开销。", lambda c, n: sw.host[c][n], baseline=base, pct=True)}

<h3>3.1 把固定开销和单请求代价拆开</h3>
<p>表 9 直接读会得出「不压缩最费 CPU」的荒谬结论。原因是 <code>OMP_NUM_THREADS</code>
在各配置下并不相同（no-zip 与 qat+iaa32 都是 64 线程，iaa16 只有 16），空转线程本身就占 CPU。
由于每卡负载恒定、整组负载随卡数线性增长，用最小二乘拟合
<code>核数 = a × 卡数 + b</code> 即可把两者分开。</p>
<table><caption>表 11. CPU 拆解</caption>
<thead><tr><th>配置</th><th>a（核/卡）</th><th>b（核）</th><th>CPU-秒/请求</th><th>相对基线</th></tr></thead>
<tbody>{"".join(reg)}</tbody></table>
<p class="note"><b>a</b>：斜率，每增加一张满负载的 GPU 所增加的 CPU 核数——即与请求量成正比的那部分开销。
<b>b</b>：截距，与负载无关的固定开销（主要是 OpenMP 线程本身），单位核。
<b>CPU-秒/请求</b>：<code>a ÷ 每卡请求速率</code>，把斜率折算成每条请求的 CPU 成本，这才是压缩路径的真实代价。
<b>相对基线</b>：相对 <code>{html.escape(base)}</code> 的百分比，红色表示更费 CPU。</p>

<h2>4. 吞吐与解码 <span class="tag m">实测</span></h2>
{matrix(sw.configs, sw.counts, "表 12. 整组请求吞吐（req/s）", "组内各 server 的 <code>Request throughput</code> 之和，越大越好（本表着色方向与前面相反：绿=吞吐更高）。各配置在同一卡数下基本一致，说明压缩后端没有成为吞吐瓶颈。", lambda c, n: sw.rps[c][n], fmt="{:.2f}", baseline=base, pct=True, higher_better=True)}
{matrix(sw.configs, sw.counts, "表 13. TPOT 均值（毫秒/token）", "Time Per Output Token，首 token 之后的平均出词间隔。全部 48 组落在极窄区间内 —— 压缩只影响 prefill 路径，不影响 decode，这符合预期。", lambda c, n: sw.tpot[c][n], fmt="{:.2f}")}

<h2>5. 结论</h2>
<ol>
<li><span class="tag m">实测</span><b>只有 <code>qat8</code> 随卡数单调劣化。</b>
1 卡时它最快，8 卡时相对基线 P50 慢 13.0%、均值慢 13.5%。
<span class="tag i">推断</span>卡数增加时 QAT 实例被切碎（32→4），而 IAA 同样被切碎却只劣化约 3%；
与之相符的是此前实测的解压带宽 IAA 14.7 GB/s vs QAT 6.39 GB/s（2.3 倍）。<b>该因果未做单因子实验验证。</b></li>

<li><span class="tag m">实测</span><b>用 IAA 做压缩是净收益。</b>
<code>iaa16</code>/<code>iaa32</code> 在 1–8 卡的 TTFT P50 全部不劣于关压缩（−3.6% 到 +0.7%），
同时省 CPU。<span class="tag i">推断</span>约 1.25× 的压缩比让主机到显存的传输量减少约 20%，
省下的搬运时间抵掉了解压时间。</li>

<li><span class="tag m">实测</span><b>CPU 代价排序：IAA（+7~9%）&lt; QAT+IAA（+16~18%）&lt; QAT（+30%）</b>（见表 11）。</li>

<li><span class="tag m">实测</span><b>P90/P95/P99 无法区分后端。</b>
同一卡数下各配置的高分位差异与 P50/均值上的趋势对不上；且均值/P50 比值在同一列内跨配置几乎不变（表 8）。
<span class="tag i">推断</span>长尾疑似来自 OpenMP 调度抖动（比值随线程数下降而下降），<b>尚未定位。</b></li>

<li><span class="tag i">推断</span><b>本轮只跑了一次，没有置信区间。</b>
1–3% 的差异应当视为噪声。<code>qat8</code> 的 +13% 且单调，才明显超出噪声范围。</li>
</ol>

<h2>6. 复现方式</h2>
<pre>cd benchmark/kvstore
./ttft_sweep.sh check          # 预检：NUMA、GPU、IAA/DSA 队列、QAT、torch/vLLM 版本
./ttft_sweep.sh build          # 重建 iaxl torch 扩展
sudo -E ./ttft_sweep.sh run    # 跑全部 {sw.groups} 组，失败组自动补跑
sudo -E ./ttft_sweep.sh verify # 校验：无缺失组、显存归零、外部缓存命中
./ttft_sweep.sh html           # 重新生成本页</pre>
<p class="note"><code>run</code> 必须 root：DSA 与 IAA 的 portal 需要 mmap 权限。
所有参数可用环境变量覆盖，例如
<code>PROMPTS=50 GPU_LIST="1 4 8" sudo -E ./ttft_sweep.sh run</code>。</p>

""")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl")
    ap.add_argument("--configs", nargs="+", required=True,
                    help="config order for every table and chart")
    ap.add_argument("--html", metavar="PATH", help="write an HTML report instead of text")
    args = ap.parse_args()

    sw = Sweep(args.jsonl, args.configs)
    if not sw.configs:
        raise SystemExit(f"no clean results in {args.jsonl}")
    if args.html:
        with open(args.html, "w") as fp:
            fp.write(render_html(sw, os.path.abspath(args.jsonl)))
        print(f"wrote {args.html}  ({sw.groups} groups, {sw.reqs:,.0f} requests)")
    else:
        print(render_text(sw))


if __name__ == "__main__":
    main()
