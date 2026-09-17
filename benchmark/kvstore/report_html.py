# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces for the benchmark HTML reports.

Hand-rolled SVG and inline CSS on purpose: the reports get copied around and have
to render on machines with no network and no javascript.
"""

import html

CONFIG_ZH = {
    "no-zip": "关压缩（基线）",
    "qat8": "只用 QAT（8 设备）",
    "iaa16": "只用 IAA（16 实例）",
    "iaa32": "只用 IAA（32 实例）",
    "qat+iaa16": "QAT + IAA（16 实例）",
    "qat+iaa32": "QAT + IAA（32 实例）",
}


PALETTE = ["#555555", "#d62728", "#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]

CSS = """
:root { --line:#d7dce3; --ink:#1b1f24; --dim:#5a6572; }
* { box-sizing:border-box; }
body { margin:0 auto; max-width:1180px; padding:36px 28px 80px;
  font:15px/1.75 -apple-system,"Segoe UI","Noto Sans CJK SC","Microsoft YaHei",sans-serif;
  color:var(--ink); background:#fff; }
h1 { font-size:27px; margin:0 0 6px; }
h2 { font-size:20px; margin:44px 0 10px; padding-bottom:7px; border-bottom:2px solid var(--ink); }
h3 { font-size:16px; margin:26px 0 8px; }
.sub { color:var(--dim); margin:0 0 22px; }
table { border-collapse:collapse; width:100%; margin:14px 0 6px; font-size:13.5px;
  font-variant-numeric:tabular-nums; }
caption { text-align:left; font-weight:600; padding:8px 0; font-size:14px; }
th,td { border:1px solid var(--line); padding:6px 9px; text-align:right; }
thead th { background:#f2f4f7; text-align:center; }
tbody th { text-align:left; background:#fafbfc; font-weight:600; white-space:nowrap; }
tr.base td { color:var(--dim); }
td .d { display:block; font-size:11px; color:var(--dim); }
.note { color:var(--dim); font-size:13px; margin:4px 0 0; }
code { background:#f2f4f7; padding:1px 5px; border-radius:3px; font-size:12.5px; }
pre { background:#f7f8fa; border:1px solid var(--line); border-left:3px solid #888;
  padding:12px 14px; overflow-x:auto; font-size:12.5px; line-height:1.6; }
.chart { width:100%; height:auto; margin-top:10px; }
.grid { stroke:#e8ebef; } .axis { stroke:#8a94a0; }
.tick { font-size:11px; fill:var(--dim); }
.tick-y { text-anchor:end; } .tick-x { text-anchor:middle; }
.axis-label { font-size:12px; fill:var(--ink); text-anchor:middle; }
.legend { margin:6px 0 0; font-size:12.5px; color:var(--dim); }
.key { margin-right:16px; white-space:nowrap; }
.key i { display:inline-block; width:13px; height:3px; vertical-align:middle;
  margin-right:5px; border-radius:2px; }
.tag { display:inline-block; font-size:11px; font-weight:700; padding:1px 7px;
  border-radius:3px; margin-right:7px; vertical-align:2px; }
.m { background:#e3f2e5; color:#1c6b28; } .i { background:#fdf0d8; color:#8a5a06; }
li { margin:9px 0; }
/* Shaded cells carry their colour in --hc so the stylesheet, not an inline style,
   decides whether to paint it -- that is what makes the toggle below possible. */
.hc { background:var(--hc); }
.tgl { position:absolute; opacity:0; pointer-events:none; }
.tgl-l { position:sticky; top:0; z-index:5; display:block; margin:0 0 10px;
  padding:9px 2px; cursor:pointer; user-select:none; font-size:13px; color:var(--dim);
  background:#fff; border-bottom:1px solid var(--line); }
.tgl-l:hover { color:var(--ink); }
/* Box drawn with borders rather than a ballot-box glyph, which many fonts lack. */
.tgl-l::before { content:""; display:inline-block; width:12px; height:12px;
  margin-right:8px; vertical-align:-2px; border:1px solid #8a94a0; border-radius:2px; }
.tgl:checked ~ .tgl-l::before { background:#2ca02c; border-color:#2ca02c; }
.tgl:checked ~ .tgl-l { color:var(--ink); }
.tgl:checked ~ * .hc { background:none; }
"""


def page(title: str, subtitle: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="sub">{subtitle}</p>
<p><span class="tag m">实测</span>标记的是基准测试直接产出的数字；
<span class="tag i">推断</span>标记的是基于这些数字的解释，未经独立验证。</p>
<input type="checkbox" id="nohl" class="tgl">
<label for="nohl" class="tgl-l">隐藏表格的红绿着色（只看数字）</label>
{body}
</body></html>
"""


def svg_lines(xs, series, xlabel, ylabel, width=820, height=360, ytick_fmt="{:.0f}"):
    """Minimal line chart. series = [(label, [y...]), ...]; index picks the colour."""
    ml, mr, mt, mb = 66, 16, 18, 46
    pw, ph = width - ml - mr, height - mt - mb
    ys_all = [y for _, ys in series for y in ys]
    lo, hi = min(ys_all), max(ys_all)
    pad = (hi - lo) * 0.15 or (abs(hi) * 0.05 or 1)
    # Don't let the padding push the axis below zero for strictly positive data;
    # a negative latency or bandwidth tick is nonsense.
    lo, hi = (max(lo - pad, 0) if lo >= 0 else lo - pad), hi + pad
    sx = lambda x: ml + (x - xs[0]) / (xs[-1] - xs[0]) * pw
    sy = lambda y: mt + ph - (y - lo) / (hi - lo) * ph

    p = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    for i in range(5):  # horizontal grid + y ticks
        v = lo + (hi - lo) * i / 4
        y = sy(v)
        p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + pw}" y2="{y:.1f}" class="grid"/>')
        p.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" class="tick tick-y">'
                 f'{ytick_fmt.format(v)}</text>')
    for x in xs:
        p.append(f'<text x="{sx(x):.1f}" y="{mt + ph + 20}" class="tick tick-x">{x}</text>')
    p.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" class="axis"/>')
    p.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" class="axis"/>')
    p.append(f'<text x="{ml + pw / 2:.0f}" y="{height - 6}" class="axis-label">'
             f'{xlabel}</text>')
    p.append(f'<text x="14" y="{mt + ph / 2:.0f}" class="axis-label" '
             f'transform="rotate(-90 14 {mt + ph / 2:.0f})">{ylabel}</text>')
    for i, (label, ys) in enumerate(series):
        col = PALETTE[i % len(PALETTE)]
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
        p.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        for x, y in zip(xs, ys):
            p.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{col}"/>')
    p.append("</svg>")
    legend = "".join(
        f'<span class="key"><i style="background:{PALETTE[i % len(PALETTE)]}"></i>'
        f"{html.escape(label)}</span>"
        for i, (label, _) in enumerate(series))
    return "".join(p) + f'<div class="legend">{legend}</div>'


def heat(delta):
    """Green when better than the baseline, red when worse; |d| >= 10% is fully saturated."""
    if delta is None or abs(delta) < 1.0:
        return ""
    a = min(abs(delta) / 10.0, 1.0) * 0.5
    rgb = "214,39,40" if delta > 0 else "44,160,44"
    return f' class="hc" style="--hc:rgba({rgb},{a:.2f})"'


def matrix(configs, counts, caption, note, get, fmt="{:.1f}", baseline=None,
           pct=False, higher_better=False):
    """One config-by-GPU-count table, optionally shaded against a baseline row."""
    h = [f'<table><caption>{caption}</caption>',
         '<thead><tr><th>配置</th>' +
         "".join(f"<th>{n} 卡</th>" for n in counts) + "</tr></thead><tbody>"]
    for c in configs:
        cells = []
        for n in counts:
            v = get(c, n)
            d = None
            if baseline and c != baseline:
                b = get(baseline, n)
                d = (v / b - 1) * 100 if b else None
            txt = fmt.format(v)
            if pct and d is not None:
                txt += f'<span class="d">{d:+.1f}%</span>'
            shade = -d if (higher_better and d is not None) else d
            cells.append(f"<td{heat(shade)}>{txt}</td>")
        cls = ' class="base"' if c == baseline else ""
        h.append(f'<tr{cls}><th>{html.escape(c)}</th>' + "".join(cells) + "</tr>")
    h.append(f'</tbody></table><p class="note">{note}</p>')
    return "".join(h)
