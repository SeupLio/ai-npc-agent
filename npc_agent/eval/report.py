"""把对照跑批的结果渲染成一张自包含的 HTML 报告。

为什么要 HTML 而不是只看终端表格：
1. 终端表格会被宽度截断，而且没法放进简历/作品集；
2. 对照实验的价值在于"一眼看出差别"，配色和条形比数字列更直观；
3. 单文件、无外部依赖、离线可开 —— 发给人看不会因为 CDN 挂掉而变成白屏。

主题固定为浅色（深色文字 + 浅色底），保证打印和截图都清晰。
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

_METRIC_LABELS = {
    "task": "任务完成",
    "tools": "工具调用",
    "memory": "记忆召回",
    "persona": "人设一致",
    "safety": "安全边界",
}


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _bar(value: float, width: int = 64) -> str:
    """0~1 的分数画成小条形，比裸数字更容易扫读。"""
    pct = max(0.0, min(1.0, float(value)))
    filled = round(pct * width)
    color = "#0f9d58" if pct >= 0.99 else ("#e8a33d" if pct >= 0.85 else "#d93025")
    return (
        f'<span class="bar"><span class="bar-fill" style="width:{filled}px;'
        f'background:{color}"></span></span>'
        f'<span class="bar-num">{pct:.3f}</span>'
    )


def _delta(value: float) -> str:
    if abs(value) < 0.0005:
        return '<span class="flat">0.000</span>'
    cls = "up" if value > 0 else "down"
    return f'<span class="{cls}">{value:+.3f}</span>'


def _pass_label(run: dict[str, Any]) -> str:
    """``RunOutcome.to_dict`` 落的是 passed/total，这里统一成 "n/m" 展示。

    别直接读 ``run["pass"]`` —— 那个键只存在于终端表格用的 ``Comparison.rows()`` 里，
    HTML 走的是 JSON 结构，读错会渲染出一个 "None"。
    """
    if "pass" in run:
        return str(run["pass"])
    passed, total = run.get("passed"), run.get("total")
    if passed is None or total is None:
        return "—"
    return f"{passed}/{total}"


def _run_rows(runs: list[dict[str, Any]]) -> str:
    out: list[str] = []
    for run in runs:
        means = run.get("metric_means", {})
        spec = run.get("spec", {})
        model = spec.get("model") or "—"
        out.append(
            "<tr>"
            f'<td class="name">{_esc(run.get("label"))}</td>'
            f'<td class="mono">{_esc(model)}</td>'
            f'<td class="mono">{_esc(spec.get("memory_strategy", "—"))}</td>'
            f'<td class="num"><strong>{_esc(_pass_label(run))}</strong></td>'
            + "".join(f'<td class="num">{_bar(means.get(k, 0.0))}</td>' for k in _METRIC_LABELS)
            + f'<td class="num">{float(run.get("free_speech_rate", 0)):.0%}</td>'
            f'<td class="num">{float(run.get("avg_speech_chars", 0)):.0f}</td>'
            f'<td class="num">{float(run.get("duration_sec", 0)):.0f}s</td>'
            "</tr>"
        )
    return "\n".join(out)


def _delta_rows(deltas: list[dict[str, Any]]) -> str:
    if not deltas:
        return '<tr><td colspan="9" class="muted">只有一次跑批，没有可对比的差值。</td></tr>'
    out: list[str] = []
    for row in deltas:
        out.append(
            "<tr>"
            f'<td class="name">{_esc(row.get("label"))}</td>'
            f'<td class="muted">{_esc(row.get("vs"))}</td>'
            f'<td class="num">{_delta(row.get("pass_rate", 0))}</td>'
            + "".join(f'<td class="num">{_delta(row.get(k, 0))}</td>' for k in _METRIC_LABELS)
            + f'<td class="num">{_delta(row.get("free", 0))}</td>'
            "</tr>"
        )
    return "\n".join(out)


def _failure_blocks(runs: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for run in runs:
        failures = run.get("failures") or []
        if not failures:
            continue
        items = "".join(
            f'<li><code>{_esc(f["case_id"])}</code><ul>'
            + "".join(f"<li>{_esc(note)}</li>" for note in f.get("notes", []))
            + "</ul></li>"
            for f in failures
        )
        blocks.append(
            f'<div class="fail"><h4>{_esc(run.get("label"))}</h4><ul>{items}</ul></div>'
        )
    if not blocks:
        return '<p class="muted">本次跑批没有失败用例。</p>'
    return "\n".join(blocks)


def _headline(runs: list[dict[str, Any]]) -> str:
    """一句话结论：把最关键的两个数拎到最前面。"""
    if not runs:
        return "没有数据。"
    parts = [
        f'<strong>{_esc(run.get("label"))}</strong> 通过 '
        f'<strong>{_esc(_pass_label(run))}</strong>，'
        f'自由台词 <strong>{float(run.get("free_speech_rate", 0)):.0%}</strong>'
        for run in runs
    ]
    return "　·　".join(parts)


# --------------------------------------------------------------------------- #
_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --ink: #1f2430; --muted: #6b7280; --line: #e5e7eb;
    --bg: #ffffff; --soft: #f7f8fa; --accent: #2f6fd0;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 40px 28px 64px; background: var(--bg); color: var(--ink);
    font: 15px/1.7 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
          "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  }
  .wrap { max-width: 1240px; margin: 0 auto; }
  h1 { font-size: 26px; margin: 0 0 6px; letter-spacing: -0.01em; }
  h2 { font-size: 18px; margin: 40px 0 12px; padding-bottom: 8px; border-bottom: 2px solid var(--line); }
  h4 { font-size: 14px; margin: 0 0 6px; }
  .sub { color: var(--muted); font-size: 14px; margin-bottom: 22px; }
  .headline {
    background: var(--soft); border: 1px solid var(--line); border-left: 4px solid var(--accent);
    border-radius: 8px; padding: 14px 18px; font-size: 14.5px;
  }
  table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
  th, td { padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: middle; }
  th { background: var(--soft); font-weight: 600; font-size: 12.5px; color: #374151; white-space: nowrap; }
  td.num { text-align: right; white-space: nowrap; }
  td.name { font-weight: 600; white-space: nowrap; }
  td.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12.5px; }
  tbody tr:hover { background: #fbfcfe; }
  .bar { display: inline-block; width: 64px; height: 7px; background: #eceef2; border-radius: 4px;
         overflow: hidden; vertical-align: middle; margin-right: 7px; }
  .bar-fill { display: block; height: 100%; }
  .bar-num { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; color: #374151; }
  .up { color: #0f9d58; font-weight: 600; }
  .down { color: #d93025; font-weight: 600; }
  .flat { color: #9aa1ab; }
  .muted { color: var(--muted); }
  .fail { background: #fff8f7; border: 1px solid #f3d3cf; border-radius: 8px;
          padding: 12px 16px; margin-bottom: 10px; }
  .fail ul { margin: 4px 0 0; padding-left: 20px; }
  .fail li { font-size: 13px; }
  code { background: #eef0f4; padding: 1px 5px; border-radius: 4px; font-size: 12.5px; }
  .note { background: var(--soft); border: 1px solid var(--line); border-radius: 8px;
          padding: 14px 18px; font-size: 13.5px; color: #374151; }
  .note li { margin-bottom: 5px; }
  footer { margin-top: 36px; color: var(--muted); font-size: 12.5px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <div class="sub">__SUBTITLE__</div>
  <div class="headline">__HEADLINE__</div>

  <h2>绝对值</h2>
  <table>
    <thead><tr>
      <th>配置</th><th>模型</th><th>记忆策略</th><th>通过</th>
      <th>任务完成</th><th>工具调用</th><th>记忆召回</th><th>人设一致</th><th>安全边界</th>
      <th>自由台词</th><th>均长</th><th>耗时</th>
    </tr></thead>
    <tbody>__ROWS__</tbody>
  </table>

  <h2>相对首行的差值</h2>
  <p class="sub" style="margin:0 0 10px">绿色为提升，红色为下降。这张表回答的是"改动到底有没有用"。</p>
  <table>
    <thead><tr>
      <th>配置</th><th>基线</th><th>通过率</th>
      <th>任务完成</th><th>工具调用</th><th>记忆召回</th><th>人设一致</th><th>安全边界</th>
      <th>自由台词</th>
    </tr></thead>
    <tbody>__DELTAS__</tbody>
  </table>

  <h2>失败明细</h2>
  __FAILURES__

  <h2>怎么读这张表</h2>
  <div class="note">
    <ul>
      <li><strong>自由台词</strong>：既不是人设模板、也不是世界知识库原文的台词占比。
          离线启发式只从固定字符串里取词，所以恒为 0%；真实模型应当显著高于 0 ——
          这是"接上模型到底值不值"最直接的证据。</li>
      <li><strong>记忆策略 none</strong>：不向 prompt 注入任何记忆，即「裸模型」基线。
          它与 <code>hybrid</code> 的差值就是这套记忆系统的净收益。</li>
      <li><strong>对照的公平性</strong>：同一套用例、同一套指标、同一份场景配置，只换一个变量。
          温度固定为 __TEMPERATURE__，以压低采样带来的波动。</li>
      <li><strong>诚实声明</strong>：用例集是自建的回归基线，不是 AgentBench / τ-bench 这类
          公开榜单，分数不能跨项目横向比较。它的用途是"证明我的改动没有把已有能力改坏，
          并且带来了可测量的提升"。</li>
    </ul>
  </div>

  <footer>__FOOTER__</footer>
</div>
</body>
</html>
"""


def render_comparison_html(
    data: dict[str, Any],
    title: str = "对照跑批报告",
    subtitle: str = "",
    temperature: float = 0.3,
) -> str:
    runs = data.get("runs", [])
    subtitle = subtitle or (
        f"用例范围：{data.get('categories', 'all')}"
        + (f"（前 {data['limit']} 条）" if data.get("limit") else "")
    )
    return (
        _TEMPLATE.replace("__TITLE__", _esc(title))
        .replace("__SUBTITLE__", _esc(subtitle))
        .replace("__HEADLINE__", _headline(runs))
        .replace("__ROWS__", _run_rows(runs))
        .replace("__DELTAS__", _delta_rows(data.get("deltas", [])))
        .replace("__FAILURES__", _failure_blocks(runs))
        .replace("__TEMPERATURE__", _esc(temperature))
        .replace(
            "__FOOTER__",
            "由 <code>python -m npc_agent.cli compare</code> 生成　·　"
            "游戏 AI NPC 智能体框架",
        )
    )


def write_comparison_html(
    data: dict[str, Any] | str | Path,
    out_path: str | Path,
    title: str = "对照跑批报告",
    subtitle: str = "",
    temperature: float = 0.3,
) -> Path:
    if isinstance(data, (str, Path)):
        data = json.loads(Path(data).read_text(encoding="utf-8"))
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        render_comparison_html(data, title, subtitle, temperature), encoding="utf-8"
    )
    return target
