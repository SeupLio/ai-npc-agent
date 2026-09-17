"""把「一次真实模型的跑批」渲染成一张自包含的 HTML 报告。

和 `report.py`（对照跑批）的分工：

- `report.py` 回答的是**相对问题**："改动 A 和改动 B 哪个更好"，
  它的主角是差值表。
- 这里回答的是**绝对问题**："这一次跑批的数字能不能信、信到什么程度"。

所以这份报告的第一屏不是通过率，而是**可信度**：调了多少次模型、
多少条是模板兜底、多少条根本没跑起来。原因很直接 ——

    一次没有调用过模型的跑批，通过率 100% 衡量的是框架的确定性逻辑，
    不是模型能力。

把这两个数字并排放，读者自己就能判断；只放通过率，就是在误导。
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

# 复用对照报告里的样式基元，避免两套 HTML 慢慢长歪。
# 它们是同包内的私有函数，这里是刻意复用而不是复制 —— 复制过的表头
# 已经错过一次位（见 report._metric_headers 的注释）。
from .report import _METRIC_LABELS, _bar, _delta, _esc

_CATEGORY_LABELS = {
    "task": "任务完成",
    "tools": "工具调用",
    "memory": "记忆召回",
    "persona": "人设一致",
    "safety": "安全边界",
    "turn_taking": "发言调度",
    "multi_npc": "多 NPC 协作",
    "minecraft": "Minecraft 世界",
    "generated": "生成用例",
}


def _category_label(key: str) -> str:
    return _CATEGORY_LABELS.get(key, key)


def _fmt_sec(seconds: float) -> str:
    seconds = float(seconds or 0.0)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{rest:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m"


# --------------------------------------------------------------------------- #
# 可信度
# --------------------------------------------------------------------------- #
def trust_summary(eval_payload: dict[str, Any]) -> dict[str, Any]:
    """把"这份分数可不可信"压成几个数字 + 一句话结论。

    单独抽成纯函数，因为它是这份报告里**唯一一句会被人引用的结论**，
    必须有测试钉住它在各种退化情形下说什么。
    """
    summary = eval_payload.get("summary") or {}
    batch = eval_payload.get("batch") or {}
    stats = batch.get("stats") or {}
    degraded = batch.get("degraded") or {}

    total = int(summary.get("total") or 0)
    calls = int(stats.get("llm_calls") or 0)
    failures = int(stats.get("llm_failures") or 0)
    degraded_cases = int(stats.get("degraded_cases") or 0)
    failed_cases = int(stats.get("failed_cases") or 0)

    # 规划调用失败的条数。规划失败会**静默回落到启发式规划**，
    # 所以"planner 开着"和"planner 一直在失败"跑出来的轨迹是一样的 ——
    # 这个数字是那个对照实验能不能读的前提。
    planner_failed = sum(
        1 for r in (eval_payload.get("results") or []) if r.get("planner_failures")
    )

    if total == 0:
        verdict = "没有用例，什么都没测。"
    elif calls == 0:
        verdict = (
            "这次跑批**一次模型都没调用** —— 分数衡量的是框架的确定性逻辑，"
            "不是模型能力。要测模型请加 --provider / --model。"
        )
    elif failed_cases:
        verdict = (
            f"有 {failed_cases}/{total} 条用例没能跑起来（基础设施故障，不是"
            "\"NPC 没做到\"）。通过率是**剩下的那些**算出来的，这些用例不计入任何一行。"
        )
    elif degraded_cases / max(total, 1) > 0.10:
        verdict = (
            f"有 {degraded_cases}/{total} 条用例中途掉进了模板兜底（超过 10%）。"
            "这些用例的台词不是模型写的，通过率偏高，不能当模型能力读。"
        )
    elif degraded_cases:
        verdict = (
            f"有 {degraded_cases}/{total} 条用例中途掉进了模板兜底。"
            "比例不高，但要知道有这回事。"
        )
    else:
        verdict = (
            f"全部 {total} 条用例都完整跑完，模型调用 {calls} 次无失败。"
            "这份分数可以作为该模型在该用例集上的读数。"
        )

    if planner_failed:
        verdict += (
            f"　⚠️ 另有 {planner_failed}/{total} 条的**规划调用失败**并静默回落到了"
            "启发式规划 —— 这些用例上的「模型规划」等于没开，"
            "拿它们做 planner 对照会得到「两组一样」的假结论。"
        )

    return {
        "total": total,
        "llm_calls": calls,
        "llm_failures": failures,
        "degraded_cases": degraded_cases,
        "failed_cases": failed_cases,
        "planner_failed_cases": planner_failed,
        "degraded_rate": round(degraded_cases / total, 3) if total else 0.0,
        "trustworthy": bool(total and calls and not failed_cases
                            and degraded_cases / max(total, 1) <= 0.10
                            and not planner_failed),
        "verdict": verdict,
        "degraded_note": degraded.get("verdict", ""),
    }


def _trust_block(eval_payload: dict[str, Any]) -> str:
    trust = trust_summary(eval_payload)
    cls = "ok" if trust["trustworthy"] else "warn"
    cells = [
        ("用例", str(trust["total"])),
        ("模型调用", str(trust["llm_calls"])),
        ("调用失败", str(trust["llm_failures"])),
        ("模板兜底", f'{trust["degraded_cases"]}（{trust["degraded_rate"]:.1%}）'),
        ("规划回落", str(trust["planner_failed_cases"])),
        ("跑不起来", str(trust["failed_cases"])),
    ]
    tiles = "".join(
        f'<div class="tile"><div class="tile-k">{_esc(k)}</div>'
        f'<div class="tile-v">{_esc(v)}</div></div>'
        for k, v in cells
    )
    extra = ""
    if trust["degraded_note"]:
        extra = f'<div class="tile-note">{_esc(trust["degraded_note"])}</div>'
    return (
        f'<div class="trust {cls}">'
        f'<div class="trust-v">{_esc(trust["verdict"])}</div>'
        f'<div class="tiles">{tiles}</div>{extra}</div>'
    )


# --------------------------------------------------------------------------- #
# 跑批统计
# --------------------------------------------------------------------------- #
def _stats_rows(eval_payload: dict[str, Any]) -> str:
    stats = (eval_payload.get("batch") or {}).get("stats") or {}
    if not stats:
        return '<tr><td colspan="2" class="muted">这份报告没有记录跑批统计。</td></tr>'
    executed = int(stats.get("executed_cases") or 0)
    reused = int(stats.get("reused_cases") or 0)
    speedup = float(stats.get("speedup") or 0.0)
    rows = [
        ("并发数", str(stats.get("concurrency", 1))),
        ("实际执行", f"{executed} 条"
                     + (f"（另有 {reused} 条从检查点复用）" if reused else "")),
        ("墙钟耗时", _fmt_sec(stats.get("wall_sec", 0))),
        ("串行估计", _fmt_sec(stats.get("serial_estimate_sec", 0))),
        ("实测加速比", f"{speedup:.2f}×"
                       if executed else "— （没有实际跑批，全是复用）"),
        ("并行效率", f"{float(stats.get('efficiency') or 0):.2f}"
                     "（= 加速比 / 并发数）" if executed else "—"),
        ("重试过的用例", str(stats.get("retried_cases", 0))),
        ("模型调用", f'{stats.get("llm_calls", 0)} 次'
                     f'（失败 {stats.get("llm_failures", 0)} 次）'),
    ]
    return "\n".join(
        f'<tr><td class="name">{_esc(k)}</td><td class="mono">{_esc(v)}</td></tr>'
        for k, v in rows
    )


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def _metric_table(summary: dict[str, Any], label: str) -> str:
    means = summary.get("metric_means") or {}
    passed, total = summary.get("passed", 0), summary.get("total", 0)
    head = "".join(f"<th>{_esc(v)}</th>" for v in _METRIC_LABELS.values())
    cells = "".join(f'<td class="num">{_bar(means.get(k, 0.0))}</td>' for k in _METRIC_LABELS)
    return (
        "<table><thead><tr><th>配置</th><th>通过</th>"
        f"{head}<th>通过率</th></tr></thead><tbody>"
        f'<tr><td class="name">{_esc(label)}</td>'
        f'<td class="num"><strong>{passed}/{total}</strong></td>'
        f"{cells}"
        f'<td class="num">{float(summary.get("pass_rate") or 0):.1%}</td></tr>'
        "</tbody></table>"
    )


def _category_rows(eval_payload: dict[str, Any]) -> str:
    by_cat = ((eval_payload.get("summary") or {}).get("by_category")) or {}
    if not by_cat:
        return '<tr><td colspan="4" class="muted">没有分类别数据。</td></tr>'
    rows = []
    for key in sorted(by_cat, key=lambda k: (-by_cat[k]["total"], k)):
        bucket = by_cat[key]
        rows.append(
            "<tr>"
            f'<td class="name">{_esc(_category_label(key))}</td>'
            f'<td class="mono">{_esc(key)}</td>'
            f'<td class="num">{bucket["passed"]}/{bucket["total"]}</td>'
            f'<td class="num">{_bar(bucket.get("pass_rate", 0.0))}</td>'
            "</tr>"
        )
    return "\n".join(rows)


def _baseline_rows(eval_payload: dict[str, Any], baseline: dict[str, Any] | None) -> str:
    """与离线基线的差值。**只在用例集完全相同的时候才画。**

    两次跑批的用例数不一样就相减，是"拿苹果比橘子"最常见的来源 ——
    所以这里宁可返回一句说明，也不给一张看起来很有道理的差值表。
    """
    if not baseline:
        return '<tr><td colspan="3" class="muted">没有提供基线报告，无法对比。</td></tr>'
    cur = eval_payload.get("summary") or {}
    base = baseline.get("summary") or {}
    if cur.get("total") != base.get("total"):
        return (
            '<tr><td colspan="3" class="muted">'
            f"用例数不一致（{base.get('total')} vs {cur.get('total')}），"
            "不是同一张卷子，不做差值。</td></tr>"
        )
    cur_m = cur.get("metric_means") or {}
    base_m = base.get("metric_means") or {}
    rows = [
        ("通过率", float(cur.get("pass_rate") or 0) - float(base.get("pass_rate") or 0)),
    ]
    rows += [(label, cur_m.get(k, 0.0) - base_m.get(k, 0.0))
             for k, label in _METRIC_LABELS.items()]
    return "\n".join(
        f'<tr><td class="name">{_esc(k)}</td><td class="num">{_delta(v)}</td>'
        f'<td class="muted">{"模型更好" if v > 0 else ("离线更好" if v < 0 else "持平")}</td></tr>'
        for k, v in rows
    )


# --------------------------------------------------------------------------- #
# 失败明细 / 台词
# --------------------------------------------------------------------------- #
def _failure_groups(eval_payload: dict[str, Any], limit: int = 40) -> str:
    results = eval_payload.get("results") or []
    failures = [r for r in results if not r.get("passed")]
    if not failures:
        return '<p class="muted">本次跑批没有失败用例。</p>'
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in failures:
        groups.setdefault(item.get("category") or "?", []).append(item)

    blocks = []
    for cat in sorted(groups, key=lambda k: (-len(groups[k]), k)):
        items = groups[cat][:limit]
        lis = "".join(
            f'<li><code>{_esc(r.get("case_id"))}</code>'
            f'<span class="muted">　{_esc(r.get("scenario"))}</span>'
            f'<ul>' + "".join(f"<li>{_esc(n)}</li>" for n in r.get("notes") or [])
            + "</ul></li>"
            for r in items
        )
        more = (f'<li class="muted">…另有 {len(groups[cat]) - len(items)} 条</li>'
                if len(groups[cat]) > len(items) else "")
        blocks.append(
            f'<div class="fail"><h4>{_esc(_category_label(cat))}'
            f'<span class="muted">　{len(groups[cat])} 条</span></h4>'
            f"<ul>{lis}{more}</ul></div>"
        )
    return "\n".join(blocks)


def _speech_samples(eval_payload: dict[str, Any], per_category: int = 4,
                    per_case: int = 2) -> str:
    """摊出台词样本。数字证明"比例变了"，只有台词能证明"它真的在说话"。"""
    from .compare import _norm, is_scripted, scripted_patterns

    patterns = scripted_patterns()
    results = eval_payload.get("results") or []
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        if item.get("speeches"):
            by_cat.setdefault(item.get("category") or "?", []).append(item)

    blocks = []
    for cat in sorted(by_cat, key=lambda k: (-len(by_cat[k]), k)):
        cases = by_cat[cat][:per_category]
        seen: set[str] = set()
        items: list[str] = []
        for case in cases:
            picked = 0
            for speech in case.get("speeches") or []:
                key = _norm(speech)
                if not key or key in seen:
                    continue
                seen.add(key)
                scripted = is_scripted(speech, patterns)
                tag_cls = "t-scripted" if scripted else "t-free"
                tag_txt = "脚本" if scripted else "自由"
                items.append(
                    f'<li><span class="tag {tag_cls}">{tag_txt}</span>'
                    f'<span class="muted">{_esc(case.get("case_id"))}</span>'
                    f'<span class="line">　{_esc(speech)}</span></li>'
                )
                picked += 1
                if picked >= per_case:
                    break
        if items:
            blocks.append(
                f'<div class="sample"><h4>{_esc(_category_label(cat))}'
                f'<span class="muted">　展示 {len(items)} 句</span></h4>'
                f'<ul>{"".join(items)}</ul></div>'
            )
    return "\n".join(blocks) or '<p class="muted">没有台词记录。</p>'


# --------------------------------------------------------------------------- #
# 裁判
# --------------------------------------------------------------------------- #
def _calibration_block(judge_payload: dict[str, Any] | None) -> str:
    if not judge_payload:
        return ""
    cal = judge_payload.get("calibration") or {}
    per = cal.get("per_rubric") or {}
    if not per:
        return ""
    rows = []
    for key in sorted(per):
        stats = per[key]
        conf = stats.get("confusion") or {}
        rows.append(
            "<tr>"
            f'<td class="name">{_esc(key)}</td>'
            f'<td class="num">{stats.get("n", 0)}</td>'
            f'<td class="num">{float(stats.get("agreement") or 0):.0%}</td>'
            f'<td class="num"><strong>{float(stats.get("kappa") or 0):.2f}</strong></td>'
            f'<td class="muted">{_esc(stats.get("reading", ""))}</td>'
            f'<td class="num mono">{conf.get("真阳性", 0)}/{conf.get("真阴性", 0)}/'
            f'{conf.get("假阳性", 0)}/{conf.get("假阴性", 0)}</td>'
            "</tr>"
        )
    note = (
        '<div class="warn"><strong>kappa 是上界，不是泛化能力。</strong>'
        "这三条 rubric 是**对着这套校准集改过三轮**的，"
        "所以一致性里有一部分是拟合出来的。n=8，一个样本的摆动就是 ±0.125。"
        "真正的下一步是**留出集**：新标注样本时先不看裁判输出、不回头改 rubric。</div>"
    )
    return (
        "<table><thead><tr><th>评判标准</th><th>样本</th><th>一致率</th>"
        "<th>kappa</th><th>结论</th>"
        "<th>真阳/真阴/假阳/假阴</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>{note}"
    )


def _judge_block(judge_payload: dict[str, Any] | None) -> str:
    if not judge_payload:
        return (
            '<p class="muted">这份报告没有附带裁判结果。'
            "跑 <code>judge --report ... --json ...</code> 后可以再生成一次。</p>"
        )
    summary = judge_payload.get("summary") or {}
    coverage = judge_payload.get("coverage") or {}
    by_rubric = summary.get("by_rubric") or {}

    rows = []
    for key in sorted(by_rubric):
        stats = by_rubric[key]
        rows.append(
            "<tr>"
            f'<td class="name">{_esc(key)}</td>'
            f'<td class="num">{stats.get("passed", 0)}/{stats.get("n", 0)}</td>'
            f'<td class="num">{_bar(stats.get("pass_rate", 0.0))}</td>'
            "</tr>"
        )
    body = "".join(rows) or '<tr><td colspan="3" class="muted">没有判决。</td></tr>'

    coverage_note = ""
    if coverage:
        coverage_note = (
            '<div class="tiles" style="margin-top:10px">'
            f'<div class="tile"><div class="tile-k">判过的用例</div>'
            f'<div class="tile-v">{coverage.get("cases", 0)}</div></div>'
            # 「判过多少条」和「这一轮新判了多少条」是两件事。
            # 一次 --resume 只判 1 条、复用 227 条，和从头判 228 条，
            # 报告上都是"228 条判完了" —— 不写出来就分不清。
            f'<div class="tile"><div class="tile-k">本轮新判</div>'
            f'<div class="tile-v">{coverage.get("executed", 0)}</div></div>'
            f'<div class="tile"><div class="tile-k">复用检查点</div>'
            f'<div class="tile-v">{coverage.get("reused", 0)}</div></div>'
            f'<div class="tile"><div class="tile-k">判决条数</div>'
            f'<div class="tile-v">{coverage.get("verdicts", 0)}</div></div>'
            f'<div class="tile"><div class="tile-k">未判</div>'
            f'<div class="tile-v">{coverage.get("unjudged", 0)}</div></div>'
            f'<div class="tile"><div class="tile-k">判分时炸了</div>'
            f'<div class="tile-v">{coverage.get("cases_failed", 0)}</div></div>'
            f'<div class="tile"><div class="tile-k">无对话可判</div>'
            f'<div class="tile-v">{coverage.get("cases_without_dialogue", 0)}</div></div>'
            "</div>"
            f'<div class="tile-note">{_esc(coverage.get("verdict", ""))}</div>'
        )
        reused = int(coverage.get("reused") or 0)
        if reused:
            coverage_note += (
                f'<div class="tile-note">其中 {reused} 条是从检查点复用的，'
                "没有重新调用模型 —— 这些判决来自上一次判分。</div>"
            )

    return (
        f'<table><thead><tr><th>评判标准</th><th>通过</th><th>通过率</th></tr></thead>'
        f"<tbody>{body}</tbody></table>{coverage_note}"
    )


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
  .kv { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 10px 24px; margin: 0; }
  .kv div { border-bottom: 1px solid var(--line); padding: 6px 0; font-size: 13.5px; }
  .kv b { color: var(--muted); font-weight: 500; margin-right: 8px; }
  .trust { border-radius: 8px; padding: 14px 18px; font-size: 14.5px; border: 1px solid var(--line); }
  .trust.ok { background: #f2fbf5; border-color: #c6e7d0; border-left: 4px solid #0f9d58; }
  .trust.warn { background: #fffbf0; border-color: #f0dfb8; border-left: 4px solid #e8a33d; }
  .trust-v { font-size: 15px; }
  .tiles { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
  .tile { background: #fff; border: 1px solid var(--line); border-radius: 7px;
          padding: 7px 13px; min-width: 104px; }
  .tile-k { color: var(--muted); font-size: 11.5px; }
  .tile-v { font-size: 17px; font-weight: 600;
            font-family: ui-monospace, Menlo, Consolas, monospace; }
  .tile-note { margin-top: 10px; font-size: 13px; color: #4b5563; }
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
  .fail ul ul { margin: 2px 0 6px; padding-left: 18px; }
  .sample { border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px;
            margin-bottom: 10px; background: #fcfcfd; }
  .sample ul { margin: 4px 0 0; padding: 0; list-style: none; }
  .sample li { padding: 4px 0; border-bottom: 1px dashed #eef0f3; font-size: 13.5px; }
  .sample li:last-child { border-bottom: none; }
  .tag { display: inline-block; min-width: 34px; text-align: center; font-size: 11px;
         padding: 1px 6px; border-radius: 4px; margin-right: 9px; vertical-align: 1px; }
  .t-free { background: #e6f4ea; color: #137333; border: 1px solid #c6e7d0; }
  .t-scripted { background: #f1f3f4; color: #5f6368; border: 1px solid #e0e3e6; }
  .line { color: #1f2430; }
  code { background: #eef0f4; padding: 1px 5px; border-radius: 4px; font-size: 12.5px; }
  .note { background: var(--soft); border: 1px solid var(--line); border-radius: 8px;
          padding: 14px 18px; font-size: 13.5px; color: #374151; }
  .note li { margin-bottom: 5px; }
  .warn { background: #fffbf0; border: 1px solid #f0dfb8; border-left: 4px solid #e8a33d;
          border-radius: 8px; padding: 12px 16px; margin-top: 12px; font-size: 13.5px;
          color: #6b5320; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }
  @media (max-width: 900px) { .cols { grid-template-columns: 1fr; } }
  footer { margin-top: 36px; color: var(--muted); font-size: 12.5px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <div class="sub">__SUBTITLE__</div>

  <h2>这份分数可不可信</h2>
  __TRUST__

  <h2>跑批统计</h2>
  <div class="cols">
    <table><tbody>__STATS__</tbody></table>
    <table>
      <thead><tr><th>配置项</th><th>取值</th></tr></thead>
      <tbody>__CONFIG__</tbody>
    </table>
  </div>

  <h2>六维指标</h2>
  __METRICS__

  <h2>分类别</h2>
  <table>
    <thead><tr><th>类别</th><th>key</th><th>通过</th><th>通过率</th></tr></thead>
    <tbody>__CATEGORIES__</tbody>
  </table>

  <h2>与离线基线的差值</h2>
  <p class="sub" style="margin:0 0 10px">
    绿色为模型更好，红色为离线更好。这张表回答的是"接上模型到底值不值"。
  </p>
  <table>
    <thead><tr><th>指标</th><th>差值</th><th>谁更好</th></tr></thead>
    <tbody>__DELTAS__</tbody>
  </table>

  <h2>LLM-as-judge</h2>
  <p class="sub" style="margin:0 0 10px">
    先校准，再判分。没跑过校准的裁判只是"另一个模型的意见"。
  </p>
  <h4 style="margin-top:18px">裁判校准（24 条人工标注）</h4>
  __CALIBRATION__
  <h4 style="margin-top:22px">判分结果</h4>
  __JUDGE__

  <h2>台词样本</h2>
  <p class="sub" style="margin:0 0 10px">
    <span class="tag t-free">自由</span>＝模型现场组织的语言　
    <span class="tag t-scripted">脚本</span>＝人设模板或世界知识库原文
  </p>
  __SAMPLES__

  <h2>失败明细</h2>
  __FAILURES__

  <h2>怎么读这张表</h2>
  <div class="note">
    <ul>
      <li><strong>先看"模型调用"，再看通过率。</strong>调用次数为 0 的 100% 和
          调用几千次的 100% 是两件完全不同的事，前者衡量的是框架的确定性逻辑。</li>
      <li><strong>模板兜底</strong>：模型调用失败后框架会退回离线模板，
          那条用例的台词就不是模型写的了。比例高的时候通过率会虚高。</li>
      <li><strong>加速比</strong>是<strong>量出来的</strong>：串行估计 / 实际墙钟，
          分子分母都只用本次实际执行的用例。明显低于并发数说明瓶颈在端点排队。</li>
      <li><strong>诚实声明</strong>：228 条全部是<strong>自建</strong>用例，
          不是 AgentBench / τ-bench 这类公开榜单。它证明的是"这套框架在自建回归集上
          能被测量、且改动可归因"，<strong>不代表</strong> NPC 的通用能力，
          也不能和外部分数横向比较。</li>
      <li><strong>kappa 是上界</strong>：校准集被用来改过 rubric，
          一致性里有一部分是拟合。留出集是下一步，不是已完成项。</li>
    </ul>
  </div>

  <footer>__FOOTER__</footer>
</div>
</body>
</html>
"""


def _config_rows(eval_payload: dict[str, Any]) -> str:
    config = eval_payload.get("config") or {}
    if not config:
        return '<tr><td colspan="2" class="muted">没有记录配置。</td></tr>'
    return "\n".join(
        f'<tr><td class="name">{_esc(k)}</td><td class="mono">{_esc(v)}</td></tr>'
        for k, v in config.items()
    )


def render_batch_html(
    payload: dict[str, Any],
    title: str = "真实模型跑批报告",
    subtitle: str = "",
) -> str:
    """渲染一份跑批报告。

    `payload` 的形状：
        {"eval": <eval.json 的内容>,
         "judge": <judge --json 的内容，可选>,
         "baseline": <离线 eval.json，可选>}
    """
    eval_payload = payload.get("eval") or {}
    summary = eval_payload.get("summary") or {}
    config = eval_payload.get("config") or {}

    if not subtitle:
        model = config.get("model") or "?"
        planner = "LLM 规划" if config.get("use_llm_planner") else "启发式规划"
        subtitle = (
            f"{config.get('provider', '?')} / {model}　·　{planner}　·　"
            f"{summary.get('total', 0)} 条自建用例"
        )

    return (
        _TEMPLATE.replace("__TITLE__", _esc(title))
        .replace("__SUBTITLE__", _esc(subtitle))
        .replace("__TRUST__", _trust_block(eval_payload))
        .replace("__STATS__", _stats_rows(eval_payload))
        .replace("__CONFIG__", _config_rows(eval_payload))
        .replace("__METRICS__", _metric_table(summary, "本次跑批"))
        .replace("__CATEGORIES__", _category_rows(eval_payload))
        .replace("__DELTAS__", _baseline_rows(eval_payload, payload.get("baseline")))
        .replace("__CALIBRATION__", _calibration_block(payload.get("judge")) or
                 '<p class="muted">没有校准记录。</p>')
        .replace("__JUDGE__", _judge_block(payload.get("judge")))
        .replace("__SAMPLES__", _speech_samples(eval_payload))
        .replace("__FAILURES__", _failure_groups(eval_payload))
        .replace(
            "__FOOTER__",
            "由 <code>python -m npc_agent.cli report-batch</code> 生成　·　"
            "游戏 AI NPC 智能体框架",
        )
    )


def write_batch_html(
    payload: dict[str, Any] | str | Path,
    out_path: str | Path,
    title: str = "真实模型跑批报告",
    subtitle: str = "",
) -> Path:
    if isinstance(payload, (str, Path)):
        payload = json.loads(Path(payload).read_text(encoding="utf-8"))
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_batch_html(payload, title, subtitle), encoding="utf-8")
    return target
