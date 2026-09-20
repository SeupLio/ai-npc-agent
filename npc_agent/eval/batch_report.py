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
# 离线占位符**只有一个定义**（`harness.OFFLINE_MODEL`）—— 这里再用一个字面量
# 写一遍 `"(offline)"` 就是同一条规则的第二份实现，改了那边这边不会动。
from .harness import OFFLINE_MODEL

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
#: 通过率贴到这个高度就算「用例集饱和」。0.98 是刻意的保守值：
#: 99.6% 和 98% 在"还能不能区分好坏"这件事上没有实质区别。
SATURATION_PASS_RATE = 0.98
#: 用例太少时通过率高不算饱和 —— 20 条里过 20 条说明不了什么。
SATURATION_MIN_CASES = 20


def planner_breakdown(eval_payload: dict[str, Any]) -> dict[str, Any]:
    """把每条用例的**规划来源**汇总起来，并数出真的回落了的条数。

    ## 为什么它比 `planner_failures` 更该被引用

    `planner_failures` 数的是**失败的调用**；这里数的是**真的没走上模型规划的用例**。
    两者不等价：一条用例会按 tick 多次规划，**一次**失败就足以让那个 tick 落回启发式，
    而其余九次可能全成功。读者真正要回答的问题是

        「这条用例上的『模型规划』到底发生了没有」

    —— 那要用**观测到的来源**回答，不能用调用失败次数代理。
    本项目已经把"用数值代理量当闸门"栽过一次（`urgency >= 0.9`），这里是同一个教训。

    ⚠️ 没有这个数据的报告（2026-09-19 之前跑的那些）**必须说"没记"**，
    不能默认成 0 —— "取不到 ≠ 没有"。
    """
    results = eval_payload.get("results") or []
    config = eval_payload.get("config") or {}
    stats = (eval_payload.get("batch") or {}).get("stats") or {}

    # 模型规划这一路是不是**真的活着**（配置开着 **且** 配了模型）。
    # 判据只有一份实现（`_planner_is_live`）—— 报告副标题用的是同一个函数。
    planner_live = _planner_is_live(config)

    by_source: dict[str, int] = {}
    fallback_ids: list[str] = []
    empty_ids: list[str] = []
    has_provenance = False
    for item in results:
        sources = item.get("plans_by_source")
        if sources is None:
            continue
        has_provenance = True
        for key, count in sources.items():
            by_source[key] = by_source.get(key, 0) + int(count or 0)
        if planner_live and int(sources.get("heuristic") or 0) > 0:
            fallback_ids.append(str(item.get("case_id") or "?"))
        if int(item.get("planner_empty_plans") or 0) > 0:
            empty_ids.append(str(item.get("case_id") or "?"))

    # 老报告没有来源数据 ⇒ 退回调用失败数，并且**说清这是代理量**。
    failed_cases = sum(1 for r in results if r.get("planner_failures"))
    if not has_provenance:
        return {
            "has_provenance": False,
            "planner_live": planner_live,
            "by_source": {},
            "fallback_cases": failed_cases,
            "fallback_is_proxy": True,
            "empty_plan_cases": 0,
            "model_plan_cases": 0,
            "llm_calls": int(stats.get("llm_calls") or 0),
        }

    return {
        "has_provenance": True,
        "planner_live": planner_live,
        "by_source": by_source,
        "fallback_cases": len(fallback_ids),
        "fallback_ids": fallback_ids,
        "fallback_is_proxy": False,
        "empty_plan_cases": len(empty_ids),
        "empty_plan_ids": empty_ids,
        "model_plan_cases": sum(
            1 for r in results if (r.get("plans_by_source") or {}).get("model")
        ),
        "llm_calls": int(stats.get("llm_calls") or 0),
    }


#: 计划来源的中文名。**和 `npc_agent.types.PLAN_SOURCES` 一一对应** ——
#: 那边加了新来源而这里没跟上，报告会印出裸的英文 key（不报错，只是变难读）。
_SOURCE_LABELS = {
    "model": "模型规划",
    "heuristic": "启发式规划",
    "request_template": "点单模板",
    "scenario_flow": "场景引导",
}


def _planner_is_live(config: dict[str, Any]) -> bool:
    """模型规划这一路是不是**真的活着**：配置开着 **且** 配了模型。

    ⚠️ 只看 `use_llm_planner` 是不够的 —— 它默认就是 `True`，
    离线基线报告（`provider=null`、`model=(offline)`）也带着这个 True。
    只看它就会把离线报告标成"LLM 规划"，而且会把每个启发式计划记成回落
    （"对照组被记成一片红"）。
    """
    return bool(config.get("use_llm_planner")) and str(
        config.get("model") or ""
    ) not in ("", OFFLINE_MODEL)


def _planner_block(eval_payload: dict[str, Any]) -> str:
    """规划来源 + 静默回落。**报告里最容易被漏掉的一类污染。**"""
    info = planner_breakdown(eval_payload)
    total = int((eval_payload.get("summary") or {}).get("total") or 0)

    if not info["has_provenance"]:
        note = (
            '<div class="warn"><strong>这份报告没有记录规划来源。</strong>'
            "它是 2026-09-19 之前跑的（那时计划还没有 <code>source</code> 字段），"
            "所以下面这个「规划回落」只能拿<b>调用失败次数</b>当代理量 —— "
            "而它<strong>不等于</strong>真正回落的用例数：一条用例按 tick 多次规划，"
            "一次失败就足以让那个 tick 落回启发式。<br>"
            "要看真实来源，请重跑（会带上 <code>plans_by_source</code>）。</div>"
        )
    elif not info["planner_live"]:
        cfg = eval_payload.get("config") or {}
        note = (
            '<div class="note">这次跑批<strong>没有开启模型规划</strong>'
            f"（配置 <code>use_llm_planner={_esc(str(bool(cfg.get('use_llm_planner'))))}</code>，"
            f"模型 <code>{_esc(str(cfg.get('model') or '?'))}</code>）—— "
            "所以「启发式规划」是<b>预期行为</b>，不是回落。"
            "这一列在这里是<b>对照组的基线</b>。</div>"
        )
    elif info["fallback_cases"]:
        share = info["fallback_cases"] / max(total, 1)
        note = (
            '<div class="warn"><strong>⚠️ 有 '
            f'{info["fallback_cases"]}/{total} 条（{share:.0%}）的规划调用回落到了启发式规划。</strong>'
            "框架<strong>静默</strong>回落，两条路径产出的轨迹<strong>完全一样</strong> —— "
            "这些用例上的「模型规划」等于没开。<br>"
            "<b>这批数字因此不能当模型能力读</b>，"
            "也不能拿它做 planner 对照（会得到「两组一样」的假结论）。"
            "这条要先修端点/预算，再重跑。</div>"
        )
    else:
        note = (
            '<div class="good">这次跑批<strong>没有一条用例回落过</strong> —— '
            "所有计划都真的来自模型。这批数字可以作为模型规划的读数。</div>"
        )

    if info["has_provenance"] and info["by_source"]:
        rows = "".join(
            f'<tr><td class="name">{_esc(_SOURCE_LABELS.get(k, k))}</td>'
            f'<td class="mono">{_esc(k)}</td>'
            f'<td class="num">{count}</td>'
            f'<td class="num">{"—" if not info["by_source"] else f"{count / max(sum(info['by_source'].values()), 1):.0%}"}</td>'
            "</tr>"
            for k, count in sorted(
                info["by_source"].items(), key=lambda kv: (-kv[1], kv[0])
            )
        )
        table = (
            "<table><thead><tr><th>计划来源</th><th>key</th>"
            "<th>计划数</th><th>占比</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            '<p class="muted">数的是<b>计划</b>不是用例：一条用例会按 tick 规划很多次。'
            f'本次共有 {info["model_plan_cases"]} 条用例至少产出过一个模型计划。</p>'
        )
    else:
        table = ""

    if info["has_provenance"] and info["empty_plan_cases"]:
        table += (
            '<div class="warn">另有 <b>'
            f'{info["empty_plan_cases"]}</b> 条用例出现过「模型调用成功、'
            "但返回的计划不可用」（<code>steps</code> 为空）。"
            "这一类比调用失败更容易漏掉：<b>它不抛异常</b>。"
            "注意它<strong>不</strong>等于「模型不会规划」——"
            "思维链吃穿 <code>max_tokens</code> 时返回的也是空内容（预算问题）。</div>"
        )

    return table + note


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
    #
    # ⚠️ 它是**代理量**：数的是"有失败调用"的用例，不是"真的回落了"的用例。
    # 新报告另有观测值（`plans_by_source`），两者分开报 —— 见 `planner_breakdown`。
    planner_failed = sum(
        1 for r in (eval_payload.get("results") or []) if r.get("planner_failures")
    )
    planner_info = planner_breakdown(eval_payload)
    planner_fallback = int(planner_info["fallback_cases"])

    if total == 0:
        verdict = "没有用例，什么都没测。"
    elif calls == 0:
        verdict = (
            "这次跑批一次模型都没调用 —— 分数衡量的是框架的确定性逻辑，"
            "不是模型能力。要测模型请加 --provider / --model。"
        )
    elif failed_cases:
        verdict = (
            f"有 {failed_cases}/{total} 条用例没能跑起来（基础设施故障，不是"
            "\"NPC 没做到\"）。通过率是剩下的那些算出来的，这些用例不计入任何一行。"
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

    if planner_info["has_provenance"] and planner_info["planner_live"]:
        if planner_fallback:
            verdict += (
                f"　⚠️ 另有 {planner_fallback}/{total} 条的规划回落到启发式规划"
                "（按观测到的计划来源数的，不是按调用失败次数估的）——"
                "这些用例上的「模型规划」等于没开，"
                "拿它们做 planner 对照会得到「两组一样」的假结论。"
            )
    elif planner_failed:
        verdict += (
            f"　⚠️ 另有 {planner_failed}/{total} 条的规划调用失败并静默回落到了"
            "启发式规划 —— 这些用例上的「模型规划」等于没开，"
            "拿它们做 planner 对照会得到「两组一样」的假结论。"
            "（这份报告没有记录计划来源，所以这里用的是调用失败次数这个代理量，"
            "它不等于真正回落的条数：一条用例按 tick 多次规划，"
            "一次失败就足以让那个 tick 落回启发式。）"
        )

    # 天花板效应。**这一条不是关于"可不可信"，是关于"有没有信息量"。**
    # 一份 99.6% 的分数完全可以既可信又没用：它说明这套用例集对这个模型
    # 已经饱和，不再能区分「好」和「更好」。不写出来的话，读者很容易把
    # "99.6%" 读成"NPC 做得几乎完美"，而它实际的意思是"我们的卷子太简单了"。
    pass_rate = float(summary.get("pass_rate") or 0.0)
    saturated = total >= SATURATION_MIN_CASES and pass_rate >= SATURATION_PASS_RATE
    if saturated:
        verdict += (
            f"　⚠️ 但通过率 {pass_rate:.1%} 已经贴到天花板：这套用例集对这个模型"
            "已经饱和，它不再能区分「好」和「更好」。"
            # 两处必须用同一个精度：0.996 在 .0% 下会印成 "100%"，
            # 于是同一句话里出现 "通过率 99.6%" 和 "这里的 100%" 两个数 ——
            # 读者会以为在说两件事。同一个量，同一句话，只能有一个写法。
            f"这里的 {pass_rate:.1%} 不等于「NPC 做得很好」，"
            "只等于「这套回归集没抓到问题」。"
            "真正还有区分度的是下面的裁判维度（它测的是规则断言测不了的东西："
            "像不像人设、有没有真的回应、有没有编造）。"
            "要继续往前走，需要的是更难的自建用例去压规则指标，"
            "而不是继续跑同一张卷子。"
        )

    return {
        "total": total,
        "llm_calls": calls,
        "llm_failures": failures,
        "degraded_cases": degraded_cases,
        "failed_cases": failed_cases,
        "planner_failed_cases": planner_failed,
        "planner_fallback_cases": planner_fallback,
        "planner_has_provenance": bool(planner_info["has_provenance"]),
        "degraded_rate": round(degraded_cases / total, 3) if total else 0.0,
        "pass_rate": round(pass_rate, 4),
        "saturated": saturated,
        "trustworthy": bool(total and calls and not failed_cases
                            and degraded_cases / max(total, 1) <= 0.10
                            and not planner_failed and not planner_fallback),
        "verdict": verdict,
        "degraded_note": degraded.get("verdict", ""),
    }


def _trust_block(eval_payload: dict[str, Any]) -> str:
    trust = trust_summary(eval_payload)
    cls = "ok" if trust["trustworthy"] else "warn"
    # 有观测值就用观测值（真的回落了几条），没有才退回代理量并标出来。
    if trust["planner_has_provenance"]:
        fallback_tile = str(trust["planner_fallback_cases"])
        fallback_key = "规划回落"
    else:
        fallback_tile = f'{trust["planner_failed_cases"]}（代理量）'
        fallback_key = "规划回落"
    cells = [
        ("用例", str(trust["total"])),
        ("模型调用", str(trust["llm_calls"])),
        ("调用失败", str(trust["llm_failures"])),
        ("模板兜底", f'{trust["degraded_cases"]}（{trust["degraded_rate"]:.1%}）'),
        (fallback_key, fallback_tile),
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
    """校准 + 留出集，**两块必须分开摆**。

    它们回答的不是同一个问题：
      开发集上的 kappa —— 裁判和这套标准自洽吗？
      留出集上的 kappa —— 换一批没见过的样本，还准吗？

    只摆一个数（不管哪个），读者都会把它当成后者。所以差值也要算出来印上：
    **开发 kappa − 留出 kappa = 拟合的量。**
    """
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
    dev = (
        f"<h4>开发集（{cal.get('total', 0)} 条，被用来调过 rubric）</h4>"
        "<table><thead><tr><th>评判标准</th><th>样本</th><th>一致率</th>"
        "<th>kappa</th><th>结论</th>"
        "<th>真阳/真阴/假阳/假阴</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )

    holdout = judge_payload.get("holdout")
    if not holdout:
        return dev + (
            '<div class="warn"><strong>这份报告没有留出集结果。</strong>'
            "开发集上的 kappa 含拟合成分 —— 这三条 rubric 就是对着它改的。"
            "所以上面那个数只能说明「没有明显的系统性偏差」，"
            "不能当泛化能力引用。留出集在 "
            "<code>npc_agent/eval/calibration_holdout.jsonl</code>，"
            "跑 <code>judge</code>（默认会带上）就有了。</div>"
        )

    from .judge import contrast_rows

    hrows = []
    for row in contrast_rows(cal, holdout):
        hk, dk, gap = row["holdout_kappa"], row["dev_kappa"], row["gap"]
        hrows.append(
            "<tr>"
            f'<td class="name">{_esc(str(row["name"]))}</td>'
            f'<td class="num">{row["holdout_n"] or "—"}</td>'
            f'<td class="num"><strong>{"—" if hk is None else f"{hk:.2f}"}</strong></td>'
            f'<td class="num muted">{"—" if dk is None else f"{dk:.2f}"}</td>'
            f'<td class="num">{"—" if gap is None else f"{gap:+.2f}"}</td>'
            f'<td class="muted">{_esc(str(row["holdout_reading"]))}</td>'
            "</tr>"
        )
    table = (
        f"<h4>留出集（{holdout.get('total', 0)} 条，标签写好时没见过裁判输出）</h4>"
        "<table><thead><tr><th>评判标准</th><th>留出 n</th><th>留出 kappa</th>"
        "<th>开发 kappa</th><th>差值</th><th>结论</th></tr></thead>"
        f"<tbody>{''.join(hrows)}</tbody></table>"
        '<p class="muted">差值 = 开发 kappa − 留出 kappa，就是<b>拟合的量</b>。'
        "差得多不代表裁判差，只代表那个数不能再当泛化能力用。</p>"
    )

    if holdout.get("quotable"):
        seal = holdout.get("seal") or {}
        note = (
            '<div class="good"><strong>留出集封条校验通过</strong>'
            f'（样本摘要 <code>{_esc(str(seal.get("holdout_digest", "")))}</code>，'
            f'评分标准摘要 <code>{_esc(str(seal.get("rubric_digest", "")))}</code>）。'
            "这个 kappa 可以引用。<br>"
            "<b>但它仍然只是上界：</b>样本是手写的，比真实转写干净；"
            "每个维度只有十条上下，差一条 kappa 就动 0.1 量级。"
            "所以结论只能是「这个维度大致可用 / 不可用」，"
            "不能写成「裁判准确率 90%」。</div>"
        )
    else:
        problems = holdout.get("problems") or []
        items = "".join(
            f'<li><code>{_esc(str(p.get("kind", "")))}</code>：{_esc(str(p.get("detail", "")))}</li>'
            for p in problems
        )
        note = (
            '<div class="bad"><strong>⛔ 这份留出集的封条对不上，上面的 kappa 不可引用。</strong>'
            f"<ul>{items}</ul>"
            "封条破了是<b>不可修复</b>的：重封条只把破过的事实藏起来，"
            "并不会让「标签是改之前写的」重新成立。正确做法是另攒一份新的留出集。</div>"
        )
    return dev + table + note


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
            # neutral=True：裁判的通过率不能用规则指标那套绿/黄/红阈值 ——
            # 那会把 50%~72% 整片涂红，和上面绿油油的 1.000 并排摆着，
            # 读者会读成"这个模型不行"。两列量的不是同一个东西。
            f'<td class="num">{_bar(stats.get("pass_rate", 0.0), neutral=True)}</td>'
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
            f'<div class="tile"><div class="tile-k">判分重试</div>'
            f'<div class="tile-v">{coverage.get("judge_retries", 0)}</div></div>'
            "</div>"
            f'<div class="tile-note">{_esc(coverage.get("verdict", ""))}</div>'
        )
        reused = int(coverage.get("reused") or 0)
        if reused:
            coverage_note += (
                f'<div class="tile-note">其中 {reused} 条是从检查点复用的，'
                "没有重新调用模型 —— 这些判决来自上一次判分。</div>"
            )
        # 解析失败的重试单独报：它不是网络抖动，而是**裁判预算被思维链吃穿**。
        # 这个数只要不是 0，处置办法就是加预算 / 换模型，和查网络完全不同。
        parse_retries = int(coverage.get("judge_parse_retries") or 0)
        if parse_retries:
            coverage_note += (
                '<div class="warn">有 '
                f'<b>{parse_retries}</b> 次重试是因为<b>裁判返回的内容解析不了</b>'
                "（空内容 / 没有 score 字段）。这类失败通常是思维链把输出预算吃光"
                "（实测出现过 14440 字的思维链，是常规值的 3.5 倍）。"
                "重试能救回大部分，但根因是预算偏小 —— 下次开跑前把 "
                "<code>JUDGE_MAX_TOKENS</code> 调大。"
                "注意：改了预算会作废检查点（它在恢复关键字段里），"
                "所以这件事要在开跑前决定，不能等跑完一半。</div>"
            )

    return (
        f'<table><thead><tr><th>评判标准</th><th>通过</th><th>通过率</th></tr></thead>'
        f"<tbody>{body}</tbody></table>"
        '<div class="tile-note"><b>这一列的进度条是中性色，'
        "和上面的规则指标不是同一把尺子。</b>"
        "规则指标量的是「断言有没有过」，1.000 是常态；"
        "裁判量的是「另一个模型觉得像不像」，它更严，"
        "50%~72% 属于正常范围，<b>不代表失败</b>。"
        "两列不可相减，也不该比颜色。</div>"
        f"{coverage_note}"
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
  .good { background: #f2fbf4; border: 1px solid #c9e7d0; border-left: 4px solid #3f9e58;
          border-radius: 8px; padding: 12px 16px; margin-top: 12px; font-size: 13.5px;
          color: #1f5b31; }
  .bad  { background: #fdf3f3; border: 1px solid #f0cfcf; border-left: 4px solid #c0392b;
          border-radius: 8px; padding: 12px 16px; margin-top: 12px; font-size: 13.5px;
          color: #7d2a21; }
  .bad ul { margin: 8px 0 8px 18px; padding: 0; }
  h4 { margin: 20px 0 8px; font-size: 14.5px; color: #374151; }
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

  <h2>规划来源</h2>
  <p class="sub" style="margin:0 0 10px">
    计划是<strong>谁</strong>产出的。规划调用失败时框架会<strong>静默回落</strong>到启发式规划，
    两条路径产出的轨迹完全一样 —— 不看来源就分不出"模型规划生效了"和"模型压根没被问成"。
  </p>
  __PLANNER__

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
      <li><strong>kappa 是上界</strong>：开发集被用来改过 rubric，
          一致性里有一部分是拟合 —— 所以要看的是留出集那一列，
          而留出集也只有 32 条、每个维度十条上下，仍然不是泛化能力的证明。</li>
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
        # ⚠️ 两个条件缺一不可（`_planner_is_live` 是唯一实现）。
        # 只看配置的话，离线基线报告（`use_llm_planner` 默认 True、
        # `model=(offline)`）会印成「LLM 规划」—— 一句假话，
        # 而且它恰好是那份"对照组"报告最不该说的一句话。
        planner = "LLM 规划" if _planner_is_live(config) else "启发式规划"
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
        .replace("__PLANNER__", _planner_block(eval_payload))
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
