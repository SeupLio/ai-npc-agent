"""评测的分辨率：**剂量-反应曲线** —— 掉多少，评测才开始动？

## 为什么需要它（和 `sensitivity.py` 的分工）

`sensitivity.py` 是**二值**的：注入一个缺陷，问"抓到了没有"，答案是 **6/6**。
这个读数看起来像"评测很灵敏"，但它其实**什么也没说** —— 只要掉一个点就算抓到。

> 一个只能发现"整个模块被拔掉"的评测，和一个能发现"计划少执行了一步"的评测，
> 在 "6/6" 这个读数上**长得一模一样**。

所以这里问的是另一个问题：**掉多少，评测才开始动？** 做法是把每个计划里的
步骤按比例 k 丢掉（k 从 0 到 1），测通过率随 k 怎么变。曲线离开 1.000 的那个 k，
就是这套评测的**检出阈值** —— 它是"评测分辨率"的直接读数。

**这不是 `sensitivity` 的替代，是它的补充**：`sensitivity` 回答"评测会不会报警"，
这里回答"评测的报警线画在哪"。

## 为什么用"丢步骤"作为剂量

因为它**连续可调**、又**贴着真实故障**：

* k = 0 → 原样，基线（实测 235/235）；
* k = 1 → 计划全丢，**等价于 `install_planning_disabled`**。
  这是一个免费的自洽性检验：如果这条曲线在 k=1 处不落在 `sensitivity` 里
  `planning_disabled` 那个读数上（59.1%），说明两个剂量口径不是一回事。
  实测 **0.5915 vs 0.591**，对得上。
* 中间那些点才是这份报告的价值 —— 二值测试**结构上取不到**它们。

## 剂量怎么施加

**确定性均匀抽样**，不是随机丢：计划有 n 步时保留 m = ceil((1-k)·n) 步，
下标取 `floor(j·n/m)`。这样：

* 同样的 k 每次得到同样的计划（可复现 —— 本项目要求离线路径必须确定性）；
* 丢掉的步骤**分散在计划各处**，而不是固定砍尾巴 ——
  否则 k=0.2 永远等于"丢掉最后那个改变世界的步骤"，曲线会退化成阶跃。

## ⚠️ 这条曲线给的是**下界**

均匀丢步 **不等于** 真实的模型退化：真实模型更可能在**难的那几步**上失败
（最后那个改世界状态的步骤），所以真实退化会被评测**更早**看见。
换句话说，这里量出来的是**最不利情形下的灵敏度**，是个下界不是估计值。
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Optional

from ..agent import NPCAgent
from ..config import RuntimeConfig
from .harness import EvalHarness, EvalReport
from .sensitivity import DIMENSIONS

#: 要扫的剂量（丢掉计划步骤的比例）。0.1 的步长够看出阈值在哪一格。
DOSES: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def _subsample(n: int, k: float) -> list[int]:
    """从 n 个步骤里按比例 k 丢掉之后，**保留**哪些下标（确定性、均匀）。"""
    if n <= 0:
        return []
    m = math.ceil((1.0 - k) * n)
    if m >= n:
        return list(range(n))
    if m <= 0:
        return []
    return sorted({min(n - 1, int(j * n / m)) for j in range(m)})


def install_graded_skip(k: float) -> Callable[[], None]:
    """把计划里的步骤丢掉 k 比例。k=1 等价于"不做任何规划"。

    与 `sensitivity.install_*` 同形：返回一个 restore 回调，
    **调用方必须放在 `finally` 里**，否则下一个剂量会叠在它上面。
    """
    original = NPCAgent._make_plan

    def make_plan_truncated(
        self: Any,
        utterance: Any,
        decision: Any,
        memories: list[Any],
    ) -> Optional[Any]:
        plan = original(self, utterance, decision, memories)
        if plan is None or not plan.steps:
            return plan
        keep = _subsample(len(plan.steps), k)
        if not keep:
            # 一步都不剩 —— 给 None，与 `install_planning_disabled` 同形，
            # 这样 k=1 才能和那个变异直接对账。
            return None
        if len(keep) == len(plan.steps):
            return plan
        return dataclasses.replace(plan, steps=[plan.steps[i] for i in keep])

    NPCAgent._make_plan = make_plan_truncated  # type: ignore[method-assign]

    def restore() -> None:
        NPCAgent._make_plan = original  # type: ignore[method-assign]

    return restore


def _run(config: RuntimeConfig) -> EvalReport:
    """跑一遍全部用例（离线）。"""
    harness = EvalHarness(config)
    report = EvalReport()
    for case in harness.load_cases(None):
        report.results.append(harness.run_case(case))
    return report


def sweep(config: Optional[RuntimeConfig] = None) -> dict[str, Any]:
    """跑完整条剂量-反应曲线，返回可直接序列化的结果。"""
    cfg = config or RuntimeConfig()
    points: list[dict[str, Any]] = []

    for k in DOSES:
        restore = install_graded_skip(k)
        try:
            report = _run(cfg)
        finally:
            restore()
        means = report.metric_means()
        points.append(
            {
                "k": round(k, 3),
                "passed": report.passed,
                "total": report.total,
                "pass_rate": (
                    round(report.passed / report.total, 6) if report.total else 0.0
                ),
                "metric_means": {d: round(means.get(d, 0.0), 6) for d in DIMENSIONS},
            }
        )

    baseline = points[0]["pass_rate"] if points else 0.0
    threshold = next((p["k"] for p in points if p["pass_rate"] < baseline - 1e-9), None)
    #: 整条曲线上**始终没动过**的维度。这些维度结构上看不见这次退化 ——
    #: 它比"覆盖率 52%"更硬：覆盖不足是"可能没测到"，
    #: 恒为 1.000 是"**确实**没测到"。
    blind = [
        d
        for d in DIMENSIONS
        if all(abs(p["metric_means"][d] - baseline_means(d, points)) < 1e-9 for p in points)
    ]
    return {
        "doses": list(DOSES),
        "points": points,
        "baseline_pass_rate": baseline,
        "total_cases": points[0]["total"] if points else 0,
        "detection_threshold_k": threshold,
        "blind_dimensions": blind,
    }


def baseline_means(dim: str, points: list[dict[str, Any]]) -> float:
    """这个维度在 k=0 时的值 —— 用来判断它在整条曲线上有没有动过。"""
    return points[0]["metric_means"][dim] if points else 0.0


def render_resolution(result: dict[str, Any], console: Any = None) -> None:
    """把曲线印到终端。"""
    dims = "  ".join(f"{d:>11}" for d in DIMENSIONS)
    lines = [
        f"用例 {result['total_cases']} 条｜基线 {result['baseline_pass_rate']:.4f}"
        "｜离线、0 次模型调用",
        "",
        f"  {'k':<4}  {'通过':>7}  {'通过率':>8}  {'Δ通过率':>8}  {dims}",
        "  " + "-" * (len(dims) + 44),
    ]
    for p in result["points"]:
        delta = p["pass_rate"] - result["baseline_pass_rate"]
        cells = "  ".join(f"{p['metric_means'][d]:>11.4f}" for d in DIMENSIONS)
        flag = "  ← 开始掉分" if delta < -1e-9 else ""
        lines.append(
            f"  {p['k']:<4.1f}  {p['passed']:>3}/{p['total']:<3}  "
            f"{p['pass_rate']:>8.4f}  {delta:>+8.4f}  {cells}{flag}"
        )
    lines.append("")
    threshold = result["detection_threshold_k"]
    if threshold is None:
        lines.append("检出阈值：**在 0~1 全程都没掉分** —— 这套评测对这条退化路径不敏感。")
    else:
        lines.append(
            f"检出阈值：丢掉 ≥ **{threshold:.0%}** 的计划步骤，评测才开始掉分。"
        )
        lines.append("          （更小的退化在全部用例上都是满分 —— 这就是分辨率的上限。）")
    if result["blind_dimensions"]:
        lines.append("")
        lines.append(
            "整条曲线上**恒为 1.000 的维度**（结构上看不见这次退化）："
            + "、".join(result["blind_dimensions"])
        )
        lines.append("          ⚠️ 这不是「这些维度不重要」，是「它们测不到规划」——")
        lines.append("             它们各有各的靶子，但**不能拿来当规划的证据**。")
    text = "\n".join(lines)
    if console is not None:
        console.print(text)
    else:
        print(text)


def render_resolution_html(result: dict[str, Any]) -> str:
    """HTML 版。**首行必须能被 `report_index.case_count` 解析出条数。**"""
    rows = []
    for p in result["points"]:
        delta = p["pass_rate"] - result["baseline_pass_rate"]
        cells = "".join(f"<td class='num'>{p['metric_means'][d]:.3f}</td>" for d in DIMENSIONS)
        cls = " class='hit'" if delta < -1e-9 else ""
        rows.append(
            f"<tr{cls}><td class='num'>{p['k']:.1f}</td>"
            f"<td class='num'>{p['passed']}/{p['total']}</td>"
            f"<td class='num'>{p['pass_rate']:.3f}</td>"
            f"<td class='num'>{delta:+.3f}</td>{cells}</tr>"
        )
    dims = "".join(f"<th>{d}</th>" for d in DIMENSIONS)
    threshold = result["detection_threshold_k"]
    if threshold is None:
        verdict = "在 0~1 全程都没掉分 —— 这套评测对这条退化路径<strong>不敏感</strong>。"
    else:
        verdict = (
            f"丢掉 ≥ <strong>{threshold:.0%}</strong> 的计划步骤，评测才开始掉分；"
            "更小的退化在全部用例上都是满分。"
        )
    blind = result["blind_dimensions"]
    blind_html = (
        "<div class='warn'><strong>整条曲线上恒为 1.000 的维度："
        + "、".join(f"<code>{d}</code>" for d in blind)
        + "</strong><br>这不是「这些维度不重要」—— 它们各有各的靶子。"
        "但它意味着：<strong>这 " + str(len(blind)) + " 个维度不能拿来当「规划没问题」的证据</strong>，"
        "哪怕把规划整个拔掉（k=1.0），它们照样满分。"
        "覆盖率不足是「可能没测到」，恒为 1.000 是「<strong>确实</strong>没测到」。</div>"
        if blind
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>评测的分辨率：剂量-反应曲线</title>
<style>
body{{font-family:-apple-system,'Segoe UI',sans-serif;max-width:1000px;margin:40px auto;padding:0 24px;color:#1a1a1a;line-height:1.6}}
h1{{font-size:23px;margin-bottom:6px}}
.sub{{color:#666;font-size:14px;margin-bottom:22px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid #e3e3e3;padding:6px 9px;text-align:right}}
th{{background:#fafafa;font-weight:600}}
td:first-child,th:first-child{{text-align:left}}
td.num{{font-variant-numeric:tabular-nums}}
tr.hit td{{background:#fff4f4}}
.warn{{background:#fffbe6;border-left:3px solid #e8b400;padding:12px 16px;margin:22px 0;font-size:14px}}
.note{{background:#f4f8ff;border-left:3px solid #4a7fd4;padding:12px 16px;margin:22px 0;font-size:14px}}
code{{background:#f2f2f2;padding:1px 5px;border-radius:3px}}
</style></head><body>
<h1>评测的分辨率：把计划砍掉多少，评测才开始掉分</h1>
<div class="sub">共 {result['total_cases']} 条用例｜基线 {result['baseline_pass_rate']:.4f}
｜<strong>离线、确定性、0 次模型调用</strong>。红色行 = 该剂量下评测开始掉分。</div>

<div class="note"><strong>这份报告回答的是 <code>sensitivity.html</code> 回答不了的问题。</strong>
那一份是二值的（注入缺陷 → 抓到没有 = 6/6），只要掉一个点就算抓到，
所以「一个能发现整个模块被拔掉的评测」和「一个能发现计划少走一步的评测」
在 6/6 这个读数上长得一模一样。这里把剂量做成连续的，量出<strong>报警线画在哪</strong>。</div>

<table><thead><tr><th>丢掉计划步骤比例 k</th><th>通过</th><th>通过率</th><th>Δ</th>{dims}</tr></thead>
<tbody>{''.join(rows)}</tbody></table>

<div class="warn"><strong>结论。</strong>{verdict}<br>
k=1.0 是这条曲线的一个<strong>免费自洽性检验</strong>：它应当等于
<code>sensitivity.html</code> 里 <code>planning_disabled</code> 的读数。
两处对得上，说明「丢光计划」和「不做规划」是同一个口径。</div>

{blind_html}

<div class="note"><strong>⚠️ 这条曲线给的是<strong>下界</strong>，不是估计值。</strong>
剂量是<strong>均匀</strong>丢步，而真实模型的退化更可能集中在<strong>难的那几步</strong>上
（最后那个改世界状态的步骤）。所以真实退化会被评测<strong>更早</strong>看见 ——
这里量出来的是最不利情形下的灵敏度。</div>
</body></html>"""
