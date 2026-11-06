"""守密人引擎的批量跑批与报告 —— 指标、对照与 HTML 渲染。

## 跑批回答三个问题

1. **守得住吗**：零泄密不变量在所有 seed 上是否全过（必须 100%，
   这是结构保证 —— 出现一个失败就是引擎 bug，不是"偶发"）；
2. **破得了吗**：脚本侦探在门控下的破案率 —— 这个数字衡量的是
   **场景设计的信息充分性**（门放出来的信息够不够推理），
   不是模型能力。太低=剧本太难（信息不够），太高=剧本太水
   （门形同虚设），健康区间在中间；
3. **边界独立于行为吗**：对照"正常口径"与"大嘴口径"两个模式 ——
   两个模式的泄密数都必须是 0。大嘴模式是攻击者模型：说话人
   全面叛变，门仍守得住，这一对照才把"结构保证"与"策略自觉"
   区分开。

## 报告形态

沿用仓库报告的家规：数字 + 边界说明 + 快照（reports/ 下生成，
docs/ 提交一份固化快照）。所有结论**可复现**：seed 列表印在报告里。
"""

from __future__ import annotations

import html
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .session import KeeperResult, KeeperSession, Scenario


@dataclass
class BatchStats:
    scenario_id: str
    label: str
    runs: int = 0
    solved: int = 0
    avg_rounds: float = 0.0
    avg_facts: float = 0.0
    blocked_total: int = 0
    delivered_total: int = 0
    audit_failures: int = 0
    #: 大嘴/正常模式标记，报告对照用。
    mode: str = "normal"
    seeds: list[int] = field(default_factory=list)
    #: 企业场景：边界违规计数（PII 转移数、薪酬越界数）。必须为 0。
    pii_released: int = -1
    payroll_violations: int = -1
    #: 有没有"破案"概念（剧本杀 True，企业 False）。
    #: False 时破案率显示"—"而不是误导性的 0%。
    solvable: bool = True

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "场景": self.label,
            "模式": self.mode,
            "局数": self.runs,
            "破案率": (
                f"{(self.solved / self.runs):.0%}"
                if self.runs and self.solvable
                else "—"
            ),
            "平均轮数": f"{self.avg_rounds:.1f}",
            "人均到手事实": f"{self.avg_facts:.1f}",
            "门拦截": self.blocked_total,
            "转移总数": self.delivered_total,
            "审计失败": self.audit_failures,
        }
        if self.pii_released >= 0:
            row["PII 泄出"] = self.pii_released
        if self.payroll_violations >= 0:
            row["薪酬越界"] = self.payroll_violations
        return row


def run_batch(
    scenario_data: Mapping[str, Any],
    seeds: Iterable[int],
    *,
    label: str = "",
    mode: str = "normal",
) -> tuple[list[KeeperResult], BatchStats]:
    """跑一个场景的一批 seed。返回（逐局结果, 汇总）。"""
    scenario = Scenario.from_dict(scenario_data)
    results: list[KeeperResult] = []
    seeds_list = list(seeds)
    for seed in seeds_list:
        results.append(KeeperSession(scenario, seed=seed).run())
    stats = BatchStats(
        scenario_id=scenario.scenario_id,
        label=label or scenario.title,
        runs=len(results),
        solved=sum(1 for r in results if r.solved),
        avg_rounds=(
            statistics.fmean(r.rounds_used for r in results) if results else 0.0
        ),
        avg_facts=(
            statistics.fmean(len(r.asker_known) for r in results)
            if results
            else 0.0
        ),
        blocked_total=sum(r.counters.get("blocked_attempts", 0) for r in results),
        delivered_total=sum(
            r.audit.channel_counts.get("total", 0) for r in results
        ),
        audit_failures=sum(0 if r.audit.ok else 1 for r in results),
        mode=mode,
        seeds=seeds_list,
        solvable=scenario.solution is not None,
    )
    # 企业边界统计：按"事实类别 × 越界接收人"数。
    if scenario.solution is None:
        pii = 0
        payroll_violations = 0
        for r in results:
            for d in r.audit.ledger_rows:
                if not d["allowed"]:
                    continue
                fact = scenario.fact_by_id.get(d["fact_id"])
                if fact is None:
                    continue
                if fact.category == "pii":
                    pii += 1
                if (
                    fact.category == "payroll"
                    and d["to_id"] not in fact.audience
                ):
                    payroll_violations += 1
        stats.pii_released = pii
        stats.payroll_violations = payroll_violations
    return results, stats


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #
def _esc(value: Any) -> str:
    return html.escape(str(value), quote=False)


def transcript_text(result: KeeperResult) -> str:
    """一局对局的纯文本实录（console / 报告内嵌两用）。"""
    lines = []
    for line in result.transcript:
        prefix = ""
        if line.kind == "evidence":
            prefix = "[取证] "
        elif line.kind == "whisper":
            prefix = "[私语] "
        lines.append(f"第{line.round_no}轮 | {line.speaker} | {prefix}{line.text}")
    return "\n".join(lines)


def render_batch_html(
    results: list[KeeperResult],
    stats: list[BatchStats],
    *,
    title: str = "守密人引擎跑批报告",
    sample_seed: Optional[int] = None,
) -> str:
    """守密人跑批的 HTML 报告。自包含（内联 CSS），风格沿用现有报告。"""
    rows = "".join(
        "<tr>"
        + "".join(f"<td>{_esc(v)}</td>" for v in s.as_row().values())
        + "</tr>"
        for s in stats
    )
    header = "".join(f"<th>{_esc(k)}</th>" for k in stats[0].as_row().keys())
    stat_blocks = "".join(_stat_block(s) for s in stats)
    total_runs = sum(s.runs for s in stats)
    all_seeds = sorted({r.seed for r in results})
    seeds_range = (
        f"{all_seeds[0]}–{all_seeds[-1]}" if all_seeds else "（无）"
    )

    sample = results[0] if results else None
    if sample_seed is not None:
        sample = next((r for r in results if r.seed == sample_seed), results[0] if results else None)
    sample_html = ""
    if sample is not None:
        audit = sample.audit
        inv_rows = "".join(
            f'<tr><td>{_esc(i.name)}</td>'
            f'<td class="{"ok" if i.passed else "bad"}">{"通过" if i.passed else "破坏"}</td>'
            f"<td>{_esc(i.detail)}</td></tr>"
            for i in audit.invariants
        )
        sample_html = f"""
        <h2>样例对局（seed={sample.seed}）</h2>
        <pre class="transcript">{_esc(transcript_text(sample))}</pre>
        <h3>信息流审计（五不变量）</h3>
        <table><thead><tr><th>不变量</th><th>判定</th><th>说明</th></tr></thead>
        <tbody>{inv_rows}</tbody></table>
        <h3>转移台账（前 24 条）</h3>
        <table><thead><tr><th>#</th><th>事实</th><th>从</th><th>到</th><th>通道</th><th>轮</th><th>判定</th></tr></thead>
        <tbody>{_ledger_rows(sample, 24)}</tbody></table>
        """
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{_esc(title)}</title>
<style>
  body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
         margin: 0 auto; max-width: 960px; padding: 2rem 1rem; color: #1f2430;
         line-height: 1.6; }}
  h1 {{ font-size: 1.5rem; }} h2 {{ font-size: 1.15rem; margin-top: 2rem; }}
  h3 {{ font-size: 1rem; margin-top: 1.4rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
  th, td {{ border: 1px solid #d9dee8; padding: 5px 8px; text-align: left; }}
  th {{ background: #f2f4f8; }}
  td.ok, .ok {{ color: #0a7d32; font-weight: 600; }}
  td.bad, .bad {{ color: #b3261e; font-weight: 600; }}
  pre.transcript {{ background: #f6f7fa; border: 1px solid #e3e6ee; padding: 12px;
                    font-size: 0.8rem; white-space: pre-wrap; border-radius: 6px; }}
  .note {{ background: #fffbe8; border-left: 3px solid #d4a017; padding: 8px 12px;
           font-size: 0.85rem; margin: 1rem 0; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .card {{ border: 1px solid #d9dee8; border-radius: 8px; padding: 10px 14px; min-width: 140px; }}
  .card .n {{ font-size: 1.5rem; font-weight: 700; }}
</style></head><body>
<h1>{_esc(title)}</h1>
<p>共 <strong>{total_runs}</strong> 局对局 · 零模型 · 零网络 · seed 列表：{seeds_range}</p>
<div class="cards">{stat_blocks}</div>
<h2>汇总</h2>
<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>
<div class="note">零泄密是<b>结构保证</b>：大嘴模式（说话人全面叛变）下泄出数仍为 0 才算数。
破案率衡量的是场景的信息充分性，不是模型能力。全部数字按报告内 seed 列表可复现。</div>
{sample_html}
</body></html>"""


def _stat_block(s: BatchStats) -> str:
    if s.solvable and s.runs:
        solve = f"{(s.solved / s.runs):.0%}"
    else:
        solve = "—"  # 没有破案概念的场景：别显示误导性的 0%
    return (
        f'<div class="card"><div class="n">{_esc(solve)}</div>'
        f'<div>{_esc(s.label)}<br><small>{_esc(s.mode)} · {s.runs} 局</small></div></div>'
    )


def _ledger_rows(result: KeeperResult, limit: int) -> str:
    rows = []
    for d in result.audit.ledger_rows[:limit]:
        cls = "ok" if d["allowed"] else "bad"
        verdict = "放行" if d["allowed"] else "拦截"
        rows.append(
            f'<tr><td>{d["seq"]}</td><td>{_esc(d["fact_content"][:40])}</td>'
            f'<td>{_esc(d["from_name"])}</td><td>{_esc(d["to_name"])}</td>'
            f'<td>{_esc(d["channel"])}</td><td>{d["round_no"]}</td>'
            f'<td class="{cls}">{verdict}</td></tr>'
        )
    return "".join(rows)


def write_report(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


__all__ = [
    "BatchStats",
    "render_batch_html",
    "run_batch",
    "transcript_text",
    "write_report",
]
