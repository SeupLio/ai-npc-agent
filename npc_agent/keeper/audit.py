"""守密人引擎的审计层 —— 独立重放台账，复核对局的全部不变量。

## 审计为什么必须"独立"

session 跑完时可以宣称"我零泄密"，但这句话只有**别人**复核过才算数。
审计器不复用 session 的任何结论 —— 它只信两样东西：

1. 台账（ledger）—— 每次转移记下的 fact/通道/双方/当时世界标记快照；
2. 门（kb.cleared）—— 用**当时**的标记快照重放判定。

重放结果和台账记录不一致，就是不变量被破坏。这个结构有一个重要
推论：**审计器自己必须可测**（tests 里有"篡改台账必被抓"的用例）——
一个永远说 OK 的审计器比没有审计器更糟。

## 五条不变量

    I1 零泄密    台账里每笔 allowed 的转移，重放门判定都应放行
    I2 知识单调  重放重建的知识态，每一步只增不减
    I3 无中生有  提问方最终知道的每条事实，要么是初始设定、要么有台账来源
    I4 秘密已死  谎言盖着的话题事实，从未发生任何一次成功转移
    I5 台账完整  seq 从 1 连续无缺口（防"删记录洗白"）

I2 与 I4 是用**重建**而不是记录来检查的：从空知识态出发重放 allowed
转移，如果重建结果和 kb.known 不一致，说明状态被绕过门改过。
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..env.conditions import ConditionContext
from .knowledge import KnowledgeBase

if TYPE_CHECKING:  # 只为类型标注，运行时零依赖（也解开与 session 的环）
    from .session import KeeperSession


@dataclass
class Invariant:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class AuditReport:
    """一场对局的审计结论。渲染方法只做展示，判定都在 audit_run 里。"""

    scenario_id: str
    seed: int
    invariants: list[Invariant] = field(default_factory=list)
    #: 按 actor 对账：初始种子数 / 转移获得数 / 最终知识数。
    actor_rows: list[dict[str, Any]] = field(default_factory=list)
    #: 台账摘要（转移次数/拦截次数，按通道）。
    channel_counts: dict[str, int] = field(default_factory=dict)
    #: 渲染用的台账行（含名字与事实内容，报告直接吃）。
    ledger_rows: list[dict[str, Any]] = field(default_factory=list)
    #: 违规明细（人类可读，报告里逐条列出）。
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(i.passed for i in self.invariants)

    @property
    def zero_leak(self) -> bool:
        return all(i.passed for i in self.invariants)

    def summary_line(self) -> str:
        state = "全部通过" if self.zero_leak else f"{len(self.violations)} 项违规"
        return (
            f"[{self.scenario_id}#{self.seed}] 不变量：{state}；"
            f"转移 {self.channel_counts.get('total', 0)} 次"
            f"（拦截 {self.channel_counts.get('blocked', 0)} 次）"
        )

    # ------------------------------------------------------------------ #
    # 渲染
    # ------------------------------------------------------------------ #
    def render_text(self, kb: KnowledgeBase) -> str:
        lines = [self.summary_line(), "不变量明细："]
        for inv in self.invariants:
            mark = "✓" if inv.passed else "✗"
            lines.append(f"  {mark} {inv.name}：{inv.detail}")
        if self.violations:
            lines.append("违规明细：")
            lines.extend(f"  - {v}" for v in self.violations)
        return "\n".join(lines)

    def render_html(self, kb: KnowledgeBase, title: str = "") -> str:
        head = html.escape(title or f"守密人审计 · {self.scenario_id} #{self.seed}")
        inv_rows = "".join(
            f'<tr class="{"ok" if i.passed else "bad"}">'
            f"<td>{html.escape(i.name)}</td>"
            f"<td>{'通过' if i.passed else '违规'}</td>"
            f"<td>{html.escape(i.detail)}</td></tr>"
            for i in self.invariants
        )
        chan_rows = "".join(
            f"<tr><td class=\"mono\">{html.escape(k)}</td>"
            f'<td class="num">{v}</td></tr>'
            for k, v in sorted(self.channel_counts.items())
        )
        actor_rows = "".join(
            f"<tr><td>{html.escape(r['actor'])}</td>"
            f'<td class="num">{r['seeded']}</td>'
            f'<td class="num">{r['gained']}</td>'
            f'<td class="num">{r['final']}</td></tr>'
            for r in self.actor_rows
        )
        bad = "" if self.zero_leak else "".join(
            f'<li class="bad">{html.escape(v)}</li>' for v in self.violations
        )
        bad_block = f"<h3>违规</h3><ul>{bad}</ul>" if bad else ""
        return (
            "<!DOCTYPE html><html lang=\"zh\"><head><meta charset=\"utf-8\">"
            f"<title>{head}</title><style>"
            "body{font-family:system-ui,'PingFang SC',sans-serif;margin:2rem;"
            "color:#1c2733;line-height:1.6}"
            "h1{font-size:1.3rem}h3{font-size:1rem;margin:1.2rem 0 .4rem}"
            "table{border-collapse:collapse;margin:.4rem 0;font-size:.9rem}"
            "td,th{border:1px solid #d7dee6;padding:.3rem .6rem;text-align:left}"
            "th{background:#eef2f6}.num{text-align:right}"
            ".mono{font-family:ui-monospace,monospace}"
            "tr.ok td{background:#f0f9f1}tr.bad td{background:#fdeeee}"
            "li.bad{color:#a33}.summary{color:#4a5b6d}"
            "</style></head><body>"
            f"<h1>{head}</h1>"
            f"<p class=\"summary\">{html.escape(self.summary_line())}</p>"
            f"<h3>不变量</h3><table><tr><th>不变量</th><th>判定</th>"
            f"<th>说明</th></tr>{inv_rows}</table>"
            "<h3>通道计数</h3><table><tr><th>通道</th><th>次数</th>"
            f"</tr>{chan_rows}</table>"
            "<h3>知识对账（按角色）</h3><table><tr><th>角色</th>"
            f"<th>初始</th><th>转移获得</th><th>最终</th></tr>{actor_rows}</table>"
            f"{bad_block}"
            "</body></html>"
        )


def audit_run(kb: KnowledgeBase, session: KeeperSession) -> AuditReport:
    """重放台账，复核全部不变量。session 只提供场景种子，不提供结论。"""
    scenario = session.scenario
    report = AuditReport(scenario_id=scenario.scenario_id, seed=session.seed)

    # 初始种子（审计重建的起点）：NPC 的 knows + 谎版（说谎者得
    # 记得自己编过什么，strategy 在构造时种进 kb）+ 提问方的
    # asker_knows。种子对不上，I2/I3 就会把正常的谎言账目误报成
    # "无中生有"。
    seeds: dict[str, set[str]] = {}
    for spec in scenario.npcs:
        bucket = set(spec.knows)
        bucket.update(spec.bluffs.values())
        seeds[spec.npc_id] = bucket
    seeds[scenario.asker_id] = set(scenario.asker_knows)

    # I5 台账完整：seq 连续。
    seqs = [d.seq for d in kb.ledger]
    seq_ok = seqs == list(range(1, len(seqs) + 1))
    report.invariants.append(
        Invariant("I5 台账完整", seq_ok,
                  f"seq 1..{len(seqs)}" if seq_ok else "seq 有缺口或乱序")
    )

    # I1 零泄密 + 重复交付计数（一次重放同时收集）。
    # 重复交付（听者已知又收一遍）不是违规 —— 集合语义天然幂等 ——
    # 但要计数：一场对局重复率奇高说明提问方策略在空转。
    rebuilt: dict[str, set[str]] = {k: set(v) for k, v in seeds.items()}
    duplicates = 0
    leak_violations: list[str] = []
    for d in kb.ledger:
        ctx = ConditionContext(flags=frozenset(d.flags_at_time))
        expect = kb.cleared(
            d.fact_id, d.to_id, ctx, channel=d.channel, from_id=d.from_id
        )
        if d.allowed and not expect:
            leak_violations.append(
                f"#{d.seq} {d.fact_id} {d.from_id}->{d.to_id}"
                f"（{d.channel}，重放判定=拒绝）"
            )
        if d.allowed:
            bucket = rebuilt.setdefault(d.to_id, set())
            if d.fact_id in bucket:
                duplicates += 1
            bucket.add(d.fact_id)
    report.invariants.append(
        Invariant(
            "I1 零泄密",
            not leak_violations,
            f"重放 {len(kb.ledger)} 笔，泄漏 {len(leak_violations)} 笔"
            + (f"，重复交付 {duplicates} 笔（幂等，非违规）" if duplicates else ""),
        )
    )
    report.violations.extend(leak_violations)

    # I2：重放重建态与现场知识态必须一致（不一致 = 状态被绕过门改过）。
    # 知识层从构造上只增不减（没有撤销 API），单调性由重建一致性
    # 间接保证 —— 重建过程本身只做并集。
    consistent = all(
        rebuilt.get(actor, set()) == kb.known.get(actor, set())
        for actor in set(rebuilt) | set(kb.known)
    )
    report.invariants.append(
        Invariant("I2 重放一致（知识只增不减）", consistent,
                  "重建态=现场态" if consistent else "重放重建与现场知识态不一致")
    )

    # I3 无中生有：提问方最终知识 ⊆ 初始 ∪ 台账来源。
    asker_gained = {
        d.fact_id for d in kb.ledger
        if d.allowed and d.to_id == scenario.asker_id
    }
    final_known = kb.known_by(scenario.asker_id)
    orphan = final_known - seeds.get(scenario.asker_id, set()) - asker_gained
    report.invariants.append(
        Invariant("I3 无中生有", not orphan,
                  f"无来源事实 {len(orphan)} 条"
                  + (f"：{sorted(orphan)}" if orphan else ""))
    )
    if orphan:
        report.violations.extend(f"无来源：{f}" for f in sorted(orphan))

    # I4 秘密已死：谎言盖着的话题事实从未成功转移。
    dead_ok = True
    dead_notes: list[str] = []
    for spec in scenario.npcs:
        for covered in spec.bluffs:
            moved = [
                d for d in kb.ledger
                if d.allowed and d.fact_id == covered
            ]
            if moved:
                dead_ok = False
                dead_notes.append(f"{spec.name} 的被盖话题 {covered} 泄出了 {len(moved)} 次")
    report.invariants.append(
        Invariant("I4 被盖秘密从未转移", dead_ok,
                  "；".join(dead_notes) or "所有谎版覆盖的话题零转移")
    )
    report.violations.extend(dead_notes)

    # 汇总表。
    for actor in sorted(set(rebuilt) | set(kb.known)):
        report.actor_rows.append(
            {
                "actor": scenario.npc_names().get(actor, actor),
                "seeded": len(seeds.get(actor, ())),
                "gained": len(rebuilt.get(actor, set()) - seeds.get(actor, set())),
                "final": len(kb.known.get(actor, set())),
            }
        )
    counts: dict[str, int] = {"total": 0, "blocked": 0}
    names = scenario.npc_names()
    contents = scenario.fact_contents()
    for d in kb.ledger:
        counts["total"] += 1
        if d.allowed:
            counts[d.channel] = counts.get(d.channel, 0) + 1
        else:
            counts["blocked"] += 1
        report.ledger_rows.append(
            {
                "seq": d.seq,
                "fact_id": d.fact_id,
                "fact_content": contents.get(d.fact_id, d.fact_id),
                "from_id": d.from_id,
                "to_id": d.to_id,
                "from_name": names.get(d.from_id, d.from_id),
                "to_name": names.get(d.to_id, d.to_id),
                "channel": d.channel,
                "round_no": d.round_no,
                "allowed": d.allowed,
                "via": d.via,
            }
        )
    report.channel_counts = counts
    return report


__all__ = [
    "AuditReport",
    "Invariant",
    "audit_run",
]
