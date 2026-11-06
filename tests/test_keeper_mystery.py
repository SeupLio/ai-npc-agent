"""剧本杀场景的动态护栏 —— 破案率、路径覆盖与零泄密的批量断言。

## 这里在守什么

单局看运气，40 局看设计。这组用例把"剧本调到了健康区间"
钉成断言：

1. **解出率带宽**：0.4–0.9 之间。太低 = 门把信息掐死了
   （剧本太难），太高 = 门形同虚设（剧本太水）。两条都要
   被测试逮住 —— 这是把"信息充分性"从设计直觉变成数字的
   那条线；
2. **两条破案路径都被真实走到**：至少一局纯物证破案
   （零松口）、至少一局靠证词突破口破案。只有一条路径的
   剧本是单线程的演示，不是博弈；
3. **零泄密跨 seed**：40 局的 I1 全绿 —— 结构保证不挑运气；
4. **真凶的秘密死了**：f_where_gutang 是全剧本最深的秘密
   （held_secret、无世界解锁条件、持有者咬死），任何局里
   都不允许出现在任何成功交付里；
5. **谎言只从说谎者嘴里出来**：f_bluff_gutang 的每一次
   交付，来源都只能是顾棠自己 —— 别的 NPC 不会替他圆谎。
"""

from __future__ import annotations

from collections import Counter

import pytest

from npc_agent.keeper import KeeperSession, Scenario
from npc_agent.keeper.scenarios import (
    mystery_big_mouth_scenario,
    mystery_scenario,
)

N_SEEDS = 40


def _results(factory=mystery_scenario, n: int = N_SEEDS):
    return [
        KeeperSession(Scenario.from_dict(factory()), seed=s).run()
        for s in range(n)
    ]


def _evid_path_kind(result) -> str:
    evid = [l for l in result.transcript if l.kind == "evidence"]
    if not evid:
        return "none"
    if "没有发现" in evid[0].text:
        return "miss"
    if "书房" in evid[0].text:
        return "study"
    return "room"


class TestSolveRateBand:
    def test_solve_rate_in_healthy_band(self):
        """40 局解出率必须落在 0.4–0.9：两头都是设计缺陷。"""
        results = _results()
        solved = sum(1 for r in results if r.solved)
        rate = solved / len(results)
        assert 0.4 <= rate <= 0.9, f"解出率 {rate:.2f} 出了健康带"

    def test_audit_green_across_all_seeds(self):
        """零泄密 + 全不变量：40 局无一例外（结构保证不挑运气）。"""
        for r in _results():
            assert r.audit.ok, f"seed={r.seed} 审计失败: {r.audit.violations}"


class TestBothPathsExercised:
    def test_evidence_path_solves_without_any_crack(self):
        """书房路径：纯物证破案（全程无人松口）。"""
        results = _results()
        pure = [
            r for r in results
            if _evid_path_kind(r) == "study"
            and r.solved
            and r.counters["cracks"] == 0
        ]
        assert pure, "40 局里没有一局纯物证破案 —— 物证路径死了"

    def test_witness_path_solves_via_crack(self):
        """证词路径：靠压力撬开的口供配合物证定罪。"""
        results = _results()
        via_crack = [
            r for r in results
            if _evid_path_kind(r) == "room"
            and r.solved
            and r.counters["cracks"] > 0
        ]
        assert via_crack, "40 局里没有一局靠突破口定罪 —— 证词路径死了"

    def test_missed_search_is_unsolvable_by_design(self):
        """搜查失手 + 只有一张搜查令：这类局按设计不可解。"""
        results = _results()
        missed = [r for r in results if _evid_path_kind(r) == "miss"]
        if missed:  # 有失手局的 seed 才断言（概率路径）
            assert not any(r.solved for r in missed)


class TestDeepestSecretStaysDead:
    @pytest.mark.parametrize("factory", [mystery_scenario, mystery_big_mouth_scenario])
    def test_murderer_secret_never_delivered(self, factory):
        """真凶的秘密（下过楼）在任何模式、任何局里零成功交付。"""
        for r in _results(factory, n=20):
            moved = [
                d for d in r.audit.ledger_rows
                if d["fact_id"] == "f_where_gutang" and d["allowed"]
            ]
            assert moved == [], f"seed={r.seed} 秘密泄了: {moved}"

    def test_bluff_only_from_the_liar_himself(self):
        """谎版只能出自说谎者本人 —— 别人不会替顾棠圆谎。"""
        for r in _results(n=20):
            for d in r.audit.ledger_rows:
                if d["fact_id"] == "f_bluff_gutang" and d["allowed"]:
                    assert d["from_name"] == "顾棠"


class TestBatchStats:
    def test_run_batch_aggregates(self):
        from npc_agent.keeper.bench import run_batch

        _results, stats = run_batch(mystery_scenario(), seeds=list(range(6)))
        assert stats.runs == 6
        assert 0 <= stats.solved <= 6
        assert stats.audit_failures == 0
        assert stats.label  # 报告卡片要用的标签

    def test_big_mouth_batch_zero_leak(self):
        from npc_agent.keeper.bench import run_batch

        _rs, stats = run_batch(
            mystery_big_mouth_scenario(),
            seeds=list(range(6)),
            label="剧本杀·大嘴",
            mode="big-mouth",
        )
        assert stats.audit_failures == 0
        assert stats.blocked_total > 0  # 门在干活
