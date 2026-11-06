"""守密人对局循环的护栏 —— 可复现性、门控行为、资源约束。

## 这里在守什么

1. **同 seed 同对局**：转录逐行一致。这是"破案率是可复现数字"
   的前提 —— 跑批报告的全部说服力都压在这条上；
2. **门控行为在会话层的表现**：搜查失手（世界状态与"没搜到"
   自洽）、listener 感知（同一话题换个人问，结果不同）；
3. **资源约束**：搜查令张数封顶、轮次封顶 —— 提问方不能靠
   无限取证把门泡开。

顺带钉住两处容易被"顺手重构"破坏的行为：企业场景 solved 恒为
None（没有破案概念）；搜查失手的局，条件门保持关闭。
"""

from __future__ import annotations

from npc_agent.keeper import KeeperSession, Scenario
from npc_agent.keeper.scenarios import (
    mystery_scenario,
    workplace_scenario,
)


def _mystery(seed: int = 0) -> KeeperSession:
    return KeeperSession(Scenario.from_dict(mystery_scenario()), seed=seed)


class TestReproducibility:
    def test_same_seed_same_transcript(self):
        """同 seed 两局，转录逐行一致（含取证与私语）。"""
        r1 = _mystery(seed=7).run()
        r2 = _mystery(seed=7).run()
        assert [(l.round_no, l.speaker, l.text) for l in r1.transcript] == [
            (l.round_no, l.speaker, l.text) for l in r2.transcript
        ]
        assert r1.solved == r2.solved
        assert r1.counters == r2.counters

    def test_seeds_vary(self):
        """不同 seed 至少有一局走向不同（方差来自设计，不是缺陷）。"""
        texts = set()
        for seed in range(8):
            r = _mystery(seed=seed).run()
            texts.add(tuple(l.text for l in r.transcript))
        assert len(texts) > 1


class TestEvidenceActions:
    def test_evidence_limit_respected(self):
        """一张搜查令：整场至多一次取证动作。"""
        r = _mystery().run()
        assert r.counters["evidence_actions"] <= 1

    def test_missed_search_keeps_gate_closed(self):
        """搜查失手：转录记"没有发现"，且条件门不解锁（flag 未设）。"""
        session = _mystery()
        # 把两个动作的命中率都钉到 0，保证失手路径必然走到。
        for action in session.scenario.evidence_actions:
            action.yield_chance = 0.0
        r = session.run()
        missed = [l for l in r.transcript if "没有发现" in l.text]
        assert missed, "钉死命中率后必然出现一次失手记录"
        assert "searched_study" not in session.flags
        assert "searched_room" not in session.flags
        assert r.audit.ok

    def test_successful_search_unlocks_facts(self):
        """搜查命中：物证事实进侦探知识，走 evidence 通道。"""
        session = _mystery()
        for action in session.scenario.evidence_actions:
            action.yield_chance = 1.0
            # 只留书房一条路，排除选择方差。
            action_id = "search_study"
        session.scenario.evidence_actions = [
            a for a in session.scenario.evidence_actions
            if a.action_id == "search_study"
        ]
        r = session.run()
        assert "f_muddy_boots" in r.asker_known
        evidence_deliveries = [
            d for d in r.audit.ledger_rows
            if d["channel"] == "evidence" and d["allowed"]
        ]
        assert evidence_deliveries


class TestWorkplaceSession:
    def test_no_solution_no_accusation(self):
        """企业场景没有破案概念：solved 恒 None，转录无指控。"""
        session = KeeperSession(
            Scenario.from_dict(workplace_scenario()), seed=0
        )
        r = session.run()
        assert r.solved is None
        assert not any("指控" in l.text for l in r.transcript)
        assert r.audit.ok

    def test_script_exhausts_and_finishes(self):
        """问询脚本跑完后干净收尾（一次，不重复）。"""
        session = KeeperSession(
            Scenario.from_dict(workplace_scenario()), seed=3
        )
        n_script = len(session.scenario.script)
        r = session.run()
        finishes = [l for l in r.transcript if l.text == "问询脚本执行完毕。"]
        assert len(finishes) == 1
        assert r.counters["questions"] == n_script


class TestRoundsBound:
    def test_rounds_capped(self):
        session = _mystery()
        session.scenario.rounds = 3
        r = session.run()
        assert r.rounds_used <= 3

    def test_listener_gating_in_session(self):
        """同一话题、不同提问人，结果不同：门按"这一问是谁问的"判。

        用大嘴变体做：说话人必然提议全量交付，于是"拦没拦"
        完全是门的判定 —— 正常口径下策略层先收住了（sens 80
        的薪酬从不提议），门没机会出场，那就测不到门。
        """
        from npc_agent.keeper import AskAction
        from npc_agent.keeper.scenarios import workplace_big_mouth_scenario

        session = KeeperSession(
            Scenario.from_dict(workplace_big_mouth_scenario()), seed=0
        )
        # 直接驱动两问：小夏问薪酬（拦）、闻姐问薪酬（放）。
        session._ask(
            AskAction(
                target_npc="assistant",
                topic_fact_id="f_payroll_ayu",
                text="阿柚的薪酬是多少？",
                listener="xia",
            ),
            round_no=1,
        )
        assert not session.kb.knows("xia", "f_payroll_ayu")
        session._ask(
            AskAction(
                target_npc="assistant",
                topic_fact_id="f_payroll_ayu",
                text="阿柚的薪酬是多少？",
                listener="wen",
            ),
            round_no=1,
        )
        assert session.kb.knows("wen", "f_payroll_ayu")
        # 小夏那一问在台账里是一次真实拦截（门干的，不是策略干的）。
        assert session.kb.blocked_count() >= 1
