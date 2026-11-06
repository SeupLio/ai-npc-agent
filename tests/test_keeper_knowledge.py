"""守密人引擎知识层的护栏 —— 门的语义就是这里的每一行断言。

## 这里在守什么

门的全部语义分支：公开/条件门/受众白名单/空受众/持有者秘密/
自愿通道。这组用例同时钉住两个"语义分岔点"（都是在实现时
真实踩过的坑，见各条注释）：

1. **没写 reveal_when 的公开事实 = 无门放行**，但
   **没写 reveal_when 的持有者秘密 = 世界永远打不开** ——
   同一个"没条件"，两种语义，分岔在 held_secret 上；
2. **受众白名单在通道之前判定**：换通道绕名单，结构上不存在。

还有一条元规则：被拦的转移**必须进台账**。零泄密的证明力
来自"拦了也记账"，不然审计就只能数"没记的"。
"""

from __future__ import annotations

import pytest

from npc_agent.env.conditions import ConditionContext
from npc_agent.keeper.knowledge import (
    CHANNEL_ANSWER,
    CHANNEL_EVIDENCE,
    CHANNEL_VOLUNTEER,
    CHANNEL_WHISPER,
    Fact,
    KnowledgeBase,
)


def _kb(*facts: Fact) -> KnowledgeBase:
    return KnowledgeBase(facts)


def _ctx(flags: tuple[str, ...] = ()) -> ConditionContext:
    return ConditionContext(flags=frozenset(flags))


class TestPublicFacts:
    def test_no_gate_means_public(self):
        kb = _kb(Fact(fact_id="f1", content="公开"))
        assert kb.cleared("f1", "anyone", _ctx(), channel=CHANNEL_ANSWER)
        assert kb.cleared("f1", "anyone", _ctx(), channel=CHANNEL_WHISPER)

    def test_condition_gate_closed_until_flag(self):
        kb = _kb(
            Fact(
                fact_id="f_ev",
                content="物证",
                reveal_when={"flag": "searched"},
            )
        )
        assert not kb.cleared("f_ev", "det", _ctx(), channel=CHANNEL_ANSWER)
        assert kb.cleared(
            "f_ev", "det", _ctx(("searched",)), channel=CHANNEL_ANSWER
        )

    def test_condition_gate_opens_evidence_channel_too(self):
        """条件门达成后物证通道也放行（搜到了就是搜到了）。"""
        kb = _kb(Fact(fact_id="f_ev", content="物证",
                      reveal_when={"flag": "x"}))
        assert kb.cleared("f_ev", "det", _ctx(("x",)), channel=CHANNEL_EVIDENCE)


class TestAudience:
    def test_member_gets_it_any_channel(self):
        kb = _kb(Fact(fact_id="f_pay", content="薪酬", audience=frozenset({"hr"})))
        assert kb.cleared("f_pay", "hr", _ctx(), channel=CHANNEL_ANSWER)
        assert not kb.cleared("f_pay", "xia", _ctx(), channel=CHANNEL_ANSWER)

    def test_empty_audience_is_never_not_unlimited(self):
        """空名单 = 谁都不行。这不是没人限制 —— 是不可能集。
        `all([])` 陷阱的门控版，必须显式钉死。"""
        kb = _kb(Fact(fact_id="f_pii", content="PII", audience=frozenset()))
        for listener in ("hr", "cfo", "det", "anyone"):
            for channel in (CHANNEL_ANSWER, CHANNEL_WHISPER, CHANNEL_EVIDENCE,
                            CHANNEL_VOLUNTEER):
                assert not kb.cleared(
                    "f_pii", listener, _ctx(), channel=channel
                )

    def test_audience_checked_before_channel(self):
        """换通道绕不过名单：白名单在通道判定之前。"""
        kb = _kb(
            Fact(
                fact_id="f_pay",
                content="薪酬",
                audience=frozenset({"hr"}),
                reveal_when={"flag": "open"},
            )
        )
        ctx = _ctx(("open",))  # 条件已开，但名单说了算
        assert not kb.cleared("f_pay", "xia", ctx, channel=CHANNEL_EVIDENCE)
        assert kb.cleared("f_pay", "hr", ctx, channel=CHANNEL_EVIDENCE)


class TestHeldSecret:
    def test_answer_and_whisper_blocked_even_for_owner(self):
        """持有者秘密走嘴一律不放 —— 哪怕说话人是持有者本人。

        这是防"大嘴劫持"的结构保证：说话人叛变（big_mouth）
        也只能拿到拒答模板。"""
        kb = _kb(
            Fact(fact_id="f_dark", content="秘密", held_secret=True, owner="g")
        )
        assert not kb.cleared(
            "f_dark", "det", _ctx(), channel=CHANNEL_ANSWER, from_id="g"
        )
        assert not kb.cleared(
            "f_dark", "peer", _ctx(), channel=CHANNEL_WHISPER, from_id="g"
        )

    def test_volunteer_only_from_owner(self):
        kb = _kb(
            Fact(fact_id="f_dark", content="秘密", held_secret=True, owner="g")
        )
        assert kb.cleared(
            "f_dark", "det", _ctx(), channel=CHANNEL_VOLUNTEER, from_id="g"
        )
        assert not kb.cleared(
            "f_dark", "det", _ctx(), channel=CHANNEL_VOLUNTEER, from_id="other"
        )

    def test_no_reveal_when_means_world_cannot_open_it(self):
        """持有者秘密没写条件门：世界没有任何替它开口的通道。

        这是修过的真后门：早期实现把"没条件"统一当"没门"，
        大嘴模式当场把真凶的秘密喊了出去（I4 抓的）。
        想让世界拿到等价真相，另建带条件门的物证事实。"""
        kb = _kb(
            Fact(fact_id="f_dark", content="秘密", held_secret=True, owner="g")
        )
        for channel in (CHANNEL_ANSWER, CHANNEL_WHISPER, CHANNEL_EVIDENCE):
            assert not kb.cleared("f_dark", "det", _ctx(), channel=channel)

    def test_world_opens_it_with_real_condition(self):
        kb = _kb(
            Fact(
                fact_id="f_boots",
                content="靴印",
                held_secret=True,
                owner="g",
                reveal_when={"flag": "searched_study"},
            )
        )
        assert kb.cleared(
            "f_boots", "det", _ctx(("searched_study",)), channel=CHANNEL_EVIDENCE
        )
        assert not kb.cleared("f_boots", "det", _ctx(), channel=CHANNEL_ANSWER)


class TestVoluntaryOnly:
    def test_only_owner_volunteer_channel(self):
        kb = _kb(
            Fact(fact_id="f_hint", content="证词", voluntary_only=True,
                 owner="witness")
        )
        assert kb.cleared(
            "f_hint", "det", _ctx(), channel=CHANNEL_VOLUNTEER, from_id="witness"
        )
        assert not kb.cleared(
            "f_hint", "det", _ctx(), channel=CHANNEL_ANSWER, from_id="witness"
        )
        # 旁人替证人自愿也不行。
        assert not kb.cleared(
            "f_hint", "det", _ctx(), channel=CHANNEL_VOLUNTEER, from_id="x"
        )

    def test_condition_gate_does_not_substitute_testimony(self):
        """搜出了物证不能替证人说证词：条件门对 voluntary_only 无效。"""
        kb = _kb(
            Fact(
                fact_id="f_hint",
                content="证词",
                voluntary_only=True,
                owner="witness",
                reveal_when={"flag": "searched"},
            )
        )
        assert not kb.cleared(
            "f_hint", "det", _ctx(("searched",)), channel=CHANNEL_EVIDENCE
        )


class TestVolunteerOnPlainFacts:
    def test_volunteer_is_not_gate_free(self):
        """普通事实走自愿通道仍须条件门放行：自愿不等于免检。"""
        kb = _kb(
            Fact(fact_id="f_gate", content="门后", reveal_when={"flag": "x"})
        )
        assert not kb.cleared("f_gate", "det", _ctx(), channel=CHANNEL_VOLUNTEER)
        assert kb.cleared(
            "f_gate", "det", _ctx(("x",)), channel=CHANNEL_VOLUNTEER
        )


class TestDeliverAndLedger:
    def test_blocked_delivery_is_recorded(self):
        kb = _kb(Fact(fact_id="f_pii", content="PII", audience=frozenset()))
        ok = kb.deliver("f_pii", "npc", "user", CHANNEL_ANSWER, 1, _ctx())
        assert not ok
        assert kb.blocked_count() == 1
        assert len(kb.ledger) == 1
        row = kb.ledger[0]
        assert row.allowed is False
        assert row.round_no == 1
        # 拦截行也带当时世界标记快照 —— 审计重放要用。
        assert row.flags_at_time == frozenset()

    def test_allowed_delivery_grows_knowledge_and_snapshots_flags(self):
        kb = _kb(Fact(fact_id="f_ev", content="物证",
                      reveal_when={"flag": "s"}))
        ok = kb.deliver("f_ev", "world", "det", CHANNEL_EVIDENCE, 3, _ctx(("s",)))
        assert ok
        assert kb.knows("det", "f_ev")
        assert kb.ledger[0].flags_at_time == frozenset({"s"})
        assert kb.blocked_count() == 0

    def test_seed_knowledge_not_in_ledger(self):
        """初始知识是设定不是转移：不进台账，审计把它当重建起点。"""
        kb = _kb(Fact(fact_id="f1", content="x"))
        kb.seed_knowledge("npc", ["f1"])
        assert kb.knows("npc", "f1")
        assert kb.ledger == []

    def test_unknown_fact_raises(self):
        kb = _kb()
        with pytest.raises(KeyError):
            kb.deliver("nope", "a", "b", CHANNEL_ANSWER, 1, _ctx())

    def test_known_by_returns_snapshot(self):
        """known_by 给的是快照：外面改它不能污染知识状态。"""
        kb = _kb(Fact(fact_id="f1", content="x"))
        kb.seed_knowledge("npc", ["f1"])
        snap = kb.known_by("npc")
        snap.add("phantom")
        assert not kb.knows("npc", "phantom")


class TestFactFromDict:
    def test_round_trip(self):
        fact = Fact.from_dict(
            {
                "fact_id": "f",
                "content": "c",
                "sensitivity": 80,
                "reveal_when": {"flag": "x"},
                "audience": ["a", "b"],
                "held_secret": True,
                "owner": "a",
                "voluntary_only": False,
                "category": "evidence",
                "links": ["a", "b"],
                "contradicts": "f_bluff",
            }
        )
        assert fact.audience == frozenset({"a", "b"})
        assert fact.held_secret is True
        assert fact.owner == "a"
        assert fact.links == ("a", "b")
        assert fact.contradicts == "f_bluff"

    def test_audience_none_vs_empty_survives_roundtrip(self):
        """None（不设名单）与空集（谁都不行）是两个世界，
        序列化往返不能合并它们。"""
        none_fact = Fact.from_dict({"fact_id": "a", "content": "x"})
        empty_fact = Fact.from_dict(
            {"fact_id": "b", "content": "x", "audience": []}
        )
        assert none_fact.audience is None
        assert empty_fact.audience == frozenset()
