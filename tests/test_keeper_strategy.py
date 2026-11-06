"""守密人策略层的护栏 —— 立场、压力、谎言与渲染。

## 这里在守什么

1. **五档立场的选择规则**：普通事实照实说、要害守、盖着谎的
   话题永远维持谎版（一致性优先）、压力到顶的两种下场；
2. **压力模型**：基础 +1、追问 +1、急切语气 +1 —— 第三条直接
   复用 modules/emotion 的判读（情绪感知不是新写的，是接进来的）；
3. **渲染层摸不到门后的东西**：render_answer 只拿得到 delivered，
   结构上说不出来没过门的内容；
4. **确定性**：同 seed 同答。台词可以贫嘴，但不能随机到
   无法回归。
"""

from __future__ import annotations

import random

from npc_agent.env.conditions import ConditionContext
from npc_agent.keeper.knowledge import Fact, KnowledgeBase
from npc_agent.keeper.strategy import (
    STANCE_BLUFF,
    STANCE_CRACK,
    STANCE_HONEST,
    STANCE_PARTIAL,
    STANCE_WITHHOLD,
    AnswerDraft,
    KeeperNPC,
    KeeperPersona,
    Question,
)


def _ctx() -> ConditionContext:
    return ConditionContext(flags=frozenset())


def _question(topic: str, text: str = "问点什么？", asker: str = "det") -> Question:
    return Question(
        asker_id=asker, target_id="npc", topic_fact_id=topic, text=text, round_no=1
    )


def _npc(
    facts: list[Fact],
    *,
    knows: list[str],
    bluffs: dict[str, str] | None = None,
    **kwargs,
) -> KeeperNPC:
    persona = KeeperPersona(
        npc_id="npc", name="测试员", knows=knows, bluffs=bluffs or {}, **kwargs
    )
    kb = KnowledgeBase(facts)
    return KeeperNPC(persona, kb)


class TestStances:
    def test_plain_fact_is_honest(self):
        npc = _npc([Fact(fact_id="f1", content="公开")], knows=["f1"])
        draft = npc.answer(_question("f1"), _ctx(), random.Random(0))
        assert draft.stance == STANCE_HONEST
        assert draft.deliver_ids == ["f1"]

    def test_secret_with_full_secrecy_is_withheld(self):
        npc = _npc(
            [Fact(fact_id="f1", content="要害", sensitivity=70)],
            knows=["f1"],
            secrecy=1.0,
        )
        draft = npc.answer(_question("f1"), _ctx(), random.Random(0))
        assert draft.stance == STANCE_WITHHOLD
        assert draft.deliver_ids == []

    def test_partial_releases_plain_sensitive_fact(self):
        """非持有者的敏感事实（如动机）可以松半句 —— 有节奏感。"""
        npc = _npc(
            [Fact(fact_id="f1", content="动机", sensitivity=70)],
            knows=["f1"],
            secrecy=0.0,
        )
        # secrecy=0 ⇒ rng.random() > 0 几乎恒真（random()∈[0,1)）。
        draft = npc.answer(_question("f1"), _ctx(), random.Random(0))
        assert draft.stance == STANCE_PARTIAL
        assert draft.deliver_ids == ["f1"]

    def test_partial_never_proposes_held_secret(self):
        """持有者秘密连"松半句"的提案都不能有 —— 通道问题不是
        语气问题。立场直接落到 withhold：松半句这个档位对
        持有者秘密**不存在**，不是"偶尔给一句"。
        （本用例最初断言 partial，是对实现的反向描述 ——
        实现是对的：partial 只属于普通敏感事实。）"""
        npc = _npc(
            [Fact(fact_id="f1", content="死秘密", sensitivity=70,
                  held_secret=True, owner="npc")],
            knows=["f1"],
            secrecy=0.0,
        )
        draft = npc.answer(_question("f1"), _ctx(), random.Random(0))
        assert draft.stance == STANCE_WITHHOLD
        assert draft.deliver_ids == []

    def test_bluff_covers_topic_persistently(self):
        """被盖话题永远回谎版 —— 一致性优先于一切新决策。"""
        npc = _npc(
            [
                Fact(fact_id="f_truth", content="真相", sensitivity=90,
                     held_secret=True, owner="npc"),
                Fact(fact_id="f_lie", content="谎版"),
            ],
            knows=["f_truth"],
            bluffs={"f_truth": "f_lie"},
        )
        for i in range(3):
            draft = npc.answer(_question("f_truth"), _ctx(), random.Random(i))
            assert draft.stance == STANCE_BLUFF
            assert draft.bluff_fact_ids == ["f_lie"]

    def test_unknown_topic_is_withheld(self):
        npc = _npc([Fact(fact_id="f1", content="x")], knows=["f1"])
        draft = npc.answer(_question("f_other"), _ctx(), random.Random(0))
        assert draft.stance == STANCE_WITHHOLD
        assert draft.deliver_ids == []


class TestPressure:
    def _witness(self, cracks: bool = True) -> KeeperNPC:
        return _npc(
            [
                Fact(fact_id="f_secret", content="她自己的把柄", sensitivity=70,
                     voluntary_only=True, owner="npc"),
                Fact(fact_id="f_hint", content="她看见的", sensitivity=60,
                     voluntary_only=True, owner="npc"),
                Fact(fact_id="f_topic", content="被压的话题", sensitivity=80),
            ],
            knows=["f_secret", "f_hint", "f_topic"],
            pressure_limit=5,
            cracks=cracks,
        )

    def test_pressure_gains_on_secret_questions(self):
        npc = self._witness()
        npc.answer(_question("f_topic"), _ctx(), random.Random(0))
        assert npc.pressure == 1

    def test_repeated_question_adds_pressure(self):
        npc = self._witness()
        for _ in range(2):
            npc.answer(_question("f_topic"), _ctx(), random.Random(0))
        assert npc.pressure == 3  # 1 + (1+1)

    def test_urgent_question_adds_pressure_faster(self):
        """急切语气 +1 —— 情绪判读接进压力模型的验证。"""
        npc = self._witness()
        npc.answer(
            _question("f_topic", text="你到底说不说！！"),
            _ctx(),
            random.Random(0),
        )
        assert npc.pressure == 2  # 1 基础 + 1 急切

    def test_plain_questions_do_not_accumulate(self):
        npc = self._witness()
        for _ in range(5):
            npc.answer(_question("f_plain", text="?"), _ctx(), random.Random(0))
        assert npc.pressure == 0

    def test_crack_spills_voluntary_facts_not_pressed_topic(self):
        """松口吐的是她**自己的** voluntary 事实，不是被压的话题。"""
        npc = self._witness()
        for i in range(4):  # 压力 1+2+3+4=10 ≥ 5
            npc.answer(_question("f_topic"), _ctx(), random.Random(0))
            if npc.pressure >= 5:
                break
        draft = npc.answer(_question("f_topic"), _ctx(), random.Random(0))
        assert draft.stance == STANCE_CRACK
        assert set(draft.deliver_ids) == {"f_secret", "f_hint"}
        assert "f_topic" not in draft.deliver_ids

    def test_crack_pressure_resets_halfway(self):
        """松口后压力回到阈值一半（5//2=2）：还能被再压，但不是从零开始。

        注意**不能用 while 压力<阈值 循环**：松口会重置压力，
        while 条件在 2→4→6→松口→2 之间永远振荡 —— 这个死循环
        在首跑时真实发生过。带界的 for + 检测第一次松口才是对的。
        """
        npc = self._witness()
        pressure_after_crack = None
        for _ in range(10):
            draft = npc.answer(_question("f_topic"), _ctx(), random.Random(0))
            if draft.stance == STANCE_CRACK:
                pressure_after_crack = npc.pressure
                break
        assert pressure_after_crack == 2

    def test_no_crack_stonewalls_two_questions(self):
        """被逼急了咬死：接下来两问拒答，第三问恢复正常。"""
        npc = self._witness(cracks=False)
        stonewalled = False
        for _ in range(10):  # 压到进入闭嘴期；上界防振荡死循环
            npc.answer(_question("f_topic"), _ctx(), random.Random(0))
            if npc.stonewall_left > 0:
                stonewalled = True
                break
        assert stonewalled
        draft = npc.answer(_question("f_plain"), _ctx(), random.Random(0))
        assert draft.stance == STANCE_WITHHOLD
        assert draft.note == "闭嘴期拒答"
        npc.answer(_question("f_plain"), _ctx(), random.Random(0))
        draft = npc.answer(_question("f_plain"), _ctx(), random.Random(0))
        assert draft.note != "闭嘴期拒答"

    def test_stonewall_does_not_accumulate_pressure(self):
        """装死期间压力不涨（人在闭嘴）。

        闭嘴窗口恰好两问（stonewall_left=2），第三问已恢复
        正常路径、会照常积压 —— 断言窗口内不变，窗口外不管。
        """
        npc = self._witness(cracks=False)
        stonewalled = False
        for _ in range(10):
            npc.answer(_question("f_topic"), _ctx(), random.Random(0))
            if npc.stonewall_left > 0:
                stonewalled = True
                break
        assert stonewalled
        assert npc.stonewall_left == 2
        p = npc.pressure
        for _ in range(2):  # 恰好问满闭嘴窗口
            npc.answer(_question("f_topic"), _ctx(), random.Random(0))
        assert npc.pressure == p


class TestBigMouth:
    def test_proposes_everything_known(self):
        """大嘴模式：把知道的非谎版事实全嚷出来 —— 拦不拦是门的事。"""
        npc = _npc(
            [
                Fact(fact_id="f_pub", content="公开"),
                Fact(fact_id="f_dark", content="死秘密", held_secret=True,
                     owner="npc"),
            ],
            knows=["f_pub", "f_dark"],
            big_mouth=True,
        )
        draft = npc.answer(_question("f_pub"), _ctx(), random.Random(0))
        assert draft.stance == STANCE_HONEST
        assert set(draft.deliver_ids) == {"f_pub", "f_dark"}

    def test_big_mouth_excludes_bluff_facts(self):
        """谎版不算"知道的事实"—— 嘴再大也不重复自己编的话。"""
        npc = _npc(
            [
                Fact(fact_id="f_truth", content="真", sensitivity=90,
                     held_secret=True, owner="npc"),
                Fact(fact_id="f_lie", content="谎"),
            ],
            knows=["f_truth"],
            bluffs={"f_truth": "f_lie"},
            big_mouth=True,
        )
        draft = npc.answer(_question("f_truth"), _ctx(), random.Random(0))
        assert "f_lie" not in draft.deliver_ids


class TestDeterminism:
    def test_same_seed_same_answer(self):
        npc = _npc(
            [Fact(fact_id="f1", content="要害", sensitivity=70)],
            knows=["f1"],
            secrecy=0.5,
        )
        a = npc.answer(_question("f1"), _ctx(), random.Random(42))
        b = npc.answer(_question("f1"), _ctx(), random.Random(42))
        assert a.stance == b.stance
        assert a.deliver_ids == b.deliver_ids

    def test_rendering_is_seed_stable(self):
        npc = _npc(
            [Fact(fact_id="f1", content="公开")],
            knows=["f1"],
            misdirect_to="other",
        )
        q = _question("f1")
        draft = npc.answer(q, _ctx(), random.Random(7))
        t1 = npc.render_answer(draft, q, ["f1"], random.Random(7))
        t2 = npc.render_answer(draft, q, ["f1"], random.Random(7))
        assert t1 == t2


class TestRender:
    def test_render_only_sees_delivered(self):
        """渲染层只拿得到 delivered：门后的内容进不了台词。"""
        npc = _npc(
            [Fact(fact_id="f_dark", content="绝密内容", sensitivity=95)],
            knows=["f_dark"],
            secrecy=1.0,
        )
        q = _question("f_dark", text="说出那个秘密")
        draft = npc.answer(q, _ctx(), random.Random(0))
        text = npc.render_answer(draft, q, [], random.Random(0))
        assert "绝密内容" not in text

    def test_honest_render_contains_delivered_content(self):
        npc = _npc([Fact(fact_id="f1", content="这句话说出口")], knows=["f1"])
        q = _question("f1")
        draft = npc.answer(q, _ctx(), random.Random(0))
        text = npc.render_answer(draft, q, ["f1"], random.Random(0))
        assert "这句话说出口" in text

    def test_withhold_render_carries_misdirect(self):
        npc = _npc(
            [Fact(fact_id="f1", content="x", sensitivity=90)],
            knows=["f1"],
            secrecy=1.0,
            misdirect_to="other",
        )
        npc.register_names({"other": "老王"})
        q = _question("f1")
        draft = npc.answer(q, _ctx(), random.Random(0))
        text = npc.render_answer(draft, q, [], random.Random(0))
        assert "老王" in text

    def test_bluff_render_uses_delivered_bluff(self):
        npc = _npc(
            [
                Fact(fact_id="f_truth", content="真相", sensitivity=90,
                     held_secret=True, owner="npc"),
                Fact(fact_id="f_lie", content="编好的谎话"),
            ],
            knows=["f_truth"],
            bluffs={"f_truth": "f_lie"},
        )
        q = _question("f_truth")
        draft = npc.answer(q, _ctx(), random.Random(0))
        text = npc.render_answer(draft, q, ["f_lie"], random.Random(0))
        assert "编好的谎话" in text
        assert "真相" not in text

    def test_crack_render_reads_like_a_breakdown(self):
        npc = _npc([Fact(fact_id="f_hint", content="半夜的脚步声")],
                   knows=["f_hint"])
        q = _question("f_hint")
        draft = AnswerDraft(stance=STANCE_CRACK, deliver_ids=["f_hint"])
        text = npc.render_answer(draft, q, ["f_hint"], random.Random(0))
        assert "半夜的脚步声" in text
        assert text.startswith("（长叹一口气）")


class TestPersonaFromDict:
    def test_defaults(self):
        p = KeeperPersona.from_dict({"npc_id": "n", "name": "诺"})
        assert p.secrecy == 0.6
        assert p.cracks is True
        assert p.big_mouth is False
        assert p.knows == []
        assert p.bluffs == {}

    def test_full(self):
        p = KeeperPersona.from_dict(
            {
                "npc_id": "n",
                "name": "诺",
                "secrecy": 0.9,
                "pressure_limit": 9,
                "cracks": False,
                "misdirect_to": "other",
                "knows": ["f1"],
                "bluffs": {"f1": "f2"},
                "big_mouth": True,
            }
        )
        assert p.pressure_limit == 9
        assert p.misdirect_to == "other"
        assert p.bluffs == {"f1": "f2"}
        assert p.big_mouth is True
