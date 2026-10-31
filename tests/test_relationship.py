"""关系养成与称号（`modules/relationship.py`）。

## 这个文件守的是"数值必须有消费方"

环境里早就在算好感（递饮品 +4、答对题 +6），但修复前**没有任何决策读它**。
一个会动、没人用的数字，比没有更糟 —— 调参的人会以为自己能改变体验。

所以这里的断言分三层：

    阶段派生   → 好感变化要真的改变"关系阶段"
    行为差异   → 阶段要真的改变 NPC 说什么（回头客问候）
    称号事件   → 称号要**说出口**才算数，而且只说一次
"""

from __future__ import annotations

import pytest

from npc_agent.cast import build_cast
from npc_agent.config import RuntimeConfig, load_scenario
from npc_agent.llm import NullLLM
from npc_agent.modules.relationship import (
    SPECIAL_TITLES,
    RelationshipBook,
    stage_of,
    stage_rank,
)


# --------------------------------------------------------------------------- #
# 阶段派生
# --------------------------------------------------------------------------- #
def test_the_default_affinity_is_a_stranger():
    """环境给的初始好感是 50。50 必须是"生客" ——

    阈值从 30 起的那一版，第一次见面的人一进门就是「面熟」，
    还会当场领到称号。实测（duet）小舟第一轮就说
    「记住这个：熟面孔。阿澈。」—— 比没有关系系统更糟，因为它一眼假。
    """
    assert stage_of(50) == "生客"


def test_stages_climb_with_affinity():
    assert stage_of(56) == "面熟"
    assert stage_of(68) == "常客"
    assert stage_of(85) == "老友"
    assert stage_of(100) == "老友"


def test_stage_rank_is_ordered():
    ranks = [stage_rank(s) for s in ("生客", "面熟", "常客", "老友")]
    assert ranks == sorted(ranks)
    assert stage_rank("老友") > stage_rank("生客")


def test_an_unknown_stage_does_not_crash():
    """配置里写错一个阶段名，不该让整场对话崩掉。"""
    assert stage_rank("查无此阶段") == 0


# --------------------------------------------------------------------------- #
# 事件
# --------------------------------------------------------------------------- #
def test_crossing_a_threshold_emits_one_event_and_one_title():
    book = RelationshipBook("ayou", "阿柚")
    events = book.sync("player_a", 58, tick=1)
    kinds = [e.kind for e in events]
    assert kinds.count("stage_up") == 1
    assert kinds.count("title") == 1
    assert book.title_for("player_a") == "熟面孔"


def test_the_same_title_is_never_granted_twice():
    book = RelationshipBook()
    book.sync("player_a", 58, tick=1)
    book.sync("player_a", 59, tick=2)
    book.sync("player_a", 58, tick=3)
    assert book.titles_granted == 1


def test_staying_at_the_same_score_emits_nothing():
    book = RelationshipBook()
    book.sync("player_a", 50, tick=1)
    events = book.sync("player_a", 50, tick=2)
    assert events == []


def test_the_special_title_needs_interactions_not_just_affinity():
    """「记得你口味的人」不能靠"站得久"拿到 —— 要真的互动过。

    按"在场回合数"计数的那一版，站够 6 轮就能拿，
    而 NPC 一句话都没跟他说过。
    """
    book = RelationshipBook()
    need_stage, need_count, title = SPECIAL_TITLES[0][1], SPECIAL_TITLES[0][2], SPECIAL_TITLES[0][3]
    book.sync("player_a", 70, tick=1, active=False)  # 好感够，但没互动
    assert title not in book.bond("player_a").titles
    for tick in range(2, 2 + need_count + 1):
        book.sync("player_a", 70, tick=tick, active=True)
    assert title in book.bond("player_a").titles
    assert stage_rank(book.stage("player_a")) >= stage_rank(need_stage)


def test_take_pending_consumes_only_titles():
    """阶段事件是给报告看的，不该被"播报"消费掉；称号必须被消费。"""
    book = RelationshipBook()
    book.sync("player_a", 58, tick=1)
    event = book.take_pending("player_a")
    assert event is not None and event.kind == "title"
    assert book.take_pending("player_a") is None


def test_a_stranger_is_not_a_returning_guest():
    book = RelationshipBook()
    book.sync("player_a", 50, tick=1, active=True)
    assert not book.is_returning("player_a")
    book.sync("player_a", 58, tick=2, active=True)
    assert book.is_returning("player_a")


# --------------------------------------------------------------------------- #
# 端到端：关系真的改变了 NPC 说什么
# --------------------------------------------------------------------------- #
def _cast():
    return build_cast(load_scenario("duet"), NullLLM(), RuntimeConfig())


def test_a_title_is_actually_said_out_loud():
    """只存进数据库的称号，玩家看不见。"""
    cast = _cast()
    ayou = cast.agents["ayou"]
    cast.env.actors["ayou"].bump_affinity("player_a", 8)  # 阿柚 → 阿澈
    turns = cast.step(cast.env.record_player_utterance("player_a", "今天路过，进来坐坐。"))
    says = {t.actor_id: t.say for t in turns if t.say}
    assert "ayou" in says
    assert "熟面孔" in says["ayou"], f"称号没有说出口：{says['ayou']}"


def test_a_title_is_not_announced_twice():
    cast = _cast()
    cast.env.actors["ayou"].bump_affinity("player_a", 8)
    cast.step(cast.env.record_player_utterance("player_a", "今天路过。"))
    cast.step(cast.env.record_player_utterance("player_a", "还是老样子。"))
    lines = [u.text for u in cast.env.utterances if "熟面孔" in u.text]
    assert len(lines) == 1, f"同一个称号播报了 {len(lines)} 次：{lines}"


def test_a_question_is_answered_before_the_title_is_announced():
    """玩家在问问题时，称号要让位 —— 否则就是答非所问。"""
    cast = _cast()
    cast.env.actors["ayou"].bump_affinity("player_a", 8)
    turns = cast.step(cast.env.record_player_utterance("player_a", "你叫什么名字？"))
    says = {t.actor_id: t.say for t in turns if t.say}
    assert "阿柚" in says.get("ayou", ""), says
    assert "熟面孔" not in says.get("ayou", "")


def test_the_relationship_reaches_the_prompt():
    """关系要进提示词，否则接上模型之后这段逻辑就断了。"""
    cast = _cast()
    agent = cast.agents["ayou"]
    agent.relationships.sync("player_a", 58, tick=1, active=True)
    block = agent.relationships.bond_block()
    assert "面熟" in block
    assert "熟面孔" in block


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
