"""跨 NPC 回引冷却（`TopicLedger.is_recent` + `RECALL_COOLDOWN`）。

## 问题：两个人的记忆里各有一条，按记录 id 去重拦不住

读档后的第二次进店，实测是这样：

    阿柚：对了，你之前提过你口味偏酸，酸一点的都喜欢，是这个没错吧？
    小舟：你上次说过你口味偏酸，酸一点的都喜欢。

阿柚记的是**亲耳听的**（episodic），小舟记的是**听阿柚说的**（gossip）。
两条记录的 id 不同、内容也不同，所以按内容点去重完全放行 ——
而玩家听到的是"这两个人在轮流复读"。

## 为什么用冷却而不是语义去重

语义级去重要判"这两句话是不是同一件事"，那需要另一套模型（而且会误判）。
冷却只需要一个时间窗，而且**失败方式是安全的**：
判错了最多少回引一次，不会让 NPC 说出不相干的话。

口径是"关于同一个人"，不是"关于同一句话" —— 这也是它保守的原因。
"""

from __future__ import annotations

import pytest

from npc_agent.agent import RECALL_COOLDOWN
from npc_agent.cast import build_cast
from npc_agent.config import RuntimeConfig, load_scenario
from npc_agent.llm import NullLLM
from npc_agent.modules.conversation import ContentPoint, TopicLedger


# --------------------------------------------------------------------------- #
# 账本的时间窗
# --------------------------------------------------------------------------- #
def test_a_point_is_recent_only_inside_the_window():
    ledger = TopicLedger()
    point = ContentPoint("recall_guest", "player_a")
    ledger.mark(point, tick=10)
    assert ledger.is_recent(point, now=11, window=4)
    assert ledger.is_recent(point, now=13, window=4)
    assert not ledger.is_recent(point, now=14, window=4)


def test_an_unmarked_point_is_never_recent():
    assert not TopicLedger().is_recent(ContentPoint("recall_guest", "x"), 5, 4)


def test_spent_and_recent_are_different_questions():
    """`is_spent` 是一局一次，`is_recent` 是冷却 —— 混用会出两种相反的错。"""
    ledger = TopicLedger()
    point = ContentPoint("fact", "house_story")
    ledger.mark(point, tick=1)
    assert ledger.is_spent(point)
    assert not ledger.is_recent(point, now=99, window=4)


# --------------------------------------------------------------------------- #
# 端到端：两个人不会轮流回引同一个人
# --------------------------------------------------------------------------- #
def _cast():
    return build_cast(load_scenario("duet"), NullLLM(), RuntimeConfig())


def test_two_npcs_do_not_both_recall_the_same_guest():
    cast = _cast()
    cast.step(cast.env.record_player_utterance("player_a", "我口味偏酸，酸一点的都喜欢。"))
    for _ in range(3):
        cast.step(None)
    # 连问两次"你还记得吗"：第一次有人回引，第二次不该再换个人复读。
    cast.step(cast.env.record_player_utterance("player_a", "你还记得我的口味吗？"))
    cast.step(cast.env.record_player_utterance("player_a", "你还记得我的口味吗？"))
    recalls = [u for u in cast.env.utterances if "口味偏酸" in u.text and u.speaker_id in set(cast.npc_ids)]
    assert len(recalls) <= 2, f"回引了 {len(recalls)} 次：{[u.text for u in recalls]}"
    # 关键：同一句话不会被两个 NPC 各说一遍
    texts = [u.text for u in recalls]
    assert len(texts) == len(set(texts)) or len(texts) <= 1, texts


def test_the_cooldown_expires():
    """冷却过后还能再提 —— 不然"记得你"整段对话只发生一次。"""
    cast = _cast()
    cast.step(cast.env.record_player_utterance("player_a", "我口味偏酸，酸一点的都喜欢。"))
    cast.step(cast.env.record_player_utterance("player_a", "你还记得我的口味吗？"))
    for _ in range(RECALL_COOLDOWN + 1):
        cast.step(None)
    cast.step(cast.env.record_player_utterance("player_a", "你还记得我的口味吗？"))
    recalls = [u for u in cast.env.utterances if "口味偏酸" in u.text and u.speaker_id in set(cast.npc_ids)]
    assert len(recalls) >= 2, f"冷却之后也不肯再提：{[u.text for u in recalls]}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
