"""一对多注意力（`modules/attention.py`）。

## 守的是"别忘了有人问过你"

两种失败方式：

1. **同时有好几个人在场，NPC 只理列表里的第一个** —— 多玩家场景里
   永远是同一个人被回应，其他人像不存在；
2. **有人问了但没被答，之后也没人再提** —— 玩家会觉得"它根本没在听"。

第 2 条是这里的重点：它不是记忆问题（NPC 记得那句话），
是**账目**问题 —— 没有一个地方记着"我欠他一个回答"。
"""

from __future__ import annotations

import pytest

from npc_agent.cast import build_cast
from npc_agent.config import RuntimeConfig, load_scenario
from npc_agent.llm import NullLLM
from npc_agent.modules.attention import AttentionBoard
from npc_agent.modules.state import PlayerModel


# --------------------------------------------------------------------------- #
# 账目
# --------------------------------------------------------------------------- #
def test_a_question_creates_a_debt_and_settling_clears_it():
    board = AttentionBoard()
    board.note_question("player_a", "这家店开了多久了？", tick=1)
    assert len(board.debts) == 1
    board.settle("player_a")
    assert board.debts == []


def test_the_same_question_asked_twice_is_one_debt():
    board = AttentionBoard()
    board.note_question("player_a", "开了多久？", tick=1)
    board.note_question("player_a", "开了多久？", tick=3)
    assert len(board.debts) == 1
    assert board.debts[0].asked == 2


def test_a_debt_needs_a_minimum_wait_before_being_paid_back():
    """同一轮里刚问完就"补答"，那不是补答，是复读。"""
    board = AttentionBoard()
    board.note_question("player_a", "开了多久？", tick=1)
    assert board.oldest(now=1) is None
    assert board.oldest(now=3) is not None


def test_the_oldest_debt_wins():
    board = AttentionBoard()
    board.note_question("player_b", "第二个问题", tick=2)
    board.note_question("player_a", "第一个问题", tick=1)
    oldest = board.oldest(now=5)
    assert oldest is not None and oldest.player_id == "player_a"


def test_settle_one_counts_the_follow_up():
    board = AttentionBoard()
    board.note_question("player_a", "问题", tick=1)
    board.settle_one("player_a")
    assert board.follow_ups == 1


# --------------------------------------------------------------------------- #
# 排序
# --------------------------------------------------------------------------- #
def _player(pid: str, *, affinity: int = 50, last: int = 0, question: bool = False):
    model = PlayerModel(pid, pid)
    model.affinity = affinity
    model.last_spoke_tick = last
    model.pending_question = question
    return model


def test_a_question_outranks_a_statement():
    board = AttentionBoard()
    players = {
        "a": _player("a", last=0, question=True),
        "b": _player("b", last=0, question=False),
    }
    assert board.rank(players, now=1)[0] == "a"


def test_waiting_longer_breaks_a_tie():
    board = AttentionBoard()
    players = {
        "a": _player("a", last=5),
        "b": _player("b", last=1),
    }
    assert board.rank(players, now=6)[0] == "b"


def test_affinity_breaks_a_tie():
    board = AttentionBoard()
    players = {
        "a": _player("a", affinity=90),
        "b": _player("b", affinity=50),
    }
    assert board.rank(players, now=1)[0] == "a"


# --------------------------------------------------------------------------- #
# 端到端：欠着的回答会被补上
# --------------------------------------------------------------------------- #
def _cast():
    return build_cast(load_scenario("duet"), NullLLM(), RuntimeConfig())


def test_an_unanswered_question_is_answered_later():
    """被让出话头 → 记一笔债 → 冷场时**先补答**，再去做自己的事。

    这是"没在听"最典型的一种：玩家问了一句，NPC 一句话都没说，
    过两轮转头去分享店史。玩家不知道那是发言权调度的结果，
    只会觉得这 NPC 没在听。
    """
    cast = _cast()
    agent = cast.agents["ayou"]
    utterance = cast.env.record_player_utterance("player_a", "这家店开了多久了？")
    agent.step(utterance, allow_speech=False)  # 话头被让出去了
    assert agent.attention.debts, "没答的问题没有记账"

    answered = None
    for _ in range(4):
        cast.env.advance_tick()
        turn = agent.step(None)
        if turn.say and "灯塔" in turn.say:
            answered = turn.say
            break
    assert answered, "欠着的问题一直没有被补答"
    assert agent.attention.follow_ups >= 1
    assert not agent.attention.debts, "补答之后账没销"


def test_a_question_that_was_answered_creates_no_debt():
    cast = _cast()
    agent = cast.agents["ayou"]
    utterance = cast.env.record_player_utterance("player_a", "你叫什么名字？")
    turn = agent.step(utterance)
    assert turn.say
    assert not agent.attention.debts


def test_changing_the_subject_cancels_the_debt():
    """玩家换了话题，旧账作废 —— 人也是这样。"""
    cast = _cast()
    agent = cast.agents["ayou"]
    first = cast.env.record_player_utterance("player_a", "豆子是哪座山头的？")
    agent.step(first, allow_speech=False)
    assert agent.attention.debts
    cast.env.advance_tick()
    second = cast.env.record_player_utterance("player_a", "今天天气不错。")
    agent.step(second)
    assert not agent.attention.debts


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
