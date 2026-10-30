"""对话线程层（`modules/conversation.py`）的单元测试。

这一层是"NPC 之间能不能聊起来"的地基，它有三个互相独立的部分：

    话题账本   TopicLedger      —— 这件事说过了没有
    接话义务   ReplyObligation  —— 这句话该谁接
    话语类型   classify         —— 这句话是"转达""问群体"还是"陈述"

三条分开测，因为它们的失败方式完全不同：账本坏了会复读，
义务坏了会冷场，分类坏了会答非所问。
"""

from __future__ import annotations

import pytest

from npc_agent.modules.conversation import (
    KIND_DELEGATE,
    KIND_GROUP_QUESTION,
    KIND_PEER_TURN,
    KIND_QUESTION,
    KIND_STATEMENT,
    ContentPoint,
    ConversationBoard,
    TopicLedger,
    classify_utterance,
    normalize_point,
)
from npc_agent.types import Utterance


# --------------------------------------------------------------------------- #
# 话题账本
# --------------------------------------------------------------------------- #
def test_the_same_point_is_spent_after_being_marked():
    ledger = TopicLedger()
    point = ContentPoint("fact", "house_story")
    assert not ledger.is_spent(point)
    ledger.mark(point, tick=3)
    assert ledger.is_spent(point)
    assert ledger.size() == 1


def test_marking_twice_does_not_double_count():
    """同一个点记两次还是一条 —— 它问的是"说过没有"，不是"说了几次"。"""
    ledger = TopicLedger()
    point = ContentPoint("fact", "brewing")
    ledger.mark(point, tick=1)
    ledger.mark(point, tick=9)
    assert ledger.size() == 1


def test_points_are_scoped_by_topic():
    """同一个 detail、不同 topic 是两个点（`delegate` 和 `ask_peer` 都带人名）。"""
    ledger = TopicLedger()
    ledger.mark(ContentPoint("ask_peer", "xiaozhou"), tick=1)
    assert ledger.is_spent(ContentPoint("ask_peer", "xiaozhou"))
    assert not ledger.is_spent(ContentPoint("delegate", "xiaozhou"))


def test_normalize_point_matches_the_repetition_rule():
    """口径必须和复读判据一致：标点与空白不算内容。

    两个模块口径不一致时，闸门会放行账本认为"已经说过"的句子，
    反过来也一样 —— 这种不一致不会报错，只会让去重看起来时灵时不灵。
    """
    assert normalize_point("这家店，开在灯塔下面。") == normalize_point(
        "这家店开在灯塔下面"
    )


# --------------------------------------------------------------------------- #
# 话语类型
# --------------------------------------------------------------------------- #
def _utt(text: str, *, role: str = "player", addressed_to: str | None = None) -> Utterance:
    return Utterance(
        speaker_id="player_a" if role == "player" else "ayou",
        speaker_name="阿澈" if role == "player" else "阿柚",
        text=text,
        tick=0,
        role=role,  # type: ignore[arg-type]
        is_question=text.rstrip().endswith(("？", "?")),
        addressed_to=addressed_to,
    )


def test_a_delegation_is_not_a_question():
    """「你让小舟弹一首呗？」是**转达请求**，不是提问。

    这是「答非所问」最典型的一条：当成提问处理，NPC 只能答"我不知道"，
    而它真正不知道的是"该转达"，不是"这件事"。
    """
    kind, target = classify_utterance(
        _utt("阿柚，你让小舟弹一首呗？"),
        self_id="ayou",
        self_name="阿柚",
        peer_ids=["xiaozhou"],
        peer_names=["小舟"],
    )
    assert kind == KIND_DELEGATE
    assert target == "xiaozhou"


def test_a_verb_without_a_peer_name_is_not_a_delegation():
    """只有"叫"类动词、没点同伴的名 —— 不是转达。判宽了会把闲聊当成指令。"""
    kind, target = classify_utterance(
        _utt("你让我想想吧"),
        self_id="ayou",
        self_name="阿柚",
        peer_ids=["xiaozhou"],
        peer_names=["小舟"],
    )
    assert kind != KIND_DELEGATE
    assert target is None


def test_group_question_is_detected():
    kind, _ = classify_utterance(
        _utt("你们俩平时都聊些什么？"),
        self_id="ayou",
        self_name="阿柚",
        peer_ids=["xiaozhou"],
        peer_names=["小舟"],
    )
    assert kind == KIND_GROUP_QUESTION


def test_peer_turn_is_recognised_by_addressee():
    kind, _ = classify_utterance(
        _utt("小舟，还有豆子吗？", role="npc", addressed_to="ayou"),
        self_id="ayou",
        self_name="阿柚",
        peer_ids=["xiaozhou"],
        peer_names=["小舟"],
    )
    assert kind == KIND_PEER_TURN


def test_a_peer_statement_not_addressed_to_me_is_not_a_peer_turn():
    """同伴说给别人听的话，不该由我接 —— 否则两个 NPC 会互相抢话。"""
    kind, _ = classify_utterance(
        _utt("今天风不小。", role="npc", addressed_to="player_a"),
        self_id="ayou",
        self_name="阿柚",
        peer_ids=["xiaozhou"],
        peer_names=["小舟"],
    )
    assert kind == KIND_STATEMENT


def test_plain_question_and_statement():
    assert (
        classify_utterance(_utt("这店开了多久了？"), self_id="ayou", self_name="阿柚")[0]
        == KIND_QUESTION
    )
    assert (
        classify_utterance(_utt("今天挺热闹。"), self_id="ayou", self_name="阿柚")[0]
        == KIND_STATEMENT
    )


# --------------------------------------------------------------------------- #
# 接话义务
# --------------------------------------------------------------------------- #
def test_obligation_can_only_be_taken_by_the_addressee():
    """别人的债不能被顺手清掉 —— 清了那句问话就永远没人接。"""
    board = ConversationBoard(chain_limit=3)
    board.open(
        target_id="xiaozhou",
        speaker_id="ayou",
        speaker_name="阿柚",
        text="小舟，来一段？",
        tick=1,
    )
    assert board.take("ayou") is None
    assert board.pending is not None
    obligation = board.take("xiaozhou")
    assert obligation is not None
    assert obligation.target_id == "xiaozhou"
    assert board.pending is None


def test_chain_limit_stops_the_exchange():
    """连聊到上限就不再开义务 —— 玩家不该被晾在一边看两个人聊十轮。"""
    board = ConversationBoard(chain_limit=1)
    first = board.open(
        target_id="xiaozhou", speaker_id="ayou", speaker_name="阿柚", text="来一段？", tick=1
    )
    assert first is not None
    board.take("xiaozhou")
    second = board.open(
        target_id="ayou", speaker_id="xiaozhou", speaker_name="小舟", text="好。", tick=2
    )
    assert second is None
    assert board.truncated == 1


def test_player_turn_clears_the_obligation():
    """玩家插话的优先级高于"NPC 还没聊完"。"""
    board = ConversationBoard(chain_limit=3)
    board.open(
        target_id="xiaozhou", speaker_id="ayou", speaker_name="阿柚", text="来一段？", tick=1
    )
    board.player_turn()
    assert board.pending is None
    assert board.chain == 0


def test_obligation_renders_as_a_peer_utterance():
    """义务还原成发言时必须是 `role="npc"` —— 下游靠它区分"玩家"和"同伴"。"""
    board = ConversationBoard(chain_limit=3)
    board.open(
        target_id="xiaozhou",
        speaker_id="ayou",
        speaker_name="阿柚",
        text="小舟，还有豆子吗？",
        tick=1,
    )
    utterance = board.pending.as_utterance()
    assert utterance.role == "npc"
    assert utterance.addressed_to == "xiaozhou"
    assert utterance.is_question


def test_recording_a_line_marks_its_point():
    board = ConversationBoard()
    point = ContentPoint("fact", "house_story")
    board.record(
        tick=1,
        speaker_id="ayou",
        speaker_name="阿柚",
        text="这家店开在灯塔下面。",
        point=point,
    )
    assert board.ledger.is_spent(point)
    assert len(board.recent(3)) == 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
