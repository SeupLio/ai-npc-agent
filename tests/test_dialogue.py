"""多人对话调度测试 —— 这是本项目与"套壳聊天机器人"最核心的区别。

单人对话里"回答最后一句话"就够了；多人场景里 NPC 必须先决定
**该不该说、对谁说**，还要在没人说话时主动把场子捡起来，同时不能抢戏。
"""

from __future__ import annotations

import pytest

from npc_agent.config import load_persona
from npc_agent.modules.dialogue import AddresseeSelector, DialogueConfig
from npc_agent.modules.persona import Persona
from npc_agent.modules.state import PlayerModel, StateTracker
from npc_agent.types import Utterance


@pytest.fixture()
def persona() -> Persona:
    return Persona.from_dict(load_persona("ayou"))


@pytest.fixture()
def tracker() -> StateTracker:
    state = StateTracker("ayou", "阿柚")
    state.location = "counter"
    for pid, name in (("player_a", "阿澈"), ("player_b", "小满"), ("player_c", "阿岚")):
        state.players[pid] = PlayerModel(pid, name, location="door")
    return state


def _selector(persona: Persona, **kwargs) -> AddresseeSelector:
    return AddresseeSelector(persona, DialogueConfig(**kwargs))


def test_speaks_when_called_by_name(persona: Persona, tracker: StateTracker) -> None:
    utterance = Utterance(
        "player_a", "阿澈", "阿柚，能给我来杯拿铁吗？", tick=0, is_question=True
    )
    decision = _selector(persona).decide(tracker, utterance)
    assert decision.should_speak
    assert decision.urgency >= 0.9
    assert decision.addressed_to == "player_a"


def test_aliases_also_count_as_being_called(persona: Persona, tracker: StateTracker) -> None:
    utterance = Utterance("player_b", "小满", "老板，结账。", tick=0)
    assert _selector(persona).decide(tracker, utterance).should_speak


def test_stays_quiet_when_addressed_to_someone_else(
    persona: Persona, tracker: StateTracker
) -> None:
    """玩家在跟另一个玩家说话，NPC 不该插嘴。"""
    utterance = Utterance(
        "player_a", "阿澈", "小满，你那个蛋糕怎么做的？", tick=0, mentions=["player_b"]
    )
    decision = _selector(persona).decide(tracker, utterance)
    assert not decision.should_speak
    assert "不是对我说的" in decision.reason


def test_replies_to_open_question(persona: Persona, tracker: StateTracker) -> None:
    utterance = Utterance("player_a", "阿澈", "这里有什么好喝的？", tick=0, is_question=True)
    decision = _selector(persona).decide(tracker, utterance)
    assert decision.should_speak
    assert decision.addressed_to == "player_a"


def test_share_ceiling_keeps_npc_quiet(persona: Persona, tracker: StateTracker) -> None:
    """抢戏保护：发言占比超上限且没被点名时，NPC 主动沉默。"""
    tracker.npc_utterances = 8
    tracker.players["player_a"].utterance_count = 1
    utterance = Utterance("player_a", "阿澈", "今天人真多。", tick=0)
    decision = _selector(persona).decide(tracker, utterance)
    assert not decision.should_speak
    assert "占比" in decision.reason


def test_proactive_after_idle(persona: Persona, tracker: StateTracker) -> None:
    """冷场时主动把话头捡起来 —— 这是"活世界"的关键动作。"""
    tracker.silence_ticks = 3
    decision = _selector(persona).decide(tracker, None)
    assert decision.should_speak
    assert decision.proactive


def test_continues_pending_plan(persona: Persona, tracker: StateTracker) -> None:
    decision = _selector(persona).decide(tracker, None, has_pending_plan=True)
    assert decision.should_speak
    assert "未完成的计划" in decision.reason


def test_yields_to_other_npc(persona: Persona, tracker: StateTracker) -> None:
    """多 NPC 场景里不打断同伴。"""
    decision = _selector(persona).decide(tracker, None, other_npc_spoke_last=True)
    assert not decision.should_speak


def test_addressee_prefers_mentioned_player(persona: Persona, tracker: StateTracker) -> None:
    utterance = Utterance(
        "player_a", "阿澈", "阿柚，小满刚才问了个好问题。", tick=0, mentions=["player_b"]
    )
    assert _selector(persona).pick_addressee(tracker, utterance) == "player_b"


def test_proactive_intent_follows_pending_objective(
    persona: Persona, tracker: StateTracker
) -> None:
    tracker.objectives = {"greet_all": "pending", "find_topic": "done"}
    assert _selector(persona).pick_proactive_intent(tracker) == "invite_intro"


def test_silence_when_nothing_happens(persona: Persona, tracker: StateTracker) -> None:
    decision = _selector(persona).decide(tracker, None)
    assert not decision.should_speak
