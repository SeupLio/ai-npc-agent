"""NPC 主动私语的护栏。

## 这里在守什么

交接文档里"下一轮优先级最高"的一条：私聊此前只有玩家能开小窗。
这一支补的是"会主动分享秘密的 NPC"—— 三条契约：

1. **三条件缺一不可**：关系到位（阶段门槛）+ 玩家进过这一场 + 一次性。
   缺第三条就是复读（同一句秘密说两遍），缺第一条就是"对陌生人掏心"；
2. **真的是私聊**：channel=private、audience 只有收件人；
   另一个 NPC 听不见、对话板不开义务、传闻网络不传播；
3. **被拦也算试过**：台词被工具层拦下（剧透/出戏）要记账 ——
   否则冷场每轮都重试同一句被拦的话。
"""

from __future__ import annotations

import pytest

from npc_agent.cast import build_cast
from npc_agent.config import RuntimeConfig, load_scenario
from npc_agent.llm import NullLLM
from npc_agent.modules.relationship import stage_of
from npc_agent.modules.tools import ToolContext
from npc_agent.types import AgentTurn


def _cast(scenario_id: str = "tutorial"):
    scenario = load_scenario(scenario_id)
    cfg = RuntimeConfig()
    cfg.llm_provider = "null"
    return build_cast(scenario, NullLLM(), cfg)


def _regular(cast, npc_id="ayou", pid="player_a", score=72):
    """把 (npc, player) 推到常客：环境层的好感才是同步源。"""
    cast.env.actors[pid].affinity[npc_id] = score
    cast.env.actors[npc_id].affinity[pid] = score
    agent = cast.agents[npc_id]
    bond = agent.relationships.bond(pid)
    bond.score = score
    bond.stage = stage_of(score)
    bond.interactions = 8
    return agent


def _ctx(cast, npc_id="ayou"):
    agent = cast.agents[npc_id]
    return ToolContext(
        actor_id=npc_id, tick=cast.env.tick, env=cast.env,
        memory=agent.memory, persona=agent.persona, tracker=agent.state,
    )


class TestTriggerConditions:
    def test_fires_once_when_stage_reached(self):
        cast = _cast()
        agent = _regular(cast)
        # 先让玩家进过这一场（state.players 需要一次观察）
        u = cast.env.record_player_utterance("player_a", "阿柚，今天人不多啊")
        cast.step(u)

        turn = AgentTurn(actor_id="ayou", tick=cast.env.tick)
        assert agent._whisper_secret(turn, _ctx(cast))
        assert turn.say
        assert turn.addressed_to == "player_a"
        # 一次性：第二次不再触发
        turn2 = AgentTurn(actor_id="ayou", tick=cast.env.tick)
        assert not agent._whisper_secret(turn2, _ctx(cast))

    def test_returning_player_whispered_without_speaking(self):
        # 老客回场一句话不说，私语照样找他 —— 关系档案在场就行。
        # 「玩家进过这一场」的真实语义是"出现在环境观察里"，
        # 不是"这一场发过言"：断线重连回来的人第一句就是听 NPC 说话。
        cast = _cast()
        agent = _regular(cast)
        turn = AgentTurn(actor_id="ayou", tick=cast.env.tick)
        assert agent._whisper_secret(turn, _ctx(cast))
        assert turn.addressed_to == "player_a"

    def test_stage_gate_blocks_strangers(self):
        cast = _cast()
        _regular(cast, score=50)  # 生客
        agent = cast.agents["ayou"]
        u = cast.env.record_player_utterance("player_a", "阿柚，今天人不多啊")
        cast.step(u)
        turn = AgentTurn(actor_id="ayou", tick=cast.env.tick)
        assert not agent._whisper_secret(turn, _ctx(cast))

    def test_prefers_higher_affinity_player(self):
        cast = _cast("duet")
        _regular(cast, pid="player_a", score=90)
        _regular(cast, pid="player_b", score=72)
        u = cast.env.record_player_utterance("player_a", "阿柚，今天人不多啊")
        cast.step(u)
        u = cast.env.record_player_utterance("player_b", "小舟，今天人不多啊")
        cast.step(u)
        agent = cast.agents["ayou"]
        turn = AgentTurn(actor_id="ayou", tick=cast.env.tick)
        assert agent._whisper_secret(turn, _ctx(cast))
        # 好感 90 的阿澈先听到，而不是先到先得
        assert turn.addressed_to == "player_a"


class TestPrivacy:
    def test_whisper_is_private_channel(self):
        cast = _cast()
        agent = _regular(cast)
        u = cast.env.record_player_utterance("player_a", "阿柚，今天人不多啊")
        cast.step(u)
        before = len(cast.env.utterances)
        turn = AgentTurn(actor_id="ayou", tick=cast.env.tick)
        assert agent._whisper_secret(turn, _ctx(cast))

        spoken = cast.env.utterances[before:]
        assert len(spoken) == 1
        line = spoken[0]
        assert line.channel == "private"
        assert set(line.audience or ()) == {"ayou", "player_a"}

    def test_blocked_whisper_marked_as_tried(self):
        # 人设的私语素材**只有**一条必被剧透闸拦下的（hint 含 spoiler 词）
        cast = _cast()
        agent = _regular(cast)
        u = cast.env.record_player_utterance("player_a", "阿柚，今天人不多啊")
        cast.step(u)
        # 替换素材（append 的话正常素材会排在前面，拦不到这条）
        agent.persona.secret_notes = [
            {"id": "spoiler_trap", "hint": "灯塔余晖的秘密明天揭晓", "min_stage": "常客"}
        ]
        turn = AgentTurn(actor_id="ayou", tick=cast.env.tick)
        fired = agent._whisper_secret(turn, _ctx(cast))
        assert fired
        assert not turn.say  # 话没说出去
        # 但记为已试过：不会每轮重试
        assert "spoiler_trap" in agent.relationships.bonds["player_a"].secrets_shared
