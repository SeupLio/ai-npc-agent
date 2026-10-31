"""同伴之间的关系（NPC ↔ NPC 的好感）。

## 为什么同伴也要有好感

`_ask_peer`（冷场时跟同伴搭话）原来是按**字典顺序**挑人的。
两个人的场景看不出来；三个人以上就是灾难：阿柚永远先找小舟，
第三个人永远没人搭话 —— 而玩家看不出那是顺序问题，
只会觉得"那个人被孤立了"。

好感在这里的作用不是"模拟社交"，是**给挑人一个非任意的依据**：
常跟你说话的人，你下次更可能找他。
"""

from __future__ import annotations

import pytest

from npc_agent.cast import build_cast
from npc_agent.config import RuntimeConfig, load_scenario
from npc_agent.llm import NullLLM
from npc_agent.modules.state import PlayerModel
from npc_agent.modules.tools import ToolContext


def _cast():
    return build_cast(load_scenario("duet"), NullLLM(), RuntimeConfig())


# --------------------------------------------------------------------------- #
# 环境 → 状态
# --------------------------------------------------------------------------- #
def test_the_observation_carries_peer_affinity():
    """观察里必须同时有玩家和同伴的好感 —— 只给玩家的话，
    `_ask_peer` 读到的永远是初始值 50。"""
    cast = _cast()
    cast.env.actors["ayou"].bump_affinity("xiaozhou", 7)
    affinity = cast.env.observe("ayou")["affinity"]
    assert affinity["xiaozhou"] == 57


def test_the_state_routes_affinity_by_kind():
    cast = _cast()
    agent = cast.agents["ayou"]
    cast.env.actors["ayou"].bump_affinity("xiaozhou", 7)
    cast.env.actors["ayou"].bump_affinity("player_a", 3)
    agent.step(None)
    assert agent.state.peers["xiaozhou"].affinity == 57
    assert agent.state.players["player_a"].affinity == 53


# --------------------------------------------------------------------------- #
# 什么会让关系变近
# --------------------------------------------------------------------------- #
def test_talking_to_a_peer_brings_you_closer():
    """点名叫人 / 转达 / 搭话 —— 说上话，关系就近一点。"""
    cast = _cast()
    before = cast.env.actors["ayou"].affinity_to("xiaozhou")
    cast.step(cast.env.record_player_utterance("player_a", "阿柚，你让小舟弹一首呗？"))
    after = cast.env.actors["ayou"].affinity_to("xiaozhou")
    assert after > before, f"转达之后关系没变：{before} → {after}"


def test_hearing_a_peers_news_brings_you_closer():
    """听到同伴分享的事，听的人对说的人好感 +1（单向）。"""
    cast = _cast()
    before = cast.env.actors["xiaozhou"].affinity_to("ayou")
    cast.step(cast.env.record_player_utterance("player_a", "我喜欢偏酸的豆子。"))
    for _ in range(4):
        cast.step(None)
    assert cast.env.actors["xiaozhou"].affinity_to("ayou") > before


def test_affinity_is_bounded():
    """好感有上下界 —— 无界的话"最亲近的人"会永远固定，排序失去意义。"""
    cast = _cast()
    actor = cast.env.actors["ayou"]
    actor.bump_affinity("xiaozhou", 999)
    assert actor.affinity_to("xiaozhou") == 100
    actor.bump_affinity("xiaozhou", -999)
    assert actor.affinity_to("xiaozhou") == 0


# --------------------------------------------------------------------------- #
# 挑人
# --------------------------------------------------------------------------- #
def _ctx(cast, agent):
    return ToolContext(
        actor_id=agent.id,
        tick=agent.state.tick,
        env=cast.env,
        memory=agent.memory,
        persona=agent.persona,
        tracker=agent.state,
        lore=agent.lore,
    )


def test_ask_peer_picks_the_closer_peer():
    """三个人在场时，先找关系更近的那个。

    用一个"假的三人场"来测：真人只有两个，而这条规则只在 >= 3 人时
    才看得出差别。不测它，排序代码就是一段没人验证的分支。
    """
    from npc_agent.types import AgentTurn

    cast = _cast()
    agent = cast.agents["ayou"]
    agent.state.peers["ghost_far"] = PlayerModel("ghost_far", "远客", "npc")
    agent.state.peers["ghost_far"].affinity = 10
    agent.state.peers["ghost_near"] = PlayerModel("ghost_near", "近客", "npc")
    agent.state.peers["ghost_near"].affinity = 95

    turn = AgentTurn(actor_id=agent.id, tick=0)
    ok = agent._ask_peer(turn, _ctx(cast, agent))
    assert ok, "没有向同伴搭话"
    assert turn.addressed_to == "ghost_near", f"挑错了人：{turn.addressed_to}"


def test_ask_peer_falls_back_when_everyone_was_asked():
    """都问过了就闭嘴 —— 反复问同一句"你那边怎么样"比不说话更糟。"""
    from npc_agent.types import AgentTurn

    cast = _cast()
    agent = cast.agents["ayou"]
    for _ in range(3):
        agent._ask_peer(AgentTurn(actor_id=agent.id, tick=0), _ctx(cast, agent))
    turn = AgentTurn(actor_id=agent.id, tick=0)
    assert not agent._ask_peer(turn, _ctx(cast, agent))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
