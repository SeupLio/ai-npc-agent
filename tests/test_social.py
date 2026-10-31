"""共享记忆 / 传闻网络（`modules/social.py`）。

## 这个模块的价值不在"能传"，在"传的是判断而不是原话"

所有 NPC 都听得见玩家说话（`observe_utterance` 是广播），
所以"把原话再传一遍"是没有信息量的 —— 接收方早就听过。
真正只属于某个 NPC 的东西是**它的判断**：
「阿柚注意到阿澈口味偏酸」。这条判断会在角色之间流动，
于是另一个 NPC 能对第三个人说出「听阿柚说，阿澈喜欢偏酸的豆子」。

测试分两层：网络本身的契约（发布/去重/阈值/投递），
以及它在剧组里的真实效果（端到端）。
"""

from __future__ import annotations

import pytest

from npc_agent.cast import build_cast
from npc_agent.config import RuntimeConfig, load_scenario
from npc_agent.llm import NullLLM
from npc_agent.modules.social import (
    GossipNetwork,
    SharedFact,
    fact_key,
    preference_clause,
    to_third_person,
)


# --------------------------------------------------------------------------- #
# 文本层
# --------------------------------------------------------------------------- #
def test_preference_clause_picks_the_clause_that_talks_about_taste():
    assert preference_clause("我口味偏酸，酸一点的我都喜欢。") == "酸一点的我都喜欢"
    assert preference_clause("我喜欢偏酸的豆子") == "我喜欢偏酸的豆子"


def test_a_plain_statement_is_not_a_preference():
    """没有偏好词的句子不进网络 —— 否则传闻会把寒暄也传出去。"""
    assert preference_clause("今天这里挺热闹的") == ""


def test_a_question_about_a_preference_is_not_a_preference():
    """「你还记得我喜欢什么吗」含"喜欢"，但它没有提供任何信息。

    这条问句曾经被当成偏好发布出去，传闻网络里于是出现了一条
    「你还记得小满喜欢什么吗」—— 玩家看到的是 NPC 在背后复读别人的问题，
    而且它还会顶掉真正的那条偏好（同一个 fact_key 之后不再接收）。
    """
    assert preference_clause("你还记得我喜欢什么吗") == ""
    assert preference_clause("阿柚，你还记得我喜欢什么吗？") == ""
    assert preference_clause("我喜欢偏酸的豆子，越酸越好") == "我喜欢偏酸的豆子"


def test_third_person_replaces_every_first_person():
    """只换句首是不够的：「酸一点的我都喜欢」会剩下一个"我"。

    那样的话，接收方记住的是「听阿柚说，阿澈酸一点的我都喜欢」——
    一句话里两个主语，读起来像另一个人。
    """
    assert to_third_person("我喜欢偏酸的豆子", "阿澈") == "阿澈喜欢偏酸的豆子"
    assert to_third_person("酸一点的我都喜欢", "阿澈") == "酸一点的阿澈都喜欢"


# --------------------------------------------------------------------------- #
# 网络契约
# --------------------------------------------------------------------------- #
def _fact(content: str, *, source: str = "ayou", importance: float = 0.75) -> SharedFact:
    return SharedFact(
        fact_id=fact_key(content),
        content=content,
        source_id=source,
        source_name="阿柚",
        tick=1,
        importance=importance,
        subject="player_a",
    )


def test_the_same_fact_is_only_published_once():
    net = GossipNetwork()
    assert net.publish(_fact("阿澈喜欢偏酸的豆子")) is True
    assert net.publish(_fact("阿澈喜欢偏酸的豆子")) is False
    assert len(net.facts()) == 1


def test_low_importance_facts_are_filtered():
    """阈值是"别把寒暄也传成传闻"的那道闸门。"""
    net = GossipNetwork(threshold=0.7)
    assert net.publish(_fact("今天天气不错", importance=0.3)) is False
    assert net.filtered_count == 1
    assert net.facts() == []


def test_the_source_does_not_receive_its_own_fact():
    net = GossipNetwork()
    net.publish(_fact("阿澈喜欢偏酸的豆子", source="ayou"))
    assert net.digest("ayou") == []
    assert len(net.digest("xiaozhou")) == 1


def test_a_fact_is_delivered_only_once_per_actor():
    """同一件事在一个 NPC 的记忆里出现两次，会让它反复提这件事。"""
    net = GossipNetwork()
    net.publish(_fact("阿澈喜欢偏酸的豆子"))
    assert len(net.digest("xiaozhou")) == 1
    assert net.digest("xiaozhou") == []


# --------------------------------------------------------------------------- #
# 端到端：剧组里真的传开了
# --------------------------------------------------------------------------- #
def test_an_impression_travels_between_npcs():
    """阿柚听见玩家的偏好 → 小舟后来能对**别人**说出来。

    这条断言覆盖了整条链：偏好抽取 → 第三人称重写 → 发布 → 投递 →
    记忆写入（带来源）→ 检索放行（传闻是第三人称，不受"关于说话人"的限制）→
    用传闻口气说出（`heard_from` 模板）。
    """
    cast = build_cast(load_scenario("duet"), NullLLM(), RuntimeConfig())
    cast.step(cast.env.record_player_utterance("player_a", "我喜欢偏酸的豆子。"))
    for _ in range(6):
        cast.step(None)

    gossip_memories = [
        r for r in cast.agents["xiaozhou"].memory.store.records if "gossip" in r.tags
    ]
    assert gossip_memories, "传闻一条都没投递到小舟那里"
    assert "阿柚" in gossip_memories[0].content
    assert "阿澈" in gossip_memories[0].content

    turns = cast.step(
        cast.env.record_player_utterance("player_b", "小舟，你还记得阿澈喜欢什么吗？")
    )
    says = {t.actor_id: t.say for t in turns if t.say}
    assert "xiaozhou" in says, "小舟没有回答"
    assert "阿柚" in says["xiaozhou"], f"没有说出信息来源：{says['xiaozhou']}"


def test_gossip_can_be_switched_off():
    """`gossip_enabled=False` 必须真的关掉它 —— 配置项不是装饰。"""
    cast = build_cast(
        load_scenario("duet"), NullLLM(), RuntimeConfig(gossip_enabled=False)
    )
    cast.step(cast.env.record_player_utterance("player_a", "我喜欢偏酸的豆子。"))
    for _ in range(4):
        cast.step(None)
    assert cast.gossip is None
    assert not [
        r for r in cast.agents["xiaozhou"].memory.store.records if "gossip" in r.tags
    ]


def test_a_plain_statement_creates_no_gossip():
    """寒暄不该在世界上留下传闻 —— 否则"信息流通"会变成"噪音流通"。"""
    cast = build_cast(load_scenario("duet"), NullLLM(), RuntimeConfig())
    cast.step(cast.env.record_player_utterance("player_a", "今天这里挺热闹的。"))
    for _ in range(4):
        cast.step(None)
    assert cast.gossip is not None
    assert cast.gossip.facts() == []


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
