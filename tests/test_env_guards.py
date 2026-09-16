"""环境护栏测试。

这一组是"可控性"的底线：世界规则必须挡得住越权和不合理动作。
每一个失败都返回**结构化的原因**，而不是抛异常 —— 因为那个原因会被
Reflection 模块消费，变成 NPC 下次的教训。
"""

from __future__ import annotations

import pytest

from npc_agent.config import load_scenario
from npc_agent.env.star_isle import Item, StarIsleEnv
from npc_agent.types import ActionCall


@pytest.fixture()
def env() -> StarIsleEnv:
    return StarIsleEnv(load_scenario("tutorial"), "ayou", "阿柚")


def test_unknown_tool_returns_reason(env: StarIsleEnv) -> None:
    result = env.dispatch("ayou", ActionCall("teleport", {"to": "moon"}))
    assert not result.ok
    assert "没有名为" in result.detail


def test_craft_requires_correct_station(env: StarIsleEnv) -> None:
    """NPC 站在吧台就想做拿铁 —— 必须被拦下，并告诉它要去哪。"""
    result = env.dispatch("ayou", ActionCall("craft_item", {"recipe": "latte"}))
    assert not result.ok
    assert "后厨" in result.detail


def test_take_item_requires_being_there(env: StarIsleEnv) -> None:
    result = env.dispatch("ayou", ActionCall("take_item", {"item": "beans"}))
    assert not result.ok
    assert "move_to" in result.detail  # 提示了补救动作，Reflection 才能据此重规划


def test_take_item_is_idempotent(env: StarIsleEnv) -> None:
    env.dispatch("ayou", ActionCall("move_to", {"location": "kitchen"}))
    first = env.dispatch("ayou", ActionCall("take_item", {"item": "beans"}))
    second = env.dispatch("ayou", ActionCall("take_item", {"item": "beans"}))
    assert first.ok and second.ok  # 重复拿不该报错，否则重规划会陷入循环


def test_give_requires_colocation(env: StarIsleEnv) -> None:
    """能听见不等于能递过去。玩家在门口时，吧台的 NPC 递不了东西。"""
    env.actors["ayou"].inventory.append("latte")
    env.items["latte"] = Item("latte", holder="ayou")
    result = env.dispatch(
        "ayou", ActionCall("give_item", {"item": "latte", "player": "player_a"})
    )
    assert not result.ok
    assert "不在你身边" in result.detail


def test_full_latte_flow_succeeds(env: StarIsleEnv) -> None:
    """完整闭环：走到后厨 → 取材料 → 制作 → 回吧台 → 交付。"""
    steps = [
        ActionCall("move_to", {"location": "kitchen"}),
        ActionCall("take_item", {"item": "beans"}),
        ActionCall("take_item", {"item": "milk"}),
        ActionCall("craft_item", {"recipe": "latte"}),
        ActionCall("move_to", {"location": "counter"}),
    ]
    for step in steps:
        assert env.dispatch("ayou", step).ok, step.render()

    env.actors["player_a"].loc = "counter"
    result = env.dispatch(
        "ayou", ActionCall("give_item", {"item": "latte", "player": "player_a"})
    )
    assert result.ok
    assert "latte" in env.actors["player_a"].inventory


def test_tell_fact_blocks_locked_topic(env: StarIsleEnv) -> None:
    """剧透红线：未解锁的话题必须拒绝。"""
    result = env.dispatch("ayou", ActionCall("tell_fact", {"topic": "hidden_menu"}))
    assert not result.ok
    assert "剧透" in result.detail


def test_tell_fact_allows_unlocked_topic(env: StarIsleEnv) -> None:
    result = env.dispatch("ayou", ActionCall("tell_fact", {"topic": "brewing"}))
    assert result.ok
    assert result.state_delta.get("text")


def test_set_flag_whitelist_blocks_overreach(env: StarIsleEnv) -> None:
    """越权保护：NPC 只能改场景授权的标记。"""
    result = env.dispatch(
        "ayou", ActionCall("set_flag", {"key": "hidden_menu_unlocked", "value": "1"})
    )
    assert not result.ok
    assert "白名单" in result.detail


def test_set_flag_allows_whitelisted(env: StarIsleEnv) -> None:
    result = env.dispatch(
        "ayou", ActionCall("set_flag", {"key": "learned_order", "value": "1"})
    )
    assert result.ok
    assert "learned_order" in env.world_flags


def test_dispatch_never_raises(env: StarIsleEnv) -> None:
    """任何畸形参数都不能把异常抛出环境边界。"""
    for bad in (
        ActionCall("move_to", {}),
        ActionCall("move_to", {"location": None}),
        ActionCall("give_item", {"item": 1, "player": []}),
        ActionCall("judge_answer", {}),
    ):
        result = env.dispatch("ayou", bad)
        assert isinstance(result.ok, bool)


def test_objective_completion_by_world_condition(env: StarIsleEnv) -> None:
    """icebreaker 的目标完成条件不是"我说了话"，而是"玩家都开口了"。"""
    ice = StarIsleEnv(load_scenario("icebreaker"), "ayou", "阿柚")
    assert ice.objectives_status()["greet_all"] == "pending"
    for pid in ("player_a", "player_b", "player_c"):
        ice.record_player_utterance(pid, "大家好")
    assert ice.objectives_status()["greet_all"] == "done"
