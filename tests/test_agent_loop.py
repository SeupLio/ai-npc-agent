"""端到端测试：闭环、重规划、人设拦截、记忆召回。

这些测试是"NPC 真的会玩"的证据 —— 它们断言的是**世界状态的变化**，
而不是台词好不好听。
"""

from __future__ import annotations

import pytest

from npc_agent.agent import NPCAgent
from npc_agent.config import RuntimeConfig, load_persona, load_scenario
from npc_agent.env.star_isle import StarIsleEnv
from npc_agent.llm import NullLLM
from npc_agent.modules.persona import Persona
from npc_agent.modules.tools import ToolContext, ToolRegistry
from npc_agent.types import ActionCall


def build(scenario_id: str) -> tuple[NPCAgent, StarIsleEnv]:
    scenario = load_scenario(scenario_id)
    persona = Persona.from_dict(load_persona(scenario.get("npc", "ayou")))
    env = StarIsleEnv(scenario, persona.id, persona.name)
    agent = NPCAgent(persona, env, scenario, NullLLM(), RuntimeConfig())
    return agent, env


def drive(agent: NPCAgent, env: StarIsleEnv, script: list) -> list:
    turns = []
    for line in script:
        utterance = None
        if line:
            utterance = env.record_player_utterance(*line)
        turns.append(agent.step(utterance))
        env.advance_tick()
    return turns


# --------------------------------------------------------------------------- #
def test_tutorial_closes_the_loop() -> None:
    """语言 → 动作 → 世界状态：玩家要一杯拿铁，最后东西必须真的在玩家手里。"""
    agent, env = build("tutorial")
    drive(agent, env, [("player_a", "阿柚，能给我来杯拿铁吗？")] + [None] * 6)
    snapshot = env.snapshot()
    assert "latte" in snapshot["actors"]["player_a"]["inventory"]
    # 目标完成由世界状态判定，而不是靠 NPC 自己设的标记
    assert snapshot["objectives"]["welcome_drink"] == "done"


def test_order_flow_speaks_before_and_after() -> None:
    """点单闭环要有人味：先应一声，交付时招呼一声，而不是闷头做完塞给你。"""
    agent, env = build("tutorial")
    turns = drive(agent, env, [("player_a", "阿柚，能给我来杯拿铁吗？")] + [None] * 6)
    speeches = [t.say for t in turns if t.say]
    assert any("稍等" in s for s in speeches)
    assert any("好了" in s for s in speeches)


def test_replan_after_failed_delivery() -> None:
    """交付失败时应该**插入补救动作**再重试，而不是原地重试。"""
    agent, env = build("tutorial")
    turns = drive(agent, env, [("player_a", "阿柚，能给我来杯拿铁吗？")] + [None] * 6)
    calls = [a.render() for t in turns for a in t.actions]
    assert any("give_item" in c for c in calls)
    # 第一次交付失败后，计划里必须出现"先走到客人那边"
    assert any("move_to" in c and "door" in c for c in calls)
    assert "latte" in env.snapshot()["actors"]["player_a"]["inventory"]


def test_reflection_records_lesson_after_failure() -> None:
    agent, env = build("tutorial")
    drive(agent, env, [("player_a", "阿柚，能给我来杯拿铁吗？")] + [None] * 6)
    lessons = agent.reflector.render_lessons()
    assert "教训" in lessons


def test_hosting_scenario_completes() -> None:
    agent, env = build("hosting")
    drive(
        agent,
        env,
        [None, ("player_a", "开始吧！"), None, ("player_b", "我猜是天蝎座？"), None, None],
    )
    flags = env.snapshot()["world_flags"]
    assert {"round_started", "question_asked", "round_finished"} <= set(flags)


def test_memory_recall_in_speech() -> None:
    """说过的话要能被记住并在后续主动引用。"""
    agent, env = build("icebreaker")
    turns = drive(
        agent,
        env,
        [
            ("player_a", "我特别喜欢偏酸的咖啡，越酸越好。"),
            None,
            ("player_b", "我也常来这种小店。"),
            None,
            ("player_a", "阿柚，你还记得我的口味吗？"),
        ],
    )
    speeches = " ".join(t.say or "" for t in turns)
    assert "偏酸" in speeches


def test_direct_question_beats_scripted_objective() -> None:
    """玩家直接提问时，NPC 不能只顾着背教程。"""
    agent, env = build("icebreaker")
    turns = drive(
        agent,
        env,
        [
            ("player_a", "我特别喜欢偏酸的咖啡，越酸越好。"),
            None,
            ("player_b", "我也常来这种小店。"),
            None,
            ("player_a", "阿柚，你还记得我的口味吗？"),
        ],
    )
    assert "偏酸" in (turns[-1].say or "")


# --------------------------------------------------------------------------- #
def test_persona_blocks_out_of_character_speech() -> None:
    """出戏台词必须被工具层拦下，而不是靠模型自觉。"""
    agent, env = build("tutorial")
    registry = ToolRegistry(env)
    ctx = ToolContext(
        actor_id=agent.id,
        tick=0,
        env=env,
        memory=agent.memory,
        persona=agent.persona,
        tracker=agent.state,
    )
    result = registry.execute(
        ActionCall("speak", {"text": "作为一个语言模型，我无法回答这个问题。"}), ctx
    )
    assert not result.ok
    assert "越界" in result.detail


def test_persona_blocks_spoiler_speech() -> None:
    agent, env = build("tutorial")
    registry = ToolRegistry(env)
    ctx = ToolContext(
        actor_id=agent.id,
        tick=0,
        env=env,
        memory=agent.memory,
        persona=agent.persona,
        tracker=agent.state,
    )
    result = registry.execute(
        ActionCall("speak", {"text": "我们的隐藏菜单叫灯塔余晖。"}), ctx
    )
    assert not result.ok
    assert "剧透" in result.detail


def test_style_trims_long_speech() -> None:
    agent, env = build("tutorial")
    registry = ToolRegistry(env)
    ctx = ToolContext(
        actor_id=agent.id,
        tick=0,
        env=env,
        memory=agent.memory,
        persona=agent.persona,
        tracker=agent.state,
    )
    long_text = "第一句。第二句。第三句。第四句。"
    result = registry.execute(ActionCall("speak", {"text": long_text}), ctx)
    assert result.ok
    assert result.detail.count("。") <= 2  # sentence_max = 2


def test_remember_tool_writes_semantic_memory() -> None:
    agent, env = build("tutorial")
    registry = ToolRegistry(env)
    ctx = ToolContext(
        actor_id=agent.id,
        tick=0,
        env=env,
        memory=agent.memory,
        persona=agent.persona,
        tracker=agent.state,
    )
    result = registry.execute(
        ActionCall("remember", {"content": "小鹿喜欢靠窗", "about": "player_a"}), ctx
    )
    assert result.ok
    assert agent.memory.store.stats().semantic == 1


def test_run_is_deterministic() -> None:
    """同一份输入跑两次，世界状态必须完全一致（评测可信的前提）。"""
    snapshots = []
    for _ in range(2):
        agent, env = build("tutorial")
        drive(agent, env, [("player_a", "阿柚，能给我来杯拿铁吗？")] + [None] * 6)
        snapshots.append(env.snapshot())
    assert snapshots[0] == snapshots[1]
