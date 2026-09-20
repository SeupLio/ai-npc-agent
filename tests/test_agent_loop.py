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
from npc_agent.llm.base import LLMUnavailable
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


def test_style_does_not_eat_a_line_that_starts_with_an_ellipsis() -> None:
    """省略号不是句末。

    小舟的开场台词是「……你好。」。把「…」当成句子边界的话，
    这句会被切成 ["…", "…", "你好。"] 三段，sentence_max=2 一裁
    就只剩「……」—— 整句台词凭空消失，而且不报任何错。
    """
    persona = Persona.from_dict(load_persona("xiaozhou"))
    assert persona.style["sentence_max"] == 2
    assert persona.apply_style("……你好。") == "……你好。"
    assert persona.sentence_count("……你好。") == 1


def test_consecutive_punctuation_counts_as_one_sentence() -> None:
    """「真的吗？！」是一个人问了一句话，不是两句。"""
    persona = Persona.from_dict(load_persona("xiaozhou"))
    assert persona.sentence_count("真的吗？！") == 1
    assert persona.sentence_count("好。第一句。第二句。第三句。") == 4


def test_replan_goes_to_the_item_not_to_where_i_already_stand() -> None:
    """重规划必须推断出**目的地**，而不是"原因串里第一个被提到的地点"。

    take_item 失败的原因是「柠檬在后厨，你现在在吧台，需要先 move_to 过去」。
    取第一个提到的地点会得到「吧台」—— NPC 原地 move_to 到自己已经站着的
    地方，再试一次还是失败，永远拿不到东西。
    """
    agent, env = build("tutorial")
    # 玩家点柠檬水：柠檬在后厨，而阿柚站在吧台 → 必然触发一次位置纠错
    drive(agent, env, [("player_a", "阿柚，能给我来杯柠檬水吗？")] + [None] * 6)
    calls = [a.render() for t in agent.turns for a in t.actions]
    assert any("move_to" in c and "kitchen" in c for c in calls), calls
    assert "lemonade" in env.snapshot()["actors"]["player_a"]["inventory"]


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


# --------------------------------------------------------------------------- #
# 规划失败必须留下痕迹
#
# 规划调用失败时会**静默回落到启发式规划**（见 NPCAgent._decide_plan 的兜底分支）。
# 回落本身是对的 —— 一次调用失败不该让 NPC 卡住 —— 但它带来一个测量陷阱：
# `--no-planner` 和"planner 开着但一直在失败"跑出来的轨迹**完全一样**。
# 于是"接上模型规划到底有没有用"这个对照实验，可能在读者不知情的情况下
# 变成自己跟自己比。所以失败次数和原因必须能被读到。
# --------------------------------------------------------------------------- #
class _BrokenPlannerLLM(NullLLM):
    """`available` 为真、但每次调用都失败 —— 模拟端点挂了或预算被思维链吃光。

    这两种原因在报错里长得不一样，但对规划器来说都是 `LLMUnavailable`，
    所以用同一个假件覆盖。
    """

    name = "broken"

    @property
    def available(self) -> bool:
        return True

    def complete(self, messages, *, temperature=0.7, max_tokens=512):  # type: ignore[override]
        raise LLMUnavailable(
            "模型返回空内容（finish_reason=length，思维链 3905 字）。"
            "推理模型需要更大的 max_tokens"
        )


def _build_with_llm(scenario_id: str, llm, planner_on: bool):
    scenario = load_scenario(scenario_id)
    persona = Persona.from_dict(load_persona(scenario.get("npc", "ayou")))
    env = StarIsleEnv(scenario, persona.id, persona.name)
    config = RuntimeConfig(use_llm_planner=planner_on)
    return NPCAgent(persona, env, scenario, llm, config), env


def test_a_failing_planner_falls_back_to_heuristics_and_says_so() -> None:
    """回落要发生（NPC 不能因为一次调用失败就卡住），但**必须留下痕迹**。"""
    agent, env = _build_with_llm("tutorial", _BrokenPlannerLLM(), planner_on=True)
    drive(agent, env, [("player_a", "阿柚，能给我来杯拿铁吗？")] + [None] * 6)

    # 回落生效：目标照样完成了（和 --no-planner 走的是同一条启发式路径）
    assert env.snapshot()["actors"]["player_a"]["inventory"] == ["latte"]
    # 但痕迹也在：否则这条轨迹和 --no-planner 完全无法区分
    assert agent.planner_failures > 0, "规划一直在失败，却一次都没被记下来"
    assert "空内容" in agent.planner_last_error
    assert "max_tokens" in agent.planner_last_error


def test_planner_off_does_not_count_as_a_planner_failure() -> None:
    """`--no-planner` 是"没开这一路"，不是"失败了 N 次"。

    两者混在一起，对照组会被记成一片红 —— 而它恰恰是基线。
    """
    agent, env = _build_with_llm("tutorial", _BrokenPlannerLLM(), planner_on=False)
    drive(agent, env, [("player_a", "阿柚，能给我来杯拿铁吗？")] + [None] * 6)
    assert agent.planner_failures == 0


def test_an_unconfigured_model_is_not_a_planner_failure() -> None:
    """离线跑批（NullLLM）同理：没配模型不是失败，但要能解释为什么走了启发式。"""
    agent, env = _build_with_llm("tutorial", NullLLM(), planner_on=True)
    drive(agent, env, [None] * 3)
    assert agent.planner_failures == 0
    assert "未配置模型" in agent.planner_last_error

# --------------------------------------------------------------------------- #
# 计划来源：**回落是静默的**，所以来源必须能被读到
#
# 实测背景（2026-09-19，231 条跑批）：**48% 的用例至少回落过一次**。
# 回落前后的轨迹完全一样 —— 不标来源，"模型规划到底有没有生效"
# 在任何地方都看不出来，包括控制台。
# --------------------------------------------------------------------------- #
class _ScriptedPlanLLM(NullLLM):
    """`available` 为真、`complete` 返回一段写死的文本。

    用来分别制造"可用的计划"和"调用成功但计划不可用"两种情况。
    后者**从前一次都不会被记** —— 它不抛异常，所以不计数。
    """

    name = "scripted"

    def __init__(self, payload: str) -> None:
        self.payload = payload

    @property
    def available(self) -> bool:
        return True

    def complete(self, messages, *, temperature=0.7, max_tokens=512):  # type: ignore[override]
        return self.payload


#: 可用：有 goal、有一步带 tool。
_USABLE_PLAN = (
    '{"goal": "招呼客人", "rationale": "先开口", "steps": '
    '[{"goal": "打招呼", "tool": "speak", "args": {"text": "欢迎光临"}}]}'
)
#: 解析得出来，但 `steps` 是空的。
_EMPTY_PLAN = '{"goal": "招呼客人", "rationale": "先开口", "steps": []}'
#: 解析得出来，但每一步都缺 `tool`。
_TOOLLESS_PLAN = '{"goal": "招呼客人", "steps": [{"goal": "打招呼"}]}'


def test_every_plan_says_who_produced_it() -> None:
    """计划必须带来源，而且来源只能是登记过的那几个。"""
    from npc_agent.types import PLAN_SOURCES

    agent, env = _build_with_llm(
        "tutorial", _ScriptedPlanLLM(_USABLE_PLAN), planner_on=True
    )
    drive(agent, env, [None] * 3)

    sources = agent.plan_sources()
    assert sources, "一个计划都没记来源 —— 那就分不清模型规划和静默回落"
    assert set(sources) <= set(PLAN_SOURCES), f"出现了没登记的来源：{set(sources)}"


def test_a_model_plan_is_tagged_model() -> None:
    """模型真给出计划时来源是 `model`，不能被兜底那条覆盖掉。"""
    agent, env = _build_with_llm(
        "tutorial", _ScriptedPlanLLM(_USABLE_PLAN), planner_on=True
    )
    drive(agent, env, [None] * 3)
    assert agent.plan_sources().get("model", 0) > 0
    assert agent.planner_failures == 0
    assert agent.planner_empty_plans == 0


@pytest.mark.parametrize("payload", [_EMPTY_PLAN, _TOOLLESS_PLAN, ""])
def test_an_unusable_plan_is_counted_not_swallowed(payload: str) -> None:
    """⚠️ 这一条补的是一个**从来没被计数过**的洞。

    模型调用成功、JSON 也解析出来了，但 `steps` 不可用 —— 从前这一支
    直接 `return None`，**什么都没记**。于是"模型给了不可用的计划"
    和"模型给了可用计划"在数据里的区别消失了，而报告里却写着
    "解析失败 0 条"（那句话只覆盖了抛异常的那一类）。

    **不抛异常不等于成功。**
    """
    agent, env = _build_with_llm(
        "tutorial", _ScriptedPlanLLM(payload), planner_on=True
    )
    drive(agent, env, [None] * 3)

    assert agent.planner_empty_plans > 0, "不可用的计划被静默吞掉了"
    assert agent.planner_failures == 0, "这不是抛异常那一类，别混在一起"
    assert "不可用" in agent.planner_last_error
    # 回落仍然发生（NPC 不能因为模型答不好就卡住），但这次它留了痕迹
    assert agent.plan_sources().get("heuristic", 0) > 0


def test_planner_off_tags_heuristic_but_records_no_failure() -> None:
    """`--no-planner` 也走启发式，但它**不是回落**。

    靠配置（`use_llm_planner`）区分，不靠来源 —— 否则对照组会被记成一片红，
    而它恰恰是基线。
    """
    agent, env = _build_with_llm("tutorial", NullLLM(), planner_on=False)
    drive(agent, env, [None] * 3)
    assert agent.plan_sources().get("heuristic", 0) > 0
    assert agent.planner_failures == 0
    assert agent.planner_empty_plans == 0
