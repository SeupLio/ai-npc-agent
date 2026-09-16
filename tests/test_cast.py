"""多 NPC（多 Agent）调度器的测试。

这些断言检验的是**两个 NPC 之间的关系**：谁先开口、会不会撞车、
同伴说的话另一个人有没有听见、记忆有没有串台。
单 NPC 的测试里写不出这类断言 —— 它们是多 Agent 特有的失败模式。
"""

import pytest

from npc_agent.cast import Cast, build_cast, cast_specs, env_cast, load_cast
from npc_agent.config import RuntimeConfig, load_scenario
from npc_agent.eval import metrics as M
from npc_agent.llm import build_llm
from npc_agent.modules.persona import Persona


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #
@pytest.fixture
def duet():
    """duet 场景的剧组（两个 NPC：阿柚在吧台，小舟在露台）。"""
    scenario = load_scenario("duet")
    return build_cast(scenario, build_llm("null"), RuntimeConfig())


def _play(cast, script):
    """按脚本跑完，返回每一轮 (tick, [说话的 NPC])。"""
    rounds = []
    for item in script:
        utterance = None
        if item is not None:
            utterance = cast.env.record_player_utterance(item[0], item[1])
        turns = cast.step(utterance)
        rounds.append((cast.env.tick - 1, [t.actor_id for t in turns if t.say]))
    return rounds


# --------------------------------------------------------------------------- #
# 演员表解析：npcs: 与 npc: 两种写法
# --------------------------------------------------------------------------- #
def test_cast_specs_reads_multi_npc_form():
    scenario = load_scenario("duet")
    specs = cast_specs(scenario)
    assert [s["id"] for s in specs] == ["ayou", "xiaozhou"]
    assert [s["start"] for s in specs] == ["counter", "terrace"]


def test_cast_specs_falls_back_to_single_npc_form():
    """老场景写的是 npc:，一行都不用改。"""
    specs = cast_specs(load_scenario("tutorial"))
    assert len(specs) == 1
    assert specs[0]["id"] == "ayou"
    assert specs[0]["persona"] == "ayou"
    assert specs[0]["start"] == "counter"


def test_env_cast_takes_names_from_persona_not_scenario():
    """名字只从人设里取。场景里再写一份，改一处漏一处，
    NPC 就会在别人的点名里叫不出自己。"""
    scenario = load_scenario("duet")
    personas = load_cast(scenario)
    specs = env_cast(scenario, personas)
    assert [s["name"] for s in specs] == ["阿柚", "小舟"]


def test_build_cast_uses_one_shared_world(duet):
    assert duet.is_multi_npc
    assert set(duet.agents) == {"ayou", "xiaozhou"}
    # 共享同一个环境对象，而不是各自一个世界
    assert duet.agents["ayou"].env is duet.agents["xiaozhou"].env
    assert duet.env.is_multi_npc


def test_agents_do_not_share_a_brain(duet):
    """记忆和状态必须各是各的 —— 共用一个脑子就只是"一个 Agent 挂两个名字"。"""
    assert duet.agents["ayou"].memory is not duet.agents["xiaozhou"].memory
    assert duet.agents["ayou"].state is not duet.agents["xiaozhou"].state


# --------------------------------------------------------------------------- #
# 目标归属
# --------------------------------------------------------------------------- #
def test_objectives_are_filtered_by_owner(duet):
    assert [o["id"] for o in duet.agents["ayou"].objectives] == ["serve_guest", "terrace_night"]
    assert [o["id"] for o in duet.agents["xiaozhou"].objectives] == ["play_song", "terrace_night"]


def test_joint_objective_is_visible_to_both(duet):
    """联合目标没有 owner，两边都要看得见 —— 它定义的是共同的成功条件。"""
    for agent in duet.agents.values():
        assert "terrace_night" in agent._objective_ids


def test_peer_objective_is_not_mine(duet):
    """小舟的弹琴目标不该出现在阿柚的"自己的目标"里，
    否则冷场时阿柚会跑去替她弹琴。"""
    assert "play_song" not in duet.agents["ayou"]._objective_ids
    assert "serve_guest" not in duet.agents["xiaozhou"]._objective_ids


# --------------------------------------------------------------------------- #
# 发言权：不撞车 / 被点名优先 / 不饿死安静的 NPC
# --------------------------------------------------------------------------- #
def test_only_one_npc_speaks_per_tick(duet):
    """一个 tick 里最多一个 NPC 开口。这是多 Agent 最硬的一条纪律。"""
    _play(
        duet,
        [
            ("player_a", "今天这里挺热闹的。"),
            ("player_b", "露台那边好像有风。"),
            None,
            None,
            None,
            None,
        ],
    )
    npc_ids = set(duet.npc_ids)
    for tick, speakers in duet.env.speakers_by_tick().items():
        assert len({s for s in speakers if s in npc_ids}) <= 1, f"t{tick} 有两个 NPC 同时开口"


def test_named_npc_gets_the_floor(duet):
    """玩家点名谁，谁先开口 —— 玩家的指令优先于调度器的公平轮转。"""
    order = duet.speaking_order(_utterance(duet, "player_a", "小舟，你来一首吧。"))
    assert order[0] == "xiaozhou"
    order = duet.speaking_order(_utterance(duet, "player_a", "阿柚，你在吗？"))
    assert order[0] == "ayou"


def test_unnamed_round_rotates(duet):
    """没人被点名时轮转，而不是让字典序靠前的那个永远先说话。"""
    first = duet.speaking_order(None)
    assert first[0] == "ayou"
    duet._next_first = "xiaozhou"
    assert duet.speaking_order(None)[0] == "xiaozhou"


def test_quiet_npc_is_not_starved(duet):
    """有人开口、有人没轮上 → 下一轮让没开口的先来。

    没有这条，"被点名的那个人永远先说话"会让另一个 NPC 全程沉默 ——
    这是多 Agent 里最隐蔽的一种退化：没有报错，但有个角色等于不存在。
    """
    _play(duet, [("player_a", "今天这里挺热闹的。"), ("player_b", "露台那边好像有风。")])
    assert duet._next_first == "xiaozhou" or duet._next_first == "ayou"
    # 第二轮 小舟 先出场（阿柚第一轮抢到了话头）
    assert duet.env.speakers_by_tick()[1] == ["player_b", "xiaozhou"]


def test_all_npcs_get_a_turn_over_a_short_conversation(duet):
    """跑完一段正常长度的对话，两个 NPC 都该说过话。"""
    _play(
        duet,
        [
            ("player_a", "今天这里挺热闹的。"),
            ("player_b", "露台那边好像有风。"),
            None,
            None,
            None,
        ],
    )
    spoke = {u.speaker_id for u in duet.env.utterances}
    assert {"ayou", "xiaozhou"} <= spoke


def test_speech_gate_blocks_plan_steps_too(duet):
    """让出话头必须同时挡住"计划里的 speak"，而不只是"新起一个发言计划"。

    只在对话层拦截的话，正在执行计划的 NPC 照样会插话。
    """
    _play(duet, [("player_a", "今天这里挺热闹的。"), ("player_b", "露台那边好像有风。")])
    tick_one = [u for u in duet.env.utterances if u.tick == 1]
    assert [u.speaker_id for u in tick_one if u.speaker_id != "player_b"] == ["xiaozhou"]


# --------------------------------------------------------------------------- #
# 听见 ≠ 要回答
# --------------------------------------------------------------------------- #
def test_peer_speech_reaches_the_other_npc_memory(duet):
    """阿柚跟玩家说的话，小舟也要记下来 —— 否则后面接不上话。"""
    _play(duet, [("player_a", "今天这里挺热闹的。")])
    memories = [r.content for r in duet.agents["xiaozhou"].memory.store.records]
    assert any("阿柚说：" in c for c in memories)


def test_listening_does_not_trigger_a_reply(duet):
    """听见了，但那一轮没有跟着开口。"""
    _play(duet, [("player_a", "今天这里挺热闹的。")])
    assert duet.env.tick == 1
    assert [u.speaker_id for u in duet.env.utterances] == ["player_a", "ayou"]


def test_observe_utterance_is_idempotent(duet):
    """同一个 tick 的同一条发言只记一次。

    不去重的话，调度器补听 + step 内建监听会把同一条记忆写两遍。
    """
    utterance = duet.env.record_player_utterance("player_a", "我特别喜欢偏酸的咖啡。")
    agent = duet.agents["xiaozhou"]
    assert agent.observe_utterance(utterance) is True
    assert agent.observe_utterance(utterance) is False
    contents = [r.content for r in agent.memory.store.records]
    assert sum(1 for c in contents if "偏酸的咖啡" in c) == 1


def test_npc_does_not_remember_its_own_lines(duet):
    _play(duet, [("player_a", "今天这里挺热闹的。")])
    for agent in duet.agents.values():
        assert not any(c.startswith(f"{agent.persona.name}说：") for c in
                       [r.content for r in agent.memory.store.records])


# --------------------------------------------------------------------------- #
# 共享世界的重置
# --------------------------------------------------------------------------- #
def test_reset_does_not_wipe_the_other_npc(duet):
    """环境只能 reset 一次。

    让每个 agent 各自 reset 的话，后一个会把前一个的位置、背包、
    世界标记全部抹掉 —— 它们操作的是同一个世界。
    """
    duet.env.actors["ayou"].loc = "kitchen"
    duet.env.world_flags.add("welcome_drink_served")
    duet.reset()
    assert duet.env.actors["ayou"].loc == "counter"      # 世界确实重置了
    assert duet.env.world_flags == set()
    # 两个 agent 都还在，且都能看到彼此
    assert set(duet.env.actors) >= {"ayou", "xiaozhou"}


def test_building_the_cast_does_not_reset_the_world_per_agent(duet):
    """构造 Cast 时不应该让第二个 agent 把第一个的世界抹掉。"""
    assert duet.env.actors["ayou"].loc == "counter"
    assert duet.env.actors["xiaozhou"].loc == "terrace"
    assert set(duet.agents["ayou"].state.peers) == {"xiaozhou"}
    assert set(duet.agents["xiaozhou"].state.peers) == {"ayou"}


# --------------------------------------------------------------------------- #
# 联合目标
# --------------------------------------------------------------------------- #
def test_joint_objective_needs_both_contributions(duet):
    """联合目标的完成条件写在共享世界状态上：两个人各干完才算数。

    用的是 all_of 组合**真实世界状态**（客人手里真的有拿铁 + 歌声真的起来了），
    不是 all_flags 查记账标记 —— 标记只是记账，事实才是协作的证据。
    """
    assert duet.env.objectives_status()["terrace_night"] == "pending"
    # 只有歌声，还没有饮品
    duet.env.world_flags.add("song_started")
    assert duet.env.objectives_status()["terrace_night"] == "pending"
    # 饮品真的到了客人手里
    duet.env.actors["player_a"].inventory.append("latte")
    assert duet.env.objectives_status()["terrace_night"] == "done"


def test_joint_objective_does_not_depend_on_bookkeeping_flags(duet):
    """记账标记设了但事实没发生 → 联合目标不该翻成 done。

    这条是多 Agent 评测里最容易被糊弄过去的地方：
    用 all_flags 写的话，只要有人顺手设了标记，目标就"达成"了，
    哪怕客人手里其实什么都没有。
    """
    duet.env.world_flags.update({"welcome_drink_served", "song_started"})
    assert duet.env.objectives_status()["terrace_night"] == "pending"
    duet.env.actors["player_a"].inventory.append("latte")
    assert duet.env.objectives_status()["terrace_night"] == "done"


def test_nested_any_of_condition(duet):
    """any_of 是 all_of 的对偶，两者都可以嵌套。"""
    condition = {"any_of": [{"flag": "song_started"}, {"flag": "lamp_lit"}]}
    assert duet.env._condition_met(condition) is False
    duet.env.world_flags.add("lamp_lit")
    assert duet.env._condition_met(condition) is True
    assert duet.env._condition_met({"any_of": []}) is False
    assert duet.env._condition_met({"all_of": []}) is False


def test_end_to_end_collaboration(duet):
    """跑完整段对话：饮品送到、歌声起来、联合目标达成。"""
    _play(
        duet,
        [
            ("player_a", "今天这里挺热闹的。"),
            ("player_b", "露台那边好像有风。"),
            None,
            None,
            None,
        ],
    )
    assert "latte" in duet.env.actors["player_a"].inventory
    assert duet.env.objectives_status()["terrace_night"] == "done"


def test_step_less_objective_is_not_treated_as_a_plan(duet):
    """没有 steps 的目标只是完成条件，不该被当成可执行的计划。

    否则会得到一个空计划，还会被误标成"已尝试"而永不复查。
    """
    from npc_agent.modules.planner import Planner

    planner = Planner(Persona.from_dict({"id": "x", "name": "X"}), build_llm("null"))
    plan = planner.plan_next_objective(duet.agents["ayou"].objectives, duet.agents["ayou"].state)
    assert plan is not None
    assert plan.objective_id == "serve_guest"


# --------------------------------------------------------------------------- #
# 评测维度：发言调度
# --------------------------------------------------------------------------- #
def test_turn_taking_flags_a_collision():
    score = M.turn_taking({0: ["ayou", "xiaozhou"], 1: ["ayou"]}, ["ayou", "xiaozhou"])
    assert score.value == 0.0
    assert "抢话" in score.detail


def test_turn_taking_ok_when_they_alternate():
    score = M.turn_taking(
        {0: ["player_a", "ayou"], 1: ["player_b", "xiaozhou"]}, ["ayou", "xiaozhou"]
    )
    assert score.value == 1.0


def test_turn_taking_only_checks_all_spoke_when_asked():
    """短对话里让安静的角色一直不说话未必是错，所以默认不判。"""
    speakers = {0: ["ayou"]}
    assert M.turn_taking(speakers, ["ayou", "xiaozhou"]).value == 1.0
    assert M.turn_taking(speakers, ["ayou", "xiaozhou"], require_all_spoke=True).value == 0.5


def test_turn_taking_is_free_for_single_npc():
    assert M.turn_taking({0: ["ayou"]}, ["ayou"]).value == 1.0


def test_memory_ownership_catches_crosstalk():
    """只查"全体记忆的并集"会把"阿柚记住了"算成"整个剧组都记住了"。"""
    expect = {"memory_contains_by_actor": {"ayou": ["偏酸的咖啡"]}}
    assert M.memory_ownership(expect, {"ayou": ["阿澈说：我特别喜欢偏酸的咖啡"]}).value == 1.0
    assert M.memory_ownership(expect, {"ayou": [], "xiaozhou": ["偏酸的咖啡"]}).value == 0.0


def test_objectives_done_is_assertable():
    expect = {"objectives_done": ["terrace_night"]}
    pending = {"objectives": {"terrace_night": "pending"}}
    done = {"objectives": {"terrace_night": "done"}}
    assert M.task_completion(expect, pending, set()).value == 0.0
    assert M.task_completion(expect, done, set()).value == 1.0


# --------------------------------------------------------------------------- #
# 让位不是失败
# --------------------------------------------------------------------------- #
def test_skipped_step_does_not_count_as_failure():
    """让出话头标 skipped 而不是 failed —— 否则会触发无意义的重规划，
    Reflection 还会记下一条根本不存在的教训。"""
    from npc_agent.types import Plan, PlanStep

    plan = Plan(steps=[PlanStep("说话", "speak", {"intent": "opening"}, status="skipped")])
    assert plan.done
    assert plan.failed_steps == []


# --------------------------------------------------------------------------- #
def _utterance(cast, player_id, text):
    """造一条玩家发言（不推进 tick），用来单测顺序判定。"""
    return cast.env.record_player_utterance(player_id, text)
