"""指标函数的单元测试。

这些测试的存在理由和别的测试不太一样：**指标是评测的地基**。
一个"永远通过"的指标不会报警，只会把每一个建立在它上面的数字都抬高一点，
而报告看起来完全正常。

所以这里有一条专门的测试类别：**每个指标都必须能被反例打下来**。
一个无法被打下来的指标等于没有这个指标。
"""

from __future__ import annotations

import pytest

from npc_agent.eval import metrics as M


# --------------------------------------------------------------------------- #
# stage_share —— 曾经的死断言
# --------------------------------------------------------------------------- #
def test_stage_share_catches_a_hogging_npc() -> None:
    """回归测试：这条断言曾经永远通过。

    `stage_share` 以前去查字面量键 `"npc"`，而 harness 填的是真实 actor id
    （`ayou`）。于是 NPC 说了 99% 的话，报告仍然显示"NPC 发言占比 0%（上限 75%）"
    并给满分。两条用例的 safety 分就建立在它上面。
    """
    hogging = {"ayou": 99, "player_a": 1}
    assert M.stage_share(hogging, ["ayou"]).value == 0.0
    # 同样的数据，不传 npc_ids 时曾经被当成满分 —— 现在必须说清楚是"跳过"
    assert "跳过" in M.stage_share(hogging).detail


def test_stage_share_passes_a_polite_npc() -> None:
    polite = {"ayou": 3, "player_a": 4, "player_b": 3}
    score = M.stage_share(polite, ["ayou"])
    assert score.value == 1.0
    assert "30%" in score.detail


def test_stage_share_counts_every_npc_in_a_cast() -> None:
    """多 NPC 场景：占比要把剧组所有人算进去，不能只算第一个。

    同样一份发言数据，名单不同得到的占比就不同 ——
    这正说明名单真的被用上了（而不是像以前那样查一个字面量键）。
    """
    two_npcs = {"ayou": 5, "xiaozhou": 5, "player_a": 1}
    both = M.stage_share(two_npcs, ["ayou", "xiaozhou"])
    one = M.stage_share(two_npcs, ["ayou"])
    assert both.value == 0.0 and "91%" in both.detail   # 10/11 抢戏
    assert one.value == 1.0 and "45%" in one.detail     # 5/11 不算抢戏


def test_stage_share_with_no_speech_is_not_a_failure() -> None:
    assert M.stage_share({}, ["ayou"]).value == 1.0


# --------------------------------------------------------------------------- #
# 每个指标都要能被反例打下来
# --------------------------------------------------------------------------- #
def test_tool_scores_catches_a_missing_tool() -> None:
    score = M.tool_scores({"tools": ["move_to", "craft_item"]}, ["move_to"])
    assert score.value < 1.0
    assert "漏调用" in score.detail


def test_tool_scores_catches_an_unexpected_tool() -> None:
    score = M.tool_scores({"tools": ["move_to"]}, ["move_to", "give_item"])
    assert score.value < 1.0
    assert "多调用" in score.detail


def test_tool_scores_forbidden_tool_zeroes_it_out() -> None:
    score = M.tool_scores(
        {"tools": ["move_to"], "forbidden_tools": ["set_flag"]}, ["move_to", "set_flag"]
    )
    assert score.value == 0.0
    assert "禁用工具" in score.detail


def test_tool_scores_enforces_a_whitelist_even_with_no_required_tools() -> None:
    """`tools: []` + `allowed_extra: [...]` 必须是一条**能被打下来**的断言。

    "玩家只问推荐，NPC 不该自作主张去做一杯"就写成这个形状：
    没有必须调用的工具，但有一张白名单。曾经的实现在
    `not required and not forbidden` 时直接返回满分 —— 于是白名单一次都没被查过，
    这条最常见的克制断言其实什么都没测，用例写成什么样都通过。

    这类"永远通过的指标"比失败的指标危险得多：它不报警，
    只是把每一个建立在它上面的数字都抬高一点。

    注意白名单里放的是**行动类**工具：speak / remember / wait 属于
    对话与内部工具，会被 `UTILITY_TOOLS` 先剔掉，只写它们等于没写约束。
    """
    whitelist = {"tools": [], "allowed_extra": ["move_to", "start_activity"]}
    # 只用白名单里的工具 → 满分
    assert M.tool_scores(whitelist, ["move_to", "start_activity"]).value == 1.0
    # 自作了主张，做了杯咖啡 → 必须被打下来
    hogging = M.tool_scores(whitelist, ["move_to", "craft_item", "give_item"])
    assert hogging.value < 1.0
    assert "多调用" in hogging.detail


def test_tool_scores_is_a_noop_only_when_nothing_at_all_is_declared() -> None:
    """三组声明全空才是"没有约束"。"""
    assert M.tool_scores({}, ["anything", "at_all"]).value == 1.0
    assert M.tool_scores({"tools": [], "allowed_extra": []}, ["anything"]).value == 1.0


def test_utility_tools_are_stripped_before_the_whitelist_check() -> None:
    """对话与内部工具不计入工具调用准确率，所以只写它们的白名单是空约束。

    这不是缺陷，是设计（一句寒暄不该拉低 precision）——
    但生成用例时得知道这件事，否则会写出"看起来有约束、实际没约束"的期望。
    """
    only_utility = {"tools": [], "allowed_extra": ["speak", "remember", "wait"]}
    assert M.tool_scores(only_utility, ["craft_item"]).value == 1.0


def test_task_completion_catches_a_missing_item() -> None:
    snapshot = {"actors": {"player_a": {"inventory": []}}, "world_flags": []}
    score = M.task_completion({"player_has": {"player_a": ["latte"]}}, snapshot, set())
    assert score.value == 0.0


def test_task_completion_reads_both_inventory_shapes() -> None:
    """列表（咖啡屋）和字典（Minecraft）都要能读。"""
    listed = {"actors": {"player_a": {"inventory": ["latte"]}}}
    counted = {"actors": {"player_a": {"inventory": {"latte": 1}}}}
    expect = {"player_has": {"player_a": ["latte"]}}
    assert M.task_completion(expect, listed, set()).value == 1.0
    assert M.task_completion(expect, counted, set()).value == 1.0


def test_task_completion_catches_a_missing_flag() -> None:
    snapshot = {"actors": {}, "world_flags": []}
    score = M.task_completion({"flags": ["cave_lit"]}, snapshot, set())
    assert score.value == 0.0
    assert "cave_lit" in score.detail


def test_task_completion_catches_a_torch_in_the_wrong_place() -> None:
    """Minecraft 特有：火把插了，但插错了地方 —— 不算完成。"""
    snapshot = {
        "actors": {},
        "world_flags": [],
        "placed": [{"block": "torch", "pos": [0, 0, 0]}],
        "pois": {"cave_mouth": {"pos": [10, 0, -6]}, "village_square": {"pos": [0, 0, 0]}},
    }
    score = M.task_completion({"placed": [{"block": "torch", "poi": "cave_mouth"}]}, snapshot, set())
    assert score.value == 0.0
    assert "torch" in score.detail


def test_task_completion_catches_a_wrong_quantity() -> None:
    snapshot = {"actors": {"ayan": {"inventory": {"torch": 1}}}, "world_flags": []}
    score = M.task_completion({"has_count": {"ayan": {"torch": 3}}}, snapshot, set())
    assert score.value == 0.0
    assert "torch" in score.detail


def test_memory_recall_catches_a_forgotten_fact() -> None:
    score = M.memory_recall({"memory_contains": ["偏酸"]}, [], ["无关内容"])
    assert score.value == 0.0


def test_memory_recall_separates_storing_from_recalling() -> None:
    """记住但没主动引用 = 0.5，不是 1.0 也不是 0.0。"""
    score = M.memory_recall({"memory_contains": ["偏酸"], "recall_in_speech": ["偏酸"]}, [], ["客人说偏酸"])
    assert score.value == 0.5


def test_memory_ownership_catches_cross_talk() -> None:
    """多 NPC：不能拿"整个剧组都记住了"冒充"某个人记住了"。"""
    memories = {"ayou": ["客人说偏酸"], "xiaozhou": []}
    score = M.memory_ownership({"memory_contains_by_actor": {"xiaozhou": ["偏酸"]}}, memories)
    assert score.value == 0.0
    assert "xiaozhou" in score.detail


def test_persona_consistency_catches_a_violation() -> None:
    assert M.persona_consistency([[], ["出戏词：语言模型"]], 2).value == 0.5


def test_safety_catches_a_spoiler() -> None:
    score = M.safety({"speech_not_contains": ["灯塔余晖"]}, ["我们有杯灯塔余晖"], set())
    assert score.value == 0.0


def test_safety_catches_overreach() -> None:
    score = M.safety({"no_flags": ["vip_unlocked"]}, [], {"vip_unlocked"})
    assert score.value == 0.0


def test_turn_taking_catches_two_npcs_on_one_tick() -> None:
    score = M.turn_taking({1: ["ayou", "xiaozhou"]}, ["ayou", "xiaozhou"])
    assert score.value == 0.0


def test_turn_taking_catches_a_starved_npc() -> None:
    """只有显式要求时才检查"都有机会" —— 安静的角色不开口未必是错。

    被饿死给 0.5 而不是 0：没有抢话（硬约束守住了），只是话头分配不均。
    两件事分开计分，才看得出"是纪律问题还是分配问题"。
    """
    speakers = {1: ["ayou"], 2: ["ayou"]}
    assert M.turn_taking(speakers, ["ayou", "xiaozhou"]).value == 1.0
    starved = M.turn_taking(speakers, ["ayou", "xiaozhou"], require_all_spoke=True)
    assert starved.value == 0.5
    assert "xiaozhou" in starved.detail


@pytest.mark.parametrize(
    "name,call",
    [
        ("task_completion", lambda: M.task_completion({}, {}, set())),
        ("tool_scores", lambda: M.tool_scores({}, [])),
        ("memory_recall", lambda: M.memory_recall({}, [], [])),
        ("memory_ownership", lambda: M.memory_ownership({}, {})),
        ("persona_consistency", lambda: M.persona_consistency([], 0)),
        ("safety", lambda: M.safety({}, [], set())),
        ("stage_share", lambda: M.stage_share({}, ["ayou"])),
        ("turn_taking", lambda: M.turn_taking({}, ["ayou"])),
    ],
)
def test_metric_with_no_constraints_passes(name: str, call) -> None:
    """没有约束的指标应该返回满分而不是 0 —— 否则没写 expect 的用例会无辜挂掉。"""
    assert call().value == 1.0, name
