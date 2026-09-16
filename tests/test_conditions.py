"""共享条件判定器的单元测试。

这个模块刻意不依赖任何环境 —— 所以这些测试是纯函数测试，
跑得飞快，也不需要构造场景配置。这正是把判定逻辑抽出来的好处之一。
"""

from __future__ import annotations

import pytest

from npc_agent.env.conditions import (
    CONDITION_KINDS,
    ConditionContext,
    condition_met,
    refresh_objectives,
    unmet_reasons,
)


def ctx(**kw) -> ConditionContext:
    """构造一份事实清单。默认：两个玩家、一个同伴，都在场。"""
    base = dict(
        flags=set(),
        inventories={"player_a": [], "player_b": [], "ayou": []},
        speech_counts={},
        player_ids=["player_a", "player_b"],
    )
    base.update(kw)
    return ConditionContext(**base)


# --------------------------------------------------------------------------- #
# 叶子条件
# --------------------------------------------------------------------------- #
def test_empty_condition_is_never_met():
    # 空条件返回 False 而不是 True：一个漏写 success_when 的目标
    # 应该表现为"永远完不成"（评测里能看见），而不是开局即完成。
    assert condition_met({}, ctx()) is False
    assert condition_met(None, ctx()) is False


def test_flag():
    assert condition_met({"flag": "song_started"}, ctx(flags={"song_started"})) is True
    assert condition_met({"flag": "song_started"}, ctx()) is False


def test_all_flags_requires_every_flag():
    c = {"all_flags": ["a", "b"]}
    assert condition_met(c, ctx(flags={"a"})) is False
    assert condition_met(c, ctx(flags={"a", "b"})) is True
    assert condition_met(c, ctx(flags={"a", "b", "c"})) is True


def test_any_flags_needs_at_least_one():
    c = {"any_flags": ["a", "b"]}
    assert condition_met(c, ctx()) is False
    assert condition_met(c, ctx(flags={"b"})) is True


def test_empty_flag_list_is_not_a_free_pass():
    # all([]) 在 Python 里是 True，如果不特判，写错的条件会静默地立刻完成。
    assert condition_met({"all_flags": []}, ctx()) is False
    assert condition_met({"any_flags": []}, ctx()) is False
    assert condition_met({"all_of": []}, ctx()) is False
    assert condition_met({"any_of": []}, ctx()) is False


# --------------------------------------------------------------------------- #
# player_has / player_has_count
# --------------------------------------------------------------------------- #
def test_player_has_checks_world_state_not_bookkeeping():
    c = {"player_has": {"player_a": ["latte"]}}
    assert condition_met(c, ctx(inventories={"player_a": []})) is False
    assert condition_met(c, ctx(inventories={"player_a": ["latte"]})) is True
    # 别人手里有不算
    assert condition_met(c, ctx(inventories={"player_a": [], "player_b": ["latte"]})) is False


def test_player_has_unknown_actor_is_false():
    assert condition_met({"player_has": {"nobody": ["latte"]}}, ctx()) is False


def test_player_has_count_needs_the_quantity():
    c = {"player_has_count": {"player_a": {"oak_log": 3}}}
    assert condition_met(c, ctx(quantities={"player_a": {"oak_log": 2}})) is False
    assert condition_met(c, ctx(quantities={"player_a": {"oak_log": 3}})) is True
    assert condition_met(c, ctx(quantities={"player_a": {"oak_log": 9}})) is True


def test_quantities_take_priority_over_inventory_list():
    # Minecraft 的背包带数量，咖啡屋的背包只是个列表。
    # 同一个 count_item 要走对路：有数量表时以数量表为准，
    # 否则「有 1 个」会被算成「有 3 个」。
    both = ctx(
        inventories={"player_a": ["oak_log", "oak_log", "oak_log"]},
        quantities={"player_a": {"oak_log": 1}},
    )
    assert condition_met({"player_has_count": {"player_a": {"oak_log": 3}}}, both) is False


def test_presence_inventory_counts_duplicates():
    listed = ctx(inventories={"player_a": ["oak_log", "oak_log"]})
    assert condition_met({"player_has_count": {"player_a": {"oak_log": 2}}}, listed) is True


# --------------------------------------------------------------------------- #
# all_players_spoke
# --------------------------------------------------------------------------- #
def test_all_players_spoke_needs_every_player():
    c = {"all_players_spoke": 1}
    assert condition_met(c, ctx(speech_counts={"player_a": 3})) is False
    assert condition_met(c, ctx(speech_counts={"player_a": 3, "player_b": 1})) is True


def test_all_players_spoke_respects_the_count():
    c = {"all_players_spoke": 2}
    assert condition_met(c, ctx(speech_counts={"player_a": 1, "player_b": 5})) is False
    assert condition_met(c, ctx(speech_counts={"player_a": 2, "player_b": 5})) is True


def test_all_players_spoke_ignores_npc_speech():
    # 同伴说话不算"玩家说过话"
    c = {"all_players_spoke": 1}
    assert condition_met(c, ctx(speech_counts={"ayou": 9})) is False


def test_all_players_spoke_without_players_is_false():
    # 没有玩家时不能返回 True —— 那会让空场景的目标开局即完成。
    assert condition_met({"all_players_spoke": 1}, ctx(player_ids=[])) is False


# --------------------------------------------------------------------------- #
# 组合条件
# --------------------------------------------------------------------------- #
def test_all_of_needs_every_sub():
    c = {"all_of": [{"flag": "song_started"}, {"player_has": {"player_a": ["latte"]}}]}
    assert condition_met(c, ctx(flags={"song_started"})) is False
    assert condition_met(
        c, ctx(flags={"song_started"}, inventories={"player_a": ["latte"]})
    ) is True


def test_any_of_needs_one_sub():
    c = {"any_of": [{"flag": "a"}, {"flag": "b"}]}
    assert condition_met(c, ctx()) is False
    assert condition_met(c, ctx(flags={"b"})) is True


def test_nested_conditions():
    c = {
        "all_of": [
            {"any_of": [{"flag": "a"}, {"flag": "b"}]},
            {"flag": "c"},
        ]
    }
    assert condition_met(c, ctx(flags={"a", "c"})) is True
    assert condition_met(c, ctx(flags={"b", "c"})) is True
    assert condition_met(c, ctx(flags={"a"})) is False


def test_unknown_condition_is_false_not_an_exception():
    # 写错的关键字应该表现为"这个目标完不成"，而不是让整场评测崩掉。
    assert condition_met({"flagz": "x"}, ctx(flags={"x"})) is False


def test_every_declared_kind_is_reachable():
    # 防止有人加了分支却忘了登记进 CONDITION_KINDS
    assert "player_has_count" in CONDITION_KINDS
    assert "all_of" in CONDITION_KINDS
    assert len(CONDITION_KINDS) == len(set(CONDITION_KINDS))


# --------------------------------------------------------------------------- #
# refresh_objectives
# --------------------------------------------------------------------------- #
def test_refresh_marks_done_and_is_monotonic():
    specs = [{"id": "sing", "success_when": {"flag": "song_started"}}]
    state: dict[str, str] = {"sing": "pending"}
    refresh_objectives(specs, state, ctx())
    assert state["sing"] == "pending"
    refresh_objectives(specs, state, ctx(flags={"song_started"}))
    assert state["sing"] == "done"
    # 世界后来变了（玩家把咖啡喝了 / 标记被清），完成记录不该被撤销
    refresh_objectives(specs, state, ctx())
    assert state["sing"] == "done"


def test_refresh_leaves_unknown_specs_alone():
    state: dict[str, str] = {}
    refresh_objectives([{"id": "x", "goal": "没有 success_when"}], state, ctx())
    assert "x" not in state


# --------------------------------------------------------------------------- #
# unmet_reasons —— 失败报告用
# --------------------------------------------------------------------------- #
def test_unmet_reasons_lists_pending_leaves():
    c = {"all_of": [{"flag": "a"}, {"player_has": {"player_a": ["latte"]}}]}
    reasons = unmet_reasons(c, ctx())
    assert len(reasons) == 2
    assert any("a" in r for r in reasons)
    assert any("latte" in r for r in reasons)


def test_unmet_reasons_is_empty_when_met():
    assert unmet_reasons({"flag": "a"}, ctx(flags={"a"})) == []


def test_unmet_reasons_skips_satisfied_any_of_branches():
    # any_of 有一个分支达成就整体达成 —— 另一个分支不该再被报成"原因"
    c = {"any_of": [{"flag": "a"}, {"flag": "b"}]}
    assert unmet_reasons(c, ctx(flags={"a"})) == []
    assert len(unmet_reasons(c, ctx())) == 2


def test_unmet_reasons_handles_quantity_conditions():
    c = {"player_has_count": {"player_a": {"oak_log": 3}}}
    reasons = unmet_reasons(c, ctx(quantities={"player_a": {"oak_log": 1}}))
    assert len(reasons) == 1
    assert "oak_log×3" in reasons[0]


@pytest.mark.parametrize("kind", CONDITION_KINDS)
def test_unmet_reasons_never_crashes_on_any_kind(kind):
    # 失败报告本身崩掉是最糟的情况：真正的错误信息会被它的 traceback 盖住。
    well_formed = {
        "all_of": [{"flag": "x"}],
        "any_of": [{"flag": "x"}],
        "flag": "x",
        "all_flags": ["x"],
        "any_flags": ["x"],
        "all_players_spoke": 1,
        "player_has": {"player_a": ["latte"]},
        "player_has_count": {"player_a": {"oak_log": 3}},
    }[kind]
    assert len(unmet_reasons({kind: well_formed}, ctx())) >= 1


@pytest.mark.parametrize("kind", CONDITION_KINDS)
def test_malformed_condition_value_is_false_not_a_crash(kind):
    # 类型写错（YAML 少个缩进、把数字写成列表）也只该是"完不成"。
    # 一次跑了 20 分钟的真实模型对照，不该因为配置里一个标点白跑。
    bad = {kind: [] if kind != "flag" else []}
    assert condition_met(bad, ctx()) is False
    assert isinstance(unmet_reasons(bad, ctx()), list)


def test_malformed_condition_reports_something_readable():
    reasons = unmet_reasons({"all_players_spoke": ["不是数字"]}, ctx())
    assert len(reasons) == 1
    assert reasons[0]
