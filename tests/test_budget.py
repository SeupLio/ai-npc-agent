"""成本与循环管控（`modules/budget.py`）。

三件事各测一组，而且每组都要**正反两面**：
既测"该拦的拦住了"，也测"不该拦的没被拦"。

第二条比第一条重要：第一版循环检测写的是"同一个动作在窗口里出现 3 次"，
它把一条正常的做单计划判成了死循环 —— `move_to(吧台)` 在一次做单里
本来就出现 3 次（去后厨、回吧台、再回吧台递东西），中间夹着别的动作。
计划被拦腰砍断，离线 235 条掉到 233。**一个会误伤的刹车比没有刹车更糟。**
"""

from __future__ import annotations

import pytest

from npc_agent.config import RuntimeConfig
from npc_agent.modules.budget import LoopGuard, TurnBudget, trim_records


# --------------------------------------------------------------------------- #
# 预算
# --------------------------------------------------------------------------- #
def test_per_turn_budget_blocks_the_third_call():
    budget = TurnBudget(max_calls_per_turn=2)
    budget.start_turn()
    assert budget.allow_call()
    budget.charge(100)
    assert budget.allow_call()
    budget.charge(100)
    assert not budget.allow_call()
    assert budget.denied == 1


def test_a_new_turn_resets_the_per_turn_counter():
    """单轮额度每轮重置 —— 不清零的话第二句台词永远用不上模型。"""
    budget = TurnBudget(max_calls_per_turn=1)
    budget.start_turn()
    budget.charge(10)
    assert not budget.allow_call()
    budget.start_turn()
    assert budget.allow_call()


def test_session_budget_survives_turn_resets():
    """会话总额不清零 —— 否则"一轮一轮慢慢烧"就拦不住。"""
    budget = TurnBudget(max_calls_per_turn=5, max_session_tokens=100)
    budget.start_turn()
    budget.charge(120)
    budget.start_turn()
    assert not budget.allow_call()
    assert budget.tokens_total == 120


# --------------------------------------------------------------------------- #
# 循环检测
# --------------------------------------------------------------------------- #
def test_three_identical_actions_trip():
    guard = LoopGuard(repeats=3)
    assert not guard.observe(("move_to", "kitchen"))
    assert not guard.observe(("move_to", "kitchen"))
    assert guard.observe(("move_to", "kitchen"))
    assert guard.trips == 1


def test_a_two_step_cycle_trips():
    """A B A B A B 也是原地打转 —— 两个动作互相踢皮球。"""
    guard = LoopGuard(repeats=3)
    sequence = [("move_to", "kitchen"), ("take_item", "beans")] * 3
    tripped = [guard.observe(sig) for sig in sequence]
    assert tripped[-1] is True


def test_a_normal_order_plan_does_not_trip():
    """回归：正常做单会**多次经过同一个地点**，那不是循环。

    实测（`multi_named_npc_priority`）：第一版按"同一动作出现 3 次"判，
    这条计划被拦腰砍断，`give_item` 永远执行不到 —— 柠檬水做了却递不出去。
    """
    guard = LoopGuard(repeats=3)
    plan = [
        ("speak", "accept_order"),
        ("move_to", "counter"),
        ("move_to", "kitchen"),
        ("take_item", "lemon"),
        ("move_to", "counter"),
        ("craft_item", "lemonade"),
        ("move_to", "counter"),
        ("give_item", "lemonade"),
        ("speak", "deliver_order"),
    ]
    for sig in plan:
        assert not guard.observe(sig), f"{sig} 被误判成循环"


def test_reset_clears_the_history():
    guard = LoopGuard(repeats=2)
    guard.observe(("a",))
    guard.observe(("a",))
    assert guard.tripped
    guard.reset()
    assert not guard.tripped
    assert not guard.observe(("a",))


# --------------------------------------------------------------------------- #
# 上下文裁剪
# --------------------------------------------------------------------------- #
class _Record:
    def __init__(self, content: str) -> None:
        self.content = content


def test_trim_keeps_the_most_relevant_records():
    records = [_Record("x" * 50) for _ in range(5)]
    kept, dropped = trim_records(records, max_chars=120)
    assert len(kept) == 2
    assert dropped == 3


def test_trim_keeps_at_least_one_record():
    """一条就超预算时也要留一条 —— 空记忆会让 NPC 变成没有过去的人。"""
    records = [_Record("x" * 500)]
    kept, dropped = trim_records(records, max_chars=100)
    assert len(kept) == 1
    assert dropped == 0


def test_pinned_records_survive_trimming():
    """刚回引过的那条必须在场，否则"你上次说过…"会引一条它看不见的记忆。"""
    records = [_Record("y" * 60) for _ in range(4)]
    pinned = records[-1]
    kept, _ = trim_records(records, max_chars=70, pinned=[pinned])
    assert kept[0] is pinned


# --------------------------------------------------------------------------- #
# 配置真的接上了
# --------------------------------------------------------------------------- #
def test_the_config_values_reach_the_objects():
    from npc_agent.cast import build_cast
    from npc_agent.config import load_scenario
    from npc_agent.llm import NullLLM

    config = RuntimeConfig(
        max_llm_calls_per_turn=7, max_session_tokens=1234, loop_guard_repeats=5
    )
    cast = build_cast(load_scenario("duet"), NullLLM(), config)
    agent = cast.lead
    assert agent.budget.max_calls_per_turn == 7
    assert agent.budget.max_session_tokens == 1234
    assert agent.loop_guard.repeats == 5


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
