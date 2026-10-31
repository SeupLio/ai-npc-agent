"""动机冲突与涌现事件（`modules/motives.py`）。

## "涌现"必须能被看见

配置里写两条动机，不等于世界有了涌现 —— 那只是两行文字。
这里守的是完整闭环：

    两条动机争同一个资源 → 按强度裁定 → 产生一次事件
    → 让步方**把这句话说出口** → 事件标记已发生，不再重复

任何一环断掉，玩家看到的都不是"世界在动"：
少了裁定，是"两个人各说各的"；少了说出口，是"设定里写着有冲突"；
少了"只发生一次"，是"每隔两轮就宣布一次让步"（那是复读）。
"""

from __future__ import annotations

import pytest

from npc_agent.cast import build_cast
from npc_agent.config import RuntimeConfig, load_scenario
from npc_agent.llm import NullLLM
from npc_agent.modules.motives import Motive, MotiveBoard


# --------------------------------------------------------------------------- #
# 冲突检测与裁定
# --------------------------------------------------------------------------- #
def _board() -> MotiveBoard:
    return MotiveBoard(
        [
            Motive("a", "ayou", "早点收摊", "营业时间", intensity=40),
            Motive("b", "xiaozhou", "再唱一首", "营业时间", intensity=70),
        ],
        min_tick=0,
    )


def test_two_motives_on_the_same_resource_conflict():
    assert len(_board().conflicts()) == 1


def test_different_resources_do_not_conflict():
    board = MotiveBoard(
        [
            Motive("a", "ayou", "早点收摊", "营业时间", 40),
            Motive("b", "xiaozhou", "换个曲子", "曲目", 70),
        ],
        min_tick=0,
    )
    assert board.conflicts() == []


def test_the_stronger_motive_wins():
    board = _board()
    a, b = board.conflicts()[0]
    winner, loser = board.resolve(a, b)
    assert winner.owner == "xiaozhou"
    assert loser.owner == "ayou"


def test_equal_intensity_resolves_deterministically():
    """同强度必须可复现 —— 否则同一段对话重放会得到不同的故事。"""
    a = Motive("a", "ayou", "x", "r", intensity=50)
    b = Motive("b", "xiaozhou", "y", "r", intensity=50)
    assert MotiveBoard.resolve(a, b) == MotiveBoard.resolve(b, a)


def test_a_conflict_is_resolved_only_once():
    board = _board()
    assert len(board.poll(1)) == 1
    assert board.poll(2) == []
    assert len(board.events) == 1


def test_nothing_happens_before_the_min_tick():
    """开场就宣布"谁让步"，玩家还没坐稳，两个人已经吵完一架。"""
    board = MotiveBoard([Motive("a", "x", "p", "r", 40), Motive("b", "y", "q", "r", 70)], min_tick=3)
    assert board.poll(0) == []
    assert board.poll(2) == []
    assert len(board.poll(3)) == 1


# --------------------------------------------------------------------------- #
# 端到端：让步要说出口，而且只说一次
# --------------------------------------------------------------------------- #
def _cast():
    return build_cast(load_scenario("duet"), NullLLM(), RuntimeConfig())


def test_the_scenario_ships_motives():
    cast = _cast()
    assert len(cast.motives) >= 2


def test_a_motive_clash_produces_a_world_event():
    cast = _cast()
    for _ in range(6):
        cast.step(None)
    assert cast.events, "动机冲突没有产生事件"
    event = cast.events[0]
    assert event.winner == "xiaozhou"
    assert event.loser == "ayou"


def test_the_loser_says_it_out_loud():
    """让步必须被玩家听见 —— 只存在事件列表里的冲突，玩家看不见。"""
    cast = _cast()
    for _ in range(8):
        cast.step(None)
    lines = [u.text for u in cast.env.utterances if u.speaker_id == "ayou"]
    assert any("依小舟" in line for line in lines), f"让步没有说出口：{lines}"


def test_the_yield_is_said_only_once():
    cast = _cast()
    for _ in range(14):
        cast.step(None)
    lines = [u.text for u in cast.env.utterances if "依小舟" in u.text]
    assert len(lines) == 1, f"让步说了 {len(lines)} 次：{lines}"


def test_the_event_is_marked_as_spoken():
    cast = _cast()
    for _ in range(8):
        cast.step(None)
    assert cast.events and cast.events[0].spoken


def test_no_motives_means_no_events():
    """没配动机的场景行为必须一个字都不变。"""
    cast = build_cast(load_scenario("tutorial"), NullLLM(), RuntimeConfig())
    for _ in range(6):
        cast.step(None)
    assert cast.events == []


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
