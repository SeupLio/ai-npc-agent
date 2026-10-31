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


def test_motives_without_a_resource_never_clash():
    """没写资源的动机不参与裁定 —— 它不知道该跟谁抢。

    按资源分组后，空字符串会成为一个"资源"，两个人各写一条空资源的
    动机就会凭空产生事件。这条钉住那个边界。
    """
    board = MotiveBoard(
        [
            Motive("a", "ayou", "随便想想", "", intensity=90),
            Motive("b", "xiaozhou", "也是随便想想", "", intensity=40),
        ],
        min_tick=0,
    )
    assert board.poll(1) == []


def test_nothing_happens_before_the_min_tick():
    """开场就宣布"谁让步"，玩家还没坐稳，两个人已经吵完一架。"""
    board = MotiveBoard([Motive("a", "x", "p", "r", 40), Motive("b", "y", "q", "r", 70)], min_tick=3)
    assert board.poll(0) == []
    assert board.poll(2) == []
    assert len(board.poll(3)) == 1


def test_three_owners_on_one_resource_get_one_event_not_three():
    """三个人抢同一件事：裁定一次，赢家一个，另外两个人都在 `losers` 里。

    按冲突对枚举会产出 3 条事件、2 句让步 —— 玩家看到的是
    "这家店为同一件事吵了三次"。
    """
    board = MotiveBoard(
        [
            Motive("a", "ayou", "早点收摊", "营业时间", intensity=40),
            Motive("b", "xiaozhou", "再唱一首", "营业时间", intensity=70),
            Motive("c", "luo", "把琴调完", "营业时间", intensity=55),
        ],
        min_tick=0,
    )
    events = board.poll(1, {"ayou": "阿柚", "xiaozhou": "小舟", "luo": "洛"})
    assert len(events) == 1
    event = events[0]
    assert event.winner == "xiaozhou"
    # 代表是让步方里最强的那个（洛 55 > 阿柚 40）。
    assert event.loser == "luo"
    assert event.losers == ["luo", "ayou"]
    assert "洛、阿柚让步" in event.text


def test_two_motives_from_the_same_owner_count_as_one_vote():
    """配置里给同一个人写了两条同一资源的动机，不能变成"他跟自己抢"。"""
    board = MotiveBoard(
        [
            Motive("a1", "ayou", "早点收摊", "营业时间", intensity=40),
            Motive("a2", "ayou", "把灯关了", "营业时间", intensity=90),
            Motive("b", "xiaozhou", "再唱一首", "营业时间", intensity=70),
        ],
        min_tick=0,
    )
    events = board.poll(1, {"ayou": "阿柚", "xiaozhou": "小舟"})
    assert len(events) == 1
    # 同一个人的两条动机只投**最强**的那一票（90），所以阿柚赢、小舟让步。
    # 如果不攒成一人一票，阿柚会同时出现在赢家和让步方里 —— 自己跟自己抢。
    assert events[0].winner == "ayou"
    assert events[0].losers == ["xiaozhou"], "同一个人被记成了两票"
    assert "阿柚要「把灯关了」" in events[0].text
    assert "早点收摊" not in events[0].text


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
