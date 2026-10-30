"""端到端：NPC 和 NPC 到底能不能聊起来。

## 为什么这些断言值得写

六维评测**结构上看不见**这件事：`turn_taking` 只看"一个 tick 里是不是
只有一个人开口"，两个 NPC 全程不说话它也满分。
所以"NPC 之间聊不起来"可以在所有指标满分的情况下存在 ——
实测过：duet 场景 8 轮，对同伴说的台词 **0 句**，被点名后接话 **0 次**，
而当时的 `turn_taking` 是 1.000。

这里的每一条都对应一个**曾经真的坏掉**的环节：

    test_a_peer_question_is_answered_next_tick  导演没把同伴的话当成输入
    test_delegation_goes_through_a_peer         转达被当成"我不知道"
    test_a_group_question_gets_a_real_answer    问"你们俩"答不上来
    test_only_one_npc_speaks_per_tick           两个 NPC 抢话
    test_the_chain_stops_at_the_limit           两个 NPC 聊到玩家插不上嘴
    test_a_player_turn_clears_the_chain         玩家插话后 NPC 还在自说自话
"""

from __future__ import annotations

import pytest

from npc_agent.cast import build_cast
from npc_agent.config import RuntimeConfig, load_scenario
from npc_agent.llm import NullLLM


def _cast(scenario: str = "duet", **overrides) -> object:
    config = RuntimeConfig(**overrides)
    return build_cast(load_scenario(scenario), NullLLM(), config)


def _say(cast, player: str, text: str) -> list:
    utterance = cast.env.record_player_utterance(player, text)
    return cast.step(utterance)


def _idle(cast, ticks: int = 1) -> list:
    turns = []
    for _ in range(ticks):
        turns = cast.step(None)
    return turns


def _says(turns: list) -> dict[str, str]:
    return {t.actor_id: t.say for t in turns if t.say}


# --------------------------------------------------------------------------- #
# 基本能力：同伴的话被当成输入，下一轮有人接
# --------------------------------------------------------------------------- #
def test_a_peer_question_is_answered_next_tick():
    """阿柚对同伴发问 → 下一轮小舟必须开口。

    修复前：同伴的发言只被"记进记忆"，从来不是"下一轮的输入"，
    所以小舟永远不知道有人在问她（实测接话率 0%）。
    """
    cast = _cast()
    # 让阿柚开口对同伴说话：玩家让她转达。
    turns = _say(cast, "player_a", "阿柚，你让小舟弹一首呗？")
    speaker = [t for t in turns if t.actor_id == "ayou"][0]
    assert speaker.addressed_to == "xiaozhou"

    # 下一轮：没有人说话，但对话板上欠着小舟一句话。
    nxt = _idle(cast)
    assert "xiaozhou" in _says(nxt), "被点名的 NPC 没有接话"
    assert nxt[0].actor_id == "xiaozhou", "被点名的人应该排第一"


def test_the_peer_actually_sees_the_words():
    """接话不是"轮到我了就随便说一句"：小舟的现场状态里要有那句话。"""
    cast = _cast()
    _say(cast, "player_a", "阿柚，你让小舟弹一首呗？")
    _idle(cast)
    state = cast.agents["xiaozhou"].state
    assert state.last_utterance is not None
    assert state.last_utterance.speaker_id == "ayou"


# --------------------------------------------------------------------------- #
# 转达
# --------------------------------------------------------------------------- #
def test_delegation_goes_through_a_peer():
    """「你让小舟弹一首」→ 阿柚**对小舟**说，而不是对玩家复述。

    修复前实测：阿柚回「这个我不太清楚，别听我瞎说」——
    它不知道的是"该转达"，不是"这件事"。
    """
    cast = _cast()
    turns = _say(cast, "player_a", "阿柚，你让小舟弹一首呗？")
    ayou = [t for t in turns if t.actor_id == "ayou"][0]
    assert ayou.say, "阿柚没有开口"
    assert ayou.addressed_to == "xiaozhou"
    assert "小舟" in ayou.say
    # 转达的话必须落进世界记录（导演靠它开义务）
    spoken = [u for u in cast.env.utterances if u.speaker_id == "ayou"]
    assert spoken and spoken[-1].addressed_to == "xiaozhou"
    assert spoken[-1].expects_reply is True


# --------------------------------------------------------------------------- #
# 群体提问
# --------------------------------------------------------------------------- #
def test_a_group_question_gets_a_real_answer():
    """「你们俩平时都聊些什么？」要有一个**有内容**的回答。

    修复前：这类问题没有答案路径，落到 `unknown`，
    NPC 回「这个我还真说不好」—— 被问到自己天天在一起的伙伴却答不上来。
    """
    cast = _cast()
    turns = _say(cast, "player_a", "你们俩平时都聊些什么？")
    says = _says(turns)
    assert says, "没人回答"
    answer = next(iter(says.values()))
    assert "说不好" not in answer and "不太清楚" not in answer
    assert "弹" in answer or "煮" in answer or "咖啡" in answer


# --------------------------------------------------------------------------- #
# 发言权
# --------------------------------------------------------------------------- #
def test_only_one_npc_speaks_per_tick():
    """一个 tick 只能有一个人开口 —— 多 Agent 最典型的翻车。"""
    cast = _cast()
    script = [
        "今天这里挺热闹的。",
        "阿柚，你让小舟弹一首呗？",
        "小舟，你那边在弹什么？",
    ]
    for text in script:
        _say(cast, "player_a", text)
        _idle(cast)
    by_tick = cast.env.speakers_by_tick()
    npc_ids = set(cast.npc_ids)
    for tick, speakers in by_tick.items():
        npcs = [s for s in speakers if s in npc_ids]
        assert len(npcs) <= 1, f"tick {tick} 有 {len(npcs)} 个 NPC 同时开口：{npcs}"


def test_the_chain_stops_at_the_limit():
    """NPC 连聊有上限。上限为 0 时，一句对同伴的话都不许接。

    `truncated` 是"机制在工作"的计数，不是失败率 ——
    所以它被单独统计，也不进任何通过率的分母。
    """
    cast = _cast(max_peer_exchanges=0)
    _say(cast, "player_a", "阿柚，你让小舟弹一首呗？")
    turns = _idle(cast)
    assert cast.board.truncated >= 1, "链长上限一次都没触发 —— 闸门是摆设"
    assert "xiaozhou" not in _says(turns), "上限为 0 时仍然开了同伴轮"


def test_the_chain_never_exceeds_its_limit():
    """跑一段真实对话，任何时刻的 chain 都不能超过上限。"""
    cast = _cast(max_peer_exchanges=2)
    _say(cast, "player_a", "阿柚，你让小舟弹一首呗？")
    for _ in range(8):
        _idle(cast)
        assert cast.board.chain <= 2


def test_a_player_turn_clears_the_chain():
    """玩家插话 → 那段 NPC 对聊结束，欠着的话作废。

    不作废的话，会出现"玩家问了一句，NPC 却先回答三轮前同伴的话"。
    """
    cast = _cast()
    _say(cast, "player_a", "阿柚，你让小舟弹一首呗？")
    assert cast.board.pending is not None
    _say(cast, "player_a", "小舟，你唱得真好听。")
    assert cast.board.pending is None
    assert cast.board.chain == 0


# --------------------------------------------------------------------------- #
# 复读 / 跨 NPC 去重
# --------------------------------------------------------------------------- #
def test_two_npcs_do_not_tell_the_same_fact():
    """同一件事不能被两个 NPC 各讲一遍。

    逐字判据（`agent.said`）是**私有**的，管不到"我刚讲完店史，他也讲一遍"。
    跨 NPC 的去重靠对话板上的话题账本 —— 这是它存在的唯一理由。
    """
    cast = _cast()
    for _ in range(14):
        _idle(cast)
    facts = [
        u.text
        for u in cast.env.utterances
        if u.speaker_id in set(cast.npc_ids) and "灯塔" in u.text
    ]
    assert len(facts) <= 1, f"同一件事被说了 {len(facts)} 次：{facts}"


def test_npc_lines_do_not_repeat_within_a_short_conversation():
    """12 轮以内不该出现逐字复读（和 `docs/repetition.html` 同一套口径）。"""
    from npc_agent.modules.repetition import find_repeats

    cast = _cast()
    _say(cast, "player_a", "今天这里挺热闹的。")
    _say(cast, "player_b", "露台那边好像有风。")
    _idle(cast, 10)
    lines = [(u.speaker_id, u.text) for u in cast.env.utterances if u.speaker_id in set(cast.npc_ids)]
    report = find_repeats(lines)
    assert report.rate <= 0.2, report.render()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
