"""分组与私聊的护栏。

## 这里在守什么

`duet` / `trio` 只在**一个现场**里调度：谁开口，全场都听得见。
多人分组之后有三条新契约，任何一条破了，页面上都只是"某个人有点怪"：

1. **并行不等于抢话。** 两个组在同一个 tick 各说一句是并行对话；
   撞车判据必须是 `(tick, 组)`。判据退回 `tick` 时，正常的分组对话
   会整段显示成"抢话"，而真正同组抢话反而看不出来。
2. **组是耳朵，不只是队列。** 吧台说的话，露台的人不该记住。
   只改排队的顺序、不筛选观察，两组就共用了一双耳朵。
3. **悄悄话不是小声的公开话。** 回复也必须是私聊；同组的人听不见、
   记不住；印象不进传闻网络。泄漏只要发生在其中一环，前面的收件范围就白设了。
"""

from __future__ import annotations

import pytest

from npc_agent.cast import build_cast
from npc_agent.config import RuntimeConfig, load_scenario
from npc_agent.llm import NullLLM
from npc_agent.studio import run_chat


def _cast(scenario_id: str = "quartet", patch=None):
    scenario = load_scenario(scenario_id)
    if patch is not None:
        patch(scenario)
    cfg = RuntimeConfig()
    cfg.llm_provider = "null"
    return build_cast(scenario, NullLLM(), cfg)


def _player_line(cast, text: str, **kwargs):
    utterance = cast.env.record_player_utterance("player_a", text, **kwargs)
    tick = cast.env.tick
    cast.step(utterance)
    return tick


def _npc_lines_at(cast, tick: int):
    return [
        u
        for u in cast.env.utterances
        if u.tick == tick and u.speaker_id in cast.agents
    ]


# --------------------------------------------------------------------------- #
# 1. 解析：没配分组就退化成一组；配了就必须分全
# --------------------------------------------------------------------------- #
def test_scenarios_without_groups_degrade_to_one_group() -> None:
    cast = _cast("duet")
    assert cast.groups == {"main": cast.order}
    assert {cast.group_of(pid) for pid in cast.order} == {"main"}
    # 单组场景不该被新逻辑改写：所有台词仍是公开频道。
    tick = _player_line(cast, "今天这里挺热闹的。")
    assert all(u.channel == "" and u.audience is None for u in _npc_lines_at(cast, tick))


def test_quartet_parses_two_groups_and_their_names() -> None:
    cast = _cast()
    assert cast.groups == {
        "counter": ["ayou", "tang"],
        "terrace": ["xiaozhou", "luo"],
    }
    assert cast.group_name("counter") == "吧台组"
    assert cast.group_of("tang") == "counter"
    assert cast.group_of("luo") == "terrace"


@pytest.mark.parametrize(
    "groups, match",
    [
        # 漏掉一个人 —— 那个 NPC 会永远不在任何发言队列里
        ({"counter": ["ayou", "tang"], "terrace": ["xiaozhou"]}, "没有分组"),
        # 一个人出现在两个组里 —— 发言权会被算两次
        (
            {
                "counter": ["ayou", "tang"],
                "terrace": ["xiaozhou", "luo", "ayou"],
            },
            "两个分组",
        ),
        # 分组里写了演员表外的人
        (
            {
                "counter": ["ayou", "tang", "ghost"],
                "terrace": ["xiaozhou", "luo"],
            },
            "不在演员表",
        ),
        # 空组
        (
            {
                "counter": ["ayou", "tang"],
                "terrace": ["xiaozhou", "luo"],
                "empty": [],
            },
            "没有成员",
        ),
    ],
)
def test_broken_group_config_fails_loudly(groups, match) -> None:
    """分组配错必须**在构造时报错**，不能静默地少一个人。

    静默的失败方式特别难查：那个人仍然出现在状态、记忆、关系账本里，
    所有模块指标都正常，只有玩家会发现"第四个人从来没说过话"。
    """
    with pytest.raises(ValueError, match=match):
        _cast(patch=lambda scenario: scenario.update({"groups": groups}))


# --------------------------------------------------------------------------- #
# 2. 并行：一组一句，撞车按 (tick, 组)
# --------------------------------------------------------------------------- #
def test_public_utterance_lets_both_groups_answer_in_the_same_tick() -> None:
    cast = _cast()
    tick = _player_line(cast, "今天这里挺热闹的。")
    by_group = cast.speakers_by_group_tick().get(tick, {})
    assert set(by_group) == {"counter", "terrace"}, (
        f"两组没有并行开口：{by_group}。第二组被排到了队尾，"
        "玩家看到的是四条人命共用一张嘴。"
    )
    assert all(len(speakers) == 1 for speakers in by_group.values()), by_group
    assert cast.collisions() == [], "并行对话被误判成同组抢话"


def test_group_speech_is_stamped_and_only_its_group_hears_it() -> None:
    """每句话都要盖上自己的组；另一组不该在记忆里听见它。"""
    cast = _cast()
    tick = _player_line(cast, "今天这里挺热闹的。")
    lines = _npc_lines_at(cast, tick)
    assert lines
    for line in lines:
        gid = cast.group_of(line.speaker_id)
        assert line.channel == gid
        assert tuple(line.audience or ()) == tuple(cast.groups[gid])

    counter_text = "".join(u.text for u in lines if cast.group_of(u.speaker_id) == "counter")
    assert counter_text
    assert any(
        counter_text[:6] in record.content
        for record in cast.agents["tang"].memory.store.records
    ), "同组的人没听见同组的话"
    assert not any(
        counter_text[:6] in record.content
        for record in cast.agents["xiaozhou"].memory.store.records
    ), "露台的人记住了吧台的话 —— 两组共用了一双耳朵"


def test_mention_activates_only_the_named_group() -> None:
    cast = _cast()
    tick = _player_line(cast, "阿柚，能给我来杯拿铁吗？")
    by_group = cast.speakers_by_group_tick().get(tick, {})
    assert by_group.get("counter"), f"被点名的吧台组没有回应：{by_group}"
    assert "terrace" not in by_group, f"露台组也接了话：{by_group}"
    for pid in ("xiaozhou", "luo"):
        assert not any(
            "拿铁" in record.content
            for record in cast.agents[pid].memory.store.records
        ), f"{pid} 记住了别组的事"


def test_each_group_rotates_its_own_floor() -> None:
    """每个组各自轮转 —— 两组不共用游标，组内也不许饿死第二个人。"""
    cast = _cast()
    texts = ["今天这里挺热闹的。", "我口味偏酸。", "外面好像起风了。", "你们这儿的派真香。"]
    seen: dict[str, list[str]] = {gid: [] for gid in cast.groups}
    for text in texts:
        tick = _player_line(cast, text)
        for gid, speakers in cast.speakers_by_group_tick().get(tick, {}).items():
            seen[gid].extend(speakers)
    for gid, members in cast.groups.items():
        assert set(seen[gid]) == set(members), (
            f"{gid} 组只有 {sorted(set(seen[gid]))} 说过话，成员是 {members}"
        )
    assert cast.collisions() == []


# --------------------------------------------------------------------------- #
# 3. 私聊：只有收件人回，回复也是私聊
# --------------------------------------------------------------------------- #
def test_private_lands_on_one_npc_and_the_reply_stays_private() -> None:
    cast = _cast()
    tick = _player_line(
        cast,
        "阿柚，我只跟你说：我其实怕黑。",
        addressed_to="ayou",
        audience=("player_a", "ayou"),
        channel="private",
    )
    by_group = cast.speakers_by_group_tick().get(tick, {})
    assert by_group == {"counter": ["ayou"]}, (
        f"悄悄话不该让同组的人或另一组接话：{by_group}"
    )
    for line in _npc_lines_at(cast, tick):
        assert line.channel == "private", (
            f"{line.speaker_name} 大声回答了悄悄话：channel={line.channel!r}"
        )
        assert "player_a" in (line.audience or ()), line.audience

    for pid in ("tang", "xiaozhou", "luo"):
        assert not any(
            "怕黑" in record.content
            for record in cast.agents[pid].memory.store.records
        ), f"{pid} 记住了不属于他的悄悄话"
    owner_records = [
        r for r in cast.agents["ayou"].memory.store.records if "怕黑" in r.content
    ]
    assert owner_records, "收件人没有记住这句悄悄话"
    assert all(r.privacy == "secret" for r in owner_records)
    assert all("player_a" in r.audience for r in owner_records)
    assert cast.gossip is None or not any(
        "怕黑" in fact.content for fact in cast.gossip.facts()
    ), "悄悄话进了传闻网络"


def test_private_without_a_target_answers_nobody() -> None:
    """没有收件人的悄悄话 → **整轮无人接话**。

    失败方向必须是"少说一句"，而不是"按公开处理" —— 后者会把
    一条本该只有一个人听见的话广播给全店。
    """
    cast = _cast()
    utterance = cast.env.record_player_utterance(
        "player_a",
        "这句话我谁也不想告诉。",
        audience=(),
        channel="private",
    )
    turns = cast.step(utterance)
    assert not [turn for turn in turns if turn.say], "无人收件却有人开口了"
    assert not _npc_lines_at(cast, cast.env.tick - 1)


def test_studio_reports_zero_collisions_for_parallel_groups() -> None:
    """控制台的撞车数必须和导演用**同一个**判据。

    这里曾经各算一份：控制台按 `tick` 全场数，导演按 `(tick, 组)` 数。
    于是分组场景明明正常，网页上却报 10 次撞车 —— 两个数字都"有出处"，
    读者只能两个都不信。
    """
    cfg = RuntimeConfig()
    cfg.llm_provider = "null"
    out = run_chat(
        cfg,
        {
            "scenario": "quartet",
            "events": [
                {"kind": "say", "speaker": "player_a", "text": "今天这里挺热闹的。"}
            ],
        },
    )
    assert out["collisions"] == 0
    said = [
        turn["say"]
        for event in out["events"]
        for turn in event["turns"]
        if turn["say"]
    ]
    assert len(said) == 2, f"两组应该各说一句，实际 {said}"


def test_studio_payload_carries_the_channel_of_every_line() -> None:
    """页面画「私聊 → 谁」「对组 · 某组」读到的字段，必须由后端给出。

    这些标签不能由页面猜：私聊回复也必须是私聊，而 NPC 的组只有导演知道。
    标签读错时页面照常渲染，只是把一句悄悄话标成公开 —— 看起来像数据没准备好。
    """
    cfg = RuntimeConfig()
    cfg.llm_provider = "null"
    out = run_chat(
        cfg,
        {
            "scenario": "quartet",
            "events": [
                {
                    "kind": "say",
                    "speaker": "player_a",
                    "text": "阿柚，我只跟你说：我其实怕黑。",
                    "target": "ayou",
                },
                {
                    "kind": "say",
                    "speaker": "player_a",
                    "text": "吧台组听一下，今天几点打烊？",
                    "group": "counter",
                },
            ],
        },
    )

    private, group = out["events"]
    assert private["scope"] == "private"
    assert private["target_name"] == "阿柚"
    replies = [t for t in private["turns"] if t["say"]]
    assert replies, "收件人没有回话"
    assert all(t["channel"] == "private" for t in replies)
    assert all(t["channel_label"] == "私聊" for t in replies)

    assert group["scope"] == "group"
    assert group["group_name"] == "吧台组"
    spoken = [t for t in group["turns"] if t["say"]]
    assert spoken, "被点名的组没有回话"
    assert all(t["channel"] == "counter" and t["channel_label"] == "吧台组" for t in spoken)


def test_studio_does_not_broadcast_a_private_line_with_a_bad_target() -> None:
    """收件人写错时页面要能看见原因，而世界**一句都不接**。

    如果后端退化成公开，一句本该只有一个人听见的话会被全店接走 ——
    页面还会显示成正常对话，错误完全不可见。
    """
    cfg = RuntimeConfig()
    cfg.llm_provider = "null"
    out = run_chat(
        cfg,
        {
            "scenario": "quartet",
            "events": [
                {"kind": "say", "speaker": "player_a", "text": "喂？", "target": "ghost"}
            ],
        },
    )
    event = out["events"][0]
    assert "ghost" in event["scope_error"]
    assert not [t for t in event["turns"] if t["say"]], "没有收件人却有人接了话"
