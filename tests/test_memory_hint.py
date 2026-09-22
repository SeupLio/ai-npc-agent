"""记忆回引机制的测试（两件事：抽 hint + 判断"值不值得回引"）。

**回归测试一**：``extract_memory_hint`` 早期是
``content.split("：", 1)[-1][:18].rstrip("。！？，、 ")``。
按字数硬截对"巩固后的摘要"无效 —— 摘要用「；」把多条 episodic 拼起来，
截出来是 ``我特别喜欢偏酸的咖啡，越酸越好。；小``，
于是 NPC 说出「…越好。；小，是这个没错吧？」这种明显坏掉的话。
正确做法是**按句子边界切，不按字数切**。

**回归测试二**：剥掉「某某说：」之后剩下的还是**第一人称**，
塞进人设模板 ``对了，你之前提过{memory_hint}`` 就变成 NPC 把玩家的话
当成了自己的话（「你之前提过**我**特别喜欢偏酸的咖啡」）。
三个人设的 recall 模板都是这个形状，所以是全量的。
六维评测当时给这些用例打了 **1.000** —— `recall_in_speech` 只查子串在不在。

**回归测试三**：玩家**问句本身**含线索词时（「你还记得我**习惯**坐哪儿吗」），
这句话会被当成"值得回引的记忆"，于是 NPC 把问题引回来当成过去的事：
「对了，你之前提过阿柚，我还记得你习惯坐哪儿吗，是这个没错吧？」
—— 顺手把自己的名字说成了玩家的。`tick` 判定挡不住它。
"""

from __future__ import annotations

import pytest

from npc_agent.agent import extract_memory_hint


def test_plain_episodic_strips_speaker_prefix():
    assert extract_memory_hint("阿澈说：我特别喜欢偏酸的咖啡，越酸越好。") == "我特别喜欢偏酸的咖啡，越酸越好"


def test_consolidated_summary_cuts_at_clause_boundary():
    """回归测试：巩固摘要用「；」拼接，硬截会切出半个词并漏出分隔符。"""
    content = "（早前对话摘要）阿澈说：我特别喜欢偏酸的咖啡，越酸越好。；小满说：我也常来这种小店。"
    hint = extract_memory_hint(content)
    assert hint == "我特别喜欢偏酸的咖啡，越酸越好"
    assert "；" not in hint          # 分隔符不能漏出来
    assert "小满" not in hint         # 不能串到下一句
    assert "小" != hint[-1] or len(hint) == 1  # 不能以孤立的单字收尾


def test_summary_prefix_is_stripped():
    hint = extract_memory_hint("（早前对话摘要）小满说：我习惯坐靠窗的位置。")
    assert hint == "我习惯坐靠窗的位置"


def test_long_text_prefers_comma_boundary():
    """超长时优先在逗号处收尾，而不是留下半个词。"""
    content = "阿澈说：我平时喜欢一个人去山里爬山，尤其是清晨出发，那样人少一些。"
    hint = extract_memory_hint(content, limit=18)
    assert len(hint) <= 18
    assert not hint.endswith("，")
    assert "尤其是清晨出发" not in hint  # 被逗号边界切掉了


def test_empty_and_whitespace_are_safe():
    assert extract_memory_hint("") == ""
    assert extract_memory_hint("   ") == ""
    assert extract_memory_hint("阿澈说：") == ""


def test_no_prefix_no_terminator():
    assert extract_memory_hint("小满说的我记下了。") == "小满说的我记下了"


def test_never_leaks_edge_punctuation():
    for content in (
        "阿澈说：我喜欢酸一点的。",
        "（早前对话摘要）小满说：我常来。；阿岚说：我也是。",
        "阿澈说：第一次来吧？",
    ):
        hint = extract_memory_hint(content)
        assert hint == hint.strip("。！？，、；:： ")


def test_question_mark_is_a_boundary():
    """「？」也是句子边界，不能在它后面继续接内容。"""
    hint = extract_memory_hint("阿澈说：第一次来吧？我请你一杯。")
    assert hint == "第一次来吧"


# --------------------------------------------------------------------------- #
# 人称：别把玩家的话塞进 NPC 嘴里
# --------------------------------------------------------------------------- #
# 这一组是**另一个**真实缺陷的回归测试，和"按字数硬截"是两回事。
#
# 记忆存的是**玩家原话**（`observe` 写成 ``阿澈说：我喜欢偏酸的咖啡``）。
# 剥掉「阿澈说：」之后剩下第一人称，塞进人设模板
# ``对了，你之前提过{memory_hint}，是这个没错吧？``，实测跑出来的是：
#
#     「对了，你之前提过**我**特别喜欢偏酸的咖啡，是这个没错吧？」
#
# —— NPC 把玩家的话当成了自己的话。三个人设的 recall 模板全是这个形状，
# 所以这是全量的。而六维评测给这些用例打了 **1.000**：
# `recall_in_speech` 只查子串在不在，看不见人称对不对。


def test_swap_person_is_simultaneous_not_one_way():
    """必须是**同时**互换，不是「我→你」单向替换。

    单向替换会在「我觉得你不错」上把自己绕进去 —— 变成"你觉得你不错"。
    这一条就是用来钉住这个区别的。
    """
    from npc_agent.agent import swap_person

    assert swap_person("我喜欢偏酸的") == "你喜欢偏酸的"
    assert swap_person("我觉得你不错") == "你觉得我不错"   # 单向替换会得到「你觉得你不错」
    assert swap_person("我们常来") == "你们常来"           # 复合词自动跟着对
    assert swap_person("我的口味偏酸") == "你的口味偏酸"
    assert swap_person("") == ""


def test_other_peoples_words_are_switched_to_second_person():
    """别人说的话 → 换成第二人称，才能塞进「你之前提过…」。"""
    hint = extract_memory_hint(
        "阿澈说：我特别喜欢偏酸的咖啡，越酸越好。", own_name="阿柚"
    )
    assert hint == "你特别喜欢偏酸的咖啡，越酸越好"
    assert "我" not in hint


def test_own_words_are_not_switched():
    """**自己**说的话不能换 —— 换了就变成"你觉得我不错"那种错位。"""
    hint = extract_memory_hint("阿柚说：我觉得你不错。", own_name="阿柚")
    assert hint == "我觉得你不错"


def test_npc_authored_memory_is_not_switched():
    """`remember` 工具写的记忆没有「某某说：」前缀，本来就是第一人称，不许换。"""
    hint = extract_memory_hint("小满说的我记下了。", own_name="阿柚")
    assert hint == "小满说的我记下了"


def test_without_own_name_behaviour_is_unchanged():
    """不传 `own_name` 时保持老行为 —— 老调用方和纯函数语义都不受影响。"""
    assert (
        extract_memory_hint("阿澈说：我特别喜欢偏酸的咖啡，越酸越好。")
        == "我特别喜欢偏酸的咖啡，越酸越好"
    )


def test_rendered_recall_line_does_not_claim_the_players_identity():
    """**端到端**：真渲染一遍三个人设的 recall 模板，断言 NPC 没把玩家的话当自己的。

    这是这条回归的**主判据** —— 上面几条测的是实现细节，这一条测的是
    "玩家会看到什么"。三个人设都要过，因为模板是各自配的。
    """
    from npc_agent.config import load_persona
    from npc_agent.modules.persona import Persona

    stored = "阿澈说：我特别喜欢偏酸的咖啡，越酸越好。"
    for persona_id in ("ayou", "ayan", "xiaozhou"):
        persona = Persona.from_dict(load_persona(persona_id))
        hint = extract_memory_hint(stored, own_name=persona.name)
        line = persona.render_template("recall", target="阿澈", memory_hint=hint)

        assert "提过我" not in line, f"{persona_id} 把玩家的话当成了自己的话：{line}"
        assert "说过我" not in line, f"{persona_id} 把玩家的话当成了自己的话：{line}"
        assert "你特别喜欢" in line, f"{persona_id} 没换人称：{line}"


# --------------------------------------------------------------------------- #
# 值不值得回引：不能把**玩家当前这句话**引回来
# --------------------------------------------------------------------------- #
def _record(
    content: str,
    *,
    tick: int = 1,
    importance: float = 0.8,
    entities: tuple[str, ...] = ("player_a",),
):
    """造一条最简的记忆记录（只用到 _recallable 会读的字段）。

    ⚠️ `entities` 不能省：它是 `_recallable` 的**来源判据**
    （这条记录关于谁）。stub 少一个字段，就等于把新判据整条绕过去 ——
    「断言读写入侧、缺陷在读取侧」这类盲区就是这么来的。
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        content=content, tick=tick, importance=importance, entities=list(entities)
    )


def _agent():
    """一个最小可用的离线 agent —— `_recallable` 现在要读"谁是玩家"和说话人。"""
    from npc_agent.cast import build_cast
    from npc_agent.config import RuntimeConfig, load_scenario
    from npc_agent.llm import build_llm

    return build_cast(load_scenario("icebreaker"), build_llm("null"), RuntimeConfig()).lead


def _utterance(text: str):
    from types import SimpleNamespace

    # `speaker_id` 不能省：回引要指向**当前说话人**（见 `_recallable` 第 4 条）。
    return SimpleNamespace(
        text=text,
        tick=99,
        speaker_id="player_a",
        is_question=text.endswith(("？", "?")),
    )


def test_current_question_is_not_recallable():
    """回归测试：玩家**问句本身**含线索词时，不能把问题当"过去的事"引回来。

    实测（`memory_seat_recall`）：

        玩家：阿柚，你还记得我习惯坐哪儿吗？
        NPC ：对了，你之前提过**阿柚，我还记得你习惯坐哪儿吗**，是这个没错吧？

    问句里的「习惯」是线索词，于是它被当成值得回引的记忆。
    """
    agent = _agent()

    question = "阿柚，你还记得我习惯坐哪儿吗？"
    # 记忆库里只有"这句问话本身"（真实路径里 observe 会把它写进去）
    only_question = [_record(f"阿澈说：{question}")]

    assert not agent._has_recallable(
        only_question, now=100, utterance=_utterance(question)
    ), "把玩家当前这句问话引回来了 —— 会说出「你之前提过阿柚，我还记得你…」"
    assert agent._recallable(only_question, now=100, utterance=_utterance(question)) == []


def test_earlier_statement_is_still_recallable():
    """反向：**更早**那条真正的偏好仍然要能被回引 —— 别把功能一起关掉。"""
    agent = _agent()

    question = "阿柚，你还记得我习惯坐哪儿吗？"
    memories = [
        _record(f"阿澈说：{question}"),          # 本轮这句，要排除
        _record("阿澈说：我习惯坐靠窗的位子。"),   # 更早那条，要留下
    ]
    picked = agent._recallable(memories, now=100, utterance=_utterance(question))
    assert [r.content for r in picked] == ["阿澈说：我习惯坐靠窗的位子。"]


def test_recallable_still_requires_every_condition():
    """`_recallable` 的每个条件一个都不能少：
    过去 / 含线索词 / 够重要 / **来源是玩家**。"""
    agent = _agent()

    q = _utterance("在吗？")
    # 含线索词、够重要、但**不是过去**（tick 不早于 now）
    assert agent._recallable([_record("我习惯坐靠窗", tick=100)], 100, q) == []
    # 是过去、含线索词、但**不重要**
    assert agent._recallable([_record("我习惯坐靠窗", importance=0.5)], 100, q) == []
    # 是过去、够重要、但**没有线索词**
    assert agent._recallable([_record("今天天气不错")], 100, q) == []
    # 是过去、够重要、含线索词，但**不是玩家说的**（entities 是同伴 NPC）
    npc_said = [_record("第一次来吧？我请你一杯。", entities=("ayou",))]
    assert agent._recallable(npc_said, 100, q) == []
    # 是过去、够重要、含线索词，但**是别人说的**（不是当前说话人）
    other = [_record("我习惯坐靠窗", entities=("player_b",))]
    assert agent._recallable(other, 100, q) == []
    # 每一条都满足 → 留下
    assert len(agent._recallable([_record("我习惯坐靠窗")], 100, q)) == 1


def test_recallable_without_utterance_is_backward_compatible():
    """不传 utterance 时保持老语义（主动开口那条路径就是不带 utterance 调的）。"""
    agent = _agent()

    memories = [_record("我习惯坐靠窗的位子。")]
    assert agent._has_recallable(memories, now=100)
    assert agent._has_recallable(memories, now=100, utterance=None)

