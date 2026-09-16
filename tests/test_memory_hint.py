"""记忆回引短句提取的测试。

**回归测试**：``extract_memory_hint`` 早期是
``content.split("：", 1)[-1][:18].rstrip("。！？，、 ")``。
按字数硬截对"巩固后的摘要"无效 —— 摘要用「；」把多条 episodic 拼起来，
截出来是 ``我特别喜欢偏酸的咖啡，越酸越好。；小``，
于是 NPC 说出「…越好。；小，是这个没错吧？」这种明显坏掉的话。
正确做法是**按句子边界切，不按字数切**。
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
