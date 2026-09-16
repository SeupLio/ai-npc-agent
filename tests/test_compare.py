"""对照跑批与"自由台词率"的测试。

这里有一条**回归测试**（``test_template_regex_with_placeholder_compiles``）：
早期实现对拼好的正则整串 ``strip(标点)``，而 ``.`` 和 ``?`` 既是标点又是正则元字符，
于是 ``.+`` 被削成 ``+``，直接抛 "nothing to repeat"。
去标点必须只作用在原始字面块上。
"""

from __future__ import annotations

import pytest

from npc_agent.config import RuntimeConfig
from npc_agent.eval.compare import (
    RunSpec,
    _norm,
    _template_regex,
    free_speech_rate,
    is_scripted,
    scripted_patterns,
)


# --------------------------------------------------------------------------- #
# 模板正则：回归测试


def test_template_regex_with_placeholder_compiles():
    """槽位开头的模板曾把 ``.+`` 削成 ``+`` 而崩溃。"""
    pattern = _template_regex("{item_name}好了，趁热。")
    assert pattern.match("拿铁好了，趁热")
    assert pattern.match("柠檬水好了，趁热")
    assert not pattern.match("完全不相干的一句话")


def test_template_regex_handles_leading_punctuation_in_literal():
    pattern = _template_regex("……{hint}，是这个没错吧？")
    assert pattern.match("……你第一次来，是这个没错吧")


def test_template_regex_plain_literal_ignores_edge_punctuation():
    pattern = _template_regex("好，稍等，我这就去弄。")
    assert pattern.match("好，稍等，我这就去弄")
    assert not pattern.match("我这就去弄")


def test_template_regex_escapes_regex_metacharacters():
    """模板里的元字符必须被转义，否则会误匹配。

    注意：``_template_regex`` 产出的是**归一化后**文本的正则（去空白、去首尾标点），
    这是内部约定；公开入口 ``is_scripted`` / ``free_speech_rate`` 会先做归一化。
    """
    pattern = _template_regex("价格是 3*2 元")
    assert pattern.match(_norm("价格是 3*2 元"))
    assert not pattern.match(_norm("价格是 332 元"))


# --------------------------------------------------------------------------- #
# 自由台词率


def test_scripted_patterns_covers_persona_templates_and_world_knowledge():
    patterns = scripted_patterns()
    assert len(patterns) >= 20
    # 人设模板（带槽位）
    assert is_scripted("拿铁好了，趁热", patterns)
    # 世界知识库原文
    assert is_scripted("这家店开在星屿的旧灯塔下面，最早是个给守塔人歇脚的地方。", patterns)


def test_free_speech_rate_is_zero_when_all_scripted():
    """台词全部来自人设模板（含槽位渲染）时，自由台词率必须是 0。"""
    patterns = scripted_patterns()
    speeches = [
        "拿铁好了，趁热",              # deliver_order 的槽位渲染结果
        "欢迎，随便坐。今天想喝点什么",  # opening 的槽位渲染结果
        "好，稍等，我这就去弄",        # accept_order 原样
    ]
    assert free_speech_rate(speeches, patterns) == 0.0


def test_free_speech_rate_is_one_when_none_scripted():
    patterns = scripted_patterns()
    speeches = ["外头的风把招牌吹得咣当响", "你今天看起来有点累"]
    assert free_speech_rate(speeches, patterns) == 1.0


def test_free_speech_rate_is_partial_and_correct():
    patterns = scripted_patterns()
    speeches = ["拿铁好了，趁热", "外头的风把招牌吹得咣当响"]
    assert free_speech_rate(speeches, patterns) == pytest.approx(0.5)


def test_free_speech_rate_empty_is_zero():
    assert free_speech_rate([], scripted_patterns()) == 0.0


# --------------------------------------------------------------------------- #
# RunSpec


def test_run_spec_offline_flag():
    assert RunSpec(label="x", provider="null", model="").is_offline
    assert RunSpec(label="x", provider="openai-compat", model="").is_offline
    assert not RunSpec(label="x", provider="openai-compat", model="kimi-k2.7-code").is_offline


def test_run_spec_apply_only_overrides_its_own_fields():
    base = RuntimeConfig(memory_top_k=6, max_steps_per_turn=3, reflect_every=6)
    spec = RunSpec(label="x", provider="openai-compat", model="m", memory_strategy="none")
    cfg = spec.apply(base)

    assert cfg.llm_provider == "openai-compat"
    assert cfg.model == "m"
    assert cfg.memory_strategy == "none"
    # 不相关的字段必须原样保留，否则对照就不公平了
    assert cfg.memory_top_k == 6
    assert cfg.max_steps_per_turn == 3
    assert cfg.reflect_every == 6


def test_run_spec_apply_does_not_mutate_base():
    base = RuntimeConfig(memory_strategy="hybrid")
    RunSpec(label="x", memory_strategy="none").apply(base)
    assert base.memory_strategy == "hybrid"
