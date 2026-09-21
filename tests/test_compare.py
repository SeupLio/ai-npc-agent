"""对照跑批与"自由台词率"的测试。

这里有一条**回归测试**（``test_template_regex_with_placeholder_compiles``）：
早期实现对拼好的正则整串 ``strip(标点)``，而 ``.`` 和 ``?`` 既是标点又是正则元字符，
于是 ``.+`` 被削成 ``+``，直接抛 "nothing to repeat"。
去标点必须只作用在原始字面块上。
"""

from __future__ import annotations

import json

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


def test_a_bare_slot_template_is_not_a_wildcard():
    """整条模板只有槽位时，**不能**编译成 ``^.+$``。

    ``answer_question: "{answer_hint}"`` 的字面量是空的 —— 说什么完全由
    运行时填进去的东西决定，这条模板对"这句话长什么样"零信息量。
    早期它被编译成 ``^.+$``，一个匹配任何非空台词的通配符，
    于是自由台词率**恒为 0**：报告会说每句台词都是脚本台词。

    这个 bug 是加多变体模板时引入的：``answer_question`` 从
    ``"嗯——{answer_hint}"``（字面量"嗯——"）变成 ``["{answer_hint}"]``（无字面量）。
    所以这条测试同时守着"以后别再写出裸槽位模板"。
    """
    assert _template_regex("{answer_hint}") is None
    assert _template_regex("{a}{b}") is None
    # 有一个字的字面量就够了 —— 判据是"有没有字面量"，不是"够不够长"
    assert _template_regex("嗯——{answer_hint}") is not None


def test_no_scripted_pattern_is_a_catch_all():
    """参照系里**一个通配符都不许有**。

    只要有一个 ``^.+$``，自由台词率就变成常数 0，而且不会报错 ——
    这正是 ``test_free_speech_rate_is_one_when_none_scripted`` 会红的原因。
    与其等那句断言，不如直接钉住"参照系里没有通配符"这个更强的性质。
    """
    catch_alls = [p.pattern for p in scripted_patterns() if p.pattern == "^.+$"]
    assert not catch_alls, f"参照系里有通配符，自由台词率会恒为 0：{catch_alls}"


def test_self_facts_replies_count_as_scripted():
    """`self_facts` 的 reply 是写在人设里的固定句子，也要算脚本台词。

    它是离线路径回答"你叫什么名字"时照念的那句（见 `Persona.answer_about_self`）。
    不算进来的话，离线模式的自由台词率会被高估 ——
    而"离线 vs 模型"这张对照表最容易误读的就是这个数。
    """
    patterns = scripted_patterns()
    assert is_scripted("我是阿柚，这家店的店主。", patterns)


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


# --------------------------------------------------------------------------- #
# 跑批与检查点


def test_checkpoint_is_written_during_run(tmp_path):
    """长跑批必须边跑边落盘，否则中途崩溃会丢掉全部结果。"""
    from npc_agent.eval.compare import Comparison

    checkpoint = tmp_path / "ckpt.json"
    comparison = Comparison(base_config=RuntimeConfig(), limit=2).run(
        [RunSpec(label="离线", provider="null")], checkpoint=checkpoint
    )

    assert checkpoint.exists()
    data = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert len(data["runs"]) == 1
    assert data["runs"][0]["passed"] == 2      # 已完成 2 条
    assert data["runs"][0]["total"] == 2
    assert data["runs"][0]["speeches"]

    # 派生指标在检查点里也要是算好的，而不是默认值。
    # 直接和内存里的结果对一遍 —— 原来的写法是断言 free_speech_rate == 0.0，
    # 那只是"头两条用例恰好没有自由台词"的巧合：换了用例集就失效，
    # 而且它和注释想说的"要算好"是两回事（断言值等于 0，恰恰是在断言没算）。
    run = comparison.outcomes[0]
    assert data["runs"][0]["free_speech_rate"] == round(run.free_speech_rate, 3)
    assert data["runs"][0]["speech_count"] == run.speech_count
    assert data["runs"][0]["avg_speech_chars"] == round(run.avg_speech_chars, 1)
    assert data["runs"][0]["metric_means"] == run.means


def test_on_case_callback_fires_per_case():
    from npc_agent.eval.compare import Comparison

    seen: list[tuple[str, int, int]] = []
    Comparison(base_config=RuntimeConfig(), limit=3).run(
        [RunSpec(label="离线", provider="null")],
        on_case=lambda spec, case, i, total, outcome: seen.append(
            (case.get("id"), i, total)
        ),
    )
    assert len(seen) == 3
    assert [i for _, i, _ in seen] == [1, 2, 3]
    assert all(total == 3 for _, _, total in seen)


def test_comparison_records_outcome_before_finishing():
    """outcome 必须先挂进列表，这样检查点里才有"部分完成"的报告。"""
    from npc_agent.eval.compare import Comparison

    comparison = Comparison(base_config=RuntimeConfig(), limit=1).run(
        [RunSpec(label="离线", provider="null")]
    )
    assert len(comparison.outcomes) == 1
    assert comparison.outcomes[0].report.total == 1


# --------------------------------------------------------------------------- #
# 配对：覆盖数不一致时必须截到共同区间


def _uneven_comparison():
    """构造一个"第二行只跑了一半"的对照 —— 真实模型跑批被打断就是这样。"""
    from npc_agent.eval.compare import Comparison

    comparison = Comparison(base_config=RuntimeConfig(), limit=4).run(
        [
            RunSpec(label="A", provider="null"),
            RunSpec(label="B", provider="null"),
        ]
    )
    comparison.outcomes[1].report.results = comparison.outcomes[1].report.results[:2]
    return comparison


def test_paired_count_and_flag():
    comparison = _uneven_comparison()
    assert comparison.paired_count() == 2
    assert comparison.is_paired() is False


def test_rows_trim_to_common_prefix():
    """不截齐就是拿苹果比橘子：A 跑 4 条、B 跑 2 条，均值不可比。"""
    rows = _uneven_comparison().rows()
    assert rows[0]["pass"] == "2/2"   # 被截到 2 条
    assert rows[1]["pass"] == "2/2"


def test_to_dict_marks_unpaired():
    data = _uneven_comparison().to_dict()
    assert data["paired"] is False
    assert data["paired_cases"] == 2
    assert all(run["total"] == 2 for run in data["runs"])


def test_even_comparison_is_paired():
    from npc_agent.eval.compare import Comparison

    comparison = Comparison(base_config=RuntimeConfig(), limit=3).run(
        [RunSpec(label="A", provider="null")]
    )
    assert comparison.is_paired() is True
    assert comparison.paired_count() == 3

# --------------------------------------------------------------------------- #
# 检查点的落盘节奏：不许退化成"每条用例一次"


def test_the_checkpoint_is_not_written_once_per_case(tmp_path, monkeypatch):
    """检查点按**时间**节流，不是每条用例落一次。

    逐条落盘是 O(N²)：`_refresh()` 每次都要遍历**迄今全部**结果，
    `save()` 每次都要序列化**整份增长中的报告**。实测 235 条 × 5 档的
    离线消融因此要 98.6s，节流后 4.5s —— 而它换来的只是"崩溃时少丢最后 5 秒"。
    """
    from npc_agent.eval.compare import Comparison

    writes: list[str] = []
    monkeypatch.setattr(Comparison, "save", lambda self, path: writes.append(str(path)))

    Comparison(base_config=RuntimeConfig(), limit=40).run(
        [RunSpec(label="离线", provider="null")], checkpoint=tmp_path / "ckpt.json"
    )

    assert writes, "跑完那一档至少要落一次盘，否则检查点功能没了"
    assert len(writes) <= 3, (
        f"40 条用例落了 {len(writes)} 次盘 —— 落盘节奏退化成逐条了。"
        "那会把离线消融从 ~5s 拉回 ~99s。"
    )


def test_the_checkpoint_cadence_guard_can_actually_fail(tmp_path, monkeypatch):
    """反向测试：把间隔设成 0，上面那条护栏必须红。

    不这么做的话，"落盘次数很少"可能只是因为 `save()` 压根没被调用 ——
    那样护栏就是靠"什么都没发生"通过的。
    """
    from npc_agent.eval import compare as compare_mod

    monkeypatch.setattr(compare_mod, "CHECKPOINT_MIN_INTERVAL_SEC", 0.0)
    writes: list[str] = []
    monkeypatch.setattr(
        compare_mod.Comparison, "save", lambda self, path: writes.append(str(path))
    )

    compare_mod.Comparison(base_config=RuntimeConfig(), limit=40).run(
        [RunSpec(label="离线", provider="null")], checkpoint=tmp_path / "ckpt.json"
    )

    assert len(writes) >= 40, (
        f"间隔设成 0 时应该退化成逐条（期望 >=40 次），实际 {len(writes)} 次 —— "
        "说明反向测试没打到点子上，上面那条护栏也就不可信。"
    )
