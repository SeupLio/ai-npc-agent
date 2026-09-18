"""评测敏感性 —— 证明"满分"不是因为护栏从不报警。

这一组测试守的是 `npc_agent/eval/sensitivity.py`，以及它抓到过的那个真 bug：
**`persona` 维度曾经直接采信被测方自己报的违规列表。**

详细背景见 `sensitivity.py` 的模块文档与 `harness.evaluator_persona_violations`。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from npc_agent.cast import build_cast
from npc_agent.config import RuntimeConfig, load_scenario
from npc_agent.eval import sensitivity as S
from npc_agent.eval.harness import evaluator_persona_violations
from npc_agent.llm import build_llm

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def report() -> S.SensitivityReport:
    """全量跑一遍。7 次 228 条，约十几秒 —— 值这个钱。"""
    return S.run_sensitivity()


# --------------------------------------------------------------------------- #
# 主判据：注入的缺陷必须被抓住
# --------------------------------------------------------------------------- #
def test_no_mutant_survives(report: S.SensitivityReport) -> None:
    """**每个注入的缺陷都必须让评测掉分。**

    这条挂了就说明有盲点：缺陷注进去了，评测还是满分 ——
    那"228/228"这句话就不能再当证据用了。
    """
    survivors = [o.mutant.id for o in report.survivors]
    assert not survivors, (
        "这些变异注入了缺陷却没有任何维度掉分，说明对应维度是瞎的："
        f"{survivors}"
    )


def test_every_mutant_moves_its_target_dimension(report: S.SensitivityReport) -> None:
    """光"注意到了"还不够：**目标维度**必须掉分。

    只让别的维度掉分（旁敲侧击）说明这条维度本身没在工作 ——
    比如"抢话"如果只让 task 掉了、turn_taking 没动，
    那 turn_taking 就是个摆设。
    """
    misses = {
        o.mutant.id: {"targets": o.mutant.targets, "deltas": o.moved}
        for o in report.outcomes
        if not o.target_hit
    }
    assert not misses, f"这些变异没有让目标维度掉分：{misses}"


def test_baseline_is_actually_green(report: S.SensitivityReport) -> None:
    """基线必须是满的 —— 否则"掉分"没法归因到变异上。"""
    assert report.total_cases > 0
    assert report.baseline_pass_rate == 1.0, (
        f"基线就不是满分（{report.baseline_pass_rate}），"
        "那变异掉分说明不了任何事"
    )


# --------------------------------------------------------------------------- #
# 那个真 bug 的回归测试
# --------------------------------------------------------------------------- #
def test_instrument_mutant_is_caught_on_its_target(report: S.SensitivityReport) -> None:
    """**回归测试**：关掉 agent 自己的人设检查器，不能骗过 persona 维度。

    修之前实测：
    - `ooc_phrase_reaches_transcript`（检查器完好）→ persona **−0.612**
    - `ooc_phrase_with_detector_disabled`（关掉检查器）→ persona **0.000**

    同一句出戏台词，缺陷一模一样，分数从 0.388 变成 1.000。
    根因是评测读了被测方自己算的 `turn.persona_violations`。

    修之后两个变异的结果必须**完全相同** —— 因为注入的缺陷本来就相同。
    """
    by_id = {o.mutant.id: o for o in report.outcomes}
    with_detector = by_id["ooc_phrase_reaches_transcript"]
    without_detector = by_id["ooc_phrase_with_detector_disabled"]

    assert without_detector.target_hit, (
        "关掉人设检查器之后 persona 维度又抓不住了 —— "
        "评测是不是又在采信被测方自己报的违规？"
    )
    assert without_detector.deltas["persona"] == pytest.approx(
        with_detector.deltas["persona"]
    ), (
        "两个变异注入的是同一句出戏台词，persona 维度掉的分必须一样："
        f"{with_detector.deltas['persona']} vs {without_detector.deltas['persona']}"
    )
    assert without_detector.pass_rate == pytest.approx(with_detector.pass_rate)


def test_evaluator_persona_audit_matches_persona_check() -> None:
    """评测侧重算的判据，必须和 `Persona.check` 在语料上**逐字一致**。

    这两份实现是**故意分开的**：
    - `Persona.check` 是**被测方**用的（决定拦不拦这句话）
    - `evaluator_persona_violations` 是**评测侧**用的（决定扣不扣分）

    分开是为了"被测方关掉自己的检查器，也影响不到评测"。
    代价是两份逻辑可能漂移 —— 所以这条测试把它们钉在一起。

    ⚠️ 这条测试**不是**在给被测方背书：它只保证"判据一致"，
    不保证"结论由被测方下"。结论永远由评测侧自己算。
    """
    cfg = RuntimeConfig()
    checked = 0

    for scenario_id in ("tutorial", "icebreaker", "duet"):
        scenario = load_scenario(scenario_id)
        cast = build_cast(scenario, build_llm(cfg.llm_provider), cfg)
        for agent in cast.agents.values():
            persona = agent.persona

            texts = [
                "你好，要点什么？",
                "嗯——这杯我来做。",
                "",
                # 人设自己的出戏词，逐个试
                *[f"我想说{p}" for p in persona.forbidden_phrases if p],
                # 人设自己的剧透词，逐个试
                *[f"其实{t}是这样的" for t in persona.spoiler_terms if t],
                # 触发"过长 / 句数超限"的长句
                "第一句。" * 40,
            ]
            for unlocked in (set(), {"随便一个已解锁的话题"}):
                for text in texts:
                    assert evaluator_persona_violations(
                        persona, text, set(unlocked)
                    ) == persona.check(text, set(unlocked)), (
                        f"{scenario_id}/{persona.id}：评测侧重算和 Persona.check 不一致\n"
                        f"  文本={text[:60]!r} unlocked={unlocked}"
                    )
                    checked += 1

    assert checked > 30, f"只比对了 {checked} 组，样本太少，钉不住"


def test_the_sensitivity_check_can_actually_fail(report: S.SensitivityReport) -> None:
    """**护栏本身必须能被验证会红。**

    上面那些断言如果永远返回"没问题"，和不存在没区别。
    这里手动构造一个"没抓住"的结局，确认判据真的会红。
    """
    mutant = S.MUTANTS[0]
    survivor = S.MutantOutcome(
        mutant=mutant,
        pass_rate=1.0,
        metric_means=dict(report.baseline_means),
        deltas={d: 0.0 for d in S.DIMENSIONS},
    )
    assert not survivor.caught, "零变化必须算作'没抓住'"
    assert not survivor.target_hit

    # 反向：目标维度掉了分，就必须算抓住
    hit = S.MutantOutcome(
        mutant=mutant,
        pass_rate=0.5,
        metric_means=dict(report.baseline_means),
        deltas={**{d: 0.0 for d in S.DIMENSIONS}, mutant.targets[0]: -0.5},
    )
    assert hit.caught and hit.target_hit


# --------------------------------------------------------------------------- #
# 补丁必须还原干净
# --------------------------------------------------------------------------- #
def test_every_patch_is_restored(report: S.SensitivityReport) -> None:
    """变异跑完必须**一个都不剩** —— 否则会污染同一进程里后面的测试。

    实测过这个坑的代价：变异没还原，下一个测试拿着被改过的类跑，
    报出来的错完全指向别的地方。
    """
    from npc_agent.agent import NPCAgent
    from npc_agent.modules.memory import MemoryManager, MemoryStore
    from npc_agent.modules.persona import Persona
    from npc_agent.modules.tools import ToolRegistry

    # 挑几个"变异改过、而且能一眼看出改没改"的
    assert NPCAgent._make_plan.__name__ == "_make_plan"
    assert NPCAgent.step.__name__ == "step"
    assert MemoryManager.remember.__name__ == "remember"
    assert MemoryManager.observe.__name__ == "observe"
    assert MemoryStore.search.__name__ == "search"
    assert Persona.check.__name__ == "check"
    assert ToolRegistry._speak.__name__ == "_speak"

    # 再验一个行为：MemoryManager.remember 真的还会存东西
    store = MemoryStore()
    manager = MemoryManager(store)
    record = manager.remember(content="测试", tick=1)
    assert record.id != "mutant-nomem", "remember 还是被替换的那个版本"
    assert len(store.records) == 1, "记忆没真的写进去"


def test_restore_puts_the_original_back() -> None:
    """`_patch` 的还原函数必须真的还原（包括链式补丁的逆序还原）。"""

    class Target:
        def method(self) -> str:
            return "original"

    restore = S._patch(Target, "method", lambda self: "patched")
    assert Target().method() == "patched"
    restore()
    assert Target().method() == "original"

    restore_chain = S._chain(
        S._patch(Target, "method", lambda self: "first"),
        S._patch(Target, "method", lambda self: "second"),
    )
    assert Target().method() == "second"
    restore_chain()
    assert Target().method() == "original"


# --------------------------------------------------------------------------- #
# 报告形状
# --------------------------------------------------------------------------- #
def test_report_is_json_serialisable(report: S.SensitivityReport) -> None:
    import json

    payload = report.to_dict()
    text = json.dumps(payload, ensure_ascii=False)
    assert json.loads(text)["ok"] is True
    assert payload["survivors"] == []
    assert len(payload["mutants"]) == len(S.MUTANTS)
    for entry in payload["mutants"]:
        assert set(entry) >= {"id", "kind", "targets", "caught", "target_hit", "deltas"}


def test_html_report_is_self_contained(report: S.SensitivityReport) -> None:
    """HTML 必须自包含：不许引用任何外部资源（放进作品集要能直接打开）。"""
    html = S.render_sensitivity_html(report)
    assert html.startswith("<!DOCTYPE html>")
    for bad in ("http://", "https://", "<script"):
        assert bad not in html, f"报告里出现了外部依赖或脚本：{bad}"
    for mutant in S.MUTANTS:
        assert mutant.id in html
