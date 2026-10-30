"""多 NPC 对话基准自身的护栏。

三条，按"最容易骗自己"的顺序排：

1. **基准必须全绿** —— 不绿就说明能力真的坏了，不能靠改文档绕过去；
2. **满分必须能被证伪** —— 注入缺陷之后该掉分的维度要掉。
   一个从不失败的评测，和一个没有评测，在报告上长得一模一样；
3. **报告必须和代码一致** —— `docs/npc_dialogue.html` 是离线可复现的，
   和代码不一致就等于在说谎。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from npc_agent.eval.dialogue_bench import (
    DIMENSIONS,
    EXPECTED_BREAKAGE,
    cases,
    render_html,
    run_all,
    sensitivity,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "npc_dialogue.html"


def test_every_case_has_a_unique_id_and_a_known_dimension():
    """用例表本身的自洽性：id 撞了、维度写错，报告会静默少统计一条。"""
    all_cases = cases()
    ids = [c.id for c in all_cases]
    assert len(ids) == len(set(ids)), f"用例 id 重复：{ids}"
    for case in all_cases:
        assert case.dim in DIMENSIONS, f"{case.id} 的维度 {case.dim} 不在 {DIMENSIONS}"


def test_the_bench_is_green():
    report = run_all()
    failures = [(r.dim, r.id, r.note) for r in report.results if not r.passed]
    assert not failures, "对话基准有失败用例：\n" + "\n".join(
        f"  [{d}] {i}: {n}" for d, i, n in failures
    )
    assert report.passed == report.total


def test_each_dimension_is_actually_covered():
    """六个维度都要有用例 —— 空维度在报告里会显示 0/0，看起来像通过。"""
    report = run_all()
    covered = set(report.by_dimension())
    assert covered == set(DIMENSIONS), f"这些维度没有用例：{set(DIMENSIONS) - covered}"


@pytest.mark.parametrize("defect", list(EXPECTED_BREAKAGE))
def test_injected_defects_are_caught(defect: str):
    """满分不是"判据从不报警"：关掉能力，对应的维度必须掉分。"""
    baseline = run_all()
    rows = {row["defect"]: row for row in sensitivity(baseline)}
    row = rows[defect]
    assert row["detected"], (
        f"注入「{defect}」之后评测没反应 —— 期望掉分的是 {row['expected']}，"
        f"实际掉分的是 {row['broken']}。这条护栏本身失效了。"
    )
    assert row["passed"] < row["total"]


def test_the_committed_report_matches_the_code():
    """报告过期 = 说谎。重新生成：`python -m npc_agent.eval.dialogue_bench --html docs/npc_dialogue.html`"""
    assert REPORT.exists(), f"{REPORT} 不存在"
    fresh = render_html(run_all(), sensitivity(run_all()))
    committed = REPORT.read_text(encoding="utf-8")
    assert committed == fresh, (
        "docs/npc_dialogue.html 和当前代码生成的结果不一致 —— 它已经过期了。\n"
        "重新生成：python -m npc_agent.eval.dialogue_bench --html docs/npc_dialogue.html"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
