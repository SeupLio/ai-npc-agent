"""世界能力基准自身的护栏。

和对话基准同一套规矩：

1. 基准必须全绿；
2. 满分必须能被证伪（掐掉能力 → 对应维度掉分）；
3. 报告必须和代码一致（离线可复现，过期就是假话）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from npc_agent.eval.living_world_bench import (
    DEFECTS,
    DIMENSIONS,
    cases,
    render_html,
    run_all,
    sensitivity,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "living_world.html"


def test_every_case_has_a_unique_id_and_a_known_dimension():
    all_cases = cases()
    ids = [c.id for c in all_cases]
    assert len(ids) == len(set(ids)), f"用例 id 重复：{ids}"
    for case in all_cases:
        assert case.dim in DIMENSIONS, f"{case.id} 的维度 {case.dim} 不在 {DIMENSIONS}"


def test_each_dimension_has_cases():
    covered = {c.dim for c in cases()}
    assert covered == set(DIMENSIONS), f"这些维度没有用例：{set(DIMENSIONS) - covered}"


def test_the_bench_is_green():
    report = run_all()
    failures = [(r.dim, r.id, r.note) for r in report.results if not r.passed]
    assert not failures, "世界能力基准有失败用例：\n" + "\n".join(
        f"  [{d}] {i}: {n}" for d, i, n in failures
    )


@pytest.mark.parametrize("defect", list(DEFECTS))
def test_injected_defects_are_caught(defect: str):
    """满分不是"判据从不报警"：掐掉能力，对应维度必须掉分。"""
    baseline = run_all()
    rows = {row["defect"]: row for row in sensitivity(baseline)}
    row = rows[defect]
    assert row["detected"], (
        f"掐掉「{defect}」之后评测没反应 —— 期望掉分 {row['expected']}，"
        f"实际掉分 {row['broken']}。这条护栏本身失效了。"
    )
    assert row["passed"] < row["total"]


def test_the_committed_report_matches_the_code():
    assert REPORT.exists(), f"{REPORT} 不存在"
    fresh = render_html(run_all(), sensitivity(run_all()))
    assert REPORT.read_text(encoding="utf-8") == fresh, (
        "docs/living_world.html 和当前代码生成的结果不一致 —— 它已经过期了。\n"
        "重新生成：python -m npc_agent.eval.living_world_bench --html docs/living_world.html"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
