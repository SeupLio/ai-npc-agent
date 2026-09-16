"""跨世界报告的测试。

重点不是"HTML 能不能生成"，而是两条**诚实性**约束：

1. 报告必须明确声明自己不是对照实验。两个世界的用例集不同，
   分数相减没有意义 —— 这一点如果只写在 commit message 里，
   下一个看报告的人就会把两行数字减掉。
2. 报告里的每个数字都必须来自真实跑批，不能有手写的常量。
"""

from __future__ import annotations

from npc_agent.config import RuntimeConfig
from npc_agent.eval.worlds import (
    METRIC_LABELS,
    WORLDS,
    render_worlds_html,
    run_worlds,
    worlds_to_json,
    write_worlds_html,
)


def _runs():
    return run_worlds(RuntimeConfig(llm_provider="null"))


# --------------------------------------------------------------------------- #
# 跑批
# --------------------------------------------------------------------------- #
def test_every_world_runs_and_passes() -> None:
    runs = _runs()
    assert len(runs) == len(WORLDS)
    for run in runs:
        assert run.report.total > 0, f"{run.spec.label} 一条用例都没跑到"
        assert run.report.passed == run.report.total, run.spec.label


def test_worlds_cover_disjoint_case_sets() -> None:
    """两个世界的用例集必须**不重叠** —— 否则"跨世界"是假的。

    重叠意味着同一个用例被算了两遍，总分看起来更漂亮，但什么也没证明。
    """
    runs = _runs()
    seen: dict[str, str] = {}
    for run in runs:
        for result in run.report.results:
            assert result.case_id not in seen, (
                f"用例 {result.case_id} 同时出现在 {seen.get(result.case_id)} 和 {run.spec.label}"
            )
            seen[result.case_id] = run.spec.label
    assert len(seen) == sum(r.report.total for r in runs)


def test_minecraft_world_only_contains_minecraft_cases() -> None:
    """每个世界只跑自己那批用例 —— 别把咖啡屋的用例算进体素世界。"""
    runs = {run.spec.env: run for run in _runs()}
    voxel = runs["minecraft"]
    assert {r.category for r in voxel.report.results} == {"minecraft"}
    assert {r.scenario for r in voxel.report.results} == {"village"}

    coffee = runs["star-isle"]
    assert "minecraft" not in {r.category for r in coffee.report.results}


def test_metric_means_are_computed_from_real_results() -> None:
    """六个维度的均值必须来自真实跑批，不是写死的常量。"""
    for run in _runs():
        means = run.means
        assert set(means) == set(METRIC_LABELS)
        # 全通过的跑批，六维都应该是 1.0；这里同时验证了"没有硬编码 0"
        for key, value in means.items():
            assert 0.0 <= value <= 1.0, key
        assert means["task"] == 1.0


# --------------------------------------------------------------------------- #
# 报告内容
# --------------------------------------------------------------------------- #
def test_report_declares_it_is_not_a_controlled_experiment() -> None:
    """最重要的那条断言。

    两个世界的用例数不同，差值没有意义。报告必须自己说清楚，
    否则读的人一定会去相减。
    """
    page = render_worlds_html(_runs())
    assert "这不是一份对照实验" in page
    assert "不同的用例集" in page
    assert "不该相减" in page
    # 受控对照要有指向
    assert "compare" in page and "ablate" in page


def test_report_has_no_delta_table() -> None:
    """没有差值表 —— 这是上面那条声明的结构性落实。

    只写一句"不可相减"、底下却摆一张差值表，等于没说。
    注意别把"所以下面没有差值表"这句话本身当成违规：这里查的是**表头**。
    """
    page = render_worlds_html(_runs())
    assert "<th>差值" not in page
    assert "<th>Δ" not in page
    assert "Δ" not in page
    # 对照报告里的"相对首行的差值"这一节不该出现在这里
    assert "相对首行" not in page


def test_report_lists_each_worlds_structural_traits() -> None:
    """光有分数看不出"不同在哪" —— 报告要把结构性差异列出来。"""
    page = render_worlds_html(_runs())
    for keyword in ["三维坐标", "数量", "昼夜"]:
        assert keyword in page, keyword


def test_report_includes_every_case() -> None:
    page = render_worlds_html(_runs())
    for run in _runs():
        for result in run.report.results:
            assert result.case_id in page, result.case_id


def test_json_report_carries_the_caveat() -> None:
    data = worlds_to_json(_runs())
    assert "不是对照实验" in data["note"]
    assert len(data["worlds"]) == len(WORLDS)
    for world in data["worlds"]:
        assert world["total"] > 0
        assert world["traits"]


def test_write_html_creates_parent_dirs(tmp_path) -> None:
    target = tmp_path / "nested" / "deep" / "worlds.html"
    write_worlds_html(_runs(), target)
    assert target.exists()
    assert "同一套 Agent" in target.read_text(encoding="utf-8")


def test_html_is_self_contained() -> None:
    """报告要能直接发给别人 —— 不能引用任何外部资源。"""
    page = render_worlds_html(_runs())
    assert "<style>" in page
    assert "http://" not in page.replace("http://www.w3.org", "")
    assert "https://" not in page
    assert "<script" not in page
