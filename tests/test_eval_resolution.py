"""评测的分辨率 —— 证明"满分"不是"评测测不出退化"。

守的是 `npc_agent/eval/resolution.py`。它和 `test_sensitivity.py` 是一对：

* `test_sensitivity.py` 管**二值**问题：注入缺陷 → 抓到没有（6/6）；
* 这里管**连续**问题：掉多少，评测才开始动（剂量-反应曲线）。

⚠️ **数值本身不在这里守。** `docs/resolution.html` 已经被
`test_docs_freshness.py` 的 `FAST_OFFLINE` 钉住了（逐字节重生成比对），
所以"曲线上的数字变了"会在那边红。这里只守**机制与契约**：

1. 剂量函数 `_subsample` 的边界与单调性（它是整条曲线的地基）；
2. **k=1 必须真的等于"不做规划"** —— 报告里写着"这是个免费的自洽性检验"，
   这条断言就是那句话的牙齿。改坏了它，报告会静默变成假话；
3. 渲染契约：HTML 里的用例条数**必须能被 `report_index` 解析出来**，
   否则报告门户会把它显示成"（未标注）"。
"""

from __future__ import annotations

import pytest

from npc_agent.eval import resolution as R
from npc_agent.eval.report_index import case_count

# --------------------------------------------------------------------------- #
# 剂量函数：整条曲线的地基
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 10])
def test_k_zero_keeps_every_step(n: int) -> None:
    assert R._subsample(n, 0.0) == list(range(n))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 10])
def test_k_one_removes_every_step(n: int) -> None:
    """**这条就是报告里那句"k=1 等价于 `planning_disabled`"的牙齿。**

    `install_graded_skip` 在 `keep` 为空时返回 `None`，而
    `install_planning_disabled` 是让 `_make_plan` 直接返回 `None` ——
    两者同形，所以 k=1 的读数才应该等于 `sensitivity.html` 里那个 59.1%。
    如果哪天 `_subsample` 改了，让 k=1 还留下一步，那个对账就**静默失效**了。
    """
    assert R._subsample(n, 1.0) == []


def test_empty_plan_is_handled() -> None:
    """没有步骤的计划不该炸 —— 它本来就没东西可丢。"""
    assert R._subsample(0, 0.0) == []
    assert R._subsample(0, 0.5) == []
    assert R._subsample(0, 1.0) == []


def test_subsample_is_deterministic() -> None:
    """同一个 k 必须给出同一份计划 —— 离线路径要可复现。"""
    for n in (3, 5, 8, 13):
        for k in R.DOSES:
            assert R._subsample(n, k) == R._subsample(n, k), f"n={n} k={k} 不稳定"


def test_subsample_shrinks_monotonically() -> None:
    """剂量越大，留下的步骤只会越来越少（不会反弹）。"""
    for n in (4, 5, 8, 13):
        sizes = [len(R._subsample(n, k)) for k in R.DOSES]
        assert sizes == sorted(sizes, reverse=True), f"n={n} 的规模不是单调递减：{sizes}"


def test_subsample_indices_are_valid_and_unique() -> None:
    """下标必须升序、去重、且落在 [0, n) 内 —— 否则 `plan.steps[i]` 会取错步。"""
    for n in (1, 2, 5, 9):
        for k in R.DOSES:
            kept = R._subsample(n, k)
            assert kept == sorted(set(kept)), f"n={n} k={k} 下标重复或乱序：{kept}"
            assert all(0 <= i < n for i in kept), f"n={n} k={k} 下标越界：{kept}"


def test_subsample_spreads_drops_instead_of_cutting_the_tail() -> None:
    """丢掉的步骤要**分散**，不能固定砍尾巴。

    固定砍尾巴的话，k=0.2 永远等于"丢掉最后那个改变世界的步骤"，
    曲线会退化成阶跃 —— 那就量不出"分辨率"，只量得出"有没有最后一步"。
    """
    n = 10
    kept = R._subsample(n, 0.3)
    assert len(kept) == 7
    # 「砍尾巴」会留下一个**连续前缀** `range(7)`；均匀抽样会在中间留洞。
    assert kept != list(range(len(kept))), f"这就是在砍尾巴：{kept}"
    # 留下的下标要跨到后半段，而不是挤在前半段
    assert kept[-1] >= int(n * 0.7), f"留下的下标没跨到后半段：{kept}"


# --------------------------------------------------------------------------- #
# 补丁必须还原干净
# --------------------------------------------------------------------------- #
def test_graded_skip_restores_the_original() -> None:
    """剂量跑完必须还原 —— 否则下一个剂量会叠在它上面，曲线就没意义了。"""
    from npc_agent.agent import NPCAgent

    original = NPCAgent._make_plan
    restore = R.install_graded_skip(0.5)
    try:
        assert NPCAgent._make_plan is not original, "补丁没生效"
    finally:
        restore()
    assert NPCAgent._make_plan is original, "补丁没还原"


# --------------------------------------------------------------------------- #
# 渲染契约
# --------------------------------------------------------------------------- #
def _fake_result(**over: object) -> dict:
    """一份形状正确的假结果，用来测渲染（不跑 235 条）。"""
    points = [
        {
            "k": k,
            "passed": 235,
            "total": 235,
            "pass_rate": 1.0,
            "metric_means": {d: 1.0 for d in R.DIMENSIONS},
        }
        for k in R.DOSES
    ]
    base: dict = {
        "doses": list(R.DOSES),
        "points": points,
        "baseline_pass_rate": 1.0,
        "total_cases": 235,
        "detection_threshold_k": None,
        "blind_dimensions": list(R.DIMENSIONS),
    }
    base.update(over)
    return base


def test_html_is_self_contained() -> None:
    """HTML 不许引用任何外部资源（放进作品集要能直接打开）。"""
    html = R.render_resolution_html(_fake_result())
    assert html.startswith("<!DOCTYPE html>")
    for bad in ("http://", "https://", "<script"):
        assert bad not in html, f"报告里出现了外部依赖或脚本：{bad}"


def test_html_case_count_is_parseable_by_the_portal() -> None:
    """**契约**：报告里的用例条数必须能被 `report_index.case_count` 读出来。

    读不出来的话，报告门户会把它显示成"（未标注）" —— 不是报错，是静默降级。
    这条把"副标题的写法"和"门户的解析器"钉在一起。
    """
    html = R.render_resolution_html(_fake_result())
    assert case_count(html) == 235


def test_html_reports_an_insensitive_result_instead_of_hiding_it() -> None:
    """阈值是 None（全程没掉分）时，报告必须**明说不敏感**，不能留白。

    "没掉分"有两种可能：评测很稳，或者评测是瞎的。报告不能让人自己去猜 ——
    这正是本项目"回落分类不许有『其他』"那条规矩的同一个形状。
    """
    html = R.render_resolution_html(_fake_result(detection_threshold_k=None))
    assert "不敏感" in html

    html_hit = R.render_resolution_html(_fake_result(detection_threshold_k=0.1))
    assert "10%" in html_hit
    assert "不敏感" not in html_hit


def test_html_names_the_blind_dimensions() -> None:
    """恒为 1.000 的维度必须被点名 —— 那是这份报告最该被看见的结论。"""
    html = R.render_resolution_html(
        _fake_result(blind_dimensions=["memory", "persona", "safety", "turn_taking"])
    )
    for dim in ("memory", "persona", "safety", "turn_taking"):
        assert f"<code>{dim}</code>" in html


def test_render_is_deterministic() -> None:
    """同样的结果必须渲染出同样的 HTML —— 新鲜度护栏靠它逐字节比对。"""
    result = _fake_result(detection_threshold_k=0.1)
    assert R.render_resolution_html(result) == R.render_resolution_html(result)


def test_html_does_not_embed_a_wall_clock() -> None:
    """报告里**不许**嵌时长。

    新鲜度护栏只抹平 `⟨时长⟩` 那一类字段；这份报告是纯计算、无计时的，
    所以更不该引入时间戳 —— 一旦引入，`test_docs_freshness` 就会常年红着，
    然后有人去加 skip，等于把护栏关掉。
    """
    html = R.render_resolution_html(_fake_result(detection_threshold_k=0.1))
    for bad in ("秒", "耗时", "elapsed"):
        assert bad not in html, f"报告里出现了时间相关字样：{bad}"
