"""跑批 HTML 报告的测试。

这份报告最容易被写坏的地方不是排版，是**结论**：
一份"通过率 100%"的跑批报告，如果模型一次都没被调用过，
它证明的是框架的确定性逻辑，不是模型能力。所以这里的测试
几乎全部钉在 `trust_summary` 的那一句话上 ——
它是整份报告里唯一会被人引用进简历/周报的结论。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from npc_agent.eval import batch_report as B

METRIC_KEYS = ("task", "tools", "memory", "persona", "safety", "turn_taking")


def _eval_payload(
    *,
    total: int = 4,
    passed: int = 4,
    llm_calls: int = 40,
    llm_failures: int = 0,
    degraded_cases: int = 0,
    failed_cases: int = 0,
    model: str = "kimi-k2.7-code",
    use_llm_planner: bool = False,
    extra_results: list[dict] | None = None,
) -> dict:
    """造一份形状正确的跑批报告。字段名照抄 `EvalReport.to_dict()`。"""
    results = [
        {
            "case_id": f"case_{i}",
            "category": "task" if i % 2 == 0 else "safety",
            "scenario": "duet",
            "description": "d",
            "passed": i < passed,
            "scores": {k: 1.0 for k in METRIC_KEYS},
            "details": {},
            "notes": [] if i < passed else ["目标未完成"],
            "transcript": ["玩家[阿澈] 你好", "阿柚: 你好呀。"],
            "speeches": ["你好呀。"] if i % 2 == 0 else ["这一局我建议先熟悉一下地图。"],
            "speakers": ["阿柚"],
        }
        for i in range(total)
    ]
    results.extend(extra_results or [])
    by_category: dict[str, dict] = {}
    for item in results:
        bucket = by_category.setdefault(
            item["category"], {"total": 0, "passed": 0, "scores": {}, "pass_rate": 0.0}
        )
        bucket["total"] += 1
        bucket["passed"] += 1 if item["passed"] else 0
    for bucket in by_category.values():
        bucket["pass_rate"] = round(bucket["passed"] / bucket["total"], 3)
        bucket["scores"] = {k: 1.0 for k in METRIC_KEYS}

    n = len(results)
    n_passed = sum(1 for r in results if r["passed"])
    return {
        "config": {
            "provider": "openai-compat",
            "model": model,
            "memory_strategy": "hybrid",
            "use_llm_planner": use_llm_planner,
            "use_llm_speech": True,
        },
        "summary": {
            "total": n,
            "passed": n_passed,
            "pass_rate": round(n_passed / n, 3) if n else 0.0,
            "metric_means": {k: 1.0 for k in METRIC_KEYS},
            "by_category": by_category,
        },
        "results": results,
        "batch": {
            "stats": {
                "wall_sec": 120.0,
                "serial_estimate_sec": 600.0,
                "speedup": 5.0,
                "concurrency": 6,
                "efficiency": 0.83,
                "cases": n,
                "reused_cases": 0,
                "executed_cases": n,
                "retried_cases": 0,
                "degraded_cases": degraded_cases,
                "failed_cases": failed_cases,
                "llm_calls": llm_calls,
                "llm_failures": llm_failures,
            },
            "degraded": {
                "total": n,
                "degraded": degraded_cases,
                "failed": failed_cases,
                "llm_calls": llm_calls,
                "llm_failures": llm_failures,
                "degraded_ids": [],
                "failed_ids": [],
                "verdict": "",
            },
        },
    }


# --------------------------------------------------------------------------- #
# 可信度：报告里唯一会被引用进结论的那句话
# --------------------------------------------------------------------------- #
def test_a_batch_with_zero_model_calls_is_flagged_as_measuring_the_framework() -> None:
    """**最重要的一条。** 通过率 100% + 调用 0 次 ≠ 模型很强。

    这正是离线跑批的样子。如果报告只说"100% 通过"，
    读者会把它读成模型能力 —— 那是纯粹的误导。
    """
    trust = B.trust_summary(_eval_payload(total=4, passed=4, llm_calls=0))
    assert trust["trustworthy"] is False
    assert "一次模型都没调用" in trust["verdict"]
    assert "不是模型能力" in trust["verdict"]


def test_high_degradation_disqualifies_the_score() -> None:
    """模板兜底超过 10% 时，通过率里混了框架的功劳，不能当模型能力读。"""
    trust = B.trust_summary(_eval_payload(total=100, passed=100, degraded_cases=30))
    assert trust["trustworthy"] is False
    assert "模板兜底" in trust["verdict"]
    assert "不能当模型能力读" in trust["verdict"]


def test_a_small_amount_of_degradation_is_reported_but_not_disqualifying() -> None:
    """5% 兜底不该让整份报告作废，但必须被说出来 —— 沉默才是最坏的选项。"""
    trust = B.trust_summary(_eval_payload(total=100, passed=100, degraded_cases=5))
    assert trust["trustworthy"] is True
    assert "5/100" in trust["verdict"]
    assert "要知道有这回事" in trust["verdict"]


def test_infrastructure_failures_are_not_dressed_up_as_npc_failures() -> None:
    """跑不起来 ≠ NPC 没做到。这两件事在报告里必须分开说。"""
    trust = B.trust_summary(_eval_payload(total=10, passed=10, failed_cases=3))
    assert trust["trustworthy"] is False
    assert "基础设施故障" in trust["verdict"]
    assert "不是" in trust["verdict"] and "NPC 没做到" in trust["verdict"]


def test_a_clean_batch_is_declared_trustworthy_with_the_call_count() -> None:
    trust = B.trust_summary(_eval_payload(total=228, passed=200, llm_calls=2600))
    assert trust["trustworthy"] is True
    assert "2600" in trust["verdict"]
    assert "无失败" in trust["verdict"]


def test_a_saturated_case_set_is_flagged_as_no_longer_discriminating() -> None:
    """**可信 ≠ 有信息量。**

    99.6% 是一份完全可以信任、却几乎没有信息量的分数：它说明这套用例集
    对这个模型已经饱和，不再能区分「好」和「更好」。
    不写出来的话，读者很容易把 99.6% 读成"NPC 做得几乎完美"，
    而它实际的意思是"我们的卷子太简单了"。
    """
    trust = B.trust_summary(_eval_payload(total=228, passed=227, llm_calls=1480))
    assert trust["saturated"] is True
    assert "天花板" in trust["verdict"]
    assert "饱和" in trust["verdict"]
    # 关键：饱和**不是**不可信 —— 两件事必须分开说，
    # 否则读者会以为分数有问题，而分数没问题，是卷子的问题。
    assert trust["trustworthy"] is True
    assert trust["pass_rate"] == pytest.approx(0.996)


def test_a_high_but_discriminating_pass_rate_is_not_called_saturated() -> None:
    """95% 还有区分度，不该报"饱和"。

    阈值定得太松会让每一份正常报告都带上警告，警告就没人看了。
    """
    trust = B.trust_summary(_eval_payload(total=200, passed=190, llm_calls=2000))
    assert trust["saturated"] is False
    assert "天花板" not in trust["verdict"]


def test_a_tiny_perfect_batch_is_not_called_saturated() -> None:
    """20 条里过 20 条说明不了什么，不能算饱和。"""
    trust = B.trust_summary(_eval_payload(total=8, passed=8, llm_calls=90))
    assert trust["saturated"] is False
    assert "天花板" not in trust["verdict"]


def test_saturation_is_not_reported_for_an_old_payload_without_pass_rate() -> None:
    """并行化之前生成的报告没有 `pass_rate`，不能因此崩掉或误报。"""
    payload = _eval_payload()
    payload["summary"].pop("pass_rate")
    trust = B.trust_summary(payload)
    assert trust["saturated"] is False


def test_an_empty_batch_says_nothing_was_measured() -> None:
    trust = B.trust_summary({"summary": {}, "batch": {}})
    assert trust["trustworthy"] is False
    assert "什么都没测" in trust["verdict"]


def test_trust_survives_a_report_with_no_batch_block() -> None:
    """老报告没有 `batch` 段（并行化之前生成的），不该把渲染炸掉。"""
    payload = _eval_payload()
    payload.pop("batch")
    trust = B.trust_summary(payload)
    assert trust["llm_calls"] == 0
    assert trust["trustworthy"] is False


def test_silent_planner_fallbacks_make_the_score_untrustworthy() -> None:
    """规划失败会**静默回落到启发式规划** —— 这件事必须出现在可信度块里。

    不报的话，`--no-planner` 和"planner 开着但一直在失败"跑出来的轨迹
    完全一样，于是"接上模型规划有没有用"这个对照实验会得到「两组一样」
    的假结论 —— 而读者看不出那是自己跟自己比。
    """
    payload = {"eval": _eval_payload(total=4, passed=4)}
    payload["eval"]["results"][1]["planner_failures"] = 3
    payload["eval"]["results"][1]["planner_last_error"] = (
        "模型返回空内容（finish_reason=length，思维链 3905 字）"
    )
    trust = B.trust_summary(payload["eval"])
    assert trust["planner_failed_cases"] == 1
    assert trust["trustworthy"] is False
    assert "规划调用失败" in trust["verdict"]
    assert "启发式规划" in trust["verdict"]

    page = B.render_batch_html(payload)
    assert "规划回落" in page
    assert "1" in page


def test_a_run_without_planner_failures_stays_trustworthy() -> None:
    """反向测试：没有规划回落时不该被误判。

    假件默认不带 `planner_failures`，所以这条同时守住
    "老报告（没有这个字段）不该被当成有失败"。
    """
    trust = B.trust_summary(_eval_payload(total=10, passed=10, llm_calls=100))
    assert trust["planner_failed_cases"] == 0
    assert trust["trustworthy"] is True


# --------------------------------------------------------------------------- #
# 差值表：用例集不同就不许相减
# --------------------------------------------------------------------------- #
def test_deltas_are_refused_when_the_two_runs_used_different_case_sets() -> None:
    """两个不同的用例集相减，是"拿苹果比橘子"最常见的来源。"""
    cur = _eval_payload(total=4)
    base = _eval_payload(total=9)
    rows = B._baseline_rows(cur, base)
    assert "用例数不一致" in rows
    assert "不做差值" in rows
    # 关键：一个数字都不许出现
    assert "0.000" not in rows


def test_deltas_are_computed_when_the_case_sets_match() -> None:
    cur = _eval_payload(total=4, passed=4)
    base = _eval_payload(total=4, passed=2)
    for key in METRIC_KEYS:
        base["summary"]["metric_means"][key] = 0.5
    rows = B._baseline_rows(cur, base)
    assert "模型更好" in rows
    assert "+0.500" in rows


def test_missing_baseline_is_stated_not_faked() -> None:
    assert "没有提供基线报告" in B._baseline_rows(_eval_payload(), None)


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #
def _render(**kwargs) -> str:
    payload = {"eval": _eval_payload(**kwargs)}
    return B.render_batch_html(payload)


def test_rendering_leaves_no_placeholder_behind() -> None:
    """模板占位符没被替换会静默地渲染出一页 `__FOO__` 文字。

    这种错不会报错，只会让报告看起来像半成品 —— 所以用正则钉住。
    """
    page = _render()
    assert not re.search(r"__[A-Z_]{3,}__", page), re.findall(r"__[A-Z_]{3,}__", page)


def test_the_html_is_self_contained() -> None:
    """单文件、无外部依赖、离线可开 —— 发给人看不会因为 CDN 挂掉变成白屏。"""
    page = _render()
    assert "<!DOCTYPE html>" in page
    assert 'lang="zh-CN"' in page
    assert "<style>" in page
    for bad in ("http://", "https://", "<script", "@import"):
        assert bad not in page, bad


def test_metric_headers_come_from_the_same_source_as_the_cells() -> None:
    """表头和数据列必须同一个数据源。

    历史上手写表头漏过一次第六维（发言调度），表头和数据整体错位一格，
    页面照常渲染、没有任何报错。所以这里数一遍 `<th>` 和 `<td>` 的维数。
    """
    page = _render()
    # 页面上有多张表，必须先定位到「六维指标」那一张，否则会数到配置表上
    section = page.split("<h2>六维指标</h2>", 1)[1].split("<h2>", 1)[0]
    header = re.search(r"<thead><tr>(.*?)</tr></thead>", section, re.S)
    assert header
    labels = list(B._METRIC_LABELS.values())
    for label in labels:
        assert f"<th>{label}</th>" in header.group(1)
    row = re.search(r"<tbody>\s*<tr>(.*?)</tr>", section, re.S)
    assert row
    # 通过 + 六维 + 通过率
    assert row.group(1).count('class="num"') == len(labels) + 2


def test_case_notes_are_escaped() -> None:
    """用例说明是从用例文件里来的自由文本，渲染时必须转义。"""
    payload = {"eval": _eval_payload(total=2, passed=1)}
    payload["eval"]["results"][1]["notes"] = ['<img src=x onerror="alert(1)">']
    page = B.render_batch_html(payload)
    assert "<img src=x" not in page
    assert "&lt;img" in page


def test_failed_cases_are_grouped_by_category_with_their_notes() -> None:
    page = _render(total=4, passed=2)
    assert "目标未完成" in page
    assert "失败明细" in page
    assert "没有失败用例" not in page


def test_a_fully_passing_batch_says_so_instead_of_showing_an_empty_table() -> None:
    page = _render(total=4, passed=4)
    assert "本次跑批没有失败用例" in page


def test_judge_section_reports_coverage_and_refuses_to_hide_unjudged() -> None:
    payload = {
        "eval": _eval_payload(),
        "judge": {
            "judge_model": "kimi-k2.7-code",
            "calibration": {
                "total": 24,
                "judged": 24,
                "unjudged": 0,
                "per_rubric": {
                    "in_character": {
                        "n": 8,
                        "agreement": 1.0,
                        "kappa": 1.0,
                        "reading": "几乎完全一致，这个维度可以用",
                        "confusion": {"真阳性": 3, "真阴性": 5, "假阳性": 0, "假阴性": 0},
                    }
                },
            },
            "summary": {"judged": 100, "unjudged": 7, "by_rubric": {
                "in_character": {"n": 100, "passed": 90, "pass_rate": 0.9}}},
            "coverage": {
                "cases": 40, "cases_failed": 1, "cases_without_dialogue": 2,
                "verdicts": 107, "unjudged": 7,
                "verdict": "有 1/40 条用例判分时炸了（不是「判了 0 分」）。",
            },
        },
    }
    page = B.render_batch_html(payload)
    assert "kappa" in page
    assert "1.00" in page
    assert "未判" in page
    assert "判分时炸了" in page
    # 校准是上界，这件事必须印在页面上，不能只留在提交信息里
    assert "上界" in page
    assert "留出集" in page


def test_judge_section_degrades_gracefully_when_absent() -> None:
    page = _render()
    assert "没有附带裁判结果" in page
    assert "没有校准记录" in page


def test_judge_section_separates_this_round_from_reused_judgements() -> None:
    """「判过 228 条」和「这一轮新判了 228 条」是两件事。

    一次 `--resume` 只判 1 条、复用 227 条，和从头判 228 条，
    报告上都是"228 条判完了"。不把复用的量写出来，
    读者会以为这一轮真的烧了 228 条的调用量 —— 而它只烧了 1 条。
    """
    payload = {
        "eval": _eval_payload(),
        "judge": {
            "judge_model": "kimi-k2.7-code",
            "summary": {"judged": 100, "unjudged": 0, "by_rubric": {
                "in_character": {"n": 100, "passed": 90, "pass_rate": 0.9}}},
            "coverage": {
                "cases": 228, "cases_failed": 0, "cases_without_dialogue": 0,
                "verdicts": 2922, "unjudged": 0,
                "reused": 227, "executed": 1,
                "verdict": "全部 2922 条判决都拿到了分数",
            },
        },
    }
    page = B.render_batch_html(payload)
    assert "本轮新判" in page
    assert "复用检查点" in page
    assert "227 条是从检查点复用的" in page, "复用量必须写在页面上"


def test_judge_section_does_not_claim_reuse_when_there_was_none() -> None:
    """没复用就不要提"复用" —— 凭空多一句会让人以为报告是恢复出来的。"""
    payload = {
        "eval": _eval_payload(),
        "judge": {
            "judge_model": "kimi-k2.7-code",
            "summary": {"judged": 10, "unjudged": 0, "by_rubric": {
                "in_character": {"n": 10, "passed": 9, "pass_rate": 0.9}}},
            "coverage": {
                "cases": 10, "cases_failed": 0, "cases_without_dialogue": 0,
                "verdicts": 30, "unjudged": 0, "reused": 0, "executed": 10,
                "verdict": "全部 30 条判决都拿到了分数",
            },
        },
    }
    page = B.render_batch_html(payload)
    assert "本轮新判" in page
    assert "从检查点复用的" not in page


def test_judge_section_still_renders_for_a_checkpoint_written_by_an_older_version() -> None:
    """旧检查点里没有 `reused`/`executed` 字段，报告不能因此崩掉。

    检查点是要跨版本读的：今天跑了一半，明天升级了代码再恢复。
    缺字段就按 0 显示，别抛 KeyError。
    """
    payload = {
        "eval": _eval_payload(),
        "judge": {
            "judge_model": "kimi-k2.7-code",
            "summary": {"judged": 10, "unjudged": 0, "by_rubric": {
                "in_character": {"n": 10, "passed": 9, "pass_rate": 0.9}}},
            "coverage": {"cases": 10, "verdicts": 30, "unjudged": 0,
                         "verdict": "全部 30 条判决都拿到了分数"},
        },
    }
    page = B.render_batch_html(payload)
    assert "本轮新判" in page
    assert "全部 30 条判决都拿到了分数" in page


def test_the_report_always_states_that_the_case_set_is_self_built() -> None:
    """诚实声明必须是页面的一部分，不能靠读者记得。

    自建用例集的分数不能和外部的 AgentBench / τ-bench 横向比较 ——
    这句话一旦从报告里消失，这份报告就开始骗人了。
    """
    page = _render()
    assert "自建" in page
    assert "不代表" in page
    assert "不能和外部分数横向比较" in page


def test_subtitle_reflects_the_planner_setting() -> None:
    page = B.render_batch_html({"eval": _eval_payload(use_llm_planner=True)})
    assert "LLM 规划" in page
    page2 = B.render_batch_html({"eval": _eval_payload(use_llm_planner=False)})
    assert "启发式规划" in page2


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_writes_the_page_and_reports_trust(tmp_path: Path) -> None:
    from npc_agent.cli import main

    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps(_eval_payload()), encoding="utf-8")
    out = tmp_path / "batch.html"

    assert main(["report-batch", "--eval", str(eval_path), "--html", str(out)]) == 0
    page = out.read_text(encoding="utf-8")
    assert "这份分数可不可信" in page
    assert "case_0" in page or "失败明细" in page


def test_cli_accepts_optional_judge_and_baseline(tmp_path: Path) -> None:
    from npc_agent.cli import main

    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps(_eval_payload()), encoding="utf-8")
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(_eval_payload(passed=2)), encoding="utf-8")
    judge_path = tmp_path / "judge.json"
    judge_path.write_text(json.dumps({"summary": {"by_rubric": {}}}), encoding="utf-8")
    out = tmp_path / "batch.html"

    code = main([
        "report-batch", "--eval", str(eval_path), "--baseline", str(base_path),
        "--judge", str(judge_path), "--html", str(out),
    ])
    assert code == 0
    assert out.exists()


def test_cli_fails_loudly_on_a_missing_input_file(tmp_path: Path) -> None:
    """输入不存在就返回 2，而不是渲染出一份"空报告"假装成功。"""
    from npc_agent.cli import main

    out = tmp_path / "batch.html"
    assert main(["report-batch", "--eval", str(tmp_path / "nope.json"),
                 "--html", str(out)]) == 2
    assert not out.exists()


def test_cli_rejects_a_missing_judge_file(tmp_path: Path) -> None:
    from npc_agent.cli import main

    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps(_eval_payload()), encoding="utf-8")
    out = tmp_path / "batch.html"
    code = main(["report-batch", "--eval", str(eval_path),
                 "--judge", str(tmp_path / "nope.json"), "--html", str(out)])
    assert code == 2
    assert not out.exists()


# --------------------------------------------------------------------------- #
# 留出集：报告必须把「拟合」和「泛化」分开摆
# --------------------------------------------------------------------------- #
def _judge_with_holdout(
    *,
    dev_kappa: float = 1.0,
    hold_kappa: float = 0.75,
    quotable: bool = True,
    problems: list[dict] | None = None,
) -> dict:
    """一份带留出集的裁判 payload。

    刻意让开发集 kappa（1.00）比留出集（0.75）高 —— 这正是真实情况：
    开发集被用来调过 rubric，所以它上面的数字必然更好看。
    报告的价值就在于把这个差印出来，而不是只印好看的那个。
    """
    def _stats(kappa: float, n: int) -> dict:
        return {"n": n, "agreement": kappa, "kappa": kappa, "reading": "基本一致，可以用，但个例要人看",
                "confusion": {"真阳性": 3, "真阴性": 4, "假阳性": 1, "假阴性": 0}}

    return {
        "judge_model": "kimi-k2.7-code",
        "calibration": {
            "total": 24, "judged": 24, "unjudged": 0,
            "per_rubric": {"in_character": _stats(dev_kappa, 8)},
        },
        "holdout": {
            "total": 32, "judged": 32, "unjudged": 0,
            "quotable": quotable,
            "problems": problems if problems is not None else [],
            "per_rubric": {"in_character": _stats(hold_kappa, 11)},
            "seal": {"holdout_digest": "878e44c2ea85b51e", "rubric_digest": "342440f72b7431f6"},
        },
        "summary": {"judged": 100, "unjudged": 0, "by_rubric": {
            "in_character": {"n": 100, "passed": 90, "pass_rate": 0.9}}},
        "coverage": {"cases": 40, "cases_failed": 0, "cases_without_dialogue": 2,
                     "verdicts": 100, "unjudged": 0, "verdict": "判分覆盖完整。"},
    }


def test_report_puts_the_dev_and_holdout_kappa_side_by_side() -> None:
    """**两个数必须同时出现。** 只印一个（不管哪个）都会被读成泛化能力。

    只印开发集 → 把拟合当本事；只印留出集 → 看不出拟合有多大。
    差值才是"这个数字有多少水分"的直接读数。
    """
    page = B.render_batch_html({"eval": _eval_payload(), "judge": _judge_with_holdout()})
    assert "留出集" in page
    assert "开发集" in page
    assert "0.75" in page          # 留出
    assert "1.00" in page          # 开发
    assert "+0.25" in page         # 差值
    assert "拟合的量" in page
    # 封条通过时必须说清"可以引用"，并且把摘要摆出来以便复核
    assert "封条校验通过" in page
    assert "878e44c2ea85b51e" in page


def test_report_refuses_to_quote_a_holdout_with_a_broken_seal() -> None:
    """封条破了就**不能**把那个 kappa 当泛化能力印出来。

    注意这里不是"不显示数字" —— 数字照显示（诊断信息不能扔），
    但必须配一块红色的、说明它为什么不可引用。
    """
    page = B.render_batch_html({
        "eval": _eval_payload(),
        "judge": _judge_with_holdout(quotable=False, problems=[{
            "kind": "rubric_changed",
            "detail": "评分标准变过，这份留出集对当前 rubric 已不再是留出集。",
        }]),
    })
    assert "封条对不上" in page
    assert "不可引用" in page
    assert "rubric_changed" in page
    assert "不可修复" in page
    assert "封条校验通过" not in page


def test_report_says_so_when_there_is_no_holdout_at_all() -> None:
    """没有留出集时不能沉默 —— 沉默会被读成"校准过了"。

    老版本写的 judge.json 没有 holdout 块，这份报告必须仍然能渲染，
    并且明说"这个数只能说明没有系统性偏差"。
    """
    payload = _judge_with_holdout()
    payload.pop("holdout")
    page = B.render_batch_html({"eval": _eval_payload(), "judge": payload})
    assert "没有留出集结果" in page
    assert "不能当泛化能力引用" in page
    assert "封条校验通过" not in page


def test_the_holdout_block_does_not_invent_a_kappa_for_a_missing_rubric() -> None:
    """只在一边出现的维度要显示成「—」，不能补 0。

    补 0 会让"没测"和"测得极差"长得一样 —— 和铁律一是同一类错误。
    """
    payload = _judge_with_holdout()
    payload["holdout"]["per_rubric"]["responsive"] = {
        "n": 11, "agreement": 1.0, "kappa": 1.0, "reading": "几乎完全一致，这个维度可以用",
        "confusion": {},
    }
    page = B.render_batch_html({"eval": _eval_payload(), "judge": payload})
    # in_character 在留出集里有，responsive 在开发集里没有
    assert "—" in page
    assert "0.00" not in page.split("留出集（32 条")[1][:2000]


# --------------------------------------------------------------------------- #
# 留出集：报告必须把「拟合」和「泛化」分开摆
# --------------------------------------------------------------------------- #
def _judge_with_holdout(
    *,
    dev_kappa: float = 1.0,
    hold_kappa: float = 0.75,
    quotable: bool = True,
    problems: list[dict] | None = None,
) -> dict:
    """一份带留出集的裁判 payload。

    刻意让开发集 kappa（1.00）比留出集（0.75）高 —— 这正是真实情况：
    开发集被用来调过 rubric，所以它上面的数字必然更好看。
    报告的价值就在于把这个差印出来，而不是只印好看的那个。
    """
    def _stats(kappa: float, n: int) -> dict:
        return {"n": n, "agreement": kappa, "kappa": kappa, "reading": "基本一致，可以用，但个例要人看",
                "confusion": {"真阳性": 3, "真阴性": 4, "假阳性": 1, "假阴性": 0}}

    return {
        "judge_model": "kimi-k2.7-code",
        "calibration": {
            "total": 24, "judged": 24, "unjudged": 0,
            "per_rubric": {"in_character": _stats(dev_kappa, 8)},
        },
        "holdout": {
            "total": 32, "judged": 32, "unjudged": 0,
            "quotable": quotable,
            "problems": problems if problems is not None else [],
            "per_rubric": {"in_character": _stats(hold_kappa, 11)},
            "seal": {"holdout_digest": "878e44c2ea85b51e", "rubric_digest": "342440f72b7431f6"},
        },
        "summary": {"judged": 100, "unjudged": 0, "by_rubric": {
            "in_character": {"n": 100, "passed": 90, "pass_rate": 0.9}}},
        "coverage": {"cases": 40, "cases_failed": 0, "cases_without_dialogue": 2,
                     "verdicts": 100, "unjudged": 0, "verdict": "判分覆盖完整。"},
    }


def test_report_puts_the_dev_and_holdout_kappa_side_by_side() -> None:
    """**两个数必须同时出现。** 只印一个（不管哪个）都会被读成泛化能力。

    只印开发集 → 把拟合当本事；只印留出集 → 看不出拟合有多大。
    差值才是"这个数字有多少水分"的直接读数。
    """
    page = B.render_batch_html({"eval": _eval_payload(), "judge": _judge_with_holdout()})
    assert "留出集" in page
    assert "开发集" in page
    assert "0.75" in page          # 留出
    assert "1.00" in page          # 开发
    assert "+0.25" in page         # 差值
    assert "拟合的量" in page
    # 封条通过时必须说清"可以引用"，并且把摘要摆出来以便复核
    assert "封条校验通过" in page
    assert "878e44c2ea85b51e" in page


def test_report_refuses_to_quote_a_holdout_with_a_broken_seal() -> None:
    """封条破了就**不能**把那个 kappa 当泛化能力印出来。

    注意这里不是"不显示数字" —— 数字照显示（诊断信息不能扔），
    但必须配一块红色的、说明它为什么不可引用。
    """
    page = B.render_batch_html({
        "eval": _eval_payload(),
        "judge": _judge_with_holdout(quotable=False, problems=[{
            "kind": "rubric_changed",
            "detail": "评分标准变过，这份留出集对当前 rubric 已不再是留出集。",
        }]),
    })
    assert "封条对不上" in page
    assert "不可引用" in page
    assert "rubric_changed" in page
    assert "不可修复" in page
    assert "封条校验通过" not in page


def test_report_says_so_when_there_is_no_holdout_at_all() -> None:
    """没有留出集时不能沉默 —— 沉默会被读成"校准过了"。

    老版本写的 judge.json 没有 holdout 块，这份报告必须仍然能渲染，
    并且明说"这个数只能说明没有系统性偏差"。
    """
    payload = _judge_with_holdout()
    payload.pop("holdout")
    page = B.render_batch_html({"eval": _eval_payload(), "judge": payload})
    assert "没有留出集结果" in page
    assert "不能当泛化能力引用" in page
    assert "封条校验通过" not in page


def test_the_holdout_block_does_not_invent_a_kappa_for_a_missing_rubric() -> None:
    """只在一边出现的维度要显示成「—」，不能补 0。

    补 0 会让"没测"和"测得极差"长得一样 —— 和铁律一是同一类错误。
    """
    payload = _judge_with_holdout()
    payload["holdout"]["per_rubric"]["responsive"] = {
        "n": 11, "agreement": 1.0, "kappa": 1.0, "reading": "几乎完全一致，这个维度可以用",
        "confusion": {},
    }
    page = B.render_batch_html({"eval": _eval_payload(), "judge": payload})
    # in_character 在留出集里有，responsive 在开发集里没有
    assert "—" in page
    assert "0.00" not in page.split("留出集（32 条")[1][:2000]


def test_report_flags_judge_parse_retries_as_a_budget_problem() -> None:
    """解析失败的重试必须单独提示 —— 它的处置办法和网络抖动完全不同。

    "裁判返回的内容解析不了"通常意味着思维链把输出预算吃光了，
    该做的是加预算/换模型；而调用失败该做的是查网络。
    两者混在一个"重试 N 次"里，读者会去查错方向。
    """
    payload = _judge_with_holdout()
    payload["coverage"]["judge_retries"] = 5
    payload["coverage"]["judge_parse_retries"] = 2
    page = B.render_batch_html({"eval": _eval_payload(), "judge": payload})
    assert "裁判返回的内容解析不了" in page
    assert "JUDGE_MAX_TOKENS" in page
    assert "作废检查点" in page


def test_report_stays_quiet_when_no_parse_retry_happened() -> None:
    """没发生过就别报 —— 常驻的警告等于没有警告。"""
    payload = _judge_with_holdout()
    payload["coverage"]["judge_retries"] = 3
    payload["coverage"]["judge_parse_retries"] = 0
    page = B.render_batch_html({"eval": _eval_payload(), "judge": payload})
    assert "裁判返回的内容解析不了" not in page


def test_report_flags_judge_parse_retries_as_a_budget_problem() -> None:
    """解析失败的重试必须单独提示 —— 它的处置办法和网络抖动完全不同。

    "裁判返回的内容解析不了"通常意味着思维链把输出预算吃光了，
    该做的是加预算/换模型；而调用失败该做的是查网络。
    两者混在一个"重试 N 次"里，读者会去查错方向。
    """
    payload = _judge_with_holdout()
    payload["coverage"]["judge_retries"] = 5
    payload["coverage"]["judge_parse_retries"] = 2
    page = B.render_batch_html({"eval": _eval_payload(), "judge": payload})
    assert "裁判返回的内容解析不了" in page
    assert "JUDGE_MAX_TOKENS" in page
    assert "作废检查点" in page


def test_report_stays_quiet_when_no_parse_retry_happened() -> None:
    """没发生过就别报 —— 常驻的警告等于没有警告。"""
    payload = _judge_with_holdout()
    payload["coverage"]["judge_retries"] = 3
    payload["coverage"]["judge_parse_retries"] = 0
    page = B.render_batch_html({"eval": _eval_payload(), "judge": payload})
    assert "裁判返回的内容解析不了" not in page
