"""HTML 报告渲染的测试。

这里有一条**回归测试**（``test_run_rows_shows_pass_count_not_none``）：
``RunOutcome.to_dict()`` 落的是 ``passed`` / ``total``，而终端表格用的
``Comparison.rows()`` 落的是 ``pass``。渲染层早期直接读 ``run["pass"]``，
于是 HTML 里的"通过"一列渲染成了字面量 ``None``。
"""

from __future__ import annotations

import json

from npc_agent.eval.report import (
    _delta,
    _pass_label,
    render_comparison_html,
    write_comparison_html,
)

SAMPLE = {
    "categories": "all",
    "limit": 0,
    "runs": [
        {
            "label": "离线启发式",
            "spec": {"provider": "null", "model": "", "memory_strategy": "hybrid"},
            "duration_sec": 0.1,
            "pass_rate": 1.0,
            "passed": 10,
            "total": 10,
            "metric_means": {
                "task": 1.0, "tools": 1.0, "memory": 1.0, "persona": 1.0, "safety": 1.0,
            },
            "free_speech_rate": 0.0,
            "avg_speech_chars": 18.0,
            "speeches": ["拿铁好了，趁热", "欢迎，随便坐。今天想喝点什么"],
            "failures": [],
        },
        {
            "label": "模型·kimi-k2.7-code",
            "spec": {"provider": "openai-compat", "model": "kimi-k2.7-code",
                     "memory_strategy": "hybrid"},
            "duration_sec": 817.0,
            "pass_rate": 1.0,
            "passed": 10,
            "total": 10,
            "metric_means": {
                "task": 1.0, "tools": 1.0, "memory": 0.95, "persona": 1.0, "safety": 1.0,
            },
            "free_speech_rate": 0.815,
            "avg_speech_chars": 25.0,
            "speeches": [
                "拿铁好了，趁热",                       # 模板槽位渲染 → 脚本
                "外头的风把招牌吹得咣当响，要变天了。",   # 模型自由组织 → 自由
            ],
            "failures": [
                {"case_id": "memory_preference_recall", "notes": ["memory: 没引用偏好"]}
            ],
        },
    ],
    "deltas": [
        {
            "label": "模型·kimi-k2.7-code", "vs": "离线启发式",
            "task": 0.0, "tools": 0.0, "memory": -0.05, "persona": 0.0, "safety": 0.0,
            "pass_rate": 0.0, "free": 0.815,
        }
    ],
}


# --------------------------------------------------------------------------- #
# 回归测试


def test_run_rows_shows_pass_count_not_none():
    html = render_comparison_html(SAMPLE)
    assert "10/10" in html
    # 早期实现读错了键，渲染出字面量 None
    assert ">None<" not in html


def test_pass_label_handles_both_shapes():
    assert _pass_label({"pass": "3/4"}) == "3/4"          # 终端表格的形态
    assert _pass_label({"passed": 10, "total": 10}) == "10/10"  # JSON 的形态
    assert _pass_label({}) == "—"


# --------------------------------------------------------------------------- #
# 渲染内容


def test_render_escapes_labels():
    data = json.loads(json.dumps(SAMPLE))
    data["runs"][0]["label"] = "<script>alert(1)</script>"
    html = render_comparison_html(data)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_render_includes_headline_free_speech_numbers():
    html = render_comparison_html(SAMPLE)
    assert "0%" in html      # 离线
    assert "82%" in html or "81%" in html  # 模型（四舍五入后 82%）


def test_render_lists_failures():
    html = render_comparison_html(SAMPLE)
    assert "memory_preference_recall" in html
    assert "没引用偏好" in html


def test_render_tags_scripted_vs_free_speech():
    """台词样本必须区分脚本与自由 —— 这是整份报告最有说服力的部分。"""
    html = render_comparison_html(SAMPLE)
    assert "外头的风把招牌吹得咣当响，要变天了。" in html
    assert 'class="tag t-free"' in html      # 模型自由组织
    assert 'class="tag t-scripted"' in html  # 模板槽位渲染


def test_speech_samples_dedupes_and_handles_missing():
    from npc_agent.eval.report import _speech_samples

    runs = [{"label": "x", "speeches": ["同一句", "同一句", "另一句"]}]
    html = _speech_samples(runs)
    assert html.count("同一句") == 1  # 去重

    assert "没有台词记录" in _speech_samples([{"label": "x", "speeches": []}])


def test_render_without_deltas_says_so():
    data = {**SAMPLE, "deltas": []}
    html = render_comparison_html(data)
    assert "没有可对比的差值" in html


def test_unpaired_report_shows_warning():
    """跑批没跑完时必须明说，否则读者会把两张不同的卷子当成同一张。"""
    html = render_comparison_html({**SAMPLE, "paired": False, "paired_cases": 4})
    assert "本次跑批未跑完" in html
    assert "4 条用例" in html


def test_paired_report_has_no_warning():
    html = render_comparison_html({**SAMPLE, "paired": True})
    assert "本次跑批未跑完" not in html
    # 老报告里没有 paired 字段，应当默认视为已配对，不误报警告
    assert "本次跑批未跑完" not in render_comparison_html(SAMPLE)


def test_render_is_self_contained():
    """不能有外部依赖，否则离线打开会白屏。"""
    html = render_comparison_html(SAMPLE)
    assert "http://" not in html and "https://" not in html
    assert "<style>" in html


def test_delta_coloring():
    assert "up" in _delta(0.5)
    assert "down" in _delta(-0.5)
    assert "flat" in _delta(0.0)


def test_write_comparison_html_roundtrip(tmp_path):
    src = tmp_path / "cmp.json"
    src.write_text(json.dumps(SAMPLE, ensure_ascii=False), encoding="utf-8")
    out = write_comparison_html(src, tmp_path / "nested" / "cmp.html")
    assert out.exists()
    assert "10/10" in out.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
def test_table_headers_line_up_with_the_data_columns():
    """表头必须和列数据来自同一个源。

    手写过一次表头，加第六个维度（发言调度）时只改了数据行、忘了表头，
    于是表头和列数据整体错位一格 —— 页面照常渲染、不报任何错，
    只有人眼能看出来。
    """
    from npc_agent.eval.report import _METRIC_LABELS, _run_rows

    html = render_comparison_html(SAMPLE)
    for label in _METRIC_LABELS.values():
        assert f"<th>{label}</th>" in html, f"表头缺少维度 {label}"

    # 绝对值表：4 个前置列 + 各维度 + 3 个尾部列
    header = html.split("<tbody>")[0]
    th_count = header.count("<th>")
    row = _run_rows(SAMPLE["runs"]).split("<tr>")[1]
    td_count = row.count("<td")
    assert th_count == td_count, f"表头 {th_count} 列 vs 数据 {td_count} 列"
    assert th_count == 4 + len(_METRIC_LABELS) + 3


def test_empty_delta_table_span_follows_the_dimension_count():
    """空表的跨列数也要跟着维度数走，否则加一维就撑歪。"""
    from npc_agent.eval.report import _METRIC_LABELS, _delta_rows

    assert f'colspan="{3 + len(_METRIC_LABELS) + 1}"' in _delta_rows([])


def test_tool_catalog_lists_the_scenarios_own_activities():
    """活动是本场景专有的，工具清单里必须列出真实可用的 id。

    曾经硬编码 start_activity(star_quiz)，duet 场景只有 song_request，
    模型照着例子抄，两个 NPC 都去调 start_activity(terrace_night)
    （把目标 id 当成活动 id），白烧两轮。
    """
    from npc_agent.cast import build_cast
    from npc_agent.config import RuntimeConfig, load_scenario
    from npc_agent.llm import build_llm

    for scenario_id, expected in (("duet", "song_request"), ("hosting", "star_quiz")):
        cast = build_cast(load_scenario(scenario_id), build_llm("null"), RuntimeConfig())
        spec = next(
            s for s in cast.lead.registry.specs(cast.lead.id) if s.name == "start_activity"
        )
        rendered = spec.render()
        assert expected in rendered
        # 别的场景的活动名不该泄漏进来
        assert "star_quiz" not in rendered or expected == "star_quiz"
