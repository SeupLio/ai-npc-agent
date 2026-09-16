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


def test_render_without_deltas_says_so():
    data = {**SAMPLE, "deltas": []}
    html = render_comparison_html(data)
    assert "没有可对比的差值" in html


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
