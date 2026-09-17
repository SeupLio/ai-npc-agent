"""判分检查点 / 恢复的回归测试。

判分比跑批更长（228 条用例约四小时、两千多次模型调用），
但断点续跑一直只有跑批那边有。这里补上，重点钉住几个
"看起来对、其实会悄悄丢数据"的形状。

和跑批那侧的 `test_runner.py` 是同一套设计（配置指纹 + 只复用真跑完的），
但**判分多一个坑**：判分失败不抛异常，见
`test_a_judgement_that_scored_nothing_must_not_be_reused`。
"""

from __future__ import annotations

import json

import pytest

from npc_agent.eval import judge as J
from npc_agent.eval.runner import Checkpoint
from npc_agent.llm import ScriptedLLM


_TRANSCRIPT = [
    "玩家[阿澈] 阿柚，来杯拿铁。",
    "  [ok] 阿柚 speak(text=好嘞，稍等, to=player_a)",
    "阿柚: 好嘞，稍等",
]


def _fake_results(n: int) -> list[dict]:
    return [
        {
            "case_id": f"case_{i:02d}",
            "category": "task",
            "scenario": "tutorial",
            "transcript": list(_TRANSCRIPT),
        }
        for i in range(n)
    ]


def _scripted_judge() -> J.LLMJudge:
    return J.LLMJudge(
        ScriptedLLM(lambda _p: '{"score": 1, "reason": "还行"}'), name="fake-judge"
    )


def _judged(case_id: str, *, score: float | None = 1.0, index: int = 0) -> J.CaseJudgement:
    return J.CaseJudgement(
        index=index,
        case_id=case_id,
        category="task",
        scenario="tutorial",
        pairs=[
            {
                "player": "来杯拿铁",
                "reply": "好嘞",
                "verdicts": [
                    {
                        "rubric": "in_character",
                        "judged": score is not None,
                        "score": score,
                        "reason": "还行" if score is not None else "",
                        "error": "" if score is not None else "模型调用失败",
                    }
                ],
            }
        ],
    )


# --------------------------------------------------------------------------- #
# 序列化
# --------------------------------------------------------------------------- #


def test_a_judgement_survives_a_checkpoint_round_trip() -> None:
    original = _judged("case_01")
    restored = J.CaseJudgement.from_dict(original.to_dict())
    assert restored.case_id == original.case_id
    assert restored.category == original.category
    assert restored.scenario == original.scenario
    assert restored.verdicts() == original.verdicts()
    assert restored.reusable is True


# --------------------------------------------------------------------------- #
# 什么能复用、什么不能
# --------------------------------------------------------------------------- #


def test_a_judgement_that_scored_nothing_must_not_be_reused() -> None:
    """**这条是判分恢复里最容易错的地方。**

    判分失败走的是 `Verdict.unjudged` —— 它**不抛异常**，所以
    `error` 是空字符串、`ok` 为真。只看 `ok` 就会把"裁判当时连不上"
    当成"判完了"永久写进检查点：以后每次 `--resume` 都跳过它，
    报告里那几条永远缺判决，而且没有任何地方提示要去重判。

    跑批那边的判据（`run.ok`，即 `result is not None`）在判分这里不够用，
    因为判分的失败是**返回值**，不是异常。
    """
    nothing_scored = _judged("case_01", score=None)
    assert nothing_scored.ok is True, "前提：它确实没抛异常"
    assert nothing_scored.reusable is False, "一条判决都没拿到，必须重判"

    # 对照组：真的判出分数了就可以复用。0 分是有效判决，不是"没判"。
    assert _judged("case_02", score=1.0).reusable is True
    assert _judged("case_03", score=0.0).reusable is True


def test_a_crashed_judgement_must_not_be_reused() -> None:
    crashed = J.CaseJudgement(index=0, case_id="case_01", error="TimeoutError: 超时")
    assert crashed.ok is False
    assert crashed.reusable is False


def test_a_case_with_no_dialogue_is_reusable() -> None:
    """没有对话可判是**确定性**结果，复用不会丢信息。"""
    empty = J.CaseJudgement(index=0, case_id="case_01")
    assert empty.reusable is True


# --------------------------------------------------------------------------- #
# 指纹：管"裁判怎么判"
# --------------------------------------------------------------------------- #


def test_report_digest_changes_when_the_speeches_change() -> None:
    """同一批 `case_id`、不同的台词，必须算出不同的摘要。

    检查点是按 `case_id` 复用的。换模型重跑之后 `case_id` 一个都没变，
    但台词全变了 —— 没有这个摘要，`--resume` 会拿旧批的判决去补新批，
    报告显示"这条判过了"，实际判的是上一批的台词。这比没判更糟，
    因为它看起来有数据。
    """
    a = _fake_results(3)
    b = [dict(r) for r in a]
    b[1]["transcript"] = [
        "玩家[阿澈] 来杯拿铁",
        "  [ok] 阿柚 speak(text=换了一句完全不同的台词, to=player_a)",
    ]

    # 同样的内容 → 同样的摘要（否则每次恢复都会被拒绝）
    assert J.report_digest(a) == J.report_digest([dict(r) for r in a])
    # 内容变了 → 摘要必须变
    assert J.report_digest(a) != J.report_digest(b)


def test_judge_resume_refuses_when_the_speeches_are_different() -> None:
    results = _fake_results(2)
    fingerprint = J.judge_fingerprint(_scripted_judge(), results)
    payload = {
        "fingerprint": fingerprint,
        "judgements": [_judged("case_00").to_dict()],
    }
    # 同一份指纹 → 可以复用
    usable, note = J.plan_judge_resume(payload, fingerprint)
    assert set(usable) == {"case_00"}, note

    # 台词变了 → 整体拒绝，而不是"能复用几条算几条"
    changed = [dict(r) for r in results]
    changed[0]["transcript"] = [
        "玩家[阿澈] 你好",
        "  [ok] 阿柚 speak(text=你好呀, to=player_a)",
    ]
    other = J.judge_fingerprint(_scripted_judge(), changed)
    usable2, note2 = J.plan_judge_resume(payload, other)
    assert usable2 == {}
    assert "拒绝恢复" in note2
    assert "report_digest" in note2, "要指明是哪一项对不上"


@pytest.mark.parametrize(
    "field,value",
    [
        ("judge", "另一个模型"),
        ("rubrics", ["in_character"]),
        ("max_tokens", 512),
        ("temperature", 1.0),
    ],
)
def test_judge_resume_refuses_when_the_judge_config_changed(field, value) -> None:
    """换裁判模型 / 改标准 / 改预算 / 改温度，都必须拒绝恢复。

    否则报告里会出现"一半是 3 条标准判的、一半是 2 条标准判的"，
    而表头只会写一套 —— 正是跑批那边踩过的"两个配置混成一列"。
    """
    results = _fake_results(2)
    fingerprint = J.judge_fingerprint(_scripted_judge(), results)
    payload = {"fingerprint": fingerprint, "judgements": [_judged("case_00").to_dict()]}

    changed = dict(fingerprint)
    changed[field] = value
    usable, note = J.plan_judge_resume(payload, changed)
    assert usable == {}, f"{field} 变了却还在复用"
    assert field in note, note


def test_judge_resume_keeps_the_good_and_drops_the_empty() -> None:
    """一次恢复里，能复用的复用，一条判决都没拿到的重判。"""
    results = _fake_results(3)
    fingerprint = J.judge_fingerprint(_scripted_judge(), results)
    payload = {
        "fingerprint": fingerprint,
        "judgements": [
            _judged("case_00").to_dict(),
            _judged("case_01", score=None).to_dict(),  # 什么都没判到 → 重判
        ],
    }
    usable, note = J.plan_judge_resume(payload, fingerprint)
    assert set(usable) == {"case_00"}
    assert "1 条一条判决都没拿到" in note


def test_an_empty_checkpoint_is_not_an_error() -> None:
    usable, note = J.plan_judge_resume({}, J.judge_fingerprint(_scripted_judge(), []))
    assert usable == {}
    assert "没有已完成的结果" in note


# --------------------------------------------------------------------------- #
# 编排：真的只跑缺的那些
# --------------------------------------------------------------------------- #


def test_judging_with_resume_only_runs_the_missing_cases() -> None:
    results = _fake_results(4)
    fingerprint = J.judge_fingerprint(_scripted_judge(), results)
    payload = {
        "fingerprint": fingerprint,
        "judgements": [_judged("case_00").to_dict(), _judged("case_02").to_dict()],
    }
    usable, _ = J.plan_judge_resume(payload, fingerprint)

    calls: list[str] = []
    original = J.LLMJudge.judge_pairs

    def counting(self, pairs, **kwargs):  # noqa: ANN001, ANN003
        calls.append("call")
        return original(self, pairs, **kwargs)

    J.LLMJudge.judge_pairs = counting
    try:
        judgements, stats = J.judge_report_cases(
            results,
            judge=_scripted_judge(),
            persona_of=lambda _s: "你是阿柚",
            scene_of=lambda _s: "你在吧台",
            concurrency=2,
            resume=usable,
        )
    finally:
        J.LLMJudge.judge_pairs = original

    assert stats["reused"] == 2
    assert stats["executed"] == 2
    # 复用的两条一次模型都没调 —— 恢复的意义就在这里
    assert len(calls) == 2
    # 复用的结果仍然落在正确的下标上，顺序和跑批报告一致
    assert [j.case_id for j in judgements] == [r["case_id"] for r in results]
    assert [j.index for j in judgements] == [0, 1, 2, 3]


def test_the_judge_checkpoint_is_written_after_every_case(tmp_path) -> None:
    """检查点必须**边判边写**，而不是判完再统一落盘。

    "判完再统一落盘"在进程被杀时一条都救不回来 ——
    这正是跑批第一版犯过的错（只存"跑过没跑过"），判分不能重犯。

    `writes == 用例数` 就证明了它是逐条写的：只写一次的话这里会是 1。

    （注意 `on_done` 在 `checkpoint.save()` **之前**被调用，
    因为调用方是在 `on_done` 里把结果收进累积字典的，save 再对那个字典
    拍快照。所以第 1 条判完的那一刻磁盘上还没有它 —— 这是设计，不是 bug。）
    """
    results = _fake_results(5)
    seen: dict = {}
    order = [r["case_id"] for r in results]

    def snapshot() -> dict:
        return {
            "fingerprint": {"judge": "fake-judge"},
            "total": len(results),
            "done": len(seen),
            "judgements": [seen[c].to_dict() for c in order if c in seen],
        }

    path = tmp_path / "judge_ckpt.json"
    checkpoint = Checkpoint(path, snapshot)

    def on_done(judgement, _done, _total):
        seen[judgement.case_id] = judgement

    J.judge_report_cases(
        results,
        judge=_scripted_judge(),
        persona_of=lambda _s: "你是阿柚",
        scene_of=lambda _s: "你在吧台",
        concurrency=2,
        on_done=on_done,
        checkpoint=checkpoint,
    )

    assert checkpoint.writes == 5, "应该每判完一条就写一次"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["done"] == 5
    assert [j["case_id"] for j in payload["judgements"]] == order


def test_a_checkpoint_taken_mid_run_can_finish_the_rest(tmp_path) -> None:
    """**这条是判分检查点存在的理由，端到端跑一遍。**

    第一轮只判前 2 条就"崩"（只给 2 条的结果），
    第二轮带着检查点恢复，必须只补判剩下的 3 条 —— 而且结果要和
    "一次判完 5 条"完全一致。如果恢复只是把下标错位、或者把复用的
    结果当成空的，这里会立刻暴露。
    """
    results = _fake_results(5)
    fingerprint = J.judge_fingerprint(_scripted_judge(), results)

    # ---- 第一轮：只判前 2 条，落一个"崩在半路"的检查点 ----
    calls: list[str] = []
    original = J.LLMJudge.judge_pairs

    def counting(self, pairs, **kwargs):  # noqa: ANN001, ANN003
        calls.append("call")
        return original(self, pairs, **kwargs)

    J.LLMJudge.judge_pairs = counting
    try:
        first_judge = _scripted_judge()
        partial: list = []
        J.judge_report_cases(
            results[:2],
            judge=first_judge,
            persona_of=lambda _s: "你是阿柚",
            scene_of=lambda _s: "你在吧台",
            concurrency=1,
            on_done=lambda j, _d, _t: partial.append(j),
        )
        assert len(calls) == 2

        # ---- 第二轮：带着第一轮的结果恢复 ----
        ckpt = {
            "fingerprint": fingerprint,
            "total": len(results),
            "done": len(partial),
            "judgements": [j.to_dict() for j in partial],
        }
        usable, note = J.plan_judge_resume(ckpt, fingerprint)
        assert len(usable) == 2, note

        calls.clear()
        judgements, stats = J.judge_report_cases(
            results,
            judge=_scripted_judge(),
            persona_of=lambda _s: "你是阿柚",
            scene_of=lambda _s: "你在吧台",
            concurrency=2,
            resume=usable,
        )

        # 只补判了 3 条
        assert len(calls) == 3, "复用的那两条不该再调模型"
        assert stats["reused"] == 2
        assert stats["executed"] == 3
    finally:
        J.LLMJudge.judge_pairs = original

    # 恢复出来的整份结果，和一次判完 5 条逐条一致
    fresh, _ = J.judge_report_cases(
        results,
        judge=_scripted_judge(),
        persona_of=lambda _s: "你是阿柚",
        scene_of=lambda _s: "你在吧台",
        concurrency=1,
    )
    assert [j.case_id for j in judgements] == [j.case_id for j in fresh]
    assert [j.verdicts() for j in judgements] == [j.verdicts() for j in fresh]
    assert [j.index for j in judgements] == [0, 1, 2, 3, 4]


def test_the_judge_checkpoint_survives_a_truncated_file(tmp_path) -> None:
    """读坏了就当没有，不能因此崩掉 —— 检查点不能变成新的故障源。"""
    from npc_agent.eval.runner import load_checkpoint

    path = tmp_path / "broken.json"
    path.write_text('{"fingerprint": {"judge": "x"}, "judg', encoding="utf-8")
    assert load_checkpoint(path) == {}


# --------------------------------------------------------------------------- #
# 覆盖情况要分开报
# --------------------------------------------------------------------------- #


def test_coverage_reports_reused_and_executed_separately() -> None:
    """一次 `--resume` 只判 1 条，和从头判 228 条，报告上都写"228 条判完了"。

    不把"这次实际判了多少"写出来，就无从判断这份报告里有多少判决
    是这一轮新产生的、这一轮到底烧了多少调用。
    """
    judgements = [_judged(f"case_{i:02d}", index=i) for i in range(5)]
    stats = J.judge_coverage(judgements, concurrency=4, reused=4, executed=1)
    assert stats["cases"] == 5
    assert stats["reused"] == 4
    assert stats["executed"] == 1

    # 不传时，默认全部算执行
    plain = J.judge_coverage(judgements, concurrency=4)
    assert plain["reused"] == 0
    assert plain["executed"] == 5
