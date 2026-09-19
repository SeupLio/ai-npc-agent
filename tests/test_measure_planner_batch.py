"""`scripts/measure_planner_batch.py` 的回归测试。

这份报告的作用是**替代**一句没法验证的话（"模型规划不稳"），
所以它自己必须比那句话更经得起查。这里钉四类事：

1. 两种读数形状都要认得（跑批中途落检查点、跑完落报告）。
2. 配对规则与「干净子集」的算法是**可算错的**，所以要单独测，
   而不是只在真读数上跑一遍看它没崩。
3. 报告里的**范围声明**（这是子集、只有一个模型、台词走模板）
   必须真的印出来 —— 这类"限制说明"最容易在改版式时被顺手删掉。
4. 整个设计依赖一个事实：加载顺序的前 60 条是 10×6 分层。
   这条被当成前提用了，所以必须钉住；哪天用例生成器改了排序，
   报告会**悄悄**从"分层样本"变成"先写的那 60 条"。
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    """按路径加载 `scripts/measure_planner_batch.py`（`scripts/` 不是包）。"""
    path = ROOT / "scripts" / "measure_planner_batch.py"
    assert path.exists(), f"{path} 不在"
    spec = importlib.util.spec_from_file_location("_measure_planner_batch_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# --------------------------------------------------------------------------- #
# 1. 两种读数形状


def _result(
    case_id: str,
    *,
    passed: bool,
    task: float = 1.0,
    planner_failures: int = 0,
    category: str = "task",
    task_detail: str = "世界状态符合预期",
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "category": category,
        "passed": passed,
        "scores": {"task": task, "tools": 1.0, "memory": 1.0,
                   "persona": 1.0, "safety": 1.0, "turn_taking": 1.0},
        "details": {"task": task_detail},
        "planner_failures": planner_failures,
        "planner_last_error": "" if not planner_failures else "JSONDecodeError",
    }


def _write_checkpoint(path: Path, results: list[dict[str, Any]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "done": len(results),
                "total": len(results),
                "config": {"model": "fake"},
                "runs": [
                    {"index": i, "ok": True, "result": r} for i, r in enumerate(results)
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _write_report(path: Path, results: list[dict[str, Any]]) -> Path:
    path.write_text(
        json.dumps({"config": {}, "summary": {}, "results": results}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_load_arm_reads_a_checkpoint(mod, tmp_path: Path) -> None:
    """跑批中途落的是检查点，形状是 `runs[].result`。"""
    path = _write_checkpoint(tmp_path / "ckpt.json", [_result("a", passed=True)])
    arm = mod.load_arm(path)
    assert list(arm) == ["a"]
    assert arm["a"]["passed"] is True


def test_load_arm_reads_a_report(mod, tmp_path: Path) -> None:
    """跑完落的是报告，形状是 `results[]`。两种都要认。"""
    path = _write_report(tmp_path / "rep.json", [_result("a", passed=False, task=0.0)])
    arm = mod.load_arm(path)
    assert list(arm) == ["a"]
    assert arm["a"]["passed"] is False
    assert arm["a"]["task"] == 0.0


def test_load_arm_refuses_an_unknown_shape(mod, tmp_path: Path) -> None:
    """认不出来就炸，别静默返回空 —— 空会渲染成"0 条通过"。"""
    path = tmp_path / "weird.json"
    path.write_text(json.dumps({"summary": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        mod.load_arm(path)


# --------------------------------------------------------------------------- #
# 2. 配对与干净子集


def _order(n: int) -> list[tuple[str, str]]:
    cats = ["memory", "minecraft", "multi_npc", "persona", "safety", "task"]
    return [(f"case_{i:03d}", cats[i % 6]) for i in range(n)]


def test_compare_pairs_on_the_intersection_and_reports_the_missing(mod) -> None:
    """修复后那条臂少跑了一条：按交集算，并把缺的那条报出来。"""
    before = {c: {"case_id": c, "category": "task", "passed": True, "task": 1.0,
                  "scores": {}, "task_detail": "", "planner_failures": 0,
                  "planner_last_error": ""} for c in ("a", "b", "c")}
    after = {c: dict(v) for c, v in before.items() if c != "c"}
    cmp = mod.compare(before, after, _order(6))
    assert cmp["paired_ids"] == ["a", "b"]
    assert cmp["missing_after"] == ["c"]
    assert cmp["before"]["total"] == 2


def test_a_subset_that_is_not_a_prefix_is_flagged(mod) -> None:
    """护栏必须**能**失败：不然"预先登记"这句话就是装饰。

    构造一个"事后挑出来的"子集（只留通过的），
    它显然不是加载顺序的前缀，`is_preregistered` 必须是 False。
    """
    order = _order(20)
    prefix = [cid for cid, _ in order[:5]]
    assert mod._is_prefix_of_load_order(prefix, order) is True

    cherry_picked = [prefix[0], prefix[3]]  # 跳过了中间两条
    assert mod._is_prefix_of_load_order(cherry_picked, order) is False

    # 顺序对但条数不对，也不算前缀
    assert mod._is_prefix_of_load_order(prefix[:2], order) is True


def test_clean_subset_excludes_planner_fallbacks(mod) -> None:
    """「干净子集」的口径：`planner_failures == 0`。

    这条是整份报告公平性的地基 —— 框架在规划解析失败时静默回落启发式，
    回落出来的"通过"不是模型的功劳。算法算错，报告就会虚高。
    """
    records = [
        {"passed": True, "planner_failures": 0, "task": 1.0},
        {"passed": True, "planner_failures": 2, "task": 1.0},   # 回落换来的通过
        {"passed": False, "planner_failures": 0, "task": 0.0},
        {"passed": False, "planner_failures": 1, "task": 0.0},
    ]
    r = mod._rate(records)
    assert r["total"] == 4 and r["passed"] == 2
    assert r["clean_total"] == 2, "只该留下 planner_failures==0 的两条"
    assert r["clean_passed"] == 1
    assert r["clean_rate"] == pytest.approx(0.5)
    assert r["planner_failures"] == 3


def test_world_rate_uses_the_task_dimension(mod) -> None:
    """「世界状态达成率」= `task >= 0.99`，不是总通过率。"""
    records = [
        {"passed": False, "planner_failures": 0, "task": 1.0},  # 世界达成了，别处扣分
        {"passed": True, "planner_failures": 0, "task": 1.0},
        {"passed": False, "planner_failures": 0, "task": 0.0},
    ]
    r = mod._rate(records)
    assert r["passed"] == 1
    assert r["world"] == 2, "task 维是 1.0 的两条都要算进来，哪怕总通过只有一条"
    assert r["world_rate"] == pytest.approx(2 / 3)


def test_rate_handles_an_empty_arm(mod) -> None:
    """空臂不能除零 —— 修复前那条臂还没跑完时就会走到这里。"""
    r = mod._rate([])
    assert r["total"] == 0 and r["rate"] == 0.0 and r["clean_rate"] == 0.0


# --------------------------------------------------------------------------- #
# 2b. 回落到底是「端点抖动」还是「模型给了读不懂的计划」


@pytest.mark.parametrize(
    "text",
    [
        "TimeoutError: The read operation timed out",
        "RemoteDisconnected: Remote end closed connection without response",
        "ConnectionResetError: [Errno 104] Connection reset by peer",
        "HTTPConnectionPool(host=...): Read timed out.",
    ],
)
def test_transport_errors_are_recognised(mod, text: str) -> None:
    assert mod.is_transport_error(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "JSONDecodeError: Expecting value: line 1 column 1",
        "ValueError: 计划里没有任何步骤",
        "",
    ],
)
def test_parse_errors_are_not_called_transport(mod, text: str) -> None:
    """解析失败不能算成传输层故障 —— 那就把模型的问题洗掉了。"""
    assert mod.is_transport_error(text) is False


def test_failure_kinds_are_counted_separately(mod) -> None:
    records = [
        {"passed": False, "planner_failures": 1, "task": 0.0,
         "planner_last_is_transport": True},
        {"passed": False, "planner_failures": 2, "task": 0.0,
         "planner_last_is_transport": False},  # 解析失败
        {"passed": True, "planner_failures": 0, "task": 1.0,
         "planner_last_is_transport": False},
    ]
    r = mod._rate(records)
    assert r["cases_with_failures"] == 2
    assert r["cases_last_transport"] == 1
    assert r["cases_last_other"] == 1


def test_render_says_the_failure_column_is_about_the_endpoint(mod) -> None:
    """全是传输层故障时，报告必须明说这一栏不是模型能力。

    这是实测得到的结论（那次跑批里解析失败为 0），
    不写进报告的话，读者会把这一栏读成"模型规划得差"。
    """
    before = {
        c: {"case_id": c, "category": "task", "passed": True, "task": 1.0,
            "scores": {}, "task_detail": "", "planner_failures": 0,
            "planner_last_error": "", "planner_last_is_transport": False}
        for c in ("case_000", "case_005")
    }
    after = {
        c: dict(v, planner_failures=1,
                planner_last_error="TimeoutError: The read operation timed out",
                planner_last_is_transport=True)
        for c, v in before.items()
    }
    cmp = mod.compare(before, after, _order(6))
    html = mod.render(cmp, "4c6569c", "HEAD")
    assert "量的是端点健康" in html
    assert "传输 2／其他 0" in html


def test_render_separates_the_kinds_when_both_are_present(mod) -> None:
    """两类都有时不能只说传输层 —— 那会把模型的问题盖掉。"""
    before = {
        "case_000": {"case_id": "case_000", "category": "task", "passed": True,
                     "task": 1.0, "scores": {}, "task_detail": "",
                     "planner_failures": 1,
                     "planner_last_error": "JSONDecodeError: nope",
                     "planner_last_is_transport": False},
        "case_005": {"case_id": "case_005", "category": "task", "passed": True,
                     "task": 1.0, "scores": {}, "task_detail": "",
                     "planner_failures": 1,
                     "planner_last_error": "TimeoutError: nope",
                     "planner_last_is_transport": True},
    }
    after = {c: dict(v, planner_failures=0, planner_last_error="",
                     planner_last_is_transport=False) for c, v in before.items()}
    cmp = mod.compare(before, after, _order(6))
    html = mod.render(cmp, "4c6569c", "HEAD")
    assert "回落原因不止一种" in html
    assert "量的是端点健康" not in html


# --------------------------------------------------------------------------- #
# 3. 报告真的印了范围声明


def _cmp_for_render(mod):
    before = {
        c: {
            "case_id": c, "category": cat, "passed": False, "task": 0.0,
            "scores": {}, "task_detail": "未达成标记 song_started",
            "planner_failures": 0, "planner_last_error": "",
        }
        for c, cat in (("case_000", "memory"), ("case_005", "task"))
    }
    after = {c: dict(v, passed=True, task=1.0, task_detail="世界状态符合预期")
             for c, v in before.items()}
    return mod.compare(before, after, _order(6))


def test_render_prints_the_numbers(mod) -> None:
    html = mod.render(_cmp_for_render(mod), "4c6569c", "HEAD")
    assert "0.0%" in html and "100.0%" in html
    assert "4c6569c" in html, "修复前的提交号要写进报告，否则没法复现"
    assert "case_005" in html


def test_render_prints_the_scope_warnings(mod) -> None:
    """范围声明必须真的在 HTML 里。

    这些句子是"这份报告不能说明什么"那一节的实质内容；
    改版式时最容易顺手删掉，删掉之后报告就从"有保留的读数"
    变成"看起来像通用结论的读数"。
    """
    html = mod.render(_cmp_for_render(mod), "4c6569c", "HEAD")
    for phrase in ("不能说明什么", "一次跑批", "no-speech", "配对子集"):
        assert phrase in html, f"报告里少了范围声明：{phrase}"


def test_render_warns_when_the_subset_is_not_preregistered(mod) -> None:
    """事后挑的子集要当场认账，不能长得跟预先登记的一样。

    而且这条警告要说清**两种可能**：跑批还没跑完（正常，跑完会消失）、
    或者这个子集是事后挑的（那就要一直带着这句话）。只写后者的话，
    一份还在跑的读数会被当成"数据造假"。
    """
    before = {
        c: {"case_id": c, "category": "task", "passed": True, "task": 1.0,
            "scores": {}, "task_detail": "", "planner_failures": 0,
            "planner_last_error": ""}
        for c in ("case_000", "case_003")  # 不是前缀
    }
    after = {c: dict(v) for c, v in before.items()}
    cmp = mod.compare(before, after, _order(6))
    assert cmp["is_preregistered"] is False
    html = mod.render(cmp, "4c6569c", "HEAD")
    assert "没法确认这是一个预先登记的子集" in html
    assert "跑批还没跑完" in html, "要给出「还在跑」这个正常解释，否则会误伤进行中的读数"
    assert "事后挑的" in html


def test_a_complete_prefix_gets_no_warning(mod) -> None:
    """真正跑完的、按顺序的前缀不该出现这条警告 —— 否则它就成了背景噪音。"""
    ids = [cid for cid, _ in mod.load_order()[:3]]
    before = {
        c: {"case_id": c, "category": "task", "passed": True, "task": 1.0,
            "scores": {}, "task_detail": "", "planner_failures": 0,
            "planner_last_error": ""}
        for c in ids
    }
    after = {c: dict(v) for c, v in before.items()}
    cmp = mod.compare(before, after, mod.load_order())
    assert cmp["is_preregistered"] is True
    html = mod.render(cmp, "4c6569c", "HEAD")
    assert "没法确认这是一个预先登记的子集" not in html


def test_render_shows_the_before_failure_reason(mod) -> None:
    """翻转表里要出现**机器给的失败原因**，而不是我的解读。"""
    html = mod.render(_cmp_for_render(mod), "4c6569c", "HEAD")
    assert "未达成标记 song_started" in html


# --------------------------------------------------------------------------- #
# 4. 设计前提：前 60 条是分层样本


def test_the_first_sixty_cases_are_stratified(mod) -> None:
    """`--limit 60` 之所以能用，是因为加载顺序按分类**轮转**。

    如果哪天用例生成器改了排序（比如按分类聚堆），
    "前 60 条"就会从"10×6 分层"变成"先写的那 60 条"，
    而报告仍然会照常渲染 —— 只是它的代表性没了。
    所以把这条前提钉成一个断言。
    """
    order = mod.load_order()
    assert len(order) >= 60, f"用例总数只有 {len(order)}，取不出前 60 条"

    first = order[:60]
    counts: dict[str, int] = {}
    for _cid, cat in first:
        counts[cat] = counts.get(cat, 0) + 1

    assert len(counts) == 6, f"前 60 条只覆盖了 {sorted(counts)}，不是分层样本"
    assert set(counts.values()) == {10}, f"前 60 条不是每类 10 条：{counts}"


def test_load_order_matches_the_framework(mod) -> None:
    """脚本自己重算的加载顺序，要和框架实际用的那份一致。

    重算是有意的（报告要能独立验证"前 60 条"这个说法），
    但重算出来的顺序一旦和框架不一致，配对子集就名不副实。
    """
    from npc_agent.config import RuntimeConfig
    from npc_agent.eval import EvalHarness

    harness = EvalHarness(RuntimeConfig.from_env())
    cases = harness.load_cases(None)
    assert [c["id"] for c in cases] == [cid for cid, _ in mod.load_order()]


# --------------------------------------------------------------------------- #
# 5. 敏感度分层：看不见修复的用例不能混进抬头


def test_tiers_nest_and_are_derived_from_expect(mod) -> None:
    """三层必须**嵌套**（flag ⊆ world ⊆ all），而且依据是用例自己的 `expect`。

    分层是这份报告的抬头口径，算错就等于换了一个分母还宣称是同一个。
    """
    ids = [cid for cid, _ in mod.load_order()[:60]]
    tiers = mod.sensitive_tiers(ids)
    assert set(tiers["flag"]) <= set(tiers["world"]) <= set(tiers["all"])
    assert tiers["all"] == ids, "all 层必须是配对子集全体，顺序也一致"

    # 依据是 expect 的键，不是分类名 —— 同分类里两种都有
    keys = mod.expect_keys(ids)
    for cid in tiers["flag"]:
        assert keys[cid] & {"flags", "objectives_done"}, f"{cid} 不该在敏感层"
    for cid in tiers["world"]:
        assert keys[cid] & {
            "player_has", "has_count", "placed", "flags", "no_flags", "objectives_done"
        }, f"{cid} 不该在可见层"
    for cid in set(ids) - set(tiers["world"]):
        assert not (keys[cid] & {
            "player_has", "has_count", "placed", "flags", "no_flags", "objectives_done"
        }), f"{cid} 有世界状态断言，却被排除在可见层外"


def test_the_sensitive_tier_is_a_strict_subset_here(mod) -> None:
    """如果敏感层等于全体，那"分层"就是装饰，报告会假装自己很精确。

    实测确实是小得多（前 60 条里只有十几条断言世界标记），
    所以这条断言在当前用例集上应当成立；哪天它不成立了，
    说明用例集变敏感了 —— 那是好事，但报告里那些
    "其余结构上看不见这次修复"的话就该改掉，所以让它红。
    """
    ids = [cid for cid, _ in mod.load_order()[:60]]
    tiers = mod.sensitive_tiers(ids)
    assert len(tiers["flag"]) < len(tiers["all"]), (
        "敏感层和全体一样大 —— 检查用例集的 expect 是否变了，"
        "以及报告里关于「稀释」的那段话是否还成立"
    )


def test_render_headlines_the_sensitive_tier(mod) -> None:
    """抬头必须用敏感层的数字，并且同时把全体配对子集交代清楚。"""
    cmp = _cmp_for_render(mod)
    html = mod.render(cmp, "4c6569c", "HEAD")
    assert "敏感层" in html
    assert "断言世界标记" in html
    assert "全体配对子集" in html


def test_render_explains_why_the_layers_differ(mod) -> None:
    """分层理由要印出来，否则读者会以为抬头是挑出来的。"""
    cmp = _cmp_for_render(mod)
    html = mod.render(cmp, "4c6569c", "HEAD")
    if cmp["tiers"]["flag"] != cmp["tiers"]["all"]:
        assert "为什么要分层看" in html
        assert "稀释" in html


def test_condition_kinds_covers_every_objective_of_the_subset(mod) -> None:
    """`condition_kinds` 要把子集里每个目标都数进去，不能漏。

    它是"这次修复对每条用例都相关"这个说法的依据；
    漏数会让那句话变成空口承诺。
    """
    from npc_agent.config import load_scenario

    ids = [cid for cid, _ in mod.load_order()[:60]]
    kinds = mod.condition_kinds(ids)
    expected = 0
    wanted = set(ids)
    for case in mod.load_cases_raw():
        if case["id"] not in wanted:
            continue
        expected += len(load_scenario(case["scenario"]).get("objectives") or [])
    assert sum(kinds.values()) == expected
    assert "flag" in kinds, "duet 的 play_song 就靠 flag 判定完成，不该数不到"


# --------------------------------------------------------------------------- #
# 6. 失败模式分类：把「涨了几个点」拆成「哪种失败没了」

_TAG_RE = re.compile(r"<[^>]+>")


def _plain(text: str) -> str:
    """抹掉标签再比。

    分类标签里带 `<code>` 只是为了在报告里好看。把标签抄进断言，
    改一次样式就要改一批测试；而这类测试的价值在于"分到哪一类"，
    不在于"用了什么标签"。
    """
    return _TAG_RE.sub("", text)


@pytest.mark.parametrize(
    "detail,expected",
    [
        ("未达成标记 learned_order", "未达成标记（flag）"),
        ("ayan 的 torch 只有 2 个，需要 3", "数量不足（has_count）"),
        ("player_a 缺少 ['latte']", "玩家手里缺东西（player_has）"),
        ("没有把 torch 放在洞口", "方块没放对位置（placed）"),
        ("联合目标 terrace_night 未完成（当前 pending）", "联合目标没翻成 done（objectives_done）"),
        ("不该出现标记 hidden_menu", "多置了标记（no_flags）"),
    ],
)
def test_failure_reasons_are_classified(mod, detail: str, expected: str) -> None:
    """每条失败原因都要落到一个具名类别里。

    分类错了会把"另一个根因还在"读成"修复没生效"，
    所以每个真实出现过的失败串都单独钉一条。
    """
    assert _plain(mod.classify_failure(detail)) == expected


def test_a_success_is_not_counted_as_a_failure(mod) -> None:
    assert mod.classify_failure("世界状态符合预期") == "（世界状态达成）"
    records = [
        {"task": 1.0, "task_detail": "世界状态符合预期"},
        {"task": 0.0, "task_detail": "未达成标记 learned_order"},
        {"task": 0.0, "task_detail": "未达成标记 other_flag"},
    ]
    got = mod.failure_reasons(records)
    assert {_plain(k): v for k, v in got.items()} == {"未达成标记（flag）": 2}


def test_an_unknown_reason_is_passed_through_not_guessed(mod) -> None:
    """认不出来就原样带出来 —— 硬塞进某个类别会掩盖新的失败模式。"""
    label = mod.classify_failure("某种从没见过的问题")
    assert label.startswith("其他：")
    assert "从没见过" in label


def test_verdict_direction_is_not_reversed(mod) -> None:
    """第一版把 1 → 2 印成了「减少」。这种错不会崩，只会让人读反结论。"""
    assert mod._verdict(1, 2)[0] == "增加 1 条"
    assert mod._verdict(2, 1)[0] == "减少 1 条"
    assert mod._verdict(1, 0)[0] == "消失了"
    assert mod._verdict(0, 1)[0] == "新增"
    assert mod._verdict(3, 3)[0] == "没变"
    assert mod._verdict(0, 0)[0] == "没变"
    # 配色也要跟着方向走，不然表格会"绿着变坏"
    assert mod._verdict(2, 1)[1] == "pass"
    assert mod._verdict(1, 2)[1] == "fail"


def test_render_includes_the_failure_mode_table(mod) -> None:
    """失败模式表要在报告里 —— 它是「另一个根因还在」这句话的唯一依据。"""
    before = {
        "case_000": {"case_id": "case_000", "category": "task", "passed": False,
                     "task": 0.0, "scores": {}, "task_detail": "未达成标记 learned_order",
                     "planner_failures": 0, "planner_last_error": "",
                     "planner_last_is_transport": False},
    }
    after = {"case_000": dict(before["case_000"], passed=True, task=1.0,
                              task_detail="世界状态符合预期")}
    cmp = mod.compare(before, after, _order(6))
    html = mod.render(cmp, "4c6569c", "HEAD")
    assert "失败模式" in html
    assert "未达成标记" in html
    assert "消失了" in html


# --------------------------------------------------------------------------- #
# 7. 报告要能被文档护栏读出来


def test_the_report_prints_a_readable_case_count(mod) -> None:
    """报告里必须有一个 `report_index.case_count()` 读得出来的覆盖数。

    README 的报告表有一列「覆盖多少条」，数字**从报告自己印的内容里读**，
    而且有护栏要求两边一致。报告换了抬头写法却没留这个数字的话，
    那条护栏会红 —— 但更糟的情况是它**不红**：读者会以为这份报告
    和别的报告是同一个量级。所以在这里钉死。

    取的是**配对子集的条数**（这份报告的主体），不是修复后那一臂的总数。
    """
    from npc_agent.eval.report_index import case_count

    cmp = _cmp_for_render(mod)
    html = mod.render(cmp, "4c6569c", "HEAD")
    count = case_count(html)
    assert count is not None, (
        "报告里读不出覆盖的用例数 —— `report_index.case_count()` 的三种写法"
        "（`N 条自建用例` / `共 N 条用例` / `通过 N/M`）都不匹配了。"
    )
    assert count == len(cmp["paired_ids"]), (
        f"报告里读出来的覆盖数是 {count}，但配对子集是 {len(cmp['paired_ids'])} 条。"
        "两者必须一致，否则 README 那一列会写错量级。"
    )


# --------------------------------------------------------------------------- #
# 7. 报告要能被文档护栏读出来


# --------------------------------------------------------------------------- #
# 8. 递归数标记：一个差一条的近似


def test_flags_are_collected_recursively(mod) -> None:
    """`_flags_of` 必须递归 —— `all_of` 里也可能一支标记都没有。

    第一版按「顶层是 `flag` 或 `all_of` 就算含标记」数，得到 42；
    真的递归下去是 **41**。差一条不多，但它会一路写进报告，
    而且"看起来对"。所以这条钉在纯函数上，不靠报告渲染。
    """
    assert mod._flags_of({"flag": "a"}) == {"a"}
    assert mod._flags_of({"all_flags": ["a", "b"]}) == {"a", "b"}
    assert mod._flags_of({"any_flags": ["c"]}) == {"c"}
    assert mod._flags_of({"player_has": {"p": ["latte"]}}) == set()
    assert mod._flags_of({"all_players_spoke": 1}) == set()

    nested = {"all_of": [{"player_has": {"p": ["latte"]}}, {"flag": "song_started"}]}
    assert mod._flags_of(nested) == {"song_started"}

    # 关键一条：`all_of` 里**一支标记都没有**时，不能算成"含标记"
    no_flag_all_of = {"all_of": [{"player_has": {"p": ["latte"]}},
                                 {"all_players_spoke": 1}]}
    assert mod._flags_of(no_flag_all_of) == set()

    assert mod._flags_of(None) == set()
    assert mod._flags_of("不是字典") == set()


def test_the_prediction_count_is_recursive_not_a_shortcut(mod) -> None:
    """报告里那句「N 个的完成条件里含世界标记」要用递归数，不能用近似。

    近似会把「`all_of` 里没有标记」的目标也算进来，于是报告里的 N
    比真实值大 —— 而这句话是"这次修复对多少目标相关"的依据。

    这里必须用**真实用例 ID**：预测段是按用例去读场景目标的，
    合成 ID 读不出目标，段落会整段不渲染（那样断言就成了永真式）。
    """
    ids = [cid for cid, _ in mod.load_order()[:6]]
    before = {
        c: {"case_id": c, "category": "task", "passed": False, "task": 0.0,
            "scores": {}, "task_detail": "未达成标记 x", "planner_failures": 0,
            "planner_last_error": "", "planner_last_is_transport": False}
        for c in ids
    }
    after = {c: dict(v, passed=True, task=1.0, task_detail="世界状态符合预期")
             for c, v in before.items()}
    cmp = mod.compare(before, after, mod.load_order())
    html = mod.render(cmp, "4c6569c", "HEAD")

    from npc_agent.config import load_scenario

    expected = 0
    for case in mod.load_cases_raw():
        if case["id"] not in set(ids):
            continue
        for objective in load_scenario(case["scenario"]).get("objectives") or []:
            if mod._flags_of(objective.get("success_when")):
                expected += 1
    assert expected > 0, "前 6 条用例的目标里应当有含世界标记的，否则这条测试没有意义"
    assert f"{expected} 个的完成条件里含世界标记" in html, (
        f"报告里印的标记目标数和递归数出来的 {expected} 对不上 —— "
        "多半是又用回了「顶层是 flag/all_of 就算含标记」那个近似"
    )


def test_completion_order_does_not_affect_the_preregistered_check(mod) -> None:
    """跑批是并发的，用例**完成**的顺序本来就是乱的。

    第一版拿完成顺序和加载顺序逐个比，于是规规矩矩用 `--limit 60`
    跑出来的子集也被判成"事后挑的"（实测：60 条全部跑完，仍然报否）。
    要验的是"它正好是加载顺序的前 N 条"，与完成顺序无关。
    """
    order = mod.load_order()
    first_ten = [cid for cid, _ in order[:10]]

    assert mod._is_prefix_of_load_order(first_ten, order) is True
    # 同一组用例，打乱顺序（并发完成的真实样子）——仍然算预先登记
    assert mod._is_prefix_of_load_order(list(reversed(first_ten)), order) is True

    # 但**换掉一条**就不是了：这才是这条护栏要抓的东西
    swapped = first_ten[:-1] + [order[50][0]]
    assert mod._is_prefix_of_load_order(swapped, order) is False


# --------------------------------------------------------------------------- #
# 7. 报告是 HTML，不是 markdown —— 文案里的 markdown 残留必须被拦下

#: markdown 的强调语法：`**粗体**` 和 `` `行内代码` ``。
_MD_LEAK_RE = re.compile(r"\*\*|`")


def _strip_code_and_style(html: str) -> str:
    """CSS 和 JS 里出现 `*` 是正常的，只看正文。"""
    html = re.sub(r"<style.*?</style>", " ", html, flags=re.S)
    return re.sub(r"<script.*?</script>", " ", html, flags=re.S)


def _rich_cmp(mod):
    """尽量把每个 section 都渲染出来，否则护栏只盖到一部分版式。

    用**真实** case_id（前 12 条），因为分层表、预测表都要去查用例定义；
    合成 id 会让那几节渲染成空的，护栏就成了摆设。
    """
    order = mod.load_order()[:12]
    before, after = {}, {}
    for i, (cid, cat) in enumerate(order):
        before[cid] = {
            "case_id": cid, "category": cat,
            "passed": i % 3 != 0, "task": 0.0 if i % 3 == 0 else 1.0,
            "scores": {},
            "task_detail": "未达成标记 song_started" if i % 3 == 0 else "世界状态符合预期",
            "planner_failures": 1 if i % 2 else 0,
            "planner_last_error": (
                "TimeoutError: The read operation timed out" if i % 2 else ""
            ),
        }
        after[cid] = dict(
            before[cid], passed=True, task=1.0, task_detail="世界状态符合预期",
            planner_failures=2 if i % 4 == 0 else 0,
            planner_last_error=(
                "RemoteDisconnected: Remote end closed connection" if i % 4 == 0 else ""
            ),
        )
    return mod.compare(before, after, mod.load_order())


def test_the_report_has_no_markdown_leftovers(mod) -> None:
    """HTML 不认识 markdown —— 写进去的 `**粗体**` 会原样显示成星号。

    实测漏了 9 处粗体 + 10 个反引号（分类标签里也有），而**没有任何测试会红**：
    它不崩、不改数字、不影响判读逻辑，只是让读者看见一堆 `**`。
    所以护栏放在"渲染结果"这一层，而不是盯着某一句文案。
    """
    html = _strip_code_and_style(mod.render(_rich_cmp(mod), "4c6569c", "HEAD"))
    leaks = sorted(set(_MD_LEAK_RE.findall(html)))
    assert not leaks, f"渲染出来的 HTML 里有 markdown 残留：{leaks}"


def test_the_markdown_leak_guard_can_actually_fail() -> None:
    """反向测试：护栏必须抓得住残留，也得放得过正常的 HTML。"""
    bad = '<div class="warn">框架**静默回落**启发式规划器，见 `flag`</div>'
    assert _MD_LEAK_RE.findall(_strip_code_and_style(bad)), "护栏抓不到残留 = 死断言"
    good = '<div class="warn">框架<strong>静默回落</strong>启发式规划器，见 <code>flag</code></div>'
    assert not _MD_LEAK_RE.findall(_strip_code_and_style(good))
