"""留出集的测试。

这个文件守的是一件事：**留出集的数字什么时候可以引用，什么时候不可以。**

为什么值得单独一个测试文件：`calibration.jsonl` 那 24 条被用来改过三次
rubric，所以它上面的 kappa 里有一部分是拟合。留出集就是为了补上这个洞 ——
但如果封条可以被随手重置，或者破了之后没人拦，那它就只是又一个
"看起来很严谨"的说法。所以这里测的不是"kappa 算得对不对"
（那是 test_judge.py 的事），而是**封条破了会不会被拦住**。
"""

from __future__ import annotations

import argparse
import dataclasses
import json

import pytest

from npc_agent import cli
from npc_agent.eval import judge as J
from npc_agent.llm import ScriptedLLM


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #
def _items() -> list[dict]:
    return J.load_calibration(J.HOLDOUT_FILE)


def _always_pass_llm() -> ScriptedLLM:
    """一个永远判 1 的裁判 —— 内容无所谓，这里只测封条和记账。"""
    return ScriptedLLM(lambda _prompt: '{"score": 1, "reason": "测试桩"}')


# --------------------------------------------------------------------------- #
# 随包发布的这份留出集本身
# --------------------------------------------------------------------------- #
def test_the_shipped_holdout_set_matches_its_shipped_seal() -> None:
    """**最重要的一条。** 仓库里那两份文件必须是自洽的。

    如果这条挂了，说明留出集被改过而封条没跟着更新（或者反过来）——
    那 README 里所有引用留出集数字的地方都成了空头承诺。
    """
    items = _items()
    assert items, "留出集是空的"
    assert J.verify_seal(J.load_seal(), items) == []


def test_every_rubric_in_the_holdout_has_both_labels() -> None:
    """每个维度都得有正例也有反例。

    全是一个标签的话，期望一致率会退化到 1.0，kappa 只能取 0 或 1 ——
    这时候算出来的数字没有意义（见 test_judge.py 里那个退化用例）。
    """
    per: dict[str, set[int]] = {}
    for item in _items():
        per.setdefault(item["rubric"], set()).add(int(item["label"]))
    assert per, "没有样本"
    for rubric, labels in per.items():
        assert labels == {0, 1}, f"{rubric} 只有 {labels}，算不出有意义的 kappa"


def test_holdout_labels_are_roughly_balanced() -> None:
    """严重偏向一边的集合会让 kappa 看起来很高。

    这里不追求精确平衡，只要求两边都不少于三分之一 —— 这是个下限检查，
    不是质量指标。
    """
    labels = [int(i["label"]) for i in _items()]
    ones = sum(labels) / len(labels)
    assert 1 / 3 <= ones <= 2 / 3, f"标签分布 {ones:.0%} 偏得太厉害"


def test_every_holdout_item_cites_the_clause_that_decides_it() -> None:
    """每条样本都要在 note 里点名是哪条子句决定的。

    这不是形式要求：留出集的价值在于**可被反驳**。如果 note 只写
    "这句读起来不像阿柚"，那么 kappa 低下来的时候没人知道该改 rubric
    还是该改标签。写了子句，争议就落在子句上。
    """
    for item in _items():
        note = item.get("note") or ""
        assert len(note) >= 30, f"{item['id']} 的 note 太短，没说清依据：{note!r}"


def test_holdout_ids_are_unique_and_rubrics_are_known() -> None:
    items = _items()
    ids = [i["id"] for i in items]
    assert len(set(ids)) == len(ids), "有重复 id"
    for item in items:
        assert item["rubric"] in J.RUBRICS, f"{item['id']} 指向未知维度 {item['rubric']}"
        assert item.get("reply"), f"{item['id']} 没有 reply"


def test_the_holdout_is_not_the_dev_set() -> None:
    """两份不能是同一批样本 —— 否则留出集立刻退化成开发集。"""
    dev = {(i.get("persona"), i.get("reply")) for i in J.load_calibration(J.CALIBRATION_FILE)}
    hold = {(i.get("persona"), i.get("reply")) for i in _items()}
    assert not (dev & hold), f"有 {len(dev & hold)} 条重复样本"


# --------------------------------------------------------------------------- #
# 封条：破了就必须被拦住
# --------------------------------------------------------------------------- #
def test_editing_a_label_breaks_the_seal() -> None:
    items = _items()
    seal = J.build_seal(items)
    items[0] = {**items[0], "label": 1 - int(items[0]["label"])}
    problems = J.verify_seal(seal, items)
    assert [p["kind"] for p in problems] == ["holdout_changed"]


def test_editing_a_reply_breaks_the_seal() -> None:
    """改输入和改标签一样严重 —— 换了台词，原本的标签就不再对应它。"""
    items = _items()
    seal = J.build_seal(items)
    items[3] = {**items[3], "reply": items[3]["reply"] + "（被改过）"}
    assert any(p["kind"] == "holdout_changed" for p in J.verify_seal(seal, items))


def test_adding_an_item_breaks_the_seal() -> None:
    items = _items()
    seal = J.build_seal(items)
    items.append({**items[0], "id": "ho_extra"})
    kinds = [p["kind"] for p in J.verify_seal(seal, items)]
    assert "holdout_changed" in kinds
    assert "count_changed" in kinds


def test_editing_a_comment_does_not_break_the_seal(tmp_path) -> None:
    """摘要盯的是「标签和输入有没有变」，不是文件字节。

    修错别字、补注释不该把留出集作废 —— 否则封条会变成"谁都不敢动"的
    东西，而它真正要防的是**悄悄改标签**。
    """
    items = _items()
    before = J.items_digest(items)
    target = tmp_path / "h.jsonl"
    raw = J.HOLDOUT_FILE.read_text(encoding="utf-8")
    target.write_text("// 新加的一行注释\n" + raw + "// 末尾再来一行\n", encoding="utf-8")
    assert J.items_digest(J.load_calibration(target)) == before


def test_changing_the_rubric_text_breaks_the_seal(monkeypatch) -> None:
    """**改 rubric 会让留出集失效**，而且是不可修复的那种。

    标签是照当时的 pass_when / fail_when 写的。标准一变，那些标签就
    不再是"正确答案"了，可它们还躺在文件里，算出来的 kappa 会变成一个
    没人能解释的数。所以 rubric 摘要必须进封条。
    """
    items = _items()
    seal = J.build_seal(items)

    tampered = dict(J.RUBRICS)
    tampered["in_character"] = dataclasses.replace(
        J.RUBRICS["in_character"], pass_when="只要句子不太长就算通过（被改过的标准）"
    )
    monkeypatch.setattr(J, "RUBRICS", tampered)

    problems = J.verify_seal(seal, items)
    assert [p["kind"] for p in problems] == ["rubric_changed"]
    assert "重新封条不能修复" in problems[0]["detail"]


def test_a_missing_seal_is_not_treated_as_valid() -> None:
    """没有封条 ≠ 封条通过。缺了就只能是"不可引用"。"""
    problems = J.verify_seal({}, _items())
    assert [p["kind"] for p in problems] == ["no_seal"]


def test_the_seal_records_the_label_balance_it_was_built_from() -> None:
    seal = J.build_seal(_items())
    assert seal["items"] == len(_items())
    assert seal["label_balance"]["pass"] + seal["label_balance"]["fail"] == seal["items"]
    assert set(seal["per_rubric"]) == set(J.RUBRICS)


# --------------------------------------------------------------------------- #
# 结果：破了也照跑，但必须标成不可引用
# --------------------------------------------------------------------------- #
def test_a_clean_seal_makes_the_result_quotable() -> None:
    items = _items()
    block = J.run_holdout(
        J.LLMJudge(_always_pass_llm()), items=items, seal=J.build_seal(items)
    )
    assert block["quotable"] is True
    assert block["problems"] == []
    assert block["judged"] == len(items)
    assert set(block["per_rubric"]) == set(J.RUBRICS)


def test_a_broken_seal_makes_the_result_unquotable() -> None:
    """**破了也照跑。** 不跑等于把诊断信息一起扔掉。

    关键是数字不能丢的同时也不能被当成泛化能力引用 —— 所以
    `quotable` 和 `problems` 必须一起进 payload。
    """
    items = _items()
    seal = J.build_seal(items)
    items[0] = {**items[0], "label": 1 - int(items[0]["label"])}
    block = J.run_holdout(J.LLMJudge(_always_pass_llm()), items=items, seal=seal)
    assert block["quotable"] is False
    assert any(p["kind"] == "holdout_changed" for p in block["problems"])
    assert block["judged"] > 0, "封条破了也要把判决留下来，否则连诊断都没得做"


def test_a_judge_that_says_nothing_is_recorded_as_unjudged() -> None:
    """留出集也不能把"没判"变成 0 分 —— 铁律一在这里同样成立。"""
    from npc_agent.llm.null import NullLLM

    block = J.run_holdout(
        J.LLMJudge(NullLLM()), items=_items(), seal=J.load_seal()
    )
    assert block["judged"] == 0
    assert block["unjudged"] == block["total"]


# --------------------------------------------------------------------------- #
# 对照：两个 kappa 的差就是拟合的量
# --------------------------------------------------------------------------- #
def _report(kappa: float, n: int = 10) -> J.CalibrationReport:
    rep = J.CalibrationReport(total=n, judged=n)
    for key in J.RUBRICS:
        rep.per_rubric[key] = {"n": n, "agreement": kappa, "kappa": kappa,
                               "reading": "x", "confusion": {}}
    return rep


def test_contrast_rows_report_the_fitting_gap() -> None:
    holdout = {"quotable": True, "per_rubric": {
        k: {"n": 11, "kappa": 0.6, "reading": "基本一致"} for k in J.RUBRICS}}
    rows = J.contrast_rows(_report(0.85), holdout)
    assert len(rows) == len(J.RUBRICS)
    for row in rows:
        assert row["dev_kappa"] == 0.85
        assert row["holdout_kappa"] == 0.6
        assert row["gap"] == pytest.approx(0.25)
        assert row["quotable"] is True


def test_contrast_rows_survive_a_rubric_missing_from_one_side() -> None:
    """只在一边出现的维度不能让渲染崩掉，也不能编出一个 0 来。"""
    holdout = {"quotable": True, "per_rubric": {
        "grounded": {"n": 10, "kappa": 0.9, "reading": "几乎完全一致"}}}
    rows = J.contrast_rows(_report(0.5), holdout)
    by_key = {r["rubric"]: r for r in rows}
    assert by_key["in_character"]["holdout_kappa"] is None
    assert by_key["in_character"]["gap"] is None
    assert by_key["grounded"]["dev_n"] == 10


# --------------------------------------------------------------------------- #
# 命令：重封条要显式 --force
# --------------------------------------------------------------------------- #
def _seal_args(force: bool = False) -> argparse.Namespace:
    return argparse.Namespace(force=force, note="")


def test_seal_command_refuses_to_reseal_a_broken_set(monkeypatch, tmp_path, capsys) -> None:
    """封条破了就**不能**随手重封。

    如果重封是免费的，"标签是改之前写的"这个前提就永远无法自证 ——
    封条也就白封了。所以这里要求显式 --force，并且在拒绝时说清
    正确做法是另攒一份。
    """
    holdout = tmp_path / "h.jsonl"
    holdout.write_text(
        json.dumps({"id": "a", "rubric": "grounded", "persona": "", "scene": "",
                    "player": "", "reply": "原句", "label": 0, "note": "x"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    seal_path = tmp_path / "seal.json"
    monkeypatch.setattr(J, "HOLDOUT_FILE", holdout)
    monkeypatch.setattr(J, "HOLDOUT_SEAL_FILE", seal_path)

    assert cli.cmd_seal_holdout(_seal_args()) == 0
    assert seal_path.exists()

    # 改一条标签 -> 封条破
    holdout.write_text(
        json.dumps({"id": "a", "rubric": "grounded", "persona": "", "scene": "",
                    "player": "", "reply": "原句", "label": 1, "note": "x"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    capsys.readouterr()
    assert cli.cmd_seal_holdout(_seal_args(force=False)) == 2
    out = capsys.readouterr().out
    assert "拒绝重封" in out
    assert "另攒一份新的留出集" in out
    # 拒绝的时候不能偷偷把封条改掉
    assert J.load_seal(seal_path)["holdout_digest"] != J.items_digest(
        J.load_calibration(holdout)
    )


def test_seal_command_is_idempotent_when_nothing_changed(monkeypatch, tmp_path) -> None:
    holdout = tmp_path / "h.jsonl"
    holdout.write_text(
        json.dumps({"id": "a", "rubric": "grounded", "persona": "", "scene": "",
                    "player": "", "reply": "原句", "label": 0, "note": "x"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(J, "HOLDOUT_FILE", holdout)
    monkeypatch.setattr(J, "HOLDOUT_SEAL_FILE", tmp_path / "seal.json")
    assert cli.cmd_seal_holdout(_seal_args()) == 0
    first = J.load_seal(tmp_path / "seal.json")
    assert cli.cmd_seal_holdout(_seal_args()) == 0
    assert J.load_seal(tmp_path / "seal.json") == first


def test_seal_command_errors_when_the_holdout_is_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(J, "HOLDOUT_FILE", tmp_path / "nope.jsonl")
    assert cli.cmd_seal_holdout(_seal_args()) == 2


def test_the_holdout_reports_progress() -> None:
    """留出集是**串行**的 32 次调用（十几分钟），必须打进度。

    不打进度的话日志十几分钟一动不动 —— 从外面看和"卡死了"完全一样。
    本项目在监控那一节反复踩这个坑（拿文件时间戳当存活判据），
    所以这里也钉一条：进度回调必须真的被转发到 `calibrate`。
    """
    seen: list[str] = []
    items = _items()
    J.run_holdout(
        J.LLMJudge(_always_pass_llm()),
        items=items,
        seal=J.build_seal(items),
        progress=seen.append,
    )
    assert len(seen) == len(items), "每条都要有一次进度输出"
    assert "1/32" in seen[0]
    assert f"{len(items)}/{len(items)}" in seen[-1]
