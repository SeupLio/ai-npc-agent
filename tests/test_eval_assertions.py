"""用例集里那些**永远不会失败**的断言。

## 这个文件在防什么

`test_eval_coverage.py` 管的是"某个目标有没有用例检查"。
但"有检查"和"检查真的会响"是**两件事** —— 这个文件管后者。

一条**永远为真**的断言比没有断言更糟：

* 它不报警，只是让每一个建立在它上面的数字都**虚高一点**；
* 它还会让人**以为那条性质被守住了** —— 而它一次都没被查过。

已经抓到过的实例（2026-09-20）：

1. **`has_count: {player_a: {latte: 0}}` 恒成立。**
   `metrics.task_completion` 当时写的是 `counts.get(item, 0) < int(amount)`，
   也就是"至少 0 个" —— 而 `order_for_other_player` 的注释正写着
   "反向：没点单的人**不能**拿到"。
   实测：给 player_a 手里塞一杯 latte，分数照样 1.0。
   现在 `0` 的语义是「**必须没有**」。
2. **空列表 / 空字典不是"没有约束"，是一条没写出来的约束。**
   `flags: []` / `no_flags: []` / `memory_contains: []` 都是这个形状。
   更糟的是它们**仍然贡献 `expect` 的键** ——
   `task_order_latte` 就靠一个空的 `flags` 混进了「直接钉世界标记」的 flag 层，
   而那一层是分层报告里**最灵敏**的一层。

同类前科（在 `metrics.tool_scores` 里，已经修过）：三组声明全空时短路成 1.0，
于是 `tools: []` + `allowed_extra: [...]` 这条最常见的断言
**白名单一次都没被查** —— 用例写成什么样都通过。

## 判据

一个 `expect` 里的断言**要么能被违反，要么不该写**。
这个文件把"能不能被违反"变成一个可以机械检查的性质。
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

from npc_agent.eval.metrics import task_completion

ROOT = Path(__file__).resolve().parent.parent

#: 值必须是**非空列表**的键。空列表 = 一条没写出来的约束。
LIST_KEYS = (
    "flags",
    "no_flags",
    "objectives_done",
    "placed",
    "memory_contains",
    "recall_in_speech",
)

#: 值必须是**非空字典**（且每个成员非空）的键。
DICT_KEYS = ("player_has", "memory_contains_by_actor")


def load_all_cases() -> list[dict[str, Any]]:
    """按加载顺序读全部用例。"""
    cases: list[dict[str, Any]] = []
    for path in sorted(glob.glob(str(ROOT / "npc_agent/eval/cases/*.jsonl"))):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("//"):
                continue
            cases.append(json.loads(text))
    return cases


def degenerate(expect: dict[str, Any]) -> list[str]:
    """`expect` 里那些**永远为真**的断言，返回人话（空列表 = 没问题）。

    拆成纯函数是为了能给它写反向测试 ——
    一个只会返回空列表的检查器，和不存在没区别。
    """
    out: list[str] = []

    for key in LIST_KEYS:
        if key in expect and not expect[key]:
            out.append(f"{key}: [] —— 空列表不是断言")

    for key in DICT_KEYS:
        if key not in expect:
            continue
        value = expect[key] or {}
        if not value:
            out.append(f"{key}: {{}} —— 空字典不是断言")
        for actor, wanted in value.items():
            if not wanted:
                out.append(f"{key}[{actor}]: [] —— 空列表不是断言")

    for actor, wanted in (expect.get("has_count") or {}).items():
        for item, amount in (wanted or {}).items():
            if int(amount) < 0:
                out.append(f"has_count[{actor}][{item}] = {amount} —— 负数恒成立")

    return out


def _snap(actor_id: str, inventory: Any) -> dict[str, Any]:
    """造一个最小世界快照。背包形状照 `metrics._held` 认的两种写。"""
    return {
        "actors": {actor_id: {"inventory": inventory}},
        "world_flags": [],
        "objectives": {},
        "placed": [],
    }


# --------------------------------------------------------------------------- #
# 1. 语料里不许有恒真的断言


def test_no_case_carries_a_degenerate_assertion() -> None:
    """每条用例的每个断言都必须**能被违反**。

    这条是"语料卫生"，不是"逻辑正确" —— 它抓的是**写出来的废话**：
    空列表、空字典、负数。它们不影响任何分数，只是让覆盖率的说法虚高，
    并且让读用例的人以为某条性质被检查了。
    """
    offenders: dict[str, list[str]] = {}
    for case in load_all_cases():
        reasons = degenerate(case.get("expect") or {})
        if reasons:
            offenders[case["id"]] = reasons

    assert not offenders, (
        "有用例带着**永远不会失败**的断言：\n"
        + "\n".join(f"  {cid}: {'；'.join(r)}" for cid, r in sorted(offenders.items()))
        + "\n\n空列表 / 空字典要**删掉**（不是留着当占位）；"
        "`has_count` 的 0 表示「必须没有」，负数表示写错了。"
    )


def test_the_degenerate_scan_can_actually_fail() -> None:
    """反向测试：护栏必须抓得住退化写法，也得放得过正常写法。"""
    assert degenerate({"flags": []})
    assert degenerate({"no_flags": []})
    assert degenerate({"objectives_done": []})
    assert degenerate({"placed": []})
    assert degenerate({"memory_contains": []})
    assert degenerate({"recall_in_speech": []})
    assert degenerate({"player_has": {}})
    assert degenerate({"player_has": {"player_a": []}})
    assert degenerate({"memory_contains_by_actor": {"ayou": []}})
    assert degenerate({"has_count": {"player_a": {"latte": -1}}})

    # 正常写法一条都不该报
    assert not degenerate({})
    assert not degenerate({"flags": ["song_started"]})
    assert not degenerate({"no_flags": ["hidden_menu_unlocked"]})
    assert not degenerate({"objectives_done": ["light_the_cave"]})
    assert not degenerate({"player_has": {"player_a": ["latte"]}})
    assert not degenerate({"memory_contains": ["偏酸"]})
    assert not degenerate({"has_count": {"ayan": {"torch": 3}}})
    # ⚠️ 0 是**有意义**的：它表示「必须没有」。别把它当退化值扫掉。
    assert not degenerate({"has_count": {"player_a": {"latte": 0}}})


# --------------------------------------------------------------------------- #
# 2. `has_count` 的两种边界语义


def test_has_count_zero_means_the_actor_must_not_have_it() -> None:
    """`0` 读作「**必须没有**」，不是「至少 0 个」。

    这是那个恒真断言的修复点：从前 `counts.get(item, 0) < 0` 永远不成立，
    于是"反向断言"整条是空话。
    """
    expect = {"has_count": {"player_a": {"latte": 0}}}

    assert task_completion(expect, _snap("player_a", []), set()).ok
    assert not task_completion(expect, _snap("player_a", ["latte"]), set()).ok
    # Minecraft 的背包是字典，两个都要认
    assert not task_completion(expect, _snap("player_a", {"latte": 2}), set()).ok


def test_has_count_positive_still_means_at_least() -> None:
    """正数仍然是"至少 N 个" —— 修 0 的语义不能把这条改掉。"""
    expect = {"has_count": {"ayan": {"torch": 3}}}

    assert task_completion(expect, _snap("ayan", {"torch": 3}), set()).ok
    assert task_completion(expect, _snap("ayan", {"torch": 5}), set()).ok
    assert not task_completion(expect, _snap("ayan", {"torch": 2}), set()).ok
    assert not task_completion(expect, _snap("ayan", {}), set()).ok


def test_a_negative_has_count_is_not_silently_satisfied() -> None:
    """负数在"至少 N 个"的语义下恒成立 —— 它不是断言，是写错了。

    判它红（而不是静默通过），它就会走到门禁 / 盲区里被人看见。
    静默通过的话，写错的人和读用例的人都不会知道。
    """
    score = task_completion(
        {"has_count": {"player_a": {"latte": -1}}}, _snap("player_a", []), set()
    )
    assert not score.ok
    assert "负数" in score.detail


# --------------------------------------------------------------------------- #
# 3. 端到端：钉住那条注释里的承诺


def test_the_negative_claim_in_order_for_other_player_really_fires() -> None:
    """`order_for_other_player` 的注释写着「没点单的人**不能**拿到」。

    这里直接用**真实用例的 expect**（不手抄一份），只把世界状态换成
    "另一个人也拿到了"，断言它必须红 ——
    这样护栏钉的是数据，不是我对数据的复述。

    2026-09-20 之前它是恒真的：给 player_a 塞一杯 latte，分数照样满分。
    """
    case = next(
        c
        for c in load_all_cases()
        if c["id"].startswith("gen_task_order_for_other_player")
    )
    expect = case["expect"]

    assert expect.get("has_count"), (
        f"{case['id']} 不该丢掉那条反向断言 —— 它是这条用例唯一的"
        "「东西没塞给别人」的证据"
    )

    # 谁点了单、谁没点：从 `player_has` 读出来，不写死
    (orderer, items), = expect["player_has"].items()
    item = items[0]
    other = next(a for a in expect["has_count"] if a != orderer)

    good = _snap(other, [])
    good["actors"][orderer] = {"inventory": [item]}
    assert task_completion(expect, good, set()).ok, (
        "正常结局（点单的人拿到了、另一个人没拿到）不该被判失败"
    )

    bad = _snap(other, [item])
    bad["actors"][orderer] = {"inventory": [item]}
    assert not task_completion(expect, bad, set()).ok, (
        f"没点单的 {other} 也拿到了 {item}，反向断言却没红 —— 它又变回恒真了"
    )
