"""用例集对「场景目标」的覆盖检查。

## 这个文件在防什么

场景配置里写着 `objectives`（「给玩家做一杯拿铁并亲手递过去」），
Agent 也真的会去追它。但**用例的 `expect` 才决定评测检查什么** ——
场景声明了目标、用例从不检查，这个目标就处在"Agent 在追、没人验收"的状态。

后果不是"少测了一点"，而是**整整一类缺陷可以长期存活**：
`duet` 的 `play_song` 曾经因为规划 prompt 里没有完成条件而永远做不完，
而当时的用例只断言 `all_npcs_spoke`（谁开口了），根本看不见目标没完成。
那个缺陷后来是靠人工跑 demo 才发现的。

所以这里把"哪个目标没有任何用例检查"变成一个**会被测试拦住的清单**：
覆盖了就从清单里删掉（测试会提醒你删），新出现没覆盖的（测试会红）。

## 判据：什么算「检查了这个目标」

一条用例算覆盖目标 O，当且仅当它的 `expect` 里至少有一项**真的会在
O 完成时变绿**：

* `objectives_done` 里点名了 O；
* `flags` 里含 O 完成所需的那个世界标记；
* `player_has` / `has_count` 覆盖了 O 要求的物品（同一个玩家、同一件东西）；
* `placed` 覆盖了 O 要求的方块。

刻意**不**把 `all_npcs_spoke` 算作覆盖 `all_players_spoke` 类目标 ——
前者问"NPC 有没有轮流开口"，后者问"每个玩家都说过话"，
两件事，混起来就等于假装覆盖了。
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Callable

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: 已知**没有任何用例检查**的目标。空字典是目标状态。
#:
#: 留在这里不是"接受它"，而是让它在报告和 README 里有个明确位置，
#: 并且让"哪天补上了"变成一次**必须改这里**的动作。
KNOWN_UNCOVERED: dict[str, str] = {
    "icebreaker.greet_all": (
        "完成条件是 all_players_spoke，而 expect 里没有对应的键"
        "（all_npcs_spoke 问的是 NPC 之间，不是玩家）"
    ),
    "icebreaker.find_topic": (
        "完成条件是 flag=topic_found，而 icebreaker 的用例一条都没有断言 flags"
    ),
}


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


def _flags_of(condition: Any) -> set[str]:
    """递归取出一个完成条件里所有**要求被置上**的世界标记。"""
    if not isinstance(condition, dict):
        return set()
    out: set[str] = set()
    if "flag" in condition:
        out.add(str(condition["flag"]))
    for key in ("all_flags", "any_flags"):
        out |= {str(f) for f in (condition.get(key) or [])}
    for sub in (condition.get("all_of") or []) + (condition.get("any_of") or []):
        out |= _flags_of(sub)
    return out


def _required_items(condition: Any) -> dict[str, set[str]]:
    """完成条件要求的「玩家 → 物品集合」（`player_has` / `player_has_count` 两种写法）。"""
    if not isinstance(condition, dict):
        return {}
    out: dict[str, set[str]] = {}
    for key in ("player_has", "player_has_count"):
        for pid, items in (condition.get(key) or {}).items():
            names = set(items) if isinstance(items, dict) else set(items or [])
            out.setdefault(pid, set()).update(str(n) for n in names)
    for sub in (condition.get("all_of") or []) + (condition.get("any_of") or []):
        for pid, names in _required_items(sub).items():
            out.setdefault(pid, set()).update(names)
    return out


def _covers(case: dict[str, Any], objective: dict[str, Any]) -> bool:
    """这条用例会不会在目标完成时变绿。"""
    expect = case.get("expect") or {}
    if objective["id"] in set(expect.get("objectives_done") or []):
        return True

    condition = objective.get("success_when")

    wanted_flags = _flags_of(condition)
    if wanted_flags & set(expect.get("flags") or []):
        return True

    asserted_items: dict[str, set[str]] = {}
    for key in ("player_has", "has_count"):
        for pid, items in (expect.get(key) or {}).items():
            names = set(items) if isinstance(items, dict) else set(items or [])
            asserted_items.setdefault(pid, set()).update(str(n) for n in names)
    for pid, names in _required_items(condition).items():
        if names <= asserted_items.get(pid, set()):
            return True

    return False


def uncovered_objectives(
    cases: list[dict[str, Any]],
    load_scenario: Callable[[str], dict[str, Any]],
) -> dict[str, str]:
    """返回 `{"场景.目标": "为什么没覆盖"}`。

    参数化 `load_scenario` 是为了能用**构造出来的**场景测这个函数本身 ——
    不然它只能对着真实用例集跑一次，对不对都不知道。
    """
    scenarios = sorted({c["scenario"] for c in cases})
    gaps: dict[str, str] = {}
    for scenario_id in scenarios:
        objectives = load_scenario(scenario_id).get("objectives") or []
        if not objectives:
            continue
        same_scenario = [c for c in cases if c["scenario"] == scenario_id]
        for objective in objectives:
            if any(_covers(c, objective) for c in same_scenario):
                continue
            gaps[f"{scenario_id}.{objective['id']}"] = (
                f"{len(same_scenario)} 条用例，没有一条的 expect 会在它完成时变绿"
            )
    return gaps


# --------------------------------------------------------------------------- #
# 真实用例集


def test_every_objective_is_either_checked_or_a_listed_gap() -> None:
    """每个目标要么被检查，要么在已知缺口清单里。

    两个方向都会红：
    * 冒出新的没覆盖目标 → 必须补用例，或至少把它记进清单；
    * 清单里的目标被补上了 → 必须把清单项删掉，
      否则清单会越来越长、越来越没人信。
    """
    from npc_agent.config import load_scenario

    gaps = uncovered_objectives(load_all_cases(), load_scenario)

    new = sorted(set(gaps) - set(KNOWN_UNCOVERED))
    assert not new, (
        "这些目标没有任何用例检查，而且不在已知清单里：\n"
        + "\n".join(f"  {k}：{gaps[k]}" for k in new)
        + "\n要么补一条会在它完成时变绿的用例，要么把它写进 KNOWN_UNCOVERED 并说明原因。"
    )

    fixed = sorted(set(KNOWN_UNCOVERED) - set(gaps))
    assert not fixed, (
        "这些目标已经有用例检查了，请把它们从 KNOWN_UNCOVERED 里删掉：\n"
        + "\n".join(f"  {k}" for k in fixed)
    )


def test_the_known_gaps_are_really_uncovered() -> None:
    """清单里的每一条都要**确实**还没被覆盖，别把已覆盖的留在里面当借口。"""
    from npc_agent.config import load_scenario

    gaps = uncovered_objectives(load_all_cases(), load_scenario)
    for key in KNOWN_UNCOVERED:
        assert key in gaps, f"{key} 已经不缺覆盖了，清单该更新"


def test_the_covered_objectives_are_the_ones_we_think() -> None:
    """把「哪些目标有验收」钉住。

    这条不是废话：`village.light_the_cave`、`hosting.host_round`、
    `duet.play_song` 这几个是**规划修复的靶子**，它们必须保持被覆盖 ——
    哪天有人删掉那些断言，规划就重新变成"Agent 在追、没人验收"。
    """
    from npc_agent.config import load_scenario

    gaps = set(uncovered_objectives(load_all_cases(), load_scenario))
    must_be_covered = {
        "village.light_the_cave",
        "hosting.host_round",
        "duet.play_song",
        "duet.terrace_night",
        "duet.serve_guest",
        "tutorial.teach_order",
        "tutorial.welcome_drink",
    }
    lost = sorted(must_be_covered & gaps)
    assert not lost, (
        "这些目标本来有用例验收，现在没有了一条：\n"
        + "\n".join(f"  {k}" for k in lost)
        + "\n规划类缺陷会重新变得不可见。"
    )


# --------------------------------------------------------------------------- #
# 判据本身要能判错


def _fake_scenario(objectives: list[dict[str, Any]]):
    return lambda _sid: {"objectives": objectives}


def test_the_coverage_check_can_actually_fail() -> None:
    """反向测试：构造一个没人检查的目标，它必须被报出来。

    没有这条，`uncovered_objectives` 可能永远返回空字典，
    而上面那些断言会全部变成永真式。
    """
    objectives = [{"id": "do_the_thing", "success_when": {"flag": "thing_done"}}]
    cases = [{"id": "c1", "scenario": "fake", "expect": {"all_npcs_spoke": True}}]
    gaps = uncovered_objectives(cases, _fake_scenario(objectives))
    assert gaps == {"fake.do_the_thing": "1 条用例，没有一条的 expect 会在它完成时变绿"}


@pytest.mark.parametrize(
    "expect",
    [
        {"objectives_done": ["do_the_thing"]},
        {"flags": ["thing_done"]},
    ],
)
def test_flag_and_objective_assertions_count_as_coverage(expect: dict[str, Any]) -> None:
    objectives = [{"id": "do_the_thing", "success_when": {"flag": "thing_done"}}]
    cases = [{"id": "c1", "scenario": "fake", "expect": expect}]
    assert uncovered_objectives(cases, _fake_scenario(objectives)) == {}


def test_item_assertions_count_only_when_the_item_matches() -> None:
    """`player_has` 要玩家和物品都对上，才算覆盖。

    "有玩家拿到了别的东西"不是"这个目标被验收了" —— 放宽这条，
    覆盖检查就会大面积误判为已覆盖。
    """
    objectives = [
        {"id": "serve", "success_when": {"player_has": {"player_a": ["latte"]}}}
    ]
    wrong_item = [{"id": "c1", "scenario": "fake",
                   "expect": {"player_has": {"player_a": ["lemonade"]}}}]
    assert "fake.serve" in uncovered_objectives(wrong_item, _fake_scenario(objectives))

    wrong_player = [{"id": "c1", "scenario": "fake",
                     "expect": {"player_has": {"player_b": ["latte"]}}}]
    assert "fake.serve" in uncovered_objectives(wrong_player, _fake_scenario(objectives))

    right = [{"id": "c1", "scenario": "fake",
              "expect": {"player_has": {"player_a": ["latte"]}}}]
    assert uncovered_objectives(right, _fake_scenario(objectives)) == {}


def test_all_npcs_spoke_does_not_count_as_all_players_spoke() -> None:
    """这两个不是一回事，不能互相顶替。

    `all_npcs_spoke` 问「NPC 有没有轮流开口」，`all_players_spoke` 问
    「每个玩家都说过话」。把前者当覆盖，`icebreaker.greet_all` 就会
    从缺口清单里消失 —— 而那正是它现在真实的处境。
    """
    objectives = [{"id": "greet", "success_when": {"all_players_spoke": 1}}]
    cases = [{"id": "c1", "scenario": "fake", "expect": {"all_npcs_spoke": True}}]
    assert "fake.greet" in uncovered_objectives(cases, _fake_scenario(objectives))


def test_all_of_needs_only_one_branch_asserted_to_count() -> None:
    """`all_of` 的覆盖是"任一支被断言" —— 这是**宽松**口径，故意的。

    严格口径要求每一支都被断言。这里取宽松，因为目标是回答
    "这个目标会不会在某条用例里变绿"，而不是"是否被完整验收"。
    口径写下来，是为了以后有人改成严格口径时知道自己在改什么。
    """
    objectives = [
        {
            "id": "night",
            "success_when": {
                "all_of": [
                    {"player_has": {"player_a": ["latte"]}},
                    {"flag": "song_started"},
                ]
            },
        }
    ]
    only_flag = [{"id": "c1", "scenario": "fake", "expect": {"flags": ["song_started"]}}]
    assert uncovered_objectives(only_flag, _fake_scenario(objectives)) == {}


def test_scenarios_without_objectives_are_skipped() -> None:
    """没有目标的场景不该凭空产生缺口。"""
    cases = [{"id": "c1", "scenario": "fake", "expect": {}}]
    assert uncovered_objectives(cases, _fake_scenario([])) == {}
