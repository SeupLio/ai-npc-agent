"""`memory` 维度的**覆盖率**护栏 —— 防止"取回来"那一半悄悄变瞎。

## 这个文件为什么存在

`memory` 维度由两半组成（见 `eval/metrics.py::memory_recall`）：

| 断言 | 问的问题 | 读什么 |
|---|---|---|
| `memory_contains` | 信息进记忆库了吗（写入 + 巩固没丢） | `agent.memory.store.records` —— **记忆库的原始列表** |
| `recall_in_speech` | 后面真的把它说出来了吗（检索 + 应用） | 台词 |

差别在第二列：`memory_contains` 读的是**库**，不是**取回来的东西**。
所以它对"检索坏掉"是**结构上**看不见的 —— 一个把每句话都存进去、
却一条也取不回来的 Agent，在 `memory_contains` 上照样满分。

## 实测（这就是这个文件要防的事）

`sensitivity` 里有个变异 `retrieval_disabled`：把检索整个关掉。
第一次跑出来，`memory` 维度只掉 **0.007** —— 只有 **3 条**用例变红。

数一下才明白：当时 34 条记忆用例里，**31 条**断言 `memory_contains`（看不见检索），
只有 **3 条**断言 `recall_in_speech`。0.007 ≈ 3 × 0.5 / 228，正好对上。

也就是说：**「NPC 记得我」这件玩家一眼能看出的事，只有 3 条用例真的在测。**

## 修法

不是改指标（两半的划分是对的：`memory_contains` 本来就是在测"巩固不丢"），
而是**补用例**：新增 3 条 needle 只有检索才能得到的用例，把断言 recall 的用例
从 3 条提到 6 条。修完实测 `retrieval_disabled` 的掉分 **−0.007 → −0.013**。

覆盖率仍然不高（37 条记忆用例里 6 条断言 recall），这件事**不藏着** ——
README 和敏感性报告里都写了这个比例。这两条护栏只保证它**不会掉回去**。
"""

from __future__ import annotations

from npc_agent.config import RuntimeConfig, load_scenario
from npc_agent.eval import EvalHarness

# 断言了 `recall_in_speech` 的用例数下限。
#
# 这个数是**实测出来的**，不是拍的：补用例之前是 3（对应 -0.007），
# 补完之后是 6（对应 -0.013）。写成下限而不是等号，是为了让"继续补用例"
# 不会把测试顶红 —— 但**掉回 3 会红**，那正是要防的。
MIN_RECALL_CASES = 6

# 所有场景的玩家名册，用来识别"假 recall 断言"。
_SCENARIO_IDS = ("tutorial", "icebreaker", "hosting", "duet", "village")


def _all_cases() -> list[dict]:
    return EvalHarness(RuntimeConfig()).load_cases()


def _player_names() -> set[str]:
    names: set[str] = set()
    for scenario_id in _SCENARIO_IDS:
        scenario = load_scenario(scenario_id)
        for player in scenario.get("players") or []:
            name = (player or {}).get("name")
            if name:
                names.add(str(name))
    return names


# --------------------------------------------------------------------------- #
# 判据（纯函数，所以下面能给它们写反向测试）
# --------------------------------------------------------------------------- #
def recall_asserting_cases(cases: list[dict]) -> list[dict]:
    return [c for c in cases if (c.get("expect") or {}).get("recall_in_speech")]


def player_name_recall_needles(
    cases: list[dict], names: set[str]
) -> list[tuple[str, str]]:
    """找出拿**玩家名**当 recall needle 的用例。

    玩家名在世界名册里就有，NPC 点名时本来就会用到。实测：把检索关掉之后，
    `name_recall` / `second_player_name` 这些用例里的「阿澈」「小满」
    **照样出现在台词里** —— 来自"要不先互相报个名字？从阿澈开始吧。"
    这种与记忆无关的句子。

    所以这类断言测的是"NPC 会说人名"，不是"NPC 记得你"：
    它会**稳定通过**，而且对检索故障完全不敏感。
    """
    out: list[tuple[str, str]] = []
    for case in cases:
        for needle in (case.get("expect") or {}).get("recall_in_speech") or []:
            if needle in names:
                out.append((str(case.get("id")), str(needle)))
    return out


# --------------------------------------------------------------------------- #
# 护栏
# --------------------------------------------------------------------------- #
def test_the_recall_half_of_memory_has_non_trivial_coverage() -> None:
    """断言"真的说出来了"的用例不能少于 `MIN_RECALL_CASES` 条。

    为什么这条比它看起来重要：`memory_contains` 读的是记忆库的原始列表，
    **结构上看不见检索**。所以一旦断言 `recall_in_speech` 的用例被删掉，
    `memory` 维度就会在"检索完全坏掉"时依然接近满分 —— 而且没有任何东西会报错。
    实测过一次：只有 3 条的时候，把检索整个关掉只掉 0.007。

    一个只会看着 1.000 的维度，和没有这个维度，在报告上长得一模一样。
    """
    cases = _all_cases()
    recall_cases = recall_asserting_cases(cases)
    memory_cases = [c for c in cases if c.get("category") == "memory"]

    assert memory_cases, "memory 类别一条用例都没有？"
    assert len(recall_cases) >= MIN_RECALL_CASES, (
        f"只有 {len(recall_cases)} 条用例断言了 `recall_in_speech`（下限 "
        f"{MIN_RECALL_CASES}，{len(memory_cases)} 条记忆用例）。"
        "`memory_contains` 读的是记忆库原始列表，看不见检索 —— "
        "少了这些用例，把检索整个关掉也不会掉分。"
        "补用例时 needle 要选**只有检索才能得到**的信息（偏好 / 约定 / 习惯），"
        "不要用玩家名（见下一条测试）。"
    )


def test_recall_needles_are_not_player_names() -> None:
    r"""`recall_in_speech` 的 needle 不能是**玩家名** —— 那是假断言。

    要测检索，needle 必须**只有记忆里才有**。玩家名不满足这个条件。
    这条护栏就是把那次实测结论固化下来。
    """
    names = _player_names()
    assert names, "没读到任何玩家名 —— 场景配置的结构变了，这条护栏正在空转"

    offenders = player_name_recall_needles(_all_cases(), names)
    assert not offenders, (
        f"这些用例拿**玩家名**当 recall 断言：{offenders}。"
        "玩家名能从世界名册拿到，关掉检索也照样出现在台词里 —— "
        "断言它会稳定通过，却测不到任何记忆能力。换一个只有记忆里才有的 needle。"
    )


def test_the_player_name_claim_is_actually_true() -> None:
    """上面那条护栏的**前提**必须是真的：玩家名确实能从世界名册拿到。

    否则这条护栏只是在防一个不存在的问题。这里证明：名册里有玩家名，
    而用例集里也真的存在只用玩家名做**写入**断言的用例
    （`memory_contains: ["阿澈"]` —— 那类断言是合法的，它测的是进库，
    不是检索），只是没有一条拿它做 recall 断言。
    """
    roster = _player_names()
    assert "阿澈" in roster, f"名册里应该有「阿澈」，实际 {roster}"

    name_only_memory = [
        c for c in _all_cases()
        if (c.get("expect") or {}).get("memory_contains") == ["阿澈"]
    ]
    assert name_only_memory, (
        "用例集里应该还有「只断言玩家名进库」的用例（那类断言合法）—— "
        "一条都没有说明用例集结构变了"
    )


# --------------------------------------------------------------------------- #
# 上面两条护栏必须能被验证会红
# --------------------------------------------------------------------------- #
def test_the_recall_coverage_guard_can_actually_fail() -> None:
    """判据必须真的数得出来 —— 否则它永远绿，和不存在没区别。"""
    cases = _all_cases()
    assert recall_asserting_cases(cases), "一条 recall 用例都没有，判据失效了"

    # 反向：把 recall 断言全部抹掉，必须数出 0 条
    stripped = [
        {**c, "expect": {k: v for k, v in (c.get("expect") or {}).items()
                         if k != "recall_in_speech"}}
        for c in cases
    ]
    assert recall_asserting_cases(stripped) == []

    # 反向：只留 1 条，必须低于下限
    assert len(recall_asserting_cases(stripped[:0] + cases[:1])) < MIN_RECALL_CASES


def test_the_player_name_guard_can_actually_fail() -> None:
    """拿玩家名当 needle 时，判据必须真的抓到。"""
    names = _player_names()
    assert names

    # 干净输入 → 无违规
    assert player_name_recall_needles(
        [{"id": "ok", "expect": {"recall_in_speech": ["靠窗"]}}], names
    ) == []

    # 污染输入 → 必须抓到
    caught = player_name_recall_needles(
        [{"id": "bad", "expect": {"recall_in_speech": ["阿澈"]}}], names
    )
    assert caught == [("bad", "阿澈")], f"漏检或误报：{caught}"

    # 没有 recall 断言的用例不该被误报
    assert player_name_recall_needles(
        [{"id": "write_only", "expect": {"memory_contains": ["阿澈"]}}], names
    ) == []
