"""Reflection 模块的护栏：**该反思的和不该反思的，必须分得开**。

## 这个文件防的是哪一类 bug

`ok=False` 原来混着两类完全不同的东西：

* 动作**真的**没做成（位置不对、材料不够、台词越界）→ 该反思。
* 动作**按策略主动让出**（发言占比到顶让出话头、这一轮已经有人开口）→ **正确行为**。

不分开的代价是量出来的（`scripts/probe_advice_coverage.py`，离线跑批 235 条）：
反思往记忆里写了 **457** 条，其中 **139 条（30.4%）** 是「让出话头」被当成失败；
而全语料**最高频**的那条反思正是这个（出现 **122** 次）：

    教训：speak 失败（发言占比 67% 已超上限，本轮主动让出话头）。
    我说话太多了，这一轮把机会留给玩家。

句子自己写着「**主动**让出话头」，却顶着「教训：」和「失败」。
而反思以 `importance=0.85` 进记忆、并被规划 prompt 以【想起的事】取走
⇒ 配了真实模型时，模型会读到一句**假的自我评价**：
**它被告知自己有个"话太多"的毛病，而那个毛病是它守规矩。**

这类 bug 的特征和本项目其他几个一模一样：**不报错、分数不变**
（修完离线基线仍是 235/235），只有去数"写进记忆的东西长什么样"才看得见。
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from npc_agent.modules.memory import MemoryManager, MemoryStore
from npc_agent.modules.persona import Persona
from npc_agent.modules.reflection import Reflector
from npc_agent.modules.state import StateTracker
from npc_agent.types import (
    OUTCOME_DECLINED,
    OUTCOME_FAILED,
    ActionResult,
    AgentTurn,
)

REPO = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# 脚手架
# --------------------------------------------------------------------------- #
def _reflector() -> Reflector:
    return Reflector(Persona(id="a", name="阿柚"), MemoryManager(MemoryStore()))


def _turn() -> AgentTurn:
    return AgentTurn(tick=1, actor_id="a")


def _declined(detail: str = "发言占比 67% 已超上限，本轮主动让出话头") -> ActionResult:
    return ActionResult(
        ok=False, tool="speak", detail=detail, outcome=OUTCOME_DECLINED
    )


def _failed(detail: str = "柠檬在后厨，你现在在吧台，需要先 move_to 过去") -> ActionResult:
    return ActionResult(ok=False, tool="take_item", detail=detail)


# --------------------------------------------------------------------------- #
# 一、`ActionResult` 的默认值必须"默认最坏"
# --------------------------------------------------------------------------- #
def test_the_default_outcome_is_failed_not_declined() -> None:
    """忘了声明 `outcome` 的调用点要被当成**真失败**。

    两个方向的代价不对称：
      * 误判成 failed  ⇒ 多写一条反思（浪费，但不会让 Agent 学错东西）。
      * 误判成 declined ⇒ **漏掉一次该学的教训**（Agent 会重复犯错）。
    所以默认值取"更坏"的那个。
    """
    assert ActionResult(ok=False, tool="t").outcome == OUTCOME_FAILED
    assert not ActionResult(ok=False, tool="t").declined


def test_a_successful_action_is_never_declined() -> None:
    """`declined` 只在 `ok=False` 时有意义 —— 成功的动作谈不上"让出"。"""
    ok = ActionResult(ok=True, tool="speak", detail="好，稍等。", outcome=OUTCOME_DECLINED)
    assert not ok.declined


def test_declined_renders_as_yield_not_fail() -> None:
    """转写里不能把"主动让出"印成 `FAIL`。

    转写是人读报告的依据之一 —— 把正确行为印成 FAIL，
    读报告的人会去查一个不存在的 bug。
    """
    assert "[yield]" in _declined().render()
    assert "FAIL" not in _declined().render()
    assert "[FAIL]" in _failed().render()


# --------------------------------------------------------------------------- #
# 二、主动让出**不许**进反思
# --------------------------------------------------------------------------- #
def test_yielding_the_floor_is_not_a_lesson() -> None:
    """核心回归：发言权让出**不该**写出任何教训。"""
    r = _reflector()
    out = r.reflect(_turn(), [_declined()], StateTracker("a", "阿柚"), turn_index=1)
    assert out is None, f"让出话头被当成失败反思了：{out}"
    assert r.lessons == []


def test_yielding_the_floor_does_not_reach_memory() -> None:
    """教训会写回记忆 —— 记忆里不能出现"我说话太多了"这种假自评。"""
    r = _reflector()
    r.reflect(_turn(), [_declined()], StateTracker("a", "阿柚"), turn_index=1)
    hits = r.memory.store.search("说话太多", k=5, now=2)
    assert hits == [], f"假教训进了记忆：{[h.content for h in hits]}"


def test_a_real_failure_still_produces_a_lesson() -> None:
    """反向验证：**真的失败必须照旧反思** —— 修的是污染，不是把反思关掉。"""
    r = _reflector()
    out = r.reflect(_turn(), [_failed()], StateTracker("a", "阿柚"), turn_index=1)
    assert out is not None
    assert out.trigger == "tool_failure"
    assert "工位" in out.note
    assert len(r.lessons) == 1


def test_a_mixed_batch_reflects_only_the_real_failure() -> None:
    """一轮里既有让出又有真失败 ⇒ 反思的是**真失败**那条。"""
    r = _reflector()
    out = r.reflect(
        _turn(), [_declined(), _failed()], StateTracker("a", "阿柚"), turn_index=1
    )
    assert out is not None
    assert "让出话头" not in out.note
    assert "move_to" in out.note or "工位" in out.note


# --------------------------------------------------------------------------- #
# 三、`should_reflect` 和 `reflect` 的判据必须一致
# --------------------------------------------------------------------------- #
def test_should_reflect_agrees_with_reflect_on_declines() -> None:
    """两处判据不一致的话，会出现"说要反思、结果返回 None"的怪状态。"""
    r = _reflector()
    results = [_declined()]
    # turn_index=1 且 reflect_every 默认 6 ⇒ 周期性回顾也不该触发。
    assert r.should_reflect(1, results) is False
    assert r.reflect(_turn(), results, StateTracker("a", "阿柚"), turn_index=1) is None


def test_should_reflect_still_fires_for_real_failures() -> None:
    r = _reflector()
    assert r.should_reflect(1, [_failed()]) is True


def test_the_periodic_review_still_comes_around() -> None:
    """周期性回顾不受这次改动影响 —— 让出话头不该把它顶掉。"""
    r = _reflector()
    out = r.reflect(
        _turn(), [_declined()], StateTracker("a", "阿柚"), turn_index=6
    )
    assert out is not None
    assert out.trigger == "periodic"


# --------------------------------------------------------------------------- #
# 四、别再让它长回来
# --------------------------------------------------------------------------- #
def test_no_llm_backed_reflection_path_comes_back() -> None:
    """`reflect_with_llm` 被删掉了，不许复活。

    它写着「比关键词匹配更准」但全仓库没有调用点，而且量出来
    它**不会更准**：表在真实失败上覆盖率 100%，而它自己 120 的预算
    对推理模型太紧（实测约 10% 概率返回空）。删掉的理由写在
    `reflection.py` 末尾。

    ⚠️ 判据是**有没有被定义 / 被调用**，不是"这个词有没有出现"：
    文件末尾那段注释**必须**能写下这个名字（否则以后没人敢记录删除理由）。
    一开始写成子串检查，结果连注释一起禁掉了 —— 那是护栏自己写错，
    不是代码有问题。
    """
    tree = ast.parse(
        (REPO / "npc_agent" / "modules" / "reflection.py").read_text(encoding="utf-8")
    )
    defined = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "reflect_with_llm" not in defined, "一个没有调用点的模型路径又长回来了"

    # 也不许它从别处被调起来（整个包扫一遍，包括 `self.reflect_with_llm`）。
    callers: list[str] = []
    for py in sorted((REPO / "npc_agent").rglob("*.py")):
        if py.name == "reflection.py":
            continue
        text = py.read_text(encoding="utf-8")
        if "reflect_with_llm" in text:
            callers.append(str(py.relative_to(REPO)))
    assert callers == [], f"有人又在调那条已删的模型路径：{callers}"

    # `Reflector` 也不该再收一个用不上的 `llm`。
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Reflector":
            init = next(
                n for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__"
            )
            args = [a.arg for a in init.args.args]
            assert "llm" not in args, f"Reflector 又开始收一个不读的 llm 了：{args}"


def test_the_decline_vocabulary_is_declared_at_the_producer() -> None:
    """`declined` 必须由**产出方**声明，不许在消费方按文案匹配。

    按 `detail` 文本认类是本项目反复踩的坑（上游一改文案就静默失效）——
    实测教训：`urgency >= 0.9` 当闸门那次就是这么坏的。
    """
    src = (REPO / "npc_agent" / "modules" / "tools.py").read_text(encoding="utf-8")
    assert "OUTCOME_DECLINED" in src, "`_speak` 不再声明「主动让出」了"
    # 两处让出都要声明（让出话头 + 发言占比到顶）。
    assert src.count("outcome=OUTCOME_DECLINED") >= 2


def test_reflection_does_not_regex_the_detail_text() -> None:
    """反思**不许**靠文案匹配来区分让出 / 失败。

    ⚠️ 判据只看**代码**里的字符串比较，不看注释 —— 注释里当然要能写
    「让出话头」（`reflect()` 里那段就是靠它讲清楚为什么要排除）。
    所以先剥掉 docstring 和注释，再找 `in detail` 这类比较。
    """
    src = (REPO / "npc_agent" / "modules" / "reflection.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    # 收集所有"出现在比较/成员判断里的字符串字面量"。
    compared: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if isinstance(op, (ast.In, ast.NotIn)):
                    compared.extend(
                        n.value
                        for n in ast.walk(node)
                        if isinstance(n, ast.Constant) and isinstance(n.value, str)
                    )

    yield_words = [s for s in compared if "让出话头" in s or "超上限" in s]
    assert yield_words == [], f"又在按文案认「让出话头」了：{yield_words}"
    assert ".declined" in src, "反思没在读结构化字段"


@pytest.mark.parametrize(
    "detail",
    [
        "发言占比 67% 已超上限，本轮主动让出话头",
        "本轮已有另一位 NPC 开口，我让出话头",
    ],
)
def test_both_yield_paths_are_declared(detail: str) -> None:
    """`_speak` 里两条"让出"分支都要打上 `declined`（漏一条就漏一类污染）。"""
    r = _reflector()
    out = r.reflect(_turn(), [_declined(detail)], StateTracker("a", "阿柚"), turn_index=1)
    assert out is None, f"这条让出分支没被声明成 declined：{detail}"
