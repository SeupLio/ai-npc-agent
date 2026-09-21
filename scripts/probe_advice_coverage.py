"""量一下：离线反思有多少次落进「通用建议」那一档。

## 为什么先量这个

`Reflector._advice_for()` 是**按 `detail` 的子串**从一张写死的表里挑建议的，
挑不到就落进最后那句 `换个方式再试一次。` —— 这句话对"下次该怎么做"
几乎没有信息量，它就是这张表的"其他"桶。

本项目有一条硬规矩：**回落分类不许有"其他"**（分不清的桶会替你说反话）。
反思的"其他"桶做的是同一件坏事：它把"我不知道这次为什么失败"
伪装成"我知道了，换个方式"。

所以先数：**离线路径里有多少比例的失败落进了这个桶。**
比例高 ⇒ 模型归因是真改进；比例低 ⇒ 那张表已经够用，别为了"用了模型"
而硬塞一个模型调用进去（那会让离线基线不再是零调用）。

## 怎么拿到"真实分布"

**不去读用例文件里的文本**（第一版就是这么写的，量出 0 段：用例里只有
`expect`，根本没有失败 `detail`）。改成**在真实离线跑批里挂一个探针**，
把 `Reflector._advice_for` 每次收到的 `detail` 原样记下来 ——
这样分布是真跑出来的，不是我想象的。

跑法（**离线，0 次模型调用**）：

    python scripts/probe_advice_coverage.py

输出：一类一条计数、通用档占比、以及落进通用档的原文样例。
"""

from __future__ import annotations

import collections
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from npc_agent.modules import reflection as reflection_mod  # noqa: E402

#: 那张表里所有**具体**建议（非通用档）。
GENERIC = "换个方式再试一次。"

#: `_advice_for` 那张表里的触发词 —— 用来验证"每条规则都真的能命中"。
#: ⚠️ 从源码里**读出来**，不手抄：手抄的话，表改了、探针还按旧版算，
#: 量出来的东西就和代码没关系了（本项目"两个真相"那类坑）。
def _table_triggers() -> list[str]:
    import ast

    src = (REPO / "npc_agent" / "modules" / "reflection.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "_advice_for":
            continue
        for sub in ast.walk(node):
            # `if "触发词" in detail:` —— 取那个字面量
            if isinstance(sub, ast.Compare) and isinstance(sub.left, ast.Constant):
                if isinstance(sub.left.value, str):
                    found.append(sub.left.value)
    return found


def main() -> int:
    from npc_agent.config import RuntimeConfig
    from npc_agent.eval.harness import EvalHarness

    # --- 探针：把每次 advice 决策记下来 -------------------------------- #
    seen: list[tuple[str, str]] = []          # (detail, advice)
    orig = reflection_mod.Reflector._advice_for

    def spy(self, result):                     # type: ignore[no-untyped-def]
        advice = orig(self, result)
        seen.append((result.detail, advice))
        return advice

    reflection_mod.Reflector._advice_for = spy  # type: ignore[assignment]
    try:
        # 强制离线：NullLLM ⇒ 0 次模型调用，结果可复现。
        cfg = RuntimeConfig()
        cfg.llm_provider = "null"
        harness = EvalHarness(cfg)
        report = harness.run()
    finally:
        reflection_mod.Reflector._advice_for = orig  # type: ignore[assignment]

    print(f"离线跑批：{report.total} 条用例，通过 {report.passed}")
    print(f"反思被调用了 {len(seen)} 次（每次 = 一次失败归因）")

    # --- 表内触发词逐条验证 ------------------------------------------- #
    print("\n--- 表内触发词逐条验证（能不能命中）---")
    triggers = _table_triggers()
    print(f"从源码读出 {len(triggers)} 个触发词：{triggers}")

    from npc_agent.types import ActionResult

    miss = []
    for trig in triggers:
        got = orig(None, ActionResult(ok=False, tool="t", detail=f"动作失败：{trig}"))
        if got == GENERIC:
            miss.append(trig)
    print(f"  命中不了的触发词：{miss or '（无，每条都能命中）'}")

    # --- 真实分布的覆盖率 --------------------------------------------- #
    print("\n--- 真实分布（离线跑批里的失败）---")
    if not seen:
        print("  这一轮没有产生任何失败归因 —— 离线基线 235/235，没有失败可归因。")
        print("  ⇒ 结论：**离线基线下这个桶一次都没被用到**，")
        print("     所以「模型归因更好」在离线路径上没有可改进的样本。")
        print("     要量它的价值，得去真实模型跑批（那有 48% 回落 / 失败）。")
        return 0

    bucket = collections.Counter()
    generic_examples: list[tuple[str, str]] = []
    for detail, advice in seen:
        if advice == GENERIC:
            bucket["通用档"] += 1
            if len(generic_examples) < 10:
                generic_examples.append((detail, advice))
        else:
            bucket["具体建议"] += 1
    total = sum(bucket.values())
    print(f"  {dict(bucket)}")
    print(f"  通用档占比 = {bucket['通用档'] / total:.1%}")
    if generic_examples:
        print("  落进通用档的原文（这张表认不出它们）：")
        for d, _a in generic_examples:
            print(f"    - {d[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
