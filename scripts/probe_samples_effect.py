"""探针（第三步）：**重复采样取多数，到底把不一致率压下去多少**。

## 为什么不能拿第一步的探针来回答

`probe_judge_samples.py` 量的是"同一条判 N 次，票型有多分裂"。
它能告诉你噪声有多大，但**回答不了"取多数之后还剩多少"** ——
因为那个脚本每次都是**独立**的 N 票，没有"两个独立的 N 票批次"这个对照。

真正要回答的问题是这个：

    同一份内容，用**同一个配置**判**两遍**，两遍的结论一致吗？
      · 单次判决（samples=1）：一致率 = ?
      · 多数票（samples=7）  ：一致率 = ?

**这才是"可复现"的定义**：换一批票，结论还是不是同一个。
取多数的意义就在于此 —— 它不保证判对，它保证**换个时间判还是这个结论**。

## ⚠️ 这个探针最容易做错的地方

1. **两批必须完全独立**。共用票会把两批"对齐"，一致率虚高到接近 100%。
   这里两批各调 `samples` 次，一共 `2 × samples × 标准数 × 对数` 次调用。

2. **必须报调用数**。第一版把 `judge_pairs` 写进列表推导式，
   调用数被放大 5 倍而判决一模一样 —— **只有调用数看得出来**。

3. **必须分开报标准**。把 `grounded` 和 `responsive` 混成一个一致率，
   等于说"改动让判决稳了 12%"，但没人知道稳的是哪一条。
   而这一步的整个意义就是归因。

用法：
    python -u scripts/probe_samples_effect.py
    NPC_AGENT_PROBE_CASES=4 NPC_AGENT_PROBE_SAMPLES=7 python -u scripts/probe_samples_effect.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from npc_agent.eval.judge import (  # noqa: E402
    JUDGE_HISTORY_TURNS,
    LLMJudge,
    dialogue_pairs,
)
from npc_agent.llm import build_llm  # noqa: E402

REPORT = ROOT / "reports" / "eval_model.json"

#: 用来对照的两档票数。`1` = 旧行为（单次判决）。
#:
#: 为什么两档都跑同一份内容：**只有同内容才能比**。
#: 换个用例集再比，差多少都说不清是"票数"还是"内容难易"造成的。
ARMS = [int(x) for x in (os.environ.get("NPC_AGENT_PROBE_ARMS") or "1,7").split(",")]

N_CASES = int(os.environ.get("NPC_AGENT_PROBE_CASES") or "2")
CONCURRENCY = int(os.environ.get("NPC_AGENT_PROBE_CONCURRENCY") or "6")

PERSONA = "你是阿柚，星屿咖啡屋的店主。说话简短、口语化，每次最多两句话。"
SCENE = "星屿咖啡屋，下午，吧台后面。"


def _score_of(verdicts: list[dict], index: int, rubric: str) -> int | None:
    """按**标准名**取判决（不能按下标取 —— 那样会永远拿到第一条标准）。"""
    try:
        for v in verdicts[index]["verdicts"]:
            if v.get("rubric") != rubric:
                continue
            if not v.get("judged"):
                return None
            return 1 if float(v.get("score") or 0) >= 1.0 else 0
    except (IndexError, KeyError):
        return None
    return None


def _pass_over(
    judge: LLMJudge, pairs: list[dict[str, str]], samples: int
) -> dict[str, list[int | None]]:
    """用指定票数把整条用例判**一遍**。

    ⚠️ `judge_pairs(...)` 只调一次（不放进列表推导式）。
    ⚠️ 票数通过 `samples=` 显式传 —— 不依赖实例属性，
       否则"两批独立"这件事就不成立了。
    """
    out = judge.judge_pairs(
        pairs,
        persona=PERSONA,
        scene=SCENE,
        history_turns=JUDGE_HISTORY_TURNS,
        samples=samples,
    )
    return {r: [_score_of(out, k, r) for k in range(len(out))] for r in judge.rubrics}


def _run_tasks(tasks: list, workers: int) -> list:
    if workers <= 1:
        return [task() for task in tasks]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda fn: fn(), tasks))


def main() -> int:
    model = os.environ.get("NPC_AGENT_MODEL") or "kimi-k2.7-code"
    if len(ARMS) < 2:
        print("ARMS 至少要两档才谈得上『比』。")
        return 1
    if not REPORT.exists():
        print(f"找不到 {REPORT.name}。")
        return 1

    llm = build_llm(
        os.environ.get("NPC_AGENT_PROVIDER") or "openai-compat",
        model=model,
        base_url=os.environ.get("NPC_AGENT_BASE_URL") or "",
        api_key=os.environ.get("NPC_AGENT_API_KEY") or "",
        timeout=180.0,
        retries=3,
    )
    if not llm.available:
        print("模型不可用 —— 探针不做任何推断。")
        return 1

    judge = LLMJudge(llm, rubrics=["grounded", "responsive"], max_tokens=8192)

    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    # ⚠️ 键是 `results`（不是 `cases`），转写字段是 `transcript`。
    cases = [c for c in payload.get("results") or [] if c.get("transcript")][:N_CASES]
    units = [
        (str(c.get("case_id") or f"case_{i}"), dialogue_pairs(c["transcript"]))
        for i, c in enumerate(cases)
    ]
    units = [(cid, pairs) for cid, pairs in units if pairs]
    if not units:
        print("报告里没有带转写的用例。")
        return 1

    print(f"模型：{model}｜票数对照：{ARMS}｜用例 {len(units)} 条｜并发 {CONCURRENCY}")
    print("每个票数档跑**两遍独立**的判决，比较两遍结论是否一致。\n")

    # 任务：每条用例 × 每个票数档 × 两批（batch 0 / 1）→ 共 len(units)*len(ARMS)*2 个
    tasks = []
    keys: list[tuple[int, int, int]] = []  # (unit, arm_pos, batch)
    for u, (_cid, pairs) in enumerate(units):
        for a, samples in enumerate(ARMS):
            for batch in (0, 1):
                keys.append((u, a, batch))
                tasks.append(lambda p=pairs, s=samples: _pass_over(judge, p, s))
    results = _run_tasks(tasks, CONCURRENCY)
    by_key = dict(zip(keys, results))

    print("=" * 78)
    print("结果：同一份内容、同一个配置判两遍，结论一致的比例")
    print("=" * 78)
    summary: dict[int, dict[str, dict[str, int]]] = {
        s: {r: {"pairs": 0, "same": 0, "unjudged": 0} for r in judge.rubrics}
        for s in ARMS
    }
    for u, (cid, pairs) in enumerate(units):
        print(f"\n--- {cid}（{len(pairs)} 对）---")
        for a, samples in enumerate(ARMS):
            first = by_key[(u, a, 0)]
            second = by_key[(u, a, 1)]
            for rubric in judge.rubrics:
                same = comparable = unjudged = 0
                for k in range(len(pairs)):
                    x, y = first[rubric][k], second[rubric][k]
                    if x is None or y is None:
                        unjudged += 1
                        continue
                    comparable += 1
                    if x == y:
                        same += 1
                bucket = summary[samples][rubric]
                bucket["pairs"] += comparable
                bucket["same"] += same
                bucket["unjudged"] += unjudged
                rate = same / comparable if comparable else 0.0
                print(f"  samples={samples} [{rubric}]: {comparable} 对可比，"
                      f"两遍一致 {same} = **{rate:.1%}**，未判 {unjudged}")
                print(f"      第1遍: {first[rubric]}")
                print(f"      第2遍: {second[rubric]}")

    print("\n" + "=" * 78)
    print("按标准汇总：两遍一致率（越高 = 越可复现）")
    print("=" * 78)
    print(f"{'标准':<14}" + "".join(f"{'samples=' + str(s):<20}" for s in ARMS))
    for rubric in judge.rubrics:
        cells = []
        for samples in ARMS:
            b = summary[samples][rubric]
            rate = b["same"] / b["pairs"] if b["pairs"] else 0.0
            cells.append(f"{rate:.1%} ({b['same']}/{b['pairs']})")
        print(f"{rubric:<14}" + "".join(f"{c:<20}" for c in cells))

    print(f"\n模型调用数：{judge.calls}（重试 {judge.retries}，"
          f"其中解析失败 {judge.parse_retries}）")
    print("\n" + "=" * 78)
    print("这个探针能说明什么 / 不能说明什么")
    print("=" * 78)
    print("能：**同内容同配置判两遍的一致性**，而且按票数分开报 ——")
    print("    这是「重复采样有没有用」的直接证据，也和调用成本摆在一起。")
    print("不能：一致 ≠ 正确。多数票只压随机噪声，压不掉系统性偏见；")
    print("      而且这里的转写**没有人工标注**，所以「多数票更准」没被验证。")
    print("      n 很小（几条用例的对数），差几个点分不清是票数的功劳还是运气。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
