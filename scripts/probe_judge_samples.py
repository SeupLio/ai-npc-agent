"""探针（第二步）：裁判判决不可复现 —— **要采样几次才够**。

## 为什么先做这个，再改代码

前一步（`probe_judge_history.py`）量出的一件事：`JUDGE_TEMPERATURE = 0.0`、
prompt **逐字节相同**，同一条台词判两次仍有 ~22% 的概率给出相反的结论。
也就是说**现有报告里的每一个判分数字（包括 kappa）都只是"某一次运行的读数"**，
不是"裁判的判断力"。修法是**重复采样取多数**（不是继续调 temperature）。

但"重复采样"要先回答一个问题：**采几次？** 拍脑袋写个 3 或 5，
然后就没人知道这个数字是哪来的 —— 那正是本项目反复在修的那类毛病。
所以这个探针先把它量出来。

## 它怎么量（关键：**同一个 prompt 反复判，只看它自己跟自己一不一致**）

对同一条台词、同一个 prompt 判 `N` 次，得到 N 个 0/1：

    全 0 或全 1        → 这条"稳定"（这 N 次里）
    既有 0 又有 1      → 这条"不稳定"，单次判决不可信

两个指标：

    **不稳定率**  不稳定条数 / 总条数        —— 有多少比例的内容是单次判不得的
    **多数票翻面率**  多数票 != 少数票时，少数票占比的期望 —— 采 N 次取多数，
                    仍然会跟"另一批 N 次"给出不同结论的概率有多大

## ⚠️ 三个会毁掉这次测量的坑（都在本文件里踩过）

1. **`or` 会把 `0` 吃掉**。`NPC_AGENT_PROBE_SAMPLES=0` 若写成 `int(os.environ.get(...) or "4")`，
   传 0 会变成 4 —— 而 0 在这里是个**有意义的值**（"不重复采样"）。
   凡是要取环境变量里的数字，一律显式判 `None`。

2. **键不能用 `turns` / 不能用序号以外的任何"看起来唯一"的东西**。
   N 次调用的配置**完全相同**，只有**第几次**不同。所以键只能是
   `(用例, 样本序号)`，并且并发时必须 `ex.map` 保序 ——
   `as_completed` 会把结果打乱，而打乱**不报错**，只会给出一个看起来正常的错误结论。

3. **`judge_pairs(...)` 必须只调一次**。第一版把它写进列表推导式里，
   模型调用数被悄悄放大 5 倍，而判决一模一样 —— 只有调用数看得出来。

用法：
    python -u scripts/probe_judge_samples.py                 # 默认 N=7，2 条真实用例
    NPC_AGENT_PROBE_SAMPLES=7 NPC_AGENT_PROBE_CASES=2 python -u scripts/probe_judge_samples.py
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


def _int_env(name: str, default: int) -> int:
    """读一个整数环境变量。

    ⚠️ **不能写 `int(os.environ.get(name) or default)`** —— `or` 会吃掉
    字符串 `"0"` 吗？不会，`"0"` 是非空字符串，永远为真。但一旦有人
    把它改成 `int(...)` 之后再 `or`，`0` 就没了。这里统一走显式判 None，
    顺便把"传了非法值"变成一次**响亮的失败**，而不是静默回落默认值。
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name}={raw!r} 不是整数") from exc


#: 每条台词判几次。**这是被测量的未知量**，不是拍出来的值。
SAMPLES = _int_env("NPC_AGENT_PROBE_SAMPLES", 7)

#: 取几条真实用例（每条用例有若干对，全部计入样本）。
N_CASES = _int_env("NPC_AGENT_PROBE_CASES", 2)

#: 并发。判分是几小时的长作业，探针也一样 —— 串行 7 次 × 若干对会很久。
CONCURRENCY = _int_env("NPC_AGENT_PROBE_CONCURRENCY", 6)

PERSONA = "你是阿柚，星屿咖啡屋的店主。说话简短、口语化，每次最多两句话。"
SCENE = "星屿咖啡屋，下午，吧台后面。"


def _score_of(verdicts: list[dict], index: int, rubric: str) -> int | None:
    """取第 `index` 对里**指定标准**的判决，按**标准名**取。

    （按 `[0]` 取会永远拿到第一条标准，而数字照样排得整整齐齐 ——
    这个错只在打印理由时才看得出来。理由同 `probe_judge_history.py`。）
    """
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


def _one_pass(judge: LLMJudge, pairs: list[dict[str, str]]) -> dict[str, list[int | None]]:
    """把整条用例判一遍（turns 用默认值，这里测的不是历史而是重复性）。"""
    out = judge.judge_pairs(
        pairs, persona=PERSONA, scene=SCENE, history_turns=JUDGE_HISTORY_TURNS
    )
    return {r: [_score_of(out, k, r) for k in range(len(out))] for r in judge.rubrics}


def _run_tasks(tasks: list, workers: int) -> list:
    """并发跑一批无参调用，**按提交顺序**返回（`ex.map` 保序）。"""
    if workers <= 1:
        return [task() for task in tasks]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda fn: fn(), tasks))


def _majority(values: list[int]) -> int:
    return 1 if sum(values) * 2 > len(values) else 0


def main() -> int:
    model = os.environ.get("NPC_AGENT_MODEL") or "kimi-k2.7-code"
    print(f"模型：{model}｜每条判 {SAMPLES} 次｜用例 {N_CASES} 条｜并发 {CONCURRENCY}")

    if SAMPLES < 2:
        print("SAMPLES < 2 —— 一次调用没有「重复」可言，这个探针不做任何推断。")
        return 1
    if not REPORT.exists():
        print(f"找不到 {REPORT.name}，没有可判的真实转写。")
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
    if not cases:
        print("报告里没有带转写的用例。")
        return 1

    # 把"用例 × 词典位置"摊平成一个个待判单元，各自判 SAMPLES 次。
    units: list[tuple[int, str, list[dict[str, str]]]] = []
    for i, case in enumerate(cases):
        pairs = dialogue_pairs(case["transcript"])
        if pairs:
            units.append((i, str(case.get("case_id") or f"case_{i}"), pairs))

    print(f"\n{'=' * 78}")
    print(f"每条判 {SAMPLES} 次：{len(units)} 条用例 × {SAMPLES} 次 × "
          f"{len(judge.rubrics)} 个标准（按对展开）")
    print(f"{'=' * 78}")

    tasks = []
    keys: list[tuple[int, int]] = []
    for u, (_i, _cid, pairs) in enumerate(units):
        for s in range(SAMPLES):
            # ⚠️ 键 = (用例序号, **第几次**)。N 次调用的配置完全相同，
            # 只有"第几次"能区分它们。用别的任何东西当键都会让后一次覆盖前一次，
            # 于是"不稳定率"被算成 0 —— 又一个漂亮但完全虚假的 0。
            keys.append((u, s))
            tasks.append(lambda p=pairs: _one_pass(judge, p))
    results = _run_tasks(tasks, CONCURRENCY)
    by_key = dict(zip(keys, results))

    # 样本总数（按对 × 标准展开）
    stats = {r: {"total": 0, "unstable": 0, "unjudged": 0, "lone": 0} for r in judge.rubrics}
    per_case: list[dict] = []

    for u, (i, cid, pairs) in enumerate(units):
        print(f"\n--- {cid}（{len(pairs)} 对）---")
        row: dict = {"case_id": cid, "pairs": []}
        for k in range(len(pairs)):
            line = []
            for rubric in judge.rubrics:
                seq = [by_key[(u, s)][rubric][k] for s in range(SAMPLES)]
                known = [v for v in seq if v is not None]
                stats[rubric]["total"] += 1
                if len(known) < SAMPLES:
                    stats[rubric]["unjudged"] += 1
                if len(known) < 2:
                    line.append(f"{rubric}=未判")
                    continue
                ones = sum(known)
                if 0 < ones < len(known):
                    stats[rubric]["unstable"] += 1
                    # 少数票有几个（用来估"多数票会不会翻"）
                    stats[rubric]["lone"] += min(ones, len(known) - ones) / len(known)
                mark = "⚠不稳定" if 0 < ones < len(known) else "稳定"
                line.append(f"{rubric}={ones}/{len(known)} {mark}")
                row["pairs"].append(
                    {"rubric": rubric, "seq": seq, "ones": ones, "n": len(known)}
                )
            print(f"  第{k + 1}对: " + "｜".join(line))
        per_case.append(row)

    print(f"\n{'=' * 78}")
    print("按标准汇总（同一 prompt 反复判）")
    print(f"{'=' * 78}")
    for rubric, s in stats.items():
        n = s["total"]
        rate = s["unstable"] / n if n else 0.0
        print(f"  {rubric}: {n} 对 | **不稳定 {s['unstable']} 对 = {rate:.1%}** | "
              f"未判成 {s['unjudged']}")

    if SAMPLES >= 3:
        print("\n采样数 → 多数票的把握（用上面实测的票型直接算，不是套公式）：")
        print("  对每一对，把 N 次投票里少数票的比例记为 p；"
              "从同一分布再抽 N 次，多数票翻面的概率 ≈ P(少数票这一侧 ≥ N/2)。")
        for rubric, s in stats.items():
            n = s["total"]
            if not n:
                continue
            print(f"  {rubric}: 少数票占总票数 {s['lone'] / (n or 1):.1%}"
                  f"（Σ min(k,N-k)/N ÷ 对数）")
    else:
        print("\nSAMPLES < 3，票型样本太少，不做多数票推算。")

    print(f"\n模型调用数：{judge.calls}（预期 {len(units) * SAMPLES}；"
          f"重试 {judge.retries}，其中解析失败 {judge.parse_retries}）")
    print("\n" + "=" * 78)
    print("这个探针能说明什么 / 不能说明什么")
    print("=" * 78)
    print("能：同一条台词、同一个 prompt 反复判，**判决本身**有多不稳（不稳定率）。")
    print("    它把「需要重复采样」从一个说法变成一个数字，并给出采样次数的量级。")
    print("不能：它**不**说明取多数票之后判决就变**对**了 ——")
    print("      多数票只能消掉随机噪声，消不掉系统性偏见（位置偏见、宽松倾向）。")
    print("      而且这里的转写**没有人工标注**，所以「多数票更准」这件事没被验证。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
