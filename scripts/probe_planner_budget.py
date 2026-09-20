"""诊断：规划调用失败，是**等得不够久**、**预算不够**，还是**端点抖了**。

## 为什么要有这个脚本

`reports/eval_model_planner.json`（228 条跑批）里 4 条 Minecraft 失败**全部**
落在"规划回落"组里，报错是：

    模型返回空内容（finish_reason=length，思维链 16354 字）

也就是说：**模型压根没被问成**，框架回落启发式，然后启发式的轨迹被当成
"模型规划得不好"记了下来。这跟"配方数量算不对"是两回事 ——
但路线图上那条写的是后者。

## 三个**独立的**假设，必须分开测

| 假设 | 失败原文长什么样 | 修法 |
|---|---|---|
| **读超时不够** | `TimeoutError`，思维链 **0** 字（响应根本没回来） | 放宽超时 |
| **预算不够** | `finish_reason=length`，思维链 **N 字** | 调大 `max_tokens` |
| **端点抖了** | 同上，但**重发一次就好** | 重试 |

⚠️ 它们是**互相掩盖**的：60s 超时下所有失败都是 `TimeoutError`，
**看不见**预算问题；把超时放宽之后预算才露出来。混在一起测，
就会把"我们等得不够久"记成"模型规划得不好"。

用法：
    python scripts/probe_planner_budget.py
    PROBE_BUDGETS=4096,16384 PROBE_TIMEOUTS=60,180 PROBE_RETRIES=1,3 \\
        python scripts/probe_planner_budget.py
"""

from __future__ import annotations

import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from npc_agent.cast import build_cast  # noqa: E402
from npc_agent.cli import DEMO_SCRIPTS  # noqa: E402
from npc_agent.config import RuntimeConfig, load_scenario  # noqa: E402
from npc_agent.llm import build_llm  # noqa: E402

BUDGETS = [int(x) for x in (os.environ.get("PROBE_BUDGETS") or "4096,16384").split(",")]

#: 客户端的读超时。
TIMEOUTS = [float(x) for x in (os.environ.get("PROBE_TIMEOUTS") or "60").split(",")]

#: 传输层重试次数（含首次）。`1` = 不重试。
RETRIES = [int(x) for x in (os.environ.get("PROBE_RETRIES") or "1").split(",")]


# --------------------------------------------------------------------------- #
# 直接数 `urlopen` 被调了几次。
#
# ⚠️ 重试发生在**客户端内部**（`OpenAICompatLLM._attempt`），从外面数
# `complete_json` 是看不见的 —— 一次逻辑调用重试 3 次，外面也只看到 1 次。
# 所以必须在这一层数，"平均试了几次"才是真的。
_URLOPEN_CALLS = [0]
_ORIG_URLOPEN = urllib.request.urlopen


def _counting_urlopen(*args: object, **kwargs: object):
    _URLOPEN_CALLS[0] += 1
    return _ORIG_URLOPEN(*args, **kwargs)  # type: ignore[arg-type]


urllib.request.urlopen = _counting_urlopen  # type: ignore[assignment]


class RecordingLLM:
    """把每次调用的预算和结局记下来。

    只做记录，不改行为 —— 规划器看到的还是原来那个 LLM，
    所以"回落"照旧发生，我们只是终于能看见它。
    """

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self.calls: list[tuple[int, bool, str]] = []

    @property
    def available(self) -> bool:
        return bool(getattr(self._inner, "available", False))

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def complete(self, *args: object, **kwargs: object) -> object:
        return self._inner.complete(*args, **kwargs)  # type: ignore[attr-defined]

    def complete_json(self, *args: object, **kwargs: object) -> object:
        budget = int(kwargs.get("max_tokens") or 0)
        try:
            out = self._inner.complete_json(*args, **kwargs)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 —— 就是要连"没抛异常"的那类也分开记
            self.calls.append((budget, False, str(exc)[:240]))
            raise
        self.calls.append((budget, True, ""))
        return out


_COT = re.compile(r"思维链\s*(\d+)\s*字")


def cot_len(text: str) -> int:
    m = _COT.search(text)
    return int(m.group(1)) if m else 0


def run(budget: int, timeout: float, retries: int) -> tuple[list[tuple[int, bool, str]], dict]:
    cfg = RuntimeConfig(
        llm_provider=os.environ.get("NPC_AGENT_PROVIDER") or "openai-compat",
        model=os.environ.get("NPC_AGENT_MODEL") or "kimi-k2.7-code",
        base_url=os.environ.get("NPC_AGENT_BASE_URL") or "",
        api_key=os.environ.get("NPC_AGENT_API_KEY") or "",
    )
    cfg.use_llm_planner = True
    cfg.use_llm_speech = False  # 只测规划：台词预算不参与
    cfg.max_tokens = budget

    llm = RecordingLLM(
        build_llm(
            cfg.llm_provider,
            model=cfg.model,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=timeout,
            retries=retries,
        )
    )
    scenario = load_scenario("village")
    cast = build_cast(scenario, llm, cfg)  # type: ignore[arg-type]
    env = cast.env

    for line in DEMO_SCRIPTS["village"]:
        utterance = None
        if line:
            player_id, text = line
            utterance = env.record_player_utterance(player_id, text)
        for _turn in cast.step(utterance):
            pass
    return llm.calls, env.snapshot()


def kind_of(error: str) -> str:
    """按失败原文分类 —— 这几类的修法完全不同。"""
    if "Timeout" in error or "timed out" in error:
        return "超时（响应没回来）"
    if "finish_reason=length" in error or "思维链" in error or "max_tokens" in error:
        return "预算（思维链吃光）"
    if "429" in error or "限额" in error:
        return "配额"
    return "其他"


def main() -> int:
    print(
        f"预算档位：{BUDGETS}｜超时档位：{TIMEOUTS}｜重试档位：{RETRIES}\n",
        flush=True,
    )
    # 平铺成一条臂列表 —— 比三层嵌套好读，而且改档位不用动缩进。
    arms = [(t, r, b) for t in TIMEOUTS for r in RETRIES for b in BUDGETS]
    summary: list[tuple[float, int, int, int, int, int, dict[str, int]]] = []
    for timeout, retries, budget in arms:
        print(
            f"=== max_tokens={budget}  timeout={timeout:g}s  retries={retries} ===",
            flush=True,
        )
        _URLOPEN_CALLS[0] = 0
        calls, snap = run(budget, timeout, retries)
        attempts = _URLOPEN_CALLS[0]
        ok = sum(1 for _b, good, _e in calls if good)
        bad = [c for c in calls if not c[1]]
        print(f"  规划调用 {len(calls)} 次，成功 {ok}，失败 {len(bad)}", flush=True)
        if calls:
            print(
                f"  HTTP 尝试 {attempts} 次 / 逻辑调用 {len(calls)} 次"
                f"（平均每次 {attempts / len(calls):.2f} 次）",
                flush=True,
            )
        kinds: dict[str, int] = {}
        for _b, _good, err in bad:
            k = kind_of(err)
            kinds[k] = kinds.get(k, 0) + 1
            print(f"    失败[{k}]：{err}", flush=True)
        cots = [cot_len(e) for _b, _g, e in bad]
        if cots:
            print(f"  失败时的思维链长度：{cots}（最长 {max(cots)} 字）", flush=True)
        ayan = (snap.get("actors") or {}).get("ayan") or {}
        print(f"  终局背包：{ayan.get('inventory')}", flush=True)
        print(f"  终局目标：{snap.get('objectives')}", flush=True)
        print(f"  世界标记：{snap.get('world_flags')}", flush=True)
        print(flush=True)
        summary.append(
            (timeout, retries, budget, len(calls), len(bad), attempts, kinds)
        )

    print("=== 汇总 ===", flush=True)
    print(
        f"  {'timeout':>8s} {'retries':>8s} {'max_tokens':>10s} {'调用':>5s}"
        f" {'失败':>5s} {'HTTP尝试':>8s} {'成功率':>7s}  失败构成",
        flush=True,
    )
    for timeout, retries, budget, total, failed, attempts, kinds in summary:
        rate = (total - failed) / total * 100 if total else 0.0
        detail = "、".join(f"{k}×{v}" for k, v in sorted(kinds.items())) or "—"
        print(
            f"  {timeout:8g} {retries:8d} {budget:10d} {total:5d} {failed:5d}"
            f" {attempts:8d} {rate:6.1f}%  {detail}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
