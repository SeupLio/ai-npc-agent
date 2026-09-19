"""诊断：duet 场景下"第二个 NPC 的计划为什么推不完"。

背景：`docs/ENGINEERING.md` 记着"LLM 规划不可靠，duet 跑 4 次只有 1 次完成目标"。
复现之后发现失败**不是随机的** —— 4 次里有 3 次悬着的是**同一对**目标
（`play_song` + 它的联合目标 `terrace_night`），而 `serve_guest` 每次都完成了。
所以真正的问题不是"模型不会规划"，是**某一个 NPC 的计划推不动**。

这个脚本逐 tick 打印两个 NPC 的：决策理由 / 计划 / 实际动作，
把"哪一环断的"变成看得见的东西。

用法：
    python scripts/diagnose_duet_planner.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from npc_agent.cast import build_cast  # noqa: E402
from npc_agent.cli import DEMO_SCRIPTS  # noqa: E402
from npc_agent.config import RuntimeConfig, load_scenario  # noqa: E402
from npc_agent.llm import build_llm  # noqa: E402

cfg = RuntimeConfig(
    llm_provider=os.environ.get("NPC_AGENT_PROVIDER") or "openai-compat",
    model=os.environ.get("NPC_AGENT_MODEL") or "kimi-k2.7-code",
    base_url=os.environ.get("NPC_AGENT_BASE_URL") or "",
    api_key=os.environ.get("NPC_AGENT_API_KEY") or "",
)
cfg.use_llm_planner = True
cfg.use_llm_speech = False  # 只测规划

scenario = load_scenario("duet")
cast = build_cast(scenario, build_llm(cfg.llm_provider, model=cfg.model,
                                      base_url=cfg.base_url, api_key=cfg.api_key), cfg)
env = cast.env

for index, line in enumerate(DEMO_SCRIPTS["duet"], 1):
    utterance = None
    if line:
        player_id, text = line
        utterance = env.record_player_utterance(player_id, text)
    print(f"\n=== 第 {index} 轮 ===" + (f"  玩家：{text}" if line else "  （空轮）"),
          flush=True)
    for turn in cast.step(utterance):
        print(f"  [{turn.actor_id}] 决策：{turn.decision_reason}", flush=True)
        if turn.plan:
            p = turn.plan
            print(f"     计划({p.objective_id})：{p.goal}", flush=True)
            for st in p.steps:
                print(f"       - {st.status:8s} {st.tool} {st.args}  {st.note}",
                      flush=True)
        else:
            print("     计划：无", flush=True)
        for call, res in zip(turn.actions, turn.results):
            flag = "ok" if res.ok else "FAIL"
            print(f"     动作 {flag}: {call.tool} {call.args} -> {res.detail}",
                  flush=True)

print("\n=== 终局 ===", flush=True)
print(env.snapshot().get("objectives"), flush=True)
