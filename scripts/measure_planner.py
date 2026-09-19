"""复现 ENGINEERING「LLM 规划仍然不如启发式规划」那一段的 duet 实验。

原文的读数（父提交 `4fd89f5` 上量的）：
    同一个 duet 场景跑 **4 次**，只有 **1 次**两个 NPC 各自完成了目标，
    另外 3 次都**中途停住**，目标停在 pending。

那段文字里的代码路径被 `4c6569c` 动过（`step()` 改成"先回答、再跑计划"，
`_run_plan()` 加了"一轮只说一句"），所以这个数字**必须重新量**，
否则文档里就留着一句描述旧代码的话。

用法：
    python scripts/measure_planner.py --runs 4
    python scripts/measure_planner.py --runs 4 --mode heuristic   # 对照组
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from npc_agent.cast import build_cast  # noqa: E402
from npc_agent.cli import DEMO_SCRIPTS  # noqa: E402
from npc_agent.config import RuntimeConfig, load_scenario  # noqa: E402
from npc_agent.llm import build_llm  # noqa: E402


def _run_once(scenario_id: str, use_model_planner: bool, cfg_kwargs: dict) -> dict:
    cfg = RuntimeConfig(**cfg_kwargs)
    cfg.use_llm_planner = use_model_planner
    # 台词一律走模板：这一节要测的是**规划**，只变一个自变量。
    cfg.use_llm_speech = False

    scenario = load_scenario(scenario_id)
    llm = build_llm(
        cfg.llm_provider, model=cfg.model, base_url=cfg.base_url, api_key=cfg.api_key
    )
    cast = build_cast(scenario, llm, cfg)
    env = cast.env

    for line in DEMO_SCRIPTS.get(scenario_id, []):
        utterance = None
        if line:
            player_id, text = line
            utterance = env.record_player_utterance(player_id, text)
        cast.step(utterance)

    objs = env.snapshot().get("objectives") or {}
    done = [k for k, v in objs.items() if v == "done"]
    pending = [k for k, v in objs.items() if v != "done"]
    return {
        "total": len(objs),
        "done": done,
        "pending": pending,
        "all_done": bool(objs) and not pending,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--scenario", default="duet")
    ap.add_argument(
        "--mode",
        default="model",
        choices=("model", "heuristic"),
        help="model = 模型规划（去掉 --no-planner）；heuristic = 对照组的启发式规划",
    )
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cfg_kwargs = {
        "llm_provider": os.environ.get("NPC_AGENT_PROVIDER") or "openai-compat",
        "model": os.environ.get("NPC_AGENT_MODEL") or "kimi-k2.7-code",
        "base_url": os.environ.get("NPC_AGENT_BASE_URL") or "",
        "api_key": os.environ.get("NPC_AGENT_API_KEY") or "",
    }
    use_model = args.mode == "model"

    print(
        f"场景 {args.scenario}｜规划={args.mode}｜台词=模板｜"
        f"模型={cfg_kwargs['model']}｜跑 {args.runs} 次",
        flush=True,
    )
    results = []
    for i in range(1, args.runs + 1):
        t0 = time.perf_counter()
        r = _run_once(args.scenario, use_model, cfg_kwargs)
        r["seconds"] = round(time.perf_counter() - t0, 1)
        results.append(r)
        mark = "✅ 全部完成" if r["all_done"] else f"❌ 悬着 {r['pending']}"
        print(
            f"  第 {i} 次：{mark}　"
            f"done={len(r['done'])}/{r['total']}　{r['seconds']}s",
            flush=True,
        )

    ok = sum(1 for r in results if r["all_done"])
    print(
        f"\n{args.mode}：{ok}/{args.runs} 次全部完成目标"
        f"（{ok / args.runs:.0%}）",
        flush=True,
    )
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "scenario": args.scenario,
                    "mode": args.mode,
                    "model": cfg_kwargs["model"],
                    "runs": results,
                    "completed": ok,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
