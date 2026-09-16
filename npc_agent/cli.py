"""命令行入口。

    python -m npc_agent.cli demo  --scenario icebreaker   # 看一段完整交互
    python -m npc_agent.cli chat  --scenario tutorial     # 自己上手玩
    python -m npc_agent.cli eval                          # 跑评测出数字
    python -m npc_agent.cli tools                         # 看工具清单
    python -m npc_agent.cli info                          # 看人设与场景

切换模型（任何 OpenAI 兼容端点）：
    python -m npc_agent.cli demo --provider openai-compat \
        --base-url http://localhost:8000/v1 --model Qwen3-8B-Instruct
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from .agent import NPCAgent
from .config import RuntimeConfig, list_scenarios, load_persona, load_scenario
from .env.star_isle import StarIsleEnv
from .llm import build_llm
from .modules.persona import Persona

# 每个场景配一段固定的演示脚本，保证 Demo 可复现（录屏/截图用）
DEMO_SCRIPTS: dict[str, list[Optional[tuple[str, str]]]] = {
    "icebreaker": [
        ("player_a", "大家好，我是阿澈，平时喜欢爬山。"),
        None,
        ("player_b", "我叫小满，最近在学做甜点。"),
        None,
        ("player_c", "我是阿岚，做插画的。"),
        None,
        ("player_a", "阿柚，能给我来杯拿铁吗？"),
        None,
        None,
        None,
        None,
    ],
    "tutorial": [
        ("player_a", "阿柚，我第一次来，这里怎么点单呀？"),
        None,
        None,
        None,
        None,
        None,
        None,
    ],
    "hosting": [
        None,
        ("player_a", "开始吧！"),
        None,
        ("player_b", "我猜是天蝎座？"),
        None,
        None,
    ],
}


# --------------------------------------------------------------------------- #
def _build(args: argparse.Namespace):
    cfg = RuntimeConfig.from_env()
    if getattr(args, "provider", None):
        cfg.llm_provider = args.provider
    if getattr(args, "model", None):
        cfg.model = args.model
    if getattr(args, "base_url", None):
        cfg.base_url = args.base_url
    if getattr(args, "api_key", None):
        cfg.api_key = args.api_key
    cfg.verbose = bool(getattr(args, "verbose", False))

    scenario_id = getattr(args, "scenario", "tutorial")
    scenario = load_scenario(scenario_id)
    persona = Persona.from_dict(load_persona(scenario.get("npc", "ayou")))
    env = StarIsleEnv(scenario, persona.id, persona.name)
    llm = build_llm(
        cfg.llm_provider, model=cfg.model, base_url=cfg.base_url, api_key=cfg.api_key
    )
    agent = NPCAgent(persona, env, scenario, llm, cfg)
    return cfg, scenario, persona, env, agent


# --------------------------------------------------------------------------- #
def cmd_demo(args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    cfg, scenario, persona, env, agent = _build(args)

    mode = "在线模型" if agent.llm.available else "离线启发式（未配置模型）"
    console.print(
        Panel.fit(
            f"[bold]{scenario.get('name')}[/bold] · {scenario.get('description','')}\n"
            f"NPC：[bold]{persona.name}[/bold]（{persona.role}）\n"
            f"推理模式：{mode}",
            title="星屿咖啡屋",
            border_style="blue",
        )
    )

    script = DEMO_SCRIPTS.get(args.scenario, DEMO_SCRIPTS["tutorial"])
    if args.ticks:
        script = script[: args.ticks]

    for index, line in enumerate(script):
        utterance = None
        console.print(f"\n[dim]── 第 {index + 1} 轮 ──[/dim]")
        if line:
            player_id, text = line
            utterance = env.record_player_utterance(player_id, text)
            console.print(f"[cyan]玩家 {utterance.speaker_name}[/cyan]：{text}")
        turn = agent.step(utterance)

        if turn.decision_reason:
            console.print(f"  [dim]（{turn.decision_reason}）[/dim]")
        for action, result in zip(turn.actions, turn.results):
            if action.tool == "speak":
                continue
            style = "green" if result.ok else "red"
            console.print(f"  [{style}]▸ {action.render()}[/{style}]")
            if not result.ok:
                console.print(f"      [red]✗ {result.detail}[/red]")
        if turn.say:
            console.print(f"  [yellow]NPC[/yellow]：{turn.say}")

        env.advance_tick()

    snapshot = env.snapshot()
    table = Table(title="结束时世界状态", show_header=True, header_style="bold")
    table.add_column("项目")
    table.add_column("值")
    table.add_row("世界标记", ", ".join(snapshot["world_flags"]) or "（无）")
    table.add_row(
        "目标",
        ", ".join(f"{k}={v}" for k, v in snapshot["objectives"].items()) or "（无）",
    )
    stats = agent.memory.store.stats()
    table.add_row(
        "记忆",
        f"episodic={stats.episodic} semantic={stats.semantic} "
        f"reflection={stats.reflection} 巩固={stats.consolidated}",
    )
    table.add_row("发言占比", f"{agent.state.npc_share():.0%}")
    console.print()
    console.print(table)

    if agent.reflector.lessons:
        console.print("\n[bold]沉淀下来的教训[/bold]")
        console.print(agent.reflector.render_lessons())
    return 0


# --------------------------------------------------------------------------- #
def cmd_chat(args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    cfg, scenario, persona, env, agent = _build(args)
    players = [p["id"] for p in scenario.get("players", [])]
    current = players[0] if players else "player_a"

    console.print(
        Panel(
            f"正在和 [bold]{persona.name}[/bold] 对话（场景：{scenario.get('name')}）。\n"
            f"当前身份：{env.actors[current].name}　切换玩家：/as player_b　退出：/quit",
            border_style="blue",
        )
    )
    if not agent.llm.available:
        console.print(
            "[dim]提示：当前是离线启发式模式。配置 NPC_AGENT_PROVIDER=openai-compat "
            "与 NPC_AGENT_MODEL 后可用真实模型对话。[/dim]"
        )

    while True:
        try:
            raw = console.input(f"[cyan]{env.actors[current].name}[/cyan] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            continue
        if raw in ("/quit", "/exit"):
            break
        if raw.startswith("/as "):
            target = raw[4:].strip()
            if target in players:
                current = target
                console.print(f"[dim]已切换为 {env.actors[current].name}[/dim]")
            else:
                console.print(f"[red]可用玩家：{', '.join(players)}[/red]")
            continue

        utterance = env.record_player_utterance(current, raw)
        turn = agent.step(utterance)
        for action, result in zip(turn.actions, turn.results):
            if action.tool == "speak":
                continue
            style = "green" if result.ok else "red"
            console.print(f"  [{style}]▸ {action.render()}[/{style}]")
        console.print(f"[yellow]{persona.name}[/yellow]：{turn.say or '（沉默）'}")
        env.advance_tick()
    return 0


# --------------------------------------------------------------------------- #
def cmd_eval(args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.table import Table

    from .eval import EvalHarness

    console = Console()
    cfg = RuntimeConfig.from_env()
    if getattr(args, "provider", None):
        cfg.llm_provider = args.provider
    if getattr(args, "model", None):
        cfg.model = args.model
    if getattr(args, "base_url", None):
        cfg.base_url = args.base_url
    if getattr(args, "api_key", None):
        cfg.api_key = args.api_key

    categories = args.category or None
    report = EvalHarness(cfg).run(categories)

    table = Table(title="评测结果", header_style="bold")
    for column in ("用例", "场景", "任务", "工具", "记忆", "人设", "安全", "结论"):
        table.add_column(column, justify="left" if column in ("用例", "场景") else "center")

    for result in report.results:
        scores = result.metrics.as_dict()
        table.add_row(
            result.case_id,
            result.scenario,
            f"{scores['task']:.2f}",
            f"{scores['tools']:.2f}",
            f"{scores['memory']:.2f}",
            f"{scores['persona']:.2f}",
            f"{scores['safety']:.2f}",
            "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]",
        )
    console.print(table)

    means = report.metric_means()
    console.print(
        f"\n[bold]通过率[/bold] {report.passed}/{report.total} "
        f"（{report.passed / report.total:.0%}）　"
        f"[bold]各维度均值[/bold] "
        + "　".join(f"{k}={v:.3f}" for k, v in means.items())
    )

    for result in report.results:
        if not result.passed:
            console.print(f"\n[red]{result.case_id}[/red]")
            for note in result.notes:
                console.print(f"  - {note}")

    if args.json:
        path = report.save(args.json)
        console.print(f"\n报告已写入 {path}")
    return 0 if report.passed == report.total else 1


# --------------------------------------------------------------------------- #
def cmd_tools(args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    cfg, scenario, persona, env, agent = _build(args)
    table = Table(title=f"{persona.name} 可用工具（场景：{scenario.get('name')}）", header_style="bold")
    table.add_column("工具", style="bold")
    table.add_column("参数")
    table.add_column("说明")
    for spec in agent.registry.specs(agent.id):
        kind = "[dim]内部[/dim]" if spec.internal else "世界"
        params = ", ".join(f"{k}: {v}" for k, v in spec.params.items()) or "-"
        table.add_row(f"{spec.name} ({kind})", params, spec.description)
    console.print(table)
    return 0


# --------------------------------------------------------------------------- #
def cmd_info(args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    scenarios = list_scenarios()
    console.print(Panel.fit("可用场景\n" + "\n".join(f"  - {s}" for s in scenarios), title="info"))
    for scenario_id in scenarios:
        scenario = load_scenario(scenario_id)
        persona = Persona.from_dict(load_persona(scenario.get("npc", "ayou")))
        console.print(
            f"\n[bold]{scenario_id}[/bold]　{scenario.get('name')}\n"
            f"  说明：{scenario.get('description')}\n"
            f"  NPC：{persona.name}（{persona.role}）\n"
            f"  玩家：{'、'.join(p['name'] for p in scenario.get('players', []))}\n"
            f"  目标：{'、'.join(o['goal'] for o in scenario.get('objectives', []))}"
        )
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="npc_agent", description="游戏 AI NPC 智能体框架"
    )
    parser.add_argument("--provider", help="LLM provider: null | openai-compat")
    parser.add_argument("--model", help="模型名，例如 Qwen3-8B-Instruct")
    parser.add_argument("--base-url", dest="base_url", help="OpenAI 兼容端点")
    parser.add_argument("--api-key", dest="api_key", help="API key")
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="跑一段可复现的演示对话")
    p_demo.add_argument("--scenario", default="icebreaker", choices=list_scenarios())
    p_demo.add_argument("--ticks", type=int, default=0, help="只跑前 N 轮")
    p_demo.set_defaults(func=cmd_demo)

    p_chat = sub.add_parser("chat", help="交互式对话")
    p_chat.add_argument("--scenario", default="tutorial", choices=list_scenarios())
    p_chat.set_defaults(func=cmd_chat)

    p_eval = sub.add_parser("eval", help="运行评测")
    p_eval.add_argument("--category", action="append", help="只跑某类用例，可重复")
    p_eval.add_argument("--json", help="把报告写入指定路径")
    p_eval.set_defaults(func=cmd_eval)

    p_tools = sub.add_parser("tools", help="列出当前场景的工具清单")
    p_tools.add_argument("--scenario", default="tutorial", choices=list_scenarios())
    p_tools.set_defaults(func=cmd_tools)

    p_info = sub.add_parser("info", help="查看场景与人设")
    p_info.set_defaults(func=cmd_info)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
