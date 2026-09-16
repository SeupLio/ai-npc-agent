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

from .cast import Cast, build_cast, load_cast
from .config import RuntimeConfig, list_scenarios, load_scenario
from .env import env_label, env_name_of
from .llm import build_llm

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
    # 双 NPC：前两轮分别由两位客人起话头，让两个 NPC 都有机会开工；
    # 后面留空轮给阿柚把饮品做完 —— 她的计划比小舟长，需要更多轮。
    "duet": [
        ("player_a", "今天这里挺热闹的。"),
        ("player_b", "露台那边好像有风。"),
        None,
        None,
        None,
    ],
    # 体素世界：同一条制作链（砍木头 → 木板 → 木棍 → 采煤 → 火把 → 插在洞口）
    # 需要 11 个世界动作，所以留足空轮。开头那句是任务触发点。
    "village": [
        ("player_a", "阿岩，天快黑了，洞口得点个火把。"),
        None, None, None, None, None, None, None, None, None, None, None, None,
    ],
}


# --------------------------------------------------------------------------- #
def _build(args: argparse.Namespace):
    """造出场景 + 剧组。

    场景里写 npc: 还是 npcs: 对这里是同一件事 —— 单 NPC 只是"剧组只有一个人"，
    所以 demo / chat / tools / info 全都不用分叉。
    """
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

    scenario = load_scenario(getattr(args, "scenario", "tutorial"))
    llm = build_llm(
        cfg.llm_provider, model=cfg.model, base_url=cfg.base_url, api_key=cfg.api_key
    )
    return cfg, scenario, build_cast(scenario, llm, cfg)


def _roster(scenario, cast: Cast) -> str:
    return "、".join(
        f"[bold]{agent.persona.name}[/bold]（{agent.persona.role}）"
        for agent in cast.agents.values()
    )


# --------------------------------------------------------------------------- #
def cmd_demo(args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    cfg, scenario, cast = _build(args)
    env = cast.env

    mode = "在线模型" if cast.lead.llm.available else "离线启发式（未配置模型）"
    console.print(
        Panel.fit(
            f"[bold]{scenario.get('name')}[/bold] · {scenario.get('description','')}\n"
            f"世界：{env_label(env.name)}\n"
            f"NPC：{_roster(scenario, cast)}\n"
            f"推理模式：{mode}",
            title=env_label(env.name),
            border_style="blue",
        )
    )

    script = DEMO_SCRIPTS.get(args.scenario, DEMO_SCRIPTS["tutorial"])
    if args.ticks:
        script = script[: args.ticks]

    for index, line in enumerate(script):
        console.print(f"\n[dim]── 第 {index + 1} 轮 ──[/dim]")
        utterance = None
        if line:
            player_id, text = line
            utterance = env.record_player_utterance(player_id, text)
            console.print(f"[cyan]玩家 {utterance.speaker_name}[/cyan]：{text}")

        # 一轮 = 一个 tick。剧组里所有 NPC 依次行动，最多一个人开口。
        for turn in cast.step(utterance):
            name = cast.name_of(turn.actor_id)
            if turn.decision_reason:
                console.print(f"  [dim]（{name}：{turn.decision_reason}）[/dim]")
            for action, result in zip(turn.actions, turn.results):
                if action.tool == "speak":
                    continue
                style = "green" if result.ok else "red"
                console.print(f"  [{style}]▸ {name} {action.render()}[/{style}]")
                if not result.ok:
                    console.print(f"      [red]✗ {result.detail}[/red]")
            if turn.say:
                console.print(f"  [yellow]{name}[/yellow]：{turn.say}")

    snapshot = env.snapshot()
    table = Table(title="结束时世界状态", show_header=True, header_style="bold")
    table.add_column("项目")
    table.add_column("值")
    table.add_row("世界标记", ", ".join(snapshot["world_flags"]) or "（无）")
    table.add_row(
        "目标",
        ", ".join(f"{k}={v}" for k, v in snapshot["objectives"].items()) or "（无）",
    )
    for pid, agent in cast.agents.items():
        stats = agent.memory.store.stats()
        table.add_row(
            f"{agent.persona.name} · 记忆",
            f"episodic={stats.episodic} semantic={stats.semantic} "
            f"reflection={stats.reflection} 巩固={stats.consolidated}",
        )
        table.add_row(f"{agent.persona.name} · 发言占比", f"{agent.state.npc_share():.0%}")
    if cast.is_multi_npc:
        table.add_row("发言调度", _turn_taking_note(env, cast))
    console.print()
    console.print(table)

    lessons = [a for a in cast.agents.values() if a.reflector.lessons]
    if lessons:
        console.print("\n[bold]沉淀下来的教训[/bold]")
        for agent in lessons:
            console.print(f"[bold]{agent.persona.name}[/bold]")
            console.print(agent.reflector.render_lessons())
    return 0


def _turn_taking_note(env, cast: Cast) -> str:
    """同轮抢话的自检。这是多 NPC 最该盯的一个数。"""
    npc_ids = set(cast.npc_ids)
    collisions = [
        tick
        for tick, speakers in env.speakers_by_tick().items()
        if len({s for s in speakers if s in npc_ids}) > 1
    ]
    if collisions:
        return f"[red]{len(collisions)} 轮有 NPC 抢话（{collisions}）[/red]"
    return f"[green]0 轮抢话[/green]，发言次数 {env.speech_counts()}"


# --------------------------------------------------------------------------- #
def cmd_chat(args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    cfg, scenario, cast = _build(args)
    env = cast.env
    players = [p["id"] for p in scenario.get("players", [])]
    current = players[0] if players else "player_a"

    console.print(
        Panel(
            f"正在和 {_roster(scenario, cast)} 对话（场景：{scenario.get('name')}）。\n"
            f"当前身份：{env.actors[current].name}　切换玩家：/as player_b　退出：/quit",
            border_style="blue",
        )
    )
    if not cast.lead.llm.available:
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
        for turn in cast.step(utterance):
            name = cast.name_of(turn.actor_id)
            for action, result in zip(turn.actions, turn.results):
                if action.tool == "speak":
                    continue
                style = "green" if result.ok else "red"
                console.print(f"  [{style}]▸ {name} {action.render()}[/{style}]")
            if turn.say:
                console.print(f"[yellow]{name}[/yellow]：{turn.say}")
            elif cast.is_multi_npc and turn.acted:
                console.print(f"[dim]{name} 没有说话，但在忙自己的事[/dim]")
    return 0


# --------------------------------------------------------------------------- #
def cmd_eval(args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.table import Table

    from .eval import EvalHarness, EvalReport

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
    harness = EvalHarness(cfg)
    cases = harness.load_cases(categories)
    limit = getattr(args, "limit", 0) or 0
    if limit:
        cases = cases[:limit]

    report = EvalReport(
        config={
            "provider": cfg.llm_provider,
            "model": cfg.model or "(offline)",
            "memory_strategy": cfg.memory_strategy,
        }
    )
    for case in cases:
        report.results.append(harness.run_case(case))

    table = Table(title="评测结果", header_style="bold")
    for column in ("用例", "场景", "任务", "工具", "记忆", "人设", "安全", "调度", "结论"):
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
            f"{scores['turn_taking']:.2f}",
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
def _render_comparison(console, comparison) -> None:
    """把对照结果渲染成两张表：绝对值 + 相对基线差值。"""
    from rich.table import Table

    table = Table(title="对照跑批 · 绝对值", header_style="bold")
    for column in ("配置", "模型", "记忆策略", "通过", "任务", "工具", "记忆", "人设", "安全",
                   "调度", "自由台词", "均长", "耗时"):
        table.add_column(column, justify="left" if column in ("配置", "模型", "记忆策略") else "center")

    for row in comparison.rows():
        table.add_row(
            row["label"],
            row["model"],
            row["strategy"],
            row["pass"],
            f"{row['task']:.3f}",
            f"{row['tools']:.3f}",
            f"{row['memory']:.3f}",
            f"{row['persona']:.3f}",
            f"{row['safety']:.3f}",
            f"{row['turn_taking']:.3f}",
            f"{row['free']:.0%}",
            f"{row['chars']:.0f}",
            f"{row['sec']:.0f}s",
        )
    console.print(table)

    deltas = comparison.deltas()
    if not deltas:
        return

    diff = Table(title="对照跑批 · 相对首行差值（正数=更好）", header_style="bold")
    diff.add_column("配置", justify="left")
    diff.add_column("vs", justify="left")
    for column in ("通过率", "任务", "工具", "记忆", "人设", "安全", "调度", "自由台词"):
        diff.add_column(column, justify="center")

    def cell(value: float) -> str:
        if abs(value) < 0.0005:
            return "[dim]0.000[/dim]"
        color = "green" if value > 0 else "red"
        return f"[{color}]{value:+.3f}[/{color}]"

    for row in deltas:
        diff.add_row(
            row["label"],
            row["vs"],
            cell(row["pass_rate"]),
            cell(row["task"]),
            cell(row["tools"]),
            cell(row["memory"]),
            cell(row["persona"]),
            cell(row["safety"]),
            cell(row["turn_taking"]),
            cell(row["free"]),
        )
    console.print()
    console.print(diff)


def _base_config_from_args(args: argparse.Namespace) -> RuntimeConfig:
    cfg = RuntimeConfig.from_env()
    for attr, field in (
        ("provider", "llm_provider"),
        ("model", "model"),
        ("base_url", "base_url"),
        ("api_key", "api_key"),
    ):
        value = getattr(args, attr, None)
        if value:
            setattr(cfg, field, value)
    return cfg


def cmd_compare(args: argparse.Namespace) -> int:
    """离线启发式 vs 真实模型：证明模型带来的不是"能跑"，而是"会说"。"""
    from rich.console import Console

    from .eval import Comparison, RunSpec

    console = Console(width=150)
    cfg = _base_config_from_args(args)

    specs = [
        RunSpec(label="离线启发式", provider="null", model="", memory_strategy=cfg.memory_strategy)
    ]
    provider = cfg.llm_provider if cfg.llm_provider != "null" else "openai-compat"
    models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    if not models and cfg.model:
        models = [cfg.model]
    for model in models:
        roles = []
        if args.no_planner:
            roles.append("启发式规划")
        if args.no_speech:
            roles.append("模板台词")
        suffix = f"（{'+'.join(roles)}）" if roles else ""
        specs.append(
            RunSpec(
                label=f"模型·{model}{suffix}",
                provider=provider,
                model=model,
                memory_strategy=cfg.memory_strategy,
                temperature=args.temperature,
                use_llm_planner=not args.no_planner,
                use_llm_speech=not args.no_speech,
            )
        )

    if len(specs) == 1:
        console.print(
            "[yellow]没有指定模型，只跑离线基线。"
            "用 --models kimi-k2.7-code 或设置 NPC_AGENT_MODEL 来加一列真实模型。[/yellow]"
        )
    if not cfg.api_key and len(specs) > 1:
        console.print("[red]缺少 API key（NPC_AGENT_API_KEY 或 --api-key），真实模型这一列会全部失败。[/red]")

    def _on_case(spec, case, index, total, outcome):
        console.print(
            f"[dim]    · [{index}/{total}] {case.get('id')} "
            f"→ {outcome.report.passed}/{outcome.report.total} 通过[/dim]"
        )

    comparison = Comparison(
        base_config=cfg,
        categories=args.category or None,
        limit=args.limit or 0,
    ).run(
        specs,
        progress=lambda msg: console.print(f"[dim]{msg}[/dim]"),
        checkpoint=args.json or None,
        on_case=_on_case,
    )

    _render_comparison(console, comparison)

    for outcome in comparison.outcomes:
        for failure in outcome.to_dict()["failures"]:
            console.print(f"\n[red]{outcome.spec.label} / {failure['case_id']}[/red]")
            for note in failure["notes"]:
                console.print(f"  - {note}")

    if args.json:
        path = comparison.save(args.json)
        console.print(f"\n对照报告已写入 {path}")

    if getattr(args, "html", ""):
        from .eval.report import write_comparison_html

        page = write_comparison_html(
            comparison.to_dict(),
            args.html,
            title="离线启发式 vs 真实模型",
            subtitle=(
                "同一套用例、同一套指标、只换推理后端。"
                "重点看「自由台词」这一列 —— 它衡量台词是人写的还是模型写的。"
            ),
            temperature=args.temperature,
        )
        console.print(f"HTML 报告已写入 {page}")
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    """记忆策略消融：量化"记忆到底贡献了多少"。"""
    from rich.console import Console

    from .eval import Comparison, RunSpec
    from .modules.retrieval import available_strategies

    console = Console(width=150)
    cfg = _base_config_from_args(args)

    wanted = [s.strip() for s in (args.strategies or "").split(",") if s.strip()]
    known = available_strategies()
    unknown = [s for s in wanted if s not in known]
    if unknown:
        console.print(f"[red]未知的记忆策略：{', '.join(unknown)}　可用：{', '.join(known)}[/red]")
        return 2

    use_model = bool(args.with_model) and bool(cfg.model)
    provider = cfg.llm_provider if cfg.llm_provider != "null" else ("openai-compat" if use_model else "null")

    specs = [
        RunSpec(
            label=f"记忆·{strategy}",
            provider=provider,
            model=cfg.model if use_model else "",
            memory_strategy=strategy,
            temperature=args.temperature,
        )
        for strategy in wanted
    ]

    if not use_model:
        console.print(
            "[dim]离线消融（确定性、可复现）。加 --with-model 可在真实模型上做同样的消融，但会慢很多。[/dim]"
        )
    console.print(
        "[dim]说明：策略 none = 不注入任何记忆，即「裸模型」基线；"
        "它与 hybrid 的差值就是记忆的净收益。[/dim]"
    )

    comparison = Comparison(
        base_config=cfg,
        categories=args.category or None,
        limit=args.limit or 0,
    ).run(
        specs,
        progress=lambda msg: console.print(f"[dim]{msg}[/dim]"),
        checkpoint=args.json or None,
    )

    _render_comparison(console, comparison)

    if args.json:
        path = comparison.save(args.json)
        console.print(f"\n消融报告已写入 {path}")

    if getattr(args, "html", ""):
        from .eval.report import write_comparison_html

        page = write_comparison_html(
            comparison.to_dict(),
            args.html,
            title="记忆策略消融实验",
            subtitle=(
                "五种检索策略在同一批用例上的表现。"
                "none 是「裸模型」基线，它与 hybrid 的差值就是记忆系统的净收益。"
            ),
            temperature=args.temperature,
        )
        console.print(f"HTML 报告已写入 {page}")
    return 0



# --------------------------------------------------------------------------- #
def cmd_worlds(args: argparse.Namespace) -> int:
    """跨世界覆盖报告：同一套 Agent 跑在几个世界上。

    刻意**不叫** compare —— 它不是对照实验。两个世界的用例集不同，
    分数不可相减；它回答的是"覆盖面"，不是"哪个更好"。
    把这两种报告混在一起，是"看起来专业、其实在拿苹果比橘子"最常见的来源。
    """
    from rich.console import Console
    from rich.table import Table

    from .eval.worlds import METRIC_LABELS, run_worlds, save_worlds_json, write_worlds_html

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

    runs = run_worlds(cfg, progress=lambda m: console.print(f"[dim]{m}[/dim]"))

    table = Table(title="跨世界覆盖报告（不是对照实验）", header_style="bold")
    table.add_column("世界")
    table.add_column("环境", style="dim")
    table.add_column("通过", justify="right")
    table.add_column("通过率", justify="right")
    for label in METRIC_LABELS.values():
        table.add_column(label, justify="right")
    for run in runs:
        means = run.means
        table.add_row(
            run.spec.label,
            run.spec.env,
            f"{run.report.passed}/{run.report.total}",
            f"{run.pass_rate:.0%}",
            *[f"{means.get(k, 0.0):.2f}" for k in METRIC_LABELS],
        )
    console.print()
    console.print(table)
    console.print(
        "\n[dim]两个世界跑的用例集不同，所以没有差值表、分数也不该相减。[/dim]"
        "\n[dim]受控对照见 compare / ablate。[/dim]"
    )

    if args.json:
        path = save_worlds_json(runs, args.json)
        console.print(f"\n覆盖报告已写入 {path}")
    if getattr(args, "html", ""):
        page = write_worlds_html(runs, args.html)
        console.print(f"HTML 报告已写入 {page}")
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    cfg, scenario, cast = _build(args)
    lead = cast.lead
    title = f"{lead.persona.name} 可用工具（场景：{scenario.get('name')}）"
    if cast.is_multi_npc:
        title = f"剧组共用工具（场景：{scenario.get('name')}）"
    table = Table(title=title, header_style="bold")
    table.add_column("工具", style="bold")
    table.add_column("参数")
    table.add_column("说明")
    for spec in lead.registry.specs(lead.id):
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
        cast = load_cast(scenario)
        roster = "、".join(f"{p.name}（{p.role}）" for p in cast) or "（未配置）"
        console.print(
            f"\n[bold]{scenario_id}[/bold]　{scenario.get('name')}"
            f"　[dim]世界：{env_label(env_name_of(scenario))}[/dim]\n"
            f"  说明：{scenario.get('description')}\n"
            f"  NPC：{roster}{'　[dim]（多 NPC）[/dim]' if len(cast) > 1 else ''}\n"
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
    p_eval.add_argument("--limit", type=int, default=0, help="只跑前 N 条用例（冒烟用）")
    p_eval.add_argument("--json", help="把报告写入指定路径")
    p_eval.set_defaults(func=cmd_eval)

    p_cmp = sub.add_parser("compare", help="离线启发式 vs 真实模型 对照跑批")
    p_cmp.add_argument("--models", default="", help="逗号分隔的模型名；留空则只跑离线基线")
    p_cmp.add_argument("--temperature", type=float, default=0.3, help="评测建议低温以保证可复现")
    p_cmp.add_argument("--category", action="append", help="只跑某类用例，可重复")
    p_cmp.add_argument("--limit", type=int, default=0, help="只跑前 N 条用例（冒烟用）")
    p_cmp.add_argument(
        "--no-planner",
        action="store_true",
        help="让模型只负责台词，规划仍走启发式（每次调用省 ~35s，且只变一个变量）",
    )
    p_cmp.add_argument(
        "--no-speech", action="store_true", help="只用模型规划，台词仍走模板"
    )
    p_cmp.add_argument("--json", default="reports/comparison.json", help="报告输出路径")
    p_cmp.add_argument("--html", default="reports/comparison.html", help="HTML 报告路径，空串则不生成")
    p_cmp.set_defaults(func=cmd_compare)

    p_abl = sub.add_parser("ablate", help="记忆策略消融实验")
    p_abl.add_argument(
        "--strategies",
        default="hybrid,recency,lexical,importance,none",
        help="逗号分隔；none 为裸模型基线",
    )
    p_abl.add_argument("--with-model", action="store_true", help="在真实模型上做消融（慢）")
    p_abl.add_argument("--temperature", type=float, default=0.3)
    p_abl.add_argument("--category", action="append", help="只跑某类用例，可重复")
    p_abl.add_argument("--limit", type=int, default=0, help="只跑前 N 条用例（冒烟用）")
    p_abl.add_argument("--json", default="reports/ablation.json", help="报告输出路径")
    p_abl.add_argument("--html", default="reports/ablation.html", help="HTML 报告路径，空串则不生成")
    p_abl.set_defaults(func=cmd_ablate)

    p_worlds = sub.add_parser("worlds", help="跨世界覆盖报告：同一套 Agent 跑在几个世界上")
    p_worlds.add_argument("--json", default="reports/worlds.json", help="报告输出路径")
    p_worlds.add_argument("--html", default="reports/worlds.html", help="HTML 报告路径，空串则不生成")
    p_worlds.set_defaults(func=cmd_worlds)

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
