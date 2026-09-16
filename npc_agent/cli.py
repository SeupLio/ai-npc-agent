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
from pathlib import Path
from typing import Optional

from .cast import Cast, build_cast, load_cast
from .config import RuntimeConfig, list_scenarios, load_scenario
from .env import env_label, env_name_of
from .eval.harness import CASES_DIR
from .eval.judge import DEFAULT_JUDGE_CONCURRENCY, JUDGE_MAX_TOKENS
from .eval.judge import DEFAULT_RUBRICS as DEFAULT_RUBRIC_KEYS
from .eval.runner import DEFAULT_BACKOFF, DEFAULT_CONCURRENCY, DEFAULT_MAX_RETRIES
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
    from .eval.runner import Checkpoint, degraded_summary, render_batch_stats, run_cases

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
    # 两个开关把"模型负责台词"和"模型负责规划"拆成可独立测量的变量。
    # 先测便宜的那个（台词）：规划那一列不稳，混在一起测会污染台词那一列。
    if getattr(args, "no_planner", False):
        cfg.use_llm_planner = False
    if getattr(args, "no_speech", False):
        cfg.use_llm_speech = False

    categories = args.category or None
    harness = EvalHarness(cfg)
    cases = harness.load_cases(categories)
    limit = getattr(args, "limit", 0) or 0
    if limit:
        cases = cases[:limit]

    concurrency = max(1, getattr(args, "concurrency", DEFAULT_CONCURRENCY) or 1)
    report = EvalReport(
        config={
            "provider": cfg.llm_provider,
            "model": cfg.model or "(offline)",
            "memory_strategy": cfg.memory_strategy,
            "use_llm_planner": cfg.use_llm_planner,
            "use_llm_speech": cfg.use_llm_speech,
        }
    )

    # 进度回调：并发下完成顺序是乱的，所以只报"完成了几条"，不报"第几条"。
    def on_done(run, done: int, total: int) -> None:
        flag = ""
        if run.degraded:
            flag = " [yellow](降级)[/yellow]"
        elif not run.ok:
            flag = " [red](故障)[/red]"
        console.print(f"  [{done}/{total}] {run.case_id}{flag}", highlight=False)

    checkpoint = None
    ckpt_path = getattr(args, "checkpoint", "") or ""
    runs_holder: list = []
    if ckpt_path:
        checkpoint = Checkpoint(
            ckpt_path,
            lambda: {
                "done": len(runs_holder),
                "total": len(cases),
                "runs": [r.to_dict() for r in runs_holder],
            },
        )

    runs, stats = run_cases(
        cases,
        cfg,
        concurrency=concurrency,
        max_retries=getattr(args, "retries", DEFAULT_MAX_RETRIES),
        backoff=getattr(args, "backoff", DEFAULT_BACKOFF),
        cases_dir=harness.cases_dir,
        on_done=on_done if getattr(args, "progress", False) else None,
        checkpoint=checkpoint,
    )
    runs_holder.extend(runs)

    for run in runs:
        if run.result is not None:
            report.results.append(run.result)
        else:
            # 基础设施故障的用例没有 CaseResult，不能混进通过率 ——
            # 把它算成"用例失败"会让评测报告替代码 bug 背锅。
            console.print(f"[red]{run.case_id} 没跑完[/red]：{run.error}")

    report.batch = {"stats": stats.to_dict(), "degraded": degraded_summary(runs)}

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

    if report.total:
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

    # 跑批口径单独说清楚 —— 这一段的重点是让"降级"无法伪装成"答得好"。
    console.print(f"\n[bold]{render_batch_stats(stats)}[/bold]")
    console.print(report.batch["degraded"]["verdict"])

    if args.json:
        path = report.save(args.json)
        console.print(f"\n报告已写入 {path}")

    # 退出码：有故障或降级超 10% 都不算干净通过。
    # 只按通过率给退出码，CI 就会在"端点挂了但用例全绿"时放行。
    if stats.failed:
        return 2
    if report.total and report.passed != report.total:
        return 1
    if report.batch["degraded"]["degraded"] > report.batch["degraded"]["total"] * 0.10:
        return 3
    return 0


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
def cmd_gencases(args: argparse.Namespace) -> int:
    """生成用例集：结构性指纹去重 → 离线可达性门禁 → 落盘。

    产出三样东西，缺一不可：
      cases/generated.jsonl    回归集（离线基线全绿的那部分）
      reports/blindspots.jsonl 基线盲区（结构正确、离线做不到）
      reports/generation.json  门禁报告（剔了谁、为什么）
    """
    from rich.console import Console

    from .eval.generator import (
        GENERATED_FILE,
        build_case_set,
        render_coverage,
        write_blindspots,
        write_cases,
        write_gate_report,
    )

    console = Console()
    result = build_case_set(
        target=args.target,
        seed=args.seed,
        verbose=not args.quiet,
    )
    cases = result["cases"]
    gate = result["gate"]

    cases_path = write_cases(cases, CASES_DIR / GENERATED_FILE, seed=args.seed)
    blind = write_blindspots(
        gate.dropped, Path(args.reports_dir) / "blindspots.jsonl"
    )
    report_path = write_gate_report(
        gate, Path(args.reports_dir) / "generation.json", target=args.target, seed=args.seed
    )

    console.print(render_coverage(result["coverage"], gate.summary()))
    console.print(f"\n回归集 → {cases_path}")
    if blind:
        console.print(f"基线盲区 → {blind}")
    console.print(f"门禁报告 → {report_path}")
    if gate.dropped:
        console.print(
            "\n[yellow]被剔除的用例不是坏用例[/yellow]：结构正确，但确定性离线路径做不到。"
            "\n它们已单独落盘，模型跑批时可以加回来衡量「换模型多做到了什么」。"
        )
    return 0


# --------------------------------------------------------------------------- #
def cmd_judge(args: argparse.Namespace) -> int:
    """LLM-as-judge：先校准裁判，再（可选）用它判一份跑批报告。

    顺序不能反。**没校准就判分，得到的只是"另一个模型的意见"** ——
    把它写进结论是拿权威感代替证据。所以 `--calibrate` 是默认动作，
    而且校准结果（一致率 / kappa）会先打出来。
    """
    import json as _json

    from rich.console import Console
    from rich.table import Table

    from .eval.judge import (
        DEFAULT_JUDGE_CONCURRENCY,
        DEFAULT_RUBRICS,
        RUBRICS,
        LLMJudge,
        calibrate,
        judge_report_cases,
        load_calibration,
        render_calibration,
    )

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

    llm = build_llm(cfg.llm_provider, model=cfg.model, base_url=cfg.base_url, api_key=cfg.api_key)
    rubrics = [r.strip() for r in (args.rubrics or "").split(",") if r.strip()] or list(DEFAULT_RUBRICS)
    unknown = [r for r in rubrics if r not in RUBRICS]
    if unknown:
        console.print(f"[red]未知的评判标准 {unknown}，可选：{sorted(RUBRICS)}[/red]")
        return 2

    judge = LLMJudge(
        llm,
        rubrics=rubrics,
        max_tokens=getattr(args, "judge_max_tokens", 0) or JUDGE_MAX_TOKENS,
        name=cfg.model or "judge",
    )

    if not judge.available:
        console.print(
            "[yellow]模型不可用 —— 裁判不会给出任何分数。[/yellow]\n"
            "这不是「判了 0 分」：未判就是未判，它不会混进任何均值。\n"
            "指定模型后重跑：--provider openai-compat --base-url ... --model ..."
        )
        return 1

    payload: dict[str, object] = {"judge_model": cfg.model or cfg.llm_provider}

    # ---- 1) 校准（永远先做） ----
    items = load_calibration()
    console.print(f"用 {len(items)} 条人工标注样本校准裁判 …")
    report = calibrate(judge, items, progress=lambda m: console.print(f"[dim]{m}[/dim]"))
    console.print(render_calibration(report))

    table = Table(title="裁判校准", header_style="bold")
    table.add_column("评判标准")
    table.add_column("样本", justify="right")
    table.add_column("一致率", justify="right")
    table.add_column("kappa", justify="right")
    table.add_column("结论")
    for key, stats in sorted(report.per_rubric.items()):
        table.add_row(
            f"{RUBRICS[key].name}",
            str(stats["n"]),
            f"{stats['agreement']:.0%}",
            f"{stats['kappa']:.2f}",
            stats["reading"],
        )
    console.print(table)
    payload["calibration"] = report.to_dict()

    # ---- 2) 可选：判一份跑批报告 ----
    if args.report:
        report_path = Path(args.report)
        data = _json.loads(report_path.read_text(encoding="utf-8"))
        results = data.get("results") or []
        limit_cases = getattr(args, "limit_cases", 0) or 0
        if limit_cases:
            results = results[:limit_cases]
        concurrency = max(1, getattr(args, "concurrency", DEFAULT_JUDGE_CONCURRENCY) or 1)
        console.print(
            f"\n对 {len(results)} 条用例的台词判分（{report_path}，并发 {concurrency}）…"
        )

        # 事后判分能拿到的东西是有限的：转写里有完整对话，但**现场状态已经过去了**。
        # 所以默认只跑不依赖现场的两条标准。
        if "grounded" in rubrics:
            console.print(
                "[yellow]注意：「事实一致」需要当时的现场状态，"
                "事后判分只能拿到场景的初始配置。[/yellow]\n"
                "        这一项的结果只能当参考信号，不能当结论。"
            )

        # 人设块和现场块按场景缓存：228 条用例只涉及 5 个场景，
        # 每条都重新渲染一遍纯属浪费，而且 load_persona 会反复读盘。
        persona_cache: dict[str, str] = {}
        scene_cache: dict[str, str] = {}

        def persona_of(sid: str) -> str:
            if sid not in persona_cache:
                persona_cache[sid] = _persona_block(load_scenario(sid))
            return persona_cache[sid]

        def scene_of(sid: str) -> str:
            if sid not in scene_cache:
                scene_cache[sid] = _scene_block(load_scenario(sid), sid)
            return scene_cache[sid]

        def on_judged(judgement, done: int, total: int) -> None:
            flag = " [red](炸了)[/red]" if not judgement.ok else ""
            console.print(f"  [{done}/{total}] {judgement.case_id}{flag}", highlight=False)

        judgements, coverage = judge_report_cases(
            results,
            judge=judge,
            persona_of=persona_of,
            scene_of=scene_of,
            concurrency=concurrency,
            on_done=on_judged if getattr(args, "progress", False) else None,
        )

        per_case = [j.to_dict() for j in judgements if j.pairs]
        summary = _summarise_judgements(judgements, rubrics)
        payload["report"] = report_path.name
        payload["cases"] = per_case
        payload["summary"] = summary
        payload["coverage"] = coverage

        console.print(
            f"\n判出 {summary['judged']} 条｜未判 {summary['unjudged']}"
            f"（未判不会被当成 0 分）"
        )
        console.print(coverage["verdict"])
        for key, stats in sorted(summary["by_rubric"].items()):
            console.print(
                f"  {RUBRICS[key].name}: 通过率 {stats['pass_rate']:.0%}"
                f"（{stats['passed']}/{stats['n']}）"
            )

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"\n报告 → {out}")
    return 0


def _persona_block(scenario: dict) -> str:
    """从场景配置里取 NPC 的人设卡，渲染成裁判能读的一段话。"""
    from .config import load_persona
    from .modules.persona import Persona

    npc = scenario.get("npc")
    if not npc:
        cast = scenario.get("npcs") or []
        npc = cast[0].get("persona") if cast else None
    if not npc:
        return ""
    try:
        return Persona.from_dict(load_persona(npc)).system_block()
    except FileNotFoundError:
        return ""


def _scene_block(scenario: dict, scenario_id: str) -> str:
    """场景的**静态**配置（物品在哪、有哪些地点）。

    刻意在函数名和注释里说清楚这是"初始配置"：事后判分拿不到当时的现场，
    用它去判「事实一致」会把"世界已经变了"误判成"NPC 在编造"。
    """
    world = scenario.get("world") or {}
    items = world.get("items") or {}
    where: dict[str, list[str]] = {}
    for item, spec in items.items():
        where.setdefault((spec or {}).get("loc", "?"), []).append(item)
    parts = [f"场景「{scenario.get('name', scenario_id)}」（这是**初始**配置，不是当时的状态）"]
    for loc, names in sorted(where.items()):
        parts.append(f"  {loc}: {'、'.join(sorted(names))}")
    pois = scenario.get("pois") or {}
    if pois:
        parts.append("  地点: " + "、".join(f"{k}({v.get('name', k)})" for k, v in sorted(pois.items())))
    return "\n".join(parts)


def _summarise_judgements(judgements: list, rubrics: list[str]) -> dict:
    """汇总判决。**未判的条目单独计数，绝不并进通过率的分母。**

    直接吃 `CaseJudgement` 对象而不是它的 dict 形式：
    走一遍 `to_dict()` 再读回来，等于把"判分结果长什么样"这件事
    在判分模块和 CLI 之间抄了两遍，两边一旦不一致就会静默算错。
    """
    by_rubric: dict[str, dict[str, int]] = {key: {"n": 0, "passed": 0} for key in rubrics}
    judged = unjudged = 0
    for judgement in judgements:
        for entry in judgement.pairs:
            for verdict in entry.get("verdicts") or []:
                bucket = by_rubric.setdefault(
                    verdict["rubric"], {"n": 0, "passed": 0}
                )
                if not verdict["judged"]:
                    unjudged += 1
                    continue
                judged += 1
                bucket["n"] += 1
                if verdict["score"] >= 0.99:
                    bucket["passed"] += 1
    for stats in by_rubric.values():
        stats["pass_rate"] = round(stats["passed"] / stats["n"], 3) if stats["n"] else 0.0
    return {"judged": judged, "unjudged": unjudged, "by_rubric": by_rubric}


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
def _add_llm_args(p: argparse.ArgumentParser) -> None:
    """把模型相关参数挂到子命令上。

    用 `default=argparse.SUPPRESS` 而不是 `None` / `""`：argparse 的子解析器
    会把**自己所有参数**的默认值写回命名空间，所以 `default=None` 会覆盖掉
    用户在顶层写的 `--provider X` —— 变成"顶层明明写了却没生效"这种最难查的 bug。
    SUPPRESS 的语义正是"没写就别动它"。
    """
    p.add_argument("--provider", default=argparse.SUPPRESS, help="LLM provider: null | openai-compat")
    p.add_argument("--model", default=argparse.SUPPRESS, help="模型名，例如 kimi-k2.7-code")
    p.add_argument("--base-url", dest="base_url", default=argparse.SUPPRESS, help="OpenAI 兼容端点")
    p.add_argument("--api-key", dest="api_key", default=argparse.SUPPRESS, help="API key")


def _add_batch_args(p: argparse.ArgumentParser) -> None:
    """跑批相关的参数（并发 / 重试 / 检查点）。

    默认值故意保守：端点通常有并发上限，调太高会变成重试风暴，
    总耗时反而更长，而且降级比例会上升。
    """
    p.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"并发数（默认 {DEFAULT_CONCURRENCY}；1 = 串行，用于给并行结果做基线）",
    )
    p.add_argument(
        "--retries", type=int, default=DEFAULT_MAX_RETRIES, help="单条用例的重试次数（不含首发）"
    )
    p.add_argument("--backoff", type=float, default=DEFAULT_BACKOFF, help="退避基数（秒），指数增长")
    p.add_argument(
        "--checkpoint",
        default="",
        help="边跑边把进度写到这个文件；长跑（真实模型）建议打开。空串 = 不写",
    )
    p.add_argument("--progress", action="store_true", help="逐条打印进度")


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
    p_eval.add_argument(
        "--no-planner",
        action="store_true",
        help="让模型只负责台词，规划仍走启发式（每次调用省 ~35s，且只变一个变量）",
    )
    p_eval.add_argument("--no-speech", action="store_true", help="只用模型规划，台词仍走模板")
    _add_llm_args(p_eval)
    _add_batch_args(p_eval)
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

    p_gen = sub.add_parser("gencases", help="生成用例集（指纹去重 + 离线可达性门禁）")
    p_gen.add_argument("--target", type=int, default=240, help="用例数上限（不是配额，实际由结构数决定）")
    p_gen.add_argument("--seed", type=int, default=20260916, help="随机种子；固定种子 → 同一批用例")
    p_gen.add_argument("--reports-dir", default="reports", help="门禁报告与盲区文件的目录")
    p_gen.add_argument("--quiet", action="store_true", help="不打印门禁进度")
    p_gen.set_defaults(func=cmd_gencases)

    p_judge = sub.add_parser("judge", help="LLM-as-judge：先校准裁判，再判跑批报告")
    p_judge.add_argument("--report", default="", help="要判分的跑批报告 JSON（eval --json 的产物）")
    p_judge.add_argument("--rubrics", default="", help=f"逗号分隔，可选：{','.join(sorted(DEFAULT_RUBRIC_KEYS))}")
    p_judge.add_argument("--json", default="reports/judge.json", help="裁判报告输出路径，空串则不写")
    p_judge.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_JUDGE_CONCURRENCY,
        help=f"判分并发（默认 {DEFAULT_JUDGE_CONCURRENCY}）；串行判 228 条约十小时",
    )
    p_judge.add_argument("--limit-cases", dest="limit_cases", type=int, default=0, help="只判前 N 条用例")
    p_judge.add_argument(
        "--judge-max-tokens",
        dest="judge_max_tokens",
        type=int,
        default=0,
        help=f"裁判的输出预算（默认 {JUDGE_MAX_TOKENS}）。推理模型要给足，否则思维链会把预算吃光、返回空内容",
    )
    p_judge.add_argument("--progress", action="store_true", help="逐条打印判分进度")
    _add_llm_args(p_judge)
    p_judge.set_defaults(func=cmd_judge)

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
