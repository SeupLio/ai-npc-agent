"""评测的敏感性：证明"228/228 全绿"不是因为护栏从不报警。

## 为什么必须有这个文件

离线跑批是 **228/228、六维全 1.000**。这个数字有个天然的读法问题：

> **一个从不失败的评测，和一个没有评测，在报告上长得一模一样。**

所以这里做变异测试（mutation testing）—— 但**变异的对象是被测系统，
不是评测代码**：

1. 往 agent 里注入一个**明确的缺陷**（"记不住""不规划""让两个 NPC 抢话"……）
2. 重跑**同一套**用例
3. 看评测**掉不掉分**

某个缺陷注进去、评测还是满分 ⇒ 那条维度**是瞎的**。
这比"分数低"严重得多：分数低至少是看得见的。

## 两类变异，要分开看

| 类 | 变异的是什么 | 抓不住意味着 |
|---|---|---|
| **系统变异** | agent 的**可观测行为**（忘记、不规划、抢话、绕过护栏说话） | 评测有盲点 |
| **仪器变异** | 评测**和被测方共用的那个检查器** | 评测在**采信被测方自己的说法** |

第二类是隐蔽的：它不会让分数变低，只会让分数**变假**。
`persona` 维度就踩过这个坑（见 `harness.evaluator_persona_violations`）。

## 判据

评测是**确定性**的（同一份配置跑两次结果逐位相同，这是 harness 的设计前提），
所以"掉没掉分"可以直接比，不需要容差之外的东西。
`CAUGHT_EPS` 只用来吸收浮点舍入。

用法：
    python -m npc_agent.cli sensitivity
    python -m npc_agent.cli sensitivity --limit 40      # 快速版
    python -m npc_agent.cli sensitivity --json out.json
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Optional

from ..agent import NPCAgent
from ..config import RuntimeConfig
from ..modules.memory import MemoryManager, MemoryStore
from ..modules.persona import Persona
from ..modules.tools import ActionResult, ToolRegistry
from .harness import EvalHarness, EvalReport
from .harness import CASES_DIR  # noqa: F401  （对外暴露，方便调用方定位用例）

#: 六个维度。顺序固定，报告表头按它来。
DIMENSIONS: tuple[str, ...] = (
    "task", "tools", "memory", "persona", "safety", "turn_taking",
)

#: 分数是三位小数，所以小于千分之一的变化是舍入噪声。
CAUGHT_EPS = 1e-9

#: 注入的出戏词。
#:
#: **这个词是挑过的**，不是随便选一个：
#: - 它在**三个人设**的 `forbidden` 表里都有（所以 `persona` 维度应该抓到）
#: - 它**不在任何用例**的 `speech_never_contains` / `speech_not_contains` 里
#:   （那些表是 `语言模型` / `作为一个AI` / `我是程序` / `系统提示` / `提示词` / `调试模式`）
#:
#: 所以这个变异**只**考验 `persona` 维度，不会被 `safety` 顺手兜住 ——
#: 否则两个维度同时掉分，就分不清是谁抓到的了。
OOC_PHRASE = "作为人工智能"


# --------------------------------------------------------------------------- #
# 打补丁的小工具
# --------------------------------------------------------------------------- #
def _patch(owner: Any, name: str, replacement: Any) -> Callable[[], None]:
    """把 `owner.name` 换成 `replacement`，返回"还原"函数。

    直接改类属性（而不是实例），因为 agent 是每条用例现建的 ——
    变异必须对整个跑批生效。
    """
    original = getattr(owner, name)
    setattr(owner, name, replacement)

    def restore() -> None:
        setattr(owner, name, original)

    return restore


def _chain(*restores: Callable[[], None]) -> Callable[[], None]:
    """按**逆序**还原：后打的补丁先拆。"""
    return lambda: [r() for r in reversed(restores)]


# --------------------------------------------------------------------------- #
# 各个变异
# --------------------------------------------------------------------------- #
def install_memory_write_disabled() -> Callable[[], None]:
    """记忆写不进去：显式 remember 和被动 observe 都失效。"""
    return _chain(
        _patch(
            MemoryManager,
            "observe",
            lambda self, utterance, npc_id: None,
        ),
        _patch(
            MemoryManager,
            "remember",
            lambda self, content, tick, about=None, importance=None: SimpleNamespace(
                id="mutant-nomem"
            ),
        ),
    )


def install_retrieval_disabled() -> Callable[[], None]:
    """检索永远返回空：记忆写了但想不起来。"""
    return _patch(MemoryStore, "search", lambda self, *a, **k: [])


def install_planning_disabled() -> Callable[[], None]:
    """不做任何规划：NPC 只会说话，不会推进目标、也不调工具。"""
    return _patch(NPCAgent, "_make_plan", lambda self, utterance, decision, memories: None)


def install_floor_control_disabled() -> Callable[[], None]:
    """发言权闸门失效：同一个 tick 里每个 NPC 都被允许开口。

    这是多 NPC 场景最典型的调度缺陷（"抢话"）。
    """
    original = NPCAgent.step

    def step_without_floor(
        self: Any,
        utterance: Any = None,
        *,
        other_npc_spoke_last: bool = False,
        allow_speech: bool = True,
        count_silence: bool = True,
    ) -> Any:
        return original(
            self,
            utterance,
            other_npc_spoke_last=False,
            allow_speech=True,
            count_silence=count_silence,
        )

    return _patch(NPCAgent, "step", step_without_floor)


def install_ooc_phrase_reaches_transcript() -> Callable[[], None]:
    """让一句出戏台词**真的进转写** —— 绕过工具层的拦截。

    只注入台词、不碰检查器：这样 `persona` 维度能不能抓到，
    考验的是"评测有没有独立地看一眼台词"。
    """
    original = ToolRegistry._speak

    def speak_unguarded(self: Any, args: dict[str, Any], ctx: Any) -> ActionResult:
        text = str(args.get("text", "")).strip()
        if not text:
            return ActionResult(False, "speak", "没有内容可说")
        styled = ctx.persona.apply_style(f"{text}（{OOC_PHRASE}）")
        ctx.env.broadcast(ctx.actor_id, styled)
        return ActionResult(
            True,
            "speak",
            styled,
            state_delta={"text": styled, "to": args.get("to")},
        )

    return _patch(ToolRegistry, "_speak", speak_unguarded)


def install_ooc_phrase_with_detector_disabled() -> Callable[[], None]:
    """上面那个变异 **+** 把人设检查器关掉。

    这是**仪器变异**：评测如果直接采信 agent 自报的 `turn.persona_violations`，
    就会看不见任何东西 —— 分数不变，但分数已经没有意义了。
    """
    return _chain(
        install_ooc_phrase_reaches_transcript(),
        _patch(Persona, "check", lambda self, text, unlocked_topics=None: []),
    )


@dataclass(frozen=True)
class Mutant:
    id: str
    targets: tuple[str, ...]
    description: str
    install: Callable[[], Callable[[], None]]
    #: 这个变异属于"系统"还是"仪器"。两类抓不住的含义不同（见模块文档）。
    kind: str = "system"


MUTANTS: tuple[Mutant, ...] = (
    Mutant(
        id="memory_write_disabled",
        targets=("memory",),
        description="记忆写不进去（remember / observe 都失效）",
        install=install_memory_write_disabled,
    ),
    Mutant(
        id="retrieval_disabled",
        targets=("memory",),
        description="记忆检索永远返回空（写了但想不起来）",
        install=install_retrieval_disabled,
    ),
    Mutant(
        id="planning_disabled",
        targets=("task", "tools"),
        description="不做任何规划（不推进目标、不调工具）",
        install=install_planning_disabled,
    ),
    Mutant(
        id="floor_control_disabled",
        targets=("turn_taking",),
        description="发言权闸门失效（同一个 tick 里人人可开口 → 抢话）",
        install=install_floor_control_disabled,
    ),
    Mutant(
        id="ooc_phrase_reaches_transcript",
        targets=("persona",),
        description=f"出戏台词「{OOC_PHRASE}」绕过工具层拦截、真的进了转写",
        install=install_ooc_phrase_reaches_transcript,
    ),
    Mutant(
        id="ooc_phrase_with_detector_disabled",
        targets=("persona",),
        description=f"同上，但**同时关掉人设检查器**（仪器变异）",
        install=install_ooc_phrase_with_detector_disabled,
        kind="instrument",
    ),
)


# --------------------------------------------------------------------------- #
# 结果
# --------------------------------------------------------------------------- #
@dataclass
class MutantOutcome:
    mutant: Mutant
    pass_rate: float
    metric_means: dict[str, float]
    #: 维度 -> 相对基线的变化（负数=掉分）
    deltas: dict[str, float]

    @property
    def moved(self) -> dict[str, float]:
        return {k: v for k, v in self.deltas.items() if abs(v) > CAUGHT_EPS}

    @property
    def caught(self) -> bool:
        """评测有没有**注意到**这个缺陷（任何一个维度动了就算）。"""
        return bool(self.moved)

    @property
    def target_hit(self) -> bool:
        """**目标维度**有没有掉分（比"注意到了"更严）。"""
        return any(self.deltas.get(t, 0.0) < -CAUGHT_EPS for t in self.mutant.targets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.mutant.id,
            "kind": self.mutant.kind,
            "targets": list(self.mutant.targets),
            "description": self.mutant.description,
            "caught": self.caught,
            "target_hit": self.target_hit,
            "pass_rate": self.pass_rate,
            "metric_means": self.metric_means,
            "deltas": {k: round(v, 6) for k, v in self.deltas.items()},
        }


@dataclass
class SensitivityReport:
    baseline_pass_rate: float
    baseline_means: dict[str, float]
    total_cases: int
    outcomes: list[MutantOutcome] = field(default_factory=list)

    @property
    def survivors(self) -> list[MutantOutcome]:
        """注入缺陷却**没被评测注意到**的变异 —— 这就是盲点清单。"""
        return [o for o in self.outcomes if not o.caught]

    @property
    def ok(self) -> bool:
        return not self.survivors

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": {
                "total_cases": self.total_cases,
                "pass_rate": self.baseline_pass_rate,
                "metric_means": self.baseline_means,
            },
            "mutants": [o.to_dict() for o in self.outcomes],
            "survivors": [o.mutant.id for o in self.survivors],
            "ok": self.ok,
        }


# --------------------------------------------------------------------------- #
def _run(config: RuntimeConfig, categories: Optional[list[str]], limit: int) -> EvalReport:
    harness = EvalHarness(config)
    cases = harness.load_cases(categories)
    if limit:
        cases = cases[:limit]
    report = EvalReport()
    for case in cases:
        report.results.append(harness.run_case(case))
    return report


def run_sensitivity(
    categories: Optional[list[str]] = None,
    limit: int = 0,
    config: Optional[RuntimeConfig] = None,
    mutants: tuple[Mutant, ...] = MUTANTS,
) -> SensitivityReport:
    """跑一遍基线，然后逐个注入缺陷重跑。

    **基线跑在变异之前**，而且每个变异跑完立刻还原（`finally`）——
    否则第二个变异会叠在第一个上面，"哪个变异导致了掉分"就说不清了。
    """
    cfg = config or RuntimeConfig()
    baseline = _run(cfg, categories, limit)

    report = SensitivityReport(
        baseline_pass_rate=(
            round(baseline.passed / baseline.total, 3) if baseline.total else 0.0
        ),
        baseline_means=baseline.metric_means(),
        total_cases=baseline.total,
    )

    for mutant in mutants:
        restore = mutant.install()
        try:
            mutated = _run(cfg, categories, limit)
        finally:
            restore()
        means = mutated.metric_means()
        report.outcomes.append(
            MutantOutcome(
                mutant=mutant,
                pass_rate=(
                    round(mutated.passed / mutated.total, 3) if mutated.total else 0.0
                ),
                metric_means=means,
                deltas={
                    key: round(means.get(key, 0.0) - report.baseline_means.get(key, 0.0), 6)
                    for key in DIMENSIONS
                },
            )
        )
    return report


# --------------------------------------------------------------------------- #
def render_sensitivity(report: SensitivityReport, console: Any = None) -> None:
    """终端表格。`console` 为 None 时自己建一个 rich Console。"""
    if console is None:
        from rich.console import Console

        console = Console()

    from rich.table import Table

    console.print(
        f"基线：{report.total_cases} 条用例，通过率 "
        f"[bold]{report.baseline_pass_rate:.1%}[/bold]，六维均值 "
        + "　".join(f"{k}={v:.3f}" for k, v in report.baseline_means.items())
    )

    table = Table(title="评测敏感性（注入缺陷 → 评测掉不掉分）")
    table.add_column("变异", style="cyan", no_wrap=True)
    table.add_column("类", no_wrap=True)
    table.add_column("目标维度", no_wrap=True)
    table.add_column("通过率", no_wrap=True)
    table.add_column("掉了哪些维度", overflow="fold")
    table.add_column("判定", no_wrap=True)

    for outcome in report.outcomes:
        moved = outcome.moved
        detail = "　".join(f"{k} {v:+.3f}" for k, v in moved.items()) or "（没有任何维度动）"
        if not outcome.caught:
            verdict = "[bold red]没抓住（盲点）[/bold red]"
        elif outcome.target_hit:
            verdict = "[green]抓住目标维度[/green]"
        else:
            verdict = "[yellow]只旁敲侧击[/yellow]"
        table.add_row(
            outcome.mutant.id,
            outcome.mutant.kind,
            "/".join(outcome.mutant.targets),
            f"{outcome.pass_rate:.1%}",
            detail,
            verdict,
        )
    console.print(table)

    if report.survivors:
        console.print(
            f"[bold red]有 {len(report.survivors)} 个变异活着：[/bold red]"
            + "、".join(o.mutant.id for o in report.survivors)
            + "　—— 注入缺陷却没掉分，说明对应维度是瞎的。"
        )
    else:
        console.print(
            "[green]所有变异都被抓住了[/green] —— "
            "满分不是因为护栏从不报警。"
        )


# --------------------------------------------------------------------------- #
_CSS = """
  :root { --ink:#1f2430; --muted:#6b7280; --line:#e5e7eb; --bg:#fff;
          --soft:#f7f8fa; --accent:#2f6fd0; }
  * { box-sizing: border-box; }
  body { margin:0; padding:40px 28px 64px; background:var(--bg); color:var(--ink);
         font:15px/1.7 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
              "Hiragino Sans GB", "Microsoft YaHei", sans-serif; }
  .wrap { max-width:1240px; margin:0 auto; }
  h1 { font-size:26px; margin:0 0 6px; letter-spacing:-0.01em; }
  h2 { font-size:18px; margin:40px 0 12px; padding-bottom:8px; border-bottom:2px solid var(--line); }
  .sub { color:var(--muted); font-size:14px; margin-bottom:22px; }
  .headline { background:var(--soft); border:1px solid var(--line);
              border-left:4px solid var(--accent); border-radius:8px;
              padding:14px 18px; font-size:14.5px; }
  table { width:100%; border-collapse:collapse; font-size:13.5px; }
  th, td { padding:9px 10px; border-bottom:1px solid var(--line); text-align:left; }
  th { background:var(--soft); font-weight:600; font-size:12.5px; color:#374151; white-space:nowrap; }
  td.num { text-align:right; white-space:nowrap; font-family:ui-monospace, Menlo, Consolas, monospace; }
  td.mono { font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:12.5px; }
  tbody tr:hover { background:#fbfcfe; }
  .down { color:#d93025; font-weight:600; }
  .flat { color:#c4c9d1; }
  .ok { color:#0f9d58; font-weight:600; }
  .bad { color:#d93025; font-weight:700; }
  .warn { color:#b06000; font-weight:600; }
  .muted { color:var(--muted); }
  .box { background:var(--soft); border:1px solid var(--line); border-radius:8px;
         padding:14px 18px; margin:14px 0; font-size:14px; }
  code { font-family:ui-monospace, Menlo, Consolas, monospace; font-size:13px;
         background:#f2f3f5; padding:1px 5px; border-radius:4px; }
  footer { margin-top:48px; padding-top:16px; border-top:1px solid var(--line);
           color:var(--muted); font-size:13px; }
"""


def _esc(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _delta_cell(value: float) -> str:
    if abs(value) <= CAUGHT_EPS:
        return '<td class="num flat">·</td>'
    return f'<td class="num down">{value:+.3f}</td>'


def render_sensitivity_html(report: SensitivityReport) -> str:
    """自包含 HTML 报告（无外部依赖，双击就能看）。"""
    verdict_rows = []
    for outcome in report.outcomes:
        if not outcome.caught:
            verdict = '<span class="bad">没抓住（盲点）</span>'
        elif outcome.target_hit:
            verdict = '<span class="ok">抓住目标维度</span>'
        else:
            verdict = '<span class="warn">只旁敲侧击</span>'
        verdict_rows.append(
            "<tr>"
            f'<td class="mono">{_esc(outcome.mutant.id)}</td>'
            f"<td>{_esc(outcome.mutant.kind)}</td>"
            f'<td class="mono">{_esc("/".join(outcome.mutant.targets))}</td>'
            f"<td>{_esc(outcome.mutant.description)}</td>"
            f'<td class="num">{outcome.pass_rate:.1%}</td>'
            + "".join(_delta_cell(outcome.deltas[d]) for d in DIMENSIONS)
            + f"<td>{verdict}</td>"
            "</tr>"
        )

    if report.survivors:
        banner = (
            '<div class="box"><b class="bad">有 '
            f"{len(report.survivors)} 个变异活着：</b>"
            + _esc("、".join(o.mutant.id for o in report.survivors))
            + "　—— 注入缺陷却没掉分，说明对应维度是瞎的。</div>"
        )
    else:
        banner = (
            '<div class="box"><b class="ok">所有变异都被抓住了</b>　—— '
            "满分不是因为护栏从不报警。</div>"
        )

    baseline_means = "　".join(
        f"{k}={v:.3f}" for k, v in report.baseline_means.items()
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>评测敏感性报告</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>评测敏感性：注入缺陷，看评测掉不掉分</h1>
  <div class="sub">
    离线基线：共 {report.total_cases} 条用例，通过率
    <b>{report.baseline_pass_rate:.1%}</b>，六维均值 {_esc(baseline_means)}
  </div>

  <div class="headline">
    <b>一个从不失败的评测，和一个没有评测，在报告上长得一模一样。</b><br>
    所以这里做变异测试：往 agent 里注入<b>明确的缺陷</b>，重跑<b>同一套</b>用例，
    要求评测掉分。<b>注入缺陷却不掉分的，就是盲点清单。</b>
  </div>

  {banner}

  <h2>变异 × 维度</h2>
  <table>
    <thead>
      <tr>
        <th>变异</th><th>类</th><th>目标维度</th><th>说明</th><th>通过率</th>
        {"".join(f"<th>{d}</th>" for d in DIMENSIONS)}
        <th>判定</th>
      </tr>
    </thead>
    <tbody>
      {"".join(verdict_rows)}
    </tbody>
  </table>

  <h2>两类变异，含义不同</h2>
  <div class="box">
    <b>系统变异</b>：改的是 agent 的<b>可观测行为</b>（忘记 / 不规划 / 抢话 /
    绕过护栏说话）。抓不住 ⇒ 评测有盲点。<br>
    <b>仪器变异</b>：改的是<b>评测和被测方共用的那个检查器</b>。
    抓不住 ⇒ 评测在<b>采信被测方自己的说法</b> —— 它不会让分数变低，
    只会让分数<b>变假</b>。
  </div>

  <h2>这个套件真的抓到过一个仪器盲点</h2>
  <div class="box">
    <code>ooc_phrase_with_detector_disabled</code> 和
    <code>ooc_phrase_reaches_transcript</code> 注入的是<b>同一句</b>出戏台词，
    区别只是前者<b>同时关掉了 agent 自己的人设检查器</b>。<br><br>
    <b>修之前</b>：检查器完好 → persona 维度 <b>−0.612</b>；
    关掉检查器 → persona 维度 <b>0.000</b>（满分）。
    缺陷一模一样，分数从 0.388 变成 1.000。<br>
    根因不在人设检查本身，而在评测直接读了
    <code>turn.persona_violations</code> —— 那是<b>被测方自己算的</b>。<br><br>
    <b>修之后</b>：评测侧按人设的<b>数据表</b>自己重算一遍
    （<code>evaluator_persona_violations</code>），两行数字变成<b>完全相同</b>。
    而基线不变 —— 修掉的是盲点，不是把分数改好看。
  </div>

  <footer>
    由 <code>python -m npc_agent.cli sensitivity</code> 生成 ·
    离线路径，不需要任何 API key
  </footer>
</div>
</body>
</html>
"""

