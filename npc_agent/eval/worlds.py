"""跨世界报告：同一套 Agent 在两个世界上的成绩。

## 这不是一份对照实验，必须说清楚

`compare` 做的是**受控对照**：同一套用例、只换一个自变量（模型 / 记忆策略），
所以它敢把两行相减。这份报告不是 —— 两个世界跑的**用例集本身不同**
（咖啡屋 12 条，体素世界 3 条），相减没有意义。

它回答的是另一个问题：**同一套 Agent、同一套指标、同一套 expect 词汇表，
在结构完全不同的两个世界上分别是什么成绩。**
换句话说，它验证的是**覆盖面**，不是**因果**。

把这两种报告混为一谈是很容易犯的错，也是"看起来专业、其实在拿苹果比橘子"
最典型的来源。所以这里不生成 deltas 表，并且把上面这段话直接印在报告里。
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..config import RuntimeConfig
from .harness import EvalHarness, EvalReport

#: 六个维度。与 report.py 的标签表保持一致。
METRIC_LABELS: dict[str, str] = {
    "task": "任务",
    "tools": "工具",
    "memory": "记忆",
    "persona": "人设",
    "safety": "安全",
    "turn_taking": "调度",
}


@dataclass
class WorldSpec:
    """一个世界在评测里的样子。"""

    label: str
    env: str
    categories: list[str]
    blurb: str
    #: 这个世界的结构性特点。报告里逐条列出来 —— 光有分数看不出"不同在哪"。
    traits: list[str] = field(default_factory=list)


@dataclass
class WorldRun:
    spec: WorldSpec
    report: EvalReport
    duration: float = 0.0

    @property
    def pass_rate(self) -> float:
        total = self.report.total
        return round(self.report.passed / total, 3) if total else 0.0

    @property
    def means(self) -> dict[str, float]:
        return self.report.metric_means()

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.spec.label,
            "env": self.spec.env,
            "categories": self.spec.categories,
            "blurb": self.spec.blurb,
            "traits": self.spec.traits,
            "duration_sec": round(self.duration, 2),
            "passed": self.report.passed,
            "total": self.report.total,
            "pass_rate": self.pass_rate,
            "metric_means": self.means,
            "cases": [
                {
                    "case_id": r.case_id,
                    "scenario": r.scenario,
                    "description": r.description,
                    "passed": r.passed,
                    "notes": r.notes,
                }
                for r in self.report.results
            ],
        }


#: 两个世界。加第三个世界只需要在这里加一条 —— 和 `env/__init__.py` 的注册表一样，
#: 是纯加法。
WORLDS: list[WorldSpec] = [
    WorldSpec(
        label="星屿咖啡屋（文字世界）",
        env="star-isle",
        categories=["task", "memory", "persona", "safety", "multi_npc"],
        blurb="确定性文字世界。回归基线与消融实验都在这里做 —— 毫秒级、零依赖。",
        traits=[
            "位置是离散的几个房间",
            "背包是物品列表（只问有没有）",
            "拿东西 = take_item；交付 = give_item",
            "没有时间压力",
        ],
    ),
    WorldSpec(
        label="Minecraft 体素世界",
        env="minecraft",
        categories=["minecraft"],
        blurb="结构不同的世界，用来把「环境无关」从主张变成证据。",
        traits=[
            "位置是三维坐标 + 命名地点（POI）",
            "背包是 {物品: 数量}（还问有几个）",
            "拿东西 = mine（挖）/ craft（合成）；交付 = transfer",
            "昼夜循环：夜里没光源就干不了活",
        ],
    ),
]


def run_worlds(
    config: RuntimeConfig | None = None,
    *,
    progress: Callable[[str], None] | None = None,
    specs: list[WorldSpec] | None = None,
) -> list[WorldRun]:
    """把每个世界的用例集跑一遍。"""
    import time

    cfg = config or RuntimeConfig()
    out: list[WorldRun] = []
    for spec in specs or WORLDS:
        if progress:
            progress(f"{spec.label}（{len(spec.categories)} 个类别）…")
        harness = EvalHarness(cfg)
        cases = harness.load_cases(spec.categories)
        report = EvalReport(
            config={
                "provider": cfg.llm_provider,
                "model": cfg.model or "(offline)",
                "env": spec.env,
            }
        )
        started = time.time()
        for case in cases:
            report.results.append(harness.run_case(case))
        run = WorldRun(spec=spec, report=report, duration=time.time() - started)
        out.append(run)
        if progress:
            progress(f"    {report.passed}/{report.total} 通过，耗时 {run.duration:.1f}s")
    return out


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
_CSS = """
:root { --ink:#1f2430; --muted:#6b7280; --line:#e5e7eb; --bg:#fff;
        --soft:#f7f8fa; --accent:#2f6fd0; --good:#0f9d58; --bad:#d93025; }
* { box-sizing: border-box; }
body { margin:0; padding:40px 28px 64px; background:var(--bg); color:var(--ink);
  font:15px/1.7 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
       "Hiragino Sans GB", "Microsoft YaHei", sans-serif; }
.wrap { max-width:1180px; margin:0 auto; }
h1 { font-size:26px; margin:0 0 6px; letter-spacing:-0.01em; }
h2 { font-size:18px; margin:38px 0 12px; padding-bottom:8px; border-bottom:2px solid var(--line); }
h3 { font-size:15px; margin:22px 0 8px; }
.sub { color:var(--muted); font-size:14px; margin-bottom:22px; }
.headline { background:var(--soft); border:1px solid var(--line); border-left:4px solid var(--accent);
  border-radius:8px; padding:14px 18px; font-size:14.5px; }
.warn { background:#fffbf0; border:1px solid #f0dfb8; border-left:4px solid #e8a33d;
  border-radius:8px; padding:14px 18px; font-size:13.5px; margin:16px 0; }
table { width:100%; border-collapse:collapse; font-size:13.5px; }
th,td { padding:9px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:middle; }
th { background:var(--soft); font-weight:600; font-size:12.5px; color:#374151; white-space:nowrap; }
td.num { text-align:right; white-space:nowrap; font-family:ui-monospace,Menlo,Consolas,monospace; }
td.name { font-weight:600; }
tbody tr:hover { background:#fbfcfe; }
.pass { color:var(--good); font-weight:600; }
.fail { color:var(--bad); font-weight:600; }
.card { border:1px solid var(--line); border-radius:8px; padding:14px 18px; margin-bottom:12px; background:#fcfcfd; }
.card ul { margin:6px 0 0; padding-left:20px; }
.card li { font-size:13.5px; }
.muted { color:var(--muted); }
code { background:#eef0f4; padding:1px 5px; border-radius:4px; font-size:12.5px; }
.foot { margin-top:34px; color:var(--muted); font-size:12.5px; }
"""


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _world_rows(runs: list[WorldRun]) -> str:
    rows: list[str] = []
    for run in runs:
        means = run.means
        cells = "".join(
            f'<td class="num">{means.get(key, 0.0):.2f}</td>' for key in METRIC_LABELS
        )
        mark = "pass" if run.report.passed == run.report.total else "fail"
        rows.append(
            f'<tr><td class="name">{_esc(run.spec.label)}</td>'
            f'<td class="num">{_esc(run.spec.env)}</td>'
            f'<td class="num">{run.report.passed}/{run.report.total}</td>'
            f'<td class="num {mark}">{run.pass_rate:.0%}</td>'
            f"{cells}"
            f'<td class="num muted">{run.duration:.2f}s</td></tr>'
        )
    return "\n".join(rows)


def _world_cards(runs: list[WorldRun]) -> str:
    blocks: list[str] = []
    for run in runs:
        traits = "".join(f"<li>{_esc(t)}</li>" for t in run.spec.traits)
        cases = "".join(
            f'<tr><td class="name">{_esc(c["case_id"])}</td>'
            f'<td class="num">{_esc(c["scenario"])}</td>'
            f'<td>{_esc(c["description"])}</td>'
            f'<td class="{"pass" if c["passed"] else "fail"}">'
            f'{"PASS" if c["passed"] else "FAIL"}</td></tr>'
            for c in run.to_dict()["cases"]
        )
        blocks.append(
            f'<div class="card"><h3>{_esc(run.spec.label)}'
            f'　<span class="muted">{_esc(run.spec.env)}</span></h3>'
            f'<div class="muted">{_esc(run.spec.blurb)}</div>'
            f"<ul>{traits}</ul>"
            f'<table style="margin-top:12px"><thead><tr><th>用例</th><th>场景</th>'
            f"<th>说明</th><th>结论</th></tr></thead><tbody>{cases}</tbody></table></div>"
        )
    return "\n".join(blocks)


def render_worlds_html(
    runs: list[WorldRun],
    title: str = "同一套 Agent，两个世界",
    subtitle: str = "",
) -> str:
    headers = "".join(f"<th>{_esc(label)}</th>" for label in METRIC_LABELS.values())
    total_cases = sum(r.report.total for r in runs)
    total_pass = sum(r.report.passed for r in runs)
    headline = (
        f"<strong>{len(runs)} 个世界</strong>、共 {total_cases} 条用例、"
        f"{total_pass}/{total_cases} 通过。"
        "同一份 <code>NPCAgent</code> 代码，两个结构完全不同的世界 —— "
        "Agent 侧一行没改。"
    )
    subtitle = subtitle or (
        "环境无关的可执行证据：观测 schema 一致、工具调用走同一个 dispatch、"
        "目标完成由同一份条件判定器决定。"
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>{_esc(title)}</h1>
  <div class="sub">{_esc(subtitle)}</div>
  <div class="headline">{headline}</div>

  <div class="warn">
    <strong>这不是一份对照实验。</strong>
    两个世界跑的是<strong>不同的用例集</strong>（咖啡屋与体素世界的用例数不同），
    所以下面没有"差值"表，两行分数也不该相减。
    它回答的是<strong>覆盖面</strong>问题 —— 同一套 Agent、同一套指标、
    同一套 <code>expect</code> 词汇表，在两个世界上分别是什么成绩；
    不是"哪个世界更好"。
    <br>受控对照见 <code>python -m npc_agent.cli compare</code> 与 <code>ablate</code>。
  </div>

  <h2>总分</h2>
  <table>
    <thead><tr>
      <th>世界</th><th>环境</th><th>通过</th><th>通过率</th>
      {headers}<th>耗时</th>
    </tr></thead>
    <tbody>
{_world_rows(runs)}
    </tbody>
  </table>

  <h2>每个世界跑的是什么</h2>
{_world_cards(runs)}

  <h2>怎么读这张表</h2>
  <div class="card">
    <ul>
      <li><strong>六维指标在两个世界里含义相同。</strong>
          <code>task</code> 问的都是"期望的世界状态达成了吗"，
          <code>safety</code> 问的都是"越界了吗" —— 只是世界状态长什么样不同。</li>
      <li><strong>体素世界特有的断言只有两个新的</strong>：
          <code>placed</code>（方块真的放在指定地点）与
          <code>has_count</code>（背包里有几个）。
          其余断言词汇完全复用 —— 加一个世界不需要加一套新的断言语言。</li>
      <li><strong>分数相同不代表难度相同。</strong>
          体素世界的任务链更长（11 个世界动作 vs 咖啡屋的 5～8 个），
          而且多了一条昼夜约束。用例少，所以这条对比只是"能跑通"，不是"更擅长"。</li>
    </ul>
  </div>

  <div class="foot">
    由 <code>python -m npc_agent.cli worlds</code> 生成　·　游戏 AI NPC 智能体框架
  </div>
</div>
</body>
</html>
"""


def write_worlds_html(
    runs: list[WorldRun],
    out_path: str | Path,
    title: str = "同一套 Agent，两个世界",
    subtitle: str = "",
) -> Path:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_worlds_html(runs, title, subtitle), encoding="utf-8")
    return target


def worlds_to_json(runs: list[WorldRun]) -> dict[str, Any]:
    return {
        "note": "跨世界覆盖报告，不是对照实验：两个世界的用例集不同，分数不可相减。",
        "worlds": [r.to_dict() for r in runs],
    }


def save_worlds_json(runs: list[WorldRun], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(worlds_to_json(runs), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target
