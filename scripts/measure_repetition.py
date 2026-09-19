"""量「复读率」并生成 `docs/repetition.html`。

## 这个脚本回答什么

用户在控制台实测时发现 NPC 反复重复同一话题、对话推不动。这份脚本把
"推不动"变成一个数：**同一段对话里，NPC 说出的话有多大比例是之前说过的**。

三个数各回答一个问题：

| 运行方式 | 回答的问题 |
|---|---|
| 离线 10 轮 | 演示对话的**真实长度**。控制台默认走的就是这条离线启发式路径 |
| 离线 20 轮 | 模板池耗尽之后会怎样（**已知局限**，不是目标） |
| 模型 20 轮 | prompt 里带上【你最近说过】之后，长对话还复读吗 |

## 判据不抄第二份

复读的判据住在包里（`npc_agent.modules.repetition`）—— 这里只负责
"跑场景、印表格、出报告"。抄一份判据就会漂移，
而漂移的表现是"脚本说没问题、测试说有问题"。

## 复现「修复前」的读数

父提交还没有判据模块，所以要把那个纯函数模块拷过去（它没有依赖，
这也是当初把它放进包里、而不是放进 `reports/` 的原因）：

    git worktree add ../_prefix HEAD
    cp npc_agent/modules/repetition.py ../_prefix/npc_agent/modules/
    cd ../_prefix
    python scripts/measure_repetition.py --skip-model --json before.json --html none

回到主树，把 `before.json` 喂进来即可：

    python scripts/measure_repetition.py --before before.json --html docs/repetition.html

用法：
    python scripts/measure_repetition.py                      # 离线 + 模型，写 docs/repetition.html
    python scripts/measure_repetition.py --skip-model         # 只跑离线（秒级）
    python scripts/measure_repetition.py --json out.json      # 顺便落一份原始读数
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npc_agent.config import RuntimeConfig, list_scenarios, load_scenario  # noqa: E402
from npc_agent.modules.repetition import find_repeats  # noqa: E402
from npc_agent.studio import run_chat  # noqa: E402

#: 一段固定的、贴近真人聊天节奏的对话。**必须和
#: `tests/test_repetition.py` 的 `CONVERSATION_LONG` 逐字一致** ——
#: 两处各写一份必然漂移，而漂移的表现是"报告说 29%、测试说 22%"，
#: 读者无从判断哪个是真的。有测试钉住这一点
#: （`test_the_report_and_the_test_use_the_same_conversation`）。
#:
#: 刻意**以陈述句为主**（"我想喝点酸的" / "我下次还来"）——
#: 那正是当前实现最薄弱的输入：非问句一律落到 acknowledge，
#: 而 acknowledge 的模板池最小。用问句为主的对话去量，会把这个弱点盖住。
#: 也刻意夹了两句**没有问号的问题**（"你叫什么名字" / "这店开了多久了"），
#: 它们曾经被当成陈述句，于是 NPC 只应声、不回答 —— "推不动"的最直接原因。
CONVERSATION = [
    "你好呀",
    "这里有什么好喝的？",
    "我想喝点酸的",
    "你平时都在这儿吗",
    "那给我来一杯吧",
    "谢谢！",
    "你叫什么名字",
    "这店开了多久了",
    "我下次还来",
    "那明天见",
    "今天人真多",
    "你们这儿有座位吗",
    "我最喜欢靠窗的位置",
    "外面下雨了",
    "你推荐什么",
    "我朋友一会儿也来",
    "他不太喝咖啡",
    "那就来两杯吧",
    "麻烦你了",
    "改天再聊",
]

REGIMES = (
    ("offline10", "离线 10 轮", 10, False),
    ("offline20", "离线 20 轮", 20, False),
    ("model20", "模型 20 轮", 20, True),
)


def _run(scenario_id: str, turns: int, use_model: bool, model: str) -> dict[str, Any]:
    """跑一个场景的一个运行方式，返回台词与复读读数。"""
    cfg = RuntimeConfig.from_env()
    # 规划与复读无关，且模型规划单次约 35s —— 关掉，把预算全留给台词。
    cfg.use_llm_planner = False
    cfg.use_llm_speech = bool(use_model)
    if use_model:
        cfg.model = model

    lines = [
        {"kind": "say", "speaker": "player_a", "text": text}
        for text in CONVERSATION[:turns]
    ]
    out = run_chat(cfg, {"scenario": scenario_id, "events": lines})

    spoken = [
        (turn["name"], turn["say"])
        for event in out["events"]
        for turn in event["turns"]
        if turn["say"]
    ]
    report = find_repeats(spoken)
    return {
        "total": report.total,
        "distinct": report.distinct,
        "repeats": len(report.repeats),
        "rate": report.rate,
        "examples": [
            {"at": i + 1, "collides_with": j + 1, "text": text, "similarity": sim}
            for i, j, text, _prev, sim in report.repeats[:3]
        ],
        "lines": [{"speaker": name, "text": text} for name, text in spoken],
    }


def measure(scenarios: list[str], model: str, skip_model: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"scenarios": scenarios, "model": model, "regimes": {}}
    for key, label, turns, use_model in REGIMES:
        if use_model and skip_model:
            continue
        rows: dict[str, Any] = {}
        for sid in scenarios:
            load_scenario(sid)  # 提前炸掉，别在表格中途出错
            started = time.perf_counter()
            row = _run(sid, turns, use_model, model)
            row["seconds"] = round(time.perf_counter() - started, 1)
            rows[sid] = row
            print(
                f"  [{label}] {sid:<12} 台词 {row['total']:>3} "
                f"去重后 {row['distinct']:>3} 复读 {row['repeats']:>3} "
                f"（{row['rate']:>6.1%}） {row['seconds']:>6.1f}s",
                flush=True,
            )
        out["regimes"][key] = {"label": label, "turns": turns, "rows": rows}
    return out


def _total(regime: dict[str, Any]) -> dict[str, Any]:
    total = sum(r["total"] for r in regime["rows"].values())
    repeats = sum(r["repeats"] for r in regime["rows"].values())
    return {"total": total, "repeats": repeats, "rate": (repeats / total) if total else 0.0}


# --------------------------------------------------------------------------- #
# HTML

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
.sample { font-size:12.5px; color:var(--muted); font-family:ui-monospace,Menlo,Consolas,monospace; }
"""


def _rate_cell(rate: float) -> str:
    cls = "pass" if rate == 0 else "fail"
    return f'<td class="num {cls}">{rate:.1%}</td>'


def _table(
    regime: dict[str, Any],
    scenarios: list[str],
    before: dict[str, Any] | None = None,
) -> str:
    has_before = before is not None and any(
        before["rows"].get(sid, {}).get("total") for sid in scenarios
    )
    head = "<th>场景</th><th>台词数</th><th>去重后</th><th>复读</th><th>复读率</th>"
    if has_before:
        head = "<th>场景</th><th>修复前复读率</th><th>修复后复读率</th><th>台词数</th>"
    rows = []
    for sid in scenarios:
        row = regime["rows"].get(sid)
        if not row:
            continue
        if has_before:
            old = (before or {}).get("rows", {}).get(sid, {}).get("rate")
            old_cell = (
                f'<td class="num fail">{old:.1%}</td>'
                if old is not None
                else '<td class="num muted">·</td>'
            )
            rows.append(
                f'<tr><td class="name">{sid}</td>{old_cell}'
                f"{_rate_cell(row['rate'])}"
                f'<td class="num">{row["total"]}</td></tr>'
            )
        else:
            rows.append(
                f'<tr><td class="name">{sid}</td><td class="num">{row["total"]}</td>'
                f'<td class="num">{row["distinct"]}</td>'
                f'<td class="num">{row["repeats"]}</td>'
                f"{_rate_cell(row['rate'])}</tr>"
            )
    agg = _total(regime)
    if has_before:
        old_agg = _total(before) if before else None
        rows.append(
            "<tr><td class=\"name\">合计</td>"
            + (
                f'<td class="num fail">{old_agg["rate"]:.1%}</td>'
                if old_agg
                else '<td class="num muted">·</td>'
            )
            + f"{_rate_cell(agg['rate'])}"
            + f'<td class="num">{agg["total"]}</td></tr>'
        )
    else:
        rows.append(
            '<tr><td class="name">合计</td>'
            f'<td class="num">{agg["total"]}</td><td class="num">·</td>'
            f'<td class="num">{agg["repeats"]}</td>{_rate_cell(agg["rate"])}</tr>'
        )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def render(data: dict[str, Any], before: dict[str, Any] | None) -> str:
    scenarios = data["scenarios"]
    regimes = data["regimes"]
    n_runs = len(regimes) * len(scenarios)

    offline10 = regimes.get("offline10")
    offline20 = regimes.get("offline20")
    model20 = regimes.get("model20")

    headline = (
        f"共 {n_runs} 条用例（{len(scenarios)} 个场景 × {len(regimes)} 种运行方式）。"
    )
    if offline10:
        headline += f" 离线 10 轮复读率 <strong>{_total(offline10)['rate']:.1%}</strong>"
    if model20:
        headline += f"、模型 20 轮 <strong>{_total(model20)['rate']:.1%}</strong>"
    headline += "。"

    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>复读：一个只看单句的评测看不见的缺陷</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        '<div class="wrap">',
        "<h1>复读：一个只看单句的评测看不见的缺陷</h1>",
        '<div class="sub">NPC 反复重复同一话题、对话推不动 —— 把它变成一个数，'
        "再修掉它。判据住在 <code>npc_agent/modules/repetition.py</code>。</div>",
        f'<div class="headline">{headline}</div>',
    ]

    parts.append(
        '<div class="warn"><strong>为什么六维评测看不见这个问题。</strong>'
        "六维（人设 / 记忆 / 任务 / 工具 / 安全 / 发言调度）"
        "<strong>每一维都只看一行</strong>：这一句像不像这个角色、这一轮目标达成没有。"
        "<strong>没有任何一维把「这一句」和「上一句」放在一起看。</strong>"
        "所以复读 62% 和「六维全 1.000、231 条全过」可以同时成立 —— "
        "而且确实同时成立了。修完复读之后离线基线仍然是 231/231、六维全 1.000，"
        "这反过来证明评测真的看不见复读。"
        "<br>判据取「同一说话人自己说过的两句话，去标点后相似度 ≥ 0.80」。"
        "阈值不是拍的：同义加词 0.86（判复读）、不同内容 0.55（不判），两侧余量都 ≥0.05。</div>"
    )

    if offline10 and before:
        parts.append("<h2>修复前 vs 修复后（离线 10 轮）</h2>")
        parts.append(
            '<div class="sub">10 轮就是演示对话的真实长度。'
            "「修复前」跑的是父提交（<code>git worktree</code> + 把判据模块拷过去）。</div>"
        )
        parts.append(_table(offline10, scenarios, before=before.get("regimes", {}).get("offline10")))
    elif offline10:
        parts.append("<h2>离线 10 轮</h2>")
        parts.append(_table(offline10, scenarios))

    if offline20:
        parts.append("<h2>离线 20 轮 —— 已知局限</h2>")
        parts.append(
            '<div class="sub">离线启发式的词汇量是<strong>有限</strong>的'
            "（每人设约 10 条低信息模板 + 3~4 个话题），长对话必然耗尽它。"
            "复读率的下界就是 <code>(轮数 − 词汇量) / 轮数</code>。</div>"
        )
        parts.append(_table(offline20, scenarios))

    if model20:
        parts.append("<h2>模型 20 轮 —— 同一个长度，接上模型之后</h2>")
        parts.append(
            '<div class="sub">同一个场景、同一段对话、同一个长度，'
            "只把台词换成真实模型（<code>"
            + data["model"]
            + "</code>）。台词 prompt 里带了【你最近说过】+ 复读闸门。</div>"
        )
        parts.append(_table(model20, scenarios))

    parts.append("<h2>修了什么</h2>")
    parts.append(
        '<div class="card"><ul>'
        "<li><strong>意图塌缩</strong>：玩家说陈述句一律映射成 <code>acknowledge</code>，"
        "而它只有一句固定模板 —— 实测 10 轮里同一句说了 7 遍。"
        "改成按「用得最少」在 <code>acknowledge</code> / <code>probe</code> 之间轮换。</li>"
        "<li><strong>模板单变体</strong>：一个意图一个字符串 ⇒ 同一意图必然逐字相同。"
        "高频意图改成多变体列表，轮换由计数器决定（<strong>确定性</strong>，"
        "因为控制台每次请求都从头重放整段对话）。</li>"
        "<li><strong>记忆反复回引同一条</strong>：检索每轮都返回打分最高的那条。"
        "加 <code>_recalled</code> 集合，回引过一次就不再回引。</li>"
        "<li><strong>问题识别只看 <code>?</code>/<code>？</code></strong>："
        "「你叫什么名字」「这店开了多久了」被当成陈述句 ⇒ 只应声不回答。"
        "这是「推不动」的<strong>最直接</strong>原因（不是重复，是压根没答）。</li>"
        "<li><strong>回避问题的常量</strong>：<code>answer_hint</code> 原来是写死的"
        "「这个我还没想过，你怎么看？」⇒ NPC 对<em>每个</em>问题都用同一句话"
        "<strong>把问题踢回给玩家</strong>。改成先答「问到自己」"
        "（<code>self_facts</code>）、再答世界知识，答不上来才老实说不知道。</li>"
        "</ul></div>"
    )

    parts.append(
        '<div class="warn"><strong>这份报告不能说明什么。</strong>'
        "它量的是<strong>同一段固定对话</strong>上的复读率，不是「对话质量」。"
        "复读率 0% 不代表 NPC 有趣、记得住事、或者目标推进得对 —— "
        "那些由六维评测和裁判负责。反过来，这里也没有把 20 轮离线那个数藏起来："
        "它是<strong>已知局限</strong>，护栏钉的是 0.35 的上界，"
        "而且 docstring 里写明「这是局限，不是目标」。</div>"
    )

    parts.append(
        '<div class="foot">重生成：'
        "<code>python scripts/measure_repetition.py --html docs/repetition.html</code>"
        "（离线部分秒级；模型部分要模型 + 额度，所以这份是<strong>一次跑批的快照</strong>）。"
        "<br>判据与回归测试：<code>npc_agent/modules/repetition.py</code>、"
        "<code>tests/test_repetition.py</code>。</div>"
    )
    parts += ["</div>", "</body>", "</html>", ""]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="量复读率并生成报告")
    parser.add_argument("--html", default="docs/repetition.html", help="输出 HTML；none 表示不写")
    parser.add_argument("--json", default=None, help="把原始读数落一份 JSON")
    parser.add_argument("--before", default=None, help="父提交跑出来的 JSON（当基线用）")
    parser.add_argument("--model", default=None, help="模型名（默认读环境变量）")
    parser.add_argument("--skip-model", action="store_true", help="只跑离线路径")
    parser.add_argument("--scenario", action="append", default=None, help="只跑指定场景（可重复）")
    args = parser.parse_args()

    cfg = RuntimeConfig.from_env()
    model = args.model or getattr(cfg, "model", "") or "(未配置)"
    scenarios = args.scenario or list_scenarios()

    print(f"场景 {len(scenarios)} 个｜模型 {model}｜skip_model={args.skip_model}\n", flush=True)
    data = measure(scenarios, model, args.skip_model)

    if args.json:
        Path(args.json).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n原始读数 → {args.json}")

    if args.html and args.html != "none":
        before = None
        if args.before:
            before = json.loads(Path(args.before).read_text(encoding="utf-8"))
        out = Path(args.html)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render(data, before), encoding="utf-8")
        print(f"报告 → {out}")

    for key, regime in data["regimes"].items():
        agg = _total(regime)
        print(f"{regime['label']:<12} 合计 复读 {agg['repeats']}/{agg['total']}（{agg['rate']:.1%}）")


if __name__ == "__main__":
    main()
