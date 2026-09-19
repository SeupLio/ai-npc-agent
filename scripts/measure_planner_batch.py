"""把「模型自己规划」这条路上的修复，量成一个能读的前后对比报告。

## 这份报告回答什么

`docs/batch_planner.html` 说的是**旧代码**：那次 228 条跑批里，规划交给模型之后
目标经常做不完。第四个根因（规划 prompt 里没有 `success_when`）修掉之后，
需要一份新的读数，否则 README 里那句话就只能停在"1/4 → 10/10、单场景 n=10"。

这份脚本把两个检查点读进来，渲染 `docs/planner_batch.html`。

## 为什么是「配对子集」而不是两个满量跑批

231 条 × 2 条臂，每条臂实测均值 104s，串行 ≈ 6.7 小时。而**要回答的问题
只需要一个配对样本**：同一批用例、同一个模型、只差规划 prompt 里那几行。

于是取**加载顺序的前 60 条**当配对子集。它正好是 10×6 ——
`npc_agent/eval/cases/*.jsonl` 里生成用例是按分类**轮转**排的
（memory / minecraft / multi_npc / persona / safety / task 循环），
所以「前 60 条」天然是分层样本，不是"先写的那 60 条"。
子集的定义是**先定好再跑的**（`--limit 60`），不是事后挑的。

脚本会**验证**这一点：如果修复前那条臂的用例集不是加载顺序的前缀，
报告里会明着说"这不是一个预先登记的子集"，而不是默默按交集算。

## 公平性：只比「两边都真的让模型规划成功」的那些

框架在**规划解析失败时会静默回落启发式规划器**（`planner_failures` 记着次数）。
两条臂的回落次数不一样，直接相减就会把"模型没规划成功"混进
"规划质量"里。所以：

* 主表给的是**框架口径的通过率**（读者看到的那个数）；
* 同时给**只算 `planner_failures == 0` 的干净子集**通过率（保守的那个数）；
* 两个数并排放，回落次数单独列一栏 —— 不藏。

## 用法

    python scripts/measure_planner_batch.py \\
        --before "E:/WB AI/_prefix2/reports/planner_batch_before_checkpoint.json" \\
        --after  reports/planner_batch_after_checkpoint.json \\
        --html docs/planner_batch.html

修复前那条臂这样跑（在父提交的 worktree 里）：

    git worktree add "../_prefix2" 4c6569c
    cd "../_prefix2"
    NPC_AGENT_PROVIDER=openai-compat NPC_AGENT_BASE_URL=... NPC_AGENT_MODEL=... \\
    python -m npc_agent.cli eval --no-speech --limit 60 --concurrency 6 \\
        --checkpoint reports/planner_batch_before_checkpoint.json \\
        --json reports/planner_batch_before.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- #
# 读

#: 传输层故障的特征串。**这个分类是这份报告里最容易被误读的一栏，所以单列。**
#:
#: 实测（2026-09-19 那次 231 条跑批，前 16 条）里 `planner_failures` 全部是
#: `TimeoutError: The read operation timed out` 和 `RemoteDisconnected` ——
#: **一条解析失败都没有**。也就是说这一栏量的是**端点健康**，不是"模型规划得差"。
#: 把两者混成一栏，读者会把网络抖动读成模型能力问题。
_TRANSPORT_MARKERS = (
    "TimeoutError",
    "timed out",
    "RemoteDisconnected",
    "ConnectionError",
    "ConnectionResetError",
    "BrokenPipeError",
    "URLError",
)


def is_transport_error(text: str) -> bool:
    return any(marker in text for marker in _TRANSPORT_MARKERS)


def _read_arm(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """读一份读数，返回（原始结果列表，元信息）。

    元信息里带**声明总数**（检查点的 `total`）—— 报告要能说清
    "这一臂跑到了 106/231"，而不是把 106 说成全部。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw: list[dict[str, Any]] = []
    if isinstance(data.get("runs"), list):
        raw = [r["result"] for r in data["runs"] if r.get("result")]
    elif isinstance(data.get("results"), list):
        raw = list(data["results"])
    else:
        raise ValueError(f"{path}：既没有 runs 也没有 results，认不出这份读数")
    return raw, {
        "declared_total": data.get("total"),
        "done": data.get("done"),
        "path": str(path),
    }


def load_arm(path: str | Path) -> dict[str, dict[str, Any]]:
    """读一条臂的读数，归一成 `{case_id: record}`。

    同时接受两种形状，因为跑批中途落的是检查点、跑完落的是报告：

    * 检查点：`{"done":…, "runs":[{"index":…, "result": {…}}]}`
    * 报告：  `{"results":[{…}]}`

    两种都要认，否则"用检查点先出一版、跑完再刷新"就得改脚本。
    """
    raw, _meta = _read_arm(path)
    out: dict[str, dict[str, Any]] = {}
    for res in raw:
        scores = res.get("scores") or {}
        details = res.get("details") or {}
        last_error = str(res.get("planner_last_error") or "")
        out[res["case_id"]] = {
            "case_id": res["case_id"],
            "category": res.get("category") or "?",
            "passed": bool(res.get("passed")),
            "task": float(scores.get("task", 0.0)),
            "scores": {k: float(v) for k, v in scores.items()},
            "task_detail": details.get("task", ""),
            "planner_failures": int(res.get("planner_failures") or 0),
            "planner_last_error": last_error,
            # ⚠️ 只有**最后一条**错误被留下（检查点不存全部错误），
            # 所以这是"按最后一条归类"的近似，不是逐条归类。
            "planner_last_is_transport": bool(last_error) and is_transport_error(last_error),
        }
    return out


def load_cases_raw() -> list[dict[str, Any]]:
    """按加载顺序读全部用例（原始 dict，含 `scenario`）。"""
    cases: list[dict[str, Any]] = []
    for path in sorted(glob.glob(str(ROOT / "npc_agent/eval/cases/*.jsonl"))):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("//"):
                continue
            cases.append(json.loads(text))
    return cases


def load_order() -> list[tuple[str, str]]:
    """用例的**加载顺序**，与 `EvalHarness.load_cases()` 一致。

    这里重算一遍而不是调用框架，是因为报告要能**独立验证**
    "前 60 条"这个说法 —— 调用同一个函数验证自己等于没验证。
    有测试钉住"重算出来的顺序和框架一致"。
    """
    return [(c["id"], c.get("category") or "?") for c in load_cases_raw()]


_CONDITION_KINDS = (
    "flag",
    "player_has",
    "player_has_count",
    "all_flags",
    "any_flags",
    "all_players_spoke",
)


def _kind_of(condition: Any) -> str:
    if not isinstance(condition, dict) or not condition:
        return "（空条件）"
    if condition.get("all_of"):
        return "all_of"
    if condition.get("any_of"):
        return "any_of"
    for kind in _CONDITION_KINDS:
        if kind in condition:
            return kind
    return "无法识别：" + ",".join(sorted(condition))


def _flags_of(condition: Any) -> set[str]:
    """递归取出一个完成条件里所有**要求被置上**的世界标记。

    ⚠️ 必须**递归**。第一版按"顶层是 `flag` 或 `all_of` 就算含标记"来数，
    得到 42；真的递归下去是 **41** —— `all_of` 里也可能一支标记都没有。
    差一条不多，但这种"看起来对"的近似会一路写进报告，
    所以宁可多写十行递归。
    """
    if not isinstance(condition, dict):
        return set()
    out: set[str] = set()
    if "flag" in condition:
        out.add(str(condition["flag"]))
    for key in ("all_flags", "any_flags"):
        out |= {str(f) for f in (condition.get(key) or [])}
    for sub in (condition.get("all_of") or []) + (condition.get("any_of") or []):
        out |= _flags_of(sub)
    return out


def condition_kinds(case_ids: list[str]) -> dict[str, int]:
    """配对子集里，各目标完成条件（`success_when`）的**类型**分布。

    ## 为什么这件事要在跑之前算

    这次修复改的是「规划 prompt 里有没有 `success_when`」。
    如果配对子集里根本没有靠 `success_when` 判定完成的目标，
    那前后对比必然没有差 —— 而这个"没有差"跟修复好坏无关。

    所以先数一遍。实测：配对子集里每个目标都有 `success_when`，
    其中一部分的完成条件是「某个世界标记被置上」——
    那正是这次修复要告诉模型的东西。

    ## ⚠️ 但"场景有目标"不等于"用例会检查目标"

    这是这条路上最容易踩空的地方，也是这份脚本存在的直接原因：

    **场景**里写着 `objectives`，但**用例**的 `expect` 才决定评测检查什么。
    实测 231 条用例里只有 118 条断言了世界状态，其中断言 `flags` 的 33 条、
    断言 `objectives_done` 的 18 条 —— 也就是说有相当一部分用例
    **结构上就看不见这次修复**（比如 `duet` 的生成用例只断言 `all_npcs_spoke`，
    而 `play_song` 目标要的是 `song_started` 这个标记）。

    把看不见的用例混进分母，真实效应会被稀释向零。
    所以下面还要算一层「敏感子集」（`sensitive_tiers`），
    报告的抬头用那一层，而不是全体。
    """
    from npc_agent.config import load_scenario  # 局部导入：脚本其它部分用不到

    wanted = set(case_ids)
    counts: dict[str, int] = {}
    for case in load_cases_raw():
        if case["id"] not in wanted:
            continue
        scenario = load_scenario(case["scenario"])
        for objective in scenario.get("objectives") or []:
            kind = _kind_of(objective.get("success_when"))
            counts[kind] = counts.get(kind, 0) + 1
    return counts


#: 用例的 `expect` 里这些键是**世界状态**断言 —— 只有它们能看见规划的结果。
_WORLD_KEYS = frozenset(
    {"player_has", "has_count", "placed", "flags", "no_flags", "objectives_done"}
)

#: 其中这两个直接钉在**世界标记**上，是这次修复最直接的靶子。
_FLAG_KEYS = frozenset({"flags", "objectives_done"})


def expect_keys(case_ids: list[str]) -> dict[str, set[str]]:
    """每条用例的 `expect` 用了哪些断言键。"""
    wanted = set(case_ids)
    return {
        case["id"]: set((case.get("expect") or {}).keys())
        for case in load_cases_raw()
        if case["id"] in wanted
    }


def sensitive_tiers(case_ids: list[str]) -> dict[str, list[str]]:
    """按「这条用例能不能看见规划的结果」分成三层。

    * `flag`  —— `expect` 里有 `flags` / `objectives_done`，直接钉世界标记。
      **这是最灵敏的一层**，修复的靶子就在这。
    * `world` —— `expect` 里有任意世界状态断言（含 `player_has` / `placed` 等）。
      规划错到拿不到东西、放不下方块，这一层也看得见。
    * `all`   —— 配对子集全体。**这一层包含结构上看不见修复的用例**，
      所以它的差会被稀释；放在这里是为了不让人以为只有敏感子集存在。
    """
    keys = expect_keys(case_ids)
    flag = [cid for cid in case_ids if keys.get(cid, set()) & _FLAG_KEYS]
    world = [cid for cid in case_ids if keys.get(cid, set()) & _WORLD_KEYS]
    return {"flag": flag, "world": world, "all": list(case_ids)}


# --------------------------------------------------------------------------- #
# 算


def _rate(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    passed = sum(1 for r in records if r["passed"])
    clean = [r for r in records if r["planner_failures"] == 0]
    clean_passed = sum(1 for r in clean if r["passed"])
    world = sum(1 for r in records if r["task"] >= 0.99)
    with_failures = [r for r in records if r["planner_failures"]]
    transport = sum(1 for r in with_failures if r.get("planner_last_is_transport"))
    return {
        "total": total,
        "passed": passed,
        "rate": (passed / total) if total else 0.0,
        "clean_total": len(clean),
        "clean_passed": clean_passed,
        "clean_rate": (clean_passed / len(clean)) if clean else 0.0,
        "world": world,
        "world_rate": (world / total) if total else 0.0,
        "planner_failures": sum(r["planner_failures"] for r in records),
        "cases_with_failures": len(with_failures),
        "cases_last_transport": transport,
        "cases_last_other": len(with_failures) - transport,
    }


def _is_prefix_of_load_order(ids: list[str], order: list[tuple[str, str]]) -> bool:
    """这组用例是不是「加载顺序的前 N 条」？

    ⚠️ 比的是**集合**，不是顺序。跑批是并发的，用例完成的顺序本来就是乱的 ——
    拿"完成顺序"去和加载顺序逐个比，一份规规矩矩用 `--limit 60` 跑出来的
    子集也会被判成"事后挑的"（第一版就是这么错的）。
    要验的是"它正好是前 N 条"，不是"它按顺序完成"。
    """
    return set(ids) == {cid for cid, _ in order[: len(ids)]}


#: 做"长度控制"时的轮数门槛。见 `fallback_confound()`。
MIN_TURNS = 6


def case_turns() -> dict[str, int]:
    """用例**定义里**的轮数（外生量）。

    ⚠️ 不能用转写长度当"用例有多难"的代理 —— 那是**内生**的：
    一条失败的用例会一直重试，转写自然更长。
    轮数写在用例文件里，和跑成什么样无关。
    """
    return {c["id"]: len(c.get("turns") or []) for c in load_cases_raw()}


def _split_by_fallback(
    records: list[dict[str, Any]], turns: dict[str, int], min_turns: int
) -> dict[str, Any]:
    """把一条臂拆成「干净组」和「回落组」，只保留轮数 ≥ `min_turns` 的用例。"""
    keep = [r for r in records if turns.get(r["case_id"], 0) >= min_turns]
    return {
        "clean": _rate([r for r in keep if r["planner_failures"] == 0]),
        "fallback": _rate([r for r in keep if r["planner_failures"] > 0]),
    }


def fallback_confound(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """回落到底把这次测量污染到什么程度。

    ## 为什么必须算这个

    「干净子集 100%」看起来像"模型一规划就成"，但它**可能只是
    "干净的都是短用例"** —— 规划调用是按 tick 发生的，用例越长调用越多、
    撞上超时的机会越大，也越难做完。所以两个变量是缠在一起的。

    实测（前 60 条配对跑批）：

    | | 干净组轮数中位 | 回落组轮数中位 |
    |---|---|---|
    | 修复前 | 2 | 11.5 |
    | 修复后 | 6 | 7 |

    干净组明显更短。所以这里做**长度控制**：只留轮数 ≥ `MIN_TURNS` 的用例，
    再比干净组和回落组。控制之后两组仍差很多，才说明问题出在回落本身。

    同时数一下**有多少翻转落在回落组里** —— 如果全部落在回落组，
    那这份 before/after 的差就主要是"超时打在哪里"的运气，
    而不是 prompt 改动的效果。
    """
    turns = case_turns()
    paired = [cid for cid in before if cid in after]

    flips = [
        cid
        for cid in paired
        if (before[cid]["task"] >= 0.99) != (after[cid]["task"] >= 0.99)
    ]
    flips_in_fallback = [
        cid
        for cid in flips
        if before[cid]["planner_failures"] > 0 or after[cid]["planner_failures"] > 0
    ]
    both_clean = [
        cid
        for cid in paired
        if before[cid]["planner_failures"] == 0 and after[cid]["planner_failures"] == 0
    ]

    def _share(records: dict[str, dict[str, Any]]) -> float:
        if not records:
            return 0.0
        return sum(1 for r in records.values() if r["planner_failures"] > 0) / len(records)

    return {
        "min_turns": MIN_TURNS,
        "before": _split_by_fallback(list(before.values()), turns, MIN_TURNS),
        "after": _split_by_fallback(list(after.values()), turns, MIN_TURNS),
        "share_before": _share(before),
        "share_after": _share(after),
        "n_flips": len(flips),
        "n_flips_in_fallback": len(flips_in_fallback),
        "flips": flips,
        "both_clean": {
            "n": len(both_clean),
            "before": _rate([before[c] for c in both_clean]),
            "after": _rate([after[c] for c in both_clean]),
        },
        # 干净组里两臂都是满分 ⇒ 这份用例集**分辨不出**修复前后。
        "cannot_tell_apart": (
            bool(both_clean)
            and all(before[c]["task"] >= 0.99 for c in both_clean)
            and all(after[c]["task"] >= 0.99 for c in both_clean)
        ),
    }


def compare(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    order: list[tuple[str, str]],
) -> dict[str, Any]:
    before_ids = list(before)
    # 配对子集 = 修复前那条臂跑过的用例（它是用 --limit 定的）。
    # 修复后那条臂如果少跑了其中某条，就按交集算，并把缺的条数报出来。
    paired_ids = [cid for cid in before_ids if cid in after]
    missing_after = [cid for cid in before_ids if cid not in after]

    pairs = []
    for cid in paired_ids:
        b, a = before[cid], after[cid]
        pairs.append(
            {
                "case_id": cid,
                "category": b["category"],
                "before": b,
                "after": a,
                "flip": ("fail→pass" if (a["passed"] and not b["passed"])
                         else "pass→fail" if (b["passed"] and not a["passed"])
                         else ""),
                "world_flip": ("未达成→达成" if (a["task"] >= 0.99 and b["task"] < 0.99)
                               else "达成→未达成" if (b["task"] >= 0.99 and a["task"] < 0.99)
                               else ""),
            }
        )

    by_category: dict[str, list[dict[str, Any]]] = {}
    for p in pairs:
        by_category.setdefault(p["category"], []).append(p)

    # 事先说好的失败特征：修复前模型不知道该置哪个标记，
    # 所以失败原因里应该出现「未达成标记 X」。这一条**可以证伪** ——
    # 如果修复前后这个计数没变，说明我给的机制解释是错的。
    flag_failures = {
        arm: sum(
            1
            for p in pairs
            if "未达成标记" in (p[arm]["task_detail"] or "")
        )
        for arm in ("before", "after")
    }

    tiers = sensitive_tiers(paired_ids)
    tier_rates = {
        name: {
            "before": _rate([before[c] for c in ids]),
            "after": _rate([after[c] for c in ids]),
        }
        for name, ids in tiers.items()
    }

    return {
        "paired_ids": paired_ids,
        "paired": pairs,
        "missing_after": missing_after,
        "is_preregistered": _is_prefix_of_load_order(before_ids, order),
        "condition_kinds": condition_kinds(paired_ids),
        "tiers": {name: len(ids) for name, ids in tiers.items()},
        "tier_rates": tier_rates,
        "failure_reasons": {
            "before": failure_reasons([before[c] for c in paired_ids]),
            "after": failure_reasons([after[c] for c in paired_ids]),
        },
        "flag_failures": flag_failures,
        "confound": fallback_confound(before, after),
        "before": _rate([before[c] for c in paired_ids]),
        "after": _rate([after[c] for c in paired_ids]),
        "after_all": _rate(list(after.values())),
        "by_category": {
            cat: {
                "before": _rate([p["before"] for p in ps]),
                "after": _rate([p["after"] for p in ps]),
            }
            for cat, ps in sorted(by_category.items())
        },
    }


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


def _pct_cell(rate: float, *, good_when_high: bool = True) -> str:
    cls = "pass" if (rate >= 0.99 if good_when_high else rate == 0) else "fail"
    return f'<td class="num {cls}">{rate:.1%}</td>'


def _delta_cell(old: float, new: float) -> str:
    d = new - old
    cls = "pass" if d > 0 else ("fail" if d < 0 else "muted")
    sign = "+" if d > 0 else ""
    return f'<td class="num {cls}">{sign}{d * 100:.1f}pp</td>'


def _summary_table(rows: list[tuple[str, dict[str, Any], dict[str, Any]]]) -> str:
    head = (
        "<th>用例组</th><th>n</th>"
        "<th>修复前通过</th><th>修复后通过</th><th>差</th>"
        "<th>修复前世界状态</th><th>修复后世界状态</th>"
    )
    body = []
    for label, b, a in rows:
        body.append(
            f'<tr><td class="name">{label}</td><td class="num">{a["total"]}</td>'
            f"{_pct_cell(b['rate'])}"
            f"{_pct_cell(a['rate'])}"
            f"{_delta_cell(b['rate'], a['rate'])}"
            f"{_pct_cell(b['world_rate'])}"
            f"{_pct_cell(a['world_rate'])}</tr>"
        )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _planner_health_table(b: dict[str, Any], a: dict[str, Any]) -> str:
    head = "<th>口径</th><th>修复前</th><th>修复后</th><th>说明</th>"

    def _kind(arm: dict[str, Any]) -> str:
        if not arm["cases_with_failures"]:
            return "0"
        return (
            f'{arm["cases_with_failures"]} 条'
            f'（传输 {arm["cases_last_transport"]}／其他 {arm["cases_last_other"]}）'
        )

    rows = [
        (
            "规划回落次数",
            b["planner_failures"],
            a["planner_failures"],
            "模型没能给出可解析的计划、框架<strong>静默回落</strong>启发式规划器的次数",
        ),
        (
            "涉及用例数",
            _kind(b),
            _kind(a),
            "按<strong>最后一条</strong>错误归类；检查点不存全部错误，所以这是近似",
        ),
        (
            "干净子集 n",
            b["clean_total"],
            a["clean_total"],
            "该臂里 <code>planner_failures == 0</code> 的用例条数",
        ),
        (
            "干净子集通过率",
            f"{b['clean_rate']:.1%}",
            f"{a['clean_rate']:.1%}",
            "只算「模型真的规划成功了」的那些 —— 这是<strong>保守</strong>的那个数",
        ),
    ]
    body = "".join(
        f'<tr><td class="name">{k}</td><td class="num">{x}</td>'
        f'<td class="num">{y}</td><td class="muted">{note}</td></tr>'
        for k, x, y, note in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _transport_note(b: dict[str, Any], a: dict[str, Any]) -> str:
    """回落到底是谁的锅 —— 实测结论写进报告，别让读者自己猜。"""
    total_kind = b["cases_with_failures"] + a["cases_with_failures"]
    transport = b["cases_last_transport"] + a["cases_last_transport"]
    other = b["cases_last_other"] + a["cases_last_other"]
    if not total_kind:
        return ""
    if transport and not other:
        return (
            '<div class="warn"><strong>这一栏量的是端点健康，不是模型能力。</strong>'
            f"两条臂加起来 {total_kind} 条用例出现过规划回落，"
            f"其中 <strong>{transport} 条的末次错误是传输层故障</strong>"
            "（<code>TimeoutError</code> / <code>RemoteDisconnected</code>），"
            "<strong>0 条是解析失败</strong>。"
            "也就是说这些回落不是「模型给出了读不懂的计划」，"
            "而是「这一次请求没回来」 —— 框架把它当成规划失败、回落启发式，"
            "于是网络抖动被记进了这一栏。"
            "两条臂的抖动次数不一样，所以<strong>总通过率的差里混着端点的运气</strong>；"
            "干净子集那一行才是把这份运气扣掉之后的读数。</div>"
        )
    return (
        '<div class="warn"><strong>回落原因不止一种。</strong>'
        f"末次错误里传输层 {transport} 条、其他 {other} 条。"
        "「其他」里可能包含真正的解析失败 —— 那才是模型的问题。"
        "引用这一栏时请把两类分开说。</div>"
    )


#: 失败原因的分类规则。顺序有意义：先匹配到的先算。
#:
#: 这一栏是这份报告里**最有诊断价值**的部分：它把"通过率涨了几个点"
#: 拆成"哪一种失败模式消失了、哪一种还在"。
#: 实测里最有用的两条是 `未达成标记`（这次修复的靶子，应该消失）
#: 和 `只有 N 个，需要 M`（数量规划，**两条臂都有**，说明是另一个根因）。
_FAILURE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("未达成标记", "未达成标记（<code>flag</code>）"),
    ("只有", "数量不足（<code>has_count</code>）"),
    ("缺少", "玩家手里缺东西（<code>player_has</code>）"),
    ("没有把", "方块没放对位置（<code>placed</code>）"),
    ("联合目标", "联合目标没翻成 done（<code>objectives_done</code>）"),
    ("不该出现标记", "多置了标记（<code>no_flags</code>）"),
)


def classify_failure(task_detail: str) -> str:
    """把 `details.task` 的失败原因归成一类。认不出就原样返回，不硬猜。"""
    text = task_detail or ""
    if not text or "符合预期" in text:
        return "（世界状态达成）"
    for needle, label in _FAILURE_PATTERNS:
        if needle in text:
            return label
    return f"其他：{text[:60]}"


def failure_reasons(records: list[dict[str, Any]]) -> dict[str, int]:
    """按失败模式计数（只算世界状态没达成的那些用例）。"""
    counts: dict[str, int] = {}
    for r in records:
        if r["task"] >= 0.99:
            continue
        label = classify_failure(r["task_detail"])
        counts[label] = counts.get(label, 0) + 1
    return counts


_TIER_LABELS = {
    "flag": "敏感层：断言世界标记（<code>flags</code> / <code>objectives_done</code>）",
    "world": "可见层：断言任意世界状态（含 <code>player_has</code> / <code>placed</code>）",
    "all": "全体配对子集（<strong>含结构上看不见修复的用例</strong>）",
}


def _verdict(before: int, after: int) -> tuple[str, str]:
    """把「前后各几条」翻成一句判读，并给出配色。

    抽成纯函数是因为第一版把方向写反了（1 → 2 印成「减少」）——
    这种错不会崩，只会让人读反结论，所以单独测。
    """
    if before and after == 0:
        return "消失了", "pass"
    if after == 0 and before == 0:
        return "没变", "muted"
    if before == 0:
        return "新增", "fail"
    if after < before:
        return f"减少 {before - after} 条", "pass"
    if after > before:
        return f"增加 {after - before} 条", "fail"
    return "没变", "muted"


def _failure_section(cmp: dict[str, Any]) -> str:
    """失败模式的前后对比 —— 「哪种失败消失了、哪种还在」。"""
    b = cmp["failure_reasons"]["before"]
    a = cmp["failure_reasons"]["after"]
    labels = sorted(set(b) | set(a), key=lambda k: -(b.get(k, 0) + a.get(k, 0)))
    rows = []
    for label in labels:
        if label == "（世界状态达成）":
            continue
        nb, na = b.get(label, 0), a.get(label, 0)
        text, cls = _verdict(nb, na)
        rows.append(
            f'<tr><td class="name">{label}</td>'
            f'<td class="num">{nb}</td><td class="num">{na}</td>'
            f'<td class="{cls}">{text}</td></tr>'
        )
    if not rows:
        return "<h2>失败模式</h2><div class='card muted'>配对子集里没有世界状态未达成的用例。</div>"
    head = "<th>世界状态失败原因</th><th>修复前</th><th>修复后</th><th>判读</th>"
    return (
        "<h2>失败模式：哪种失败消失了、哪种还在</h2>"
        f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _confound_section(cmp: dict[str, Any], conf: dict[str, Any]) -> str:
    """回落把这份对比污染到什么程度 —— 这份报告里最该读的一节。

    结论可能是"这份跑批测不出修复"，那也要明说。一份诚实的
    "测不出来，原因是这些" 比一份编出来的 "+18pp" 有用得多。
    """
    mt = conf["min_turns"]
    rows = []
    for arm, label in (("before", "修复前"), ("after", "修复后")):
        d = conf[arm]
        c, f = d["clean"], d["fallback"]
        rows.append(
            f'<tr><td class="name">{label}</td>'
            f'<td class="num">{c["total"]}</td>'
            f'<td class="num">{c["world"]}/{c["total"]}</td>'
            f'<td class="num">{f["total"]}</td>'
            f'<td class="num">{f["world"]}/{f["total"]}</td></tr>'
        )
    table = (
        f"<table><thead><tr><th>臂</th>"
        f"<th>干净组 n（轮数 ≥ {mt}）</th><th>干净组世界状态</th>"
        f"<th>回落组 n（轮数 ≥ {mt}）</th><th>回落组世界状态</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )

    share = (
        f'<div class="sub">至少发生过一次规划回落的用例：'
        f"修复前 <strong>{conf['share_before']:.0%}</strong>、"
        f"修复后 <strong>{conf['share_after']:.0%}</strong>。"
        "回落 = 那一次规划调用超时/断连，框架<strong>静默换成启发式规划器</strong>。"
        "所以「把规划交给模型」这件事，在这一半用例上其实没发生。</div>"
    )

    flips = conf["n_flips"]
    in_fb = conf["n_flips_in_fallback"]
    bc = conf["both_clean"]

    parts = [
        "<h2>⚠️ 回落把这份对比污染到什么程度</h2>",
        '<div class="warn"><strong>先读这一节，再读上面的数字。</strong>'
        "规划调用是按 tick 发生的：用例越长，调用越多，撞上超时的机会越大，"
        "而越长的用例本来也越难做完 —— 两个变量缠在一起。"
        "所以「干净子集满分」有可能只是「干净的都是短用例」。"
        "下面做<strong>长度控制</strong>（只留轮数 ≥ "
        f"{mt} 的用例）再看一遍。</div>",
        share,
        table,
        f'<div class="card"><ul>'
        f"<li>世界状态翻转的用例共 <strong>{flips} 条</strong>，"
        f"其中 <strong>{in_fb} 条</strong>在至少一条臂上发生过回落。</li>"
        f"<li>两条臂都干净的配对用例 <strong>{bc['n']} 条</strong>："
        f"世界状态 {bc['before']['world']}/{bc['n']} → {bc['after']['world']}/{bc['n']}。</li>"
        "</ul></div>",
    ]

    if in_fb == flips and flips:
        parts.append(
            '<div class="warn"><strong>全部翻转都落在回落组里。</strong>'
            "也就是说：这份 before/after 的差，主要是"
            "<strong>「超时打在哪一次规划调用上」的运气</strong>，"
            "而不是 prompt 改动的效果。"
            "一次超时发生在关键 tick 上，目标就悬着没做完 —— "
            "这跟 prompt 里有没有 <code>success_when</code> 是两件事。</div>"
        )

    if conf["cannot_tell_apart"]:
        parts.append(
            '<div class="warn"><strong>结论：这份跑批分辨不出修复前后。</strong>'
            f"两边都干净的 {bc['n']} 条用例上，修复前后<strong>都是满分</strong> —— "
            "说明这些用例<strong>结构上不经过那个缺陷</strong>。"
            "要测出修复，得让敏感用例（需要世界标记那些）"
            "<strong>真的由模型规划到底</strong>，而不是半路回落。"
            "<br>所以这次修复的证据仍然是那个<strong>受控的 duet 实验</strong>"
            "（1/4 → 10/10，见 <code>docs/ENGINEERING.md</code>）；"
            "这份跑批的价值在于<strong>证明了批量测量现在测不出来</strong>，"
            "以及<strong>为什么</strong>。</div>"
        )

    return "".join(parts)


def _tier_section(cmp: dict[str, Any]) -> str:
    """三层敏感度分开报 —— 这是这份报告的核心表。"""
    rows = []
    for name in ("flag", "world", "all"):
        r = cmp["tier_rates"][name]
        b, a = r["before"], r["after"]
        rows.append(
            f'<tr><td class="name">{_TIER_LABELS[name]}</td>'
            f'<td class="num">{a["total"]}</td>'
            f'<td class="num">{b["passed"]}/{b["total"]}</td>'
            f'<td class="num">{a["passed"]}/{a["total"]}</td>'
            f"{_delta_cell(b['rate'], a['rate'])}"
            f'<td class="num">{b["world"]}/{b["total"]}</td>'
            f'<td class="num">{a["world"]}/{a["total"]}</td>'
            "</tr>"
        )
    head = (
        "<th>用例层</th><th>n</th><th>修复前通过</th><th>修复后通过</th>"
        "<th>差</th><th>修复前世界状态</th><th>修复后世界状态</th>"
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _tier_note(cmp: dict[str, Any]) -> str:
    t = cmp["tiers"]
    if t["flag"] == t["all"]:
        return ""
    return (
        '<div class="warn"><strong>为什么要分层看。</strong>'
        f"配对子集 {t['all']} 条里，只有 <strong>{t['world']} 条</strong>断言了世界状态，"
        f"其中只有 <strong>{t['flag']} 条</strong>直接钉在世界标记上。"
        "剩下的用例（记忆、人设、安全、发言调度）<strong>结构上就看不见规划的结果</strong> —— "
        "它们就算通过，也不能作为「规划修好了」的证据。"
        "把三层混在一个分母里算总通过率，真实效应会被稀释向零；"
        "所以抬头那一行用的是敏感层，而不是全体。"
        "这不是「挑一个好看的子集」：分层依据是<strong>用例自己的 <code>expect</code> 写了什么</strong>，"
        "而且三层都印在表里，读者可以自己换算。</div>"
    )


def _prediction_section(cmp: dict[str, Any]) -> str:
    """把「这次修复该动哪一列、失败长什么样」在数字之前说清楚。

    这一段是这份报告里最该保留的部分：它把"数字变好了"从**故事**
    变成**预测**。预测在跑批之前就定下来（依据是 `success_when` 的类型分布），
    所以它是可以证伪的 —— 如果翻转表里没出现「未达成标记 X」，
    那我对机制的解释就是错的，哪怕总通过率涨了。
    """
    kinds = cmp["condition_kinds"]
    n_objectives = sum(kinds.values())
    # 递归数，别用"顶层是 flag/all_of"近似 —— 那会多算一条（见 `_flags_of`）。
    flag_like = 0
    for case in load_cases_raw():
        if case["id"] not in set(cmp["paired_ids"]):
            continue
        from npc_agent.config import load_scenario

        for objective in load_scenario(case["scenario"]).get("objectives") or []:
            if _flags_of(objective.get("success_when")):
                flag_like += 1
    b = cmp["flag_failures"]["before"]
    a = cmp["flag_failures"]["after"]

    if not n_objectives:
        return ""

    parts = [
        "<h2>事先说好的预测（跑批之前定的）</h2>",
        f'<div class="card">配对子集里一共 <strong>{n_objectives} 个目标</strong>，'
        "它们的完成条件类型是："
        + "、".join(f"<code>{k}</code>×{n}" for k, n in sorted(kinds.items()))
        + f"。其中 <strong>{flag_like} 个的完成条件里含世界标记</strong>"
        "（标记可能嵌在 <code>all_of</code> 里，所以这个数是递归数出来的）。"
        "<br>修复前，规划 prompt 只给了模型目标的<strong>文字</strong>"
        "（「为玩家演奏一首曲子」），没给<strong>完成条件</strong>"
        "（「世界标记 <code>song_started</code> 被置上」）。"
        "模型于是规划出「演奏」这个动作，但没有任何一步去置那个标记 —— "
        "听起来做完了，机器判定没做完。"
        "<br>所以预测是两条，都可以证伪："
        "<strong>(1)</strong> 修复前失败原因里应出现「未达成标记 X」；"
        "<strong>(2)</strong> 修复后这一类应该显著减少。"
        "</div>",
        _prediction_table(b, a, cmp["before"]["total"]),
    ]
    return "".join(parts)


def _prediction_table(before_count: int, after_count: int, n: int) -> str:
    head = "<th>预测</th><th>修复前</th><th>修复后</th><th>判读</th>"
    delta = after_count - before_count
    verdict = (
        "符合预测"
        if delta < 0
        else ("没有变化 —— 机制解释不成立" if delta == 0 else "与预测相反")
    )
    cls = "pass" if delta < 0 else "fail"
    rows = (
        f'<tr><td class="name">失败原因含「未达成标记」的用例数</td>'
        f'<td class="num">{before_count} / {n}</td>'
        f'<td class="num">{after_count} / {n}</td>'
        f'<td class="{cls}">{verdict}</td></tr>'
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def _flip_table(pairs: list[dict[str, Any]], key: str, title: str) -> str:
    flips = [p for p in pairs if p[key]]
    if not flips:
        return (
            f"<h2>{title}</h2>"
            '<div class="card muted">这一组里没有翻转的用例。</div>'
        )
    # 两侧的原因**都**要印。只印"修复前的失败原因"的话，
    # `达成→未达成` 那一行会显示「世界状态符合预期」——
    # 那是它成功时的原因，读起来像在说反话。
    head = (
        "<th>用例</th><th>分类</th><th>方向</th>"
        "<th>修复前</th><th>修复后</th>"
    )
    body = ""
    for p in flips:
        good = p[key].endswith("→达成") or p[key] == "fail→pass"
        body += (
            f'<tr><td class="name sample">{p["case_id"]}</td>'
            f'<td>{p["category"]}</td>'
            f'<td class="{"pass" if good else "fail"}">{p[key]}</td>'
            f'<td class="muted">{p["before"]["task_detail"] or "（无）"}</td>'
            f'<td class="muted">{p["after"]["task_detail"] or "（无）"}</td></tr>'
        )
    return f"<h2>{title}（{len(flips)} 条）</h2><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render(cmp: dict[str, Any], before_label: str, after_label: str) -> str:
    b, a = cmp["before"], cmp["after"]
    after_all = cmp["after_all"]
    n = b["total"]

    # 抬头用**敏感层**（断言世界标记的那些），因为只有它们看得见这次修复。
    # 全体配对子集的数字紧随其后 —— 不藏，但也不让它当主角。
    sensitive = cmp["tier_rates"]["flag"]
    sb, sa = sensitive["before"], sensitive["after"]

    meta = cmp.get("meta") or {}
    declared = meta.get("after_total")
    if declared and declared > after_all["total"]:
        arm_note = (
            f"修复后那一臂跑到 <strong>{after_all['total']}/{declared} 条</strong>"
            f"（<strong>还没跑完</strong>，这个数不能当整臂读数），"
            f"其中通过 {after_all['passed']} 条（{after_all['rate']:.1%}）。"
        )
    else:
        arm_note = (
            f"修复后那一臂总共跑了 {after_all['total']} 条，"
            f"通过 {after_all['passed']} 条（{after_all['rate']:.1%}）。"
        )

    headline = (
        f"敏感层 <strong>{sb['total']} 条</strong>（断言世界标记的用例）："
        f"世界状态达成 <strong>{sb['world']}/{sb['total']} → {sa['world']}/{sa['total']}</strong>，"
        f"总通过 <strong>{sb['passed']}/{sb['total']} → {sa['passed']}/{sa['total']}</strong>。"
        f"<br>全体配对子集 {n} 条：通过 {b['rate']:.1%} → {a['rate']:.1%}"
        f"（其中只有 {cmp['tiers']['world']} 条断言了世界状态，"
        "其余结构上看不见这次修复）。"
        f"<br>{arm_note}"
    )

    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>模型自己规划：修复前 vs 修复后</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        '<div class="wrap">',
        "<h1>模型自己规划：修复前 vs 修复后</h1>",
        # ⚠️ 这一行里必须有一个**能被 `report_index.case_count()` 读出来**的覆盖数，
        # 否则 README 的报告表没法写「这份覆盖多少条」，而那条护栏会红。
        # 取配对子集的条数（这份报告的主体就是它），不是修复后那一臂的总数。
        f'<div class="sub">把规划交给模型之后，NPC 到底能不能把一件事做完。'
        f"共 {cmp['tiers']['all']} 条用例的配对前后对比，"
        "只差规划 prompt 里那几行（<code>success_when</code>）。</div>",
        f'<div class="headline">{headline}</div>',
    ]

    if not cmp["is_preregistered"]:
        ids = cmp["paired_ids"]
        prefix = {cid for cid, _ in load_order()[: len(ids)]}
        in_prefix = sum(1 for cid in ids if cid in prefix)
        parts.append(
            '<div class="warn"><strong>没法确认这是一个预先登记的子集。</strong>'
            f"修复前那条臂跑了 {len(ids)} 条，其中只有 {in_prefix} 条落在"
            f"加载顺序的前 {len(ids)} 条里 —— 所以它的用例集不是加载顺序的前缀。"
            "两种可能：<strong>(1)</strong> 跑批还没跑完（并发完成的顺序本来就是乱的），"
            "跑完之后这条警告会自己消失；<strong>(2)</strong> 这个子集是事后挑的。"
            "跑完之后如果它还在，那就是第二种，引用这些数字时要把这句话一起带上。</div>"
        )

    parts.append(
        '<div class="warn"><strong>两条臂只差一个变量。</strong>'
        f"同一条用例集、同一个模型、同样的 <code>--no-speech</code>"
        "（模型只负责规划，台词走模板）。修复前那条臂跑在父提交 "
        f"<code>{before_label}</code> 的 worktree 里，修复后是 "
        f"<code>{after_label}</code>。"
        "所以这里的差可以归到规划 prompt 上，而不是「模型今天心情好」。</div>"
    )

    parts.append("<h2>配对结果</h2>")
    parts.append(_confound_section(cmp, cmp["confound"]))
    parts.append(_tier_section(cmp))
    parts.append(_tier_note(cmp))
    rows = [("全部（配对子集）", b, a)]
    for cat, pair in cmp["by_category"].items():
        rows.append((cat, pair["before"], pair["after"]))
    parts.append("<h3>按分类</h3>")
    parts.append(_summary_table(rows))
    parts.append(
        '<div class="sub">「世界状态」= 用例期望的世界状态真的达成了'
        "（玩家手里真有那件东西 / 世界标记真的置上了 / 联合目标真的翻成 done），"
        "就是六维里的 <code>task</code> 那一维。它比总通过率更贴近这次修复 —— "
        "这次修的是「模型不知道做到什么才算做完」，所以它应该先动这一列。</div>"
    )

    parts.append(_prediction_section(cmp))

    parts.append("<h2>规划本身的健康状况</h2>")
    parts.append(_planner_health_table(b, a))
    parts.append(_transport_note(b, a))
    parts.append(
        '<div class="warn"><strong>为什么要单独列这一栏。</strong>'
        "框架在<strong>规划解析失败时会静默回落到启发式规划器</strong>。"
        "回落不等于失败 —— 启发式规划器也能完成任务，只是那不是模型的功劳。"
        "如果两条臂的回落次数差很多，直接比总通过率就会把"
        "「这一次请求没回来」记成「规划质量变差」。"
        "所以干净子集那一行才是保守读数：它只算两边都真的让模型规划成功的用例。</div>"
    )

    parts.append(_flip_table(cmp["paired"], "world_flip", "世界状态翻转的用例"))
    parts.append(_failure_section(cmp))
    parts.append(
        '<div class="sub">右列是修复前的<strong>机器给出的失败原因</strong>'
        "（<code>details.task</code>），不是我的解读。"
        "如果修复真的按预期起作用，这里应该出现"
        "「未达成标记 X」「玩家手里缺少 Y」这类句子 —— "
        "而修复后同一条用例不再出现在这一列。</div>"
    )
    parts.append(_flip_table(cmp["paired"], "flip", "总通过率翻转的用例"))

    parts.append("<h2>这份报告不能说明什么</h2>")
    parts.append(
        '<div class="card"><ul>'
        f"<li>配对子集只有 <strong>{n} 条</strong>，不是全部 231 条。"
        "它够看出一个大的效应，不够给小数点后一位的结论。</li>"
        "<li>只有<strong>一个模型</strong>（"
        f"<code>{'kimi-k2.7-code'}</code>）和<strong>一次跑批</strong>。"
        "模型采样有方差，这份报告没有重复跑，所以没有置信区间。</li>"
        "<li>这是<strong>规划</strong>这一条路径的读数。台词仍然走模板"
        "（<code>--no-speech</code>），所以它完全不代表「接上模型台词之后」的表现。</li>"
        "<li>它量的是<strong>目标有没有做完</strong>，不是对话好不好。"
        "复读、语气、趣味性不在这一列。</li>"
        "<li>旧的那份全量报告 <code>docs/batch_planner.html</code> 描述的是修复前的代码，"
        "两份报告的数字不可混用。</li>"
        "</ul></div>"
    )

    parts.append(
        '<div class="foot">重生成：'
        "<code>python scripts/measure_planner_batch.py --before &lt;父提交检查点&gt; "
        "--after &lt;当前检查点&gt; --html docs/planner_batch.html</code>"
        "<br>每次重跑要真实模型额度，所以这份是<strong>一次跑批的快照</strong>。"
        "<br>根因分析与回归测试：<code>docs/ENGINEERING.md</code>、"
        "<code>tests/test_planner.py</code>。</div>"
    )
    parts += ["</div>", "</body>", "</html>", ""]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="渲染规划修复的前后对比报告")
    parser.add_argument("--before", required=True, help="修复前那条臂的检查点/报告")
    parser.add_argument("--after", required=True, help="修复后那条臂的检查点/报告")
    parser.add_argument("--html", default="docs/planner_batch.html", help="输出 HTML；none 表示不写")
    parser.add_argument("--json", default=None, help="把对比读数落一份 JSON")
    parser.add_argument("--before-label", default="4c6569c", help="修复前的提交号（写进报告）")
    parser.add_argument("--after-label", default="HEAD", help="修复后的提交号（写进报告）")
    args = parser.parse_args()

    before = load_arm(args.before)
    after = load_arm(args.after)
    _raw_before, meta_before = _read_arm(args.before)
    _raw_after, meta_after = _read_arm(args.after)
    order = load_order()
    print(f"修复前 {len(before)} 条｜修复后 {len(after)} 条｜用例总数 {len(order)}", flush=True)

    cmp = compare(before, after, order)
    cmp["meta"] = {
        "before_total": meta_before["declared_total"] or len(before),
        "after_total": meta_after["declared_total"] or len(after),
    }
    b, a = cmp["before"], cmp["after"]
    print(
        f"配对 {b['total']} 条：通过 {b['passed']} → {a['passed']}"
        f"（{b['rate']:.1%} → {a['rate']:.1%}）"
        f"｜世界状态 {b['world']} → {a['world']}（{b['world_rate']:.1%} → {a['world_rate']:.1%}）",
        flush=True,
    )
    print(f"预先登记子集：{'是' if cmp['is_preregistered'] else '否'}", flush=True)
    for cat, pair in cmp["by_category"].items():
        print(
            f"  {cat:<12} n={pair['after']['total']:<3}"
            f" 通过 {pair['before']['passed']} → {pair['after']['passed']}"
            f"｜世界状态 {pair['before']['world']} → {pair['after']['world']}",
            flush=True,
        )

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "paired_ids": cmp["paired_ids"],
                    "is_preregistered": cmp["is_preregistered"],
                    "missing_after": cmp["missing_after"],
                    "before": cmp["before"],
                    "after": cmp["after"],
                    "after_all": cmp["after_all"],
                    "by_category": cmp["by_category"],
                    "world_flips": [p["case_id"] for p in cmp["paired"] if p["world_flip"]],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n对比读数 → {args.json}")

    if args.html and args.html != "none":
        out = Path(args.html)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render(cmp, args.before_label, args.after_label), encoding="utf-8")
        print(f"报告 → {out}")


if __name__ == "__main__":
    main()
