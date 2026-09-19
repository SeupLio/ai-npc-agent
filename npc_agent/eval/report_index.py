"""`docs/` 里那些报告的**索引**：谁是谁、覆盖多少条、回答什么问题。

## 为什么单独成一个模块

这份清单原来只活在 `scripts/regen_docs.py` 里，于是只有"重生成脚本"认识它。
但同一个事实现在有三个消费方：

1. `scripts/regen_docs.py` —— 知道**怎么重生成**（命令 + 输出文件名）
2. `tests/test_docs_freshness.py` —— 知道**哪些必须和代码一致**
3. `npc_agent/studio.py` 的报告门户 —— 知道**该给用户看什么**

三份手抄必然漂移（这个项目已经因为"手抄的清单"栽过好几次：`--only` 的
帮助文本漏了 `sensitivity`、README 的覆盖列没人核对）。所以收拢到这里，
**能推导的别手抄**。

⚠️ 这里**不写用例条数**。条数会变（15 → 228 → 231），写死就会像文档里的数字
一样过期。要读就读报告自己印的那一行 —— `case_count()`。
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"

# 报告里"覆盖了多少条用例"有三种写法 —— 都是历史原因，不统一。
# 这里按顺序试，**解析不出来就返回 None**，绝不返回 0：
# 0 会被下游当成一个合法的覆盖数，而"我没解析出来"必须被当成错误。
_CASE_COUNT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(\d+)\s*条自建用例"),          # 跑批报告：副标题
    re.compile(r"共\s*(\d+)\s*条用例"),          # 跨世界报告：副标题
    re.compile(r"通过\s*<strong>(\d+)/\d+</strong>"),  # 对照/消融：头条的分母
)

# 离线可复现的报告：命令 → 输出文件名
OFFLINE_REPORTS: dict[str, tuple[tuple[str, ...], str]] = {
    "ablation": (("ablate",), "ablation.html"),
    "worlds": (("worlds",), "worlds.html"),
    # 敏感性报告也是离线可复现的：它只跑启发式路径，不调模型。
    # 它比另外两份更该被钉住 —— 它是「基线全绿」这句话的**证据**，
    # 过期了就等于在给一个已经不准的结论背书。
    "sensitivity": (("sensitivity",), "sensitivity.html"),
}

# 一次跑批的快照：需要模型 + 额度，**故意不自动化**。
# 列在这里是为了让"哪些不能重生成"变成一个可被测试读取的事实，
# 而不是散落在文档里的口头说明。
SNAPSHOT_REPORTS: dict[str, str] = {
    "batch_model.html": "compare --models kimi-k2.7-code（228 条跑批 + 裁判）",
    "batch_planner.html": "compare --models kimi-k2.7-code（规划也交给模型）",
    "comparison.html": "compare --models kimi-k2.7-code（离线 vs 模型对照）",
    "multi_npc.html": "compare --models kimi-k2.7-code --category multi_npc",
    # 离线那半是秒级可复现的，但模型那半要额度 —— 整份按快照对待，
    # 否则 `test_the_classification_does_not_drift_from_reality` 会红
    # （报告里嵌着模型名，就不可能是"不需要模型就能重生成"）。
    "repetition.html": "scripts/measure_repetition.py（离线 + 模型 20 轮）",
    # 两条臂各要真实模型额度，而且修复前那条臂跑在父提交的 worktree 里 ——
    # 不可能"不联网重生成"。
    "planner_batch.html": "scripts/measure_planner_batch.py（父提交 vs 当前，各跑一遍）",
}

# 给报告门户用的两句话：**叫什么**、**回答什么问题**。
# 键必须覆盖 docs/ 里每一个 .html —— 漏一个就没有标题可显示，
# 而"漏了"必须被测试抓到，不能静默显示成文件名。
REPORT_TITLES: dict[str, str] = {
    "ablation.html": "记忆策略消融",
    "worlds.html": "跨世界覆盖",
    "sensitivity.html": "评测敏感性（最该先看的一份）",
    "batch_model.html": "全模型跑批 · 台词侧",
    "batch_planner.html": "全模型跑批 · 规划侧",
    "comparison.html": "离线启发式 vs 真实模型",
    "multi_npc.html": "多 NPC 接上模型的对话样本",
    "repetition.html": "复读：只看单句的评测看不见的缺陷",
    "planner_batch.html": "规划修复 · 配对前后对比",
}

REPORT_ANSWERS: dict[str, str] = {
    "ablation.html": "记忆系统的净收益（五种检索策略对照）",
    "worlds.html": "「环境无关」的覆盖面证据：同一套 Agent 跑在两个世界上",
    "sensitivity.html": "满分是不是「护栏从不报警」—— 注入缺陷，要求评测掉分",
    "batch_model.html": "真实模型读数 + 裁判校准 + 留出集（含「已饱和」提示）",
    "batch_planner.html": "把规划也交给模型 —— 掉 11 个点，且报告自标「这批不干净」",
    "comparison.html": "用例集还小的时候跑的，不可与现在的跑批相比",
    "multi_npc.html": "同上，只作存档",
    "repetition.html": "同一段对话里 NPC 有多少话是之前说过的（修复前 61% → 修复后 0%）",
    "planner_batch.html": "规划 prompt 补上完成条件之后，同一个配对子集上的前后对比"
    "（含预先写下的失败特征、分层口径、以及**没修好的另一个根因**）",
}

# 报告里嵌了**时长**（`0.18s` / `14s` / `1184s`），它天然每次都不一样。
# 所以"离线可复现"只能定义成"**除时长外**逐字节一致"。
# 不把时长抠掉，护栏就只能常年红着 —— 于是没人再看它，等于没有护栏。
_DURATION_CELL = re.compile(r'(<td class="num[^"]*">)\s*\d+(?:\.\d+)?s\s*(</td>)')
DURATION_PLACEHOLDER = "⟨时长⟩"


def normalize_report(html: str) -> str:
    """把报告里天然会变的字段抹平，只留下**应该稳定**的内容。

    ⚠️ **只抹时长。** 别顺手把别的数字也抹掉 —— 那些数字正是要守的东西
    （通过数、指标均值、用例名）。抹多了这条护栏就变成永真式。
    """
    return _DURATION_CELL.sub(
        lambda m: f"{m.group(1)}{DURATION_PLACEHOLDER}{m.group(2)}", html
    )


def case_count(html: str) -> int | None:
    """从报告里解析出它覆盖的用例数。

    为什么值得解析而不是写死：README 里写了不少"这份是 12 条""那份是 2 条"，
    但**没有任何东西把这些说法和报告本身绑在一起**。报告哪天被重新生成、
    用例集变了，那些说法就会静默变成假话 —— 和 `ablation.html` 那次一模一样。
    """
    for pattern in _CASE_COUNT_PATTERNS:
        match = pattern.search(html)
        if match:
            return int(match.group(1))
    return None


def kind_of(filename: str) -> str:
    """`offline`（不联网可重生成）还是 `snapshot`（一次跑批的快照）。"""
    return "snapshot" if filename in SNAPSHOT_REPORTS else "offline"


def index(docs_dir: Path | None = None) -> list[dict[str, object]]:
    """扫一遍 `docs/`，给出报告门户要的全部信息。

    **以文件系统为准，不以清单为准**：清单里写了但文件不在 ⇒ 跳过；
    文件在但清单里没写 ⇒ 也列出来（标题退回文件名），
    由测试去抓"清单漏了"这件事。反过来做的话，一个孤儿文件会**彻底隐身**，
    而它恰恰是最该被看见的（那正是 `ablation.html` 停在 12 条用例时代的成因）。
    """
    folder = docs_dir or DOCS_DIR
    out: list[dict[str, object]] = []
    for path in sorted(folder.glob("*.html")):
        name = path.name
        try:
            html = path.read_text(encoding="utf-8")
        except OSError:
            continue
        count = case_count(html)
        out.append(
            {
                "file": name,
                "url": f"/report/{name}",
                "title": REPORT_TITLES.get(name, name),
                "answers": REPORT_ANSWERS.get(name, ""),
                # 解析不出来就明说，**不要填 0** —— 0 会被读成一个合法的覆盖数。
                "coverage": f"{count} 条" if count is not None else "（未标注）",
                "cases": count,
                "kind": kind_of(name),
                "known": name in REPORT_TITLES,
            }
        )
    return out
