"""重新生成 `docs/` 里**离线可复现**的报告。

## 为什么要有这个脚本

`docs/` 里放了两种完全不同的东西，以前文档里没区分：

| 种类 | 谁能重生成 | 例子 |
|---|---|---|
| **离线可复现** | 任何人，不需要模型、不需要网络 | `ablation.html`、`worlds.html` |
| **一次跑批的快照** | 要有模型 + 额度 | `batch_*.html`、`comparison.html`、`multi_npc.html` |

区别很要紧：**离线那份如果和代码不一致，就是假的** ——
读者会以为它是当前代码的输出。快照那份冻结在某个时间点是合理的，
但必须说清"它是什么时候、用什么模型跑的"。

这个脚本只负责第一类。第二类重生成要烧额度，不该被自动化。

用法：
    python scripts/regen_docs.py                     # 写回 docs/
    python scripts/regen_docs.py --out /tmp/x        # 写到别处（给护栏比对用）
    python scripts/regen_docs.py --only worlds       # 只生成一个
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 报告里会嵌**时长**（`0.18s` / `14s` / `1184s`），它天然每次都不一样。
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


# 报告里"覆盖了多少条用例"有三种写法 —— 都是历史原因，不统一。
# 这里按顺序试，**解析不出来就返回 None**，绝不返回 0：
# 0 会被下游当成一个合法的覆盖数，而"我没解析出来"必须被当成错误。
_CASE_COUNT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(\d+)\s*条自建用例"),          # 跑批报告：副标题
    re.compile(r"共\s*(\d+)\s*条用例"),          # 跨世界报告：副标题
    re.compile(r"通过\s*<strong>(\d+)/\d+</strong>"),  # 对照/消融：头条的分母
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

# 离线可复现的报告：命令 → 输出文件名
OFFLINE_REPORTS: dict[str, tuple[tuple[str, ...], str]] = {
    "ablation": (("ablate",), "ablation.html"),
    "worlds": (("worlds",), "worlds.html"),
}

# 一次跑批的快照：需要模型 + 额度，**故意不自动化**。
# 列在这里是为了让"哪些不能重生成"变成一个可被测试读取的事实，
# 而不是散落在文档里的口头说明。
SNAPSHOT_REPORTS: dict[str, str] = {
    "batch_model.html": "compare --models kimi-k2.7-code（228 条跑批 + 裁判）",
    "batch_planner.html": "compare --models kimi-k2.7-code（规划也交给模型）",
    "comparison.html": "compare --models kimi-k2.7-code（离线 vs 模型对照）",
    "multi_npc.html": "compare --models kimi-k2.7-code --category multi_npc",
}


def regen(only: str | None, out_dir: Path) -> list[Path]:
    """生成离线报告，返回写出的文件路径列表。"""
    names = [only] if only else list(OFFLINE_REPORTS)
    unknown = [n for n in names if n not in OFFLINE_REPORTS]
    if unknown:
        raise SystemExit(f"未知的报告名 {unknown}；可选：{list(OFFLINE_REPORTS)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in names:
        args, filename = OFFLINE_REPORTS[name]
        target = out_dir / filename
        cmd = [sys.executable, "-m", "npc_agent.cli", *args, "--html", str(target)]
        print(f"[regen] {name} → {target}", flush=True)
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-2000:], flush=True)
            print(proc.stderr[-2000:], file=sys.stderr, flush=True)
            raise SystemExit(f"生成 {name} 失败（退出码 {proc.returncode}）")
        written.append(target)
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs", help="输出目录（默认 docs/）")
    ap.add_argument("--only", default=None, help="只生成某一个（ablation / worlds）")
    args = ap.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    written = regen(args.only, out_dir)
    print(f"\n完成，写了 {len(written)} 份：")
    for p in written:
        print(f"  {p}")

    print("\n以下报告是**一次跑批的快照**，需要模型 + 额度才能重生成，脚本不碰：")
    for name, how in SNAPSHOT_REPORTS.items():
        print(f"  {name:22s} ← {how}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
