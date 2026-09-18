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

## 清单在哪

报告清单（谁是谁、哪些能重生成）住在 **`npc_agent/eval/report_index.py`**。
它原来只活在这个脚本里，于是"报告门户"和"过期护栏"各自抄了一份 ——
手抄的清单必然漂移。这里只是**转出**那几个名字，保持
`regen_docs.OFFLINE_REPORTS` 这些老入口不变（测试在用）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# `scripts/` 不是包，直接 `python scripts/regen_docs.py` 时 sys.path[0] 是
# `scripts/` 而不是仓库根 —— 不插这一行，下面的 `npc_agent` 导入会失败。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 转出（re-export）：清单的唯一真相来源在库里，不在脚本里。
from npc_agent.eval.report_index import (  # noqa: E402
    DURATION_PLACEHOLDER,
    OFFLINE_REPORTS,
    SNAPSHOT_REPORTS,
    case_count,
    normalize_report,
)

__all__ = [
    "OFFLINE_REPORTS",
    "SNAPSHOT_REPORTS",
    "DURATION_PLACEHOLDER",
    "case_count",
    "normalize_report",
    "regen",
    "main",
]


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
    ap.add_argument(
        "--only",
        default=None,
        # 列表从 OFFLINE_REPORTS 生成 —— 手写列表在加 sensitivity 那次就漏了，
        # 于是 --help 上少了一个可选项。**能推导的别手抄。**
        help=f"只生成某一个（{' / '.join(OFFLINE_REPORTS)}）",
    )
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
