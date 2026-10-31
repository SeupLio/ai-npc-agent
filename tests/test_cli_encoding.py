"""CLI 的输出必须能在中文 Windows 的 GBK 控制台里打印。

## 为什么这值得一条测试

中文 Windows 的控制台默认编码是 **GBK（cp936）**。`▸`（U+25B8）和
`✗`（U+2717）不在 GBK 里，rich 会直接抛 `UnicodeEncodeError`：

    UnicodeEncodeError: 'gbk' codec can't encode character '\u25b8'

崩的不是某个高级功能 —— 是 `demo` 的第一句动作日志。用户只是想看一眼
NPC 说话，程序直接退出。而项目自带的冒烟命令（`demo --scenario trio`）
在这台机器上就跑不过。

## 两道护栏

1. **静态**：`cli.py` 里所有面向输出的字符串字面量都能被 GBK 编码
   （docstring 不算 —— 它们不会进控制台）。
2. **动态**：真的用 `PYTHONIOENCODING=gbk` 跑一遍 trio 演示。
   静态只覆盖源码里的字面量，动态还能抓到 rich 自己生成的符号。
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "npc_agent" / "cli.py"


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """所有 docstring 的 `id()` —— 它们不参与终端输出。"""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                out.add(id(body[0].value))
    return out


def _non_gbk_literals(source: str) -> list[tuple[int, str]]:
    tree = ast.parse(source)
    docstrings = _docstring_nodes(tree)
    bad: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in docstrings:
            continue
        try:
            node.value.encode("gbk")
        except UnicodeEncodeError:
            bad.append((node.lineno, node.value[:60]))
    return sorted(bad)


def test_every_cli_literal_is_printable_on_a_gbk_console() -> None:
    bad = _non_gbk_literals(CLI.read_text(encoding="utf-8"))
    assert not bad, (
        "这些字符串在 GBK 控制台里打印会抛 UnicodeEncodeError：\n"
        + "\n".join(f"  cli.py:{line}: {text!r}" for line, text in bad)
        + "\n换成 GBK 里存在的符号（例如 → 代替 ▸，× 代替 ✗）。"
    )


def test_the_scanner_can_actually_fail() -> None:
    """反向测试：判据必须真的抓得到那个曾经让 demo 崩掉的字符。"""
    source = 'console.print("▸ action")\n'
    assert _non_gbk_literals(source) == [(1, "▸ action")]
    # docstring 里的符号不参与输出，不该报。
    assert _non_gbk_literals('"""▸ 这是注释"""\n') == []


def test_the_trio_demo_runs_under_gbk() -> None:
    """真跑一次：rich 自己生成的制表符与中文都必须编得出来。"""
    env = {**os.environ, "PYTHONIOENCODING": "gbk"}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "npc_agent.cli",
            "demo",
            "--scenario",
            "trio",
            "--ticks",
            "4",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        "GBK 输出下 demo 崩了：\n"
        f"stdout tail:\n{proc.stdout[-1500:]!r}\n"
        f"stderr tail:\n{proc.stderr[-1500:]!r}"
    )
    assert proc.stdout, "demo 什么都没打印"


def test_resolution_terminal_text_is_printable_on_gbk() -> None:
    """`regen_docs --only resolution` 的必经路径不能在 GBK 下崩。

    这条是真事故：报告的终端文本里曾经有 `⚠️`，在中文 Windows 上
    `scripts/regen_docs.py` 一跑就 `UnicodeEncodeError`，
    而测试只看到"报告和代码对不上"。
    """
    from npc_agent.eval.resolution import DIMENSIONS, render_resolution

    class _Capture:
        text = ""

        def print(self, text: str) -> None:
            self.text = text

    result = {
        "total_cases": 2,
        "baseline_pass_rate": 1.0,
        "points": [
            {
                "k": 0.0,
                "passed": 2,
                "total": 2,
                "pass_rate": 1.0,
                "metric_means": {d: 1.0 for d in DIMENSIONS},
            }
        ],
        "detection_threshold_k": None,
        "blind_dimensions": list(DIMENSIONS),
    }
    cap = _Capture()
    render_resolution(result, cap)
    cap.text.encode("gbk")
