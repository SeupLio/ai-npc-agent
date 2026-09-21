"""`--help` 必须真的能打印出来，而且打印出来的东西要能读。

## 为什么要有这个文件

这个仓库里有两个 bug **同时**活到了推送，而且 pytest 全绿：

| # | 症状 | 谁会发现 |
|---|---|---|
| 1 | `judge --help` 抛 `ValueError: unsupported format character '?' (0x7684)` | 只要有人敲一次 `judge --help` |
| 2 | `judge --help` 抛 `NameError: name 'JUDGE_HISTORY_TURNS' is not defined` | 同上 |

两个都是"**加了一个命令行选项**"的副产物，两个都**只在渲染 help 时才炸**。
整个测试套件里没有任何一条会去渲染 help —— `build_parser()` 和 `parse_args()`
都不会求值 `help=`，所以它们照常通过。

**这就是为什么这个文件存在**：把"敲一下 `--help`"变成一条断言。

## 坑 1 的机制（值得单独记住）

`argparse` 会对 help 串**再做一次 `%` 格式化**（`HelpFormatter._expand_help` 里的
`self._get_help_string(action) % params`）。所以 help 里写一个裸 `%`：

```python
help="实测有约 7% 的调用会超过 60s"     # ← 炸
```

`% ` 不是合法的格式符 ⇒ `ValueError`。要写 `7%%`。
`%(default)s` 是 argparse 的**合法**占位符，不算裸 `%`。

## 坑 2 的机制

`build_parser()` 里引用了一个没导入的常量。它在**函数体**里，所以
`import npc_agent.cli` 成功、`build_parser()` 被调用时才炸 —— 而调用它的人
只有 `main()` 和 `--help`。

## 四条断言

1. 每个子命令的 `--help` 都能渲染，且非空（动态，抓坑 1 和坑 2）
2. 顶层 `--help` 同样能渲染
3. 源码里每个 `help=` 串都不含裸 `%`（静态，抓还没被渲染到的选项）
4. `--judge-history-turns 0` 必须真的是 `0`

第 4 条是另一个坑：`args.judge_history_turns or JUDGE_HISTORY_TURNS` 会把显式的
`0`（"退回旧行为"）悄悄改成 `4`，而报告里会印着 `history_turns: 4`。
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import pytest

from npc_agent.cli import build_parser, judge_history_turns, judge_samples
from npc_agent.eval.judge import JUDGE_HISTORY_TURNS, JUDGE_SAMPLES

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "npc_agent" / "cli.py"


# --------------------------------------------------------------------------- #
# 枚举子命令
# --------------------------------------------------------------------------- #
def _subcommands() -> list[str]:
    """从解析器里**读**子命令，不手抄清单。

    手抄的清单会过期：新加一个子命令，护栏还在检查旧的 15 个，而它全绿。
    读不到子命令就**直接炸** —— 那说明 `build_parser()` 的结构变了，
    这个护栏已经失效，继续绿下去比红更危险。
    """
    parser = build_parser()
    for action in parser._actions:  # noqa: SLF001 - argparse 没给公开的遍历方式
        if isinstance(action, argparse._SubParsersAction):
            names = sorted(action.choices)
            assert names, "子命令清单是空的 —— 枚举方式失效了"
            return names
    raise AssertionError("`build_parser()` 里找不到子命令动作，枚举方式失效了")


SUBCOMMANDS = _subcommands()


def _render_help(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    """跑一次 `--help` 并把它打印的东西取回来。"""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(argv)
    assert exc.value.code == 0, f"`{' '.join(argv)}` 不是正常退出（code={exc.value.code}）"
    return capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 1 / 2 —— 动态：真的渲染一遍
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sub", SUBCOMMANDS)
def test_every_subcommand_can_render_its_help(
    sub: str, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _render_help([sub, "--help"], capsys)
    assert out.strip(), f"`{sub} --help` 什么都没打印"
    assert sub in out, f"`{sub} --help` 的输出里没有子命令名"


def test_top_level_help_renders(capsys: pytest.CaptureFixture[str]) -> None:
    out = _render_help(["--help"], capsys)
    for sub in SUBCOMMANDS:
        assert sub in out, f"顶层 `--help` 里没列出子命令 `{sub}`"


def test_rendered_help_never_shows_a_doubled_percent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`%%` 是给 argparse 的转义写法，**渲染出来的东西里不该再有它**。

    它出现在输出里就说明有人写成了 `%%%%`，或者 argparse 的转义没被消费掉。
    """
    for sub in SUBCOMMANDS:
        out = _render_help([sub, "--help"], capsys)
        assert "%%" not in out, f"`{sub} --help` 的输出里出现了 `%%`"


# --------------------------------------------------------------------------- #
# 3 —— 静态：扫源码，抓还没被渲染到的选项
# --------------------------------------------------------------------------- #
#: argparse 允许的两种 `%` 用法：转义 `%%`，和 `%(name)s` 占位符。
_ALLOWED_PERCENT = re.compile(r"%%|%\([A-Za-z_][A-Za-z0-9_]*\)[sdr]")


def stray_percents(text: str) -> int:
    """数一个 help 串里**不合法**的 `%` 有几个。`0` = 安全。"""
    return _ALLOWED_PERCENT.sub("", text).count("%")


def _string_parts(node: ast.expr) -> str | None:
    """把一个字符串表达式还原成文本；f-string 里的插值当空串。

    插值部分（`{JUDGE_MAX_TOKENS}`）渲染出来是数字，不可能含 `%`，
    所以只取字面量部分是安全的。
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value
            for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
    if isinstance(node, ast.BinOp):  # 显式拼接： "a" + "b"
        left, right = _string_parts(node.left), _string_parts(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def help_strings_in(path: Path) -> list[tuple[int, str]]:
    """`(行号, 文本)` —— 文件里所有 `help=` 关键字实参。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "help":
                continue
            text = _string_parts(kw.value)
            if text is not None:
                out.append((kw.value.lineno, text))
    return out


def test_no_help_string_contains_a_bare_percent() -> None:
    """源码里每个 `help=` 串都不许有裸 `%`。

    比动态那条更宽：动态只覆盖"被渲染到的"选项，这条覆盖全部。
    """
    found = help_strings_in(CLI)
    assert found, "一个 `help=` 都没扫到 —— 扫描方式失效了"
    bad = [
        (line, text)
        for line, text in found
        if stray_percents(text)
    ]
    assert not bad, (
        "这些 help 串里有裸 `%`，会让 `--help` 抛 "
        "`ValueError: unsupported format character`；要写成 `%%`：\n"
        + "\n".join(f"  cli.py:{line}: {text[:80]!r}" for line, text in bad)
    )


# --------------------------------------------------------------------------- #
# 反向测试：证明这两条护栏不是永真式
# --------------------------------------------------------------------------- #
def test_the_percent_rule_can_actually_fail() -> None:
    """先证明"裸 `%` 真的会炸" —— 否则上面那条可能只是碰巧绿。"""
    parser = argparse.ArgumentParser(prog="demo")
    parser.add_argument("--x", default=0, help="实测有约 7% 的调用会超时")
    with pytest.raises(ValueError, match="unsupported format character"):
        parser.format_help()


def test_the_percent_helper_flags_the_exact_string_that_shipped_broken() -> None:
    """被测坏过的那一行原文，必须被判成"不合法"。"""
    shipped = "判分的 prompt 比台词长得多，实测有约 7% 的调用会超过 60s —— "
    assert stray_percents(shipped) == 1
    assert stray_percents(shipped.replace("7%", "7%%")) == 0
    # argparse 的合法占位符不许被误报。
    assert stray_percents("默认 %(default)s") == 0


def test_the_scanner_actually_reads_help_strings() -> None:
    """静态扫描要能证明它**真的读到了**已知的那条 help 串。"""
    texts = [t for _, t in help_strings_in(CLI)]
    assert any("读超时" in t for t in texts), "扫描没读到 `--timeout` 的 help"
    assert any("判一条台词时往前带几轮对话" in t for t in texts), (
        "扫描没读到 `--judge-history-turns` 的 help"
    )


# --------------------------------------------------------------------------- #
# 4 —— `--judge-history-turns 0` 必须真的是 0
# --------------------------------------------------------------------------- #
def test_zero_means_zero_and_is_not_confused_with_unset() -> None:
    """`0`（显式关掉历史）和"没传"必须区分开。"""
    assert JUDGE_HISTORY_TURNS != 0, (
        "默认值本身是 0 ⇒ `0` 和「没传」无法区分，下面几条断言会变成永真式"
    )
    assert judge_history_turns(argparse.Namespace(judge_history_turns=None)) == (
        JUDGE_HISTORY_TURNS
    )
    assert judge_history_turns(argparse.Namespace(judge_history_turns=0)) == 0
    assert judge_history_turns(argparse.Namespace(judge_history_turns=7)) == 7
    # 负数当 0 —— 不能让它一路传到 `pairs[start:index]` 里。
    assert judge_history_turns(argparse.Namespace(judge_history_turns=-3)) == 0


def test_the_cli_default_is_the_sentinel_not_the_value() -> None:
    """参数定义那边必须是 `default=None`。

    写成 `default=JUDGE_HISTORY_TURNS` 也能跑，但 `0` 就再也表达不出
    "我要关掉它"了 —— 翻译函数看到的永远是 4，两种意图挤成一种。
    """
    parser = build_parser()
    assert parser.parse_args(["judge"]).judge_history_turns is None
    assert parser.parse_args(["judge", "--judge-history-turns", "0"]).judge_history_turns == 0


def test_the_flag_reaches_the_judge_constructor() -> None:
    """`cmd_judge` 必须把 `judge_history_turns(args)` 传给 `LLMJudge`。

    这条是静态的：真的跑 `cmd_judge` 要一个可用模型。用源码断言挡住
    "加了参数但忘了接上"——那正是这次两个 bug 的共同形状。
    """
    src = CLI.read_text(encoding="utf-8")
    assert "history_turns=judge_history_turns(args)" in src, (
        "`cmd_judge` 没有把 `--judge-history-turns` 接到 `LLMJudge` 上"
    )


# --------------------------------------------------------------------------- #
# 5 —— `--judge-samples` 的翻译
#
# 同一个 `or` 陷阱，但这里 `0` **更有理由被传**（有人会想"关掉重复采样省钱"），
# 所以它比 `--judge-history-turns` 更危险：静默变成默认票数之后，
# 报告里还写着 `samples: 5`，看起来完全正常。
# --------------------------------------------------------------------------- #
def test_zero_samples_falls_back_to_one_not_to_the_default() -> None:
    """`0` 票 = 必然"未判"的死配置 ⇒ 夹到 **1**，不是夹到默认票数。

    ⚠️ 这里刻意区分两个"安全值"：
      - `None`（没传）→ 默认票数（由 `JUDGE_SAMPLES` 决定）；
      - `0`（显式传）→ **1**（旧行为），因为 0 票什么都不判。
    如果两者都翻成默认票数，那么"我想省掉重复采样的钱"这个意图
    会被悄悄执行成"请多花 (默认票数-1) 倍的钱" —— 方向正好相反。
    """
    assert JUDGE_SAMPLES >= 1
    assert judge_samples(argparse.Namespace(judge_samples=None)) == JUDGE_SAMPLES
    assert judge_samples(argparse.Namespace(judge_samples=0)) == 1
    assert judge_samples(argparse.Namespace(judge_samples=-4)) == 1
    assert judge_samples(argparse.Namespace(judge_samples=5)) == 5


def test_the_samples_default_is_the_sentinel_not_the_value() -> None:
    """参数定义必须是 `default=None`，否则 `0` 表达不出"关掉重复采样"。"""
    parser = build_parser()
    assert parser.parse_args(["judge"]).judge_samples is None
    assert parser.parse_args(["judge", "--judge-samples", "5"]).judge_samples == 5
    assert parser.parse_args(["judge", "--judge-samples", "0"]).judge_samples == 0


def test_the_samples_flag_reaches_the_judge_constructor() -> None:
    """`cmd_judge` 必须把 `judge_samples(args)` 接到 `LLMJudge` 上。

    ⚠️ 断言里带 `=`（`samples=judge_samples(args)`）而不是只搜 `samples=`：
    `history_turns=judge_history_turns(args)` 里也含子串 `samples` 吗？不含。
    但反过来，若只搜 `judge_samples(` 就会漏掉"参数加了、函数写了、
    却没接到构造函数上"这个形状 —— 而它正是前一次两个 bug 的共同长相。
    """
    src = CLI.read_text(encoding="utf-8")
    assert "samples=judge_samples(args)" in src, (
        "`cmd_judge` 没有把 `--judge-samples` 接到 `LLMJudge` 上"
    )
    # 报告里也必须能读到票数 —— 它决定"这条判决是单次读数还是多数票"。
    assert '"judge_samples": judge.samples' in src, (
        "票数没有写进报告 ⇒ 读报告的人不知道这些判决采了几票"
    )
