"""测试文件自身的卫生检查。

## 为什么需要这个文件

Python 里**同名函数后者覆盖前者**，pytest 只收集最后那个。
所以一个被重复定义两次的测试，看起来有两份，实际只有一份会跑 ——
而**改第一份是完全没有效果的**。这是最坏的一类静默失效：
你以为在改测试，其实改的是一段死代码，而测试照样"通过"。

实测这个仓库里曾经有 **10 处**重复定义（`test_batch_report.py` 7 处、
`test_generator.py` / `test_holdout.py` / `test_runner.py` 各 1 处），
都是同一段内容被粘了两遍、两份**逐字节相同** —— 所以没有丢断言，
但"改错那一份"的陷阱是真实存在的。

修完之后加了这个检查，让它不能再长回来。
"""

from __future__ import annotations

import ast
import collections
import re
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
README = REPO_ROOT / "README.md"


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _top_level_functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _test_files() -> list[Path]:
    return sorted(TESTS_DIR.glob("test_*.py"))


def test_no_test_file_defines_the_same_function_twice() -> None:
    """同名函数只能有一个定义 —— 否则只有最后一个会跑。

    注意这里**故意不检查"两份内容是否相同"**：内容相同也一样是陷阱，
    因为下一次有人改的正是被覆盖的那一份。
    """
    offenders: dict[str, dict[str, int]] = {}
    for path in _test_files():
        names = [node.name for node in _top_level_functions(_module(path))]
        dupes = {n: c for n, c in collections.Counter(names).items() if c > 1}
        if dupes:
            offenders[path.name] = dupes

    assert not offenders, (
        "这些测试文件里有重复定义，只有最后一个会跑（改前面那些是白改）："
        f"{offenders}"
    )


def test_nothing_shadows_a_test_function() -> None:
    """重复定义只是**已知的一种**遮蔽方式。

    这条防的是另一种：模块顶层后面又用赋值/类/导入把 `test_xxx` 这个名字
    重新绑到别的东西上 —— 那样函数还在源码里，但已经收不到了。
    """
    offenders: dict[str, list[str]] = {}
    for path in _test_files():
        tree = _module(path)
        func_names = {node.name for node in _top_level_functions(tree)}
        test_names = {n for n in func_names if n.startswith("test_")}

        shadowed: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(node, ast.ClassDef) and node.name in test_names:
                shadowed.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id in test_names:
                        shadowed.add(target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    if bound in test_names:
                        shadowed.add(bound)
        if shadowed:
            offenders[path.name] = sorted(shadowed)

    assert not offenders, f"这些 test_ 函数被后面的语句遮住了，收不到：{offenders}"


def test_the_hygiene_check_itself_can_fail(tmp_path: Path) -> None:
    """**护栏必须能被验证会红。**

    上面两条检查如果永远返回"没问题"，那它和不存在没区别。
    这里造一份**故意有重复定义**的假测试文件，确认检查逻辑真的会抓到它。
    """
    fake = tmp_path / "test_fake.py"
    fake.write_text(
        "def test_same() -> None:\n"
        "    assert True\n"
        "\n"
        "\n"
        "def test_same() -> None:\n"
        "    assert False\n",
        encoding="utf-8",
    )
    names = [node.name for node in _top_level_functions(_module(fake))]
    dupes = {n: c for n, c in collections.Counter(names).items() if c > 1}
    assert dupes == {"test_same": 2}, "重复检测逻辑失效了"


# --------------------------------------------------------------------------- #
# README 里的测试数
# --------------------------------------------------------------------------- #

# README 承诺了测试数的地方。故意**逐个写死模式**，而不是"把文档里所有数字
# 抓出来对一遍" —— 后者会把版本号、用例数、kappa 全卷进来，测试立刻变噪音。
README_COUNT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("目录树里的测试数", re.compile(r"tests/\s+(\d+) 个单元与端到端测试")),
    ("路线图里的测试数", re.compile(r"六维评测 harness \+ (\d+) 个测试")),
    ("测试一节的输出", re.compile(r"#\s*(\d+) passed")),
)

# `pytest --collect-only -q` 的每一行形如 `tests/test_x.py: 12`
_COLLECT_LINE = re.compile(r"^tests/(\w+\.py): (\d+)$")


def _collected_test_count() -> int:
    """跑一次 collect-only 数一遍 —— 这是唯一的真相来源。

    用**子进程**而不是在当前进程里调 pytest：在收集期再嵌套触发一次收集，
    pytest 的行为不保证（而且当前进程里已经有一份 session）。
    """
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", "tests",
            "-p", "no:cacheprovider", "--collect-only", "-q",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.fail(
            "收集测试失败，没法核对 README 里的数字：\n"
            f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
        )

    total = 0
    for line in proc.stdout.splitlines():
        match = _COLLECT_LINE.match(line.strip())
        if match:
            total += int(match.group(2))

    # 解析出 0 是最危险的情况：所有断言都会"通过"，因为没有一个数对得上。
    if total == 0:
        pytest.fail(
            "没能从 collect-only 的输出里解析出任何用例数（输出格式变了？）：\n"
            f"{proc.stdout[-2000:]}"
        )
    return total


def test_readme_test_counts_match_reality() -> None:
    """README 里的测试数必须是真的。

    这个仓库里这个数已经过期过五次（546 → 557 → 565 → 568 → 569），
    每次都是"加了测试、忘了改文档"。文档里的数字是**写给读者看的断言**，
    过期了就是假话 —— 和报告里写死"开发集（24 条）"是同一类毛病。

    注意这条测试**自己也计入总数**（它就是新加的那一条）。这是故意的：
    断言的是"README 等于实际收集数"，不是"README 等于实际数减一"。
    代价是每加一条测试都要顺手改 README —— 而这正是想要的。
    """
    actual = _collected_test_count()
    text = README.read_text(encoding="utf-8")

    found: dict[str, list[int]] = {}
    for label, pattern in README_COUNT_PATTERNS:
        matches = pattern.findall(text)
        # 模式过期比数字过期更隐蔽：找不到就静默通过，等于没有这条护栏。
        assert matches, f"README 里找不到「{label}」，正则过期了：{pattern.pattern}"
        found[label] = [int(m) for m in matches]

    wrong = {k: v for k, v in found.items() if v != [actual] * len(v)}
    assert not wrong, (
        f"README 里的测试数和实际收集到的不一致（实际 {actual}）：{wrong}。"
        "改了测试就顺手把 README 里那三处数字一起改掉。"
    )
