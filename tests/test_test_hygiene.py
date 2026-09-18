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
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

# 这些环境变量会把"默认跳过"的用例打开（慢速 / 需要外部依赖）。
# 数"默认跳过几条"时必须把它们摘掉，否则：
# 外层带着 `NPC_AGENT_DOC_FRESHNESS=1` 跑整套时，子进程会真的去跑那 2 分钟，
# 于是既慢、又数出 0 条跳过，护栏反而红 —— 一个"因为环境太全"而失败的护栏。
OPT_IN_ENV_VARS = ("NPC_AGENT_DOC_FRESHNESS", "NPC_AGENT_MC_E2E")


def _default_env() -> dict[str, str]:
    """去掉所有 opt-in 开关之后的环境 —— 也就是"默认环境"。"""
    env = dict(os.environ)
    for key in OPT_IN_ENV_VARS:
        env.pop(key, None)
    return env


def _pytest_subprocess_args() -> list[str]:
    """跑子进程 pytest 时统一带的参数。

    ⚠️ **`--basetemp` 不是可有可无的。** 不给它，子进程会用 pytest 的
    全局临时根（`%TEMP%/pytest-of-<user>`）。那个目录会攒下几百个
    `garbage-*` 条目，pytest 在会话结束时清理它们 —— 在这个环境里，
    一次删 208 个条目会撞上沙箱的批量删除保护，**进程被杀掉**。

    症状极具误导性：测试其实跳过了（进度条上是 `s`），但进程死在
    `-rs` 摘要**之前**，于是 stdout 里一条 `SKIPPED [n]` 都没有，
    调用方数出 0 条跳过，报出来的是"README 的跳过数对不上" ——
    **完全指向错误的方向**。给一个自己的 basetemp 就绕开了全局状态。
    """
    return [
        "-q", "-rs", "-p", "no:cacheprovider",
        f"--basetemp={REPO_ROOT / '.pytest_bt_skipcount'}",
    ]
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
)

# 测试那一节印的是 pytest 的真实输出，给的是 **passed**，不是收集数。
# 默认会跳过需要真实服务端的用例，所以 `passed + skipped == 收集数`。
#
# 这里以前直接拿收集数去比 `(\d+) passed`，于是护栏**逼着文档写一句假话**：
# 579 条被收集、只有 578 条真的跑，README 却必须印 "# 579 passed"。
# 一个把标签和数值配错的护栏，比没有护栏更糟 —— 它会给假话盖章。
_PASSED_LINE = re.compile(r"#\s*(\d+) passed(?:,\s*(\d+) skipped)?")

# 默认会被跳过的目标 —— README 里那句 "2 skipped" 说的就是它们。
# 可以写整个文件，也可以写到具体用例（node id）。
#
# 这份名单**故意写死**：它是"默认环境下哪些测试不跑"的唯一真相来源。
# 新增一个默认跳过的用例时，这里要改，README 也要改 —— 这正是想要的。
# 忘了改不会静默通过：`skipped` 对不上就会红，逼你回来看这份名单。
DEFAULT_SKIPPED_TARGETS: tuple[str, ...] = (
    # 整个文件都跳过（需要真实 Minecraft 服务端）
    "tests/test_minecraft_e2e.py",
    # 只有这条跳过（跑一次约 2 分钟，所以默认不跑）
    "tests/test_docs_freshness.py::test_slow_offline_reports_match_the_code",
)

# `-rs` 会给每个跳过组印一行 `SKIPPED [n] 路径:行号: 原因`
_SKIPPED_MARK = re.compile(r"SKIPPED \[(\d+)\]")


def _default_skip_count() -> int:
    """数默认环境下会跳过多少条。

    为什么不能从 `--collect-only` 得到：跳过是在 **fixture 体内**调的
    `pytest.skip()`，收集期看不到（`--setup-plan` 也看不到，因为它不执行
    fixture 体）。所以只能真跑一遍这些目标 —— 而它们本来就是
    "跑起来立刻跳过"，代价是秒级，不是分钟级。

    为什么要支持 node id 而不只是文件：慢速用例**和快用例在同一个文件里**
    （`test_docs_freshness.py` 里只有一条是默认跳过的）。
    按文件粒度算，会把同文件里那些正常跑的用例也当成跳过，于是数出 6 而不是 2。
    """
    total = 0
    for target in DEFAULT_SKIPPED_TARGETS:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", target, *_pytest_subprocess_args()],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=_default_env(),
        )
        marks = _SKIPPED_MARK.findall(proc.stdout)
        # 目标跑挂了却静默算 0 条，会让"跳过数对不上"报成一句看不懂的错。
        # 这里必须把子进程的失败原样带出来 —— 否则查这个错要花掉一整个下午。
        if proc.returncode != 0:
            pytest.fail(
                f"数跳过条数时，目标 {target!r} 自己跑挂了"
                f"（退出码 {proc.returncode}）：\n"
                f"stdout:\n{proc.stdout[-1500:]}\n"
                f"stderr:\n{proc.stderr[-1500:]}"
            )
        if not marks:
            pytest.fail(
                f"目标 {target!r} 一条 SKIPPED 都没有 —— "
                "要么它其实跑了（那就该从 DEFAULT_SKIPPED_TARGETS 里去掉），"
                "要么它根本没被收集到（node id 写错了）：\n"
                f"stdout:\n{proc.stdout[-1500:]}"
            )
        total += sum(int(m) for m in marks)
    return total


# `pytest --collect-only -q` 的每一行形如 `tests/test_x.py: 12`
_COLLECT_LINE = re.compile(r"^tests/(\w+\.py): (\d+)$")


def _collected_test_count() -> int:
    """跑一次 collect-only 数一遍 —— 这是唯一的真相来源。

    用**子进程**而不是在当前进程里调 pytest：在收集期再嵌套触发一次收集，
    pytest 的行为不保证（而且当前进程里已经有一份 session）。
    也带 `--basetemp`，理由见 `_pytest_subprocess_args`（别用全局临时根）。
    """
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", "tests",
            "--collect-only", *_pytest_subprocess_args(),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=_default_env(),
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
    r"""README 里的测试数必须是真的。

    这个仓库里这个数已经过期过五次（546 → 557 → 565 → 568 → 569），
    每次都是"加了测试、忘了改文档"。文档里的数字是**写给读者看的断言**，
    过期了就是假话 —— 和报告里写死"开发集（24 条）"是同一类毛病。

    注意这条测试**自己也计入总数**（它就是新加的那一条）。这是故意的：
    断言的是"README 等于实际收集数"，不是"README 等于实际数减一"。
    代价是每加一条测试都要顺手改 README —— 而这正是想要的。

    ## 这条护栏自己出过一次错，记在这里

    它一开始拿**收集数**去比 README 里的 `(\d+) passed`。加了默认跳过的
    端到端测试之后，579 条被收集、只有 578 条真的跑，于是护栏
    **逼着 README 印一句假话**：`# 579 passed`。

    一个把标签和数值配错的护栏比没有护栏更糟 —— 它会给假话盖章。
    现在的判据拆成两条：`passed + skipped == 收集数`，**而且**
    `skipped` 必须等于默认环境里真正跳过的条数（见 `_default_skip_count`）。
    只校验前者的话，`579 passed, 0 skipped` 照样能混过去。
    """
    actual = _collected_test_count()
    text = README.read_text(encoding="utf-8")

    found: dict[str, list[int]] = {}
    for label, pattern in README_COUNT_PATTERNS:
        matches = pattern.findall(text)
        # 模式过期比数字过期更隐蔽：找不到就静默通过，等于没有这条护栏。
        assert matches, f"README 里找不到「{label}」，正则过期了：{pattern.pattern}"
        found[label] = [int(m) for m in matches]

    wrong: dict[str, object] = {
        k: v for k, v in found.items() if v != [actual] * len(v)
    }

    # 测试一节单独算：它印的是 passed，而 passed + skipped 才等于收集数。
    # 直接拿收集数比 passed，会在有跳过用例时逼文档写错 —— 所以这里比的是关系。
    passed_line = _PASSED_LINE.search(text)
    assert passed_line, (
        f"README 里找不到测试一节的输出行，正则过期了：{_PASSED_LINE.pattern}"
    )
    passed = int(passed_line.group(1))
    skipped = int(passed_line.group(2) or 0)
    if passed + skipped != actual:
        wrong["测试一节的输出"] = (
            f"{passed} passed + {skipped} skipped = {passed + skipped}，"
            f"但收集到 {actual}"
        )
    else:
        # 光校验"加起来对"还不够：`579 passed, 0 skipped` 也能凑出 579，
        # 而那**正是**这条护栏以前逼出来的假话。所以跳过数要单独验一遍。
        real_skips = _default_skip_count()
        if skipped != real_skips:
            wrong["测试一节的跳过数"] = (
                f"README 说跳过 {skipped} 条，默认环境实际跳过 {real_skips} 条"
                f"（{', '.join(DEFAULT_SKIPPED_TARGETS)}）"
            )

    assert not wrong, (
        f"README 里的测试数和实际对不上（实际收集 {actual}）：{wrong}。"
        "改了测试就顺手把 README 里那几处数字一起改掉：目录树、路线图写收集数，"
        "测试一节写 `passed, skipped`，且两者相加必须等于收集数。"
    )
