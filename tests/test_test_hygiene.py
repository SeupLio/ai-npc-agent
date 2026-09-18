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
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

# 六个维度从 `_METRIC_LABELS` 取，**不手抄** ——
# 将来加第七个维度时，README 那段输出和这条护栏会一起长出来。
from npc_agent.eval.report import _METRIC_LABELS

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


# --------------------------------------------------------------------------- #
# README 里那条命令：它印出来的东西必须和文档写的一样
# --------------------------------------------------------------------------- #

# README 测试一节里的命令行。故意只认 `python -m pytest tests` 这一种写法 ——
# 换成别的写法（`pytest -q tests` / `python -m pytest ./tests`）这条护栏会失效，
# 所以**找不到就红**，而不是静默跳过。
_README_TEST_CMD = re.compile(r"^python -m pytest tests(?P<flags>[^\n]*)$", re.M)

# pytest 的汇总行：`42 passed in 0.11s` / `591 passed, 2 skipped in 234.56s`
_SUMMARY_LINE = re.compile(r"^\d+ passed(?:, \d+ \w+)* in \d", re.M)

# 拿它当靶子跑一遍：够小（几十条、0.1 秒），但足以让 pytest 走到"印汇总"那一步。
_SUMMARY_PROBE_TARGET = "tests/test_conditions.py"


def _documented_pytest_flags() -> list[str]:
    """把 README 里那条命令的**参数**抠出来（`tests` 之后的部分）。"""
    match = _README_TEST_CMD.search(README.read_text(encoding="utf-8"))
    assert match, (
        "README 的测试一节里找不到 `python -m pytest tests ...` 这行命令，"
        f"正则过期了：{_README_TEST_CMD.pattern}"
    )
    return match.group("flags").split()


def test_the_documented_test_command_prints_the_summary_it_promises() -> None:
    """README 里那条命令，必须真的印出 README 承诺的那行汇总。

    ## 为什么"数字对"还不够

    上面那条护栏只管**数字**。但文档可以是另一种假话：
    **数字是对的，而照着文档敲出来的命令根本印不出这行数字。**

    这就是真发生过的事：README 写的是

    ```bash
    python -m pytest tests -q
    # 591 passed, 2 skipped
    ```

    而 `pyproject.toml` 里已经有 `addopts = "-q"` —— 命令行那个 `-q` 叠上去
    变成 **`-qq`**，而 `-qq` 会把汇总行**整个吞掉**。照着文档敲的人只看到
    进度点和 `[100%]`，然后什么都没有，**退出码还是 0**。
    看起来像"套件崩了"，其实全绿 —— 一个把读者往错误方向引的文档。

    ## 做法

    把 README 里的参数**原样抠出来**，套在一个小靶子上跑一遍。
    靶子只要能让 pytest 走到"印汇总"那一步就够了，所以是秒级，
    不是把整套跑一遍。
    """
    probe = REPO_ROOT / _SUMMARY_PROBE_TARGET
    assert probe.exists(), (
        f"这条护栏的靶子 {_SUMMARY_PROBE_TARGET} 不存在了 —— "
        "换一个跑得快的测试文件，别把这条护栏一起删掉"
    )

    flags = _documented_pytest_flags()
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", _SUMMARY_PROBE_TARGET,
            *flags, "-p", "no:cacheprovider",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=_default_env(),
    )

    if not _SUMMARY_LINE.search(proc.stdout):
        pytest.fail(
            "照着 README 敲 `python -m pytest tests "
            f"{' '.join(flags)}`，pytest 没有印出汇总行。\n"
            "最常见的原因：README 里的参数和 `pyproject.toml` 的 `addopts` "
            "叠成了 `-qq`，而 `-qq` 会吞掉汇总行。\n"
            f"靶子 {_SUMMARY_PROBE_TARGET} 的实测输出（退出码 {proc.returncode}）：\n"
            f"{proc.stdout[-1500:]}"
        )


# --------------------------------------------------------------------------- #
# README 里**所有**被文档化的 pytest 命令
# --------------------------------------------------------------------------- #

# 允许三种前缀：Markdown 引用（`> `）、环境变量赋值（`FOO=bar `）、缩进。
# 故意只认 `python -m pytest`：换成裸 `pytest` 这条护栏会静默失效，
# 所以下面先断言"至少找到一条"，一条都找不到就红。
_DOCUMENTED_PYTEST = re.compile(
    r"(?m)^[>\s]*(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*python -m pytest\b[^\n]*"
)

_QUIET_FLAGS = {"-q", "--quiet"}


def _documented_pytest_commands() -> list[str]:
    return [
        match.group(0).strip()
        for match in _DOCUMENTED_PYTEST.finditer(README.read_text(encoding="utf-8"))
    ]


def test_the_documented_command_pattern_catches_the_hard_shapes() -> None:
    r"""正则必须抓得到 README 里**真实出现过**的几种写法。

    抓不到就等于这条护栏在空转 —— 而空转的护栏比没有更糟，因为它会
    给"我已经检查过了"盖章。真实出现过两种：

    - 普通代码块：`python -m pytest tests`
    - 引用块 + 环境变量前缀：
      `> NPC_AGENT_MC_E2E=1 python -m pytest tests/test_minecraft_e2e.py`

    第二种如果漏了，那条命令里的 `-q` 就永远查不出来。
    """
    samples: dict[str, int] = {
        "python -m pytest tests": 1,
        "> NPC_AGENT_MC_E2E=1 python -m pytest tests/test_minecraft_e2e.py": 1,
        "  python -m pytest tests -k foo": 1,
        "python -m pytest": 1,
        "pytest tests": 0,  # 没有 `python -m`，故意不认（所以下面才要断言非空）
    }
    for line, expected in samples.items():
        found = len(_DOCUMENTED_PYTEST.findall(line))
        assert found == expected, f"{line!r} 期望 {expected} 条，实际 {found} 条"


def test_no_documented_pytest_command_brings_its_own_quiet_flag() -> None:
    r"""README 里**任何一条** pytest 命令都不能自己带 `-q`。

    `pyproject.toml` 里已经有 `addopts = "-q"`。命令行再带一个，两个叠成
    **`-qq`** —— 而 `-qq` 会把最后那行汇总**整个吞掉**：只剩进度点和
    `[100%]`，退出码还是 0。照着文档敲的人会以为套件崩了。

    上面那条护栏只盯"主命令"，这条盯**全部**命令 —— 因为同一个错
    在这个 README 里出现过**两处**（主命令 + Minecraft 那条），
    只修一处、只测一处，另一处照样是假话。
    """
    commands = _documented_pytest_commands()
    assert commands, (
        "README 里一条 `python -m pytest` 命令都找不到 —— 正则过期了，"
        f"这条护栏正在空转：{_DOCUMENTED_PYTEST.pattern}"
    )

    offenders = [cmd for cmd in commands if _QUIET_FLAGS & set(cmd.split())]
    assert not offenders, (
        "这些文档化的命令自带 `-q`，会和 `pyproject.toml` 里的 "
        f'`addopts = "-q"` 叠成 `-qq`，把汇总行吞掉：{offenders}。'
        "`addopts` 已经给了 `-q`，命令里不必再写。"
    )


def test_the_quiet_flag_check_can_actually_fail() -> None:
    """上面那条的判据必须真的抓得到 `-q`，否则它和不存在没区别。"""
    sample = [
        "python -m pytest tests",
        "NPC_AGENT_MC_E2E=1 python -m pytest tests/test_minecraft_e2e.py -q",
        "python -m pytest tests --quiet",
    ]
    offenders = [cmd for cmd in sample if _QUIET_FLAGS & set(cmd.split())]
    assert len(offenders) == 2, f"漏检或误报：{offenders}"


# --------------------------------------------------------------------------- #
# README 里手抄的离线基线
# --------------------------------------------------------------------------- #

# README「评测」一节里那段基线输出。它和 `docs/*.html` 里"覆盖"那一列是
# 同一个毛病：**数字不是从代码里长出来的，是人手打上去的**。
#
# 而它比那几份报告更该被钉住 —— 它是**离线可复现**的：
# 任何人跑一遍 `python -m npc_agent.cli eval` 就能拿到真值。
# 一份离线可复现的文档却停在几个月前，就是在说谎，而且没人会发现。
#
# ⚠️ 只钉**确定性**的字段（通过率、六维均值、每类条数）。
# `墙钟 2s` / `实测加速 3.97×` **故意不钉** —— 它们随机器和负载变，
# 钉住只会逼着文档写一个假的固定值。一条把噪声也钉死的护栏，
# 最后一定会被 `--force` 掉。
_BASELINE_BLOCK = re.compile(
    r"通过率\s*(?P<passed>\d+)/(?P<total>\d+)\s*（(?P<pct>\d+)%）\s*各维度均值\s*"
    r"task=(?P<task>[\d.]+)\s*tools=(?P<tools>[\d.]+)\s*memory=(?P<memory>[\d.]+)\s*"
    r"persona=(?P<persona>[\d.]+)\s*safety=(?P<safety>[\d.]+)\s*"
    r"turn_taking=(?P<turn_taking>[\d.]+)"
)

# `分布：`task 76 / persona 39 / memory 34 / safety 31 / multi_npc 27 / minecraft 21`。`
_BASELINE_DIST = re.compile(
    r"分布：`task (?P<task>\d+) / persona (?P<persona>\d+) / memory (?P<memory>\d+) / "
    r"safety (?P<safety>\d+) / multi_npc (?P<multi_npc>\d+) / minecraft (?P<minecraft>\d+)`"
)

# 六个维度从 `_METRIC_LABELS` 取（import 在文件顶部），**不手抄**。
_BASELINE_DIMS: tuple[str, ...] = tuple(_METRIC_LABELS)


def _baseline_mismatches(text: str, summary: dict) -> dict[str, object]:
    """README 里那段基线输出和**实际跑出来**的 summary 对不上的地方。

    拆成纯函数是为了能给它写反向测试（见 `..._can_actually_fail`）：
    一个只会返回空字典的护栏，和不存在没区别。
    """
    wrong: dict[str, object] = {}

    block = _BASELINE_BLOCK.search(text)
    if not block:
        return {"基线输出段": "README 里找不到，正则过期了"}

    dist = _BASELINE_DIST.search(text)
    if not dist:
        wrong["分布行"] = "README 里找不到，正则过期了"

    total = int(summary["total"])
    passed = int(summary["passed"])

    if (int(block["passed"]), int(block["total"])) != (passed, total):
        wrong["通过率"] = (
            f"README 写 {block['passed']}/{block['total']}，"
            f"实际 {passed}/{total}"
        )
    expected_pct = round(passed / total * 100) if total else 0
    if int(block["pct"]) != expected_pct:
        wrong["百分比"] = f"README 写 {block['pct']}%，实际 {expected_pct}%"

    means = summary["metric_means"]
    for dim in _BASELINE_DIMS:
        shown = float(block[dim])
        real = float(means.get(dim, 0.0))
        # README 印的是三位小数（`1.000`），所以比到 1e-9 就够。
        if abs(shown - real) > 1e-9:
            wrong[f"维度 {dim}"] = f"README 写 {shown:.3f}，实际 {real:.3f}"

    if dist:
        by_cat = {k: int(v["total"]) for k, v in summary["by_category"].items()}
        for cat in ("task", "persona", "memory", "safety", "multi_npc", "minecraft"):
            shown = int(dist[cat])
            real = by_cat.get(cat)
            if real is None:
                wrong[f"分布 {cat}"] = "这次跑批里根本没有这一类用例（用例集被改了？）"
            elif shown != real:
                wrong[f"分布 {cat}"] = f"README 写 {shown} 条，实际 {real} 条"

    return wrong


def _offline_eval_summary(tmp_path: Path) -> dict:
    """跑一遍 README 里那条命令，拿回真实的 summary。

    用**子进程**而不是在当前进程里调 harness：README 承诺的是
    "敲这行命令你会看到这些数字"，那验证的就该是**那行命令**，
    不是一段碰巧等价的库调用。
    """
    out = tmp_path / "eval_offline.json"
    proc = subprocess.run(
        [sys.executable, "-m", "npc_agent.cli", "eval", "--json", str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=_default_env(),
    )
    if proc.returncode != 0:
        pytest.fail(
            f"跑 README 里的离线基线命令失败（退出码 {proc.returncode}）：\n"
            f"stdout:\n{proc.stdout[-1500:]}\nstderr:\n{proc.stderr[-1500:]}"
        )
    if not out.exists():
        pytest.fail(f"命令退出码 0，但没写出 JSON：{out}")
    return json.loads(out.read_text(encoding="utf-8"))["summary"]


def test_readme_offline_baseline_matches_reality(tmp_path: Path) -> None:
    r"""README 里那段手抄的基线输出，必须和真跑一遍的结果逐字一致。

    ## 为什么值得单独加一条

    这段输出是**离线可复现**的（不调模型、不联网、秒级），
    所以它和 `docs/` 里那几份离线报告是一个性质：
    **和代码不一致就是在说谎** —— 读者会以为这是当前代码的成绩。

    它已经过期过一次：用例集从 12 条长到 228 条，而这类"手抄块"
    没有任何机制会跟着动。加这条护栏之后，任何让六维均值或分布
    发生变化的改动（比如某条断言被写松了、用例集被重新生成），
    都会先把 README 顶红 —— 而不是静默留下一句假话。
    """
    text = README.read_text(encoding="utf-8")
    summary = _offline_eval_summary(tmp_path)
    wrong = _baseline_mismatches(text, summary)

    assert not wrong, (
        f"README 里的离线基线和实际对不上：{wrong}。"
        "跑 `python -m npc_agent.cli eval --json reports/eval_offline.json` "
        "拿到真值，把「评测」一节那段输出和 `分布：` 那一行一起改掉。"
    )


def test_the_readme_baseline_check_can_actually_fail(tmp_path: Path) -> None:
    """上面那条的判据必须真的抓得到过期数字，否则它和不存在没区别。

    做法：拿真跑出来的 summary，配一份**被人改坏**的 README 文本，
    断言每类改动都被点名。只测"改一个数"不够 —— 那只能证明某一条分支活着。
    """
    text = README.read_text(encoding="utf-8")
    summary = _offline_eval_summary(tmp_path)
    assert not _baseline_mismatches(text, summary), "原文本本来就对不上，先修 README"

    broken = text.replace("通过率 228/228（100%）", "通过率 227/228（100%）")
    broken = broken.replace("task=1.000 tools=1.000", "task=0.900 tools=1.000")
    broken = broken.replace("分布：`task 76", "分布：`task 99")
    wrong = _baseline_mismatches(broken, summary)

    assert "通过率" in wrong, f"改了通过率却没抓到：{wrong}"
    assert "维度 task" in wrong, f"改了六维均值却没抓到：{wrong}"
    assert "分布 task" in wrong, f"改了分布却没抓到：{wrong}"

    # 正则过期是最隐蔽的失效方式：抓不到就静默返回空字典。
    assert _baseline_mismatches("这里没有基线输出", summary), (
        "文本里没有基线块时必须报错，不能静默通过"
    )



