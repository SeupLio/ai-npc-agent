"""死代码卫生：**模块级 `def` / `class` 不许没人调用。**

`test_import_hygiene` 的姊妹文件。那一份管"导进来了但没人用"，
这一份管"定义出来了但没人调"。

## 为什么需要这个文件

今天用 AST 扫了一遍全仓库，抓到 2 处，**两处都是"两条实现里死掉的那一条"**：

| # | 残留 | 为什么留下来了 |
|---|---|---|
| 1 | `eval/batch_report.py` `write_batch_html` | 活的那条路径在 `cli.py` 里**内联**写了同一件事（含 `mkdir`）。这条是**第二条写出路径** |
| 2 | `eval/runner.py` `metered_factory` | 活的那条是 `runner.py` 里一个内联的 `factory()` 闭包，做了同一件事 |

第 1 条最值得记：它不是"没用的代码"，它是**同一件事的第二份实现**。
它现在无害（没人调），但**将来有人给活的那条加东西**（编码 / 行尾 / 原子写 / 建目录），
死的那条不会跟着走 —— 而它看起来仍然是一个可以调的公开函数。
本项目已经把"两个真相"列为最强的一类缺陷（见 `MEMORY.md` C 组），
这一条属于同一族，只是还没发作。

**为什么值得专门配护栏**：它们不报错、不掉分，而且**删之前得先证明它真的没人用** ——
人不会去证，于是它会一直留着。

## 判据（刻意保守）

只报**高置信度**的，因为**会误报的护栏比没有护栏更糟**（它迟早被 `--deselect` 掉，
然后连真的问题也一起放过）：

* 只看 `tree.body`（**模块级**）。方法一律不看 —— 方法经由 `self.x()` 调用，
  名字里根本不出现 `x`，按"没人用"报它是纯误报。
* 跳过 dunder（`__init__` 之类由语言调用）与 `pytest_*`（由 pytest 按约定调用）。
* 跳过任何出现在 `__all__` 里的名字 —— 那是"再导出"的正当理由。
* 跳过测试文件与 `conftest.py`（它们的函数由 pytest 按名字收集）。
* **引用语料故意放宽**：不只 `.py`，还包括 README / `docs/` / `configs/` / studio 前端
  —— 前端按名字取后端接口，只扫 `.py` 会把它们误报成死的。
* 判据是"整个仓库里这个词只出现 1 次"，也就是**只有它自己的定义那一次**。
"""

from __future__ import annotations

import ast
import re
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "npc_agent"

_EXTRA_TEXT = ("README.md", "pyproject.toml", "requirements.txt")
_TEXT_GLOBS = (
    "docs/**/*.md",
    "docs/**/*.html",
    "configs/**/*.yaml",
    "configs/**/*.yml",
    "npc_agent/studio/**/*.js",
    "npc_agent/studio/**/*.html",
)


def _package_files() -> list[Path]:
    return sorted(p for p in PACKAGE_DIR.rglob("*.py") if "__pycache__" not in str(p))


def _corpus_files() -> list[Path]:
    """引用语料：**故意放宽**。漏报只是少抓一个；误报会让护栏被关掉。"""
    files: list[Path] = []
    for d in (PACKAGE_DIR, REPO_ROOT / "tests", REPO_ROOT / "scripts"):
        if d.exists():
            files.extend(sorted(d.rglob("*.py")))
    for name in _EXTRA_TEXT:
        p = REPO_ROOT / name
        if p.exists():
            files.append(p)
    for pattern in _TEXT_GLOBS:
        files.extend(sorted(REPO_ROOT.glob(pattern)))
    return [p for p in files if "__pycache__" not in str(p)]


def _corpus_text() -> str:
    return "\n".join(
        p.read_text(encoding="utf-8", errors="replace") for p in _corpus_files()
    )


def _exported_names() -> set[str]:
    """包内所有 `__all__` 列出的字符串 —— 这些是**正当的再导出**。"""
    names: set[str] = set()
    for path in _package_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
            ):
                for el in getattr(node.value, "elts", []):
                    if isinstance(el, ast.Constant) and isinstance(el.value, str):
                        names.add(el.value)
    return names


def _dead_module_defs(source: str, corpus: str, exported: set[str]) -> list[str]:
    """返回形如 `L12: foo` 的列表。语法错时抛 `SyntaxError`（由调用方处理）。"""
    problems: list[str] = []
    for node in ast.parse(source).body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        name = node.name
        if name.startswith("__") or name.startswith("pytest_"):
            continue
        if name in exported:
            continue
        if len(re.findall(rf"\b{re.escape(name)}\b", corpus)) <= 1:
            problems.append(f"L{node.lineno}: {name}")
    return problems


# --------------------------------------------------------------------------- #
# 主护栏
# --------------------------------------------------------------------------- #
def test_the_package_has_no_dead_module_level_defs() -> None:
    """`npc_agent/` 里每个模块级 `def` / `class` 都必须有人调用。"""
    corpus = _corpus_text()
    exported = _exported_names()
    assert exported, "一个 __all__ 都没读到 —— 扫描范围写错了"

    offenders: dict[str, list[str]] = {}
    for path in _package_files():
        if path.name == "conftest.py":
            continue
        problems = _dead_module_defs(
            path.read_text(encoding="utf-8"), corpus, exported
        )
        if problems:
            offenders[str(path.relative_to(REPO_ROOT))] = problems

    assert not offenders, (
        "有定义出来却没人调用的模块级名字：\n"
        + "\n".join(
            f"  {f}\n" + "\n".join(f"      {p}" for p in ps)
            for f, ps in offenders.items()
        )
        + "\n  删掉它们。**如果那是有意的公开 API，把它加进本模块的 `__all__`**"
        "（那才是「我确实对外提供它」的写法）；"
        "\n  ⚠️ 特别注意「同一件事的第二份实现」—— 先 grep 活的那条路径，"
        "确认没人调之后再删。"
    )


# --------------------------------------------------------------------------- #
# 反向测试：这条检查必须能红
# --------------------------------------------------------------------------- #
def test_the_dead_code_check_can_actually_fail() -> None:
    """喂一个**真的**有死函数的模块，必须被抓出来。

    没有这一条，`_dead_module_defs` 哪天因为 AST 遍历写错而恒返回 `[]`，
    上面那条会**永远是绿的** —— 而它会以"没有死代码"的姿态骗过所有人。
    """
    source = (
        "def used():\n"
        "    return 1\n"
        "\n"
        "def nobody_calls_me():\n"
        "    return 2\n"
        "\n"
        "print(used())\n"
    )
    corpus = source  # 语料就是它自己：used 出现 2 次，nobody_calls_me 只出现 1 次
    problems = _dead_module_defs(source, corpus, set())
    assert problems == ["L4: nobody_calls_me"], (
        f"检查没能识别出死函数（抓到的是 {problems}）—— 这条护栏是摆设。\n"
        "  期望只抓到 ['L4: nobody_calls_me']：`used` 被 `print` 调了，另一个一次都没被提。"
    )


def test_the_check_does_not_flag_an_exported_name() -> None:
    """`__all__` 里列出来的名字是**正当的再导出**，不许报。"""
    source = "def public_api():\n    return 1\n"
    assert _dead_module_defs(source, source, {"public_api"}) == []
    # 反向：不把它放进 __all__ 就必须报 —— 证明上面那条不是因为判据失效才绿的
    assert _dead_module_defs(source, source, set()) == ["L1: public_api"]


def test_the_check_ignores_methods_dunders_and_pytest_hooks() -> None:
    """方法 / dunder / `pytest_*` 钩子都**不该**被报。

    方法经由 `self.x()` 调用，名字里根本不出现 `x` —— 按"没人用"报它是纯误报。
    这三类误报只要出现一次，这条护栏就会被 deselect 掉，那就比没有更糟。

    ⚠️ 语料里必须有 `Thing()`：**类本身**是模块级的，没人用它就该被报
    （第一版忘了加，于是这条测试拿到 `['L1: Thing']` —— 那是**正确**行为，
    错的是我的前提）。这里要验的是"它的**方法**没被报"。
    """
    source = (
        "class Thing:\n"
        "    def a_method_nobody_names(self):\n"
        "        return 1\n"
        "\n"
        "    def __init__(self):\n"
        "        pass\n"
        "\n"
        "def pytest_configure(config):\n"
        "    pass\n"
    )
    corpus = source + "\nThing()\n"
    assert _dead_module_defs(source, corpus, set()) == []


def test_the_check_reads_names_from_a_wider_corpus_than_the_file() -> None:
    """只在**别的文件**里被调用的名字不算死。

    这是本检查最容易误报的地方：studio 的前端按名字取后端接口，
    只扫 `.py` 会把它报成死的。
    """
    source = "def called_from_the_frontend():\n    return 1\n"
    corpus = source + '\nfetch("/api/called_from_the_frontend")\n'
    assert _dead_module_defs(source, corpus, set()) == []


def test_the_guard_would_catch_a_regression_in_the_real_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """往**真实包的副本**里注入一个死函数，主护栏的扫描逻辑必须抓到它。

    前几条测的是纯函数；这一条测的是"主护栏真的在扫 `npc_agent/`"。
    两者缺一不可 —— 纯函数对了但扫描范围写错（`glob` 拼错之类）照样是死护栏。

    ⚠️ 用**副本**而不是原地改真文件：原地改的话，测试中途崩掉会把仓库
    留在一个被注入过的状态。
    """
    # ⚠️ 名字**拼出来**，不要写字面量。引用语料**包含本文件自己**，
    # 写字面量的话本文件里那几处出现会把它"喂饱"（计数 >1），
    # 于是注入的函数反而不会被报 —— 这条测试会**静默变成永远绿**。
    # （第一版就是这么翻的：`offenders == {}`，看起来像"护栏坏了"，其实是语料被自己污染。）
    injected = "injected_dead_" + "def"

    shadow = tmp_path / "npc_agent"
    shutil.copytree(PACKAGE_DIR, shadow, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(sys.modules[__name__], "PACKAGE_DIR", shadow)

    target = shadow / "modules" / "repetition.py"
    original = target.read_text(encoding="utf-8")
    assert f"def {injected}(" not in original, "锚点已经存在，换个名字"
    target.write_text(
        original.replace(
            "from __future__ import annotations",
            f"from __future__ import annotations\n\n\ndef {injected}():\n    return 1",
            1,
        ),
        encoding="utf-8",
    )
    assert f"def {injected}(" in target.read_text(encoding="utf-8"), "注入没落盘"

    # 关键：走一遍**主护栏的扫描逻辑**（`_package_files()` 读的是被改过的根）。
    corpus = _corpus_text()
    # **前提断言**：语料里这个名字必须只出现一次（就是注入的那一行）。
    # 不写这一条，上面那个"被自己污染"的坑就会静默复发。
    assert corpus.count(injected) == 1, (
        f"语料里 {injected} 出现 {corpus.count(injected)} 次，不是 1 次 —— "
        "有别的文件把它写进了语料，这条测试会变成永远绿"
    )
    exported = _exported_names()
    scanned = {
        str(p.relative_to(shadow)): _dead_module_defs(
            p.read_text(encoding="utf-8"), corpus, exported
        )
        for p in _package_files()
    }
    offenders = {f: ps for f, ps in scanned.items() if ps}
    assert any(
        "repetition.py" in f and any(injected in p for p in ps)
        for f, ps in offenders.items()
    ), f"主护栏没扫到被注入的那个文件（扫到的问题：{offenders}）"

    # 真仓库没被动过
    assert f"def {injected}(" not in (
        REPO_ROOT / "npc_agent" / "modules" / "repetition.py"
    ).read_text(encoding="utf-8"), "真文件被改到了，测试没有隔离好"
