"""import 卫生：**不许有"导进来了但没人用"的名字。**

## 为什么需要这个文件

`npc_agent/` 里有 5 处这样的残留，全都是**改动搬了家、import 忘了跟着走**：

| # | 残留 | 为什么留下来了 |
|---|---|---|
| 1 | `agent.py` `similarity` | 复读判定搬走后没人用了 |
| 2 | `agent.py` `ActionResult` | 同上 |
| 3 | `batch_report.py` `import html` | `_esc` 搬去 `report.py` 了，转义跟着走，import 没走 |
| 4 | `modules/memory.py` `field` | 那个 `field(...)` 的字段后来删了 |
| 5 | `eval/sensitivity.py` `CASES_DIR` | 挂着 `# noqa: F401 （对外暴露…）`，但**没有任何调用方从本模块取它** |

第 5 条最值得记：那行注释描述了一个**没被实现的意图**。
读者看到 `对外暴露，方便调用方定位用例` 会以为"这是给别人用的，别删"，
于是它会一直留着 —— 而真正的调用方全都直接 `from .harness import CASES_DIR`。

**为什么这类残留值得专门配一条护栏**：它们不报错、不影响分数，
但会让人**读错代码的骨架**（"这个模块用到了 html 转义" / "这里做了相似度计算"），
而且删掉一个"看起来被用着"的 import 需要先证明它真的没用 —— 人不会去证。

## 判据

用 `ast` 走一遍每个文件的**读**（`Name` / `Attribute` / 字符串注解），
再看每条 import 的名字有没有出现在里面。只报**高置信度**的：

* `from __future__ import ...` 是**编译器指令**，永远不出现在 `Name` 节点里
  ⇒ 必须显式跳过，否则 41 个文件里报 37 个假阳性，
  而**会误报的护栏比没有护栏更糟**（它迟早被 `--deselect` 掉）。
* `import a.b.c` 只看顶层名 `a`（`a.b.c` 全写出来时 `Name` 节点里只有 `a`）。
* `as` 别名按别名判。
* `import *` 跳过（名字是动态的，判不了）。
"""

from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "npc_agent"


def _used_names(tree: ast.AST) -> set[str]:
    """收集所有被"读"的名字。

    三类都要收：
      * `Name` —— 普通引用；
      * `Attribute` —— 取 `a.b` 时把根名 `a` 记上（`a.b.c` 的 `Name` 只有 `a`）；
      * `Constant(str)` —— 字符串型注解 / 前向引用（`-> "Foo"`）里的名字。
    """
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            cur: ast.AST = node
            while isinstance(cur, ast.Attribute):
                cur = cur.value
            if isinstance(cur, ast.Name):
                used.add(cur.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for token in (
                node.value.replace("[", " ")
                .replace("]", " ")
                .replace(",", " ")
                .replace("|", " ")
                .split()
            ):
                used.add(token.strip("\"'"))
    return used


def _unused_imports(source: str) -> list[str]:
    """返回形如 `L12: similarity` 的列表。语法错时抛 SyntaxError（由调用方处理）。"""
    tree = ast.parse(source)
    used = _used_names(tree)
    problems: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = (alias.asname or alias.name).split(".")[0]
                if name not in used:
                    problems.append(f"L{node.lineno}: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            # 编译器指令，不是名字 —— 跳过（否则全是假阳性）。
            if node.module == "__future__":
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                name = alias.asname or alias.name
                if name not in used:
                    where = node.module or "."
                    problems.append(f"L{node.lineno}: {alias.name}（from {where}）")
    return problems


def _package_files() -> list[Path]:
    return sorted(
        p for p in PACKAGE_DIR.rglob("*.py") if "__pycache__" not in str(p)
    )


# --------------------------------------------------------------------------- #
# 主护栏
# --------------------------------------------------------------------------- #
def test_the_package_has_no_unused_imports() -> None:
    """`npc_agent/` 里每个 import 都必须有人用。"""
    offenders: dict[str, list[str]] = {}
    for path in _package_files():
        problems = _unused_imports(path.read_text(encoding="utf-8"))
        if problems:
            offenders[str(path.relative_to(REPO_ROOT))] = problems

    assert not offenders, (
        "有导进来却没人用的名字：\n"
        + "\n".join(
            f"  {f}\n" + "\n".join(f"      {p}" for p in ps)
            for f, ps in offenders.items()
        )
        + "\n  删掉它们。如果那是有意的**再导出**（别的模块从这里取），"
        "请在注释里写清**是哪个调用方**在取 —— 写不出调用方就说明没人取。"
    )


# --------------------------------------------------------------------------- #
# 反向测试：这条检查必须能红
# --------------------------------------------------------------------------- #
def test_the_unused_import_check_can_actually_fail(tmp_path: Path) -> None:
    """喂一个**真的**有未使用 import 的模块，必须被抓出来。

    没有这一条，`_unused_imports` 哪天因为 AST 遍历写错而恒返回 `[]`，
    上面那条会**永远是绿的** —— 而它会以"import 都很干净"的姿态骗过所有人。
    """
    source = (
        "import os\n"
        "import json\n"
        "\n"
        "def f(x):\n"
        "    return json.dumps(x)\n"
    )
    problems = _unused_imports(source)
    assert problems == ["L1: os"], (
        f"检查没能识别出未使用的 import（抓到的是 {problems}）—— 这条护栏是摆设。\n"
        "  期望只抓到 ['L1: os']：`json` 被 `f()` 用了，`os` 一次都没被用。"
    )


def test_the_check_does_not_flag_a_future_import() -> None:
    """**`from __future__ import annotations` 不是未使用。**

    它永远不出现在 `Name` 节点里（是编译器指令），按"未使用"报它是纯误报。
    第一版就是这么写的：41 个文件里报了 37 个假阳性。
    **会误报的护栏比没有护栏更糟** —— 它迟早被 `--deselect` 掉，
    然后连真的问题也一起放过。
    """
    source = (
        "from __future__ import annotations\n"
        "\n"
        "def f() -> int:\n"
        "    return 1\n"
    )
    assert _unused_imports(source) == []


def test_the_check_reads_attribute_roots_and_aliases() -> None:
    """`import a.b.c` 只写 `a` 也算用到；`as` 别名按别名判。"""
    # a.b.c 的 Name 节点里只有 a
    assert _unused_imports("import urllib.request\nurllib.request.urlopen('x')\n") == []
    # 别名：用了别名就不算未使用；用了原名而没定义它 ⇒ 会被报（正确行为）
    assert _unused_imports("import numpy as np\nnp.array([1])\n") == []
    assert _unused_imports("import numpy as np\n") == ["L1: numpy"]


def test_the_check_sees_names_used_only_in_annotations() -> None:
    """只在**字符串注解**里出现的名字也算用到。

    本项目大量用 `from __future__ import annotations`，
    注解会被解析成字符串常量 —— 漏掉这一类就会把真正在用的 import 报成未使用。
    """
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from decimal import Decimal\n"
        "\n"
        "def f(x: 'Decimal') -> 'Decimal':\n"
        "    return x\n"
    )
    assert _unused_imports(source) == []


def test_the_guard_would_catch_a_regression_in_the_real_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """往**真实文件的副本**里注入一个未使用的 import，主护栏必须变红。

    前几条测的是纯函数；这一条测的是"主护栏真的在扫 `npc_agent/`"。
    两者缺一不可 —— 纯函数对了但扫描范围写错（比如 `glob` 拼错）照样是死护栏。

    ⚠️ 用**副本**而不是原地改真文件：原地改的话，测试中途崩掉会把仓库
    留在一个被注入过的状态（而 `--basetemp` 那类事故已经教过我们这个教训）。
    """
    # 把真个包复制到 tmp，再把扫描根指过去 —— 文件列表仍然是真实的那份。
    shadow = tmp_path / "npc_agent"
    shutil.copytree(
        PACKAGE_DIR, shadow, ignore=shutil.ignore_patterns("__pycache__")
    )
    # 本目录没有 `__init__.py`（`tests` 不是包），所以按模块对象改全局。
    monkeypatch.setattr(sys.modules[__name__], "PACKAGE_DIR", shadow)

    target = shadow / "modules" / "repetition.py"
    original = target.read_text(encoding="utf-8")
    assert "import collections" not in original, "锚点已经存在，换个名字"
    target.write_text(
        original.replace(
            "from __future__ import annotations",
            "from __future__ import annotations\n\nimport collections  # 注入",
            1,
        ),
        encoding="utf-8",
    )

    problems = _unused_imports(target.read_text(encoding="utf-8"))
    assert any("collections" in p for p in problems), (
        f"注入的未使用 import 没被抓到（抓到的是 {problems}）"
    )

    # 关键：走一遍**主护栏的扫描逻辑**（`_package_files()` 读的是被改过的根），
    # 否则"扫描范围写错"这类缺陷测不出来。
    scanned = {
        str(p.relative_to(shadow)): _unused_imports(p.read_text(encoding="utf-8"))
        for p in _package_files()
    }
    offenders = {f: ps for f, ps in scanned.items() if ps}
    assert any(
        "repetition.py" in f and any("collections" in p for p in ps)
        for f, ps in offenders.items()
    ), f"主护栏没扫到被注入的那个文件（扫到的问题：{offenders}）"

    # 真仓库没被动过
    assert "import collections" not in (
        REPO_ROOT / "npc_agent" / "modules" / "repetition.py"
    ).read_text(encoding="utf-8"), "真文件被改到了，测试没有隔离好"
