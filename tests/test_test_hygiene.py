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
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent


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
