"""配置字段的卫生检查：**不许有"写了但没人读"的字段。**

## 为什么需要这个文件

这个仓库里同一个坑踩了**三次**，而且一次比一次严重：

| # | 死掉的东西 | 在哪 | 为什么危险 |
|---|---|---|---|
| 1 | `QUESTION_MARKERS = ("?", "？")` | `modules/dialogue.py` | 名字让人以为"问句判定就在这儿"，真判据在 `types.looks_like_question` |
| 2 | `HOST_INTENTS = (...)` | `modules/dialogue.py` | 同上，读者会去改一个没人读的常量 |
| 3 | `DialogueConfig.min_urgency_to_speak` | `modules/dialogue.py` | **长得像可调阈值** —— 调它不会有任何效果 |
| 4 | `Persona.locked_topics` | `modules/persona.py` | **从 YAML 读进来的**，配置里还写着注释"防止一上来就剧透" |
| 5 | `Persona.relationships` | `modules/persona.py` | **从 YAML 读进来的**，三个 persona 都写了 |

第 1、2 条只骗**读代码**的人。第 4、5 条连**改配置**的人都骗了 ——
`configs/personas/ayou.yaml` 里写着 `locked: [hidden_menu]`，
`from_dict` 老老实实解析进字段，然后全项目没有一处读它。
一个改配置的人会以为自己防住了剧透，其实什么都没发生。

**配置里写着一个不生效的开关，等于对使用者撒谎。**
这比"没有这个字段"糟得多，因为对方连怀疑的入口都没有。

## 这条检查怎么判

只扫**配置类**（使用者会去改的那些），不扫数据载体：

- `Persona` —— 从 `configs/personas/*.yaml` 读
- `DialogueConfig` —— 发言调度阈值
- `RuntimeConfig` —— 一次运行的全部可调参数

判据：**每个注解字段都必须在 `npc_agent/` 里被当作属性读过**（`xxx.<字段名>`）。
只出现定义、不出现读取 ⇒ 红。

为什么不扫 `types.py` / `reflection.py` 那些类：它们的字段是**数据载体**
（`Plan.rationale`、`Verdict.raw`、`MemoryRecord.tags`…），
靠 `dataclasses.asdict()` 或报告渲染消费，属性访问是 0 次但**并不是死字段**。
把它们一起扫会得到 7 个假阳性 —— 而**会误报的护栏比没有护栏更糟**，
它迟早被人 `--deselect` 掉（这个仓库在 `ablation.html` 那次事故里学到的）。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
PKG = REPO_ROOT / "npc_agent"

#: 要检查的配置类：类名 → 它所在的文件。
#:
#: 加新配置类时**加到这里**。别为了让它变绿而把字段删出检查范围 ——
#: 要么这个字段真有人读，要么它就不该存在。
CONFIG_CLASSES: dict[str, str] = {
    "Persona": "npc_agent/modules/persona.py",
    "DialogueConfig": "npc_agent/modules/dialogue.py",
    "RuntimeConfig": "npc_agent/config.py",
}

#: 允许"只定义、没人读"的字段，格式 `类名.字段名` → 理由。
#:
#: 现在是空的，而且**应该保持为空**。真要用它，理由必须是
#: "这个字段被 X 机制消费，属性访问看不见" —— 而不是"删不掉"。
ALLOWED_UNREAD: dict[str, str] = {}


def _package_source() -> str:
    """整个包的源码拼在一起，用来数"这个字段被读过几次"。"""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in PKG.rglob("*.py")
    )


def _annotated_fields(path: Path, class_name: str) -> list[str]:
    """取出某个类里带注解的字段名（`x: int = 0` 这种）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name),
        None,
    )
    assert node is not None, f"{path} 里找不到类 {class_name} —— 改名了就来改这里"
    return [
        st.target.id
        for st in node.body
        if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)
    ]


def _unread_fields(class_name: str, path: Path, source: str) -> list[str]:
    """只被定义、从没被当作属性读过的字段。"""
    out = []
    for field in _annotated_fields(path, class_name):
        if f"{class_name}.{field}" in ALLOWED_UNREAD:
            continue
        if not re.search(rf"\b\w+\.{re.escape(field)}\b", source):
            out.append(field)
    return out


@pytest.mark.parametrize("class_name", sorted(CONFIG_CLASSES))
def test_every_config_field_is_actually_read(class_name: str) -> None:
    """配置类的每个字段都必须有人读。

    这条抓的是"写了但没人读"的字段 —— 见模块开头那张表（同一个坑踩了三次）。
    """
    path = REPO_ROOT / CONFIG_CLASSES[class_name]
    source = _package_source()
    unread = _unread_fields(class_name, path, source)
    assert not unread, (
        f"{class_name} 里有**没人读**的字段：{unread}。\n"
        "  这些字段会被解析、会出现在配置里，但**不会有任何效果** ——\n"
        "  改配置的人会以为它生效了。要么接上消费方，要么删掉它\n"
        "  （连着 YAML 里那一行一起删，否则配置里留着一个骗人的开关）。\n"
        "  若确有机制在属性访问之外消费它，把它加进 ALLOWED_UNREAD 并写明理由。"
    )


def test_the_unread_field_check_can_actually_fail(tmp_path: Path) -> None:
    """反向：造一个真没人读的字段，这条检查必须红。

    没有这一条的话，`_unread_fields` 哪天因为正则写错而恒返回 `[]`，
    上面那条会**永远是绿的** —— 而它会以"配置字段都有人读"的姿态骗过所有人。
    这个仓库对"不会红的护栏"有专门的教训（见 `test_test_hygiene.py` 开头）。

    做法：写一个临时模块，里面有一个**真没人读**的字段和一个有人读的字段，
    断言检查只抓前者。
    """
    module = tmp_path / "fake_config.py"
    module.write_text(
        "class FakeConfig:\n"
        "    used: int = 0\n"      # 有人读（下面读了）
        "    dead: int = 1\n"      # 没人读 —— 必须被抓出来
        "\n"
        "def consume(cfg: FakeConfig) -> int:\n"
        "    return cfg.used\n",
        encoding="utf-8",
    )
    source = module.read_text(encoding="utf-8")

    unread = _unread_fields("FakeConfig", module, source)
    assert unread == ["dead"], (
        f"检查没能识别出没人读的字段（抓到的是 {unread}）—— 这条护栏是摆设。\n"
        "  期望只抓到 ['dead']：`used` 被 `consume()` 读了，`dead` 一次都没被读。"
    )


def test_the_real_config_classes_have_no_unread_fields() -> None:
    """上面那条按类参数化，这条给一个总览 —— 顺便把字段数钉住。

    字段数是**故意**钉的：删字段时如果忘了改这里，说明这次改动没被审过。
    """
    source = _package_source()
    counts = {}
    for class_name, rel in CONFIG_CLASSES.items():
        path = REPO_ROOT / rel
        fields = _annotated_fields(path, class_name)
        assert not _unread_fields(class_name, path, source), (
            f"{class_name} 有没人读的字段"
        )
        counts[class_name] = len(fields)
    assert counts == {"Persona": 11, "DialogueConfig": 2, "RuntimeConfig": 19}, (
        f"配置类的字段数变了：{counts}。\n"
        "  加字段是好事，但要顺手改这个数 —— 改的时候请确认新字段**真的有人读**\n"
        "  （删掉 `min_urgency_to_speak` / `locked_topics` / `relationships` 时\n"
        "   正是因为没人读）。"
    )
