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

from npc_agent.config import load_persona, load_scenario

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
    assert counts == {"Persona": 11, "DialogueConfig": 2, "RuntimeConfig": 22}, (
        f"配置类的字段数变了：{counts}。\n"
        "  加字段是好事，但要顺手改这个数 —— 改的时候请确认新字段**真的有人读**\n"
        "  （删掉 `min_urgency_to_speak` / `locked_topics` / `relationships` 时\n"
        "   正是因为没人读）。\n"
        "  2026-09-20：`RuntimeConfig` 21 → 22，加的是 `llm_parse_retries`；\n"
        "  它在 `cli` / `harness` / `runner` / `studio` 四处被读、且进了恢复指纹。"
    )


# --------------------------------------------------------------------------- #
# 配置解析缓存
#
# 实测一次离线 eval（235 条）里 `load_yaml` 被调用 510 次、只涉及 8 个文件
# （去重率 1.6%），单次解析约 2.9ms ⇒ 约 1.5s 花在反复读同一批 YAML。
# 缓存把 510 次连续加载从 1.49s 降到 0.018s。
#
# 但缓存有个**不会报错**的失败模式：返回**共享**对象。谁改了一份，
# 下一个调用方就拿到被改过的配置 —— 于是某几条用例莫名其妙地行为不同。
# 所以下面三条护栏：缓存要**真的生效**、返回的对象要**互不共享**、
# 且通用的 `load_yaml` **不许**被缓存（它接任意路径）。
# --------------------------------------------------------------------------- #
def _count_yaml_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """把 config.load_yaml 换成一个计数版，返回被读过的路径列表。"""
    from npc_agent import config as cfg

    reads: list[str] = []
    real = cfg.load_yaml

    def counting(path):
        reads.append(str(path))
        return real(path)

    monkeypatch.setattr(cfg, "load_yaml", counting)
    cfg.clear_config_cache()
    return reads


def test_the_persona_and_scenario_loads_are_cached(monkeypatch) -> None:
    """同一份配置读 20 次，只应该真的解析 1 次。"""
    reads = _count_yaml_reads(monkeypatch)
    for _ in range(20):
        load_persona("ayou")
        load_scenario("village")
    assert len(reads) == 2, (
        f"缓存没生效：20 轮读了两份配置，实际解析 {len(reads)} 次（期望 2）。\n"
        f"  读过的路径：{sorted(set(reads))}"
    )


def test_the_cache_returns_objects_that_are_not_shared(monkeypatch) -> None:
    """**返回的必须是各自的副本** —— 这是缓存唯一的危险面。

    如果两份拿到同一个 dict，改一份会污染缓存，进而污染**之后所有**调用方。
    这类串味（aliasing）不报错，只让行为变得依赖于调用顺序。
    """
    _count_yaml_reads(monkeypatch)
    first = load_scenario("village")
    second = load_scenario("village")

    assert first == second, "内容应该一致"
    assert first is not second, "两次加载不能是同一个对象"

    # 改第一份，第二份和"之后新拿的"都不该受影响
    first["name"] = "被我改坏了"
    assert second.get("name") != "被我改坏了", "改一份污染了另一份"
    assert load_scenario("village").get("name") != "被我改坏了", "改一份污染了缓存"


def test_the_cache_key_includes_the_folder(tmp_path, monkeypatch) -> None:
    """`personas/x.yaml` 和 `scenarios/x.yaml` 不许互相覆盖。

    ⚠️ **第一版这条护栏是假的。** 它拿 `ayou`（persona）和 `village`（scenario）
    去测 —— 两者 **id 本来就不同**，就算缓存键里不带目录也不会撞车，
    于是这条断言**永远绿**（反向测试注入"键不带 kind"时它照样绿）。
    这正是本项目栽过好几次的那个坑：**护栏没跑在它要抓的现象会发生的场景上。**

    修法：**先造出同名文件**（`personas/同名.yaml` + `scenarios/同名.yaml`，
    内容不同），再断言两边读出来不一样。前提一成立，鉴别力就有了。
    """
    # 造一个只含"同名但内容不同"两个文件的假 configs/
    (tmp_path / "personas").mkdir()
    (tmp_path / "scenarios").mkdir()
    (tmp_path / "personas" / "twin.yaml").write_text(
        "name: 我是人设\n", encoding="utf-8"
    )
    (tmp_path / "scenarios" / "twin.yaml").write_text(
        "name: 我是场景\n", encoding="utf-8"
    )
    from npc_agent import config as cfg

    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    cfg.clear_config_cache()

    persona = load_persona("twin")
    scenario = load_scenario("twin")

    # 前提断言：两个文件确实同名、内容确实不同 —— 不然这条测试没有鉴别力
    assert persona["name"] == "我是人设", f"前提不成立：{persona}"
    assert scenario["name"] == "我是场景", f"前提不成立：{scenario}"

    # 换个顺序再读一遍：缓存键不带 kind 的话，第二次会拿到第一次的结果
    assert load_scenario("twin")["name"] == "我是场景", (
        "`scenarios/twin.yaml` 读到了 `personas/twin.yaml` 的内容 —— "
        "缓存键里没带目录（kind）"
    )
    assert load_persona("twin")["name"] == "我是人设", (
        "`personas/twin.yaml` 读到了 `scenarios/twin.yaml` 的内容 —— "
        "缓存键里没带目录（kind）"
    )

    cfg.clear_config_cache()



def test_the_generic_yaml_loader_is_not_cached(monkeypatch) -> None:
    """**通用的 `load_yaml` 不许被缓存。**

    它接任意路径（`eval/compare.py` 就传外部路径）。缓存它会让
    "改了文件再读"静默拿到旧内容 —— 那是本项目最忌讳的那类静默故障。
    这条护栏钉的是"缓存只加在按 id 的两个入口上"这个边界。
    """
    from npc_agent import config as cfg

    reads: list[str] = []
    real = cfg.load_yaml
    monkeypatch.setattr(
        cfg, "load_yaml", lambda p: (reads.append(str(p)), real(p))[1]
    )
    cfg.clear_config_cache()

    cfg.load_yaml(REPO_ROOT / "configs" / "personas" / "ayou.yaml")
    cfg.load_yaml(REPO_ROOT / "configs" / "personas" / "ayou.yaml")
    assert len(reads) == 2, (
        f"通用 load_yaml 被缓存了（读了两次文件但只解析 {len(reads)} 次）—— "
        "外部路径会被静默地缓存成旧内容"
    )


def test_clear_config_cache_actually_invalidates(monkeypatch) -> None:
    """`clear_config_cache()` 之后必须重新解析 —— 否则它是个假开关。"""
    reads = _count_yaml_reads(monkeypatch)
    load_persona("ayou")
    load_persona("ayou")
    assert len(reads) == 1, "缓存没生效，先看上面那条"

    from npc_agent import config as cfg

    cfg.clear_config_cache()
    load_persona("ayou")
    assert len(reads) == 2, (
        f"clear_config_cache() 之后没有重新解析（仍是 {len(reads)} 次）—— "
        "它没有真的清掉缓存"
    )


def test_the_cache_does_not_perturb_the_resume_fingerprint() -> None:
    """**加缓存不许改变喂给裁判的上下文** —— 那会静默作废 `--resume`。

    `judge_fingerprint` 里有一项 `prompt_digest`，覆盖人设块 + 现场块。
    它的存在就是因为一条真实事故：改好现场块之后再 `--resume`，
    新旧两批判决会被拼在一起，而它们是在两套 prompt 下判的 ——
    **不报错，看起来只是"判完了"**。

    所以缓存必须保证：**清不清缓存，渲染出来的块逐字节相同**。
    深拷贝保证了这一点（内容相同、对象不同），但那是"实现上碰巧成立"，
    得有一条护栏把它变成契约。
    """
    from npc_agent import config as cfg
    from npc_agent.cli import _persona_block, _scene_block
    from npc_agent.eval.judge import prompt_digest

    def render() -> dict[str, str]:
        scenario = load_scenario("village")
        return {
            "scene:village": _scene_block(scenario, "village"),
            "persona:village:ayan": _persona_block(scenario, "阿岩"),
        }

    # ⚠️ **必须渲染三次**，而且中间那次不能清缓存 —— 否则整条测试
    # 每次都在"缓存未命中"的路径上跑，**永远碰不到缓存命中**，
    # 于是它测不出任何缓存引起的差异（第一版就是这么写的，
    # 反向测试注入"命中时篡改内容"照样绿）。
    cold = render()                     # ① 冷：真的解析
    warm = render()                     # ② 热：**命中缓存** ← 关键
    cfg.clear_config_cache()
    cold_again = render()               # ③ 清后再冷

    assert cold == warm, (
        "缓存命中时渲染出的上下文和首次解析时不一样 —— "
        "说明缓存改了喂给裁判的内容。\n"
        f"  差异：{ {k: (cold[k], warm[k]) for k in cold if cold[k] != warm[k]} }"
    )
    assert cold == cold_again, (
        "清缓存后渲染出的上下文变了。\n"
        f"  差异：{ {k: (cold[k], cold_again[k]) for k in cold if cold[k] != cold_again[k]} }"
    )
    assert prompt_digest(cold) == prompt_digest(warm) == prompt_digest(cold_again), (
        "指纹跟着缓存变了 —— `--resume` 会把两套 prompt 下的判决拼在一起"
    )



