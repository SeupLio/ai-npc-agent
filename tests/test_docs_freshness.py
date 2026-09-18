"""`docs/` 里的报告必须和代码对得上 —— 至少**离线那份**必须。

## 为什么要有这条

`docs/` 里混了两种东西，以前文档里没区分：

| 种类 | 谁能重生成 | 例子 |
|---|---|---|
| **离线可复现** | 任何人，不需要模型/网络 | `ablation.html`、`worlds.html` |
| **一次跑批的快照** | 要有模型 + 额度 | `batch_*.html`、`comparison.html`、`multi_npc.html` |

**离线那份一旦和代码不一致，它就是在说谎**：读者会以为那是当前代码的输出。
这个仓库真出过事 —— 入库的 `ablation.html` 还是 12 条用例时代的产物
（5 列指标，没有「发言调度」），而代码早就是 228 条 / 6 列了。

快照那份冻结在某个时间点是**合理的**，但要有人明确把它归到"快照"里，
不能靠"没人注意"。

## 三条断言

1. 每个 `docs/*.html` 都出现在 README 的报告表里（新报告不能偷偷加）
2. 每个 `docs/*.html` 都被**显式分类**成离线可复现或快照（不许"没分类"）
3. 离线那几份和当前代码生成的结果**除时长外逐字节一致**

第 3 条要说明一下：报告里嵌了每条的**耗时**（`0.18s` / `14s`），
它天然每次都不一样，所以比的是"抹掉时长之后一致"。
归一化只抹时长 —— 别的数字（通过数、指标均值、用例名）一个都不许抹，
抹多了这条断言就变成永真式。函数在 `scripts/regen_docs.py:normalize_report`。

第 3 条里 `ablation` 要跑约 2 分钟，所以默认跳过（和端到端测试同一个约定）：
设 `NPC_AGENT_DOC_FRESHNESS=1` 才跑。`worlds` 是秒级，默认就跑。
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
README = ROOT / "README.md"

_REGEN_PATH = ROOT / "scripts" / "regen_docs.py"


def _load_regen():
    """按路径加载 scripts/regen_docs.py（scripts/ 不是包）。"""
    spec = importlib.util.spec_from_file_location("_regen_docs_under_test", _REGEN_PATH)
    assert spec and spec.loader, f"加载不了 {_REGEN_PATH}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


regen_docs = _load_regen()

# 秒级，默认就跑
FAST_OFFLINE = ("worlds",)
# 约 2 分钟，默认跳过；设 NPC_AGENT_DOC_FRESHNESS=1 才跑
SLOW_OFFLINE = ("ablation",)


def _docs_html() -> list[str]:
    return sorted(p.name for p in DOCS.glob("*.html"))


def test_every_doc_report_is_documented_in_the_readme() -> None:
    """新加一份报告却不写进 README，读者就不知道它存在。"""
    text = README.read_text(encoding="utf-8")
    missing = [n for n in _docs_html() if f"docs/{n}" not in text]
    assert not missing, (
        f"这些报告在 docs/ 里但 README 没提：{missing}。"
        "README 的报告表要列全，否则没人知道它存在。"
    )


def test_every_doc_report_is_explicitly_classified() -> None:
    """每份报告都要**明确**属于「离线可复现」或「一次跑批的快照」。

    这一条是这次事故的核心：以前没有这个分类，于是"离线报告早就过期了"
    没有任何机制能发现。分类本身不需要人聪明，只需要人**做一次选择**。
    """
    offline = {f for _, f in regen_docs.OFFLINE_REPORTS.values()}
    snapshots = set(regen_docs.SNAPSHOT_REPORTS)
    unclassified = [n for n in _docs_html() if n not in offline and n not in snapshots]
    assert not unclassified, (
        f"这些报告没有被分类：{unclassified}。"
        "请把它加进 scripts/regen_docs.py 的 OFFLINE_REPORTS（离线可复现）"
        "或 SNAPSHOT_REPORTS（需要模型 + 额度的一次跑批快照）。"
    )


def test_the_classification_does_not_drift_from_reality() -> None:
    """分类不能和文件本身矛盾：含模型输出的不许被归成"离线可复现"。

    离线可复现的意思是"不需要模型就能重生成"。一份报告里如果嵌着模型名，
    它就不可能满足这个条件 —— 把它归错类，第 3 条断言就会变成
    "拿模型跑批的结果去和离线生成的结果比"，永远红。
    """
    model_marker = "kimi-k2.7-code"
    wrong = []
    for _, filename in regen_docs.OFFLINE_REPORTS.values():
        path = DOCS / filename
        if path.exists() and model_marker in path.read_text(encoding="utf-8"):
            wrong.append(filename)
    assert not wrong, (
        f"这些报告里嵌着模型输出（{model_marker}），却被归成「离线可复现」：{wrong}。"
    )


def _assert_matches_code(name: str, tmp_path: Path) -> None:
    committed = DOCS / regen_docs.OFFLINE_REPORTS[name][1]
    assert committed.exists(), f"{committed} 不存在"

    out = tmp_path / name
    regen_docs.regen(name, out)
    fresh = out / regen_docs.OFFLINE_REPORTS[name][1]

    # 报告里嵌了**时长**，它天然每次都不同，所以比的是"除时长外逐字节一致"。
    # 归一化函数在 scripts/regen_docs.py 里，只抹时长 —— 别的数字一个都不许抹。
    got = regen_docs.normalize_report(committed.read_text(encoding="utf-8"))
    want = regen_docs.normalize_report(fresh.read_text(encoding="utf-8"))
    if got == want:
        return

    # 只说"不一致"没用，得让下一个人知道怎么修。
    pytest.fail(
        f"docs/{committed.name} 和当前代码生成的结果不一致 —— 它已经过期了。\n"
        f"重新生成：python scripts/regen_docs.py --only {name}\n"
        f"（这是一份**离线可复现**的报告，和代码不一致就等于在说谎。）"
    )


@pytest.mark.parametrize("name", FAST_OFFLINE)
def test_fast_offline_reports_match_the_code(name: str, tmp_path: Path) -> None:
    """秒级的那几份，每次都验。"""
    _assert_matches_code(name, tmp_path)


@pytest.mark.parametrize("name", SLOW_OFFLINE)
def test_slow_offline_reports_match_the_code(name: str, tmp_path: Path) -> None:
    """分钟级的那几份，默认跳过（和端到端测试同一个约定）。"""
    if os.environ.get("NPC_AGENT_DOC_FRESHNESS") != "1":
        pytest.skip(f"没设 NPC_AGENT_DOC_FRESHNESS=1，跳过慢速报告校验（{name}）")
    _assert_matches_code(name, tmp_path)


def test_the_freshness_check_itself_can_fail(tmp_path: Path) -> None:
    """这条护栏必须能红。

    做法：故意把一份报告的入库版本改坏，确认比对会失败 ——
    而不是"两边都是空文件所以相等"这种假通过。
    """
    name = FAST_OFFLINE[0]
    filename = regen_docs.OFFLINE_REPORTS[name][1]
    committed = DOCS / filename
    backup = tmp_path / "backup.html"
    shutil.copy2(committed, backup)
    try:
        committed.write_text("故意改坏的内容", encoding="utf-8")
        # 注意：pytest 的 `Failed` 继承自 **BaseException**，不是 Exception ——
        # 写 `pytest.raises(Exception)` 抓不住它，于是这条自检会自己红，
        # 而"护栏坏了"和"护栏抓到了"看起来一模一样。用确切的类型。
        with pytest.raises(pytest.fail.Exception):
            _assert_matches_code(name, tmp_path / "out")
    finally:
        shutil.copy2(backup, committed)
    # 恢复之后必须又能通过 —— 否则这条测试本身是坏的
    _assert_matches_code(name, tmp_path / "out2")
