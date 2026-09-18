"""自测控制台（`npc_agent/studio.py`）的护栏。

## 这里在守什么

控制台是**一层壳**：它不重写任何逻辑，只是把 `Cast` / `EvalHarness` /
`run_sensitivity` 包了一层 HTTP。所以最该守的是这一条：

> **网页上的数字必须和命令行逐字一致。**

不一致只有一种成因 —— 有人在这里抄了一份逻辑。那正是这个项目
反复栽跟头的地方（`dialogue.py` 和 `tools.py` 各硬编码了一份 0.62；
报告清单被抄成三份）。所以这里用**差分测试**钉住：控制台的输出
和直接调库的输出必须相等。

另外两条也是踩出来的：

1. **注解不是默认值。** `_Handler` 上写 `config: RuntimeConfig` 只是类型注解，
   类上并没有这个属性；而 `send_response` 内部会调 `log_message`，
   它读 `self.config` —— 于是**每一个请求**都以 `AttributeError` 收场，
   客户端只看到 `RemoteDisconnected`（服务端把连接关了，什么都没回）。
   错误发生在日志函数里，病因完全看不见。
2. **页面里不许有硬编码数字。** 页面一旦写上"231 条用例"，
   它就变成了一份会过期的文档 —— 和 README 里那段手抄的基线是同一个病。
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from npc_agent import studio as S
from npc_agent.config import RuntimeConfig
from npc_agent.eval.harness import EvalHarness, EvalReport
from npc_agent.eval.report_index import DOCS_DIR
from npc_agent.eval.report_index import index as report_index
from npc_agent.eval.sensitivity import MUTANTS, run_sensitivity
from npc_agent.studio_ui import PAGE

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cfg() -> RuntimeConfig:
    """**刻意不读环境变量** —— 用 `RuntimeConfig()` 而不是 `from_env()`。

    否则在一台配了 `NPC_AGENT_PROVIDER` 的机器上，这套测试会真的去调模型：
    慢、烧额度、而且结果不可复现。测试必须离线。
    """
    return RuntimeConfig()


# --------------------------------------------------------------------------- #
# 1. 控制台 == 命令行
# --------------------------------------------------------------------------- #
def test_eval_numbers_equal_the_library_path(cfg: RuntimeConfig) -> None:
    """控制台报的 summary 必须等于直接跑 harness 的 summary。

    判据是**整个 summary 字典相等**，不是"通过数差不多"。
    只比通过数的话，"六维均值算错了"照样能混过去。
    """
    got = S.run_eval(cfg, {"category": "persona", "limit": 6})

    harness = EvalHarness(cfg)
    report = EvalReport()
    for case in harness.load_cases(["persona"])[:6]:
        report.results.append(harness.run_case(case))

    assert got["summary"] == report.to_dict()["summary"]
    assert got["summary"]["total"] == 6
    assert [row["id"] for row in got["cases"]] == [
        r.case_id for r in report.results
    ]


def test_mutant_numbers_equal_the_sensitivity_path(cfg: RuntimeConfig) -> None:
    """控制台的变异结果必须等于直接跑 `run_sensitivity` 的结果。

    这一条比上一条更要紧：变异会**临时改类属性**。控制台自己再写一套注入
    逻辑的话，"网页说抓到了、命令行说没抓到"会变成一个无法排查的幽灵。
    """
    payload = {"mutant": "floor_control_disabled", "category": "multi_npc", "limit": 12}
    got = S.run_mutant(cfg, payload)

    want = run_sensitivity(
        categories=["multi_npc"],
        limit=12,
        config=cfg,
        mutants=tuple(m for m in MUTANTS if m.id == "floor_control_disabled"),
    )
    outcome = want.outcomes[0]

    assert got["deltas"] == outcome.deltas
    assert got["baseline_means"] == want.baseline_means
    assert got["mutant_means"] == outcome.metric_means
    assert got["baseline_pass"] == want.baseline_pass_rate
    assert got["mutant_pass"] == outcome.pass_rate
    assert got["caught"] == outcome.caught


def test_retrieval_actually_moves_the_memory_dimension(cfg: RuntimeConfig) -> None:
    """「关掉检索」必须在 memory 这一类上**被抓到**。

    钉的是一个真实的历史 bug：memory 维度曾经 37 条里只有 3 条真的在断言
    "回忆"（其余 31 条断言的是 `memory_contains`，读的是记忆库的原始列表，
    对检索结构上盲）。于是关掉检索只掉 **0.007** —— 而 `caught` 仍然是 `True`，
    报告仍然全绿。补到 6 条之后才掉到 0.013。

    如果哪天有人把回忆用例删回去，这条会红。
    """
    got = S.run_mutant(cfg, {"mutant": "retrieval_disabled", "category": "memory"})
    assert got["caught"], "关掉检索却没被抓住 —— memory 维度又变成'只写不读'了"
    assert got["target_hit"]
    assert got["deltas"]["memory"] < 0


def test_weak_flag_is_only_meaningful_when_caught(cfg: RuntimeConfig) -> None:
    """`weak` 的判据：**抓到** 且 目标维度掉分小于阈值。

    没抓到的时候"目标维度没动"是必然的，再说一遍"覆盖面薄"
    会把两个不同的结论搅在一起。
    """
    got = S.run_mutant(cfg, {"mutant": "planning_disabled", "category": "task", "limit": 20})
    expected = bool(
        got["caught"]
        and got["targets"]
        and abs(got["deltas"][got["targets"][0]]) < S.WEAK_DELTA
    )
    assert got["weak"] is expected


def test_unknown_mutant_is_rejected_not_silently_ignored(cfg: RuntimeConfig) -> None:
    """拼错变异名必须报错，不能"什么都不注入然后报全绿"。"""
    with pytest.raises(ValueError, match="未知的变异"):
        S.run_mutant(cfg, {"mutant": "no_such_mutant"})


# --------------------------------------------------------------------------- #
# 2. 子集不能当结论
# --------------------------------------------------------------------------- #
def test_partial_flag_marks_a_subset(cfg: RuntimeConfig) -> None:
    """跑了子集就必须标出来。

    真实症状：`retrieval_disabled` 在前 80 条里**没有**回忆用例，
    于是显示"没抓到 ✗ —— 这是一个评测盲区"。那是**取样造成的**，
    不是评测的结论。不标 `partial`，界面就会把人引向一个错误结论。
    """
    full = S.run_mutant(cfg, {"mutant": "planning_disabled", "category": "minecraft"})
    part = S.run_mutant(
        cfg, {"mutant": "planning_disabled", "category": "minecraft", "limit": 5}
    )
    assert full["partial"] is False
    assert part["partial"] is True
    assert part["total"] == 5


# --------------------------------------------------------------------------- #
# 3. 对话：可复现 + 输入要挡
# --------------------------------------------------------------------------- #
CHAT_EVENTS = [
    {"kind": "say", "speaker": "player_a", "text": "今天这里挺热闹的。"},
    {"kind": "idle"},
    {"kind": "say", "speaker": "player_b", "text": "露台那边好像有风。"},
]


def test_chat_is_reproducible(cfg: RuntimeConfig) -> None:
    """同一段输入必须给出**逐字节相同**的输出。

    控制台是无状态的（每次请求从头重放），所以这本来就该成立 ——
    但"本来就该"不是证据。可复现是这个项目的底线：
    一个偶尔不一样的演示，没法录屏，也没法当证据。
    """
    payload = {"scenario": "duet", "events": CHAT_EVENTS}
    first = S.run_chat(cfg, payload)
    second = S.run_chat(cfg, payload)
    dump = lambda d: json.dumps(d, ensure_ascii=False, sort_keys=True)
    assert dump(first) == dump(second)
    assert len(first["events"]) == len(CHAT_EVENTS)


def test_chat_skips_blank_text_and_unknown_speaker(cfg: RuntimeConfig) -> None:
    """空白输入和伪造的说话人都不该把请求搞崩，也不该被当成一次发言。"""
    got = S.run_chat(
        cfg,
        {
            "scenario": "tutorial",
            "events": [
                {"kind": "say", "speaker": "player_a", "text": "   "},
                {"kind": "say", "speaker": "根本不存在的玩家", "text": "你好"},
            ],
        },
    )
    # 第一条被跳过；第二条回落到场景里的第一个玩家
    assert len(got["events"]) == 1
    assert got["events"][0]["speaker_id"] == "player_a"


def test_chat_caps_the_event_count(cfg: RuntimeConfig) -> None:
    """事件数有上限 —— 前端每发一句就把**整段**历史重发一遍。

    没上限的话，一个循环发消息的页面能把单次请求的计算量顶到任意大。
    """
    events = [{"kind": "idle"}] * (S.MAX_EVENTS + 7)
    got = S.run_chat(cfg, {"scenario": "tutorial", "events": events})
    assert len(got["events"]) == S.MAX_EVENTS


def test_chat_reports_world_state_and_memories(cfg: RuntimeConfig) -> None:
    """界面右侧那三栏必须有真数据，否则它只是个装饰。"""
    got = S.run_chat(cfg, {"scenario": "duet", "events": CHAT_EVENTS})
    assert got["world_label"]
    assert "world_flags" in got["snapshot"] and "objectives" in got["snapshot"]
    assert got["snapshot"]["objectives"], "目标状态不该是空的"
    assert got["memories"], "记忆库不该是空的"
    for payload in got["memories"].values():
        assert {"episodic", "semantic", "reflection", "consolidated"} <= set(payload)
    assert got["collisions"] == 0, "这两个 NPC 不该抢话"
    assert got["speech"], "发言统计不该是空的"


# --------------------------------------------------------------------------- #
# 4. 报告门户
# --------------------------------------------------------------------------- #
def test_portal_lists_every_report_in_docs() -> None:
    """`docs/` 里每一个 html 都要出现在门户里。

    漏一个的后果不是"少了个链接"，是那份报告**彻底隐身** ——
    而它恰恰可能是最该被看见的那个（`ablation.html` 就曾经停在
    12 条用例时代、5 列指标，没人发现）。
    """
    on_disk = {p.name for p in DOCS_DIR.glob("*.html")}
    assert on_disk, "docs/ 里一份报告都没有？先确认工作目录"
    listed = {row["file"] for row in report_index()}
    assert listed == on_disk


def test_every_listed_report_has_a_title_and_an_answer() -> None:
    """每份报告都要有"叫什么"和"回答什么问题"。

    没有标题，门户只能显示文件名；没有"回答什么"，读者不知道要不要点开。
    判据是 `known` 标志 —— 它就是为这条护栏准备的。
    """
    unknown = [row["file"] for row in report_index() if not row["known"]]
    assert not unknown, (
        f"这些报告没有标题/说明：{unknown}。"
        "请把它们加进 `npc_agent/eval/report_index.py` 的 REPORT_TITLES / REPORT_ANSWERS。"
    )


def test_the_portal_guard_can_actually_flag_an_unknown_file(tmp_path: Path) -> None:
    """反向测试：上面那条必须真的能红。

    造一个 `docs/` 里没有的新报告，门户要把它标成 `known=False`
    并且**照样列出来**（以文件系统为准，不以清单为准）。
    """
    (tmp_path / "brand_new.html").write_text(
        "<html><body>共 5 条用例</body></html>", encoding="utf-8"
    )
    rows = report_index(tmp_path)
    assert len(rows) == 1
    assert rows[0]["known"] is False
    assert rows[0]["coverage"] == "5 条"


def test_unparseable_coverage_is_reported_not_zeroed(tmp_path: Path) -> None:
    """解析不出覆盖数时**不能填 0**。

    0 会被下游当成一个合法的覆盖数（"这份报告覆盖 0 条"），
    而真相是"我没解析出来"。这个区别决定排查方向对不对。
    """
    (tmp_path / "weird.html").write_text("<html>没有用例数</html>", encoding="utf-8")
    rows = report_index(tmp_path)
    assert rows[0]["cases"] is None
    assert rows[0]["coverage"] == "（未标注）"


def _load_regen_docs():
    """按路径加载 `scripts/regen_docs.py`（`scripts/` 不是包）。"""
    path = ROOT / "scripts" / "regen_docs.py"
    spec = importlib.util.spec_from_file_location("_regen_docs_studio_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_index_and_regen_docs_share_one_source() -> None:
    """报告清单只能有一份。

    原来它只活在 `scripts/regen_docs.py` 里，于是"重生成"、"过期护栏"、
    "报告门户"三个消费方各抄一份 —— 抄的东西必然漂移。
    这条断言的是**同一对象**（`is`），不是"内容相等"：
    内容相等只说明这次碰巧一样，明天就不一定了。
    """
    from npc_agent.eval import report_index as RI

    regen = _load_regen_docs()
    assert regen.OFFLINE_REPORTS is RI.OFFLINE_REPORTS
    assert regen.SNAPSHOT_REPORTS is RI.SNAPSHOT_REPORTS
    for sample in (
        "<div>228 条自建用例</div>",
        "<div>2 个世界、共 231 条用例</div>",
        "通过 <strong>12/12</strong>，自由台词 0%",
        "<html>没有数字</html>",
    ):
        assert regen.case_count(sample) == RI.case_count(sample)


# --------------------------------------------------------------------------- #
# 5. 页面本身
# --------------------------------------------------------------------------- #
def test_page_is_self_contained() -> None:
    """自包含：没有外链、没有 CDN、没有外部脚本。

    和 `docs/` 里那几份报告同一条约定 —— 它要能直接放进作品集里打开，
    而且离线环境（面试现场的网络）也打不开 CDN。
    """
    lowered = PAGE.lower()
    assert "http://" not in lowered and "https://" not in lowered
    assert "<script src" not in lowered
    assert "<link " not in lowered
    assert lowered.lstrip().startswith("<!doctype html")


def test_page_has_no_hardcoded_case_count() -> None:
    """页面里不许写死用例条数。

    写死了它就变成一份**会过期的文档** —— 和 README 里那段手抄的
    离线基线是同一个病（那段已经过期过两次：12 → 228 → 231）。
    所有数字都必须从 `/api/meta` 拿。
    """
    total = len(EvalHarness(RuntimeConfig()).load_cases())
    assert str(total) not in PAGE, (
        f"页面里出现了用例总数 {total} —— 这个数会变，必须从 /api/meta 读，不能硬编码"
    )


# --------------------------------------------------------------------------- #
# 5b. 页面真的能渲染吗
#
# 这一段是"我没有浏览器也要验证前端"的替代方案。缺了它，我交付的是一个
# **从来没被打开过**的页面 —— 一个拼错的字段名不会让任何测试变红，
# 只会让某一栏永远空白，而且看起来像"数据没准备好"。
# --------------------------------------------------------------------------- #
def _page_script() -> str:
    return PAGE.split("<script>", 1)[1].rsplit("</script>", 1)[0]


def _strip_js_literals(src: str) -> str:
    """去掉字符串字面量和注释。

    不去掉的话，`document.querySelectorAll(".panel")` 里的 `.panel`
    会被当成"页面读了字段 `panel`" —— 于是护栏被 CSS 选择器逼着加白名单，
    慢慢变成噪音。**先剥掉，再谈字段。**
    """
    out: list[str] = []
    index, size = 0, len(src)
    while index < size:
        char = src[index]
        if char in "\"'`":
            quote, index = char, index + 1
            while index < size:
                if src[index] == "\\":
                    index += 2
                    continue
                if src[index] == quote:
                    index += 1
                    break
                index += 1
            out.append(" ")
            continue
        if char == "/" and index + 1 < size and src[index + 1] == "/":
            while index < size and src[index] != "\n":
                index += 1
            continue
        if char == "/" and index + 1 < size and src[index + 1] == "*":
            index += 2
            while index + 1 < size and not (src[index] == "*" and src[index + 1] == "/"):
                index += 1
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _property_names(script: str) -> set[str]:
    return {
        match.group(1)
        for match in re.finditer(r"\.([A-Za-z_$][\w$]*)", _strip_js_literals(script))
    }


#: 页面读到的属性名，要么是**真实响应里出现过的字段**，要么是下面这些
#: JS / DOM 内置成员。**故意写死**：它是"这条护栏不误报"的唯一依据。
#: 新增一个内置成员时这里要改 —— 这正是想要的，
#: 否则白名单会被一条条放宽，直到护栏变成永真式。
JS_BUILTINS: frozenset[str] = frozenset(
    {
        # String / Array
        "length", "join", "map", "filter", "find", "forEach", "includes",
        "push", "pop", "replace", "trim", "toFixed", "entries",
        # Math / JSON
        "abs", "round", "max", "min", "stringify",
        # fetch / Promise
        "catch", "json", "status",
        # Error
        "message",
        # KeyboardEvent
        "key",
        # DOM
        "body", "classList", "className", "click", "dataset", "disabled",
        "innerHTML", "insertAdjacentHTML", "onchange", "onclick", "onkeydown",
        "placeholder", "querySelector", "querySelectorAll", "scrollHeight",
        "scrollTop", "tab", "textContent", "toggle", "value",
    }
)


def _api_field_vocabulary(cfg: RuntimeConfig) -> set[str]:
    """真实响应里出现过的**所有**键名（递归收集，含错误体）。

    拿真响应当词表，而不是手写一份"前端需要哪些字段" ——
    手写的那份会和后端一起漂移，而它恰恰是用来发现漂移的。
    """
    payloads = [
        S.build_meta(cfg),
        S.run_chat(
            cfg,
            {
                "scenario": "tutorial",
                "events": [{"kind": "say", "speaker": "player_a", "text": "你好"}],
            },
        ),
        S.run_eval(cfg, {"category": "safety", "limit": 2}),
        S.run_mutant(cfg, {"mutant": "floor_control_disabled", "limit": 3}),
        # 错误体也是真实响应的一种形状（`{"error": "..."}`）
        {"error": "x"},
    ]
    names: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                names.add(str(key))
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for payload in payloads:
        walk(payload)
    return names


def _unknown_property_names(script: str, vocabulary: set[str]) -> set[str]:
    """页面读的字段里，既不是内置成员、也不在真实响应里的那些。"""
    return _property_names(script) - vocabulary - JS_BUILTINS


@pytest.fixture(scope="module")
def api_vocabulary(cfg: RuntimeConfig) -> set[str]:
    return _api_field_vocabulary(cfg)


def test_page_only_reads_fields_the_api_actually_returns(
    api_vocabulary: set[str],
) -> None:
    """页面读的每个字段，必须真的出现在 API 的响应里。

    **这条测试替代的是"用浏览器打开看一眼"。** 一个拼错的字段名
    （`d.sumary`）不会让任何别的测试变红：页面照常加载、照常发出请求，
    只是那一栏永远空白，而空白看起来像"数据没准备好"。

    判据不靠猜：词表是**递归收集真实响应**得到的，不是手写的
    "前端需要哪些字段"。
    """
    unknown = _unknown_property_names(_page_script(), api_vocabulary)
    assert not unknown, (
        f"页面读了这些字段，但 API 响应里没有：{sorted(unknown)}。"
        "要么是拼错了，要么后端改了字段名 —— 两种都会让界面静默空白。"
        "如果它其实是 JS/DOM 内置成员，请加进 JS_BUILTINS 并写清理由。"
    )


def test_the_field_guard_can_actually_fail() -> None:
    """反向测试：上面那条必须真的抓得到拼错的字段名。

    只证明"当前代码是绿的"不够 —— 那可能只是因为判据什么都没在查。

    **这条测试刻意不读真实页面、也不读真实词表。** 用真实页面就等于把
    "页面是干净的"又断言了一遍：页面一旦有拼错，这条会跟着红，
    于是失败信息指向上一条测试，而你看不出判据本身是否还在工作。
    这里只喂合成输入，验的是**判据的区分能力**。
    """
    vocab = {"summary", "cases", "name"}

    # 正例：合法字段不该被误报（否则判据就是"凡读必错"的永真式）
    assert _unknown_property_names("const x = d.summary;", vocab) == set()

    # 反例：拼错的字段必须被抓出来
    assert _unknown_property_names("const x = d.sumary;", vocab) == {"sumary"}
    assert _unknown_property_names("x.lengh", vocab) == {"lengh"}

    # 白名单不能顺手放过：内置成员与拼错同时出现时，只报拼错的那个
    got = _unknown_property_names("const n = list.length; const m = d.sumary;", vocab)
    assert got == {"sumary"}

    # 字符串字面量里的"字段名"不算读取，否则词表会被噪声撑大，
    # 白名单就不得不一条条放宽，直到判据失去意义
    got = _property_names('x.setAttribute("data-x", "d.bogus");')
    assert "bogus" not in got, "字符串字面量被当成了字段读取"


def test_the_field_guard_actually_parses_the_page(api_vocabulary: set[str]) -> None:
    """判据必须真的从页面里读出了东西。

    这是上一条测试自身的"空转检查"：如果 `_property_names` 哪天因为
    正则失效而返回空集，那么"未知字段为空"会**静默通过**，
    而判据已经什么都不查了 —— 这正是这个项目里反复出现的那类
    "永远是绿的断言"。

    不用"至少 N 个"这种阈值，而是点名几个页面**一定**会读的字段：
    阈值会在某次合法重构时被顺手调小，点名不会。
    """
    props = _property_names(_page_script())

    # 点名：页面渲染评测结果时必然要读这些（`/api/eval` 的响应字段）
    for expected in ("summary", "cases", "by_category"):
        assert expected in props, f"判据没能从页面里解析出 {expected!r}"
        assert expected in api_vocabulary, f"{expected!r} 不在真实响应词表里"

    # 词表本身也得是活的：真响应里字段很多，不该只剩三五个
    assert len(api_vocabulary) > 20, "API 词表太小，判据可能已经失效"


def test_the_builtin_whitelist_never_shadows_a_real_field(
    api_vocabulary: set[str],
) -> None:
    """白名单里不能出现任何**真实存在的** API 字段。

    这条堵的是"遇到红灯就改白名单"这条后门。它挡不住全部
    （把一个拼错的字段名加进白名单仍能溜过去），但能挡住最容易发生的一种：
    某个字段确实存在、只是页面把它写错了，于是顺手加进白名单 ——
    那等于把断言改成永真式，还留下了"这个字段是内置成员"的假注释。

    剩下的一半靠人工：白名单每次变长都应该是**显式**的、有理由的改动，
    在 code review 里看得见。所以这里不设"白名单最多 N 条"这种阈值 ——
    阈值只会在某次合法新增时被顺手调大，然后失去意义。
    """
    shadowed = JS_BUILTINS & api_vocabulary
    assert not shadowed, (
        f"这些名字既是真实 API 字段、又在内置白名单里：{sorted(shadowed)}。"
        "从白名单里删掉它们 —— 页面读这些字段时应该拿真实响应来核对。"
    )


def test_page_script_is_syntactically_valid() -> None:
    """页面的 JS 必须能过 `node --check`。

    一个语法错误会让**整个页面**死掉（连 `boot()` 都跑不到），
    而 `PAGE` 只是 Python 里的一段字符串 —— 没有任何东西会告诉你它坏了。
    没有 node 就跳过（和真实桥那两条测试同一个处理方式）。
    """
    node = os.environ.get("NPC_AGENT_NODE") or shutil.which("node")
    if not node:
        pytest.skip("找不到 node（装 Node.js，或用 NPC_AGENT_NODE 指定路径）")

    handle, path = tempfile.mkstemp(suffix=".js", text=True)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(_page_script())
        proc = subprocess.run(
            [node, "--check", path], capture_output=True, text=True, timeout=60
        )
        assert proc.returncode == 0, (
            f"页面 JS 有语法错误：\n{proc.stdout}\n{proc.stderr}"
        )
    finally:
        os.unlink(path)


# --------------------------------------------------------------------------- #
# 6. HTTP 外壳
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def live():
    """起一个真的 HTTP 服务。`port=0` 让系统挑端口 —— 写死会互相抢。"""
    httpd, base = S.start_in_thread(port=0, llm_provider="null")
    yield base
    httpd.shutdown()
    httpd.server_close()


def _get(base: str, path: str) -> tuple[int, bytes, str]:
    with urllib.request.urlopen(base + path, timeout=120) as resp:
        return resp.status, resp.read(), resp.headers.get("Content-Type", "")


def _post(base: str, path: str, body: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _get_status(base: str, path: str) -> tuple[int, dict | str]:
    try:
        status, body, ctype = _get(base, path)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        if "json" in exc.headers.get("Content-Type", ""):
            return exc.code, json.loads(raw)
        return exc.code, raw.decode("utf-8", "replace")
    return status, json.loads(body) if "json" in ctype else body.decode("utf-8", "replace")


def test_handler_has_a_real_config_attribute() -> None:
    r"""`config: RuntimeConfig` 只是**注解**，不是默认值 —— 这个坑真踩过。

    `send_response` 内部会调 `log_message`，而它读 `self.config`。
    注解不会在类上创建属性，于是**每一个请求**都抛 `AttributeError`，
    客户端只看到 `RemoteDisconnected`：服务端把连接关了，什么都没回。
    错误发生在日志函数里，所以真正的病因完全看不见。

    断言的是"类上真的有这个属性"，不是"注解写对了"。
    """
    assert "config" in vars(S._Handler), "必须写成赋值，只写注解等于没有"
    assert isinstance(vars(S._Handler)["config"], RuntimeConfig)


def test_every_request_gets_a_response_not_a_dropped_connection(live: str) -> None:
    """连一个**必然 404** 的路径都必须回一个 404。

    上一条钉的是"类上有属性"，这一条钉的是**可观测的后果**：
    只要有任何一条路径让连接被静默关掉，这里就会炸。
    """
    status, payload = _get_status(live, "/api/definitely-not-a-route")
    assert status == 404
    assert isinstance(payload, dict) and payload.get("error")


def test_console_page_and_meta_are_served(live: str) -> None:
    status, body, ctype = _get(live, "/")
    assert status == 200 and "text/html" in ctype
    assert b"<!doctype html" in body.lower()

    status, body, ctype = _get(live, "/api/meta")
    meta = json.loads(body)
    assert status == 200 and "application/json" in ctype
    assert meta["case_total"] == len(EvalHarness(RuntimeConfig()).load_cases())
    assert meta["mutant_total"] == len(MUTANTS)
    assert len(meta["reports"]) == len(list(DOCS_DIR.glob("*.html")))


def test_http_endpoints_work(live: str) -> None:
    status, payload = _post(
        live, "/api/chat", {"scenario": "tutorial", "events": [{"kind": "idle"}]}
    )
    assert status == 200 and payload["events"]

    status, payload = _post(live, "/api/eval", {"category": "safety", "limit": 4})
    assert status == 200 and payload["summary"]["total"] == 4

    status, payload = _post(live, "/api/mutant", {"mutant": "floor_control_disabled", "limit": 4})
    assert status == 200 and payload["id"] == "floor_control_disabled"


def test_bad_requests_are_400_with_a_readable_message(live: str) -> None:
    """参数错是**用户**的问题，要用 400 说清楚，而不是 500 甩一个栈。"""
    status, payload = _post(live, "/api/mutant", {"mutant": "nope"})
    assert status == 400 and "未知的变异" in payload["error"]

    status, payload = _post(live, "/api/eval", {"limit": "abc"})
    assert status == 400 and "int" in payload["error"]

    # 请求体不是 JSON 对象 —— 也要 400，不能 500
    request = urllib.request.Request(
        live + "/api/chat",
        data=b"[1,2,3]",
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(request, timeout=30)
        raise AssertionError("应该 400")
    except urllib.error.HTTPError as exc:
        assert exc.code == 400
        assert "JSON 对象" in json.loads(exc.read())["error"]


def test_report_route_serves_docs_and_blocks_traversal(live: str) -> None:
    """报告路由只服务 `docs/` 下的 `.html`。

    这个服务会把文件内容回给浏览器，所以拼路径必须校验 ——
    不校验就等于把 `../` 交给调用方。
    """
    name = sorted(p.name for p in DOCS_DIR.glob("*.html"))[0]
    status, body, _ = _get(live, f"/report/{name}")
    assert status == 200 and b"<" in body

    for bad in ("nope.html", "..%2FREADME.md", "%2E%2E%2FREADME.md", "README.md"):
        status, _ = _get_status(live, f"/report/{bad}")
        assert status in (400, 404), f"{bad} 竟然被放过去了：{status}"


def test_free_port_returns_a_usable_port() -> None:
    port = S.free_port()
    assert 1024 < port < 65536
