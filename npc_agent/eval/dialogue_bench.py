"""多 NPC 对话基准 —— 量「NPC 之间能不能聊起来、聊得像不像话」。

## 为什么单独一套用例

六维评测（`eval/harness.py`）量的是**单 NPC 的任务完成**：
目标达成没有、工具调对没有、人设崩没有。它对"两个 NPC 之间"这件事
**结构上盲** —— `turn_taking` 只看"一个 tick 里是不是只有一个人开口"，
两个 NPC 全程不说话它也满分。

所以这套基准补的是那个空档。它**不改**六维用例集（那是回归基线，
数字被文档和测试钉着），而是新增一份可以独立引用的读数：

    peer_turn      同伴点了我，我接不接
    delegation     玩家让我叫同伴，我转不达得到
    grounding      问"你们俩"、问"你是谁"，答不答得上来
    novelty        复不复读、会不会两个 NPC 抢着说同一件事
    memory_flow    印象传不传到另一个 NPC 那里
    turn_taking    一个 tick 一个人、链长有上限、被点名的优先

## 判据只断言"可观测的行为"

每条用例都只看**世界里的发言记录**（谁对谁说了什么），不看内部变量。
理由：内部变量可以为了通过测试而设成某个值，发言记录不能 ——
玩家听到的就是它。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npc_agent.cast import Cast, build_cast  # noqa: E402
from npc_agent.config import RuntimeConfig, load_scenario  # noqa: E402
from npc_agent.llm import NullLLM  # noqa: E402
from npc_agent.modules.repetition import find_repeats  # noqa: E402
from npc_agent.types import Utterance  # noqa: E402

DIMENSIONS = {
    "peer_turn": "同伴接话",
    "delegation": "转达",
    "grounding": "有依据的回答",
    "novelty": "不复读",
    "memory_flow": "印象流动",
    "turn_taking": "发言权",
}

#: 答非所问的典型句子。出现它们就算"没答上来"。
_DODGE = ("说不好", "不太清楚", "不知道", "没听过", "别听我瞎说")


# --------------------------------------------------------------------------- #
# 用例
# --------------------------------------------------------------------------- #
@dataclass
class DialogueCase:
    id: str
    dim: str
    desc: str
    script: list[Optional[str]]
    check: Callable[[Cast], tuple[bool, str]]
    scenario: str = "duet"
    config: dict[str, Any] = field(default_factory=dict)


# -- 小工具 ----------------------------------------------------------------- #
def _npc_lines(cast: Cast) -> list[Utterance]:
    ids = set(cast.npc_ids)
    return [u for u in cast.env.utterances if u.speaker_id in ids]


def _npc_said(cast: Cast) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for u in _npc_lines(cast):
        out.setdefault(u.speaker_id, []).append(u.text)
    return out


def _all_npc_text(cast: Cast) -> str:
    return "".join(u.text for u in _npc_lines(cast))


def _speakers_by_tick(cast: Cast) -> dict[int, list[str]]:
    ids = set(cast.npc_ids)
    out: dict[int, list[str]] = {}
    for u in _npc_lines(cast):
        out.setdefault(u.tick, []).append(u.speaker_id)
    return {t: [s for s in v if s in ids] for t, v in out.items()}


# -- peer_turn --------------------------------------------------------------- #
def _check_peer_reply(cast: Cast) -> tuple[bool, str]:
    replies = [
        u
        for u in _npc_lines(cast)
        if u.speaker_id == "xiaozhou" and u.addressed_to == "ayou"
    ]
    if not replies:
        return False, "小舟没有回应阿柚"
    return True, f"小舟接话 {len(replies)} 次"


def _check_peer_chain_alternates(cast: Cast) -> tuple[bool, str]:
    """两个 NPC 之间要**真的**有来有回，不是各自说话。

    ⚠️ 这条判据的第一版只检查"两个人都开过口、相邻两句不是同一人"，
    于是**关掉 NPC 对聊之后它照样通过** —— 因为小舟会自己找话讲，
    两个人的名字都会出现在记录里。那是**假通过**：它量的是"有没有两个人说话"，
    不是"两个人有没有在对话"。
    现在的判据要求：至少有一句是**对另一个 NPC** 说的，而且对方下一轮接了。
    """
    ids = set(cast.npc_ids)
    cross = [u for u in _npc_lines(cast) if u.addressed_to in ids]
    if not cross:
        return False, "没有任何一句是对另一个 NPC 说的"
    answered = 0
    for line in cross:
        replies = [
            u
            for u in _npc_lines(cast)
            if u.speaker_id == line.addressed_to and u.tick == line.tick + 1
        ]
        if replies:
            answered += 1
    if answered == 0:
        return False, f"{len(cross)} 句对同伴说的话，一句都没被接"
    return True, f"{len(cross)} 句对同伴说，{answered} 句被接住"


def _check_peer_addressed_by_name(cast: Cast) -> tuple[bool, str]:
    """小舟被要求去问阿柚 → 小舟必须**对阿柚**开口。"""
    delegated = [u for u in _npc_lines(cast) if u.speaker_id == "xiaozhou"]
    if not delegated:
        return False, "小舟没有开口"
    target = delegated[0].addressed_to
    return (target == "ayou", f"小舟这句说给 {target}")


# -- delegation -------------------------------------------------------------- #
def _check_delegate_target(cast: Cast) -> tuple[bool, str]:
    ayou = [u for u in _npc_lines(cast) if u.speaker_id == "ayou"]
    if not ayou:
        return False, "阿柚没有开口"
    first = ayou[0]
    if first.addressed_to != "xiaozhou":
        return False, f"阿柚这句说给 {first.addressed_to}，应该转达给小舟"
    if "小舟" not in first.text:
        return False, f"转达时没叫名字：{first.text}"
    return True, first.text


def _check_delegate_not_to_player(cast: Cast) -> tuple[bool, str]:
    ayou = [u for u in _npc_lines(cast) if u.speaker_id == "ayou"]
    if not ayou:
        return False, "阿柚没有开口"
    if ayou[0].addressed_to in ("player_a", "player_b"):
        return False, "把转达说给了玩家（三个人里两个人会听错）"
    return True, "收件人是同伴"


def _check_delegate_gets_reply(cast: Cast) -> tuple[bool, str]:
    """被转达的人要**对转达者**接话，不是"之后开过口"。

    ⚠️ 判据原来只查 `xiaozhou 在 tick>=1 说过话`。关掉 NPC 对聊之后，
    小舟会在冷场时自说自话（`……你好。`，addressed_to=None）——
    整条用例照样通过，于是「关掉 NPC 对聊」这个缺陷只打掉了 `peer_turn`，
    `delegation` 维度看起来毫发无伤。那测的是"后来有人说话"，
    不是"转达被接住了"。
    """
    reply = [
        u
        for u in _npc_lines(cast)
        if u.speaker_id == "xiaozhou"
        and u.tick >= 1
        and u.addressed_to == "ayou"
    ]
    return (bool(reply), "小舟接了转达" if reply else "转达之后没人接")


# -- grounding --------------------------------------------------------------- #
def _check_group_question(cast: Cast) -> tuple[bool, str]:
    said = _all_npc_text(cast)
    if not said:
        return False, "没人回答"
    if any(word in said for word in _DODGE):
        return False, f"回避了问题：{said}"
    return True, said


def _check_self_question(cast: Cast) -> tuple[bool, str]:
    said = _all_npc_text(cast)
    if "阿柚" not in said:
        return False, f"被问到名字却没说名字：{said}"
    return True, said


def _check_recommend(cast: Cast) -> tuple[bool, str]:
    said = _all_npc_text(cast)
    menu = ("拿铁", "柠檬水", "手冲", "咖啡", "豆子")
    if not any(item in said for item in menu):
        return False, f"问推荐却没报出菜单上的东西：{said}"
    return True, said


# -- novelty ----------------------------------------------------------------- #
def _check_no_repeat(cast: Cast) -> tuple[bool, str]:
    lines = [(u.speaker_id, u.text) for u in _npc_lines(cast)]
    report = find_repeats(lines)
    ok = report.rate <= 0.2
    return (ok, f"{len(lines)} 句，复读 {len(report.repeats)} 句（{report.rate:.0%}）")


def _check_no_duplicate_fact(cast: Cast) -> tuple[bool, str]:
    told = [u.text for u in _npc_lines(cast) if "灯塔" in u.text]
    return (len(told) <= 1, f"店史被说了 {len(told)} 次")


def _check_topic_switch(cast: Cast) -> tuple[bool, str]:
    """话题用尽之后要**换话题**，不是把最后一个话题再说一遍。

    判据用"有内容的长句有几条不同的"：`嗯。`、`……还行。` 这类应声不算 ——
    它们是对话的黏合剂，不是内容。只数超过 10 个字的句子，
    要求至少出现 2 条**不同**的。少于 2 条说明 NPC 已经没话找话。
    """
    facts = [u.text for u in _npc_lines(cast) if len(u.text) > 10]
    distinct = len(set(facts))
    return (distinct >= 2, f"有内容的台词 {len(facts)} 条，去重后 {distinct} 条")


# -- memory_flow ------------------------------------------------------------- #
def _check_gossip_propagates(cast: Cast) -> tuple[bool, str]:
    mem = cast.agents["xiaozhou"].memory.store.records
    gossip = [r for r in mem if "gossip" in r.tags]
    if not gossip:
        return False, "小舟没有收到任何传闻"
    return True, gossip[0].content


def _check_gossip_attributed(cast: Cast) -> tuple[bool, str]:
    said = _all_npc_text(cast)
    if "阿柚" not in said:
        return False, f"说起传闻时没提来源：{said}"
    return True, said


def _check_no_gossip_on_smalltalk(cast: Cast) -> tuple[bool, str]:
    facts = cast.gossip.facts() if cast.gossip is not None else []
    return (not facts, f"寒暄产生了 {len(facts)} 条传闻")


# -- turn_taking ------------------------------------------------------------- #
def _check_one_speaker(cast: Cast) -> tuple[bool, str]:
    for tick, speakers in _speakers_by_tick(cast).items():
        if len(speakers) > 1:
            return False, f"tick {tick} 有 {len(speakers)} 个 NPC 同时开口"
    return True, "每个 tick 最多一个 NPC 开口"


def _check_chain_limit(cast: Cast) -> tuple[bool, str]:
    if cast.board.chain > cast.config.max_peer_exchanges:
        return False, f"链长 {cast.board.chain} 超过上限 {cast.config.max_peer_exchanges}"
    return True, f"链长 {cast.board.chain}/{cast.config.max_peer_exchanges}"


def _check_named_npc_first(cast: Cast) -> tuple[bool, str]:
    said = _npc_said(cast)
    if "xiaozhou" not in said:
        return False, "被点名的小舟没有开口"
    return True, "被点名的先开口"


# --------------------------------------------------------------------------- #
# 用例表
# --------------------------------------------------------------------------- #
def cases() -> list[DialogueCase]:
    return [
        # --- 同伴接话 ---
        DialogueCase(
            "peer_reply_after_delegation",
            "peer_turn",
            "玩家让阿柚叫小舟 → 小舟下一轮接话",
            ["阿柚，你让小舟弹一首呗？", None, None],
            _check_peer_reply,
        ),
        DialogueCase(
            "peer_chain_alternates",
            "peer_turn",
            "两个 NPC 之间有来有回，而不是一个人自说自话",
            ["阿柚，你让小舟弹一首呗？", None, None, None, None],
            _check_peer_chain_alternates,
        ),
        DialogueCase(
            "peer_addressed_by_name",
            "peer_turn",
            "小舟被要求去问阿柚 → 小舟对阿柚开口",
            ["小舟，你问问阿柚今天忙不忙。", None, None],
            _check_peer_addressed_by_name,
        ),
        # --- 转达 ---
        DialogueCase(
            "delegate_goes_to_peer",
            "delegation",
            "转达的收件人是同伴，而且叫得出名字",
            ["阿柚，你让小舟弹一首呗？", None],
            _check_delegate_target,
        ),
        DialogueCase(
            "delegate_not_to_player",
            "delegation",
            "转达不能说给玩家（三个人里两个人会听错）",
            ["阿柚，你让小舟弹一首呗？", None],
            _check_delegate_not_to_player,
        ),
        DialogueCase(
            "delegate_gets_reply",
            "delegation",
            "转达之后，被叫的人真的接了",
            ["阿柚，你让小舟弹一首呗？", None, None],
            _check_delegate_gets_reply,
        ),
        # --- 有依据的回答 ---
        DialogueCase(
            "group_question_answered",
            "grounding",
            "「你们俩平时都聊些什么？」要答得上来",
            ["你们俩平时都聊些什么？"],
            _check_group_question,
        ),
        DialogueCase(
            "self_question_answered",
            "grounding",
            "被问到名字要说得出自己的名字",
            ["你叫什么名字？"],
            _check_self_question,
        ),
        DialogueCase(
            "recommend_answered",
            "grounding",
            "问推荐要报出菜单上真有的东西",
            ["有什么推荐？"],
            _check_recommend,
        ),
        # --- 不复读 ---
        DialogueCase(
            "no_repeat_in_12_ticks",
            "novelty",
            "12 轮以内不出现逐字复读",
            ["今天这里挺热闹的。", "露台那边好像有风。"]
            + [None] * 10,
            _check_no_repeat,
        ),
        DialogueCase(
            "no_duplicate_fact",
            "novelty",
            "同一件事不能被两个 NPC 各讲一遍",
            [None] * 14,
            _check_no_duplicate_fact,
        ),
        DialogueCase(
            "topic_switch_when_exhausted",
            "novelty",
            "话题用尽之后换话题，而不是原地打转",
            [None] * 16,
            _check_topic_switch,
        ),
        # --- 印象流动 ---
        DialogueCase(
            "gossip_propagates",
            "memory_flow",
            "玩家说了偏好 → 另一个 NPC 记住了这条判断",
            ["我喜欢偏酸的豆子。", None, None, None, None],
            _check_gossip_propagates,
        ),
        DialogueCase(
            "gossip_attributed",
            "memory_flow",
            "说起听来的事时要提来源",
            [
                "我喜欢偏酸的豆子。",
                None,
                None,
                None,
                None,
                "小舟，你还记得阿澈喜欢什么吗？",
            ],
            _check_gossip_attributed,
        ),
        DialogueCase(
            "no_gossip_on_smalltalk",
            "memory_flow",
            "寒暄不该在世界上留下传闻",
            ["今天这里挺热闹的。", None, None, None],
            _check_no_gossip_on_smalltalk,
        ),
        # --- 发言权 ---
        DialogueCase(
            "one_speaker_per_tick",
            "turn_taking",
            "一个 tick 只能有一个 NPC 开口",
            ["今天这里挺热闹的。", "阿柚，你让小舟弹一首呗？", "小舟，你那边在弹什么？", None, None],
            _check_one_speaker,
        ),
        DialogueCase(
            "chain_within_limit",
            "turn_taking",
            "NPC 连聊不超过上限",
            ["阿柚，你让小舟弹一首呗？"] + [None] * 8,
            _check_chain_limit,
            config={"max_peer_exchanges": 2},
        ),
        DialogueCase(
            "named_npc_speaks_first",
            "turn_taking",
            "被点名的 NPC 优先开口",
            ["小舟，你那边在弹什么？"],
            _check_named_npc_first,
        ),
    ]


# --------------------------------------------------------------------------- #
# 跑批
# --------------------------------------------------------------------------- #
@dataclass
class CaseResult:
    id: str
    dim: str
    desc: str
    passed: bool
    note: str
    transcript: list[str] = field(default_factory=list)


def run_case(case: DialogueCase) -> CaseResult:
    config = RuntimeConfig(**case.config)
    cast = build_cast(load_scenario(case.scenario), NullLLM(), config)
    for line in case.script:
        utterance = (
            cast.env.record_player_utterance("player_a", line) if line is not None else None
        )
        cast.step(utterance)
    try:
        passed, note = case.check(cast)
    except Exception as exc:  # 用例自己崩了也要算失败，不能吞掉
        passed, note = False, f"用例异常：{type(exc).__name__}: {exc}"
    transcript = [f"{u.speaker_name}{'→' + u.addressed_to if u.addressed_to else ''}: {u.text}" for u in cast.env.utterances]
    return CaseResult(case.id, case.dim, case.desc, passed, note, transcript)


@dataclass
class BenchReport:
    results: list[CaseResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    def by_dimension(self) -> dict[str, tuple[int, int]]:
        out: dict[str, list[int]] = {}
        for r in self.results:
            bucket = out.setdefault(r.dim, [0, 0])
            bucket[1] += 1
            if r.passed:
                bucket[0] += 1
        return {k: (v[0], v[1]) for k, v in out.items()}


def run_all(config_overrides: Optional[dict[str, Any]] = None) -> BenchReport:
    """跑全部用例。`config_overrides` 用来做**敏感性测试**（见 `sensitivity`）。"""
    results = []
    for case in cases():
        if config_overrides:
            merged = dict(case.config)
            merged.update(config_overrides)
            case = DialogueCase(
                case.id, case.dim, case.desc, case.script, case.check, case.scenario, merged
            )
        results.append(run_case(case))
    return BenchReport(results)


# --------------------------------------------------------------------------- #
# 敏感性：满分是不是"判据从不报警"
# --------------------------------------------------------------------------- #
#: 两个**故意打断**的开关。它们各自应该打掉哪几个维度，是**事先写死**的 ——
#: 事后看哪个掉了就记哪个，等于让判据自己解释自己。
DEFECTS: dict[str, dict[str, Any]] = {
    "关掉 NPC 对聊": {"max_peer_exchanges": 0},
    "关掉传闻": {"gossip_enabled": False},
}

EXPECTED_BREAKAGE: dict[str, tuple[str, ...]] = {
    "关掉 NPC 对聊": ("peer_turn", "delegation"),
    "关掉传闻": ("memory_flow",),
}

#: 注入缺陷之后**不该**掉分的维度，以及为什么。这条比"掉了几个"更有信息量：
#: 它说明某个维度测的到底是什么。
UNCHANGED_EXPLANATION = {
    "turn_taking": (
        "关掉 NPC 对聊之后它仍然是 3/3 —— 因为这一维测的是"
        "「别抢话」（一个 tick 一个人、链长有上限），不是「会不会聊」。"
        "把它当成「对话能力」的证据是误读。"
    ),
}


def sensitivity(baseline: BenchReport) -> list[dict[str, Any]]:
    """逐个注入缺陷，看**该掉分的维度是不是真的掉了**。

    一个从不失败的评测，和一个没有评测，在报告上长得一模一样。
    所以这里配了一层反向验证：满分必须能被证伪。
    """
    out = []
    for name, overrides in DEFECTS.items():
        report = run_all(overrides)
        broken = [
            dim
            for dim in DIMENSIONS
            if report.by_dimension().get(dim, (0, 0))[0]
            < baseline.by_dimension().get(dim, (0, 0))[0]
        ]
        expected = EXPECTED_BREAKAGE[name]
        missed = [d for d in expected if d not in broken]
        out.append(
            {
                "defect": name,
                "passed": report.passed,
                "total": report.total,
                "broken": broken,
                "expected": list(expected),
                "detected": not missed,
                "missed": missed,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
_CSS = """
:root { --ink:#1f2430; --muted:#6b7280; --line:#e5e7eb; --bg:#fff;
        --soft:#f7f8fa; --accent:#2f6fd0; --good:#0f9d58; --bad:#d93025; }
* { box-sizing:border-box; }
body { margin:0; padding:40px 28px 64px; background:var(--bg); color:var(--ink);
  font:15px/1.7 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
       "Hiragino Sans GB", "Microsoft YaHei", sans-serif; }
.wrap { max-width:1100px; margin:0 auto; }
h1 { font-size:26px; margin:0 0 6px; }
h2 { font-size:18px; margin:36px 0 12px; padding-bottom:8px; border-bottom:2px solid var(--line); }
.sub { color:var(--muted); font-size:14px; margin-bottom:20px; }
.headline { background:var(--soft); border:1px solid var(--line); border-left:4px solid var(--accent);
  border-radius:8px; padding:14px 18px; font-size:14.5px; }
.warn { background:#fffbf0; border:1px solid #f0dfb8; border-left:4px solid #e8a33d;
  border-radius:8px; padding:14px 18px; font-size:13.5px; margin:16px 0; }
table { width:100%; border-collapse:collapse; font-size:13.5px; }
th,td { padding:9px 10px; border-bottom:1px solid var(--line); text-align:left; }
th { background:var(--soft); font-weight:600; font-size:12.5px; color:#374151; }
td.num { text-align:right; font-family:ui-monospace,Menlo,Consolas,monospace; white-space:nowrap; }
.pass { color:var(--good); font-weight:600; }
.fail { color:var(--bad); font-weight:600; }
code { background:#eef0f4; padding:1px 5px; border-radius:4px; font-size:12.5px; }
.foot { margin-top:32px; color:var(--muted); font-size:12.5px; }
details { margin:6px 0 0; }
summary { cursor:pointer; color:var(--muted); font-size:12.5px; }
pre { background:var(--soft); border:1px solid var(--line); border-radius:6px;
  padding:10px 12px; font-size:12.5px; overflow:auto; margin:8px 0 0; }
"""


def render_html(report: BenchReport, defects: Optional[list[dict[str, Any]]] = None) -> str:
    dims = report.by_dimension()
    rows = []
    for name, label in DIMENSIONS.items():
        got, total = dims.get(name, (0, 0))
        cls = "pass" if got == total else "fail"
        rows.append(
            f"<tr><td>{label}</td><td class='num {cls}'>{got}/{total}</td>"
            f"<td class='num'>{got / total:.0%}</td></tr>"
            if total
            else f"<tr><td>{label}</td><td class='num'>—</td><td class='num'>—</td></tr>"
        )
    detail = []
    for r in report.results:
        mark = "pass" if r.passed else "fail"
        label = "PASS" if r.passed else "FAIL"
        transcript = "".join(f"{line}\n" for line in r.transcript)
        detail.append(
            f"<tr><td><code>{r.id}</code></td><td>{DIMENSIONS.get(r.dim, r.dim)}</td>"
            f"<td>{r.desc}</td><td class='{mark}'>{label}</td><td>{r.note}</td></tr>"
            f"<tr><td colspan='5'><details><summary>对话记录</summary>"
            f"<pre>{transcript}</pre></details></td></tr>"
        )

    defect_rows = ""
    defect_note = ""
    if defects:
        rows = []
        for d in defects:
            cls = "pass" if d["detected"] else "fail"
            verdict = "抓到了" if d["detected"] else "没抓到"
            rows.append(
                f"<tr><td>{d['defect']}</td><td class='num'>{d['passed']}/{d['total']}</td>"
                f"<td>{'、'.join(DIMENSIONS.get(x, x) for x in d['broken']) or '—'}</td>"
                f"<td class='{cls}'>{verdict}</td></tr>"
            )
        defect_rows = (
            "<h2>敏感性：满分能不能被证伪</h2>"
            "<div class='sub'>往框架里注入一个**故意的缺陷**，该掉分的维度必须掉。"
            "一个从不失败的评测，和一个没有评测，在报告上长得一模一样。</div>"
            "<table><thead><tr><th>注入的缺陷</th><th>通过</th><th>掉分的维度</th>"
            "<th>判定</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )
        defect_note = (
            "<div class='warn'><strong>一个反直觉的读数。</strong>"
            + UNCHANGED_EXPLANATION["turn_taking"]
            + "<br>这条是**事先写死**的预期，不是事后看到哪个掉了就记哪个 —— "
            "否则判据会自己解释自己。</div>"
        )
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>多 NPC 对话基准</title><style>{_CSS}</style></head><body><div class="wrap">
<h1>多 NPC 对话基准</h1>
<div class="sub">量的是「NPC 之间能不能聊起来、聊得像不像话」——六维评测结构上看不见的那一块。</div>
<div class="headline">共 <strong>{report.total}</strong> 条用例、{len(DIMENSIONS)} 个维度，通过 <strong>{report.passed}/{report.total}</strong>（{report.passed / report.total:.0%}）。全部离线运行，<strong>0 次模型调用</strong>，clone 下来就能复现。</div>
<div class="warn"><strong>这份读数不能说明什么。</strong>它衡量的是一条**确定的对话链路**是否接通：话有没有说给对的人、接不接得住、会不会复读、印象传不传得到。
它<strong>不衡量</strong>对话是否有趣、是否像真人 —— 那要真人玩家来判，不是工程能替代的（见 README 的「边界」一节）。
离线路径的词汇量是有限的，长对话的复读率有下界；接上模型之后同一个位置的表现会不同，那需要另外跑批。</div>
<h2>按维度</h2>
<table><thead><tr><th>维度</th><th>通过</th><th>通过率</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
{defect_rows}
{defect_note}
<h2>逐条</h2>
<table><thead><tr><th>用例</th><th>维度</th><th>断言</th><th>结果</th><th>读数</th></tr></thead>
<tbody>{''.join(detail)}</tbody></table>
<div class="foot">重生成：<code>python -m npc_agent.eval.dialogue_bench --html docs/npc_dialogue.html</code><br>
用例与判据：<code>npc_agent/eval/dialogue_bench.py</code>；护栏：<code>tests/test_dialogue_bench.py</code>。</div>
</div></body></html>"""


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="多 NPC 对话基准")
    parser.add_argument("--html", default="")
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)

    report = run_all()
    defects = sensitivity(report)
    for r in report.results:
        print(f"[{'PASS' if r.passed else 'FAIL'}] {r.dim:12s} {r.id:32s} {r.note}")
    print()
    print(f"通过 {report.passed}/{report.total}（{report.passed / report.total:.0%}）")
    for name, (got, total) in report.by_dimension().items():
        print(f"  {DIMENSIONS.get(name, name):8s} {got}/{total}")
    print()
    for d in defects:
        print(
            f"[{'抓到了' if d['detected'] else '没抓到'}] {d['defect']} → "
            f"{d['passed']}/{d['total']}，掉分维度：{d['broken']}"
        )

    if args.html:
        path = ROOT / args.html
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_html(report, defects), encoding="utf-8")
        print(f"\n写入 {path}")
    if args.json:
        path = ROOT / args.json
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "total": report.total,
            "passed": report.passed,
            "dimensions": {k: list(v) for k, v in report.by_dimension().items()},
            "cases": [
                {"id": r.id, "dim": r.dim, "passed": r.passed, "note": r.note}
                for r in report.results
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"写入 {path}")
    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
