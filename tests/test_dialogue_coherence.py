"""双 NPC 场景的「答非所问」回归 —— 钉住 2026-09-22 控制台实测到的那段对话。

## 用户报的原话（duet，默认路径，逐字）

    第 1 轮 玩家 阿澈：今天这里好热闹
      阿柚：第一次来吧？我请你一杯，想喝什么？   ← 计划里的开场白，与这句话无关
    第 2 轮 玩家 阿澈：有什么推荐
      小舟：你上次说过第一次来吧。               ← 玩家从没说过这句
    第 3 轮 玩家 阿澈：我是第一次来
      阿柚：阿澈说的我记下了。
    第 4 轮 玩家 阿澈：那我来杯卡布奇诺吧
      小舟：我记得。你是第一次来。               ← 单没接，也没说做不了

四个症状，四个**独立的**根因（都离线复现过）：

| 症状 | 根因 |
|---|---|
| 开场白顶掉回应 | `step()` 的回应闸门只认「提问 / 被点名」。玩家说**陈述句**时它不成立 ⇒ 这一轮说什么由 NPC 自己的计划决定 |
| 假记忆 | `_recallable()` 不看记忆**是谁的**。`MemoryStore.observe()` 把"听到的一切"记成同一种形状，于是**另一个 NPC** 的话被当成玩家说的 |
| 单没人接 | `planner.is_request` 认不出饮品名 ⇒「卡布奇诺」计划为空；而闸门又因为"这是下单"而不回应 ⇒ 掉进陈述句分支 |
| 问句被回忆顶掉 | `_respond()` 里 `recall` 排在 `answer_question` **前面** |

顺带钉住一条更早的盲区：`recall` 用的 `{memory_hint}` 是**检索**的结果，
所以"另一个 NPC 的话"能被回引成"玩家说过的话"——`小舟` 那句
「你上次说过第一次来吧」里的「第一次来吧」正是 `阿柚` 的开场白。

## 这个文件只测玩家看得见的东西：转写

所有期望值都**从数据推导**（玩家原话、人设配置），**不手抄任何一句台词**。
手抄的期望值会跟着实现一起腐烂，而且腐烂时是**静默**的 ——
它变成一条永远绿的断言，比没有断言更糟。

## 哪几条是「回归钉子」，哪几条是「不许退步」

把根因一个个放回去实测过（脚本：`scripts/verify_dialogue_coherence.py`），
结论写在这里，免得读的人高估覆盖：

| 断言 | 性质 | 放回根因时会红吗 |
|---|---|---|
| `test_a_players_statement_is_never_answered_with_a_canned_opening` | **回归钉子**（症状 1） | 会。闸门收回成 `_player_is_asking_me` ⇒ 第 1 轮变成 `第一次来吧？我请你一杯，想喝什么？` |
| `test_an_order_off_the_menu_is_declined_not_ignored` | **回归钉子**（症状 4） | 会。`_unfilled_order` 恒假 ⇒ 第 4 轮变成 `你上次说过你是第一次来。` |
| `test_every_quote_is_something_the_player_actually_said` | 不许退步 | 这两次变异都没红 —— 引文机制本身没坏 |
| `test_every_round_gets_a_reply` | 不许退步 | 没红（修复前也满足：开场白顶掉回应时，那轮**确实**有人说话） |
| `test_the_memory_chain_spans_rounds` | 不许退步 | 没红（修复前的记忆链本来就不短） |

**症状 2（把另一个 NPC 的话当成玩家说的）不在这个文件里** ——
它的根因在 `_recallable()` 的判据上，所以钉在单元层：
`tests/test_memory_hint.py::test_recallable_still_requires_every_condition`
（peer-NPC 的 `entities` / 别的玩家的 `entities` 都不可回引）。
本文件的引文断言抓不到它：`{memory_hint}` 不带 `「」`，
所以「你上次说过第一次来吧」这种假回忆在转写上**没有引文痕迹**。
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from npc_agent.cast import build_cast
from npc_agent.config import RuntimeConfig, load_scenario
from npc_agent.llm import build_llm

#: 用户实测时说的话，逐字。
REPORTED_ROUNDS: tuple[tuple[str, str], ...] = (
    ("player_a", "今天这里好热闹"),
    ("player_a", "有什么推荐"),
    ("player_a", "我是第一次来"),
    ("player_a", "那我来杯卡布奇诺吧"),
)

#: 台词里引用玩家原话的写法。
QUOTE_RE = re.compile(r"「([^」]*)」")

#: 模板里的槽位。
SLOT_RE = re.compile(r"\{[^}]*\}")


# --------------------------------------------------------------------------- #
# 判据（都是纯函数 —— 所以下面能给每一条写反向测试）
# --------------------------------------------------------------------------- #
def fixed_fragments(template: str, *, min_chars: int = 2) -> list[str]:
    """切出模板里**不含槽位**的固定片段。

    `{said_hint}……这儿没有。问阿柚。` → `['……这儿没有。问阿柚。']`

    这样断言就不依赖槽位里到底填了什么 —— 既不用手抄台词，
    也不会因为换一个 `{said_hint}` 就误报。
    """
    return [p for p in SLOT_RE.split(template) if len(p.strip()) >= min_chars]


def templates_of(cast, intent: str) -> list[str]:
    """把剧组里**所有**人设的这个意图模板收集起来。

    用"全体人设"而不是"某一个人"：这段话里谁开口是调度决定的，
    把断言绑在某一个人身上，换一次发言顺序就会误报。
    """
    out: list[str] = []
    for agent in cast.agents.values():
        raw = agent.persona.templates.get(intent)
        if isinstance(raw, str):
            out.append(raw)
        elif isinstance(raw, (list, tuple)):
            out.extend(str(x) for x in raw)
    return out


def fabricated_quotes(speeches: list[str], player_lines: list[str]) -> list[str]:
    """找出**玩家没说过的**引文。

    `「…」` 是回引玩家原话的写法。引文不在玩家原话里 ⇒ 这句"原话"是编的。
    """
    haystack = "\n".join(player_lines)
    return [q for say in speeches for q in QUOTE_RE.findall(say) if q not in haystack]


def canned_opening_hits(speeches: list[str], openings: list[str]) -> list[str]:
    """找出用「计划里的开场白」回答玩家的话。"""
    return [say for say in speeches if any(op in say for op in openings)]


def decline_hits(speeches: list[str], templates: list[str]) -> list[str]:
    """找出明说「做不了」的话。"""
    out: list[str] = []
    for say in speeches:
        for tmpl in templates:
            if any(frag in say for frag in fixed_fragments(tmpl)):
                out.append(say)
                break
    return out


# --------------------------------------------------------------------------- #
# 重放
# --------------------------------------------------------------------------- #
class Round:
    """一轮：玩家说了什么、每个 NPC 说了什么、各自用了哪些记忆。"""

    def __init__(self, text: str) -> None:
        self.text = text
        self.speeches: list[tuple[str, str]] = []
        self.memories: dict[str, list[str]] = {}

    @property
    def lines(self) -> list[str]:
        return [say for _, say in self.speeches]

    @property
    def used(self) -> set[str]:
        return {mid for ids in self.memories.values() for mid in ids}


def replay():
    """离线重放这段对话（镜像 `studio.run_chat`，但**不接模型**）。

    `use_llm_planner` / `use_llm_speech` 显式关掉：这样"走的是启发式"是
    **写下来的**，不是"端点恰好不可用"碰出来的 —— 否则这条测试会在
    端点恢复的那天悄悄换一条代码路径。
    """
    cfg = replace(RuntimeConfig(), use_llm_planner=False, use_llm_speech=False)
    cast = build_cast(load_scenario("duet"), build_llm("null"), cfg)

    rounds: list[Round] = []
    for speaker, text in REPORTED_ROUNDS:
        current = Round(text)
        utterance = cast.env.record_player_utterance(speaker, text)
        for turn in cast.step(utterance):
            if turn.say:
                current.speeches.append((turn.actor_id, turn.say))
            current.memories[turn.actor_id] = list(turn.used_memories)
        rounds.append(current)
    return cast, rounds


@pytest.fixture(scope="module")
def played():
    return replay()


# --------------------------------------------------------------------------- #
# 1. 每一轮都要有人接话
# --------------------------------------------------------------------------- #
def test_every_round_gets_a_reply(played) -> None:
    """玩家说了一句话，不能没人理。

    「没人理」有两种：NPC 真的沉默，或者**有人说了话，但说的不是这件事**。
    后者由下面几条分头钉 —— 这一条只管第一种。
    """
    _, rounds = played
    silent = [r.text for r in rounds if not r.speeches]
    assert not silent, f"这些轮次玩家说了话，却一个 NPC 都没回应：{silent}"


def test_a_players_statement_is_never_answered_with_a_canned_opening(played) -> None:
    """陈述句不能被计划里的开场白顶掉。

    期望值从人设配置里读（`greet_new`），不手抄。修复前实测：

        玩家：今天这里好热闹
        阿柚：第一次来吧？我请你一杯，想喝什么？   ← `greet_new` 原文
    """
    cast, rounds = played
    openings = templates_of(cast, "greet_new")
    assert openings, "读不到任何 `greet_new` —— 人设结构变了，这条护栏正在空转"

    offenders = [
        (r.text, hit)
        for r in rounds
        for hit in canned_opening_hits(r.lines, openings)
    ]
    assert not offenders, (
        f"玩家的陈述句被计划里的开场白回答了：{offenders}。"
        "闸门必须让「回应玩家这句话」优先于「推进自己的计划」。"
    )


# --------------------------------------------------------------------------- #
# 2. 引文必须是玩家真的说过的话
# --------------------------------------------------------------------------- #
def test_every_quote_is_something_the_player_actually_said(played) -> None:
    """台词里引用的原话必须真的出自玩家之口。

    修复前实测：`小舟` 说「你上次说过第一次来吧」—— 那句话是**阿柚**的开场白。
    根因是 `MemoryStore.observe()` 把"听到的一切"记成同一种形状，
    `_recallable()` 又不看这条记忆**是谁的**，于是另一个 NPC 的话
    被回引成"玩家说过的话"。
    """
    _, rounds = played
    lines = [say for r in rounds for say in r.lines]
    quotes = [q for say in lines for q in QUOTE_RE.findall(say)]

    # 先钉住「回引机制真的在跑」—— 否则下面的断言在空转。
    assert quotes, "整段对话一句玩家原话都没引用？回引机制是不是坏了"

    fabricated = fabricated_quotes(lines, [text for _, text in REPORTED_ROUNDS])
    assert not fabricated, (
        f"这些引文玩家根本没说过：{fabricated}。"
        "记忆必须带上「这是谁说的」，另一个 NPC 的话不能当成玩家说的。"
    )


# --------------------------------------------------------------------------- #
# 3. 菜单上没有的那一杯要说「做不了」
# --------------------------------------------------------------------------- #
def test_an_order_off_the_menu_is_declined_not_ignored(played) -> None:
    """玩家点了菜单上没有的一杯：必须**明说做不了**。

    修复前这一句掉进了回忆分支，答「我记得。你是第一次来。」
    —— 单没接、也没说做不了，玩家只会以为 NPC 没听懂。
    """
    cast, rounds = played
    last = rounds[-1]
    assert last.speeches, "最后一轮没人说话"

    declines = templates_of(cast, "unavailable_order")
    assert declines, "读不到任何 `unavailable_order` —— 人设结构变了"
    # ⚠️ 前提写成断言，不写成注释：整个是槽位的模板会让判据**恒不命中**，
    # 那样这条测试会永远绿，而理由是错的。
    assert all(fixed_fragments(t) for t in declines), (
        f"有 `unavailable_order` 模板整个是槽位，判据对它恒不命中：{declines}"
    )

    hits = decline_hits(last.lines, declines)
    assert hits, (
        f"最后一轮点了「卡布奇诺」，没人说做不了：{last.lines}。"
        f"（`unavailable_order` 的固定片段：{declines}）"
    )


# --------------------------------------------------------------------------- #
# 4. 记忆链要跨轮
# --------------------------------------------------------------------------- #
def test_the_memory_chain_spans_rounds(played) -> None:
    """「记忆链尽可能长」—— 判据是**跨轮**，不是条数。

    一轮里列 6 条本轮刚写的记忆也叫 6 条，但那不叫链。
    所以这里问的是：这一轮用到的记忆里，有没有一条是**更早的轮次**留下的。
    """
    _, rounds = played
    assert rounds[0].used, "第一轮一条记忆都没用上"

    earlier: set[str] = set()
    for index, r in enumerate(rounds):
        assert r.used, f"第 {index + 1} 轮一条记忆都没用上：{r.text}"
        if index:
            shared = r.used & earlier
            assert shared, (
                f"第 {index + 1} 轮用到的记忆里没有一条来自更早的轮次"
                f"（本轮 {sorted(r.used)}，之前 {sorted(earlier)}）—— 记忆链断了"
            )
        earlier |= r.used

    assert len(earlier) >= len(rounds), (
        f"整段对话只留下 {len(earlier)} 条不同记忆（共 {len(rounds)} 轮）"
    )


# --------------------------------------------------------------------------- #
# 上面四条判据必须能被验证会红
#
# 不手抄"要改坏的那个值"：反向用例的输入是**构造的**，
# 期望值是**从被判据自身读出来的**（比如把真实台词喂进判据当反面样本）。
# --------------------------------------------------------------------------- #
def test_the_quote_judgement_can_actually_fail() -> None:
    """引文判据必须抓得住编造的原话，也放得过真的原话。"""
    player = ["今天这里好热闹"]

    assert fabricated_quotes(["阿澈说的「今天这里好热闹」，我记下了。"], player) == []
    assert fabricated_quotes(["你上次说过「第一次来吧」。"], player) == ["第一次来吧"]
    # 没有引文就不该被误报
    assert fabricated_quotes(["……苹果派。"], player) == []


def test_the_canned_opening_judgement_can_actually_fail() -> None:
    """开场白判据必须抓得住 `greet_new`，放得过正常回应。"""
    openings = ["第一次来吧？我请你一杯，想喝什么？"]

    assert canned_opening_hits(["第一次来吧？我请你一杯，想喝什么？"], openings)
    assert canned_opening_hits(["好。第一次来吧？我请你一杯，想喝什么？"], openings)
    assert canned_opening_hits(["阿澈说的「今天这里好热闹」，我记下了。"], openings) == []


def test_the_decline_judgement_can_actually_fail() -> None:
    """「做不了」判据必须忽略槽位 —— 槽位里填什么都能认出来。"""
    templates = ["{said_hint}……这儿没有。问阿柚。", "这个真没有——手冲要不要试试？"]

    assert decline_hits(["「那我来杯卡布奇诺吧」……这儿没有。问阿柚。"], templates)
    assert decline_hits(["这个真没有——手冲要不要试试？"], templates)
    # 回忆模板不能被误认成「做不了」
    assert decline_hits(["我记得。你是第一次来。"], templates) == []


def test_an_npc_does_not_butt_in_when_the_player_addresses_the_other_one() -> None:
    """玩家点名另一个 NPC 时，我不插嘴。

    ⚠️ 这条是给**刚放宽的回应闸门**补的。闸门从「提问 / 被点名」
    放宽成「提问 / 被点名 / 对我陈述 / 点了一杯没有的」——
    放宽的东西必须有一条护栏说清「放宽到哪儿为止」。
    否则下一个改它的人只会看到"闸门越宽越好"，然后把抢话放进来。
    """
    cfg = replace(RuntimeConfig(), use_llm_planner=False, use_llm_speech=False)
    cast = build_cast(load_scenario("duet"), build_llm("null"), cfg)

    other = "xiaozhou"
    mentioned = cast.agents[other].persona.name
    utterance = cast.env.record_player_utterance(
        "player_a", f"{mentioned}，你唱得真好听"
    )
    spoke = {turn.actor_id for turn in cast.step(utterance) if turn.say}

    assert spoke == {other}, (
        f"玩家点名了「{mentioned}」，开口的却是 {sorted(spoke)} —— 抢话了。"
        "「这句话是冲我来的」那个判据必须拦住这个。"
    )


def test_the_fixed_fragment_splitter_ignores_slots() -> None:
    """切片段要真的把槽位切掉，否则断言会被槽位内容带偏。"""
    assert fixed_fragments("{said_hint}我们这儿做不了，换一杯？") == [
        "我们这儿做不了，换一杯？"
    ]
    # 全是槽位 ⇒ 一个片段都没有（这条判据在这种模板上应当"空转"，
    # 所以调用方必须另有办法 —— 见下面这条）
    assert fixed_fragments("{a}{b}") == []
