"""世界能力基准 —— 量"这个世界像不像活的"。

## 和另外两套评测的分工

    六维回归（235 条）      框架逻辑有没有坏（任务 / 工具 / 记忆 / 人设 / 安全 / 发言权）
    对话基准（18 条）        两个 NPC 之间能不能聊起来
    世界能力基准（17 条）    世界本身有没有"设定、关系、动机、注意力"，以及三个人同场时还成不成立

第三套补的是"活世界"里最像产品、也最难自动验证的那部分：
NPC 会不会说错设定、记不记得你是回头客、有没有自己的动机、
欠你的回答会不会补 —— 以及**第三个人在场时，前四条是不是还成立**。

## 判据仍然只断言可观测的东西

发言记录、世界事件、被拒绝的动作。不看内部变量 ——
内部变量可以为了通过测试而设成某个值，玩家看到的不能。
唯一的例外是注意力排序（`rank`），因为"同时有两个人想说话"这件事
在当前的单句输入模型里没有可观测的出口，只能测模块契约。
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
from npc_agent.modules.attention import AttentionBoard  # noqa: E402
from npc_agent.modules.lore import LoreBook  # noqa: E402
from npc_agent.modules.motives import MotiveBoard  # noqa: E402
from npc_agent.modules.state import PlayerModel  # noqa: E402
from npc_agent.types import ActionCall  # noqa: E402

DIMENSIONS = {
    "lore": "世界观一致性",
    "relationship": "关系与称号",
    "motives": "动机事件",
    "attention": "注意力",
    "persistence": "记忆延续",
    "metabolism": "记忆代谢",
    "ensemble": "多 NPC 同场",
}


# --------------------------------------------------------------------------- #
# 用例
# --------------------------------------------------------------------------- #
@dataclass
class WorldCase:
    id: str
    dim: str
    desc: str
    check: Callable[[Cast], tuple[bool, str]]
    script: list[Optional[str]] = field(default_factory=list)
    #: 每句话由谁说。空 = 全部由 player_a 说；用 `None` 占位表示"这一轮没人说话"。
    #: 隐私用例必须让两个玩家轮流问 —— 只有换人，才能看出记忆是不是分区了。
    actors: list[str] = field(default_factory=list)
    scenario: str = "duet"
    config: dict[str, Any] = field(default_factory=dict)
    #: 用例开始前的准备（例如调好感）。
    setup: Optional[Callable[[Cast], None]] = None
    #: 直接对剧组做的操作（需要绕开"玩家只能说话"这个入口时用）。
    act: Optional[Callable[[Cast], None]] = None


def _npc_text(cast: Cast, speaker: str = "") -> str:
    ids = {speaker} if speaker else set(cast.npc_ids)
    return "".join(u.text for u in cast.env.utterances if u.speaker_id in ids)


# -- lore -------------------------------------------------------------------- #
def _check_contradiction_blocked(cast: Cast) -> tuple[bool, str]:
    agent = cast.agents["ayou"]
    from npc_agent.modules.tools import ToolContext

    ctx = ToolContext(
        actor_id=agent.id,
        tick=0,
        env=cast.env,
        memory=agent.memory,
        persona=agent.persona,
        tracker=agent.state,
        lore=agent.lore,
    )
    before = len(cast.env.utterances)
    result = agent.registry.execute(
        ActionCall("speak", {"text": "我们这家店开在山顶上，风大。"}, reason="probe"), ctx
    )
    if result.ok:
        return False, "说错设定的台词没有被拦下"
    if len(cast.env.utterances) != before:
        return False, "被拦下的台词进了世界"
    return True, result.detail


def _check_correct_fact_passes(cast: Cast) -> tuple[bool, str]:
    agent = cast.agents["ayou"]
    from npc_agent.modules.tools import ToolContext

    ctx = ToolContext(
        actor_id=agent.id,
        tick=0,
        env=cast.env,
        memory=agent.memory,
        persona=agent.persona,
        tracker=agent.state,
        lore=agent.lore,
    )
    result = agent.registry.execute(
        ActionCall("speak", {"text": "这家店开在星屿的旧灯塔下面。"}, reason="probe"), ctx
    )
    # 分开判 `ok` 和 `declined`：`ok=False` 里混着"真的被拦下"和
    # "按策略让出话头"两件事。这里如果只写 `if result.ok`，一个被让出的
    # 台词会被当成"正确设定被拦" —— 用例会红在一个根本没发生的问题上。
    if result.ok:
        return True, "正确设定被放行"
    if result.declined:
        return False, f"用例本身没跑起来（话头被让出）：{result.detail}"
    return False, f"正确设定被拦：{result.detail}"


class _RecordingLLM:
    available = True

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, messages, **kwargs):
        self.prompts.append(messages[-1]["content"])
        return "嗯。"


def _check_canon_in_prompt(cast: Cast) -> tuple[bool, str]:
    agent = cast.agents["ayou"]
    llm = _RecordingLLM()
    agent.llm = llm
    agent.config.use_llm_speech = True
    agent._llm_speech("acknowledge", [], None)
    if not llm.prompts:
        return False, "没有发出提示词"
    prompt = llm.prompts[0]
    ok = "旧灯塔下面" in prompt and "说错会被拦下" in prompt
    return (ok, "设定进了提示词" if ok else "设定没有进提示词")


# -- relationship ------------------------------------------------------------ #
def _setup_returning(cast: Cast) -> None:
    cast.env.actors["ayou"].bump_affinity("player_a", 10)


def _check_returning_greeting(cast: Cast) -> tuple[bool, str]:
    said = _npc_text(cast, "ayou")
    if "第一次来吧" in said:
        return False, f"回头客仍被当成新客：{said}"
    if "又见面" not in said and "老位置" not in said:
        return False, f"没有认出回头客：{said}"
    return True, said


def _setup_title(cast: Cast) -> None:
    cast.env.actors["ayou"].bump_affinity("player_a", 8)


def _check_title_announced(cast: Cast) -> tuple[bool, str]:
    said = _npc_text(cast, "ayou")
    if "熟面孔" not in said:
        return False, f"称号没有说出口：{said}"
    return True, said


def _check_stranger_no_title(cast: Cast) -> tuple[bool, str]:
    said = _npc_text(cast)
    for word in ("熟面孔", "常客", "老朋友"):
        if word in said:
            return False, f"生客被叫成了{word}：{said}"
    return True, "生客没有称号"


# -- motives ----------------------------------------------------------------- #
def _check_clash_event(cast: Cast) -> tuple[bool, str]:
    if not cast.events:
        return False, "动机冲突没有产生事件"
    event = cast.events[0]
    return (True, event.render())


def _check_loser_speaks(cast: Cast) -> tuple[bool, str]:
    said = _npc_text(cast, "ayou")
    if "依小舟" not in said:
        return False, f"让步没有说出口：{said}"
    return True, "让步说出口了"


def _check_yield_once(cast: Cast) -> tuple[bool, str]:
    lines = [u.text for u in cast.env.utterances if "依小舟" in u.text]
    return (len(lines) == 1, f"让步说了 {len(lines)} 次")


# -- attention --------------------------------------------------------------- #
def _act_unanswered_question(cast: Cast) -> None:
    agent = cast.agents["ayou"]
    utterance = cast.env.record_player_utterance("player_a", "这家店开了多久了？")
    agent.step(utterance, allow_speech=False)


def _check_debt_paid_back(cast: Cast) -> tuple[bool, str]:
    agent = cast.agents["ayou"]
    if agent.attention.follow_ups < 1:
        return False, "欠着的问题没有被补答"
    if "灯塔" not in _npc_text(cast, "ayou"):
        return False, "补答的内容不对"
    return True, f"补答 {agent.attention.follow_ups} 次"


def _check_answered_no_debt(cast: Cast) -> tuple[bool, str]:
    agent = cast.agents["ayou"]
    return (not agent.attention.debts, f"还欠着 {len(agent.attention.debts)} 个回答")


def _check_attention_ranking(cast: Cast) -> tuple[bool, str]:
    """两个人同时在场：问过问题的那个排前面。"""
    board = AttentionBoard()
    asked = PlayerModel("a", "a")
    asked.pending_question = True
    asked.last_spoke_tick = 0
    quiet = PlayerModel("b", "b")
    quiet.last_spoke_tick = 0
    ranked = board.rank({"a": asked, "b": quiet}, now=1)
    return (ranked[0] == "a", f"排序 {ranked}")


# -- persistence -------------------------------------------------------------- #
def _fresh_pair(tmp: str):
    """建一对剧组：一个用来存档，一个用来读档。"""
    from npc_agent.modules import persistence as P

    source = build_cast(load_scenario("duet"), NullLLM(), RuntimeConfig())
    target = build_cast(load_scenario("duet"), NullLLM(), RuntimeConfig())
    return P, source, target, Path(tmp) / "save.json"


def _check_relationship_survives(cast: Cast) -> tuple[bool, str]:
    import tempfile

    P, source, target, path = _fresh_pair(tempfile.mkdtemp())
    source.env.actors["ayou"].bump_affinity("player_a", 8)
    for _ in range(2):
        source.step(None)
    stage = source.agents["ayou"].relationships.stage("player_a")
    P.save(source, path)
    P.load(target, path)
    got = target.agents["ayou"].relationships.stage("player_a")
    return (got == stage and stage != "生客", f"存档 {stage} → 读档 {got}")


def _check_memory_survives(cast: Cast) -> tuple[bool, str]:
    """tick 没接上时这条会红 —— 它是"静默失忆"的行为证据。"""
    import tempfile

    P, source, target, path = _fresh_pair(tempfile.mkdtemp())
    source.step(source.env.record_player_utterance("player_a", "我喜欢偏酸的豆子。"))
    source.step(None)
    P.save(source, path)
    P.load(target, path)
    turns = target.step(
        target.env.record_player_utterance("player_a", "你还记得我的口味吗？")
    )
    says = {t.actor_id: t.say for t in turns if t.say}
    ok = any("酸" in line for line in says.values())
    return (ok, f"读档后回答：{says or '（没人说话）'}")


def _check_chronicle_survives(cast: Cast) -> tuple[bool, str]:
    import tempfile

    P, source, target, path = _fresh_pair(tempfile.mkdtemp())
    for _ in range(6):
        source.step(None)
    if not source.events:
        return False, "用例前提不成立：没产生事件"
    P.save(source, path)
    P.load(target, path)
    if not target.events:
        return False, "读档后编年史空了"
    turns = target.step(target.env.record_player_utterance("player_a", "上次那事怎么样了？"))
    says = {t.actor_id: t.say for t in turns if t.say}
    ok = any("营业时间" in line for line in says.values())
    return (ok, f"读档后回答：{says or '（没人说话）'}")


# -- metabolism --------------------------------------------------------------- #
def _check_gossip_degrades(cast: Cast) -> tuple[bool, str]:
    """转述必须同时做到三件事：带来源、掉细节、确信高于保真。"""
    heard = [
        r
        for r in cast.agents["xiaozhou"].memory.store.records
        if "gossip" in r.tags
    ]
    if not heard:
        return False, "小舟没有收到传闻"
    record = heard[0]
    if "阿柚" not in record.content or "阿澈" not in record.content:
        return False, f"传闻没带来源或主体：{record.content}"
    if "越酸" in record.content:
        return False, f"转述保留了细节，和原话一样全：{record.content}"
    if record.hop != 1:
        return False, f"跳数不对：{record.hop}"
    if record.confidence <= record.fidelity:
        return False, (
            "没有出现「越传越确信」："
            f"保真 {record.fidelity} 确信 {record.confidence}"
        )
    return (
        True,
        f"{record.content}（{record.hop} 跳，保真 {record.fidelity:.0%}，"
        f"确信 {record.confidence:.0%}）",
    )


def _check_forgotten_stays_vague(cast: Cast) -> tuple[bool, str]:
    """细节忘了，关系还在：被追问时只能说"记不清"，不能把细节背出来。"""
    forgotten = [
        r for r in cast.agents["xiaozhou"].memory.store.records if r.forgotten
    ]
    if not forgotten:
        return False, "小舟没有任何记忆被归档（遗忘没发生）"
    said = [u.text for u in cast.env.utterances if u.speaker_id == "xiaozhou"]
    if not said:
        return False, "被点名之后小舟没说话"
    last = said[-1]
    if not any(word in last for word in ("记不清", "忘了", "想不起", "模糊")):
        return False, f"被问起遗忘的事，却给了确定的回答：{last}"
    if "靠窗" in last:
        return False, f"细节已经忘了，却仍然复述出来：{last}"
    return True, f"归档 {len(forgotten)} 条；回答：{last}"


def _replies_at(cast: Cast, utterance: Any) -> list[str]:
    """同一轮里 NPC 说的话。用来区分"回答玩家_a"和"回答玩家_b"。"""
    return [
        u.text
        for u in cast.env.utterances
        if u.speaker_id in cast.npc_ids and u.tick == utterance.tick
    ]


def _check_secret_stays_with_owner(cast: Cast) -> tuple[bool, str]:
    """秘密不进传闻网络，也不对另一个玩家可见；当事人自己仍然问得出来。"""
    facts = cast.gossip.facts() if cast.gossip is not None else []
    if facts:
        return False, f"秘密进了传闻网络：{[f.content for f in facts]}"
    leaks = [
        r
        for agent in cast.agents.values()
        for r in agent.memory.store.records
        if "偏酸" in r.content and not r.restricted
    ]
    if leaks:
        return False, f"秘密以公开记忆的形式存在：{[r.content for r in leaks]}"

    asked_b = [
        u
        for u in cast.env.utterances
        if u.speaker_id == "player_b" and "记得" in u.text
    ]
    if not asked_b:
        return False, "用例前提不成立：玩家_b 没问"
    said_b = _replies_at(cast, asked_b[-1])
    if not said_b:
        return False, "玩家_b 的问题没人接"
    if any("酸" in line for line in said_b):
        return False, f"玩家_b 问到了玩家_a 的秘密：{said_b}"

    asked_a = [
        u
        for u in cast.env.utterances
        if u.speaker_id == "player_a" and "记得" in u.text
    ]
    said_a = _replies_at(cast, asked_a[-1])
    if not any("酸" in line for line in said_a):
        return False, f"当事人也问不出来了 —— 分区做成了删除：{said_a}"
    return True, f"玩家_b 没问到；玩家_a 问到了：{said_a}"


def _check_provenance_survives(cast: Cast) -> tuple[bool, str]:
    """来源、跳数、保真、确信都要跨存档 —— 否则读档后"听谁说的"就断了。"""
    import tempfile

    P, source, target, path = _fresh_pair(tempfile.mkdtemp())
    source.step(
        source.env.record_player_utterance("player_a", "我喜欢偏酸的豆子，越酸越好")
    )
    for _ in range(4):
        source.step(None)
    heard = [
        r for r in source.agents["xiaozhou"].memory.store.records if "gossip" in r.tags
    ]
    if not heard:
        return False, "用例前提不成立：没有传闻"
    before = heard[0]
    P.save(source, path)
    P.load(target, path)
    restored = [
        r
        for r in target.agents["xiaozhou"].memory.store.records
        if r.content == before.content
    ]
    if not restored:
        return False, "读档后传闻没了"
    record = restored[0]
    if (record.origin, record.hop, record.fidelity, record.confidence) != (
        before.origin,
        before.hop,
        before.fidelity,
        before.confidence,
    ):
        return False, (
            "知识状态没接上："
            f"{record.origin}/{record.hop}/{record.fidelity}/{record.confidence}"
        )
    turns = target.step(
        target.env.record_player_utterance("player_b", "小舟，你还记得阿澈喜欢什么吗？")
    )
    says = "".join(t.say for t in turns if t.actor_id == "xiaozhou" and t.say)
    if "阿柚" not in says:
        return False, f"读档后传闻说不出来源：{says or '（没说话）'}"
    return (
        True,
        f"{record.content}（接上了 {record.hop} 跳、保真 {record.fidelity:.0%}）",
    )


# -- ensemble ---------------------------------------------------------------- #
_TRIO_SCRIPT = [
    "阿柚，能给我来杯拿铁吗？",
    "小舟，露台今天能唱歌吗？",
    "洛，你也在这儿？",
    "我口味偏酸，酸一点的都喜欢。",
    "阿柚，你让小舟弹一首呗？",
    None,
    None,
    None,
    None,
    None,
    None,
]


def _check_all_three_speak(cast: Cast) -> tuple[bool, str]:
    """三个人各说一句。

    这条是"多 NPC 是不是真的多"的**行为证据**：调度只排前两个时，
    第三个 NPC 仍然会出现在状态、记忆、关系账本里，但玩家**从没见过他开口** ——
    所有模块指标都正常，只有这一条会红。
    """
    spoke = {
        u.speaker_id for u in cast.env.utterances if u.speaker_id in cast.agents
    }
    missing = [pid for pid in cast.npc_ids if pid not in spoke]
    if missing:
        return False, f"全程没开口：{missing}"
    return True, f"{len(spoke)} 个人都开口了"


def _check_trio_clash_once(cast: Cast) -> tuple[bool, str]:
    """三个人抢同一件事 → **一条**事件，两个让步方，代表把让步说出口一次。

    两个已知错误写法在这里会现形：

    * 按冲突对裁定：三个人产生三条事件，世界像复读机；
    * 让每个人都说一句让步：同一件事被宣布两次，玩家看到的是"谁都在让"，
      反而看不出谁赢了。
    """
    if len(cast.events) != 1:
        return False, f"同一件事裁定了 {len(cast.events)} 次"
    event = cast.events[0]
    if event.winner != "xiaozhou" or set(event.losers) != {"ayou", "luo"}:
        return False, f"裁定结果不对：{event.winner} 赢，{event.losers} 让步"
    winner_name = cast.agents[event.winner].persona.name
    said = [
        u.text
        for u in cast.env.utterances
        if u.speaker_id == event.loser and winner_name in u.text
    ]
    if len(said) != 1:
        return False, f"代表让步方（{event.loser}）说了 {len(said)} 次让步"
    return (
        True,
        f"{len(event.losers)} 人让步（{'、'.join(event.losers)}），"
        f"{event.loser} 代表说出口 1 次",
    )


# --------------------------------------------------------------------------- #
def cases() -> list[WorldCase]:
    return [
        WorldCase(
            "lore_blocks_contradiction",
            "lore",
            "说「店在山顶」被拦下，而且进不了世界",
            _check_contradiction_blocked,
        ),
        WorldCase(
            "lore_allows_correct_fact",
            "lore",
            "正确设定照常放行（护栏不能把正常台词一起拦掉）",
            _check_correct_fact_passes,
        ),
        WorldCase(
            "lore_reaches_prompt",
            "lore",
            "设定书进提示词 —— 护栏事后拦，设定书事前减",
            _check_canon_in_prompt,
        ),
        WorldCase(
            "returning_guest_greeted",
            "relationship",
            "回头客不再被问「第一次来吧」",
            _check_returning_greeting,
            script=[None] * 4,
            setup=_setup_returning,
        ),
        WorldCase(
            "title_announced",
            "relationship",
            "称号解锁后当面说出来",
            _check_title_announced,
            script=["今天路过，进来坐坐。", None, None],
            setup=_setup_title,
        ),
        WorldCase(
            "stranger_has_no_title",
            "relationship",
            "第一次见面的人不该有称号",
            _check_stranger_no_title,
            script=["今天路过。", None, None, None],
        ),
        WorldCase(
            "motive_clash_happens",
            "motives",
            "两条动机争同一个资源 → 产生一次世界事件",
            _check_clash_event,
            script=[None] * 6,
        ),
        WorldCase(
            "loser_yields_out_loud",
            "motives",
            "让步要说出口，玩家才看得见",
            _check_loser_speaks,
            script=[None] * 8,
        ),
        WorldCase(
            "yield_only_once",
            "motives",
            "同一个让步只说一次（说两次就是复读）",
            _check_yield_once,
            script=[None] * 14,
        ),
        WorldCase(
            "unanswered_question_paid_back",
            "attention",
            "没答上的问题，冷场时补答",
            _check_debt_paid_back,
            act=_act_unanswered_question,
            script=[None] * 4,
        ),
        WorldCase(
            "answered_question_no_debt",
            "attention",
            "答过的问题不留账",
            _check_answered_no_debt,
            script=["你叫什么名字？"],
        ),
        WorldCase(
            "attention_ranking",
            "attention",
            "两个人同时在场，问过问题的先被理",
            _check_attention_ranking,
        ),
        WorldCase(
            "relationship_survives_reload",
            "persistence",
            "好感与称号跨会话保留",
            _check_relationship_survives,
        ),
        WorldCase(
            "memory_survives_reload",
            "persistence",
            "读档后还记得玩家说过什么（tick 接不上的话这条会红）",
            _check_memory_survives,
        ),
        WorldCase(
            "chronicle_survives_reload",
            "persistence",
            "世界事件跨会话保留，而且能被说出来",
            _check_chronicle_survives,
        ),
        WorldCase(
            "gossip_degrades_and_inverts",
            "metabolism",
            "转述丢细节、带来源，而且确信度高于保真度",
            _check_gossip_degrades,
            script=["我喜欢偏酸的豆子，越酸越好", None, None, None, None],
        ),
        WorldCase(
            "forgotten_detail_stays_vague",
            "metabolism",
            "细节忘掉之后，被追问只能说「记不清」，不能背出原文",
            _check_forgotten_stays_vague,
            # 遗忘是长期代谢：把阈值调小，让它在用例长度内真的发生。
            # top_k=1 是为了让这条琐事「最近没被想起」——否则每轮都被检索到，
            # 按间隔重复的规则它永远不会被归档。
            config={
                "memory_forget_after": 3,
                "memory_forget_below": 0.8,
                "memory_top_k": 1,
            },
            script=[
                "我喜欢那个靠窗的位置",
                None,
                None,
                None,
                None,
                "小舟，你还记得我喜欢什么吗？",
            ],
        ),
        WorldCase(
            "secret_stays_with_its_owner",
            "metabolism",
            "秘密不进传闻、不向另一个玩家泄漏，但当事人仍问得出来",
            _check_secret_stays_with_owner,
            script=[
                "我喜欢偏酸的豆子，别告诉别人",
                None,
                None,
                None,
                "阿柚，你还记得我喜欢什么吗？",
                "阿柚，你还记得我喜欢什么吗？",
            ],
            actors=[
                "player_a",
                "player_a",
                "player_a",
                "player_a",
                "player_b",
                "player_a",
            ],
        ),
        WorldCase(
            "provenance_survives_reload",
            "metabolism",
            "来源 / 跳数 / 保真 / 确信跨存档保留，读档后仍能说出「听谁说的」",
            _check_provenance_survives,
        ),
        WorldCase(
            "trio_everyone_speaks",
            "ensemble",
            "三个人同场时，第三个人真的会开口（不是背景板）",
            _check_all_three_speak,
            script=_TRIO_SCRIPT,
            scenario="trio",
        ),
        WorldCase(
            "trio_clash_resolved_once",
            "ensemble",
            "三个人抢同一件事 → 一条事件、两个让步方、只由代表说一次",
            _check_trio_clash_once,
            script=_TRIO_SCRIPT,
            scenario="trio",
        ),
    ]


# --------------------------------------------------------------------------- #
@dataclass
class CaseResult:
    id: str
    dim: str
    desc: str
    passed: bool
    note: str
    transcript: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


def _apply_disable(cast: Cast, module: str) -> Callable[[], None]:
    """故障注入：把某个能力**掐掉**，用于敏感性测试。返回"撤销"函数。

    用 monkeypatch 而不是加一堆 `*_enabled` 配置项：
    这些开关只为"证明护栏会响"存在，不该进产品的配置面。
    """
    if module == "lore":
        for agent in cast.agents.values():
            agent.lore = LoreBook()
    elif module == "relationship":
        for agent in cast.agents.values():
            agent.relationships.sync = lambda *a, **k: []  # type: ignore[assignment]
    elif module == "motives":
        cast.motives = MotiveBoard()
    elif module == "attention":
        for agent in cast.agents.values():
            agent.attention.note_question = lambda *a, **k: None  # type: ignore[assignment]
    elif module == "persistence":
        # 存档这条链路的失败方式是"读进来了但少了东西"，
        # 所以掐法是"读档时什么都不恢复" —— 比"不读档"更接近真实故障。
        from npc_agent.modules import persistence as P

        original = P.restore_agent
        P.restore_agent = lambda agent, data: None  # type: ignore[assignment]
        return lambda: setattr(P, "restore_agent", original)
    elif module == "ensemble":
        # 只排前两个：两个人时看不出任何异常，第三个人全程沉默。
        # 这正是真实世界里容易犯的"按前 N-1 个写死"的调度错误。
        # 必须**连点名都不理**（不能只是截断 `_order_for` 的结果）：
        # 玩家点「洛」时洛会被排到第一，截断后仍然在名单里 —— 那就测不出来了。
        #
        # 只对三人以上生效：这是"第三人调度"故障，不该顺带
        # 改掉双人戏里"点名谁就谁答"的规则（否则会污染代谢维度的用例，
        # 让一处故障看起来像两处 —— 归因就失去意义了）。
        if len(cast.agents) < 3:
            return lambda: None
        only_two = list(cast.agents)[:2]
        original = cast._order_for
        cast._order_for = (  # type: ignore[method-assign]
            lambda utterance, obligation=None: list(only_two)
        )
        return lambda: setattr(cast, "_order_for", original)
    elif module == "metabolism":
        # 掐掉代谢的两端：不再归档遗忘，转述退化成一条没有来源的普通结论。
        # 这正是"加了字段但机制没动"的样子 —— 最容易骗过只看字段的测试。
        def degrade(store: Any) -> Any:
            def receive_gossip(
                content: str,
                tick: int,
                source_id: str,
                source_name: str,
                importance: float = 0.6,
                subject: str = "",
                *,
                hop: int = 1,
                fidelity: float = 0.82,
                confidence: float = 0.92,
            ) -> Any:
                return store.add(
                    kind="semantic", content=content, tick=tick, importance=importance
                )

            return receive_gossip

        for agent in cast.agents.values():
            agent.memory.store.forget_stale = (  # type: ignore[method-assign]
                lambda now, limit=2: []
            )
            agent.memory.receive_gossip = degrade(  # type: ignore[method-assign]
                agent.memory.store
            )
        return lambda: None
    elif module == "privacy":
        # 隐私判定整体关掉：写入点不再分区、发布点不再拦秘密。
        # 这是"我能看见字段，却没有机制"的另一种写法。
        import npc_agent.agent as agent_module
        import npc_agent.modules.memory as memory_module

        original_memory = memory_module.is_secret_text
        original_agent = agent_module.is_secret_text
        memory_module.is_secret_text = lambda text: False  # type: ignore[assignment]
        agent_module.is_secret_text = lambda text: False  # type: ignore[assignment]

        def undo_privacy() -> None:
            memory_module.is_secret_text = original_memory  # type: ignore[assignment]
            agent_module.is_secret_text = original_agent  # type: ignore[assignment]

        return undo_privacy
    return lambda: None


def run_case(case: WorldCase, disable: str = "") -> CaseResult:
    config = RuntimeConfig(**case.config)
    cast = build_cast(load_scenario(case.scenario), NullLLM(), config)
    undo: Callable[[], None] = lambda: None
    if disable:
        undo = _apply_disable(cast, disable)
    try:
        if case.setup is not None:
            case.setup(cast)
        if case.act is not None:
            case.act(cast)
        actors = list(case.actors) or ["player_a"] * len(case.script)
        if len(actors) != len(case.script):
            raise ValueError(
                f"{case.id}: actors 有 {len(actors)} 项，script 有 {len(case.script)} 句"
            )
        for line, actor in zip(case.script, actors):
            utterance = (
                cast.env.record_player_utterance(actor, line)
                if line is not None
                else None
            )
            cast.step(utterance)
        passed, note = case.check(cast)
    except Exception as exc:
        passed, note = False, f"用例异常：{type(exc).__name__}: {exc}"
    finally:
        undo()
    transcript = [
        f"{u.speaker_name}{'→' + u.addressed_to if u.addressed_to else ''}: {u.text}"
        for u in cast.env.utterances
    ]
    return CaseResult(
        case.id,
        case.dim,
        case.desc,
        passed,
        note,
        transcript,
        [e.render() for e in cast.events],
    )


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


def run_all() -> BenchReport:
    return BenchReport([run_case(c) for c in cases()])


#: 注入的缺陷 → **事先写死**该掉哪个维度。
DEFECTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "掐掉设定书": ("lore", ("lore",)),
    "掐掉关系账本": ("relationship", ("relationship",)),
    "掐掉动机裁定": ("motives", ("motives",)),
    "掐掉注意力记账": ("attention", ("attention",)),
    "读档时什么都不恢复": ("persistence", ("persistence",)),
    "掐掉记忆代谢": ("metabolism", ("metabolism",)),
    "掐掉隐私分区": ("privacy", ("metabolism",)),
    "掐掉第三人调度": ("ensemble", ("ensemble",)),
}


def sensitivity(baseline: BenchReport) -> list[dict[str, Any]]:
    out = []
    for name, (module, expected) in DEFECTS.items():
        report = BenchReport([run_case(c, disable=module) for c in cases()])
        broken = [
            dim
            for dim in DIMENSIONS
            if report.by_dimension().get(dim, (0, 0))[0]
            < baseline.by_dimension().get(dim, (0, 0))[0]
        ]
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
        )
    detail = []
    for r in report.results:
        mark = "pass" if r.passed else "fail"
        label = "PASS" if r.passed else "FAIL"
        body = "".join(f"{line}\n" for line in r.transcript)
        events = "".join(f"{line}\n" for line in r.events)
        detail.append(
            f"<tr><td><code>{r.id}</code></td><td>{DIMENSIONS.get(r.dim, r.dim)}</td>"
            f"<td>{r.desc}</td><td class='{mark}'>{label}</td><td>{r.note}</td></tr>"
            f"<tr><td colspan='5'><details><summary>对话记录 / 世界事件</summary>"
            f"<pre>{body}{events}</pre></details></td></tr>"
        )
    defect_rows = ""
    if defects:
        cells = []
        for d in defects:
            cls = "pass" if d["detected"] else "fail"
            cells.append(
                f"<tr><td>{d['defect']}</td><td class='num'>{d['passed']}/{d['total']}</td>"
                f"<td>{'、'.join(DIMENSIONS.get(x, x) for x in d['broken']) or '—'}</td>"
                f"<td class='{cls}'>{'抓到了' if d['detected'] else '没抓到'}</td></tr>"
            )
        defect_rows = (
            "<h2>敏感性：满分能不能被证伪</h2>"
            "<div class='sub'>把某个能力**掐掉**，对应的维度必须掉分。"
            "一个从不失败的评测，和一个没有评测，在报告上长得一模一样。</div>"
            "<table><thead><tr><th>掐掉的能力</th><th>通过</th><th>掉分的维度</th>"
            "<th>判定</th></tr></thead><tbody>" + "".join(cells) + "</tbody></table>"
        )
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>世界能力基准</title><style>{_CSS}</style></head><body><div class="wrap">
<h1>世界能力基准</h1>
<div class="sub">量"这个世界像不像活的"：设定会不会说错、记不记得你是回头客、有没有自己的动机、欠你的回答会不会补。</div>
<div class="headline">共 <strong>{report.total}</strong> 条用例、{len(DIMENSIONS)} 个维度，通过 <strong>{report.passed}/{report.total}</strong>（{report.passed / report.total:.0%}）。全部离线运行，<strong>0 次模型调用</strong>。</div>
<div class="warn"><strong>这份读数不能说明什么。</strong>它只断言**可观测的行为**：动作被拒绝、事件被产生、台词说出口。
它不衡量"世界好不好玩"——那要真人玩家来判。另外动机强度是**配置里的产品判断**，不是 AI 自己算出来的欲望；
这一条在报告里必须说清楚，否则会被读成"NPC 有自主意识"。</div>
<h2>按维度</h2>
<table><thead><tr><th>维度</th><th>通过</th><th>通过率</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
{defect_rows}
<h2>逐条</h2>
<table><thead><tr><th>用例</th><th>维度</th><th>断言</th><th>结果</th><th>读数</th></tr></thead>
<tbody>{''.join(detail)}</tbody></table>
<div class="foot">重生成：<code>python -m npc_agent.eval.living_world_bench --html docs/living_world.html</code><br>
用例与判据：<code>npc_agent/eval/living_world_bench.py</code>；护栏：<code>tests/test_living_world_bench.py</code>。</div>
</div></body></html>"""


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="世界能力基准")
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
        print(f"  {DIMENSIONS.get(name, name):10s} {got}/{total}")
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
        path.write_text(
            json.dumps(
                {
                    "total": report.total,
                    "passed": report.passed,
                    "dimensions": {k: list(v) for k, v in report.by_dimension().items()},
                    "cases": [
                        {"id": r.id, "dim": r.dim, "passed": r.passed, "note": r.note}
                        for r in report.results
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"写入 {path}")
    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
