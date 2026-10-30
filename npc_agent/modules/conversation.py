"""对话线程 —— 让"NPC 和 NPC 说话"变成一条真的有来回的线程。

## 这个模块解决的是哪一种失败

单 NPC 的对话只有一条边：玩家 → NPC。
场上站着两个 NPC，就多出一条边：NPC → NPC。而这条边**不会自己出现**，
因为每个 NPC 都只看得到自己的状态：

    阿柚：小舟，你来一首？
    小舟：（她不知道这句话是给她的，于是继续弹自己的琴）

实测（`scripts/probe_npc_dialogue.py`，duet，8 轮离线）：
**NPC 台词 5 句，其中对另一个 NPC 说的 0 句，被点名后接话 0 次。**
也就是说「NPC 之间沟通交流」在这个框架里原本**不存在** ——
不是做得不好，是那条边没有接上。

## 三件事，缺一不可

    1. 谁说给谁听   `ReplyObligation` —— 一条"这一句是给你的，下一轮你得接"的债
    2. 接下来聊什么 `TopicLedger`   —— 记下这个话题已经说过哪些内容点，别再说一遍
    3. 什么时候停   `chain_limit`   —— NPC 之间不能无限对聊，聊够了要把话头还给玩家

第 3 点看起来是"限制"，其实是**产品判断**：玩家站在旁边看两个 NPC 聊十轮，
不会觉得世界很活，只会觉得自己被晾着。所以链长默认 3 ——
够形成"你一句我一句"的来回，不够形成一段没有玩家的小剧场。

## 为什么"接话义务"要显式建模，而不是靠 NPC 自觉

靠自觉的写法是：让每个 NPC 自己判断"上一句是不是在跟我说话"。
问题是**上一句是谁说的、说给谁的**，这件事只有导演知道 ——
NPC 自己的现场状态里，同伴的发言和玩家的发言混在同一个列表里。
把它交给 NPC 判断，两个 NPC 会同时对同一句话负责（都接），
或者都不接（都以为对方会接）。这和多 NPC 抢话是同一类问题的两面。

所以义务由**导演**开、由**导演**销：一个 tick 只有一个人被欠一句话。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from ..types import Utterance

# --------------------------------------------------------------------------- #
# 话题账本
# --------------------------------------------------------------------------- #
def normalize_point(text: str) -> str:
    """内容点的归一化。和复读判据用同一套口径（去标点与空白）。

    口径必须一致：复读判据说"这两句是同一句"，账本却说"这是两个内容点"，
    两个模块就会互相拆台 —— 闸门放行的句子，账本认为没说过。
    """
    return re.sub(r"[\s，。！？、；：…—～·「」『』“”‘’（）()\[\]【】《》〈〉,.!?;:\-—_/\\|~`\"']+", "", text or "")


@dataclass(frozen=True)
class ContentPoint:
    """一个"内容点" = 话题 + 具体说法。

    为什么不是"这句话的哈希"：同一件事可以有无数种说法
    （「这家店开在灯塔下面」/「灯塔底下那家」），按句子去重永远去不干净。
    所以内容点由**产出方声明**（`intent` + 关键参数），而不是由消费者猜。
    这条原则在这个项目里已经踩过一次坑（见 `types.OUTCOME_*` 的注释）。
    """

    topic: str
    detail: str


class TopicLedger:
    """记下"这个话题里哪些内容点已经用过了"。

    它比 `agent.said`（逐字比对）高一层：
    `said` 管的是"别把同一句话说两遍"，账本管的是"别把同一件事说两遍"。
    两者都要有 —— 前者抓逐字复读，后者抓换了个说法的复读。

    实测动机（duet，10 轮，修复前）：玩家问「有什么推荐」，
    NPC 回「你上次说过第一次来吧。」—— 那句记忆被回引了 4 次。
    逐字判据在第 2 次就该拦住它，但它每次都被**换了个前缀**
    （「对了，你之前提过…」/「说起来，你上次提过…」），于是逐字判据放行。
    """

    def __init__(self) -> None:
        self._used: dict[ContentPoint, int] = {}
        self._by_topic: dict[str, set[str]] = {}

    # -- 写 ---------------------------------------------------------- #
    def mark(self, point: ContentPoint, tick: int) -> None:
        self._used.setdefault(point, tick)
        self._by_topic.setdefault(point.topic, set()).add(point.detail)

    # -- 读 ---------------------------------------------------------- #
    def is_spent(self, point: ContentPoint) -> bool:
        return point in self._used

    def spent_in(self, topic: str) -> set[str]:
        return set(self._by_topic.get(topic, ()))

    def topics(self) -> list[str]:
        return list(self._by_topic)

    def size(self) -> int:
        return len(self._used)

    def render(self) -> str:
        if not self._used:
            return "（还没有说过任何内容点）"
        lines = []
        for topic, details in self._by_topic.items():
            lines.append(f"  {topic}: {len(details)} 个内容点已用")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 接话义务
# --------------------------------------------------------------------------- #
@dataclass
class ReplyObligation:
    """"这一句是给你的" —— 下一轮该谁接、接什么。

    `chain` 是这条义务在**同一段连续 NPC 对聊**里的序号。
    它让"聊够了就停"变成一个可数的判据，而不是靠感觉。
    """

    target_id: str
    speaker_id: str
    speaker_name: str
    text: str
    tick: int
    chain: int
    requires_reply: bool = True
    reason: str = ""

    def as_utterance(self) -> Utterance:
        """把这条义务还原成"被点名者听到的那句话"。

        注意 `role="npc"` —— 下游（`Agent._respond` / 意图分类）靠它
        区分"玩家在跟我说话"和"同伴在跟我说话"，这两者的正确反应不一样：
        对玩家要答他的问题，对同伴要接他的话头（可能只是应一声）。
        """
        return Utterance(
            speaker_id=self.speaker_id,
            speaker_name=self.speaker_name,
            text=self.text,
            tick=self.tick,
            role="npc",
            is_question=looks_like_question_zh(self.text),
            addressed_to=self.target_id,
            # ⚠️ 这个字段必须一起带过来。漏了它，接收方会把"转达"当成
            # 一句普通陈述 —— 于是「阿柚：小舟，阿澈叫你呢。」得到的是
            # 「……还行。」（应一声），而不是「……来了。」
            # 这正是"答非所问"在同伴之间最轻的一种表现，也是最难发现的：
            # 语法上通顺，语义上完全错位。
            expects_reply=self.requires_reply,
        )


#: 中文问句的粗筛。**刻意和 `types.looks_like_question` 分开**：
#: 那边判的是"玩家在问问题"（误判代价低），这边判的是"同伴这句话要不要接"
#: （误判代价是 NPC 之间答非所问）。两件事的代价不同，就不该共用一个阈值 ——
#: 但口径必须写在同一处注释里，否则下一个人会以为它们是一回事。
#:
#: ⚠️ **这里不含「呢」。** `types` 那边含，因为"玩家问了一句"误判成陈述
#: 只是少回答一句；这边含「呢」会把「小舟，阿澈叫你呢。」判成问句 ——
#: 于是被叫的人回一句「……还行。」而不是「……来了。」
#: （实测：demo 里就是这个表现，语法通顺、语义完全错位）。
#: 「呢」在中文里大量用作语气助词（叫你呢 / 记着呢 / 还早呢），
#: 只有带问号时才是疑问 —— 那一条由句末 `？` 覆盖。
def looks_like_question_zh(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if t.endswith(("?", "？")):
        return True
    return t.rstrip("。！!~～… ").endswith(("吗", "么"))


# --------------------------------------------------------------------------- #
# 话语类型
# --------------------------------------------------------------------------- #
#: 一条发言的"类型"。回应的意图由类型决定，而不是由"有没有匹配到模板"决定。
#:
#: 这是治「答非所问」的核心：修复前，意图是**从模板池里挑**的
#: （先看能不能回引记忆，再看能不能直接答，最后落到 acknowledge），
#: 于是「你让小舟弹一首呗？」这种**转达请求**找不到模板，落到 `unknown`，
#: NPC 回一句「这个我不太清楚」—— 它不知道的是"该转达"，不是"这件事"。
KIND_PEER_TURN = "peer_turn"          # 同伴点了我
KIND_DELEGATE = "delegate"            # 让我去叫另一个 NPC（转达）
KIND_GROUP_QUESTION = "group_question"  # 问"你们"（群体，谁都该答）
KIND_SELF_QUESTION = "self_question"  # 问我自己的事
KIND_QUESTION = "question"            # 一般提问
KIND_STATEMENT = "statement"          # 陈述

#: 转达请求的措辞。**必须成对出现**（一个"叫"类动词 + 一个 NPC 名字）才算，
#: 只看动词会把「你让我想想」也判成转达。
#:
#: 「问」也在表里，而且故意排在最前：`你问问阿柚今天忙不忙` 是最自然的一种转达，
#: 漏掉它，小舟会把它当成"问我自己的问题"，答一句「这个我说不好」。
#: 判宽一点是安全的 —— 它同时要求句子里出现**同伴的名字**。
_DELEGATE_VERBS = ("问问", "问", "让", "叫", "请", "找", "喊", "通知", "转告")

#: 群体指代。中文里"你们"是最可靠的多人信号 —— 它几乎不可能指一个人。
_GROUP_MARKERS = ("你们", "你俩", "两位", "大家", "二位")


def classify_utterance(
    utterance: Optional[Utterance],
    *,
    self_id: str,
    self_name: str,
    peer_ids: Iterable[str] = (),
    peer_names: Iterable[str] = (),
) -> tuple[str, Optional[str]]:
    """判断这句话是什么类型；转达类还要给出"该转达给谁"。

    返回 `(kind, target_peer_id)`。`target_peer_id` 只有 `delegate` 才有。

    ⚠️ 这里**不做语义理解**，只做模式识别，误判是必然的。
    所以每条判据都按"误判之后会怎样"排过序：
    把陈述误判成提问，代价是 NPC 多问一句；把转达误判成提问，
    代价是 NPC 说"我不知道"（就是修复前那句「这个我不太清楚」）。
    后者更难看，所以转达优先判、判宽一点。
    """
    if utterance is None:
        return KIND_STATEMENT, None
    text = utterance.text or ""
    if not text.strip():
        return KIND_STATEMENT, None

    if utterance.role == "npc":
        # 同伴的话只有一种情况需要"回应"：它是冲我来的。
        if utterance.addressed_to == self_id or self_name in text:
            return KIND_PEER_TURN, None
        return KIND_STATEMENT, None

    # 转达：文本里既有一个 NPC 名字，又有一个"叫"类动词。
    peers = dict(zip(list(peer_ids), list(peer_names)))
    for pid, pname in peers.items():
        if pid == self_id or not pname:
            continue
        if pname in text and any(verb in text for verb in _DELEGATE_VERBS):
            return KIND_DELEGATE, pid

    if any(marker in text for marker in _GROUP_MARKERS):
        return KIND_GROUP_QUESTION, None

    if utterance.is_question:
        if self_name in text or any(a in text for a in (self_name,)):
            return KIND_SELF_QUESTION, None
        return KIND_QUESTION, None

    return KIND_STATEMENT, None


# --------------------------------------------------------------------------- #
# 对话板
# --------------------------------------------------------------------------- #
@dataclass
class DialogueLine:
    """一句已经说出口的话，带"说给谁"。转写与报告都从这里取。"""

    tick: int
    speaker_id: str
    speaker_name: str
    text: str
    addressed_to: Optional[str] = None
    point: Optional[ContentPoint] = None


class ConversationBoard:
    """跨 NPC 共享的对话状态。由导演（`Cast`）持有，NPC 只读。

    **它不是记忆。** 记忆是"我记得你说过什么"，是私有的、会衰减的；
    这块板子是"这一轮谁欠谁一句话"，是公共的、只活几轮。
    把两者混在一起，就会出现"三天前的一句问话还在要求回应"这种怪事。
    """

    def __init__(self, *, chain_limit: int = 3) -> None:
        self.ledger = TopicLedger()
        self.lines: list[DialogueLine] = []
        self.pending: Optional[ReplyObligation] = None
        self.chain_limit = max(0, int(chain_limit))
        #: 当前这段连续 NPC 对聊已经走到第几轮。玩家一开口就归零。
        self.chain = 0
        #: 因为链长上限而**主动掐断**的次数。它是"机制在工作"的证据，
        #: 不是失败 —— 所以单独计数，不混进任何失败率。
        self.truncated = 0

    # -- 义务 -------------------------------------------------------- #
    def can_chain(self) -> bool:
        return self.chain < self.chain_limit

    def open(
        self,
        *,
        target_id: str,
        speaker_id: str,
        speaker_name: str,
        text: str,
        tick: int,
        reason: str = "",
    ) -> Optional[ReplyObligation]:
        """开一条接话义务。链长到顶时返回 None（并记一次掐断）。"""
        if not self.can_chain():
            self.truncated += 1
            return None
        obligation = ReplyObligation(
            target_id=target_id,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            text=text,
            tick=tick,
            chain=self.chain,
            reason=reason,
        )
        self.pending = obligation
        return obligation

    def take(self, actor_id: str) -> Optional[ReplyObligation]:
        """把义务交给该接的人。不是他 → 不动它（别人的债不该被清掉）。"""
        if self.pending is None or self.pending.target_id != actor_id:
            return None
        obligation = self.pending
        self.pending = None
        self.chain = obligation.chain + 1
        return obligation

    def expire(self) -> None:
        """这一轮没人接 —— 义务作废。

        作废而不是留着：留着会让"阿柚问了一句、小舟没接"变成
        "下一轮小舟还得接"，于是玩家插进来一句话之后，小舟突然回答
        三轮前的一个问题。**过期是对话的一部分**，人也是这样。
        """
        self.pending = None

    def player_turn(self) -> None:
        """玩家开口 → 这段 NPC 对聊结束，义务清空。"""
        self.pending = None
        self.chain = 0

    # -- 记录 -------------------------------------------------------- #
    def record(
        self,
        *,
        tick: int,
        speaker_id: str,
        speaker_name: str,
        text: str,
        addressed_to: Optional[str] = None,
        point: Optional[ContentPoint] = None,
    ) -> DialogueLine:
        line = DialogueLine(
            tick=tick,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            text=text,
            addressed_to=addressed_to,
            point=point,
        )
        self.lines.append(line)
        if point is not None:
            self.ledger.mark(point, tick)
        return line

    def recent(self, n: int = 6) -> list[DialogueLine]:
        return self.lines[-n:]

    def stats(self) -> dict[str, int]:
        return {
            "lines": len(self.lines),
            "points": self.ledger.size(),
            "chain": self.chain,
            "truncated": self.truncated,
        }


__all__ = [
    "ContentPoint",
    "ConversationBoard",
    "DialogueLine",
    "KIND_DELEGATE",
    "KIND_GROUP_QUESTION",
    "KIND_PEER_TURN",
    "KIND_QUESTION",
    "KIND_SELF_QUESTION",
    "KIND_STATEMENT",
    "ReplyObligation",
    "TopicLedger",
    "classify_utterance",
    "looks_like_question_zh",
    "normalize_point",
]
