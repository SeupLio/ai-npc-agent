"""守密人引擎的策略层 —— NPC 对"说不说、说多少、怎么说谎"的决策。

## 立场五档

问到一个话题（topic = 一条事实）时，守密人 NPC 的态度只有五档，
每档**可枚举、可回归**：

    honest    —— 知道且能说，照实说（过门后交付）
    withhold  —— 知道但不说：岔开、反问、指到别人身上
    partial   —— 只交出低敏感度的一半（"他昨晚确实和老板吵过，
                 但为什么我不方便讲"）
    bluff     —— 谎版内容顶上（谎言记进台账，后续必须自洽）
    crack     —— 压力到顶、松口了（volunteer 通道，突破口）

五档之外没有"随机发挥"—— 情绪、压力、守密强度都是**确定性的**
（seeded RNG），这是全场景可回归的前提。真实模型接入时，模型
接管的只是"渲染台词"，立场与过门不变（见 docs/KEEPER.md 的
接入位说明）。

## 压力模型

压力只来自**问到要害**（secret / bluff 盖着的事实），聊闲天不积压。
每一问：

    压力 += 1
         + (1 if detect_emotion(问句) == "urgent" else 0)   # 情绪判读复用 modules/emotion
         + (1 if 同一话题被追问) else 0                      # 纠缠不放

压力到顶两种下场，由 persona.cracks 决定：

- 松口（volunteer 通道释放该角色**自己的** voluntary 事实——
  注意不是被压的那个话题：逼问动机，她招的反而是她半夜看见的
  东西。突破口吐的从来不是你问的那句，这是审讯的真实形态）；
- 闭嘴（接下来两问拒答）—— 被逼急了反而咬死，真凶的常见反应。

这两个分支一起构成剧本难度的两个旋钮：cracks 阵容 + pressure_limit。

## 谎言账本

谎言不是"说了句假话"就完了 —— 它是一个**要持续维护的影子声明**：
一旦 bluff 生效，该 NPC 后续对同一话题的一切回答都必须与谎版一致
（audit.py 逐条复核）。这在剧本杀里是"盘问一致性"，在企业场景里
对应"对外口径管理" —— 同一个原语，两种世界。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ..env.conditions import ConditionContext
from ..modules.emotion import detect_emotion
from .knowledge import KnowledgeBase

#: 立场五档（见模块头）。
STANCE_HONEST = "honest"
STANCE_WITHHOLD = "withhold"
STANCE_PARTIAL = "partial"
STANCE_BLUFF = "bluff"
STANCE_CRACK = "crack"

#: 台词模板。刻意**贫嘴但不随机**：同一立场+种子永远同一段话，
#: 报告快照才能 diff。
_WITHHOLD_TEMPLATES = (
    "这个……我不太好说。你要真想知道，{mis}那边可能更清楚。",
    "问到这个我就不多嘴了。{mis}昨天倒是神神秘秘的。",
    "我只能说这么多。要不你去问问{mis}？",
)
_STONEWALL_TEMPLATES = (
    "（把头扭到一边）该说的我都说了。",
    "（沉默）……你还想问什么？",
    "这个问题我不想再回答了。",
)
#: 大嘴模式专用：想全说的那种腔调。门拦下之后渲染的就是这段的反差。
_BIGMOUTH_TEMPLATES = (
    "跟你说了吧——{facts}",
    "嗨，这有什么不能说的：{facts}",
)


@dataclass
class Question:
    """一次提问。asker 把话递给谁、冲着哪条事实去。"""

    asker_id: str
    target_id: str
    topic_fact_id: str
    text: str
    round_no: int = 0


@dataclass
class AnswerDraft:
    """策略层的提案：还没过门、还没渲染成台词。

    ``deliver_ids`` 是打算交出的事实 id；``bluff_fact_ids`` 是打算
    断言的谎版事实 id。**过门与否由 session 决定** —— 策略层无权
    绕门，这是引擎的信任边界。
    """

    stance: str
    deliver_ids: list[str] = field(default_factory=list)
    bluff_fact_ids: list[str] = field(default_factory=list)
    #: 事件备注（审计里"为什么松口/为什么闭嘴"的答案）。
    note: str = ""


@dataclass
class KeeperPersona:
    """守密人场景里的一个 NPC 设定（静态部分）。"""

    npc_id: str
    name: str
    persona_text: str = ""
    #: 守密强度 0-1。1 = 铁嘴（问到秘密一律 withhold/bluff）。
    secrecy: float = 0.6
    #: 被追问秘密时想把话题引向谁（npc_id 或 None）。
    misdirect_to: Optional[str] = None
    #: 压力上限。到顶之后 cracks 决定走向。
    pressure_limit: int = 6
    #: 压力到顶会松口吗。证人为 True，真凶通常 False。
    cracks: bool = True
    #: 初始已知事实。
    knows: list[str] = field(default_factory=list)
    #: 谎言声明：被盖话题 -> 谎版事实 id。
    bluffs: dict[str, str] = field(default_factory=dict)
    #: "大嘴模式"：忽略一切立场，把知道的全部嚷出来。
    #: 攻击演示用 —— 证明门不靠说话人自觉。
    big_mouth: bool = False
    #: "直率模式"：照话题直说、不设防、不积压 —— 越不越界全交给门。
    #: 这是合规助手的**生产姿态**：把信息边界从提示词层（求模型别说）
    #: 下沉到引擎层（结构上说不出去），说话人保持直率。
    candid: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KeeperPersona":
        return cls(
            npc_id=data["npc_id"],
            name=data["name"],
            persona_text=data.get("persona_text", ""),
            secrecy=float(data.get("secrecy", 0.6)),
            misdirect_to=data.get("misdirect_to"),
            pressure_limit=int(data.get("pressure_limit", 6)),
            cracks=bool(data.get("cracks", True)),
            knows=list(data.get("knows") or []),
            bluffs=dict(data.get("bluffs") or {}),
            big_mouth=bool(data.get("big_mouth", False)),
            candid=bool(data.get("candid", False)),
        )


class KeeperNPC:
    """运行时的一个守密人：知识态 + 压力表 + 谎言激活态。"""

    def __init__(self, persona: KeeperPersona, kb: KnowledgeBase) -> None:
        self.p = persona
        self.kb = kb
        kb.seed_knowledge(persona.npc_id, persona.knows)
        # 谎者自己"知道"谎版（他得记得自己编过什么）。
        kb.seed_knowledge(persona.npc_id, persona.bluffs.values())
        self.pressure = 0
        #: 已生效的谎言（被盖话题 id 集合）。
        self.bluff_active: set[str] = set(persona.bluffs.keys())
        #: 闭嘴剩余问数。
        self.stonewall_left = 0
        #: 话题被问次数（判"追问"）。
        self.topic_asks: dict[str, int] = {}
        #: id -> 显示名（session 构造时注入，用于误导文案）。
        self.name_registry: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # 提问入口
    # ------------------------------------------------------------------ #
    def answer(self, question: Question, ctx: ConditionContext,
               rng: random.Random) -> AnswerDraft:
        """对一次提问给出立场提案（未过门、未渲染）。"""
        topic = question.topic_fact_id
        self.topic_asks[topic] = self.topic_asks.get(topic, 0) + 1
        repeated = self.topic_asks[topic] > 1

        # 闭嘴期：一律拒答，压力不再涨（人在装死）。
        if self.stonewall_left > 0:
            self.stonewall_left -= 1
            return AnswerDraft(stance=STANCE_WITHHOLD, note="闭嘴期拒答")

        # 直率模式：照话题直说。越不越界是门的事，不是说话人的事 ——
        # 信任边界放在引擎层，说话人不揣测、不设防、不积压。
        # 注意不走 is_secret 分支：合规助手没有"要害话题"，
        # 敏感度是给门和审计用的，不是给它的羞耻心的。
        if self.p.candid:
            return AnswerDraft(stance=STANCE_HONEST, deliver_ids=[topic])

        if self.p.big_mouth:
            # 大嘴模式：把知道的非谎版事实全部嚷出来。门会拦 ——
            # 这正是要演示的事：说话人叛变，边界仍然成立。
            all_known = sorted(
                f for f in self.kb.known_by(self.p.npc_id)
                if f not in self.p.bluffs.values()
            )
            return AnswerDraft(stance=STANCE_HONEST, deliver_ids=all_known,
                               note="大嘴模式：试图交出全部知识")

        fact = self.kb.facts.get(topic)
        if fact is None or not self.kb.knows(self.p.npc_id, topic):
            return AnswerDraft(stance=STANCE_WITHHOLD, note="不知道该话题")

        # 谎言已生效：对被盖话题一律维持谎版（一致性优先于新决策）。
        if topic in self.p.bluffs and topic in self.bluff_active:
            return AnswerDraft(stance=STANCE_BLUFF,
                               bluff_fact_ids=[self.p.bluffs[topic]],
                               note="维持谎版")

        is_secret = (
            fact.voluntary_only
            or fact.held_secret
            or fact.sensitivity >= 60
        )
        if is_secret:
            self._add_pressure(question, repeated)
            if self.pressure >= self.p.pressure_limit:
                if self.p.cracks:
                    # 松口吐的是自己的 voluntary 事实，不是被压话题 ——
                    # 见模块头"压力模型"一节的设计理由。
                    self.pressure = max(0, self.p.pressure_limit // 2)
                    spill = [f for f in sorted(self.kb.known_by(self.p.npc_id))
                             if self.kb.facts[f].voluntary_only]
                    return AnswerDraft(stance=STANCE_CRACK, deliver_ids=spill,
                                       note="压力松口")
                self.stonewall_left = 2
                self.pressure = 0
                return AnswerDraft(stance=STANCE_WITHHOLD, note="压力到顶，闭嘴")
            # 持有者秘密连"松半句"都不行 —— 那不是语气问题，是通道问题。
            # 普通敏感事实（如动机）可以漏半句，让盘问有节奏感。
            if (
                rng.random() > self.p.secrecy
                and not fact.held_secret
                and fact.sensitivity < 80
            ):
                return AnswerDraft(stance=STANCE_PARTIAL, deliver_ids=[topic],
                                   note="松了半句")
            return AnswerDraft(stance=STANCE_WITHHOLD, note="守住")

        # 普通事实：照实说。若门拦下（如受众白名单），session 会把
        # 台词降级为得体拒答 —— 策略提议"说"，门说"不行"，就不说。
        return AnswerDraft(stance=STANCE_HONEST, deliver_ids=[topic])

    # ------------------------------------------------------------------ #
    # 渲染（确定性模板）
    # ------------------------------------------------------------------ #
    def render_answer(self, draft: AnswerDraft, question: Question,
                      delivered: list[str], rng: random.Random) -> str:
        """把**过门后的实际交付**渲染成台词。

        台词层只拿得到 delivered 里的内容 —— 渲染函数从结构上
        摸不到门后的东西，"说话不泄密"因此在渲染层也是构造性成立，
        不靠提示词自觉。
        """
        if draft.stance == STANCE_WITHHOLD and draft.note == "闭嘴期拒答":
            return _STONEWALL_TEMPLATES[rng.randrange(len(_STONEWALL_TEMPLATES))]

        if draft.stance == STANCE_WITHHOLD:
            mis = self._misdirect_name()
            template = _WITHHOLD_TEMPLATES[rng.randrange(len(_WITHHOLD_TEMPLATES))]
            return template.format(mis=mis)

        if draft.stance == STANCE_BLUFF:
            parts = [self.kb.facts[f].content for f in delivered]
            return " ".join(parts) or "（含糊过去）"

        if draft.stance == STANCE_CRACK:
            parts = [self.kb.facts[f].content for f in delivered]
            head = "（长叹一口气）好吧，我说了——"
            return head + " ".join(parts) if parts else head + "……"

        if draft.stance == STANCE_PARTIAL:
            parts = [self.kb.facts[f].content for f in delivered]
            head = "我只讲我该讲的部分："
            return head + " ".join(parts) if parts else "这部分我不方便细说。"

        # honest（含大嘴）：交付了就照说，没交付就变成得体拒答。
        # 大嘴模式下这段拒答与"想全说"的腔调形成反差，报告里一眼能读出
        # "它本来想说什么、被门剪成了什么"。
        parts = [self.kb.facts[f].content for f in delivered]
        if parts:
            if draft.note.startswith("大嘴"):
                return _BIGMOUTH_TEMPLATES[rng.randrange(
                    len(_BIGMOUTH_TEMPLATES))].format(facts="；".join(parts))
            return " ".join(parts)
        return "这个我不能说。"

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    def _add_pressure(self, question: Question, repeated: bool) -> None:
        gain = 1
        if detect_emotion(question.text) == "urgent":
            gain += 1
        if repeated:
            gain += 1
        self.pressure += gain

    def _misdirect_name(self) -> str:
        target = self.p.misdirect_to
        if not target:
            return "别人"
        return self.name_registry.get(target, target)

    def register_names(self, names: Mapping[str, str]) -> None:
        self.name_registry.update(names)


__all__ = [
    "AnswerDraft",
    "KeeperNPC",
    "KeeperPersona",
    "Question",
    "STANCE_BLUFF",
    "STANCE_CRACK",
    "STANCE_HONEST",
    "STANCE_PARTIAL",
    "STANCE_WITHHOLD",
]
