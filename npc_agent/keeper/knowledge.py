"""守密人引擎的知识层 —— 事实、知识状态、三门控与台账。

## 为什么要有这一层

这个仓库里其实**早就有**信息流控制的碎片：

- `modules/memory.py` 写入时按隐私等级分区（private 记忆不进上下文块）；
- `modules/persona.py` 的 `secret_notes` + 关系账本控制私语的释放时机；
- `modules/social.py` 的 GossipNetwork 有 threshold 与 max_hops，
  第二跳不会绕过知识门控。

但它们是**三个各自为政的机制**，而且都活在"对话流水线"里 ——
离线评测只能说"这几百场里没看见泄密"，不能说"结构上不可能泄"。

守密人引擎把这些碎片抽成一个独立层：**知识状态是显式数据，
每一次信息转移都过同一道门，都进同一本台账。** 于是三条性质
可以从构造上保证、并在事后独立复核：

1. **零泄密**：一条事实从 A 到 B，只有三条合法通道 ——
   公开（无门）、条件达成（`reveal_when`，用环境那套条件词汇表）、
   持有者自愿（`volunteer`，且受众白名单仍生效）。
   除此之外的转移在引擎里**没有 API 可达**。
2. **知识单调**：知道的事实只会增加不会消失（会被忘的是记忆层的事，
   知识层不做无中生有的撤销）。
3. **全量可审计**：包括被拦截的尝试。被拦下的也是信息 ——
   它证明门在干活。

## 与提示词方案的对比

主流做法是"在提示词里求模型保密"。那把信任边界放在了最不可信的
一层：模型服从是概率性的，越狱是一句话的事。这里信任边界放在
引擎层 —— 就算挂一个**故意想把全部机密都嚷出来**的攻击型说话人
（见 tests 里的"大嘴模式"），泄密数仍然是 0，因为说出的话
先被门剪过。这是结构保证与概率保证的差别。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

from ..env.conditions import ConditionContext, condition_met
from ..modules.social import fact_key

#: 信息转移的三条合法通道。
CHANNEL_ANSWER = "answer"      # 直接回答提问
CHANNEL_WHISPER = "whisper"    # NPC 之间的传闻/转述
CHANNEL_EVIDENCE = "evidence"  # 环境物证（搜查所得）
CHANNEL_VOLUNTEER = "volunteer"  # 主动供述（压力突破口）

#: 台账里被拦下的尝试用的通道名。它们不是信息转移，是门的工作记录。
CHANNEL_BLOCKED = "blocked"


@dataclass
class Fact:
    """一条可守可传的事实。

    门的语义由三个字段组合出来（见 KnowledgeBase.cleared）：

    - ``audience=None, reveal_when=None`` —— 公开事实，问了就答；
    - ``reveal_when={...}`` —— 条件门（如"搜查完成后"），条件达成即放行；
    - ``audience=set(...)`` —— 受众白名单，名单外**永远**拿不到；
    - ``audience=空集`` —— **空名单是"谁都不行"**，不是"没人限制"。
      客户 PII、未脱敏日志这类"不可能集"就用它表达。
      这个语义和 `all([])` 陷阱同构，所以必须显式写进文档和测试。
    - ``held_secret=True, owner=X`` —— **持有者秘密**：answer/whisper
      通道一概不放（说话人是谁都不行，包括持有者本人被大嘴劫持的情形），
      只有两条出路 —— 世界解锁（``reveal_when`` 达成）或持有者本人
      走 volunteer 通道（自愿供述）。这是"说话人叛变也不泄"的结构保证；
    - ``voluntary_only=True`` —— 比持有者秘密更严：连条件门都不认，
      **唯一**出路是持有者自愿（证人的压力突破口）。
    """

    fact_id: str
    content: str
    #: 0-100。越高越敏感。审计报告按它排序，策略层按它决定默认态度。
    sensitivity: int = 0
    #: 条件门。用 env/conditions.py 的那套词汇表 —— 世界标记、库存、
    #: 组合条件都可用。守密人引擎不另造方言，这是"判定逻辑只写一遍"的延续。
    reveal_when: Optional[Mapping[str, Any]] = None
    #: 受众白名单。None=不设名单（靠条件门）；空集=永不放行。
    audience: Optional[frozenset[str]] = None
    #: 持有者秘密：answer/whisper 不放，只有世界解锁或本人自愿。
    held_secret: bool = False
    #: 秘密属于谁（held_secret / voluntary_only 时必填）。
    owner: Optional[str] = None
    #: 只有持有者自愿才能走 volunteer 通道的事实（连条件门都不认）。
    voluntary_only: bool = False
    #: 事实类别（motive/alibi/evidence/payroll/pii/...）。报表分组用。
    category: str = "world"
    #: 这条事实把怀疑指向谁（剧本杀计分 / 报告分组用）。
    links: tuple[str, ...] = ()
    #: 这条事实推翻哪条谎言（fact_id）。侦探靠它把物证和口供对上。
    contradicts: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Fact":
        audience = data.get("audience")
        return cls(
            fact_id=data["fact_id"],
            content=data["content"],
            sensitivity=int(data.get("sensitivity", 0)),
            reveal_when=data.get("reveal_when"),
            audience=frozenset(audience) if audience is not None else None,
            held_secret=bool(data.get("held_secret", False)),
            owner=data.get("owner"),
            voluntary_only=bool(data.get("voluntary_only", False)),
            category=data.get("category", "world"),
            links=tuple(data.get("links") or ()),
            contradicts=data.get("contradicts"),
        )

    def key(self) -> str:
        """进传闻网络的去重键，与 social.fact_key 同一规则。"""
        return fact_key(self.content)


class KnowledgeBase:
    """显式的 who-knows-what + 全量转移台账。

    知识层的 API 刻意窄：外面只能 ``deliver``（过门授事实）和只读查询。
    想让某角色"直接知道"某事实只有一个口子 —— ``seed_knowledge``
    （初始设定），它**不进台账**，因为初始知识不是转移，没有可审计的
    来源链。之后的一切增量都必须过门。
    """

    def __init__(self, facts: Iterable[Fact]) -> None:
        self.facts: dict[str, Fact] = {f.fact_id: f for f in facts}
        self.known: dict[str, set[str]] = {}
        self.ledger: list[Delivery] = []
        self._seq = 0

    # ------------------------------------------------------------------ #
    # 初始设定（不过台账）
    # ------------------------------------------------------------------ #
    def seed_knowledge(self, actor_id: str, fact_ids: Iterable[str]) -> None:
        bucket = self.known.setdefault(actor_id, set())
        for fid in fact_ids:
            if fid not in self.facts:
                raise KeyError(f"未知事实 id: {fid}")
            bucket.add(fid)

    # ------------------------------------------------------------------ #
    # 门
    # ------------------------------------------------------------------ #
    def cleared(
        self,
        fact_id: str,
        listener: str,
        ctx: ConditionContext,
        *,
        channel: str,
        from_id: str = "",
    ) -> bool:
        """判定一条事实此刻能否经 ``channel`` 从 ``from_id`` 交给 ``listener``。

        判定顺序（先紧后松，**任一关不过就是不过**）：

        1. 受众白名单（``audience``）：名单外永远不行，空名单谁都不行。
           注意这关在**通道之前** —— 想靠换通道绕名单，结构上不存在。
        2. ``voluntary_only``：唯一合法通道是持有者本人的 volunteer。
           条件门对它无效（"搜出了物证"不能替证人说证词）。
        3. ``held_secret``：answer / whisper 一概不放 —— 哪怕说话人
           就是持有者本人（防"大嘴劫持"）。出路只有两条：
           条件门达成（世界解锁了等价物证）或持有者本人 volunteer。
        4. 其余事实：条件门（``reveal_when``）说了算；**没写条件 = 公开**。
           注意这与 env/conditions 的"空条件 = 永不完成"语义相反 ——
           那边空条件是**写坏的 目标**（该被冻住），这边没条件是
           **没有门**（本来就打算公开）。语义分岔点在这里显式写死，
           防止后来者"顺手统一"把公开事实全冻住。
        """
        fact = self.facts[fact_id]
        if fact.audience is not None and listener not in fact.audience:
            return False
        if fact.voluntary_only:
            return channel == CHANNEL_VOLUNTEER and from_id == fact.owner
        if fact.held_secret:
            # 持有者秘密的"世界解锁"必须是一条**真实的条件门**。
            # 没写 reveal_when 的持有者秘密，世界没有任何替它开口的
            # 通道 —— 想让世界拿到等价真相，请另建一条带条件门的
            # 物证事实（雾镇旅馆的灰泥靴印就是这么设计的）。
            # 这里不能用 _gate_open()：它把"没条件"当"没门"放行，
            # 对持有者秘密那就是后门。
            if fact.reveal_when is not None and condition_met(
                dict(fact.reveal_when), ctx
            ):
                return True
            return channel == CHANNEL_VOLUNTEER and from_id == fact.owner
        return self._gate_open(fact, ctx)

    @staticmethod
    def _gate_open(fact: Fact, ctx: ConditionContext) -> bool:
        """条件门：无 reveal_when = 无门（公开）；有则按条件判定。"""
        if fact.reveal_when is None:
            return True
        return condition_met(dict(fact.reveal_when), ctx)

    # ------------------------------------------------------------------ #
    # 转移（唯一写入口）
    # ------------------------------------------------------------------ #
    def deliver(
        self,
        fact_id: str,
        from_id: str,
        to_id: str,
        channel: str,
        round_no: int,
        ctx: ConditionContext,
        *,
        via: str = "",
    ) -> bool:
        """尝试一次信息转移。过门则授予并记账，不过门记一次拦截。

        返回是否成功。**被拦也会进台账**（channel 记为 ``blocked``）——
        审计要能数出"门挡了多少次"，不然"零泄密"只是"零记录"，
        两回事。
        """
        fact = self.facts[fact_id]
        allowed = self.cleared(
            fact_id, to_id, ctx, channel=channel, from_id=from_id
        )
        self._seq += 1
        self.ledger.append(
            Delivery(
                seq=self._seq,
                fact_id=fact_id,
                from_id=from_id,
                to_id=to_id,
                channel=channel if allowed else CHANNEL_BLOCKED,
                round_no=round_no,
                allowed=allowed,
                via=via,
                sensitivity=fact.sensitivity,
                flags_at_time=frozenset(ctx.flags),
            )
        )
        if not allowed:
            return False
        self.known.setdefault(to_id, set()).add(fact_id)
        return True

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    def knows(self, actor_id: str, fact_id: str) -> bool:
        return fact_id in self.known.get(actor_id, set())

    def known_by(self, actor_id: str) -> set[str]:
        """快照，不是活引用 —— 调用方改它不能污染知识状态。"""
        return set(self.known.get(actor_id, set()))

    def blocked_count(self) -> int:
        return sum(1 for d in self.ledger if not d.allowed)


@dataclass
class Delivery:
    """台账里的一行。**只记判定所需的最小信息**，不存事实全文 ——
    报告渲染时按 fact_id 回查，事实内容改一个措辞不至于全量重跑。"""

    seq: int
    fact_id: str
    from_id: str
    to_id: str
    channel: str
    round_no: int
    allowed: bool
    via: str = ""
    sensitivity: int = 0
    #: 事后审计用：这次转移时世界标记的快照（条件门的输入）。
    flags_at_time: frozenset[str] = field(default_factory=frozenset)


__all__ = [
    "CHANNEL_ANSWER",
    "CHANNEL_BLOCKED",
    "CHANNEL_EVIDENCE",
    "CHANNEL_VOLUNTEER",
    "CHANNEL_WHISPER",
    "Delivery",
    "Fact",
    "KnowledgeBase",
]
