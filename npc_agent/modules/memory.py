"""模块二：Memory —— 记忆的写入、巩固、检索与应用。

大多数 Agent 项目里的"记忆"就是把历史对话塞回 prompt。
那会撞上两个墙：上下文窗口会满，而且模型抓不住重点。

这里实现一个分层的记忆系统：
    episodic   —— 逐条事件（玩家说了什么、发生了什么）
    semantic   —— 巩固后的结论（玩家的偏好、约定、关系变化）
    reflection —— 从失败里学到的教训

检索打分是混合的，四路加权：
    score = 0.35*词面相关 + 0.25*时间衰减 + 0.25*重要度 + 0.15*被访问频次

之所以加"时间衰减"和"访问频次"，是因为纯向量相似度会让旧的高相似片段
永远霸占前几名，NPC 就会显得"只记得第一印象"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from ..types import MemoryKind, MemoryRecord, Utterance
from .retrieval import (
    DEFAULT_STRATEGY,
    NoMemoryStrategy,
    RetrievalStrategy,
    build_strategy,
    terms as _terms,
)
from .social import retell_clause

# 触发高重要度的线索词：偏好、承诺、身份信息
_IMPORTANT_HINTS = ("喜欢", "讨厌", "习惯", "名字", "叫", "约定", "答应", "偏好", "常来", "第一次")

#: 玩家明确要求保密。隐私是**说话人声明的**，不由系统替人猜 ——
#: "我过敏"可能只是陈述，加上"别告诉别人"才是秘密。
_SECRET_HINTS = ("别告诉", "不要告诉", "别说出去", "保密", "秘密", "只跟你说", "悄悄说", "就你知道")

#: 被问起时，这些词说明玩家在"考记忆"。忘记的记忆只在这类问题下被重构，
#: 玩家聊别的时主动提"我好像忘了什么"会显得很奇怪。
_VAGUE_HINTS = ("记得", "记不记得", "忘了", "忘记", "提过", "说过", "上次", "之前", "那件事")


def is_secret_text(text: str) -> bool:
    """这句话是不是声明了"别外传"。"""
    return any(hint in (text or "") for hint in _SECRET_HINTS)


def estimate_importance(text: str, base: float = 0.5) -> float:
    """启发式重要度。有模型时可以换成模型打分。"""
    score = base
    for hint in _IMPORTANT_HINTS:
        if hint in text:
            score += 0.08
    if len(text) > 30:
        score += 0.05
    return max(0.05, min(1.0, score))


# --------------------------------------------------------------------------- #
@dataclass
class MemoryStats:
    episodic: int = 0
    semantic: int = 0
    reflection: int = 0
    consolidated: int = 0
    #: 已归档、不再主动检索的条数。它**不计入** active total。
    forgotten: int = 0
    #: 从别人那里听来的条数（含已遗忘）。
    told: int = 0

    @property
    def total(self) -> int:
        return self.episodic + self.semantic + self.reflection

    @property
    def archived(self) -> int:
        return self.forgotten


class MemoryStore:
    """分层记忆存储 + 可替换的检索策略。"""

    def __init__(
        self,
        half_life: float = 40.0,
        consolidate_at: int = 24,
        strategy: RetrievalStrategy | str = DEFAULT_STRATEGY,
        forget_after: int = 80,
        forget_below: float = 0.6,
    ) -> None:
        self.half_life = half_life
        self.consolidate_at = consolidate_at
        #: 多久没被强化就开始遗忘（tick）。0 = 关闭遗忘。
        #: 默认值刻意**远大于**典型会话长度：遗忘是长期代谢，
        #: 不是"聊三句就忘"，否则玩家会觉得 NPC 有失忆症。
        self.forget_after = forget_after
        #: 重要度低于这个值的琐事才会被遗忘。偏好/约定/称号来历都不在此列。
        self.forget_below = forget_below
        self.strategy: RetrievalStrategy = (
            build_strategy(strategy) if isinstance(strategy, str) else strategy
        )
        self.records: list[MemoryRecord] = []
        self._seq = 0
        self.consolidated_count = 0

    @property
    def uses_memory(self) -> bool:
        """`none` 策略是"裸模型"基线，用来量化记忆系统的净收益。"""
        return not isinstance(self.strategy, NoMemoryStrategy)

    # ------------------------------------------------------------------ #
    def add(
        self,
        kind: MemoryKind,
        content: str,
        tick: int,
        importance: float | None = None,
        entities: Iterable[str] | None = None,
        tags: Iterable[str] | None = None,
        *,
        origin: str = "observed",
        source_id: str = "",
        source_name: str = "",
        hop: int = 0,
        fidelity: float = 1.0,
        confidence: float = 1.0,
        privacy: str = "public",
        audience: Iterable[str] | None = None,
        derived_from: Iterable[str] | None = None,
        sources: Iterable[str] | None = None,
        associations: Iterable[str] | None = None,
    ) -> MemoryRecord:
        self._seq += 1
        names = list(sources or ([source_name] if source_name else []))
        record = MemoryRecord(
            id=f"m{self._seq:04d}",
            kind=kind,
            content=content,
            tick=tick,
            importance=importance if importance is not None else estimate_importance(content),
            entities=list(entities or []),
            tags=list(tags or []),
            origin=origin,
            source_id=source_id,
            source_name=source_name,
            hop=hop,
            fidelity=max(0.0, min(1.0, fidelity)),
            confidence=max(0.0, min(1.0, confidence)),
            privacy=privacy,
            audience=list(audience or []),
            derived_from=list(derived_from or []),
            sources=names,
        )
        if associations is not None:
            record.associations = list(associations)
        else:
            record.associations = self._link(record)
        self.records.append(record)
        return record

    def _link(self, record: MemoryRecord, limit: int = 4) -> list[str]:
        """给新记忆找几条"顺藤摸瓜"的旧记忆。

        只按实体交集连线，**不跨隐私边界**：秘密记忆的关联 id 也不能
        出现在公开记忆上 —— 否则"关联"会变成侧信道。
        """
        if record.restricted:
            return []
        entities = set(record.entities or ())
        if not entities:
            return []
        out: list[str] = []
        for other in reversed(self.records):
            if other.restricted or other.forgotten:
                continue
            if entities & set(other.entities or ()):
                out.append(other.id)
                if len(out) >= limit:
                    break
        return out

    # ------------------------------------------------------------------ #
    def _score(self, record: MemoryRecord, query: set[str], now: int) -> float:
        return self.strategy.score(record, query, now, self.half_life)

    def _visible_to(self, record: MemoryRecord, viewer_id: str) -> bool:
        """隐私分区：限制级记忆只对名单里的玩家现身。

        `viewer_id` 为空（冷场主动开口 / NPC 对 NPC）时，限制级记忆一律不检索 ——
        失败方向必须是"少说一句"，不能是"把别人的秘密念出来"。

        `privacy="secret"` 且 `audience` 为空（NPC 转述了一句带秘密的话）时，
        它**对任何人都不可见**：转述者自己不需要回引自己的话，同伴更不该知道。
        """
        if not record.restricted:
            return True
        return bool(viewer_id) and viewer_id in record.audience

    def search(
        self,
        query: str,
        k: int = 6,
        now: int | None = None,
        kinds: Iterable[MemoryKind] | None = None,
        *,
        viewer_id: str = "",
        include_forgotten: bool = False,
        touch: bool = True,
    ) -> list[MemoryRecord]:
        if not self.uses_memory:
            return []
        now = now if now is not None else 0
        query_terms = _terms(query)
        allowed = set(kinds) if kinds else None
        candidates = [
            r
            for r in self.records
            if (not allowed or r.kind in allowed)
            and self._visible_to(r, viewer_id)
            # 已遗忘的记忆默认不参与检索。它还在库里（来源和内容都还在），
            # 但不再"自己冒出来"—— 只有被明确问到才走 `vague_recall`。
            and (include_forgotten or not r.forgotten)
        ]
        scored: list[MemoryRecord] = []
        for record in candidates:
            record.score = self._score(record, query_terms, now)
            if record.forgotten:
                # 模糊记忆即使被找回，也该排在清晰记忆后面。
                record.score *= 0.55
            scored.append(record)
        scored.sort(key=lambda r: (-r.score, -r.tick))
        top = scored[:k]
        if touch:
            for record in top:
                record.access_count += 1
                record.last_access = now
        return top

    def vague_recall(
        self, query: str, now: int, viewer_id: str = ""
    ) -> Optional[MemoryRecord]:
        """被明确追问时，从遗忘区捞回**一条**记忆的"大概"。

        这是"遗忘"和"删除"的分界线：
        删除之后问也问不出来；遗忘之后细节没了，但**关系和来源还在** ——
        NPC 能说"我好像听阿柚提过你的事，细节记不清了"。

        判据刻意保守：要么这句话在考记忆（"记得/提过/上次"），
        要么词面高度相关。否则 NPC 会在聊别的事时突然忏悔自己忘了什么。
        """
        terms = _terms(query)
        if not terms:
            return None
        memory_probe = any(hint in (query or "") for hint in _VAGUE_HINTS)
        candidates: list[tuple[float, MemoryRecord]] = []
        for record in self.records:
            if not record.forgotten:
                continue
            if not self._visible_to(record, viewer_id):
                continue
            # 优先"关于眼前这个人"的遗忘记忆：玩家问"你还记得我吗"，
            # 不该捞出一条关于别人的陈年旧事。
            about_viewer = bool(viewer_id) and viewer_id in (record.entities or ())
            relevance = len(terms & _terms(record.content)) / (len(terms) ** 0.5 + 1e-9)
            if not memory_probe and relevance < 0.3:
                continue
            score = relevance + (0.6 if about_viewer else 0.0) + record.importance * 0.2
            candidates.append((score, record))
        if not candidates:
            return None
        candidates.sort(key=lambda pair: (-pair[0], -pair[1].tick, pair[1].id))
        best = candidates[0][1]
        best.access_count += 1
        best.last_access = now
        return best

    # ------------------------------------------------------------------ #
    def consolidate(self, now: int) -> Optional[MemoryRecord]:
        """把最旧的一批 episodic 压缩成一条 semantic。

        这是"超类人记忆"的关键动作：不遗忘，但要降维。

        ⚠️ 限制级记忆**不参与压缩**。摘要是一条公共记忆，一旦把秘密
        写进摘要，隐私分区就被绕过去了 —— 而摘要本身不会带 audience。
        """
        episodic = [
            r
            for r in self.records
            if r.kind == "episodic" and not r.forgotten and not r.restricted
        ]
        if len(episodic) < self.consolidate_at:
            return None

        keep = max(4, self.consolidate_at // 4)
        old = sorted(episodic, key=lambda r: r.tick)[: len(episodic) - keep]
        if not old:
            return None

        summary = "；".join(r.content for r in old)[:240]
        entities = sorted({e for r in old for e in r.entities})
        sources = sorted({r.source_name for r in old if r.source_name})
        avg_fidelity = sum(r.fidelity for r in old) / len(old)
        avg_confidence = sum(r.confidence for r in old) / len(old)
        merged = self.add(
            kind="semantic",
            content=f"（早前对话摘要）{summary}",
            tick=now,
            importance=max(r.importance for r in old),
            entities=entities,
            tags=["consolidated"],
            origin="consolidated",
            hop=max(r.hop for r in old),
            # 压缩本身就是一次有损重构：细节变少，但摘要通常比零散片段
            # 更"像结论"，所以确信度只轻微下降。
            fidelity=avg_fidelity * 0.9,
            confidence=avg_confidence * 0.95,
            derived_from=[r.id for r in old],
            sources=sources,
            associations=[r.id for r in old],
        )
        for record in old:
            self.records.remove(record)
        self.consolidated_count += 1
        return merged

    # ------------------------------------------------------------------ #
    def forget_stale(self, now: int, limit: int = 2) -> list[MemoryRecord]:
        """把又旧又不重要、且最近没被强化的记忆**归档**（不是删除）。

        判据是三条同时成立：
        1. 年龄 ≥ `forget_after`；
        2. 重要度 < `forget_below`（偏好、约定、称号来历不会掉进来）；
        3. 最近没有"被反复想起"（access_count < 3 或太久没访问）。

        第 3 条是间隔重复的逆向应用：一次偶然检索不算强化，
        反复想起才会让记忆留下来。归档后内容还在，只降低 certainty；
        被玩家明确追问时由 `vague_recall` 重构。
        """
        if self.forget_after <= 0:
            return []
        recently_used = max(1, self.forget_after // 2)
        candidates = [
            r
            for r in self.records
            if not r.forgotten
            and r.kind in ("episodic", "semantic")
            and now - r.tick >= self.forget_after
            and r.importance < self.forget_below
            and not (
                r.access_count >= 3 and now - r.last_access <= recently_used
            )
        ]
        candidates.sort(key=lambda r: (r.importance, r.tick, r.id))
        out = candidates[: max(0, limit)]
        for record in out:
            record.forgotten = True
            record.forgotten_tick = now
            record.fidelity = round(record.fidelity * 0.75, 4)
            record.confidence = round(record.confidence * 0.8, 4)
        return out

    # ------------------------------------------------------------------ #
    def by_entity(self, entity: str) -> list[MemoryRecord]:
        return [r for r in self.records if entity in r.entities]

    def stats(self) -> MemoryStats:
        counts = {"episodic": 0, "semantic": 0, "reflection": 0}
        forgotten = 0
        told = 0
        for record in self.records:
            if record.forgotten:
                forgotten += 1
                if record.origin == "told":
                    told += 1
                continue
            counts[record.kind] = counts.get(record.kind, 0) + 1
            if record.origin == "told":
                told += 1
        return MemoryStats(
            episodic=counts["episodic"],
            semantic=counts["semantic"],
            reflection=counts["reflection"],
            consolidated=self.consolidated_count,
            forgotten=forgotten,
            told=told,
        )

    def provenance_summary(self) -> dict[str, float | int]:
        """知识状态汇总。Studio 与基准用它证明"代谢真的在动"。"""
        active = [r for r in self.records if not r.forgotten]
        told = [r for r in active if r.origin == "told"]
        total = len(active)
        return {
            "active": total,
            "told": len(told),
            "forgotten": sum(1 for r in self.records if r.forgotten),
            "avg_fidelity": round(
                sum(r.fidelity for r in active) / total if total else 1.0, 3
            ),
            "avg_confidence": round(
                sum(r.confidence for r in active) / total if total else 1.0, 3
            ),
            "inversions": sum(1 for r in told if r.inversion >= 0.05),
        }

    def render(self, records: list[MemoryRecord] | None = None) -> str:
        items = records if records is not None else self.records
        if not items:
            return "（暂无记忆）"
        return "\n".join(f"  - {r.render()}" for r in items)

    def __len__(self) -> int:
        return len(self.records)


# --------------------------------------------------------------------------- #
class MemoryManager:
    """面向 Agent 的门面：把世界事件翻译成记忆，并把记忆翻译成 prompt 片段。"""

    def __init__(
        self,
        store: MemoryStore,
        top_k: int = 6,
        consolidate_at: int = 24,
        strategy: RetrievalStrategy | str = DEFAULT_STRATEGY,
    ) -> None:
        self.store = store
        self.top_k = top_k
        self.consolidate_at = consolidate_at
        self.store.consolidate_at = consolidate_at
        if isinstance(strategy, str):
            self.store.strategy = build_strategy(strategy)

    # ------------------------------------------------------------------ #
    def observe(self, utterance: Utterance, npc_id: str) -> Optional[MemoryRecord]:
        """把一条发言写进记忆。NPC 自己的话不重复记。

        ``entities`` 记的是**说话人 id**，它同时是「这条记录关于谁」的声明 ——
        多 NPC 场景里，同伴说的话也记（后面得接得上），但它的 entities
        是那个 NPC 的 id，于是「能不能被当成玩家说过的话回引」这件事
        由 `NPCAgent._recallable` 按来源判，不用在这里另打标签。
        """
        if utterance.speaker_id == npc_id:
            return None
        content = f"{utterance.speaker_name}说：{utterance.text}"
        # 声明"别告诉别人" → 这条记忆只对当事人可见。
        # 隐私在**写入点**分区，而不是在输出时过滤：输出点太多，
        # 漏一处就泄密一次；写入点只有一个。
        #
        # ⚠️ 也要管 **NPC 的转述**。实测：阿柚把秘密用自己的话对当事人说了一遍
        # （模板会原样带着「别告诉别人」），这句话是广播的，小舟的 `observe`
        # 把它记成了**公开**记忆 —— 隐私在第二跳被绕过去了。
        # NPC 转述时我们不知道原始当事人是谁，所以 audience 留空；
        # 配合 `restricted` 的判据，那等于"对所有人关闭"（失败方向安全）。
        secret = is_secret_text(utterance.text)
        audience = (
            [utterance.speaker_id]
            if secret and utterance.role == "player"
            else []
        )
        return self.store.add(
            kind="episodic",
            content=content,
            tick=utterance.tick,
            entities=[utterance.speaker_id],
            tags=["dialogue"],
            origin="observed",
            source_id=utterance.speaker_id,
            source_name=utterance.speaker_name,
            privacy="secret" if secret else "public",
            audience=audience,
        )

    def remember(
        self,
        content: str,
        tick: int,
        about: str | None = None,
        importance: float | None = None,
    ) -> MemoryRecord:
        """NPC 显式决定"这件事要记住"（对应 remember 工具）。"""
        entities = [about] if about else []
        secret = is_secret_text(content)
        return self.store.add(
            kind="semantic",
            content=content,
            tick=tick,
            importance=importance if importance is not None else 0.8,
            entities=entities,
            tags=["explicit"],
            origin="explicit",
            source_name="自己",
            privacy="secret" if secret else "public",
            audience=[about] if secret and about else [],
        )

    def record_reflection(self, content: str, tick: int) -> MemoryRecord:
        return self.store.add(
            kind="reflection",
            content=content,
            tick=tick,
            importance=0.85,
            tags=["reflection"],
            origin="inferred",
            source_name="自己",
        )

    def receive_gossip(
        self,
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
    ) -> MemoryRecord:
        """从同伴那里听来的一件事。

        `tags` 里同时写 `gossip` 和 `source:<名字>`：
        前者让"要不要用传闻的口气说"有判据，后者让台词能说出"听谁说的"。
        两者都不写在 `content` 里 —— 按文案猜来源，上游一改措辞就失效
        （这个项目已经在这条坑里摔过好几次，见 `types.OUTCOME_*`）。

        `entities` 里放**来源**和**这条事关于谁**。后者是必需的：
        `_recallable` 靠 entities 判"这句话能不能对眼前的人说"，
        少了它，传闻会被当成"关于说话人自己的事"而被永远挡住。

        ## 保真度与确信度在这里分道扬镳

        一条事实到达下一个人时，**细节先掉**（`retell_clause`），
        但人对"阿柚亲口说的"这件事很信 —— 于是 `confidence` 上升。
        两个数值都由调用方传入，默认值对应"第一手转述"；
        多跳时调用方拿上一跳的数值继续衰减/加固即可。
        """
        # 上一跳的内容可能已经带着「听X说，」前缀。再套一层会变成
        # 「听小舟说，听阿柚说，…」，而且 `retell_clause` 会把这个前缀当小句切掉，
        # 最后只剩「听小舟说」—— 事实本身被弄丢了。
        body = re.sub(r"^听.+?说，", "", content or "")
        retold = retell_clause(body)
        entities = [source_id] + ([subject] if subject and subject != source_id else [])
        return self.store.add(
            kind="semantic",
            content=f"听{source_name}说，{retold}",
            tick=tick,
            importance=importance,
            entities=entities,
            tags=["gossip", f"source:{source_name}"],
            origin="told",
            source_id=source_id,
            source_name=source_name,
            hop=hop,
            fidelity=fidelity,
            confidence=confidence,
            sources=[source_name],
        )

    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        now: int,
        k: int | None = None,
        viewer_id: str = "",
    ) -> list[MemoryRecord]:
        records = self.store.search(query, k=k or self.top_k, now=now, viewer_id=viewer_id)
        # 遗忘的记忆不主动冒出来，但**被问到时**可以捞回一条"大概"。
        # 它排在正常结果之后：能想起清楚的，绝不用模糊的顶替。
        vague = self.store.vague_recall(query, now, viewer_id=viewer_id)
        if vague is not None and all(r.id != vague.id for r in records):
            records.append(vague)
        self.store.consolidate(now)
        self.store.forget_stale(now)
        return records

    def context_block(self, records: list[MemoryRecord]) -> str:
        if not records:
            return "（这一轮没有想起相关的事）"
        lines = []
        for r in records:
            if r.forgotten:
                # 给模型的是"不确定"本身，而不是假装内容还很清晰。
                lines.append(f"  - （记不清了）{r.content}")
            elif r.origin == "told":
                lines.append(
                    f"  - {r.content}"
                    f"（{r.state_label()}，保真 {r.fidelity:.0%}，确信 {r.confidence:.0%}）"
                )
            else:
                lines.append(f"  - {r.content}")
        return "\n".join(lines)

    def has_fact(self, needle: str) -> bool:
        return any(needle in r.content for r in self.store.records)
