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

from dataclasses import dataclass, field
from typing import Iterable, Optional

from ..types import MemoryKind, MemoryRecord, Utterance
from .retrieval import (
    DEFAULT_STRATEGY,
    NoMemoryStrategy,
    RetrievalStrategy,
    build_strategy,
    terms as _terms,
)

# 触发高重要度的线索词：偏好、承诺、身份信息
_IMPORTANT_HINTS = ("喜欢", "讨厌", "习惯", "名字", "叫", "约定", "答应", "偏好", "常来", "第一次")


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

    @property
    def total(self) -> int:
        return self.episodic + self.semantic + self.reflection


class MemoryStore:
    """分层记忆存储 + 可替换的检索策略。"""

    def __init__(
        self,
        half_life: float = 40.0,
        consolidate_at: int = 24,
        strategy: RetrievalStrategy | str = DEFAULT_STRATEGY,
    ) -> None:
        self.half_life = half_life
        self.consolidate_at = consolidate_at
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
    ) -> MemoryRecord:
        self._seq += 1
        record = MemoryRecord(
            id=f"m{self._seq:04d}",
            kind=kind,
            content=content,
            tick=tick,
            importance=importance if importance is not None else estimate_importance(content),
            entities=list(entities or []),
            tags=list(tags or []),
        )
        self.records.append(record)
        return record

    # ------------------------------------------------------------------ #
    def _score(self, record: MemoryRecord, query: set[str], now: int) -> float:
        return self.strategy.score(record, query, now, self.half_life)

    def search(
        self,
        query: str,
        k: int = 6,
        now: int | None = None,
        kinds: Iterable[MemoryKind] | None = None,
    ) -> list[MemoryRecord]:
        if not self.uses_memory:
            return []
        now = now if now is not None else 0
        query_terms = _terms(query)
        allowed = set(kinds) if kinds else None
        candidates = [r for r in self.records if not allowed or r.kind in allowed]
        scored: list[MemoryRecord] = []
        for record in candidates:
            record.score = self._score(record, query_terms, now)
            scored.append(record)
        scored.sort(key=lambda r: (-r.score, -r.tick))
        top = scored[:k]
        for record in top:
            record.access_count += 1
            record.last_access = now
        return top

    # ------------------------------------------------------------------ #
    def consolidate(self, now: int) -> Optional[MemoryRecord]:
        """把最旧的一批 episodic 压缩成一条 semantic。

        这是"超类人记忆"的关键动作：不遗忘，但要降维。
        """
        episodic = [r for r in self.records if r.kind == "episodic"]
        if len(episodic) < self.consolidate_at:
            return None

        keep = max(4, self.consolidate_at // 4)
        old = sorted(episodic, key=lambda r: r.tick)[: len(episodic) - keep]
        if not old:
            return None

        summary = "；".join(r.content for r in old)[:240]
        entities = sorted({e for r in old for e in r.entities})
        merged = self.add(
            kind="semantic",
            content=f"（早前对话摘要）{summary}",
            tick=now,
            importance=max(r.importance for r in old),
            entities=entities,
            tags=["consolidated"],
        )
        for record in old:
            self.records.remove(record)
        self.consolidated_count += 1
        return merged

    # ------------------------------------------------------------------ #
    def by_entity(self, entity: str) -> list[MemoryRecord]:
        return [r for r in self.records if entity in r.entities]

    def stats(self) -> MemoryStats:
        counts = {"episodic": 0, "semantic": 0, "reflection": 0}
        for record in self.records:
            counts[record.kind] = counts.get(record.kind, 0) + 1
        return MemoryStats(
            episodic=counts["episodic"],
            semantic=counts["semantic"],
            reflection=counts["reflection"],
            consolidated=self.consolidated_count,
        )

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
        """把一条玩家发言写进记忆。NPC 自己的话不重复记。"""
        if utterance.speaker_id == npc_id:
            return None
        content = f"{utterance.speaker_name}说：{utterance.text}"
        return self.store.add(
            kind="episodic",
            content=content,
            tick=utterance.tick,
            entities=[utterance.speaker_id],
            tags=["dialogue"],
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
        return self.store.add(
            kind="semantic",
            content=content,
            tick=tick,
            importance=importance if importance is not None else 0.8,
            entities=entities,
            tags=["explicit"],
        )

    def record_reflection(self, content: str, tick: int) -> MemoryRecord:
        return self.store.add(
            kind="reflection", content=content, tick=tick, importance=0.85, tags=["reflection"]
        )

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, now: int, k: int | None = None) -> list[MemoryRecord]:
        records = self.store.search(query, k=k or self.top_k, now=now)
        self.store.consolidate(now)
        return records

    def context_block(self, records: list[MemoryRecord]) -> str:
        if not records:
            return "（这一轮没有想起相关的事）"
        return "\n".join(f"  - {r.content}" for r in records)

    def has_fact(self, needle: str) -> bool:
        return any(needle in r.content for r in self.store.records)
