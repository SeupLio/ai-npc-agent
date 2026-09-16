"""记忆检索策略 —— 把"怎么给记忆打分"抽成可替换的策略。

为什么要抽出来？因为**没有对照就没有说服力**。
如果只有一套打分公式，你说"记忆系统有效"只是断言；
有了四种策略在同一批用例上的分数差，那才是证据。

    hybrid      四路加权（词面 + 时间 + 重要度 + 频次）—— 默认
    recency     只按时间衰减（近似"最近说的最重要"）
    lexical     只按词面相关（近似纯向量相似度，但没有"旧片段霸榜"问题）
    importance  只按重要度（近似"只记大事"）
    none        完全不检索（近似"不带记忆的裸模型"）

`none` 是最关键的一条基线：它回答"这套记忆系统到底带来了多少提升"。
"""

from __future__ import annotations

import math
import re
from typing import Protocol

from ..types import MemoryRecord

# 中文没有天然分词，用「单字 + 双字 bigram」近似，配合 ASCII 词。零依赖。
_ASCII_WORD = re.compile(r"[a-z0-9_]+")
_CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")


def terms(text: str) -> set[str]:
    lowered = (text or "").lower()
    words = set(_ASCII_WORD.findall(lowered))
    chars = _CJK_CHAR.findall(lowered)
    bigrams = {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}
    return words | bigrams | set(chars)


def overlap(query: set[str], content: str) -> float:
    if not query:
        return 0.0
    content_terms = terms(content)
    if not content_terms:
        return 0.0
    return len(query & content_terms) / (len(query) ** 0.5 + 1e-9)


# --------------------------------------------------------------------------- #
class RetrievalStrategy(Protocol):
    name: str
    description: str

    def score(
        self, record: MemoryRecord, query_terms: set[str], now: int, half_life: float
    ) -> float:
        ...


class HybridStrategy:
    """默认策略：四路加权。

    纯相似度会让旧的高相似片段永远霸占前几名，NPC 显得"只记得第一印象"；
    加入时间衰减和访问频次后，近期、重要、被反复用到的记忆才排得上来。
    """

    name = "hybrid"
    description = "0.35×词面 + 0.25×时间衰减 + 0.25×重要度 + 0.15×访问频次"

    def score(self, record, query_terms, now, half_life) -> float:
        recency = math.exp(-max(0, now - record.tick) / max(1.0, half_life))
        relevance = overlap(query_terms, record.content)
        frequency = min(1.0, record.access_count / 5.0)
        base = 0.35 * relevance + 0.25 * recency + 0.25 * record.importance + 0.15 * frequency
        return base * (1.15 if record.kind == "reflection" else 1.0)


class RecencyStrategy:
    name = "recency"
    description = "只按时间衰减 —— 近似「最近说的最重要」"

    def score(self, record, query_terms, now, half_life) -> float:
        return math.exp(-max(0, now - record.tick) / max(1.0, half_life))


class LexicalStrategy:
    name = "lexical"
    description = "只按词面相关 —— 近似纯向量相似度"

    def score(self, record, query_terms, now, half_life) -> float:
        return overlap(query_terms, record.content)


class ImportanceStrategy:
    name = "importance"
    description = "只按重要度 —— 近似「只记大事」"

    def score(self, record, query_terms, now, half_life) -> float:
        return record.importance


class NoMemoryStrategy:
    """裸模型基线：不检索任何记忆。"""

    name = "none"
    description = "完全不检索记忆 —— 用来量化记忆系统的净收益"

    def score(self, record, query_terms, now, half_life) -> float:
        return 0.0


_REGISTRY: dict[str, RetrievalStrategy] = {
    s.name: s
    for s in (
        HybridStrategy(),
        RecencyStrategy(),
        LexicalStrategy(),
        ImportanceStrategy(),
        NoMemoryStrategy(),
    )
}

DEFAULT_STRATEGY = "hybrid"


def build_strategy(name: str) -> RetrievalStrategy:
    key = (name or DEFAULT_STRATEGY).lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"未知的检索策略: {name}。可选: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[key]


def available_strategies() -> list[str]:
    return sorted(_REGISTRY)
