"""检索策略的单元测试。

重点是**策略之间真的不一样**：如果四种策略在同一份数据上给出相同排序，
那这个抽象就是白做的，消融实验也就没有意义。
"""

from __future__ import annotations

import pytest

from npc_agent.modules.retrieval import (
    DEFAULT_STRATEGY,
    HybridStrategy,
    ImportanceStrategy,
    LexicalStrategy,
    NoMemoryStrategy,
    RecencyStrategy,
    available_strategies,
    build_strategy,
    overlap,
    terms,
)
from npc_agent.types import MemoryRecord


def rec(content: str, tick: int = 0, importance: float = 0.5, kind: str = "episodic",
        access_count: int = 0) -> MemoryRecord:
    return MemoryRecord(
        id=f"m{tick}-{content[:4]}",
        kind=kind,
        content=content,
        tick=tick,
        importance=importance,
        access_count=access_count,
    )


# --------------------------------------------------------------------------- #
# 分词与词面相关


def test_terms_mixes_cjk_bigram_and_ascii():
    got = terms("阿柚 latte 第一次")
    assert "latte" in got          # ASCII 词保留
    assert "阿柚" in got           # CJK bigram
    assert "阿" in got             # 单字兜底


def test_terms_on_empty_is_empty():
    assert terms("") == set()


def test_overlap_zero_for_empty_query():
    assert overlap(set(), "任何内容") == 0.0


def test_overlap_positive_for_matching_text():
    assert overlap(terms("第一次来"), "他说他第一次来这里") > 0


def test_overlap_zero_for_unrelated_text():
    assert overlap(terms("星座运势"), "拿铁咖啡的做法") == 0.0


# --------------------------------------------------------------------------- #
# 注册表


def test_available_strategies_lists_all_five():
    assert available_strategies() == ["hybrid", "importance", "lexical", "none", "recency"]


def test_build_strategy_known_names():
    for name in available_strategies():
        assert build_strategy(name).name == name


def test_build_strategy_defaults_when_blank():
    assert build_strategy("").name == DEFAULT_STRATEGY


def test_build_strategy_unknown_raises_with_helpful_message():
    with pytest.raises(ValueError) as excinfo:
        build_strategy("vector-db")
    assert "hybrid" in str(excinfo.value)  # 报错信息里要列可选值


# --------------------------------------------------------------------------- #
# 策略语义：每个策略只看它该看的东西


def test_recency_prefers_recent_regardless_of_relevance():
    strategy = RecencyStrategy()
    fresh_irrelevant = rec("完全无关的内容", tick=100)
    stale_relevant = rec("玩家说他第一次来", tick=1)
    query = terms("第一次来")
    assert strategy.score(fresh_irrelevant, query, 100, 40.0) > strategy.score(
        stale_relevant, query, 100, 40.0
    )


def test_lexical_ignores_time_entirely():
    strategy = LexicalStrategy()
    query = terms("第一次来")
    old = rec("他说他第一次来", tick=0)
    new = rec("他说他第一次来", tick=999)
    assert strategy.score(old, query, 999, 40.0) == pytest.approx(
        strategy.score(new, query, 999, 40.0)
    )


def test_importance_ignores_time_and_relevance():
    strategy = ImportanceStrategy()
    query = terms("第一次来")
    assert strategy.score(rec("无关", tick=0, importance=0.9), query, 999, 40.0) == pytest.approx(0.9)
    assert strategy.score(rec("第一次来", tick=999, importance=0.1), query, 999, 40.0) == pytest.approx(0.1)


def test_none_strategy_always_zero():
    strategy = NoMemoryStrategy()
    assert strategy.score(rec("第一次来", tick=0, importance=1.0), terms("第一次来"), 0, 40.0) == 0.0


def test_hybrid_reflection_gets_boost():
    strategy = HybridStrategy()
    query = terms("点单")
    plain = rec("点单要先去吧台", tick=5, importance=0.5)
    lesson = rec("点单要先去吧台", tick=5, importance=0.5, kind="reflection")
    assert strategy.score(lesson, query, 5, 40.0) > strategy.score(plain, query, 5, 40.0)


def test_hybrid_balances_relevance_and_recency():
    """hybrid 的关键性质：高分既需要相关，也需要不太旧。"""
    strategy = HybridStrategy()
    query = terms("第一次来")
    relevant_recent = rec("他说他第一次来", tick=10, importance=0.6)
    relevant_ancient = rec("他说他第一次来", tick=-10000, importance=0.6)
    assert strategy.score(relevant_recent, query, 10, 40.0) > strategy.score(
        relevant_ancient, query, 10, 40.0
    )


# --------------------------------------------------------------------------- #
# 策略之间必须真的不同


def test_strategies_disagree_on_a_deliberate_dataset():
    """构造一组"各策略会选出不同赢家"的数据，验证抽象不是摆设。"""
    query = terms("第一次来")
    recent_irrelevant = rec("今天天气不错", tick=100, importance=0.3)
    relevant_old = rec("他说他第一次来这里", tick=1, importance=0.5)
    important_middling = rec("约定明天再来", tick=50, importance=1.0)

    now, half_life = 100, 40.0
    ranking = {
        name: sorted(
            (recent_irrelevant, relevant_old, important_middling),
            key=lambda r: -build_strategy(name).score(r, query, now, half_life),
        )[0].content
        for name in ("recency", "lexical", "importance")
    }

    assert ranking["recency"] == "今天天气不错"          # 只看新
    assert ranking["lexical"] == "他说他第一次来这里"    # 只看相关
    assert ranking["importance"] == "约定明天再来"       # 只看重要
    assert len(set(ranking.values())) == 3               # 三家结论互不相同
