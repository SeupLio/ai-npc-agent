"""记忆系统测试：写入、检索、巩固、遗忘曲线。"""

from __future__ import annotations

from npc_agent.modules.memory import MemoryManager, MemoryStore, estimate_importance


def test_importance_detects_preferences() -> None:
    """偏好类信息的重要度应高于普通寒暄 —— 否则检索时永远排不上号。"""
    assert estimate_importance("我特别喜欢偏酸的咖啡") > estimate_importance("今天天气不错")


def test_search_prefers_relevant_over_old() -> None:
    store = MemoryStore()
    store.add("episodic", "阿澈说：我特别喜欢偏酸的咖啡。", tick=0)
    store.add("episodic", "小满说：外面下雨了。", tick=1)
    hits = store.search("偏酸的咖啡", k=2, now=2)
    assert "偏酸" in hits[0].content


def test_recency_decay_favours_recent() -> None:
    """两条同样相关时，更近的一条应该排前面。"""
    store = MemoryStore(half_life=10.0)
    store.add("episodic", "客人点了柠檬水", tick=0, importance=0.5)
    store.add("episodic", "客人点了柠檬水", tick=100, importance=0.5)
    hits = store.search("柠檬水", k=2, now=100)
    assert hits[0].tick == 100


def test_consolidation_preserves_early_facts() -> None:
    """巩固 ≠ 遗忘：早期信息必须活下来，只是被降维成摘要。"""
    store = MemoryStore(consolidate_at=6)
    store.add("episodic", "阿澈说：我特别喜欢偏酸的咖啡。", tick=0)
    for i in range(10):
        store.add("episodic", f"小满说：第{i}句闲聊。", tick=i + 1)
    store.consolidate(now=20)
    assert store.stats().consolidated >= 1
    assert any("偏酸" in r.content for r in store.records)


def test_consolidation_reduces_record_count() -> None:
    store = MemoryStore(consolidate_at=6)
    for i in range(12):
        store.add("episodic", f"第{i}句", tick=i)
    before = len(store)
    store.consolidate(now=30)
    assert len(store) < before


def test_reflection_gets_boosted() -> None:
    """教训要比普通记忆更容易被想起来。"""
    store = MemoryStore()
    store.add("episodic", "柠檬水卖完了", tick=0, importance=0.6)
    store.add("reflection", "教训：柠檬水卖完了就别再承诺", tick=0, importance=0.6)
    hits = store.search("柠檬水", k=1, now=0)
    assert hits[0].kind == "reflection"


def test_access_count_increases_on_hit() -> None:
    store = MemoryStore()
    record = store.add("episodic", "阿澈喜欢靠窗的位置", tick=0)
    store.search("阿澈 位置", k=1, now=1)
    assert record.access_count == 1


def test_manager_skips_npc_own_speech() -> None:
    from npc_agent.types import Utterance

    manager = MemoryManager(MemoryStore())
    own = Utterance(speaker_id="ayou", speaker_name="阿柚", text="欢迎", tick=0, role="npc")
    assert manager.observe(own, npc_id="ayou") is None
    assert len(manager.store) == 0


def test_manager_records_player_speech() -> None:
    from npc_agent.types import Utterance

    manager = MemoryManager(MemoryStore())
    utterance = Utterance(
        speaker_id="player_a", speaker_name="阿澈", text="我常来", tick=0, role="player"
    )
    record = manager.observe(utterance, npc_id="ayou")
    assert record is not None
    assert "player_a" in record.entities
