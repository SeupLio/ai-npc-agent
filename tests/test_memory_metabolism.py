"""记忆代谢：知识状态、转述失真、遗忘重构、隐私分区。

## 这组测试守的不是"记没记住"

记忆系统的常见评测只问"有没有这条记忆"。真实的人不是这样：

- 亲耳听见和听人转述**不是同一种知识**；
- 太久没想起的琐事会淡掉，但被追问时还能说出"我好像听谁提过"；
- 别人叮嘱过"别告诉别人"的事，不能从另一个 NPC 嘴里漏出去。

所以这里的断言全部落在**知识状态**上：
`origin / hop / fidelity / confidence / forgotten / audience`，
以及它们对**检索结果**和**台词**的影响 ——
只改元数据、行为不变，等于没有实现。

## 为什么"越传越确信"必须是可证的

如果只给每条记忆加一个来源字符串，那只是一个展示字段。
真正有产品意义的是一条**反直觉的事实**：
转述会丢掉细节（保真度下降），但对转述者的信任会让语气更笃定
（确信度上升）。两个数朝相反方向走，才叫"记忆代谢"。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from npc_agent import studio as S
from npc_agent.cast import build_cast
from npc_agent.config import RuntimeConfig, load_scenario
from npc_agent.llm import NullLLM
from npc_agent.modules.memory import MemoryManager, MemoryStore
from npc_agent.modules.persistence import load, save, snapshot
from npc_agent.modules.social import retell_clause


def _cast(config: RuntimeConfig | None = None):
    return build_cast(load_scenario("duet"), NullLLM(), config or RuntimeConfig())


def _manager(**kwargs) -> MemoryManager:
    return MemoryManager(MemoryStore(**kwargs))


def _old_save_payload(cast) -> dict:
    payload = snapshot(cast)
    payload["agents"]["ayou"]["memory"] = [
        {
            "id": "m0001",
            "kind": "episodic",
            "content": "阿澈说：你好",
            "tick": 0,
            "importance": 0.5,
        }
    ]
    payload["agents"]["ayou"]["memory_seq"] = 1
    return payload


# --------------------------------------------------------------------------- #
# 来源 / 保真 / 确信
# --------------------------------------------------------------------------- #
def test_a_first_hand_observation_is_marked_as_such():
    manager = _manager()
    utterance = _cast().env.record_player_utterance("player_a", "我喜欢偏酸的豆子。")
    record = manager.observe(utterance, "ayou")
    assert record is not None
    assert record.origin == "observed"
    assert record.hop == 0
    assert record.fidelity == 1.0
    assert record.confidence == 1.0
    assert record.source_id == "player_a"
    assert record.source_name == "阿澈"
    assert record.privacy == "public"
    assert record.state_label() == "亲眼所见"


def test_a_retelling_drops_the_last_detail_clause():
    """`retell_clause` 是**可复现**的有损模型，不是随机篡改。

    人复述时会先保住主干（谁、什么偏好），先掉的是限定条件。
    这里钉住"丢掉的是最后一小句"，因为只有确定的行为才能被评测。
    """
    assert retell_clause("阿澈喜欢偏酸的豆子，越酸越好") == "阿澈喜欢偏酸的豆子"
    assert retell_clause("只有一个小句") == "只有一个小句"

    manager = _manager()
    record = manager.receive_gossip(
        "阿澈喜欢偏酸的豆子，越酸越好",
        tick=3,
        source_id="ayou",
        source_name="阿柚",
        subject="player_a",
    )
    assert record.content == "听阿柚说，阿澈喜欢偏酸的豆子"
    assert record.origin == "told"
    assert record.hop == 1
    assert record.tags == ["gossip", "source:阿柚"]
    assert record.source_name == "阿柚"
    assert record.inversion > 0, (
        f"转述没有体现「越传越确信」：fidelity={record.fidelity}, "
        f"confidence={record.confidence}"
    )


def test_retelling_the_same_clause_keeps_the_text_but_loses_certainty():
    """只有一个小句时文本不变 —— 但保真度仍然下降。

    保真度是**知识状态**，不是"文本改了几个字"。
    转述者把细节说全了，也不代表他传的是原话的完整上下文。
    """
    manager = _manager()
    record = manager.receive_gossip(
        "阿澈喜欢偏酸的豆子",
        tick=1,
        source_id="ayou",
        source_name="阿柚",
    )
    assert "阿澈喜欢偏酸的豆子" in record.content
    assert record.fidelity < 1.0
    assert record.confidence > record.fidelity


def test_multi_hop_keeps_losing_detail_and_gaining_confidence():
    """每转一手：保真度 ×0.82，确信度 +0.05，直到封顶。"""
    manager = _manager()
    first = manager.receive_gossip(
        "阿澈喜欢偏酸的豆子，越酸越好",
        tick=1,
        source_id="ayou",
        source_name="阿柚",
        subject="player_a",
    )
    second = manager.receive_gossip(
        first.content,
        tick=2,
        source_id="xiaozhou",
        source_name="小舟",
        subject="player_a",
        hop=2,
        fidelity=round(first.fidelity * 0.82, 4),
        confidence=min(0.98, round(first.confidence + 0.05, 4)),
    )
    assert second.hop == 2
    assert second.fidelity < first.fidelity
    assert second.confidence > first.confidence
    assert second.inversion > first.inversion
    # 前缀不能叠加成「听小舟说，听阿柚说，…」—— 那会把事实本身挤掉。
    assert second.content == "听小舟说，阿澈喜欢偏酸的豆子"


def test_consolidation_keeps_the_source_graph_and_refuses_secrets():
    """压缩可以删旧记录，但**来源图不能跟着消失**。

    `derived_from` / `sources` / `associations` 是摘要唯一的可解释性。
    少了它们，读档之后没人知道这条结论是从哪几句话推出来的。
    """
    store = MemoryStore(consolidate_at=4)
    ids = []
    for i in range(8):
        record = store.add(
            "episodic",
            f"阿澈说：第{i}句话。",
            tick=i,
            importance=0.5,
            entities=["player_a"],
            origin="observed",
            source_name="阿澈",
        )
        ids.append(record.id)
    secret = store.add(
        "episodic",
        "阿澈说：我有个秘密，别告诉别人。",
        tick=8,
        importance=0.8,
        entities=["player_a"],
        privacy="secret",
        audience=["player_a"],
    )

    merged = store.consolidate(now=10)
    assert merged is not None
    assert merged.origin == "consolidated"
    assert merged.derived_from, "摘要没有记下它从哪几条来"
    assert set(merged.derived_from) <= set(ids)
    assert merged.sources == ["阿澈"]
    assert merged.fidelity < 1.0
    assert merged.confidence < 1.0
    assert set(merged.associations) == set(merged.derived_from)
    # 限制级记忆不参与压缩，内容更不能出现在公开摘要里。
    assert secret in store.records
    assert "秘密" not in merged.content


# --------------------------------------------------------------------------- #
# 遗忘：归档，不是删除
# --------------------------------------------------------------------------- #
def test_forgetting_archives_the_record_instead_of_deleting_it():
    store = MemoryStore(forget_after=10, forget_below=0.6)
    trivial = store.add("episodic", "今天风挺大", tick=0, importance=0.3)
    important = store.add("semantic", "阿澈喜欢偏酸的豆子", tick=0, importance=0.9)
    reinforced = store.add("episodic", "常来坐靠窗的位置", tick=0, importance=0.3)
    reinforced.access_count = 3
    reinforced.last_access = 16

    archived = store.forget_stale(now=20, limit=10)
    assert archived == [trivial]
    assert len(store.records) == 3, "遗忘不该删记录"
    assert trivial.forgotten is True
    assert trivial.forgotten_tick == 20
    assert trivial.fidelity < 1.0
    assert important.forgotten is False, "高重要度的偏好不该被忘"
    assert reinforced.forgotten is False, "反复想起的记忆不该被忘"

    stats = store.stats()
    assert stats.forgotten == 1
    assert stats.total == 2, "遗忘的条数不该混进 active total"


def test_a_forgotten_memory_does_not_surface_unless_asked():
    store = MemoryStore(forget_after=10, forget_below=0.6)
    record = store.add("episodic", "今天风挺大", tick=0, importance=0.3)
    store.add("semantic", "阿澈喜欢偏酸的豆子", tick=0, importance=0.9)
    store.forget_stale(now=20)

    # 正常检索看不到
    assert record not in store.search("风", k=5, now=21)
    # 明确要求时才能拿到（仅供审计 / 调试）
    assert record in store.search("风", k=5, now=21, include_forgotten=True)
    # 被"考记忆"时能捞回一条大概
    assert store.vague_recall("你还记得那件事吗？", now=21) is record


def test_the_prompt_says_when_a_memory_is_fuzzy():
    """给模型的上下文必须带"不确定"本身，不能假装内容还很清晰。"""
    manager = _manager(forget_after=10, forget_below=0.6)
    record = manager.store.add("episodic", "今天风挺大", tick=0, importance=0.3)
    manager.store.forget_stale(now=20)
    block = manager.context_block([record])
    assert "记不清" in block


# --------------------------------------------------------------------------- #
# 隐私：写入点分区 + 检索点过滤
# --------------------------------------------------------------------------- #
def test_privacy_filters_retrieval_by_viewer():
    store = MemoryStore()
    public = store.add("episodic", "阿澈喜欢偏酸的豆子", tick=1, entities=["player_a"])
    secret = store.add(
        "episodic",
        "阿澈说：我喜欢偏酸的豆子，别告诉别人",
        tick=1,
        entities=["player_a"],
        privacy="secret",
        audience=["player_a"],
    )
    assert secret in store.search("偏酸", k=5, now=2, viewer_id="player_a")
    assert secret not in store.search("偏酸", k=5, now=2, viewer_id="player_b")
    # 没有当前对话人（冷场 / NPC 对 NPC）时失败方向是"少说一句"。
    assert secret not in store.search("偏酸", k=5, now=2, viewer_id="")
    assert public in store.search("偏酸", k=5, now=2, viewer_id="player_b")


def test_associations_cross_entities_but_not_privacy_boundaries():
    """关联是"顺藤摸瓜"，不能变成隐私侧信道。"""
    store = MemoryStore()
    first = store.add("episodic", "阿澈说：你好", tick=1, entities=["player_a"])
    second = store.add("episodic", "阿澈说：今天风大", tick=2, entities=["player_a"])
    secret = store.add(
        "episodic",
        "阿澈说：别告诉别人",
        tick=3,
        entities=["player_a"],
        privacy="secret",
        audience=["player_a"],
    )
    third = store.add("episodic", "阿澈说：再见", tick=4, entities=["player_a"])

    assert first.associations == []
    assert second.associations == [first.id]
    assert secret.associations == [], "秘密记忆不能去连公开记忆"
    assert third.associations == [second.id, first.id]
    assert secret.id not in third.associations


def test_a_secret_never_leaks_to_another_player_end_to_end():
    """玩家 A 的秘密，玩家 B 问不出来，也不该从传闻网络流过去。

    实测踩到的第二跳泄漏：阿柚把秘密用自己的话对 A 说了一遍
    （模板原样带着「别告诉别人」），这句话是**广播**的，
    小舟的 `observe` 把它记成了公开记忆。隐私在第二跳被绕过去了。
    """
    cast = _cast()
    cast.step(
        cast.env.record_player_utterance("player_a", "我喜欢偏酸的豆子，别告诉别人")
    )
    for _ in range(3):
        cast.step(None)

    facts = cast.gossip.facts() if cast.gossip is not None else []
    assert facts == [], f"秘密进了传闻网络：{[f.content for f in facts]}"
    leaks = [
        r
        for agent in cast.agents.values()
        for r in agent.memory.store.records
        if "偏酸" in r.content and not r.restricted
    ]
    assert leaks == [], f"秘密以公开记忆的形式存在：{[r.content for r in leaks]}"

    turns_b = cast.step(
        cast.env.record_player_utterance("player_b", "阿柚，你还记得我喜欢什么吗？")
    )
    said_b = [t.say for t in turns_b if t.say]
    assert said_b and all("酸" not in line for line in said_b), (
        f"玩家 B 问到了玩家 A 的秘密：{said_b}"
    )

    turns_a = cast.step(
        cast.env.record_player_utterance("player_a", "阿柚，你还记得我喜欢什么吗？")
    )
    said_a = "".join(t.say for t in turns_a if t.say)
    assert "酸" in said_a, f"当事人自己也问不出来了 —— 分区做成了删除：{said_a}"


# --------------------------------------------------------------------------- #
# 遗忘之后仍能"重构大概"（行为层）
# --------------------------------------------------------------------------- #
def test_forgotten_details_are_reconstructed_as_fuzzy_not_recited():
    """忘了细节，但记得"有这么回事、听谁说过" —— 这才是人的遗忘。"""
    cast = _cast(
        RuntimeConfig(
            memory_forget_after=3,
            memory_forget_below=0.8,
            # 只想起最相关的一条：让这条琐事真的有机会"最近没被想起"。
            memory_top_k=1,
        )
    )
    cast.step(cast.env.record_player_utterance("player_a", "我喜欢那个靠窗的位置"))
    for _ in range(4):
        cast.step(None)

    forgotten = [
        r for r in cast.agents["xiaozhou"].memory.store.records if r.forgotten
    ]
    assert forgotten, "用例前提不成立：小舟没有遗忘任何记忆"

    turns = cast.step(
        cast.env.record_player_utterance("player_a", "小舟，你还记得我喜欢什么吗？")
    )
    said = "".join(t.say for t in turns if t.actor_id == "xiaozhou" and t.say)
    assert said, "被点名的小舟没有回答"
    assert any(word in said for word in ("记不清", "忘了", "想不起", "模糊")), said
    assert "靠窗" not in said, f"细节已经忘了，却仍然复述出来：{said}"


# --------------------------------------------------------------------------- #
# 跨存档 / Studio / 配置
# --------------------------------------------------------------------------- #
def test_the_knowledge_state_survives_a_round_trip(tmp_path: Path):
    cast = _cast()
    agent = cast.agents["ayou"]
    store = agent.memory.store

    store.add(
        "episodic",
        "阿澈说：今天风挺大",
        tick=0,
        importance=0.2,
        entities=["player_a"],
        origin="observed",
        source_name="阿澈",
    )
    store.forget_stale(now=100)
    store.add(
        "episodic",
        "阿澈说：我喜欢偏酸的豆子，别告诉别人",
        tick=1,
        importance=0.7,
        entities=["player_a"],
        origin="observed",
        source_name="阿澈",
        privacy="secret",
        audience=["player_a"],
    )
    told = agent.memory.receive_gossip(
        "阿澈喜欢偏酸的豆子",
        tick=2,
        source_id="xiaozhou",
        source_name="小舟",
        subject="player_a",
        hop=1,
        fidelity=0.57,
        confidence=0.95,
    )

    path = save(cast, tmp_path / "knowledge.json")
    fresh = _cast()
    load(fresh, path)
    restored = fresh.agents["ayou"].memory.store.records

    fuzzy = next(r for r in restored if "风挺大" in r.content)
    assert fuzzy.forgotten is True
    assert fuzzy.forgotten_tick == 100
    assert fuzzy.fidelity < 1.0

    secret = next(r for r in restored if "别告诉别人" in r.content)
    assert secret.restricted is True
    assert secret.audience == ["player_a"]
    assert secret.privacy == "secret"

    heard = next(r for r in restored if r.content == told.content)
    assert (heard.origin, heard.hop) == ("told", 1)
    assert heard.source_name == "小舟"
    assert heard.fidelity == 0.57
    assert heard.confidence == 0.95
    assert heard.tags == ["gossip", "source:小舟"]
    assert heard.inversion > 0

    summary = fresh.agents["ayou"].memory.store.provenance_summary()
    assert summary["told"] == 1
    assert summary["forgotten"] == 1


def test_an_old_save_without_knowledge_fields_is_read_as_first_hand(tmp_path: Path):
    """加字段不能把旧档读成"来源不明的空壳" —— 失败方向要可解释。"""
    cast = _cast()
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(_old_save_payload(cast), ensure_ascii=False), encoding="utf-8"
    )
    fresh = _cast()
    load(fresh, path)
    record = fresh.agents["ayou"].memory.store.records[0]
    assert record.origin == "observed"
    assert record.hop == 0
    assert record.fidelity == 1.0 and record.confidence == 1.0
    assert record.privacy == "public"
    assert record.audience == []
    assert record.forgotten is False
    assert record.state_label() == "亲眼所见"


def test_the_studio_exposes_the_knowledge_state():
    """只在引擎里成立的机制，调参的人不知道它有没有在动。"""
    cast = _cast()
    cast.step(cast.env.record_player_utterance("player_a", "我喜欢偏酸的豆子，越酸越好"))
    for _ in range(4):
        cast.step(None)

    payload = S._memory_payload(cast)
    for entry in payload.values():
        assert {"knowledge", "recent_meta", "forgotten", "told"} <= set(entry)
        assert {
            "active",
            "told",
            "forgotten",
            "avg_fidelity",
            "avg_confidence",
            "inversions",
        } <= set(entry["knowledge"])
        for meta in entry["recent_meta"]:
            assert {
                "kind",
                "content",
                "origin",
                "state",
                "source",
                "hop",
                "fidelity",
                "confidence",
                "inversion",
                "forgotten",
                "privacy",
                "links",
            } <= set(meta)

    heard = [
        meta
        for entry in payload.values()
        for meta in entry["recent_meta"]
        if meta["origin"] == "told"
    ]
    assert heard, "Studio 里看不到任何转述记忆"
    assert all(meta["fidelity"] < meta["confidence"] for meta in heard)
    assert any(meta["inversion"] for meta in heard), "倒挂没有在 Studio 里显形"


def test_forgetting_is_wired_to_the_runtime_config():
    """配置项不是装饰：调它必须真的改变 store。"""
    cast = _cast(RuntimeConfig(memory_forget_after=7, memory_forget_below=0.42))
    store = cast.agents["ayou"].memory.store
    assert store.forget_after == 7
    assert store.forget_below == pytest.approx(0.42)


def test_the_same_story_produces_the_same_knowledge_state():
    """可复现是评测的底线：两条同样的对话，知识状态必须逐字相同。"""

    def play() -> dict[str, list[tuple]]:
        cast = _cast()
        cast.step(cast.env.record_player_utterance("player_a", "我喜欢偏酸的豆子，越酸越好"))
        for _ in range(4):
            cast.step(None)
        cast.step(cast.env.record_player_utterance("player_a", "你还记得我的口味吗？"))
        return {
            pid: [
                (r.content, r.origin, r.hop, r.fidelity, r.confidence, r.forgotten)
                for r in agent.memory.store.records
            ]
            for pid, agent in cast.agents.items()
        }

    assert play() == play()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
