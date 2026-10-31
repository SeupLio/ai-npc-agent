"""存档 / 读档（`modules/persistence.py`）。

## 这条链路最容易出的错是"静默失忆"

存档读不进来会报错，那是好事；真正难查的是**读进来了但少了东西**：

- tick 没接上 → 所有记忆的 `tick` 都大于当前时钟，检索里的
  `record.tick < now` 把它们全过滤掉 → NPC 一句话都不记得，**不报错**；
- 关系没存 → 玩家昨天养到「常客」，今天回来又是「生客」；
- 编年史没接上引用 → 事件在，但 NPC 手里那个 list 是旧的 → 问"上次那事"答不上来。

所以这里的断言全部是**行为**断言，不是"字段相等"。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from npc_agent.cast import build_cast
from npc_agent.config import RuntimeConfig, load_scenario
from npc_agent.llm import NullLLM
from npc_agent.modules.persistence import (
    SCHEMA_VERSION,
    PersistenceError,
    load,
    save,
    snapshot,
)


def _cast():
    return build_cast(load_scenario("duet"), NullLLM(), RuntimeConfig())


def _trio():
    return build_cast(load_scenario("trio"), NullLLM(), RuntimeConfig())


def _play(cast, ticks: int = 6) -> None:
    for _ in range(ticks):
        cast.step(None)


# --------------------------------------------------------------------------- #
# 基本往返
# --------------------------------------------------------------------------- #
def test_save_writes_a_readable_json(tmp_path: Path):
    cast = _cast()
    _play(cast, 3)
    path = save(cast, tmp_path / "s.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == SCHEMA_VERSION
    assert set(payload["agents"]) == set(cast.agents)


def test_a_missing_file_is_a_clear_error(tmp_path: Path):
    with pytest.raises(PersistenceError, match="不存在"):
        load(_cast(), tmp_path / "nope.json")


def test_a_version_mismatch_is_refused(tmp_path: Path):
    """半读进来的存档比读不进来更危险 —— 明确拒绝。"""
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"schema": 999, "agents": {}}), encoding="utf-8")
    with pytest.raises(PersistenceError, match="版本"):
        load(_cast(), path)


def test_a_corrupt_file_is_a_clear_error(tmp_path: Path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(PersistenceError, match="读不出来"):
        load(_cast(), path)


# --------------------------------------------------------------------------- #
# tick 必须接上（否则静默失忆）
# --------------------------------------------------------------------------- #
def test_the_clock_continues_after_loading(tmp_path: Path):
    cast = _cast()
    _play(cast, 5)
    saved_tick = cast.env.tick
    save(cast, tmp_path / "s.json")

    fresh = _cast()
    load(fresh, tmp_path / "s.json")
    assert fresh.env.tick == saved_tick


def test_memories_stay_recallable_after_loading(tmp_path: Path):
    """这条是 tick 那个坑的**行为证据**：tick 没接上时它会红。

    做法：让玩家说一句带偏好线索的话 → 存档 → 新剧组读档 →
    再问"你还记得我的口味吗"，NPC 必须答得出来。
    """
    cast = _cast()
    cast.step(cast.env.record_player_utterance("player_a", "我喜欢偏酸的豆子。"))
    _play(cast, 2)
    save(cast, tmp_path / "s.json")

    fresh = _cast()
    load(fresh, tmp_path / "s.json")
    turns = fresh.step(
        fresh.env.record_player_utterance("player_a", "你还记得我的口味吗？")
    )
    says = {t.actor_id: t.say for t in turns if t.say}
    assert says, "读档之后 NPC 一句话都没说"
    assert any("酸" in line for line in says.values()), f"记忆没接上：{says}"


# --------------------------------------------------------------------------- #
# 关系 / 编年史
# --------------------------------------------------------------------------- #
def test_the_relationship_survives(tmp_path: Path):
    cast = _cast()
    cast.env.actors["ayou"].bump_affinity("player_a", 8)
    _play(cast, 2)
    stage = cast.agents["ayou"].relationships.stage("player_a")
    title = cast.agents["ayou"].relationships.title_for("player_a")
    assert stage != "生客", "用例前提不成立：好感没涨上去"
    save(cast, tmp_path / "s.json")

    fresh = _cast()
    load(fresh, tmp_path / "s.json")
    assert fresh.agents["ayou"].relationships.stage("player_a") == stage
    assert fresh.agents["ayou"].relationships.title_for("player_a") == title


def test_the_relationship_score_does_not_reset_after_the_first_step(tmp_path: Path):
    """只存关系账本不够：账本每轮**从环境观察同步**，环境好感没存就会被打回 50。

    这条是行为断言：读档后走一步，分数和称号必须同时成立。
    以前只断言 stage 相等 —— 而 stage 只升不降，所以"分数回到 50"
    在这个断言下是隐形的。实测：`bond 64 → 读档 64 → 走一步 50`。
    """
    cast = _cast()
    cast.env.actors["ayou"].bump_affinity("player_a", 14)
    _play(cast, 2)
    saved = cast.agents["ayou"].relationships.bonds["player_a"].score
    assert saved == 64, f"用例前提不成立：好感是 {saved}，不是 64"
    save(cast, tmp_path / "s.json")

    fresh = _cast()
    load(fresh, tmp_path / "s.json")
    fresh.step(None)
    assert fresh.env.actors["ayou"].affinity_to("player_a") == saved
    assert fresh.agents["ayou"].relationships.bonds["player_a"].score == saved


def test_peer_affinity_survives(tmp_path: Path):
    """NPC ↔ NPC 的好感不进关系账本，只在环境里 —— 它也要跨会话保留。"""
    cast = _cast()
    cast.env.actors["ayou"].bump_affinity("xiaozhou", 9)
    save(cast, tmp_path / "s.json")

    fresh = _cast()
    load(fresh, tmp_path / "s.json")
    fresh.step(None)
    assert fresh.env.actors["ayou"].affinity_to("xiaozhou") == 59
    assert fresh.agents["ayou"].state.peers["xiaozhou"].affinity == 59


def test_an_old_save_without_affinity_rebuilds_the_player_score_from_bonds(tmp_path: Path):
    """加字段不能把旧档变成"称号还在、分数掉了"的自相矛盾状态。"""
    cast = _cast()
    cast.env.actors["ayou"].bump_affinity("player_a", 8)
    _play(cast, 2)
    saved = cast.agents["ayou"].relationships.bonds["player_a"].score
    path = save(cast, tmp_path / "s.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("affinity")
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    fresh = _cast()
    load(fresh, path)
    fresh.step(None)
    assert fresh.agents["ayou"].relationships.bonds["player_a"].score == saved


def test_the_chronicle_survives_and_can_be_told(tmp_path: Path):
    cast = _cast()
    _play(cast, 6)
    assert cast.events, "用例前提不成立：没有产生事件"
    save(cast, tmp_path / "s.json")

    fresh = _cast()
    load(fresh, tmp_path / "s.json")
    assert fresh.events, "读档之后编年史空了"
    # 引用必须接上：`Cast.reset` 把 `self.events` 交给每个 NPC，
    # 读档时如果**重新赋值** `cast.events`，NPC 手里还是旧 list。
    assert fresh.agents["ayou"].chronicle is fresh.events

    turns = fresh.step(fresh.env.record_player_utterance("player_a", "上次那事怎么样了？"))
    says = {t.actor_id: t.say for t in turns if t.say}
    assert any("营业时间" in line for line in says.values()), f"编年史没接上：{says}"


def test_a_pending_yield_is_not_replayed_after_loading(tmp_path: Path):
    """上一局已经说过的让步，读档之后不该再说一遍。"""
    cast = _cast()
    _play(cast, 8)
    save(cast, tmp_path / "s.json")

    fresh = _cast()
    load(fresh, tmp_path / "s.json")
    _play(fresh, 3)
    lines = [u.text for u in fresh.env.utterances if "依小舟" in u.text]
    assert lines == [], f"读档之后重播了让步：{lines}"


def test_a_three_way_clash_keeps_all_losers_in_the_chronicle(tmp_path: Path):
    """三人抢同一资源的让步方不止一个 —— 存档不能只留下代表。"""
    cast = _trio()
    _play(cast, 8)
    assert cast.events, "用例前提不成立：没有产生事件"
    event = cast.events[0]
    assert len(event.losers) == 2, f"三人冲突只记了 {event.losers}"
    save(cast, tmp_path / "s.json")

    fresh = _trio()
    load(fresh, tmp_path / "s.json")
    assert fresh.events[0].losers == event.losers
    _play(fresh, 2)
    assert len(fresh.events) == 1, "读档后同一个资源被二次裁定"


# --------------------------------------------------------------------------- #
# 存档格式本身
# --------------------------------------------------------------------------- #
def test_the_snapshot_does_not_carry_the_transcript():
    """转写是"这一局发生了什么"，不是"玩家是谁" —— 跨局保留会让 NPC 一开口就翻旧账。"""
    payload = snapshot(_cast())
    assert "transcript" not in payload
    assert "utterances" not in json.dumps(payload)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
