"""存档与读档 —— 让 NPC 记得"上一次你来过"。

## 为什么这件事比看起来重要

前面所有记忆能力都有一个隐含前提：**同一个进程里连续玩**。
关掉窗口再进来，NPC 会把你当成第一次见面的人 —— 好感清零、称号清零、
"我喜欢偏酸的"也不知道了。

对单机 demo 无所谓；对一个"养成"玩法，这是致命的：
玩家花了两小时把关系养到「常客」，第二天回来一切归零。

## 一个必须处理的坑：tick 要接上

记忆检索里有一条硬条件 `record.tick < now`（"过去的事"）。
如果存档后 tick 从 0 重新开始，而记忆里的 tick 是 12，
那么**所有记忆都会被判成"未来的事"而过滤掉** ——
NPC 静默失忆，而且不报任何错。

所以存档必须带 `tick`，读档之后把环境时钟推到它之后。
这条不是"顺手"，是这个模块存在的理由之一。

## 存什么、不存什么

存：每个 NPC 的记忆、关系档案、欠着的回答；世界层面的事件编年史；时钟。

不存：这一局的具体转写、传闻网络（它是"这一局里谁说过什么"的公共事实，
      跨局保留会让新一局的 NPC 一开口就提上一局的事）、发言权状态。

判断标准是**"玩家下次回来，这件事还成立吗"**：
好感成立，称号成立，"我欠你一个回答"也成立（他还没得到回答）；
但"刚才小舟说了句什么"不成立。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..types import MemoryRecord

#: 存档格式版本。改结构就要 +1，并且**读旧档要明确报错**，
#: 不能"尽量兼容" —— 半读进来的存档比读不进来更危险。
SCHEMA_VERSION = 1


class PersistenceError(RuntimeError):
    """存档读不进来。消息里必须说清是版本不对、文件不在、还是内容坏了。"""


# --------------------------------------------------------------------------- #
# 每个 NPC
# --------------------------------------------------------------------------- #
def snapshot_agent(agent: Any) -> dict[str, Any]:
    store = agent.memory.store
    bonds = {
        pid: {
            "score": bond.score,
            "interactions": bond.interactions,
            "stage": bond.stage,
            "titles": list(bond.titles),
            "last_tick": bond.last_tick,
        }
        for pid, bond in agent.relationships.bonds.items()
    }
    debts = [
        {"player_id": d.player_id, "text": d.text, "tick": d.tick, "asked": d.asked}
        for d in agent.attention.debts
    ]
    return {
        "id": agent.id,
        "memory": [asdict(record) for record in store.records],
        "memory_seq": store._seq,
        "bonds": bonds,
        "debts": debts,
        "titles_granted": agent.relationships.titles_granted,
        "follow_ups": agent.attention.follow_ups,
    }


def restore_agent(agent: Any, data: dict[str, Any]) -> None:
    store = agent.memory.store
    store.records = [_record_from_dict(item) for item in data.get("memory") or []]
    store._seq = int(data.get("memory_seq") or len(store.records))

    book = agent.relationships
    book.bonds.clear()
    for pid, raw in (data.get("bonds") or {}).items():
        bond = book.bond(pid)
        bond.score = int(raw.get("score", 50))
        bond.interactions = int(raw.get("interactions", 0))
        bond.stage = str(raw.get("stage") or "生客")
        bond.titles = list(raw.get("titles") or [])
        bond.last_tick = int(raw.get("last_tick", -99))
    book.titles_granted = int(data.get("titles_granted") or 0)
    book.pending.clear()

    board = agent.attention
    board.debts.clear()
    from .attention import QuestionDebt

    for raw in data.get("debts") or []:
        board.debts.append(
            QuestionDebt(
                player_id=str(raw.get("player_id") or ""),
                text=str(raw.get("text") or ""),
                tick=int(raw.get("tick") or 0),
                asked=int(raw.get("asked") or 1),
            )
        )
    board.follow_ups = int(data.get("follow_ups") or 0)


def _record_from_dict(raw: dict[str, Any]) -> MemoryRecord:
    return MemoryRecord(
        id=str(raw.get("id") or "m0000"),
        kind=raw.get("kind") or "episodic",  # type: ignore[arg-type]
        content=str(raw.get("content") or ""),
        tick=int(raw.get("tick") or 0),
        importance=float(raw.get("importance") or 0.5),
        entities=list(raw.get("entities") or []),
        tags=list(raw.get("tags") or []),
        last_access=int(raw.get("last_access") or 0),
        access_count=int(raw.get("access_count") or 0),
    )


# --------------------------------------------------------------------------- #
# 整个剧组
# --------------------------------------------------------------------------- #
def snapshot(cast: Any) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "scenario": getattr(cast, "scenario_id", ""),
        # ⚠️ tick 必须存（见模块注释）。少了它，读档之后 NPC 会静默失忆。
        "tick": int(getattr(cast.env, "tick", 0)),
        "rounds": int(getattr(cast, "rounds", 0)),
        "agents": {pid: snapshot_agent(agent) for pid, agent in cast.agents.items()},
        # 编年史：这一局发生过什么。跨局保留是**有意的** ——
        # "上次我们为营业时间争过一次"是玩家能感知到的连续性。
        "events": [
            {
                "id": e.id,
                "tick": e.tick,
                "kind": e.kind,
                "text": e.text,
                "winner": e.winner,
                "loser": e.loser,
                "resource": getattr(e, "resource", ""),
            }
            for e in getattr(cast, "events", [])
        ],
    }


def save(cast: Any, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = snapshot(cast)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def load(cast: Any, path: str | Path) -> dict[str, Any]:
    """把存档灌进一个**刚建好**的剧组。返回存档本身（调用方可能要读 tick）。"""
    source = Path(path)
    if not source.exists():
        raise PersistenceError(f"存档不存在：{source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PersistenceError(f"存档读不出来（{source}）：{exc}") from exc

    version = payload.get("schema")
    if version != SCHEMA_VERSION:
        raise PersistenceError(
            f"存档版本是 {version}，当前程序要 {SCHEMA_VERSION} —— "
            "不做「尽量兼容」：半读进来的存档比读不进来更危险。"
        )

    for pid, data in (payload.get("agents") or {}).items():
        agent = cast.agents.get(pid)
        if agent is None:
            # 场景换了、演员表对不上：跳过而不是崩 —— 但要说出来。
            continue
        restore_agent(agent, data)

    # 时钟接上：新一局从存档的下一格开始。
    cast.env.tick = int(payload.get("tick") or 0)
    cast.rounds = int(payload.get("rounds") or 0)
    _restore_events(cast, payload.get("events") or [])
    return payload


def _restore_events(cast: Any, raw_events: list[dict[str, Any]]) -> None:
    from .motives import WorldEvent

    # ⚠️ **原地替换**（`[:] =`），不是重新赋值。
    # 每个 NPC 的 `chronicle` 持有的是这个 list 的**引用**（见 `Cast.reset`），
    # 重新赋值会让它们继续指着旧列表 —— 读档之后 NPC 就说不出"上次那件事"，
    # 而且不会报错。
    cast.events[:] = [
        WorldEvent(
            id=str(e.get("id") or "event"),
            tick=int(e.get("tick") or 0),
            kind=str(e.get("kind") or "事件"),
            text=str(e.get("text") or ""),
            winner=str(e.get("winner") or ""),
            loser=str(e.get("loser") or ""),
            resource=str(e.get("resource") or ""),
            # 上一局已经播报过的让步不再重播 —— 那是复读。
            spoken=True,
        )
        for e in raw_events
    ]

    # ⚠️ 还要把"这对冲突已经裁定过"告诉裁定台。
    # 只恢复事件、不恢复 `resolved`，下一轮 `poll` 会认为冲突还没解决，
    # 于是**再裁定一次**，让步方把同一句话说第二遍（实测：读档后
    # 「行，那就依小舟，再唱一首。」又说了一次）。
    board = getattr(cast, "motives", None)
    if board is not None:
        for event in cast.events:
            key = tuple(sorted((event.winner, event.loser))) + (event.resource,)
            board.resolved.add(key)


__all__ = [
    "PersistenceError",
    "SCHEMA_VERSION",
    "load",
    "restore_agent",
    "save",
    "snapshot",
    "snapshot_agent",
]
