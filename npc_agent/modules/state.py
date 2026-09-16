"""模块三：State Tracking —— 现场状态跟踪。

NPC 光有记忆不够，还得知道"此刻现场是什么情况"：
    谁在场、谁刚说过话、谁在跟谁说话、话题走到哪了、冷场多久了、
    我自己的发言占比是不是太高了（抢戏检测）。

State Tracker 把这些从观测流里提炼成结构化状态，
Planner 和 Dialogue 都从这里读现场，而不是各自去翻原始日志。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..types import Utterance


@dataclass
class PlayerModel:
    """NPC 心里对每个玩家（或同伴 NPC）的画像。"""

    id: str
    name: str
    kind: str = "player"  # player | npc —— 同伴也是"我认识的人"
    affinity: int = 50
    utterance_count: int = 0
    last_spoke_tick: int = -99
    seated: bool = False
    location: str = ""
    here: bool = False  # 是否与 NPC 同位置（决定能不能递东西）
    revealed: dict[str, str] = field(default_factory=dict)  # 玩家透露过的信息

    def render(self) -> str:
        where = "" if self.here else f"，在{self.location}"
        label = "同伴" if self.kind == "npc" else f"好感 {self.affinity}"
        bits = [f"{self.name}（{label}，已发言 {self.utterance_count} 次{where}）"]
        if self.revealed:
            detail = "；".join(f"{k}={v}" for k, v in self.revealed.items())
            bits.append(f"已知信息: {detail}")
        return " / ".join(bits)


class StateTracker:
    def __init__(self, npc_id: str, npc_name: str) -> None:
        self.npc_id = npc_id
        self.npc_name = npc_name
        self.tick = 0
        self.location = ""
        self.location_name = ""
        self.inventory: list[str] = []
        self.players: dict[str, PlayerModel] = {}
        # 同伴 NPC。单独放一个字典而不是塞进 players：
        # 两者的语义不同 —— 玩家要算"发言占比"的分母，同伴是同行、
        # 是要让出话头的对象，混在一起会让抢戏检测失真。
        self.peers: dict[str, PlayerModel] = {}
        self.world_flags: set[str] = set()
        self.objectives: dict[str, str] = {}
        self.topic: str = ""
        self.last_speaker: Optional[str] = None
        self.last_utterance: Optional[Utterance] = None
        self.pending_question_to_npc: bool = False
        self.silence_ticks: int = 0
        self.npc_utterances: int = 0
        self.recent_mentions: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def update_from_env(self, observation: dict[str, Any]) -> None:
        self.tick = observation.get("tick", self.tick)
        me = observation.get("self") or {}
        self.location = me.get("loc", self.location)
        self.location_name = me.get("loc_name", self.location_name)
        self.inventory = list(me.get("inventory") or [])
        self.world_flags = set(observation.get("world_flags") or [])
        self.objectives = dict(observation.get("objectives") or {})

        seen: set[str] = set()
        peers_seen: set[str] = set()
        # 用 present_actors 而不是 visible_actors：
        # 能听见 ≠ 能碰到。门口说话的客人也该被 NPC 记住。
        for spec in observation.get("present_actors") or observation.get("visible_actors") or []:
            if spec.get("id") == self.npc_id:
                continue
            is_peer = spec.get("kind") == "npc"
            bucket = self.peers if is_peer else self.players
            index = peers_seen if is_peer else seen
            index.add(spec["id"])
            model = bucket.get(spec["id"]) or PlayerModel(
                spec["id"], spec["name"], "npc" if is_peer else "player"
            )
            model.name = spec.get("name", model.name)
            model.kind = "npc" if is_peer else "player"
            model.location = spec.get("loc", model.location)
            model.here = bool(spec.get("here", spec.get("loc") == self.location))
            bucket[spec["id"]] = model

        affinity = observation.get("affinity") or {}
        for pid, value in affinity.items():
            if pid in self.players:
                self.players[pid].affinity = value

        # 离开视野的玩家**保留**画像：NPC 应该记得来过的人，只是标记为不同位置
        for pid, model in self.players.items():
            if pid not in seen:
                model.here = False
        for pid, model in self.peers.items():
            if pid not in peers_seen:
                model.here = False

    # ------------------------------------------------------------------ #
    def note_utterance(self, utterance: Utterance) -> None:
        self.last_utterance = utterance
        self.last_speaker = utterance.speaker_id

        if utterance.speaker_id == self.npc_id:
            self.npc_utterances += 1
            return

        # 同伴 NPC 发言：记在他头上，并且算作"现场有人在说话"
        peer = self.peers.get(utterance.speaker_id)
        if peer is not None:
            peer.utterance_count += 1
            peer.last_spoke_tick = utterance.tick
            self.silence_ticks = 0
            self.recent_mentions = list(utterance.mentions)
            self.pending_question_to_npc = bool(
                utterance.is_question
                and (self.npc_id in utterance.mentions or self.npc_name in utterance.text)
            )
            return

        model = self.players.get(utterance.speaker_id)
        if model:
            model.utterance_count += 1
            model.last_spoke_tick = utterance.tick

        self.silence_ticks = 0
        self.recent_mentions = list(utterance.mentions)

        addressed_to_npc = self.npc_id in utterance.mentions or any(
            alias in utterance.text for alias in (self.npc_name,)
        )
        self.pending_question_to_npc = bool(utterance.is_question and addressed_to_npc)
        if not self.pending_question_to_npc and utterance.is_question:
            # 没点名的提问也算半开放，NPC 可以接
            self.pending_question_to_npc = self.npc_id in utterance.mentions

    def note_npc_spoke(self) -> None:
        self.npc_utterances += 1
        self.silence_ticks = 0
        self.pending_question_to_npc = False

    def tick_silence(self) -> None:
        self.silence_ticks += 1

    # ------------------------------------------------------------------ #
    def active_players(self) -> list[PlayerModel]:
        return sorted(
            self.players.values(), key=lambda p: (-p.last_spoke_tick, p.id)
        )

    def peers_here(self) -> list[PlayerModel]:
        return [p for p in self.peers.values() if p.here]

    def speaking_load(self) -> dict[str, float]:
        """发言占比。主持场景用它检测"NPC 是否抢戏"。

        分母包含同伴 NPC 的发言 —— 否则两个 NPC 各说一半时，
        每个 NPC 都会以为自己的占比是 100%。
        """
        total = (
            self.npc_utterances
            + sum(p.utterance_count for p in self.players.values())
            + sum(p.utterance_count for p in self.peers.values())
        )
        if total == 0:
            return {self.npc_id: 0.0}
        load = {self.npc_id: self.npc_utterances / total}
        for model in self.players.values():
            load[model.id] = model.utterance_count / total
        for model in self.peers.values():
            load[model.id] = model.utterance_count / total
        return load

    def npc_share(self) -> float:
        return self.speaking_load().get(self.npc_id, 0.0)

    # ------------------------------------------------------------------ #
    def scene_block(self) -> str:
        """给 prompt 用的现场描述。"""
        lines = [f"位置：{self.location_name or self.location}"]
        if self.inventory:
            lines.append("手上拿着：" + "、".join(self.inventory))
        if self.peers:
            lines.append("一起当班的同伴：")
            lines.extend(f"  - {m.render()}" for m in self.peers.values())
        if self.players:
            lines.append("在场玩家：")
            lines.extend(f"  - {m.render()}" for m in self.players.values())
        else:
            lines.append("在场玩家：（暂时没有人）")
        if self.world_flags:
            lines.append("已发生的事：" + "、".join(sorted(self.world_flags)))
        pending = [k for k, v in self.objectives.items() if v != "done"]
        if pending:
            lines.append("还没完成的目标：" + "、".join(pending))
        if self.silence_ticks:
            lines.append(f"已经连续 {self.silence_ticks} 轮没人说话了")
        lines.append(f"你的发言占比：{self.npc_share():.0%}")
        return "\n".join(lines)

    def render_players(self) -> str:
        if not self.players:
            return "（没有玩家）"
        return "\n".join(f"  - {m.render()}" for m in self.players.values())
