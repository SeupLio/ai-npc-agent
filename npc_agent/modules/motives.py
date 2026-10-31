"""动机与涌现事件 —— 让世界自己发生一点事。

## "涌现叙事"最少需要什么

米哈游公开方案里对涌现叙事的描述是：**NPC 因各自的经历产生不同动机，
动机之间相互影响，玩家的闲聊可能改变 NPC 后续的行动意图**。

拆到工程上，最少的闭环是三件事：

    1. 每个 NPC 有一个**想要的东西**（动机）
    2. 两个动机争夺**同一个东西**（冲突）
    3. 冲突有一个**结果**，而且结果会改变世界（事件）

少了第 3 条，"动机"就只是配置里的两行文字；少了第 1 条，
冲突只能靠剧本硬写。三条都有，才叫"涌现"。

## 为什么冲突必须由强度裁定，而不是轮流

轮流是"公平"，但它不产生故事 —— 玩家看到的是一人一次，没有胜负。
按强度裁定才有"谁让步"这件事，而**让步**才是能被看见的戏剧：
一方说"行，那就依你"，世界从此不同。

强度写死在一个配置里，是**产品判断**（谁是这场戏的主角），
不是模拟结果。这一点必须说清楚：它不是"AI 自己算出来的欲望"。

## 事件只发生一次

`resolved` 记已经裁定过的冲突对。不记的话，每一轮都会重新裁定一次，
NPC 会反复说"行，那就依你" —— 那是复读，不是涌现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class Motive:
    """一个 NPC 想要的一件事。"""

    id: str
    owner: str
    want: str
    resource: str
    intensity: int = 50
    #: 让步之后说的话（模板槽位：`{peer_name}`）。
    #: 放在动机里而不是人设卡里：同一句让步话在不同冲突里意思不同。
    yield_line: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Motive":
        return cls(
            id=str(data.get("id") or "motive"),
            owner=str(data.get("owner") or ""),
            want=str(data.get("want") or ""),
            resource=str(data.get("resource") or ""),
            intensity=int(data.get("intensity") or 50),
            yield_line=str(data.get("yield_line") or ""),
        )

    def render(self) -> str:
        return f"{self.owner} 想要「{self.want}」（{self.resource}，强度 {self.intensity}）"


@dataclass
class WorldEvent:
    """一次世界层面的变化。它不是某个 NPC 的动作，是**两个人之间**发生的事。"""

    id: str
    tick: int
    kind: str
    text: str
    winner: str
    loser: str
    loser_line: str = ""
    #: 争的是什么（"营业时间"）。台词要说"上次为营业时间…"，
    #: 从 `text` 里抠字符串不如直接带一个字段。
    resource: str = ""
    #: 让步那句话有没有真的说出口。说了就不再重复 ——
    #: 反复说"行，那就依你"是复读，不是涌现。
    spoken: bool = False

    def render(self) -> str:
        return f"[t{self.tick}] {self.kind}：{self.text}"


class MotiveBoard:
    """动机冲突的裁定台。由导演（`Cast`）持有。"""

    def __init__(self, motives: Iterable[Motive] = (), *, min_tick: int = 2) -> None:
        self.motives: list[Motive] = list(motives)
        #: 前几轮不裁定：开场就宣布"谁让步"会让世界显得很急。
        self.min_tick = min_tick
        #: 已经裁定过的冲突对（用 owner 排序后的元组当键）。
        self.resolved: set[tuple[str, str, str]] = set()
        self.events: list[WorldEvent] = []

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dicts(
        cls, raw: Iterable[dict[str, Any]] | None, *, min_tick: int = 2
    ) -> "MotiveBoard":
        return cls(
            (Motive.from_dict(item) for item in (raw or []) if isinstance(item, dict)),
            min_tick=min_tick,
        )

    def __len__(self) -> int:
        return len(self.motives)

    def __bool__(self) -> bool:
        return bool(self.motives)

    # ------------------------------------------------------------------ #
    def conflicts(self) -> list[tuple[Motive, Motive]]:
        """所有"同一个资源、不同主人"的动机对。"""
        out: list[tuple[Motive, Motive]] = []
        for i, a in enumerate(self.motives):
            for b in self.motives[i + 1 :]:
                if a.owner == b.owner:
                    continue
                if not a.resource or a.resource != b.resource:
                    continue
                out.append((a, b))
        return out

    @staticmethod
    def resolve(a: Motive, b: Motive) -> tuple[Motive, Motive]:
        """按强度裁定。**同强度时按 id 稳定排序** —— 结果必须可复现，
        否则同一段对话重放会得到不同的故事，评测就没法比。"""
        if a.intensity == b.intensity:
            return (a, b) if a.id <= b.id else (b, a)
        return (a, b) if a.intensity > b.intensity else (b, a)

    def poll(self, tick: int, names: dict[str, str] | None = None) -> list[WorldEvent]:
        """这一轮该发生的事件。同一对冲突只裁定一次。

        `names` 是 owner id → 显示名。事件文本会被印进控制台和报告，
        里面出现 `xiaozhou要…` 这种内部 id，读的人得先去查配置才知道是谁。
        """
        if tick < self.min_tick:
            return []
        shown = names or {}
        fresh: list[WorldEvent] = []
        for a, b in self.conflicts():
            key = tuple(sorted((a.owner, b.owner))) + (a.resource,)
            if key in self.resolved:
                continue
            self.resolved.add(key)
            winner, loser = self.resolve(a, b)
            w_name = shown.get(winner.owner, winner.owner)
            l_name = shown.get(loser.owner, loser.owner)
            event = WorldEvent(
                id=f"clash:{key[0]}:{key[1]}:{a.resource}",
                tick=tick,
                kind="动机冲突",
                text=(
                    f"{w_name}要「{winner.want}」，{l_name}要「{loser.want}」，"
                    f"{l_name}让步"
                ),
                winner=winner.owner,
                loser=loser.owner,
                loser_line=loser.yield_line,
                resource=a.resource,
            )
            self.events.append(event)
            fresh.append(event)
        return fresh

    def pending_for(self, actor_id: str) -> Optional[WorldEvent]:
        """这个人有没有"还没说出口"的让步。"""
        for event in reversed(self.events):
            if event.loser == actor_id and not event.spoken and event.loser_line:
                return event
        return None

    def render(self) -> str:
        if not self.motives:
            return "（没有配置动机）"
        lines = [f"  {m.render()}" for m in self.motives]
        if self.events:
            lines.append("  事件：")
            lines.extend(f"    {e.render()}" for e in self.events)
        return "\n".join(lines)


__all__ = ["Motive", "MotiveBoard", "WorldEvent"]
