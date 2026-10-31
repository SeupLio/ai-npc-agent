"""关系养成 —— 好感、阶段、专属称号。

## 为什么这个模块必须存在（而不是"再加一个数值"）

环境里**早就有**好感度：`give_item` 给 +4，`judge_answer` 给 +6，
`PlayerModel.affinity` 也会被读进来、显示在状态里。
但**没有任何一个决策读它** —— 也就是说，好感在这套框架里
只是一个"看起来在动"的数字，玩家做任何事都不会改变 NPC 怎么对待他。

这是这个仓库里最忌讳的那类东西：**一个有数值、没消费方的功能**。
它比没有更糟 —— 调参的人会以为自己能改变体验。

所以这个模块做两件事：

1. 把好感翻译成**可观测的行为差异**（称呼、问候方式、称号）；
2. 把"称号"做成**一次性事件**，而不是一个永远挂着的标签。

## 称号为什么要有解锁事件

米哈游公开方案里提到"好感度越高，称号越专属、越稀有"。
但称号如果只是数据库里多一行，玩家永远不知道它发生过 ——
真正的体验是**NPC 当面把那句话说出口**：

    「从今天起，你是这家店的常客。」

所以解锁时会**攒一条待播报事件**，下一次跟这个人说话时优先说出来。
说完就清掉 —— 同一个称号不播报两次（那会变成复读）。

## 阶段阈值

    0–55   生客
    56–67  面熟
    68–84  常客
    85–100 老友

阈值不是拍的：环境里初始好感就是 **50**，一次递饮品 +4、答对一题 +6、
普通互动 +1。所以下限必须从 56 起 —— 从 30 起的话，第一次见面的人
一进门就是"面熟"，还会当场领称号（实测过，一眼假）。
现在 50 落在"生客"，两次有效互动（58）才到"面熟"，
一场 demo 的长度刚好够看到一次升级。
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: (下限, 阶段名)，从高到低匹配。
#:
#: ⚠️ 下限**不能从 30 开始**：环境给的初始好感就是 50，
#: 于是"第一次见面的人"一进门就是「面熟」，还会当场领到称号。
#: 实测（duet）第一轮小舟就对新客人说「记住这个：熟面孔。阿澈。」
#: —— 比没有关系系统更糟，因为它一眼假。
#: 现在的阈值让 50 落在「生客」区间：一次递饮品（+4）不够，
#: 两次（58）才到「面熟」。
STAGES: tuple[tuple[int, str], ...] = (
    (85, "老友"),
    (68, "常客"),
    (56, "面熟"),
    (0, "生客"),
)

#: 阶段 → 称号。称号比阶段名更"专属"一点（"常客"是阶段，"记得你口味的人"是称号）。
STAGE_TITLES: dict[str, str] = {
    "面熟": "熟面孔",
    "常客": "常客",
    "老友": "老朋友",
}

#: 额外称号：达成条件才给，和阶段无关。
#: 结构：(id, 需要的阶段, 需要的最少互动次数, 称号文本)
SPECIAL_TITLES: tuple[tuple[str, str, int, str], ...] = (
    ("taste_keeper", "常客", 6, "记得你口味的人"),
    ("night_regular", "老友", 10, "夜里也来的人"),
)


def stage_of(score: int) -> str:
    for floor, name in STAGES:
        if score >= floor:
            return name
    return STAGES[-1][1]


def stage_rank(stage: str) -> int:
    """阶段的序号，越大越亲近。未知阶段按最低算。"""
    order = [name for _, name in reversed(STAGES)]
    return order.index(stage) if stage in order else 0


@dataclass
class Bond:
    """一个 NPC 对一个人的关系档案。"""

    player_id: str
    score: int = 50
    interactions: int = 0
    stage: str = "生客"
    titles: list[str] = field(default_factory=list)
    last_tick: int = -99

    def render(self) -> str:
        title = f"，称号「{self.titles[-1]}」" if self.titles else ""
        return f"{self.player_id}：{self.stage}（好感 {self.score}，互动 {self.interactions} 次{title}）"


@dataclass
class RelationshipEvent:
    """一次关系变化。`kind` 决定它该怎么被消费。"""

    player_id: str
    kind: str  # stage_up | title
    detail: str
    tick: int

    def render(self) -> str:
        if self.kind == "title":
            return f"称号解锁：{self.detail}"
        return f"关系升级：{self.detail}"


class RelationshipBook:
    """一个 NPC 的关系账本。**每个 NPC 一本**，不共享 ——
    "阿柚和小舟对同一个人的好感"本来就是两件事。"""

    def __init__(self, npc_id: str = "", npc_name: str = "") -> None:
        self.npc_id = npc_id
        self.npc_name = npc_name
        self.bonds: dict[str, Bond] = {}
        #: 还没播报出去的事件。称号必须**说出口**才算数，见模块注释。
        self.pending: list[RelationshipEvent] = []
        #: 累计解锁过几个称号（报告用）。
        self.titles_granted = 0

    # ------------------------------------------------------------------ #
    def bond(self, player_id: str) -> Bond:
        return self.bonds.setdefault(player_id, Bond(player_id))

    def sync(
        self, player_id: str, affinity: int, tick: int, *, active: bool = True
    ) -> list[RelationshipEvent]:
        """把环境里的好感同步进来，返回**这次新发生**的事件。

        好感是环境的（它知道递东西、答对题），阶段和称号是这里的。
        两边各存一份就会漂移 —— 所以这里只做"读环境 + 派生"，
        不自己维护一个平行分数。

        `active` 表示"这个人这一轮真的跟我有互动"（说了话，或好感变了）。
        **不能按"在场回合数"计数**：那样站够 6 轮就能拿"记得你口味的人"，
        而 NPC 其实一句话都没跟他说过。
        """
        bond = self.bond(player_id)
        events: list[RelationshipEvent] = []
        gained = int(affinity) - bond.score
        bond.score = int(affinity)
        if active or gained:
            bond.interactions += 1
        bond.last_tick = tick

        new_stage = stage_of(bond.score)
        if stage_rank(new_stage) > stage_rank(bond.stage):
            bond.stage = new_stage
            events.append(
                RelationshipEvent(player_id, "stage_up", new_stage, tick)
            )
            title = STAGE_TITLES.get(new_stage)
            if title:
                events.extend(self._grant(player_id, title, tick))

        for tid, need_stage, need_count, title in SPECIAL_TITLES:
            if title in bond.titles:
                continue
            if stage_rank(bond.stage) >= stage_rank(need_stage) and bond.interactions >= need_count:
                events.extend(self._grant(player_id, title, tick))
                _ = tid  # id 只用于配置可读性；去重靠称号文本

        self.pending.extend(events)
        # 只留最近 20 条：`pending` 是"还没播报的"，不是日志。
        # 不封顶的话，一场长会话里它会一直长（stage_up 事件没人消费）。
        if len(self.pending) > 20:
            del self.pending[:-20]
        return events

    def _grant(self, player_id: str, title: str, tick: int) -> list[RelationshipEvent]:
        bond = self.bond(player_id)
        if title in bond.titles:
            return []
        bond.titles.append(title)
        self.titles_granted += 1
        return [RelationshipEvent(player_id, "title", title, tick)]

    # ------------------------------------------------------------------ #
    def stage(self, player_id: str) -> str:
        return self.bond(player_id).stage

    def title_for(self, player_id: str) -> str:
        bond = self.bonds.get(player_id)
        return bond.titles[-1] if bond and bond.titles else ""

    def take_pending(self, player_id: str) -> RelationshipEvent | None:
        """取走这个人待播报的事件（称号优先）。取走即消费。"""
        for i, event in enumerate(self.pending):
            if event.player_id != player_id:
                continue
            if event.kind != "title":
                continue
            return self.pending.pop(i)
        return None

    def is_returning(self, player_id: str) -> bool:
        """是不是"回来的人"。判据是阶段，不是"说过话没有" ——
        第一次来就说三句话的人仍然是生客。"""
        return stage_rank(self.stage(player_id)) >= stage_rank("面熟")

    def bond_block(self) -> str:
        """拼进 prompt 的关系摘要。没有档案时返回空串。"""
        if not self.bonds:
            return ""
        lines = []
        for bond in self.bonds.values():
            title = f"，你叫他「{bond.titles[-1]}」" if bond.titles else ""
            lines.append(f"  - {bond.player_id}：{bond.stage}，好感 {bond.score}{title}")
        return "【你认识的人】\n" + "\n".join(lines)

    def render(self) -> str:
        if not self.bonds:
            return "（还没有关系档案）"
        return "\n".join(f"  {b.render()}" for b in self.bonds.values())


__all__ = [
    "Bond",
    "RelationshipBook",
    "RelationshipEvent",
    "SPECIAL_TITLES",
    "STAGES",
    "STAGE_TITLES",
    "stage_of",
    "stage_rank",
]
