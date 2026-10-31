"""一对多注意力 —— 同时有好几个人在场时，先理谁。

## 两种"一对多"

米哈游公开方案里那句"在一对多的场景里知道该优先回应什么话题"，
拆开是两件事：

1. **同时有多个人想说话**：谁先被回应？
2. **有人问了但我没答**：什么时候补上？

第 2 条最容易被忽略，也最伤体验：玩家问了一句，NPC 正在做自己的事，
没答；过了一会儿 NPC 又开了个新话题 —— 玩家会觉得"它根本没在听"。
而它其实听见了，只是**没有一个地方记着"我欠他一个回答"**。

## 判据

优先级的组成（`score`）：

    被点名            +100   叫我名字的人永远优先
    是问句             +40   问题比陈述更需要回应
    等了多久          +5/轮  等得越久越该被理（封顶 +30）
    好感              ±10    熟人优先于生客

这些权重不是拍脑袋，但也不是量出来的 —— 它们是**产品判断**，
写在一处、可以被测试钉住，比散落在分支里的 if 好。
真正量过的是"有没有生效"：`living_world_bench` 里有一条用例专门
构造"两个人同时提问"，要求被点名/问句/等得久的那一位先被回应。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

#: 各个因素的分值。改这里就能改策略，不用改分支顺序。
W_MENTIONED = 100.0
W_QUESTION = 40.0
W_WAIT_PER_TICK = 5.0
W_WAIT_CAP = 30.0
W_AFFINITY = 0.2  # (affinity - 50) * 0.2 ⇒ 好感 0~100 对应 ±10


@dataclass
class QuestionDebt:
    """一个"问了但没被回答"的问题。

    它是**注意力**的一部分，不是记忆：记忆是"我记得你说过什么"，
    这条是"我欠你一个回答"，只活几轮，答完就销。
    """

    player_id: str
    text: str
    tick: int
    asked: int = 1

    def render(self) -> str:
        return f"{self.player_id}@t{self.tick}: {self.text}"


class AttentionBoard:
    """一个 NPC 的注意力账本。每个 NPC 一份（谁欠谁的话是私有的）。"""

    def __init__(self) -> None:
        self.debts: list[QuestionDebt] = []
        #: 补答过几次。报告用它证明"不忘记问题"这条真的在跑。
        self.follow_ups = 0

    # ------------------------------------------------------------------ #
    def note_question(self, player_id: str, text: str, tick: int) -> QuestionDebt:
        """记下一条没答的问题。同一个人重复问同一句 → 只累加次数。"""
        for debt in self.debts:
            if debt.player_id == player_id and debt.text == text:
                debt.asked += 1
                debt.tick = tick
                return debt
        debt = QuestionDebt(player_id=player_id, text=text, tick=tick)
        self.debts.append(debt)
        return debt

    def settle(self, player_id: str) -> None:
        """这个人的问题答完了（或他换了话题）→ 销账。"""
        self.debts = [d for d in self.debts if d.player_id != player_id]

    def settle_one(self, player_id: str) -> None:
        for i, debt in enumerate(self.debts):
            if debt.player_id == player_id:
                del self.debts[i]
                self.follow_ups += 1
                return

    def oldest(self, now: int, min_wait: int = 2) -> Optional[QuestionDebt]:
        """等得最久、且已经等了至少 `min_wait` 轮的那条。

        `min_wait` 不能是 0：同一轮里刚问完就"补答"，会变成
        "你问的我已经答了"——那不是补答，是复读。
        """
        ready = [d for d in self.debts if now - d.tick >= min_wait]
        if not ready:
            return None
        return min(ready, key=lambda d: d.tick)

    # ------------------------------------------------------------------ #
    def score(self, player: Any, now: int, *, mentioned: bool = False) -> float:
        """一个人的"该被理"程度。`player` 是 `state.PlayerModel`。"""
        waited = min(W_WAIT_CAP, max(0, now - getattr(player, "last_spoke_tick", now)) * W_WAIT_PER_TICK)
        affinity = getattr(player, "affinity", 50)
        return (
            (W_MENTIONED if mentioned else 0.0)
            + (W_QUESTION if getattr(player, "pending_question", False) else 0.0)
            + waited
            + (affinity - 50) * W_AFFINITY
        )

    def rank(self, players: dict[str, Any], now: int) -> list[str]:
        """按"该被理"程度排序的 player id。分数相同时按 id 稳定排序。"""
        return sorted(
            players,
            key=lambda pid: (-self.score(players[pid], now), pid),
        )

    def render(self) -> str:
        if not self.debts:
            return "（没有欠着的回答）"
        return "\n".join(f"  {d.render()}" for d in self.debts)


__all__ = ["AttentionBoard", "QuestionDebt"]
