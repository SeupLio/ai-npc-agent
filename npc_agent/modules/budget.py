"""成本与循环管控 —— 三个"事后才想起要有"的开关。

## 为什么这三件事必须放在一个模块里

它们看起来是三件事，其实是**同一次事故的三个侧面**：

> 几十个 Agent 一起跑，没人给 token 设上限，连续跑了 13 小时，
> 烧掉价值 200 万元的调用；事后补上的是**成本管控、循环检测、上下文裁剪**。

拆开看：

    成本管控   单轮 / 单会话的调用上限      —— 防止"没有刹车"
    循环检测   同一串动作重复 N 次就停      —— 防止"踩着油门空转"
    上下文裁剪 喂给模型的记忆只留最相关的一截 —— 防止"每转一圈更贵"

第三条最容易被忽略：前两条管的是"跑多久"，它管的是"每一轮有多贵"。
一个 20 轮的会话，如果每轮都把全部记忆塞进 prompt，第 20 轮的单价
是第 1 轮的几倍 —— 而这件事在账单上表现为"总价高"，看不出原因。

## 判据为什么必须是"动作序列"而不是"轮数"

按轮数限制会误伤正常的长对话：玩家就是聊了 40 轮，那不该被拦。
真正该拦的是**原地打转** —— 同一串动作重复出现。所以循环检测看的是
`(tool, args)` 序列的重复次数，不是轮数。

阈值 3 是量出来的下限：一次失败后重规划会重试 1 次
（`max_plan_retries`），加上首次执行，正常路径最多出现 2 次同样的动作。
第 3 次还在做同一件事，就不是"重试"，是"没在往前走"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------- #
# 成本
# --------------------------------------------------------------------------- #
@dataclass
class TurnBudget:
    """一场会话的调用预算。

    ⚠️ token 是**估算**，不是精确值：仓库里没有 tokenizer（那会引入一个
    依赖，破坏"clone 下来就能跑"）。估算口径是"字符数"，
    对中文大致 1 字 ≈ 1 token，对英文会高估约 2 倍。
    所以这个数只用来**兜底**（"别跑到失控"），不能用来做成本核算 ——
    报告里凡引用它，都必须写明是估算。
    """

    max_calls_per_turn: int = 2
    max_session_tokens: int = 200_000
    calls_this_turn: int = 0
    calls_total: int = 0
    tokens_total: int = 0
    #: 因为预算被挡下的次数。它是"刹车在工作"的证据，不是失败。
    denied: int = 0

    def start_turn(self) -> None:
        self.calls_this_turn = 0

    def allow_call(self) -> bool:
        """这一轮还能不能调模型。不能 → 记一次 denied 并返回 False。"""
        if self.calls_this_turn >= self.max_calls_per_turn:
            self.denied += 1
            return False
        if self.tokens_total >= self.max_session_tokens:
            self.denied += 1
            return False
        return True

    def charge(self, chars: int) -> None:
        self.calls_this_turn += 1
        self.calls_total += 1
        self.tokens_total += max(0, int(chars))

    def render(self) -> str:
        return (
            f"调用 {self.calls_total} 次 / 估算 {self.tokens_total} 字符"
            f"（上限 {self.max_session_tokens}），挡下 {self.denied} 次"
        )


# --------------------------------------------------------------------------- #
# 循环检测
# --------------------------------------------------------------------------- #
@dataclass
class LoopGuard:
    """同一串动作重复出现 → 判定为原地打转。

    `signature` 由调用方给（通常是 `(tool, 关键参数)` 的元组）。
    本模块不解释它 —— 解释权在产出方，和 `ActionResult.outcome` 同一条原则。
    """

    window: int = 12
    repeats: int = 3
    _history: list[Any] = field(default_factory=list)
    tripped: bool = False
    tripped_signature: Any = None
    #: 触发次数。报告用它证明"这条护栏真的响过"。
    trips: int = 0

    def observe(self, signature: Any) -> bool:
        """记一次动作，返回"是不是判定成循环了"。

        判定口径：**末尾出现了重复的周期块** —— 周期 1~3，重复 `repeats` 次。

            A A A                 周期 1 → 死循环
            A B A B A B           周期 2 → 两个动作互相踢皮球
            A B C A B C A B C     周期 3

        ⚠️ 这里**不能**写成"同一个 signature 在窗口里出现 3 次"。
        第一版就是那么写的，它把一条正常的做单计划判成了死循环：
        `move_to(吧台) → move_to(后厨) → … → move_to(吧台) → give_item`
        里 `move_to(吧台)` 出现 3 次，而那 3 次之间都夹着别的动作 ——
        计划被拦腰砍断，咖啡永远递不出去（离线 235 条里掉了 1 条）。
        **"重复"和"反复经过同一个地方"是两件事。**
        """
        self._history.append(signature)
        if len(self._history) > self.window:
            del self._history[:-self.window]
        history = self._history
        for period in (1, 2, 3):
            need = period * self.repeats
            if len(history) < need:
                continue
            tail = history[-need:]
            block = tail[:period]
            if all(tail[i] == block[i % period] for i in range(need)):
                self.tripped = True
                self.tripped_signature = signature
                self.trips += 1
                return True
        return False

    def reset(self) -> None:
        self._history.clear()
        self.tripped = False
        self.tripped_signature = None

    def render(self) -> str:
        if not self.trips:
            return "没有触发"
        return f"触发 {self.trips} 次，最后一次是 {self.tripped_signature}"


# --------------------------------------------------------------------------- #
# 上下文裁剪
# --------------------------------------------------------------------------- #
def trim_records(
    records: Sequence[Any],
    max_chars: int,
    *,
    pinned: Iterable[Any] = (),
) -> tuple[list[Any], int]:
    """把注入 prompt 的记忆裁到 `max_chars` 以内。返回 `(保留的, 丢掉几条)`。

    裁剪顺序：**先保 pinned（例如刚回引过的那条），再按传入顺序取**。
    传入顺序是检索打分排好的，所以"保前面的"就是"保最相关的"。

    为什么不按条数裁：一条巩固摘要可能有 300 字，一条普通记录 20 字。
    按条数裁，20 条摘要能撑爆预算，20 条短句又浪费额度。
    """
    kept: list[Any] = []
    dropped = 0
    used = 0
    pinned_ids = {id(item) for item in pinned}
    ordered = [item for item in records if id(item) in pinned_ids]
    ordered += [item for item in records if id(item) not in pinned_ids]
    for record in ordered:
        text = str(getattr(record, "content", record))
        if kept and used + len(text) > max_chars:
            dropped += 1
            continue
        kept.append(record)
        used += len(text)
    return kept, dropped


__all__ = ["LoopGuard", "TurnBudget", "trim_records"]
