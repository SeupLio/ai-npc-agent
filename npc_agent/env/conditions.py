"""目标完成条件的判定 —— 所有环境共用。

为什么要把这段逻辑单独抽出来？
因为"目标完成"的定义是**任务设计**的一部分，不是**世界实现**的一部分。
咖啡屋的 `player_has` 和 Minecraft 的 `player_has_count` 问的是同一件事：
"玩家手里真的出现了那样东西吗？"

如果每个环境各抄一份，第二个环境就会慢慢长出自己的方言：
A 环境里 `all_flags` 是"全部达成"，B 环境里被写成"任一达成"；
A 环境空列表返回 False，B 环境返回 True（`all([])` 的陷阱）。
这种不一致不会报错，只会让评测数字悄悄失去可比性。

所以判定逻辑只写一遍，环境只负责**提供事实**：
    - 世界标记有哪些
    - 谁手里有什么、有多少
    - 谁说过几句话

`ConditionContext` 就是这份"事实清单"。它刻意不依赖任何环境对象，
于是这个模块可以脱离世界单测 —— 见 tests/test_conditions.py。

条件词汇表
----------
    {flag: X}                         某个世界标记被设置
    {all_flags: [X, Y]}               多个标记全部达成
    {any_flags: [X, Y]}               任一标记达成
    {all_of: [cond, ...]}             多个条件**全部**成立（可嵌套）
    {any_of: [cond, ...]}             任一条件成立（可嵌套）
    {all_players_spoke: N}            每位玩家都至少说过 N 次话
    {player_has: {pid: [item]}}       某位玩家的背包里真的出现了某样东西
    {player_has_count: {pid: {i: N}}} 数量版（Minecraft 里"3 块木头"和"1 块木头"不同）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Collection, Mapping


@dataclass
class ConditionContext:
    """判定条件所需的事实。

    环境在每次刷新目标状态时构造一个，用完即弃 —— 它是一份快照，
    不是一个活的对象引用。这样判定过程不会意外改到世界。
    """

    #: 当前已设置的世界标记
    flags: Collection[str] = ()
    #: actor_id -> 背包里的物品（只关心有没有，不关心数量）
    inventories: Mapping[str, Collection[str]] = field(default_factory=dict)
    #: actor_id -> {item: 数量}。有数量信息时优先用它，见 count_item()
    quantities: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    #: speaker_id -> 说过多少句
    speech_counts: Mapping[str, int] = field(default_factory=dict)
    #: 哪些 actor 算"玩家"。all_players_spoke 只考察这些人
    player_ids: Collection[str] = ()

    # ------------------------------------------------------------------ #
    # 事实查询（叶子条件都走这几个方法，方便子类覆盖语义）
    # ------------------------------------------------------------------ #
    def has_flag(self, flag: str) -> bool:
        return flag in self.flags

    def count_item(self, actor_id: str, item: str) -> int:
        """某位 actor 手里有多少个 item。

        数量信息优先：Minecraft 的背包天生带数量，而咖啡屋的背包是个列表
        （"阿柚拿着咖啡豆"，不问几颗）。两种世界共用同一个函数，
        靠 `quantities` 里有没有这个 actor 来决定走哪条路。
        """
        if actor_id in self.quantities:
            return int(self.quantities[actor_id].get(item, 0) or 0)
        return sum(1 for held in self.inventories.get(actor_id, ()) if held == item)

    def has_item(self, actor_id: str, item: str) -> bool:
        return self.count_item(actor_id, item) > 0

    def speech_count(self, actor_id: str) -> int:
        return int(self.speech_counts.get(actor_id, 0) or 0)

    def knows_actor(self, actor_id: str) -> bool:
        return actor_id in self.inventories or actor_id in self.quantities


# --------------------------------------------------------------------------- #
# 叶子条件
# --------------------------------------------------------------------------- #
def flag_met(condition: Mapping[str, Any], ctx: ConditionContext) -> bool:
    return ctx.has_flag(str(condition["flag"]))


def all_flags_met(condition: Mapping[str, Any], ctx: ConditionContext) -> bool:
    needed = list(condition["all_flags"] or [])
    return bool(needed) and all(ctx.has_flag(str(f)) for f in needed)


def any_flags_met(condition: Mapping[str, Any], ctx: ConditionContext) -> bool:
    needed = list(condition["any_flags"] or [])
    return bool(needed) and any(ctx.has_flag(str(f)) for f in needed)


def all_players_spoke_met(condition: Mapping[str, Any], ctx: ConditionContext) -> bool:
    needed = int(condition["all_players_spoke"])
    players = list(ctx.player_ids)
    if not players:
        # 没有玩家就没有"每位玩家都说过话"这回事。
        # 返回 True 会让空场景的目标开局即完成，那是个很难查的 bug。
        return False
    return all(ctx.speech_count(pid) >= needed for pid in players)


def player_has_met(condition: Mapping[str, Any], ctx: ConditionContext) -> bool:
    wanted = condition["player_has"] or {}
    # 空要求不是"自动满足"。和 all_flags: [] 一样：一个不表达任何约束的条件
    # 如果算达成，目标就会开局即完成，而且不会报任何错。
    if not wanted:
        return False
    for actor_id, items in wanted.items():
        if not ctx.knows_actor(actor_id):
            return False
        if any(not ctx.has_item(actor_id, str(item)) for item in items):
            return False
    return True


def player_has_count_met(condition: Mapping[str, Any], ctx: ConditionContext) -> bool:
    """数量版 `player_has`：`{player_has_count: {player_a: {oak_log: 3}}}`。"""
    wanted = condition["player_has_count"] or {}
    if not wanted:
        return False
    for actor_id, counts in wanted.items():
        if not ctx.knows_actor(actor_id):
            return False
        for item, amount in (counts or {}).items():
            if ctx.count_item(actor_id, str(item)) < int(amount):
                return False
    return True


# --------------------------------------------------------------------------- #
# 组合条件（递归）
# --------------------------------------------------------------------------- #
def all_of_met(condition: Mapping[str, Any], ctx: ConditionContext) -> bool:
    subs = list(condition["all_of"] or [])
    # 空列表不能算"全部达成" —— all([]) 是 True，那意味着一个写错的条件
    # 会静默地让目标立刻完成。
    return bool(subs) and all(condition_met(c, ctx) for c in subs)


def any_of_met(condition: Mapping[str, Any], ctx: ConditionContext) -> bool:
    subs = list(condition["any_of"] or [])
    return bool(subs) and any(condition_met(c, ctx) for c in subs)


#: 分发顺序即优先级。`all_of` / `any_of` 排在最前面，因为它们可以嵌套别的条件。
_LEAVES: tuple[tuple[str, Any], ...] = (
    ("all_of", all_of_met),
    ("any_of", any_of_met),
    ("flag", flag_met),
    ("all_flags", all_flags_met),
    ("any_flags", any_flags_met),
    ("all_players_spoke", all_players_spoke_met),
    ("player_has", player_has_met),
    ("player_has_count", player_has_count_met),
)

#: 条件关键字，供测试与文档核对，避免有人加了分支却忘了登记
CONDITION_KINDS: tuple[str, ...] = tuple(key for key, _ in _LEAVES)

#: "配置写坏了"这一类异常。刻意不含 Exception：
#: 真的代码 bug 应该炸出来，而不是被伪装成"目标未达成"。
_MALFORMED = (TypeError, ValueError, AttributeError, KeyError, IndexError)


def condition_met(condition: Mapping[str, Any] | None, ctx: ConditionContext) -> bool:
    """判定一个 success_when 条件是否成立。

    两种情况都返回 False 而不是抛异常，理由是一样的：
    **一份写坏的目标定义，应该表现为"这个目标永远完不成"。**

    这在评测里是能被一眼看见的（完成率 0%），而抛异常会让整场跑挂掉 ——
    一次跑了 20 分钟的真实模型对照，不该因为 YAML 里少了个缩进就白跑。
    所以未知关键字、类型写错（把 `all_players_spoke: 1` 写成列表）都只是
    "完不成"，不是崩溃。
    """
    if not condition:
        return False
    for key, predicate in _LEAVES:
        if key in condition:
            try:
                return bool(predicate(condition, ctx))
            except _MALFORMED:
                return False
    return False


def refresh_objectives(
    specs: Collection[Mapping[str, Any]],
    state: dict[str, str],
    ctx: ConditionContext,
    *,
    done: str = "done",
) -> dict[str, str]:
    """把世界事实与目标定义对一遍，把已达成者标记为 done。

    只做"未达成 → 达成"的单向推进：目标一旦完成就不该因为世界后来变化
    （玩家把咖啡喝掉了）而被撤销 —— 那是"曾经做到过"的记录。
    """
    for spec in specs:
        if condition_met(spec.get("success_when") or {}, ctx):
            state[spec["id"]] = done
    return state


def unmet_reasons(condition: Mapping[str, Any] | None, ctx: ConditionContext) -> list[str]:
    """列出**还没满足**的叶子条件，用于调试与失败报告。

    只展开没达成的分支：已经达成的部分没有诊断价值。
    """
    if not condition:
        return ["条件为空"]
    # 一个条件字典只会命中一个关键字，所以找到就返回，不用继续扫。
    for key, _ in _LEAVES:
        if key not in condition:
            continue
        if key == "all_of":
            # 全部成立才算达成：把每个未达成子条件的原因都摊开
            out: list[str] = []
            for sub in condition[key] or []:
                out.extend(unmet_reasons(sub, ctx))
            return out
        if key == "any_of":
            subs = list(condition[key] or [])
            groups = [unmet_reasons(sub, ctx) for sub in subs]
            # 只要有一个子条件达成，any_of 就算达成 —— 没达成的分支不再是"原因"
            if any(not group for group in groups):
                return []
            out = []
            for group in groups:
                out.extend(group)
            return out
        if condition_met(condition, ctx):
            return []
        # 渲染本身也不能崩：诊断信息的 traceback 会盖住真正的错误信息，
        # 那比没有诊断信息更糟。
        try:
            return [_render_leaf(key, condition)]
        except _MALFORMED:
            return [f"{key}: （条件格式无法解析）"]
    return ["未知条件类型"]


def _render_leaf(key: str, condition: Mapping[str, Any]) -> str:
    if key in ("all_flags", "any_flags"):
        return f"{key}: {'、'.join(str(f) for f in (condition[key] or []))}"
    if key == "all_players_spoke":
        return f"每位玩家至少发言 {condition[key]} 次"
    if key == "player_has":
        # player_has 的值是 {pid: [item, ...]}，列表
        parts = [
            f"{pid} 需要 {'、'.join(map(str, items or []))}"
            for pid, items in (condition[key] or {}).items()
        ]
        return f"player_has: {'; '.join(parts)}"
    if key == "player_has_count":
        # player_has_count 的值是 {pid: {item: 数量}}，字典
        parts = [
            f"{pid} 需要 {'、'.join(f'{item}×{n}' for item, n in (items or {}).items())}"
            for pid, items in (condition[key] or {}).items()
        ]
        return f"player_has_count: {'; '.join(parts)}"
    return f"{key}: {condition[key]}"
