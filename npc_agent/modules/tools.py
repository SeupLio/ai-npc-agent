"""模块五：Tool Use —— 工具注册与执行。

工具分两类：
    世界工具（由 Environment 实现）—— move_to / craft_item / give_item ...
    内部工具（由 Agent 实现）      —— speak / remember

为什么要分？因为"说话"和"记住"不改变世界状态，但它们同样是 Agent 的**行动**。
把它们统一成工具之后，Planner 产出的计划可以混排语言动作和世界动作，
执行路径只有一条，Reflection 也能统一处理失败。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..env.base import Environment, ToolSpec
from ..types import ActionCall, ActionResult
from .memory import MemoryManager
from .persona import Persona
from .state import StateTracker

# 内部工具名（不由 Environment 提供）
INTERNAL_SPEAK = "speak"
INTERNAL_REMEMBER = "remember"
INTERNAL_TOOL_NAMES = {INTERNAL_SPEAK, INTERNAL_REMEMBER}

INTERNAL_SPECS = [
    ToolSpec(
        "speak",
        "对现场说一句话（受发言占比上限约束，超限会被拒绝）",
        {"text": "要说的话", "to": "可选，指定对谁说（玩家 id）"},
        ["speak(你好呀)", "speak(小鹿，你的拿铁好了, to=player_a)"],
        internal=True,
    ),
    ToolSpec(
        "remember",
        "把值得长期记住的事写进记忆（偏好、约定、关系变化）",
        {"content": "要记住的内容", "about": "可选，关联到谁（玩家 id）", "importance": "0-1"},
        ["remember(小鹿喜欢偏酸的豆子, about=player_a, importance=0.9)"],
        internal=True,
    ),
]


@dataclass
class ToolContext:
    """执行工具所需的上下文。"""

    actor_id: str
    tick: int
    env: Environment
    memory: MemoryManager
    persona: Persona
    tracker: StateTracker
    npc_share_ceiling: float = 0.62


class ToolRegistry:
    """把世界工具和内部工具合成一张表，对外只暴露 execute()。"""

    def __init__(self, env: Environment, config: Any | None = None) -> None:
        self.env = env
        self.config = config
        self._internal: dict[str, Callable[[dict[str, Any], ToolContext], ActionResult]] = {
            INTERNAL_SPEAK: self._speak,
            INTERNAL_REMEMBER: self._remember,
        }

    # ------------------------------------------------------------------ #
    def specs(self, actor_id: str) -> list[ToolSpec]:
        return list(self.env.tool_specs(actor_id)) + INTERNAL_SPECS

    def catalog(self, actor_id: str) -> str:
        return "\n".join(spec.render() for spec in self.specs(actor_id))

    def names(self, actor_id: str) -> set[str]:
        return {spec.name for spec in self.specs(actor_id)}

    # ------------------------------------------------------------------ #
    def execute(self, call: ActionCall, ctx: ToolContext) -> ActionResult:
        if call.tool in INTERNAL_TOOL_NAMES:
            handler = self._internal[call.tool]
            try:
                return handler(call.args or {}, ctx)
            except Exception as exc:  # 内部工具同样不抛异常出去
                return ActionResult(False, call.tool, f"内部工具异常: {type(exc).__name__}: {exc}")
        return self.env.dispatch(ctx.actor_id, call)

    # ------------------------------------------------------------------ #
    # 内部工具实现
    # ------------------------------------------------------------------ #
    def _speak(self, args: dict[str, Any], ctx: ToolContext) -> ActionResult:
        text = str(args.get("text", "")).strip()
        if not text:
            return ActionResult(False, INTERNAL_SPEAK, "没有内容可说")
        violations = ctx.persona.check(text, ctx.tracker.world_flags)
        if any(v.startswith("出戏词") or v.startswith("剧透") for v in violations):
            return ActionResult(
                False, INTERNAL_SPEAK, f"台词越界被拦下: {'; '.join(violations)}"
            )
        share = ctx.tracker.npc_share()
        if share > ctx.npc_share_ceiling and not args.get("to"):
            return ActionResult(
                False,
                INTERNAL_SPEAK,
                f"发言占比 {share:.0%} 已超上限，本轮主动让出话头",
            )
        styled = ctx.persona.apply_style(text)
        ctx.env.broadcast(ctx.actor_id, styled)
        return ActionResult(
            True,
            INTERNAL_SPEAK,
            styled,
            state_delta={"text": styled, "to": args.get("to")},
        )

    def _remember(self, args: dict[str, Any], ctx: ToolContext) -> ActionResult:
        content = str(args.get("content", "")).strip()
        if not content:
            return ActionResult(False, INTERNAL_REMEMBER, "没有要记住的内容")
        about = args.get("about")
        try:
            importance = float(args.get("importance", 0.8))
        except (TypeError, ValueError):
            importance = 0.8
        record = ctx.memory.remember(
            content=content, tick=ctx.tick, about=about, importance=importance
        )
        return ActionResult(
            True,
            INTERNAL_REMEMBER,
            f"记住了：{content}",
            state_delta={"memory_id": record.id},
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def parse_action(raw: dict[str, Any]) -> Optional[ActionCall]:
        """把 LLM 输出的 dict 转成 ActionCall，容错处理。"""
        tool = raw.get("tool") or raw.get("name")
        if not tool:
            return None
        args = raw.get("args") or raw.get("arguments") or {}
        if isinstance(args, str):
            args = {"value": args}
        if not isinstance(args, dict):
            args = {}
        return ActionCall(tool=str(tool), args=args, reason=str(raw.get("reason", "")))
