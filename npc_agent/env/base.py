"""环境抽象层。

这是整个项目最关键的架构决策：**把"世界"和"智能体"彻底解耦**。

NPC 的决策逻辑完全不知道自己在咖啡屋、Minecraft 还是引擎里。
它只面对两件事：
    1. observe()  —— 我看到什么
    2. dispatch() —— 我做了什么，世界怎么回应

好处：
- 换环境 = 写一个新的 Environment 子类，Agent 一行不改
- 评测可复现：文字世界是确定性的，能跑回归
- 工具调用失败会返回结构化原因，直接喂给 Reflection 模块

新增环境（例如 Minecraft）只需实现下面 5 个方法。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..types import ActionCall, ActionResult
from .conditions import ConditionContext, condition_met, refresh_objectives


@dataclass
class ToolSpec:
    """暴露给 Agent / LLM 的工具描述。"""

    name: str
    description: str
    params: dict[str, str] = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)
    internal: bool = False  # True 表示 Agent 内部工具（不改变世界状态）

    def render(self) -> str:
        if self.params:
            signature = ", ".join(f"{k}: {v}" for k, v in self.params.items())
        else:
            signature = ""
        return f"- {self.name}({signature}): {self.description}"


class Environment(ABC):
    """世界接口。所有环境实现这 5 个方法 + 若干可选钩子。

    约定：环境把目标定义放在 `self.objective_specs`（`[{id, goal, success_when, ...}]`），
    完成情况放在 `self.objective_state`（`{id: "pending"|"done"}`）。
    判定本身由基类完成 —— 环境只负责**提供事实**（`condition_context()`），
    不负责判断"算不算完成"。见 env/conditions.py 里的说明。
    """

    name: str = "env"

    # ------------------------------------------------------------------ #
    # 必须实现
    # ------------------------------------------------------------------ #
    @abstractmethod
    def reset(self) -> dict[str, Any]:
        """回到初始状态，返回初始观测。评测里每条用例都会调用。"""

    @abstractmethod
    def observe(self, actor_id: str) -> dict[str, Any]:
        """返回 actor 当前能看到的世界切片。"""

    @abstractmethod
    def dispatch(self, actor_id: str, call: ActionCall) -> ActionResult:
        """执行一次工具调用。**不允许抛异常**，失败要返回 ok=False + 原因。"""

    @abstractmethod
    def tool_specs(self, actor_id: str) -> list[ToolSpec]:
        """当前 actor 可用的工具清单（可随状态变化，例如权限解锁）。"""

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """完整世界状态快照，用于评测断言和调试回放。"""

    # ------------------------------------------------------------------ #
    # 可选钩子（有默认实现）
    # ------------------------------------------------------------------ #
    def condition_context(self) -> ConditionContext:
        """提供判定目标完成所需的事实。

        环境只负责说清楚"世界现在是什么样"：有哪些标记、谁手里有什么、谁说了几句。
        "算不算完成"由 env/conditions.py 统一判定 —— 所以新增环境不需要再写一遍
        条件求值，也不会和别的环境产生语义分歧。
        """
        return ConditionContext()

    def objectives_status(self) -> dict[str, str]:
        """刷新并返回目标完成情况。默认实现走共享判定器。

        注意用 getattr 而不是类属性默认值：类级别的可变默认值会被所有实例共享，
        一个环境的完成状态会渗进另一个环境。这里宁可多写一层。
        """
        specs = getattr(self, "objective_specs", ())
        state = getattr(self, "objective_state", None)
        if state is None:
            state = {}
            self.objective_state = state
        return dict(refresh_objectives(specs, state, self.condition_context()))

    def _condition_met(self, condition: dict[str, Any] | None) -> bool:
        """判定单个条件。保留这个方法名，因为已有环境与测试都在用它。"""
        return condition_met(condition, self.condition_context())

    def broadcast(self, actor_id: str, text: str) -> None:
        """把一次发言广播进世界，让其他 Agent / 玩家能观察到。"""

    def advance_tick(self) -> None:
        """时间推进一格。"""

    def world_facts(self) -> dict[str, Any]:
        """暴露世界规则（配方表、位置表等），供 Planner 做离线启发式规划。

        有真实模型时 Planner 走 prompt 路径，不依赖这个。
        """
        return {}

    def available_topics(self, actor_id: str) -> list[str]:
        """当前 actor 可以安全透露的话题（受知识边界与解锁状态约束）。"""
        return []
