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
    """世界接口。所有环境实现这 5 个方法 + 2 个可选钩子。"""

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
