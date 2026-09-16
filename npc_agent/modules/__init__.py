"""Agent 的七大模块，一一对应岗位 JD 第 2 条。

    persona     Persona       人设与边界控制
    state       StateTracking 现场状态跟踪
    memory      Memory        记忆写入 / 巩固 / 检索
    planner     Planning      任务分解与重规划
    tools       ToolUse       工具注册与执行
    dialogue    Dialogue      多人发言权与收件人判定
    reflection  Reflection    失败归因与教训沉淀
"""

from .dialogue import AddresseeSelector, DialogueConfig, TurnManager
from .memory import MemoryManager, MemoryStore, estimate_importance
from .persona import Persona
from .planner import Planner
from .reflection import Reflection, Reflector
from .state import PlayerModel, StateTracker
from .tools import INTERNAL_SPEAK, ToolContext, ToolRegistry

__all__ = [
    "Persona",
    "StateTracker",
    "PlayerModel",
    "MemoryStore",
    "MemoryManager",
    "estimate_importance",
    "Planner",
    "ToolRegistry",
    "ToolContext",
    "INTERNAL_SPEAK",
    "AddresseeSelector",
    "DialogueConfig",
    "TurnManager",
    "Reflector",
    "Reflection",
]
