"""game-npc-agent — 面向游戏场景的可控 AI NPC 智能体框架。

七大模块（对齐岗位 JD 第 2 条）：
    Planning / Memory / Tool Use / Action / Reflection / Persona / State Tracking

设计目标：
    1. 环境无关：Environment 抽象层可替换（内置文字世界 / Minecraft / 引擎内嵌）
    2. 模型无关：LLM 层可插拔，且在没有模型时也能端到端跑通（离线启发式回退）
    3. 可评测：所有行为可复现，带五维评测 harness，能出可写进简历的数字
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
