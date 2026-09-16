"""贯穿整条 Agent 流水线的核心数据类。

刻意保持"纯数据 + 少量渲染方法"，不依赖任何模块，避免循环导入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Role = Literal["npc", "player", "system"]
MemoryKind = Literal["episodic", "semantic", "reflection"]
StepStatus = Literal["pending", "running", "done", "failed"]


# --------------------------------------------------------------------------- #
# 对话
# --------------------------------------------------------------------------- #
@dataclass
class Utterance:
    """一次发言。玩家和 NPC 都走这个结构，方便统一做收件人判定。"""

    speaker_id: str
    speaker_name: str
    text: str
    tick: int
    role: Role = "player"
    mentions: list[str] = field(default_factory=list)
    is_question: bool = False
    addressed_to: Optional[str] = None

    def render(self) -> str:
        return f"{self.speaker_name}: {self.text}"


# --------------------------------------------------------------------------- #
# 行动
# --------------------------------------------------------------------------- #
@dataclass
class ActionCall:
    """一次工具调用意图。Agent 产出它，Environment 负责执行。"""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def render(self) -> str:
        if not self.args:
            return f"{self.tool}()"
        inner = ", ".join(f"{k}={v}" for k, v in self.args.items())
        return f"{self.tool}({inner})"


@dataclass
class ActionResult:
    """工具执行结果。失败时 detail 会作为 Reflection 的输入信号。"""

    ok: bool
    tool: str
    detail: str = ""
    state_delta: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        mark = "ok" if self.ok else "FAIL"
        return f"[{mark}] {self.tool}: {self.detail}"


# --------------------------------------------------------------------------- #
# 规划
# --------------------------------------------------------------------------- #
@dataclass
class PlanStep:
    goal: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    status: StepStatus = "pending"
    note: str = ""


@dataclass
class Plan:
    goal: str = ""
    rationale: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    objective_id: str = ""  # 关联的场景目标 id（用于"只尝试一次"的判定）

    @property
    def next_step(self) -> Optional[PlanStep]:
        for step in self.steps:
            if step.status in ("pending", "running"):
                return step
        return None

    @property
    def done(self) -> bool:
        return all(s.status in ("done", "failed") for s in self.steps)

    @property
    def failed_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status == "failed"]

    def render(self) -> str:
        lines = [f"目标: {self.goal}"]
        marks = {"pending": " ", "running": "~", "done": "x", "failed": "!"}
        for i, step in enumerate(self.steps, 1):
            lines.append(f"  [{marks[step.status]}] {i}. {step.goal} -> {step.tool}({step.args})")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 记忆
# --------------------------------------------------------------------------- #
@dataclass
class MemoryRecord:
    id: str
    kind: MemoryKind
    content: str
    tick: int
    importance: float = 0.5
    entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    last_access: int = 0
    access_count: int = 0
    score: float = 0.0

    def render(self) -> str:
        return f"[{self.kind}@t{self.tick}] {self.content}"


# --------------------------------------------------------------------------- #
# 对话调度（多人场景的核心）
# --------------------------------------------------------------------------- #
@dataclass
class DialogueDecision:
    """本轮"我要不要说话 / 对谁说"的判定结果。"""

    should_speak: bool
    addressed_to: Optional[str] = None
    reason: str = ""
    urgency: float = 0.0
    proactive: bool = False

    def render(self) -> str:
        act = "发言" if self.should_speak else "沉默"
        return f"{act} -> {self.addressed_to or '-'} ({self.reason})"


# --------------------------------------------------------------------------- #
# 一轮完整交互
# --------------------------------------------------------------------------- #
@dataclass
class AgentTurn:
    tick: int
    actor_id: str
    thought: str = ""
    say: Optional[str] = None
    addressed_to: Optional[str] = None
    actions: list[ActionCall] = field(default_factory=list)
    results: list[ActionResult] = field(default_factory=list)
    plan: Optional[Plan] = None
    used_memories: list[str] = field(default_factory=list)
    decision_reason: str = ""
    persona_violations: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def acted(self) -> bool:
        return bool(self.say) or bool(self.actions)

    @property
    def tool_names(self) -> list[str]:
        return [a.tool for a in self.actions]

    def render(self) -> str:
        parts: list[str] = []
        if self.thought:
            parts.append(f"思考: {self.thought}")
        if self.say:
            to = f"→{self.addressed_to}" if self.addressed_to else ""
            parts.append(f"说{to}: {self.say}")
        for action in self.actions:
            parts.append(f"行动: {action.render()}")
        return " | ".join(parts) or "(无动作)"
