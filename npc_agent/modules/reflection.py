"""模块七：Reflection —— 从失败里学到东西。

没有 Reflection 的 Agent，同一个错误会犯一辈子。
这里的触发条件有三类：

    1. 工具调用失败   → 把失败原因提炼成"下次该怎么做"的教训
    2. 人设/边界越界  → 记下哪句话越界了，避免重复
    3. 周期性回顾     → 每 N 轮把近期对话压成一条结论

教训会以 reflection 类型的记忆写回 MemoryStore，检索时权重更高（×1.15），
所以 NPC 在相似情境下更容易"想起上次踩的坑"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..llm.base import LLM, LLMUnavailable
from ..types import ActionResult, AgentTurn
from .memory import MemoryManager
from .persona import Persona
from .state import StateTracker


@dataclass
class Reflection:
    trigger: str
    note: str
    applied: bool = False


class Reflector:
    def __init__(
        self,
        persona: Persona,
        memory: MemoryManager,
        llm: LLM,
        reflect_every: int = 6,
    ) -> None:
        self.persona = persona
        self.memory = memory
        self.llm = llm
        self.reflect_every = max(1, reflect_every)
        self.lessons: list[Reflection] = []

    # ------------------------------------------------------------------ #
    def should_reflect(self, turn_index: int, results: list[ActionResult]) -> bool:
        if any(not r.ok for r in results):
            return True
        return turn_index > 0 and turn_index % self.reflect_every == 0

    # ------------------------------------------------------------------ #
    def reflect(
        self,
        turn: AgentTurn,
        results: list[ActionResult],
        tracker: StateTracker,
        turn_index: int = 0,
    ) -> Optional[Reflection]:
        failures = [r for r in results if not r.ok]
        if failures:
            reflection = self._from_failure(failures, turn)
        elif turn.persona_violations:
            reflection = self._from_violation(turn)
        elif turn_index > 0 and turn_index % self.reflect_every == 0:
            reflection = self._periodic(tracker, turn)
        else:
            return None

        self.memory.record_reflection(reflection.note, turn.tick)
        self.lessons.append(reflection)
        return reflection

    # ------------------------------------------------------------------ #
    def _from_failure(self, failures: list[ActionResult], turn: AgentTurn) -> Reflection:
        primary = failures[0]
        advice = self._advice_for(primary)
        note = f"教训：{primary.tool} 失败（{primary.detail}）。{advice}"
        return Reflection(trigger="tool_failure", note=note, applied=bool(advice))

    def _advice_for(self, result: ActionResult) -> str:
        detail = result.detail
        if "需要先 move_to" in detail or "需要待在" in detail:
            return "下次动手前先确认自己是不是已经站在正确的工位上。"
        if "材料不够" in detail:
            return "做饮品前先把所有材料一次拿齐。"
        if "不在你身边" in detail:
            return "交付前先确认客人还在同一位置。"
        if "白名单" in detail or "越权" in detail:
            return "不要修改场景没有授权我改的标记。"
        if "剧透" in detail or "还不能讲" in detail:
            return "这个话题还没解锁，换个方向聊。"
        if "超上限" in detail or "让出话头" in detail:
            return "我说话太多了，这一轮把机会留给玩家。"
        if "越界" in detail:
            return "台词要先过人设和剧透检查。"
        return "换个方式再试一次。"

    def _from_violation(self, turn: AgentTurn) -> Reflection:
        note = f"教训：本轮台词触发 {'; '.join(turn.persona_violations)}，下次要守住人设边界。"
        return Reflection(trigger="persona_violation", note=note, applied=True)

    def _periodic(self, tracker: StateTracker, turn: AgentTurn) -> Reflection:
        players = "、".join(m.name for m in tracker.players.values()) or "没有客人"
        note = (
            f"阶段回顾：已经和第 {turn.tick} 轮了，"
            f"现场有{players}，未完成的目标是"
            f"{'、'.join(k for k, v in tracker.objectives.items() if v != 'done') or '全部完成'}。"
        )
        return Reflection(trigger="periodic", note=note)

    # ------------------------------------------------------------------ #
    def render_lessons(self, limit: int = 5) -> str:
        if not self.lessons:
            return "（还没有教训）"
        return "\n".join(f"  - {r.note}" for r in self.lessons[-limit:])

    def reflect_with_llm(self, turn: AgentTurn, results: list[ActionResult]) -> Optional[str]:
        """有模型时用模型做归因，比关键词匹配更准。"""
        if not self.llm.available:
            return None
        failure_text = "\n".join(r.render() for r in results if not r.ok)
        if not failure_text:
            return None
        prompt = (
            "以下是一次 NPC 行动失败记录。用一句话总结失败原因，"
            "再用一句话给出下次的改进动作。\n" + failure_text
        )
        try:
            return self.llm.complete(
                [{"role": "user", "content": prompt}], max_tokens=120
            ).strip()
        except LLMUnavailable:
            return None
