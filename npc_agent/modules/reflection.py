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
    """失败归因 → 写成教训 → 下次规划时带上。

    ## 不接受 `llm` 参数（原来收，现在不收）

    原来签名里有 `llm`，只被一个**没有任何调用点**的 `reflect_with_llm`
    读走（已删，理由见文件末尾的注释）。一个收了却不用的参数和
    "配置里写着一个不生效的开关"是同一种东西 —— **它让读代码的人以为
    反思这一路会用到模型**。所以连参数一起去掉，而不是留着不读。

    归因走 `_advice_for` 那张关键词表：实测在真实失败上覆盖率 **100%**
    （`scripts/probe_advice_coverage.py`），而且它是**纯离线**的 ——
    反思因此不消耗任何模型调用。
    """

    def __init__(
        self,
        persona: Persona,
        memory: MemoryManager,
        reflect_every: int = 6,
    ) -> None:
        self.persona = persona
        self.memory = memory
        self.reflect_every = max(1, reflect_every)
        self.lessons: list[Reflection] = []

    # ------------------------------------------------------------------ #
    def should_reflect(self, turn_index: int, results: list[ActionResult]) -> bool:
        # 与 `reflect()` 保持同一个判据：**「主动让出」不算需要反思的理由**
        # （见那里的长注释）。两处判据不一致的话，"该不该反思"和
        # "反思出了什么"会各说各话 —— 前者说要、后者返回 None。
        if any(not r.ok and not r.declined for r in results):
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
        # ⚠️ **「主动让出」不是失败，不许反思。**
        #
        # `ok=False` 混着两类东西（见 `types.OUTCOME_*`）：真的没做成，
        # 和按策略主动让出（发言占比到顶、这一轮已经有人开口了）。
        # 只有前者该进反思。
        #
        # 不分开的代价是量出来的（`scripts/probe_advice_coverage.py`）：
        # 离线跑批往记忆里写了 457 条反思，其中 **139 条（30.4%）** 是
        # 「让出话头」被当成失败，而全语料**最高频**的那条就是它：
        #
        #     教训：speak 失败（发言占比 67% 已超上限，本轮主动让出话头）。
        #     我说话太多了，这一轮把机会留给玩家。        ← 122 次
        #
        # 这句话会以 `importance=0.85` 进记忆，并被规划 prompt 以【想起的事】
        # 取走 ⇒ 配了真实模型时，模型会读到一句**假的自我评价**：
        # 它被告知自己话太多，而它其实只是守了规矩。
        failures = [r for r in results if not r.ok and not r.declined]
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


# --------------------------------------------------------------------------- #
# 关于「用模型做归因」—— 曾经有一个 `reflect_with_llm`，被删掉了
# --------------------------------------------------------------------------- #
# 它写着「有模型时用模型做归因，比关键词匹配更准」，但**全仓库没有任何地方
# 调它**（`grep -rn reflect_with_llm` 只有定义那一行）—— 一个不生效的开关。
#
# 删之前先量了"它到底能不能更准"（`scripts/probe_advice_coverage.py`，
# 离线跑批 235 条）。结论是**它不会更准**，所以删掉：
#
# 1. `_advice_for` 那张表在真实失败上**覆盖率 100%**：
#    一共只有 **5 种**失败 detail，全部是位置/工位类，
#    而表里两条位置类建议正好覆盖它们（全都命中具体档，通用档 0.0%）。
#    换句话说：**没有它可改进的样本。**
# 2. 它自己的预算是写死的 `max_tokens=120`，对推理模型太紧 ——
#    实测同一条 prompt 在 120 预算下仍有约 **10%** 的概率
#    因思维链吃光预算而返回空内容（`finish_reason=length`）。
#    也就是它比关键词表**更不可靠**，而不是更准。
# 3. 它是一条**永远不会被测到**的路径：没有调用点就没有测试，
#    没有测试就会腐烂 —— 上面那个 120 的预算就是已经腐烂的证据。
#
# 留着的唯一理由是"以后可能用得上"，但那种代码在本项目已经有个名字：
# **对使用者撒谎的开关**。要用模型归因，先拿出"关键词表认不出"的失败样本，
# 再按那个样本数量预算。
