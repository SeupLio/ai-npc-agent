"""守密人引擎（Keeper）—— 可验证信息流控制的 NPC 中间件。

一句话：米哈游方案回答"NPC 怎么像人"，守密人引擎回答
"NPC 怎么守住不该说的话"。前者是表现力，后者是可审计的
信息边界 —— 在剧本杀里是公平性，在企业助手是合规性。

入口速查：

    from npc_agent.keeper import Scenario, KeeperSession, run_batch

    scenario = Scenario.from_dict(mystery_scenario())
    result = KeeperSession(scenario, seed=7).run()
    result.solved            # 这局破了吗
    result.audit.ok          # 五条不变量全过吗
    result.audit.render_html()  # 信息流审计报告

层次：knowledge（事实/门/台账）→ strategy（立场/压力/谎言）→
session（对局循环）→ audit（独立重放复核）→ bench（批量指标）。
"""

from .audit import AuditReport, Invariant, audit_run
from .bench import run_batch
from .detective import DetectiveAsker, ScriptAsker
from .knowledge import Fact, KnowledgeBase
from .session import (
    Accuse,
    AskAction,
    Asker,
    AskerAction,
    EvidenceAct,
    EvidenceAction,
    Finish,
    KeeperResult,
    KeeperSession,
    Scenario,
    TranscriptLine,
)
from .strategy import AnswerDraft, KeeperNPC, KeeperPersona, Question

__all__ = [
    "Accuse",
    "AnswerDraft",
    "AskAction",
    "Asker",
    "AskerAction",
    "AuditReport",
    "DetectiveAsker",
    "EvidenceAct",
    "EvidenceAction",
    "Fact",
    "Finish",
    "Invariant",
    "KeeperNPC",
    "KeeperPersona",
    "KeeperResult",
    "KeeperSession",
    "KnowledgeBase",
    "Question",
    "Scenario",
    "ScriptAsker",
    "TranscriptLine",
    "audit_run",
    "run_batch",
]
