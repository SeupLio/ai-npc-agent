"""评测层：把"NPC 表现好不好"变成可复现的数字。"""

from .compare import (
    Comparison,
    RunOutcome,
    RunSpec,
    free_speech_rate,
    is_scripted,
    scripted_patterns,
)
from .harness import CaseResult, EvalHarness, EvalReport
from .metrics import CaseMetrics, Score

__all__ = [
    "EvalHarness",
    "EvalReport",
    "CaseResult",
    "CaseMetrics",
    "Score",
    "Comparison",
    "RunSpec",
    "RunOutcome",
    "scripted_patterns",
    "free_speech_rate",
    "is_scripted",
]
