"""评测层：把"NPC 表现好不好"变成可复现的数字。"""

from .harness import CaseResult, EvalHarness, EvalReport
from .metrics import CaseMetrics, Score

__all__ = ["EvalHarness", "EvalReport", "CaseResult", "CaseMetrics", "Score"]
