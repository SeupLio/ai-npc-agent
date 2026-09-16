"""评测 harness —— 把一次完整跑批变成可复现的数字。

设计原则：
1. **确定性**：每条用例都 reset 环境，同一份配置跑两次结果必须一致
2. **可归因**：用例失败时给出具体原因，而不只是一个红叉
3. **可对比**：支持 --compare 拿两份报告做差，用来证明"这次改动真的更好了"

用法：
    python -m npc_agent.cli eval                     # 跑全部用例
    python -m npc_agent.cli eval --category safety   # 只跑安全用例
    python -m npc_agent.cli eval --json out.json     # 导出机器可读报告
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from ..agent import NPCAgent
from ..config import RuntimeConfig, load_persona, load_scenario
from ..env.star_isle import StarIsleEnv
from ..llm import build_llm
from ..modules.persona import Persona
from . import metrics as M

CASES_DIR = Path(__file__).resolve().parent / "cases"
CATEGORIES = ("task", "memory", "persona", "safety")


# --------------------------------------------------------------------------- #
@dataclass
class CaseResult:
    case_id: str
    category: str
    description: str
    scenario: str
    metrics: M.CaseMetrics
    transcript: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.metrics.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "scenario": self.scenario,
            "description": self.description,
            "passed": self.passed,
            "scores": self.metrics.as_dict(),
            "details": self.metrics.details(),
            "notes": self.notes,
            "transcript": self.transcript,
        }


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    def by_category(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for result in self.results:
            bucket = out.setdefault(
                result.category, {"total": 0, "passed": 0, "scores": {}}
            )
            bucket["total"] += 1
            bucket["passed"] += 1 if result.passed else 0
            for key, value in result.metrics.as_dict().items():
                bucket["scores"].setdefault(key, []).append(value)
        for bucket in out.values():
            bucket["scores"] = {
                k: round(sum(v) / len(v), 3) for k, v in bucket["scores"].items() if v
            }
            bucket["pass_rate"] = round(bucket["passed"] / bucket["total"], 3)
        return out

    def metric_means(self) -> dict[str, float]:
        keys = ("task", "tools", "memory", "persona", "safety")
        means = {}
        for key in keys:
            values = [r.metrics.as_dict()[key] for r in self.results]
            means[key] = round(sum(values) / len(values), 3) if values else 0.0
        return means

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "pass_rate": round(self.passed / self.total, 3) if self.total else 0.0,
                "metric_means": self.metric_means(),
                "by_category": self.by_category(),
            },
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target


# --------------------------------------------------------------------------- #
class EvalHarness:
    def __init__(
        self,
        config: RuntimeConfig | None = None,
        cases_dir: str | Path | None = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.cases_dir = Path(cases_dir) if cases_dir else CASES_DIR

    # ------------------------------------------------------------------ #
    def load_cases(self, categories: Optional[list[str]] = None) -> list[dict[str, Any]]:
        wanted = set(categories) if categories else None
        cases: list[dict[str, Any]] = []
        for path in sorted(self.cases_dir.glob("*.jsonl")):
            if wanted and path.stem not in wanted:
                continue
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                try:
                    case = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path.name}:{line_no} JSON 解析失败: {exc}") from exc
                case.setdefault("category", path.stem)
                cases.append(case)
        return cases

    # ------------------------------------------------------------------ #
    def run_case(self, case: dict[str, Any]) -> CaseResult:
        scenario_id = case.get("scenario", "tutorial")
        scenario = load_scenario(scenario_id)
        persona = Persona.from_dict(load_persona(scenario.get("npc", "ayou")))

        env = StarIsleEnv(scenario, persona.id, persona.name)
        llm = build_llm(
            self.config.llm_provider,
            model=self.config.model,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )
        agent = NPCAgent(persona, env, scenario, llm, self.config)

        expect = case.get("expect") or {}
        transcript: list[str] = []
        speeches: list[str] = []
        violations: list[list[str]] = []
        called_tools: list[str] = []
        speeches_by_actor: dict[str, int] = {}

        for turn_spec in case.get("turns", []):
            repeat = 1
            if isinstance(turn_spec, dict):
                repeat = int(turn_spec.get("repeat", 1))
            for _ in range(repeat):
                utterance = None
                if isinstance(turn_spec, dict) and turn_spec.get("text"):
                    utterance = env.record_player_utterance(
                        turn_spec.get("player", "player_a"), turn_spec["text"]
                    )
                    transcript.append(f"玩家[{utterance.speaker_name}] {utterance.text}")
                turn = agent.step(utterance)

                for action, result in zip(turn.actions, turn.results):
                    called_tools.append(action.tool)
                    mark = "ok" if result.ok else "!!"
                    transcript.append(f"  [{mark}] {action.render()}")
                if turn.say:
                    speeches.append(turn.say)
                    speeches_by_actor["npc"] = speeches_by_actor.get("npc", 0) + 1
                    transcript.append(f"NPC: {turn.say}")
                if turn.persona_violations:
                    violations.append(turn.persona_violations)
                else:
                    violations.append([])
                env.advance_tick()

        for utterance in env.utterances:
            if utterance.role == "player":
                speeches_by_actor[utterance.speaker_id] = (
                    speeches_by_actor.get(utterance.speaker_id, 0) + 1
                )

        snapshot = env.snapshot()
        memory_contents = [r.content for r in agent.memory.store.records]

        scores = M.CaseMetrics(
            task=M.task_completion(expect, snapshot, set(snapshot.get("world_flags", []))),
            tools=M.tool_scores(expect, called_tools),
            memory=M.memory_recall(expect, speeches, memory_contents),
            persona=M.persona_consistency(violations, len(speeches)),
            safety=M.safety(expect, speeches, set(snapshot.get("world_flags", []))),
        )
        if expect.get("check_stage_share"):
            scores.safety = M.stage_share(speeches_by_actor)

        notes: list[str] = []
        if not scores.passed:
            for key, detail in scores.details().items():
                value = scores.as_dict()[key]
                if value < 0.99:
                    notes.append(f"{key}: {detail}")

        return CaseResult(
            case_id=case.get("id", "unnamed"),
            category=case.get("category", "task"),
            description=case.get("description", ""),
            scenario=scenario_id,
            metrics=scores,
            transcript=transcript,
            notes=notes,
        )

    # ------------------------------------------------------------------ #
    def run(self, categories: Optional[list[str]] = None) -> EvalReport:
        report = EvalReport(
            config={
                "provider": self.config.llm_provider,
                "model": self.config.model or "(offline)",
                "memory_top_k": self.config.memory_top_k,
                "max_steps_per_turn": self.config.max_steps_per_turn,
                "reflect_every": self.config.reflect_every,
            }
        )
        for case in self.load_cases(categories):
            report.results.append(self.run_case(case))
        return report
