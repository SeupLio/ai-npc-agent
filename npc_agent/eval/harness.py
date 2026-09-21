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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..cast import build_cast
from ..config import RuntimeConfig, load_scenario
from ..llm import build_llm
from . import metrics as M

CASES_DIR = Path(__file__).resolve().parent / "cases"
CATEGORIES = ("task", "memory", "persona", "safety", "multi_npc", "minecraft")

#: 没配模型时写进报告的占位符。**报告要能一眼看出"这次没有模型"**，
#: 而不是留一个空字符串让读者自己猜。
OFFLINE_MODEL = "(offline)"


def eval_config(cfg: RuntimeConfig) -> dict[str, Any]:
    """跑批报告里的 `config` 块。**全仓库只有这一个产出点。**

    ⚠️ 这里从前有**两份**实现：`EvalHarness.run()` 写 `memory_top_k` 那几个，
    `cli.cmd_eval` 写 `use_llm_planner` / `use_llm_speech` 那两个。
    于是走 `run()` 出的报告里**没有 `use_llm_planner`**，
    而 `batch_report` 的副标题读的正是它 —— `config.get("use_llm_planner")`
    取不到就当成假，**副标题永远印「启发式规划」**，哪怕那次真的开了模型规划。

    **取不到 ≠ 没有：一个键的缺失被静默解释成了另一个值。**
    这类"两个真相"在本项目里出现过多次（`answer_hint` 的通配符、
    报告与测试各写一份语料），共同点都是**不报错、还更像对的**。
    """
    return {
        "provider": cfg.llm_provider,
        "model": cfg.model or OFFLINE_MODEL,
        "memory_strategy": cfg.memory_strategy,
        "memory_top_k": cfg.memory_top_k,
        "max_steps_per_turn": cfg.max_steps_per_turn,
        "reflect_every": cfg.reflect_every,
        "use_llm_planner": cfg.use_llm_planner,
        "use_llm_speech": cfg.use_llm_speech,
    }


# --------------------------------------------------------------------------- #
def evaluator_persona_violations(
    persona: Any, text: str, unlocked_topics: set[str] | None = None
) -> list[str]:
    """在**评测侧**独立算一遍人设违规 —— 不采信 agent 自报的 `turn.persona_violations`。

    ## 为什么不能直接读 `turn.persona_violations`

    因为那是**被测方自己算的**：`agent.py` 里

        turn.persona_violations = self.persona.check(turn.say, ...)

    评测如果直接拿来用，等于问被测方"你觉得自己违规了吗"。
    实测过这个后果（`eval/sensitivity.py` 的 `ooc_phrase_with_detector_disabled`）：
    **同一句出戏台词注入进转写**，

    - 检查器完好时 → `persona` 维度掉 **0.612**
    - 把检查器关掉之后 → `persona` 维度掉 **0.000**（满分）

    缺陷一模一样，分数从 0.388 变成 1.000。一个会被被测方一句话改写的分数，
    不是测量结果。

    ## 改法：判据可以复用，**结论必须评测侧自己下**

    这里只读人设的**数据表**（`forbidden_phrases` / `spoiler_terms` / `style`），
    自己算一遍，不调 `Persona.check`。判据和 `Persona.check` 是同一套，
    而且有测试钉住两者在语料上必须逐字一致
    （`test_evaluator_persona_audit_matches_persona_check`），所以不会悄悄漂移；
    但被测方关掉自己的检查器，再也影响不到评测。

    ⚠️ 注意 `unlocked_topics` 由**调用方**从环境里取（世界事实），
    不是从 agent 的镜像状态里取 —— 否则"剧透判定"又变成被测方说了算。
    """
    if not text:
        return []

    unlocked = unlocked_topics or set()
    violations: list[str] = []

    ooc = [p for p in persona.forbidden_phrases if p and p in text]
    if ooc:
        violations.append(f"出戏词: {', '.join(ooc)}")

    spoilers = [
        term
        for term in persona.spoiler_terms
        if term and term in text and not any(term in u for u in unlocked)
    ]
    if spoilers:
        violations.append(f"剧透: {', '.join(spoilers)}")

    if persona.too_long(text):
        violations.append("过长")

    max_sentences = int(persona.style.get("sentence_max") or 0)
    if max_sentences and persona.sentence_count(text) > max_sentences:
        violations.append("句数超限")

    return violations



# --------------------------------------------------------------------------- #
@dataclass
class CaseResult:
    case_id: str
    category: str
    description: str
    scenario: str
    metrics: M.CaseMetrics
    transcript: list[str] = field(default_factory=list)
    # 台词单独存一份，不要靠回头去 transcript 里按前缀捞。
    # 多 NPC 之后转写行变成了「阿柚: xxx」，靠 "NPC: " 前缀提取会一条都捞不到
    # —— 而且两个 NPC 的台词必须能分辨是谁说的。
    speeches: list[str] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: 这条用例里规划调用失败的次数与最后一次原因。
    #:
    #: 不记这个的话，`--no-planner` 和"planner 开着但一直在失败"会产出
    #: **完全一样的轨迹** —— 因为规划失败会静默回落到启发式规划。
    #: 于是"接上模型规划有没有用"这个对照实验，可能在读者不知情的情况下
    #: 变成自己跟自己比。这是"配置故障伪装成模型行为"的又一个入口。
    planner_failures: int = 0
    planner_last_error: str = ""
    #: 模型"调用成功但计划不可用"的次数。与 `planner_failures` 分开，
    #: 因为这两类的修法不同（一个修端点，一个修提示词/预算），
    #: 而它们**都会**静默回落到启发式规划。
    planner_empty_plans: int = 0
    #: 这条用例产出的计划**按来源**计数（见 `types.PLAN_SOURCES`）。
    #:
    #: `use_llm_planner` 开着而这里出现 `heuristic` ⇒ **静默回落**。
    #: 这是唯一能分辨"模型规划的"和"回落之后启发式规划的"的依据 ——
    #: 两者的轨迹在转写里完全一样。
    plans_by_source: dict[str, int] = field(default_factory=dict)

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
            "speeches": self.speeches,
            "speakers": self.speakers,
            "planner_failures": self.planner_failures,
            "planner_last_error": self.planner_last_error,
            "planner_empty_plans": self.planner_empty_plans,
            "plans_by_source": dict(self.plans_by_source),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaseResult":
        """从 `to_dict()` 的形状还原。用于跑批中断后从检查点恢复。"""
        return cls(
            case_id=str(data.get("case_id") or ""),
            category=str(data.get("category") or ""),
            description=str(data.get("description") or ""),
            scenario=str(data.get("scenario") or ""),
            metrics=M.CaseMetrics.from_dict(data),
            transcript=list(data.get("transcript") or []),
            speeches=list(data.get("speeches") or []),
            speakers=list(data.get("speakers") or []),
            notes=list(data.get("notes") or []),
            planner_failures=int(data.get("planner_failures") or 0),
            planner_last_error=str(data.get("planner_last_error") or ""),
            planner_empty_plans=int(data.get("planner_empty_plans") or 0),
            plans_by_source={
                str(k): int(v) for k, v in (data.get("plans_by_source") or {}).items()
            },
        )


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    #: 跑批的实测数字（墙钟、加速比、降级条数）。
    #: 刻意写进报告本体而不是只打在终端上：读 `eval.json` 的人（包括三个月后的
    #: 自己）必须能仅凭文件判断"这份分数可不可信"，而不是靠记得当时的终端输出。
    batch: dict[str, Any] = field(default_factory=dict)

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
        keys = ("task", "tools", "memory", "persona", "safety", "turn_taking")
        means = {}
        for key in keys:
            values = [r.metrics.as_dict()[key] for r in self.results]
            means[key] = round(sum(values) / len(values), 3) if values else 0.0
        return means

    def to_dict(self) -> dict[str, Any]:
        payload = {
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
        if self.batch:
            payload["batch"] = self.batch
        return payload

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
        llm_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.cases_dir = Path(cases_dir) if cases_dir else CASES_DIR
        # 允许注入 LLM 构造器。存在的理由只有一个：并行跑批需要给**每条用例**
        # 套一层计数器，用来发现"模型调用失败、框架静默回退到模板"这种
        # 在报告里和"模型答得不错"长得一模一样的失败。
        self._llm_factory = llm_factory

    # ------------------------------------------------------------------ #
    def load_cases(self, categories: Optional[list[str]] = None) -> list[dict[str, Any]]:
        """加载用例。`categories` 为空则全加载。

        **按用例自己的 `category` 字段过滤，而不是按文件名。**

        这条踩过坑：生成的用例全部落在 `generated.jsonl` 一个文件里，
        早期版本按文件 stem 过滤，于是 `--category safety` 只跑到了
        `safety.jsonl` 里那 3 条手写用例 —— 而安全类实际有 31 条。
        命令跑成功了、报告全绿、退出码 0，只是**测的东西比你以为的少 90%**。

        这类"静默少测"比报错危险：报错会有人去看，静默少测只会让人以为
        "安全维度没问题"。所以过滤条件同时接受 category 字段和文件名，
        两边任一命中即可（文件名命中是为了兼容把某类用例单独放一个文件的老写法）。
        """
        wanted = set(categories) if categories else None
        cases: list[dict[str, Any]] = []
        for path in sorted(self.cases_dir.glob("*.jsonl")):
            stem = path.stem
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                try:
                    case = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path.name}:{line_no} JSON 解析失败: {exc}") from exc
                case.setdefault("category", stem)
                if wanted and case["category"] not in wanted and stem not in wanted:
                    continue
                cases.append(case)
        return cases

    # ------------------------------------------------------------------ #
    def run_case(self, case: dict[str, Any]) -> CaseResult:
        scenario_id = case.get("scenario", "tutorial")
        scenario = load_scenario(scenario_id)
        llm = (
            self._llm_factory()
            if self._llm_factory is not None
            else build_llm(
                self.config.llm_provider,
                model=self.config.model,
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.llm_timeout,
                retries=self.config.llm_retries,
                parse_retries=self.config.llm_parse_retries,
            )
        )
        # 一律走 Cast，哪怕场上只有一个 NPC。
        # 单 NPC 只是"剧组只有一个人"的特例 —— 两条路径共用一套调度，
        # 多 NPC 的发言权逻辑每跑一次评测都在被使用，就不可能悄悄腐烂。
        cast = build_cast(scenario, llm, self.config)
        env = cast.env

        expect = case.get("expect") or {}
        transcript: list[str] = []
        speeches: list[str] = []
        speakers: list[str] = []
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

                # 一轮 = 一个 tick：剧组里所有 NPC 依次行动，最多一个人开口。
                for turn in cast.step(utterance):
                    speaker = cast.name_of(turn.actor_id)
                    for action, result in zip(turn.actions, turn.results):
                        called_tools.append(action.tool)
                        # ⚠️ 标记只从 `result.mark` 取，**这里不许自己再判一次 `ok`**。
                        # 原来自写一份 `"ok" if result.ok else "!!"` ⇒ 与
                        # `ActionResult.render()` 分家：render() 印 `[yield]`、
                        # 这里印 `[!!]` ⇒ **「主动让出话头」在裁判读的这段转写里
                        # 又长得像失败**（硬规矩 5「两个真相」）。
                        transcript.append(f"  [{result.mark}] {speaker} {action.render()}")
                    if turn.say:
                        speeches.append(turn.say)
                        speakers.append(turn.actor_id)
                        # 评测侧自己算，**不读 `turn.persona_violations`** ——
                        # 那是被测方自己算的（见 evaluator_persona_violations）。
                        # world_flags 也从**环境**取，不从 agent 的镜像状态取。
                        violations.append(
                            evaluator_persona_violations(
                                cast.agents[turn.actor_id].persona,
                                turn.say,
                                set(env.snapshot().get("world_flags", [])),
                            )
                        )
                        speeches_by_actor[turn.actor_id] = (
                            speeches_by_actor.get(turn.actor_id, 0) + 1
                        )
                        transcript.append(f"{speaker}: {turn.say}")

        for utterance in env.utterances:
            if utterance.role == "player":
                speeches_by_actor[utterance.speaker_id] = (
                    speeches_by_actor.get(utterance.speaker_id, 0) + 1
                )

        snapshot = env.snapshot()
        # 记忆是每个 NPC 私有的，所以分开取。"谁记住了"本身就是多 Agent 的评测点。
        memories = cast.memory_contents()
        memory_score = M.memory_recall(
            expect, speeches, [c for contents in memories.values() for c in contents]
        )
        ownership = M.memory_ownership(expect, memories)
        if ownership.value < memory_score.value:
            memory_score = ownership

        scores = M.CaseMetrics(
            task=M.task_completion(expect, snapshot, set(snapshot.get("world_flags", []))),
            tools=M.tool_scores(expect, called_tools),
            memory=memory_score,
            persona=M.persona_consistency(violations, len(speeches)),
            safety=M.safety(expect, speeches, set(snapshot.get("world_flags", []))),
            turn_taking=M.turn_taking(
                env.speakers_by_tick(),
                env.npc_ids,
                require_all_spoke=bool(expect.get("all_npcs_spoke")),
            ),
        )
        if expect.get("check_stage_share"):
            # **不能写成 `scores.safety = M.stage_share(...)`。**
            # 那样会把这条用例真正的安全断言（出戏 / 泄露 / 越权）整个丢掉：
            # style_bounds_hosting 那 4 条用例同时写了 check_stage_share 和
            # speech_never_contains，覆盖式赋值让 speech_never_contains 变成
            # 死断言 —— expect 里写了"要检查"，但没有任何代码检查它。
            # 两个都是越界性质的检查，取较差的那个，说明两个都留着。
            scores.safety = M.combine_boundaries(
                scores.safety,
                M.stage_share(speeches_by_actor, env.npc_ids),
            )

        notes: list[str] = []
        if not scores.passed:
            for key, detail in scores.details().items():
                value = scores.as_dict()[key]
                if value < 0.99:
                    notes.append(f"{key}: {detail}")

        # 计划来源按 NPC 汇总。多 NPC 场景里两个 NPC 的回落情况可能完全不同，
        # 分开看才能知道是谁在回落 —— 合起来只有一个数会把这个信息抹掉。
        plans_by_source: dict[str, int] = {}
        for agent in cast.agents.values():
            for source, count in agent.plan_sources().items():
                plans_by_source[source] = plans_by_source.get(source, 0) + count

        return CaseResult(
            case_id=case.get("id", "unnamed"),
            category=case.get("category", "task"),
            description=case.get("description", ""),
            scenario=scenario_id,
            metrics=scores,
            transcript=transcript,
            speeches=speeches,
            speakers=speakers,
            notes=notes,
            planner_failures=sum(a.planner_failures for a in cast.agents.values()),
            planner_last_error=next(
                (
                    a.planner_last_error
                    for a in cast.agents.values()
                    if a.planner_failures
                ),
                "",
            ),
            planner_empty_plans=sum(
                a.planner_empty_plans for a in cast.agents.values()
            ),
            plans_by_source=plans_by_source,
        )

    # ------------------------------------------------------------------ #
    def run(self, categories: Optional[list[str]] = None) -> EvalReport:
        report = EvalReport(config=eval_config(self.config))
        for case in self.load_cases(categories):
            report.results.append(self.run_case(case))
        return report
