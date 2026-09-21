"""贯穿整条 Agent 流水线的核心数据类。

刻意保持"纯数据 + 少量渲染方法"，不依赖任何模块，避免循环导入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Role = Literal["npc", "player", "system"]
MemoryKind = Literal["episodic", "semantic", "reflection"]
# skipped 与 failed 必须分开：多 NPC 场景里"本轮话头给别人了"是一次**让位**，
# 不是失败。混成 failed 会触发无意义的重规划，还会让 Reflection 记下一条
# 根本不存在的教训。
StepStatus = Literal["pending", "running", "done", "failed", "skipped"]


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


#: 句末疑问语气助词。**刻意不含「吧」** ——
#: 「那给我来一杯吧」是**请求**不是提问，算成提问会让 NPC 去"回答"一句点单，
#: 而正确的反应是去做那杯咖啡（`planner.REQUEST_MARKERS` 管那条路）。
QUESTION_TAILS = ("吗", "呢", "么")

#: 疑问词。中文问句**经常不带问号**，只看「?」会漏掉一大半。
#:
#: 实测（控制台默认的 10 轮对话）：`你叫什么名字`、`这店开了多久了`
#: 都被判成陈述句，于是 NPC 只应一声「阿澈说的我记下了。」——
#: 玩家问了两遍都没得到回答。这是"对话推不动"最直接的成因，
#: 比复读更根本：复读是"说了等于没说"，这是"根本没在回答"。
QUESTION_WORDS = (
    "什么", "怎么", "为什么", "哪", "谁", "多少", "多久", "几时",
    "是否", "能不能", "可不可以", "有没有", "知不知道",
)


def looks_like_question(text: str) -> bool:
    """这句话是不是在提问。

    ⚠️ **这是个粗筛，有已知的误判**：「我什么都不知道」「没什么特别的」
    都会被判成提问（它们含疑问词但不疑问）。之所以接受：

    1. 误判的代价很低 —— NPC 会把一句陈述当成提问来回应，
       说出来的是「嗯——这个我还没想过，你怎么看？」，仍然是一句**贴现场**的话；
       而漏判的代价是"玩家问了问题，NPC 只回一句『我记下了』"，对话直接停住。
    2. 它**不参与任何评测断言** —— 只用来决定 NPC 该"回答"还是该"应一声"。
       所以它错判不会让某个分数变好看或变难看，只会让对话稍微不那么准。

    想把它做准，得上模型或分词器；那会让"离线可复现"这条底线多一个依赖，
    不划算。这里选择**把误差写在明面上**。
    """
    t = (text or "").strip()
    if not t:
        return False
    if t.endswith(("?", "？")):
        return True
    if t.rstrip("。！!~～… ").endswith(QUESTION_TAILS):
        return True
    return any(word in t for word in QUESTION_WORDS)


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


#: `ActionResult.outcome` 的合法取值。
#:
#: 为什么**不能**只看 `ok`：`ok=False` 混着两类完全不同的东西 ——
#:
#: * `"failed"`  —— 动作**真的**没做成（位置不对、材料不够、台词越界）。
#:                  这是该被反思、该写成教训的。
#: * `"declined"` —— 动作**按策略主动让出**（发言占比到顶让出话头、
#:                  这一轮已经有人开口了）。**这是正确行为**，
#:                  不是失败，不该反思、不该写进记忆。
#:
#: ## 这个字段是被量出来的，不是设计洁癖
#:
#: 实测（`scripts/probe_advice_coverage.py`，离线跑批 235 条）：
#: 反思一共往记忆里写了 **457** 条，其中 **139 条（30.4%）** 是
#: 「让出话头」被当成失败 —— 而整份语料里**最高频的**那条反思正是这个：
#:
#:     教训：speak 失败（发言占比 67% 已超上限，本轮主动让出话头）。
#:     我说话太多了，这一轮把机会留给玩家。        ← 出现 122 次
#:
#: 句子自己都写着「**主动**让出话头」，却顶着「教训：」和「失败」。
#: 而反思是**以 `importance=0.85` 进记忆、并被 `Plan` 的 prompt
#: 以【想起的事】取的** ⇒ 配了真实模型时，模型会读到这份假的自我评价：
#: 「我说话太多了」。**它被告知自己有个毛病，而那个"毛病"是它守规矩。**
#:
#: 判据不能用 `detail` 文本匹配 —— 那是本项目反复踩的坑（按一串特征
#: 串认类，上游一改文案就静默失效）。所以让**产出方**直接声明意图。
OUTCOME_FAILED = "failed"
OUTCOME_DECLINED = "declined"


@dataclass
class ActionResult:
    """工具执行结果。失败时 detail 会作为 Reflection 的输入信号。

    `outcome` 区分「真的失败了」和「按策略主动让出」——
    只有前者该进反思（见上面 `OUTCOME_*` 的注释）。
    """

    ok: bool
    tool: str
    detail: str = ""
    state_delta: dict[str, Any] = field(default_factory=dict)
    #: `ok=False` 时它才被读。默认 `"failed"` —— 也就是**默认最坏**：
    #: 忘了声明的调用点会被当成真失败（多想一次），而不是被当成
    #: "主动让出"（漏掉一次该学的教训）。两者的代价不对称。
    outcome: str = OUTCOME_FAILED

    @property
    def declined(self) -> bool:
        """按策略主动让出，不是失败。"""
        return (not self.ok) and self.outcome == OUTCOME_DECLINED

    def render(self) -> str:
        if self.ok:
            mark = "ok"
        elif self.declined:
            # ⚠️ 不印 `FAIL`。这一行会进 transcript，也是人读报告时的依据 ——
            # 把"主动让出"印成 FAIL，读报告的人会去查一个不存在的 bug。
            mark = "yield"
        else:
            mark = "FAIL"
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


#: `Plan.source` 的合法取值。
#:
#: * `model`            —— 由模型规划（`Planner.plan_with_llm` 成功返回）
#: * `heuristic`        —— 启发式规划器（场景目标模板）。**模型被调用过却没给出
#:                         可用计划时也会落到这里**，所以它同时是"回落"的信号
#: * `request_template` —— 玩家明确点单 → 意图模板（确定性最高，不是回落）
#: * `scenario_flow`    —— 玩家在问"怎么用" → 场景引导流程（不是回落）
#:
#: 判"有没有静默回落"要看：`use_llm_planner` 开着，而计划来源是 `heuristic`。
PLAN_SOURCES: tuple[str, ...] = ("model", "heuristic", "request_template", "scenario_flow")


@dataclass
class Plan:
    goal: str = ""
    rationale: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    objective_id: str = ""  # 关联的场景目标 id（用于"只尝试一次"的判定）
    #: 这个计划**是谁产出的**，取值见 `PLAN_SOURCES`。
    #:
    #: ⚠️ 为什么必须显式标注：规划调用失败（或模型返回了不可用的计划）时，
    #: 框架会**静默回落到启发式规划器**。不标注来源的话，
    #: "模型规划的"和"回落之后启发式规划的"在转写、控制台、评测数据里
    #: **长得一模一样** —— 于是"接上模型规划到底有没有用"这个问题
    #: 会在读者不知情的情况下变成自己跟自己比。
    #: 实测（2026-09-19，231 条跑批）：**48% 的用例至少回落过一次**。
    source: str = ""

    @property
    def next_step(self) -> Optional[PlanStep]:
        for step in self.steps:
            if step.status in ("pending", "running"):
                return step
        return None

    @property
    def done(self) -> bool:
        return all(s.status in ("done", "failed", "skipped") for s in self.steps)

    @property
    def failed_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status == "failed"]

    def render(self) -> str:
        lines = [f"目标: {self.goal}"]
        marks = {"pending": " ", "running": "~", "done": "x", "failed": "!", "skipped": "-"}
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
