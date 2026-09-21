"""多配置对照跑批 —— 把"这次改动真的更好"变成一张能看的表。

两种典型用法：

1. **离线启发式 vs 真实模型**（roadmap 第 2 项）
   证明接上模型之后，台词不再是从模板里抠出来的，而人设护栏在真实模型下依然守得住。

2. **记忆策略消融**（roadmap 第 4 项）
   ``hybrid`` / ``recency`` / ``lexical`` / ``importance`` / ``none``，
   其中 ``none`` 是"裸模型"基线，用来量化记忆到底贡献了多少。

设计上刻意只换一个变量：同一套用例、同一套指标、同一份场景配置。
否则对比出来的差值说不清是谁带来的。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from ..config import CONFIG_DIR, RuntimeConfig, load_yaml
from .harness import EvalHarness, EvalReport

# --------------------------------------------------------------------------- #
# 一、跑批规格


@dataclass
class RunSpec:
    """一次跑批的全部自变量。"""

    label: str
    provider: str = "null"
    model: str = ""
    memory_strategy: str = "hybrid"
    temperature: float = 0.7
    use_llm_planner: bool = True
    use_llm_speech: bool = True

    def apply(self, base: RuntimeConfig) -> RuntimeConfig:
        """在基准配置上只覆盖本规格关心的字段，其余保持不变。"""
        cfg = RuntimeConfig(**{**base.__dict__})
        cfg.llm_provider = self.provider
        cfg.model = self.model
        cfg.memory_strategy = self.memory_strategy
        cfg.temperature = self.temperature
        cfg.use_llm_planner = self.use_llm_planner
        cfg.use_llm_speech = self.use_llm_speech
        return cfg

    @property
    def is_offline(self) -> bool:
        return self.provider == "null" or not self.model


@dataclass
class RunOutcome:
    """一次跑批的结果 + 只有对照才关心的派生指标。"""

    spec: RunSpec
    report: EvalReport
    duration: float = 0.0
    free_speech_rate: float = 0.0
    avg_speech_chars: float = 0.0
    speech_count: int = 0
    speeches: list[str] = field(default_factory=list)

    @property
    def means(self) -> dict[str, float]:
        return self.report.metric_means()

    @property
    def pass_rate(self) -> float:
        return round(self.report.passed / self.report.total, 3) if self.report.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.spec.label,
            "spec": asdict(self.spec),
            "duration_sec": round(self.duration, 1),
            "pass_rate": self.pass_rate,
            "passed": self.report.passed,
            "total": self.report.total,
            "metric_means": self.means,
            "free_speech_rate": round(self.free_speech_rate, 3),
            "avg_speech_chars": round(self.avg_speech_chars, 1),
            "speech_count": self.speech_count,
            "speeches": self.speeches,
            "by_category": self.report.by_category(),
            "failures": [
                {"case_id": r.case_id, "notes": r.notes}
                for r in self.report.results
                if not r.passed
            ],
        }


# --------------------------------------------------------------------------- #
# 二、模板台词识别 —— 用来衡量"模型到底有没有真的在说话"

_PUNCT = "。！？，、,.!? "


def _norm(text: str) -> str:
    """归一化：去空白、去首尾标点，避免因为空格/句号差异漏判。"""
    return re.sub(r"\s+", "", text or "").strip(_PUNCT)


def _template_regex(raw: str) -> Optional[re.Pattern[str]]:
    """把一条模板编译成正则，槽位（``{item_name}``）变成 ``.+``。

    为什么不能直接做字符串相等：模板里带槽位，渲染后文本就变了。
    例如 ``deliver_order: "{item_name}好了，趁热。"`` 渲染成 ``拿铁好了，趁热。``，
    字面比对会把它误判成"模型自由发挥"，于是离线模式的"非模板率"会虚高到 78%。
    把槽位通配掉之后，判定才反映真实情况。

    注意：去标点只能作用在**原始字面块**上，不能对拼好的正则整串 strip ——
    ``.`` 和 ``?`` 既是标点又是正则元字符，整串 strip 会把开头的 ``.+`` 削成 ``+``，
    直接抛 "nothing to repeat"。

    ⚠️ **整条模板全是槽位时返回 ``None``（不参与判定），而不是 ``^.+$``。**
    这类模板对"这句话长什么样"**零信息量**：``answer_question: "{answer_hint}"``
    的字面量是空的，说什么完全由运行时填进去的东西决定。
    以前它会编译成 ``^.+$`` —— 一个**匹配任何非空台词的通配符**，
    于是"自由台词率"恒为 0：报告会说每句台词都是脚本台词。
    这比漏判更坏，因为它把指标变成了一个常数（有测试钉住这一点）。
    """
    parts = [p for p in re.split(r"(\{[^}]*\})", raw or "") if p]
    chunks: list[str] = []
    literal_chars = 0
    for index, part in enumerate(parts):
        if part.startswith("{") and part.endswith("}"):
            chunks.append(".+")
            continue
        literal = re.sub(r"\s+", "", part)
        if index == 0:
            literal = literal.lstrip(_PUNCT)
        if index == len(parts) - 1:
            literal = literal.rstrip(_PUNCT)
        literal_chars += len(literal)
        chunks.append(re.escape(literal))
    if literal_chars == 0:
        return None
    return re.compile("^" + "".join(chunks) + "$")


def _raw_templates() -> list[str]:
    """收集所有 persona 的模板台词原文（含槽位）。

    ⚠️ 模板的值可以是 **str 或 list[str]**（多变体，见 `Persona.template`）。
    早期这里只收 `isinstance(value, str)`，加了变体之后**整个列表会被静默跳过** ——
    于是这些台词不再被算作"脚本台词"，"自由台词率"会凭空变高：
    明明是模板生成的一句话，报告会说它是模型生成的。
    静默跳过比报错更危险，所以这里显式展开。

    一并收 `self_facts[].reply`：那也是**写在人设配置里的固定句子**
    （玩家问"你叫什么名字"时 NPC 照念），和模板一样是脚本台词。
    不收的话离线模式的自由台词率会被高估 —— 而它恰好是"离线 vs 模型"
    这张对照表里最容易被误读的一个数。
    """
    lines: list[str] = []
    personas_dir = CONFIG_DIR / "personas"
    if not personas_dir.exists():
        return lines
    for path in personas_dir.glob("*.yaml"):
        data = load_yaml(path) or {}
        templates = (data.get("style") or {}).get("utterance_templates") or {}
        if not templates:
            templates = data.get("utterance_templates") or {}
        for value in templates.values():
            if isinstance(value, str):
                lines.append(value)
            elif isinstance(value, (list, tuple)):
                lines.extend(str(item) for item in value if isinstance(item, str))
        for entry in data.get("self_facts") or []:
            reply = (entry or {}).get("reply") if isinstance(entry, dict) else None
            if isinstance(reply, str) and reply.strip():
                lines.append(reply)
    return lines


def _raw_knowledge() -> list[str]:
    """世界知识库里的固定文本。

    这些也是写死的字符串（NPC 的 ``tell_fact`` 会把它们原样念出来），
    所以必须一起算作"脚本台词"，否则离线模式的自由台词率会被高估。
    """
    from ..env.star_isle import KNOWLEDGE

    return [
        entry.get("text", "")
        for entry in KNOWLEDGE.values()
        if isinstance(entry, dict) and entry.get("text")
    ]


def scripted_patterns() -> list[re.Pattern[str]]:
    """全部"脚本台词"的正则形式：人设模板 + 世界知识库。

    这是判断"这句话是人写的还是模型写的"的参照系。

    全槽位的模板会被 `_template_regex` 判成 `None`（零信息量），在这里丢掉 ——
    见那边的注释：留成 `^.+$` 会让自由台词率恒为 0。
    """
    patterns: list[re.Pattern[str]] = []
    for raw in _raw_templates() + _raw_knowledge():
        pattern = _template_regex(raw)
        if pattern is not None:
            patterns.append(pattern)
    return patterns


def is_scripted(speech: str, patterns: list[re.Pattern[str]]) -> bool:
    norm = _norm(speech)
    return any(p.match(norm) for p in patterns)


def free_speech_rate(speeches: list[str], patterns: list[re.Pattern[str]]) -> float:
    """自由台词率 = 既不是人设模板、也不是世界知识库原文的台词占比。

    离线启发式模式只从固定字符串里取词，这个值应当接近 0；
    真实模型模式下台词由模型现场组织，这个值应当显著更高 ——
    这正是"接上模型到底值不值"最直观的证据。
    """
    if not speeches:
        return 0.0
    free = sum(1 for s in speeches if not is_scripted(s, patterns))
    return free / len(speeches)


# --------------------------------------------------------------------------- #
# 三、对照跑批

#: 两次落盘之间至少间隔多少秒。
#:
#: 检查点是为了**长跑批崩了不白跑**（真实模型一跑半小时）。但"每条用例都落一次"
#: 会让代价变成 O(N²)：`_refresh()` 要遍历**迄今全部**结果、`save()` 要序列化
#: **整份增长中的报告**，而这两件事逐条做、做 N 次。
#:
#: 实测（236 条 × 5 档的离线消融，各测两遍）：逐条落盘 **1180 次**、耗时 **~98–158s**；
#: 按时间节流后 **10 次**、**~5–9s**。落盘次数是**确定性**的，耗时随机器负载漂 ——
#: 复现时以次数为准。
#: 换来的只是"崩溃时少丢最后 5 秒" —— 对几秒就跑完的离线跑批毫无意义，
#: 对半小时的模型跑批也无关痛痒。**别为了一个只在长跑批里有用的保护，
#: 让短跑批付几十倍的代价。**
#:
#: ⚠️ **别顺手把 `runner.py` / `judge.py` 的逐条落盘也改成节流。** 那两处每条用例要
#: 几秒到十几分钟（真实模型调用），落盘那几十毫秒是噪声；而且长跑批**真的需要**
#: 逐条保护（崩了最多丢一条）。判据是**比值**（落盘一次的代价 / 单条用例的代价），
#: 不是"形状一样就一起改"。
CHECKPOINT_MIN_INTERVAL_SEC = 5.0


@dataclass
class Comparison:
    base_config: RuntimeConfig
    cases_dir: Optional[Path] = None
    outcomes: list[RunOutcome] = field(default_factory=list)
    categories: Optional[list[str]] = None
    limit: int = 0

    # ------------------------------------------------------------------ #
    def run(
        self,
        specs: list[RunSpec],
        progress=None,
        checkpoint: str | Path | None = None,
        on_case=None,
    ) -> "Comparison":
        """逐个规格跑批。逐个而不是并行，是因为真实模型端点通常有并发限制。

        ``checkpoint`` 指定一个路径，跑批过程中**按时间节流**落盘
        （见 `CHECKPOINT_MIN_INTERVAL_SEC`：最多每 5 秒一次，跑完再补一次）。
        这不是过度设计：真实模型的跑批动辄半小时（单次推理 ~9s × 几十轮），
        如果只在最后写一次文件，中途任何一次崩溃都会让几十分钟白跑。
        但也**不能每条用例都落** —— 那是 O(N²)，短跑批会白付 8 倍代价。

        ``on_case`` 是每条用例结束后的回调，用来打进度 ——
        跑批过程中最怕的就是"看起来卡住了"，其实只是在慢慢跑。
        """
        patterns = scripted_patterns()
        for index, spec in enumerate(specs, 1):
            if progress:
                progress(f"[{index}/{len(specs)}] {spec.label} …")

            cfg = spec.apply(self.base_config)
            harness = EvalHarness(cfg, self.cases_dir)

            cases = harness.load_cases(self.categories)
            if self.limit:
                cases = cases[: self.limit]

            started = time.time()
            report = EvalReport(
                config={
                    "provider": cfg.llm_provider,
                    "model": cfg.model or "(offline)",
                    "memory_strategy": cfg.memory_strategy,
                    "use_llm_planner": cfg.use_llm_planner,
                    "use_llm_speech": cfg.use_llm_speech,
                }
            )
            # 先把 outcome 挂进列表；之后按 `CHECKPOINT_MIN_INTERVAL_SEC` 的节奏
            # 刷新它，这样 checkpoint 里始终是一份"已完成部分"的完整报告 ——
            # 但**不是**每条用例都刷新（那是 O(N²)，见那个常量的注释）。
            outcome = RunOutcome(spec=spec, report=report)
            self.outcomes.append(outcome)

            last_save = 0.0
            for case_index, case in enumerate(cases, 1):
                report.results.append(harness.run_case(case))
                # ⚠️ 刷新 + 落盘**不要每条用例都做** —— 逐条做就是 O(N²)。
                # 详见 `CHECKPOINT_MIN_INTERVAL_SEC`：实测这一步占掉全程 90% 以上的时间。
                if checkpoint and time.time() - last_save >= CHECKPOINT_MIN_INTERVAL_SEC:
                    self._refresh(outcome, patterns, started)
                    self.save(checkpoint)
                    last_save = time.time()
                # `on_case` 只用来打进度（它只读 `report.passed/total`），
                # 所以**不**为它刷新派生指标 —— 那会把 O(N²) 加回来。
                if on_case:
                    on_case(spec, case, case_index, len(cases), outcome)

            # 这一档跑完：刷新一次保证派生指标是完整的，再落一次盘保证检查点最新。
            self._refresh(outcome, patterns, started)
            if checkpoint:
                self.save(checkpoint)

            if progress:
                progress(
                    f"    完成：{report.passed}/{report.total} 通过，"
                    f"耗时 {outcome.duration:.1f}s，自由台词 {outcome.free_speech_rate:.0%}"
                )
        return self

    # ------------------------------------------------------------------ #
    @staticmethod
    def _refresh(outcome: RunOutcome, patterns: list, started: float) -> None:
        """按当前已完成的用例重算派生指标。"""
        speeches = _collect_speeches(outcome.report)
        outcome.duration = time.time() - started
        outcome.speeches = speeches
        outcome.speech_count = len(speeches)
        outcome.free_speech_rate = free_speech_rate(speeches, patterns)
        outcome.avg_speech_chars = (
            sum(len(s) for s in speeches) / len(speeches) if speeches else 0.0
        )

    # ------------------------------------------------------------------ #
    # 配对：真实模型的跑批可能跑很久，中途被打断是常态。
    # 如果各行的用例覆盖数不一致，直接比均值就是拿苹果比橘子 ——
    # 所以一律截到"所有行都跑完的那一段"再算指标。
    def paired_count(self) -> int:
        if not self.outcomes:
            return 0
        return min(len(o.report.results) for o in self.outcomes)

    def is_paired(self) -> bool:
        return len({len(o.report.results) for o in self.outcomes}) <= 1

    def _metrics_for(self, outcome: RunOutcome) -> tuple[int, int, dict[str, float]]:
        """取该行在"共同覆盖区间"内的通过数与各维均值。"""
        if self.is_paired():
            results = outcome.report.results
        else:
            results = outcome.report.results[: self.paired_count()]
        keys = ("task", "tools", "memory", "persona", "safety", "turn_taking")
        if not results:
            return 0, 0, {k: 0.0 for k in keys}
        means = {
            k: round(sum(r.metrics.as_dict()[k] for r in results) / len(results), 3)
            for k in keys
        }
        return sum(1 for r in results if r.passed), len(results), means

    # ------------------------------------------------------------------ #
    def rows(self) -> list[dict[str, Any]]:
        """给渲染层用的行数据。"""
        rows = []
        for outcome in self.outcomes:
            passed, total, means = self._metrics_for(outcome)
            rows.append(
                {
                    "label": outcome.spec.label,
                    "model": outcome.spec.model or "—",
                    "strategy": outcome.spec.memory_strategy,
                    "pass": f"{passed}/{total}",
                    "pass_rate": round(passed / total, 3) if total else 0.0,
                    "task": means.get("task", 0.0),
                    "tools": means.get("tools", 0.0),
                    "memory": means.get("memory", 0.0),
                    "persona": means.get("persona", 0.0),
                    "safety": means.get("safety", 0.0),
                    "turn_taking": means.get("turn_taking", 0.0),
                    "free": outcome.free_speech_rate,
                    "chars": outcome.avg_speech_chars,
                    "sec": outcome.duration,
                }
            )
        return rows

    def deltas(self) -> list[dict[str, Any]]:
        """以第一行为基线，算出每一行的差值 —— 这才是"对照"的重点。"""
        if len(self.outcomes) < 2:
            return []
        base = self.outcomes[0]
        _, _, base_means = self._metrics_for(base)
        base_rate = self.rows()[0]["pass_rate"]
        out = []
        for index, outcome in enumerate(self.outcomes[1:], 1):
            _, _, means = self._metrics_for(outcome)
            rate = self.rows()[index]["pass_rate"]
            out.append(
                {
                    "label": outcome.spec.label,
                    "vs": base.spec.label,
                    "task": round(means.get("task", 0) - base_means.get("task", 0), 3),
                    "tools": round(means.get("tools", 0) - base_means.get("tools", 0), 3),
                    "memory": round(means.get("memory", 0) - base_means.get("memory", 0), 3),
                    "persona": round(means.get("persona", 0) - base_means.get("persona", 0), 3),
                    "safety": round(means.get("safety", 0) - base_means.get("safety", 0), 3),
                    "turn_taking": round(
                        means.get("turn_taking", 0) - base_means.get("turn_taking", 0), 3
                    ),
                    "pass_rate": round(rate - base_rate, 3),
                    "free": round(outcome.free_speech_rate - base.free_speech_rate, 3),
                }
            )
        return out

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        paired = self.is_paired()
        runs: list[dict[str, Any]] = []
        for outcome in self.outcomes:
            data = outcome.to_dict()
            if not paired:
                passed, total, means = self._metrics_for(outcome)
                data["passed"] = passed
                data["total"] = total
                data["pass_rate"] = round(passed / total, 3) if total else 0.0
                data["metric_means"] = means
            runs.append(data)
        return {
            "categories": self.categories or "all",
            "limit": self.limit,
            "paired": paired,
            "paired_cases": self.paired_count(),
            "runs": runs,
            "deltas": self.deltas(),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target


# --------------------------------------------------------------------------- #
def _collect_speeches(report: EvalReport) -> list[str]:
    """把每个用例的 NPC 台词汇总起来。

    直接读 ``CaseResult.speeches``，**不要**回头去 transcript 里按 "NPC: "
    前缀捞 —— 多 NPC 之后转写行是「阿柚: xxx」，按老前缀提取会一条都捞不到，
    而且捞回来的也分不清是谁说的（自由台词率是按角色算的）。
    """
    return [line for result in report.results for line in result.speeches]
