"""守密人引擎的对局循环 —— 一个循环跑两种世界。

## 统一的循环

不管外面是剧本杀还是企业助手，里面的循环是同一个：

    提问方 action ──► NPC 立场提案 ──► 门（逐条过）──► 台词渲染 ──► 事件
         ▲                                                                  │
         └────────────────────── 观察（只看该看的）◄──────────────────────────┘

提问方（``Asker``）是个协议：它只通过 ``plan`` 给出下一步动作、
通过 ``observe`` 接收**它有权看到**的事件。侦探听不到 NPC 之间的
私语（现实里也听不到），审计听得到 —— 两份视角是不同的，这个不同
本身就是信息边界在引擎里的又一次体现。

## 台词是门剪过的

NPC 的回答模板在渲染时**只拼装过门成功的事实**。被拦的事实不进
台词，取而代之的是得体拒答/岔话模板。也就是说：门不只记台账，
它直接决定说出口的话长什么样 —— "说出的话先被门剪过"在这里是
字面成立的，不是比喻。

## 为什么 audit 在 session 里调

对局结束时的审计是对**这场对局的台账**做的独立复核（重放门判定），
它不属于 KnowledgeBase（自己不能审自己），也不属于提问方（它立场
不中立）。session 是台账的属主，所以由它发起 —— 但审计逻辑本体在
audit.py，可单独测试。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol

from ..env.conditions import ConditionContext
from .audit import AuditReport, audit_run
from .knowledge import (
    CHANNEL_ANSWER,
    CHANNEL_EVIDENCE,
    CHANNEL_VOLUNTEER,
    CHANNEL_WHISPER,
    Fact,
    KnowledgeBase,
)
from .strategy import (
    STANCE_CRACK,
    STANCE_PARTIAL,
    STANCE_WITHHOLD,
    AnswerDraft,
    KeeperNPC,
    KeeperPersona,
    Question,
)


# --------------------------------------------------------------------------- #
# 场景与动作
# --------------------------------------------------------------------------- #
@dataclass
class EvidenceAction:
    """世界侧的取证动作：设标记，解锁条件门后的事实。

    ``yield_chance`` 是"搜了但没搜到"的概率表达（催款信塞在
    行李夹层、靴印被雨水糊掉——搜查本来就不是稳赚的）。失手时
    标记不设、事实不解锁：世界状态与"没搜到"自洽，门保持关着。
    """

    action_id: str
    label: str
    sets_flag: str
    #: 完成后能拿到的事实（进台账的 evidence 通道 —— 物证不经过嘴）。
    yields: list[str] = field(default_factory=list)
    #: 命中率 0-1。默认 1.0（稳拿）。
    yield_chance: float = 1.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceAction":
        return cls(
            action_id=data["action_id"],
            label=data["label"],
            sets_flag=data["sets_flag"],
            yields=list(data.get("yields") or []),
            yield_chance=float(data.get("yield_chance", 1.0)),
        )


@dataclass
class Scenario:
    """一个守密人场景的全部静态设定。"""

    scenario_id: str
    title: str
    intro: str
    npcs: list[KeeperPersona]
    facts: list[Fact]
    asker_id: str
    asker_name: str
    rounds: int = 8
    questions_per_round: int = 3
    #: 每轮 NPC 之间的私语交换次数（0 = 关闭传闻通道）。
    whisper_per_round: int = 1
    evidence_actions: list[EvidenceAction] = field(default_factory=list)
    #: 侦探可以钓着问的话题：fact_id -> 问句文本。
    topics: dict[str, str] = field(default_factory=dict)
    #: 企业场景的问询脚本：[{npc, topic, text}, ...]。
    script: list[dict[str, str]] = field(default_factory=list)
    initial_flags: set[str] = field(default_factory=set)
    #: 提问方的初始知识（案发这类"人人都知道"的事实）。
    asker_knows: list[str] = field(default_factory=list)
    #: 剧本杀类场景的答案（企业场景为 None：无"破案"，只有合规结论）。
    solution: Optional[Mapping[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Scenario":
        solution = data.get("solution")
        return cls(
            scenario_id=data["scenario_id"],
            title=data["title"],
            intro=data.get("intro", ""),
            npcs=[KeeperPersona.from_dict(n) for n in data["npcs"]],
            facts=[Fact.from_dict(f) for f in data["facts"]],
            asker_id=data["asker_id"],
            asker_name=data["asker_name"],
            rounds=int(data.get("rounds", 8)),
            questions_per_round=int(data.get("questions_per_round", 3)),
            whisper_per_round=int(data.get("whisper_per_round", 1)),
            evidence_actions=[
                EvidenceAction.from_dict(e) for e in data.get("evidence_actions") or []
            ],
            topics=dict(data.get("topics") or {}),
            script=[dict(r) for r in data.get("script") or []],
            initial_flags=set(data.get("initial_flags") or []),
            asker_knows=list(data.get("asker_knows") or []),
            solution=dict(solution) if solution is not None else None,
        )

    def npc_names(self) -> dict[str, str]:
        return {n.npc_id: n.name for n in self.npcs}

    @property
    def fact_by_id(self) -> dict[str, Fact]:
        return {f.fact_id: f for f in self.facts}

    def fact_contents(self) -> dict[str, str]:
        return {f.fact_id: f.content for f in self.facts}


# --------------------------------------------------------------------------- #
# 提问方协议
# --------------------------------------------------------------------------- #
@dataclass
class AskAction:
    """一次普通提问。text 走情绪判读（问法凶不凶影响压力）。

    ``listener`` 是这条信息真正的接收人（企业场景里一次提问的
    是小夏还是闻姐，门按这个人判）。None = 场景默认提问方。
    """

    target_npc: str
    topic_fact_id: str
    text: str
    listener: Optional[str] = None


@dataclass
class EvidenceAct:
    action_id: str


@dataclass
class Accuse:
    target_npc: str
    reason: str = ""


@dataclass
class Finish:
    summary: str = ""


AskerAction = AskAction | EvidenceAct | Accuse | Finish


class SessionView(Protocol):
    """提问方能看到的世界（刻意不给压力、不给私语 —— 那是隐藏状态）。"""

    def known_facts(self) -> set[str]: ...

    def round_no(self) -> int: ...

    def asked_topics(self) -> set[tuple[str, str]]: ...


class Asker(Protocol):
    """提问方：侦探或企业里的问询脚本。"""

    def plan(self, view: SessionView) -> Optional[AskerAction]: ...

    def observe_answer(
        self,
        target: str,
        topic: str,
        text: str,
        delivered: list[str],
        nervous: bool = False,
        stonewall: bool = False,
    ) -> None: ...

    def observe_evidence(self, action_id: str, found: list[str]) -> None: ...

    def observe_crack(self, npc_id: str, spilled: list[str]) -> None: ...


# --------------------------------------------------------------------------- #
# 对局记录
# --------------------------------------------------------------------------- #
@dataclass
class TranscriptLine:
    round_no: int
    speaker: str
    text: str
    kind: str = "talk"  # talk / evidence / whisper / accuse / result


@dataclass
class KeeperResult:
    scenario_id: str
    seed: int
    solved: Optional[bool]
    accused: Optional[str]
    rounds_used: int
    transcript: list[TranscriptLine]
    audit: AuditReport
    #: 侦探（或问询方）最终知识态。
    asker_known: set[str]
    #: 关键计数。
    counters: dict[str, int] = field(default_factory=dict)

    def metrics(self) -> dict[str, Any]:
        base = {
            "scenario": self.scenario_id,
            "seed": self.seed,
            "rounds_used": self.rounds_used,
            "asker_facts": len(self.asker_known),
            "zero_leak": self.audit.zero_leak,
            **self.counters,
        }
        if self.solved is not None:
            base["solved"] = self.solved
        return base


# --------------------------------------------------------------------------- #
# 会话
# --------------------------------------------------------------------------- #
class KeeperSession:
    """一场守密人对局。确定性：同 scenario + 同 seed => 同一切。"""

    def __init__(self, scenario: Scenario, seed: int = 0) -> None:
        self.scenario = scenario
        self.seed = seed
        self.rng = random.Random(seed)
        self.kb = KnowledgeBase(scenario.facts)
        self.npcs: dict[str, KeeperNPC] = {}
        for spec in scenario.npcs:
            npc = KeeperNPC(spec, self.kb)
            npc.register_names(scenario.npc_names())
            self.npcs[spec.npc_id] = npc
        # 提问方的初始知识（案发事实这类"人人皆知"）也走 seed ——
        # 它是设定不是转移，不进台账；审计会把它当重建起点。
        self.kb.seed_knowledge(scenario.asker_id, scenario.asker_knows)
        self.flags: set[str] = set(scenario.initial_flags)
        self.asker: Asker = self._build_asker()
        self.round = 0
        self.transcript: list[TranscriptLine] = []
        #: 每问的压力问数（侦探视角不可见）。
        self._asked: set[tuple[str, str]] = set()
        self._actions_done: set[str] = set()
        self._crack_events: list[tuple[int, str, list[str]]] = []
        self.accused: Optional[str] = None
        self.solved: Optional[bool] = None
        self._finished: bool = False
        self._counters: dict[str, int] = {
            "questions": 0,
            "evidence_actions": 0,
            "whispers": 0,
            "cracks": 0,
            "blocked_attempts": 0,
        }

    # ------------------------------------------------------------------ #
    def _build_asker(self) -> Asker:
        """按场景选择提问方实现。延迟导入避免循环依赖。"""
        from .detective import DetectiveAsker, ScriptAsker

        if self.scenario.solution is not None:
            # 共享 self.rng：plan/NPC/搜查失手共用一个随机流，
            # 调用顺序确定 ⇒ 整场对局对 seed 完全可复现。
            return DetectiveAsker(self.scenario, rng=self.rng)
        return ScriptAsker(self.scenario)

    def ctx(self) -> ConditionContext:
        return ConditionContext(flags=frozenset(self.flags))

    # ------------------------------------------------------------------ #
    def run(self) -> KeeperResult:
        for round_no in range(1, self.scenario.rounds + 1):
            self.round = round_no
            self._run_round(round_no)
            if self._finished or self.solved is not None or self.accused is not None:
                break
        if self.scenario.solution is not None and self.solved is None:
            # 没敢定罪（提前放弃或轮次耗尽）也是"未破"：
            # 与"指控错误"在 solved=False 上合并，但在转录里可区分。
            self.solved = False
            if not self._finished:
                self.transcript.append(
                    TranscriptLine(self.round, "结果", "证据不足，未能定罪。", "result")
                )
        audit = audit_run(self.kb, self)
        return KeeperResult(
            scenario_id=self.scenario.scenario_id,
            seed=self.seed,
            solved=self.solved,
            accused=self.accused,
            rounds_used=self.round,
            transcript=self.transcript,
            audit=audit,
            asker_known=self.kb.known_by(self.scenario.asker_id),
            counters=dict(self._counters),
        )

    # ------------------------------------------------------------------ #
    def _run_round(self, round_no: int) -> None:
        for _ in range(self.scenario.questions_per_round):
            action = self.asker.plan(self._view())
            if action is None or isinstance(action, Finish):
                if isinstance(action, Finish):
                    self.transcript.append(
                        TranscriptLine(round_no, "提问方", action.summary, "result")
                    )
                    self._finished = True
                break
            if isinstance(action, Accuse):
                self._accuse(action, round_no)
                return
            if isinstance(action, EvidenceAct):
                self._do_evidence(action, round_no)
                continue
            if isinstance(action, AskAction):
                self._ask(action, round_no)
        self._whisper_phase(round_no)

    # ------------------------------------------------------------------ #
    def _ask(self, action: AskAction, round_no: int) -> None:
        npc = self.npcs[action.target_npc]
        # 门按"这一问真正的接收人"判：企业场景里小夏问薪酬和闻姐问
        # 薪酬，走的是同一道门、同一套判定，结果不同只因为人不同。
        listener = action.listener or self.scenario.asker_id
        question = Question(
            asker_id=listener,
            target_id=action.target_npc,
            topic_fact_id=action.topic_fact_id,
            text=action.text,
            round_no=round_no,
        )
        self._asked.add((action.target_npc, action.topic_fact_id))
        self._counters["questions"] += 1
        draft = npc.answer(question, self.ctx(), self.rng)
        delivered, text = self._execute_draft(
            npc, draft, round_no, listener, question
        )
        if draft.stance == STANCE_CRACK and delivered:
            self._counters["cracks"] += 1
            self._crack_events.append((round_no, npc.p.npc_id, list(delivered)))
            self.asker.observe_crack(npc.p.npc_id, list(delivered))
        asker_label = self.scenario.npc_names().get(
            listener, self.scenario.asker_name
        )
        self.transcript.append(
            TranscriptLine(
                round_no,
                f"{asker_label}→{npc.p.name}",
                action.text,
                "talk",
            )
        )
        self.transcript.append(
            TranscriptLine(round_no, npc.p.name, text or "（沉默）", "talk")
        )
        self.asker.observe_answer(
            npc.p.npc_id,
            action.topic_fact_id,
            text,
            list(delivered),
            nervous=draft.stance in (STANCE_PARTIAL, STANCE_WITHHOLD),
            stonewall=draft.note == "闭嘴期拒答",
        )

    def _execute_draft(
        self,
        npc: KeeperNPC,
        draft: AnswerDraft,
        round_no: int,
        listener: Optional[str] = None,
        question: Optional[Question] = None,
    ) -> tuple[list[str], str]:
        """把立场提案过门、渲染台词。返回（交付成功的 fact ids, 台词）。"""
        target = listener or self.scenario.asker_id
        delivered: list[str] = []
        # 谎版优先渲染：谎言是普通事实（非门控），但要"像一句人话"。
        bluff_delivered: list[str] = []
        for bluff_id in draft.bluff_fact_ids:
            ok = self.kb.deliver(
                bluff_id,
                npc.p.npc_id,
                target,
                CHANNEL_ANSWER,
                round_no,
                self.ctx(),
                via="谎版声明",
            )
            if ok:
                bluff_delivered.append(bluff_id)
        channel = (
            CHANNEL_VOLUNTEER if draft.stance == STANCE_CRACK else CHANNEL_ANSWER
        )
        for fact_id in draft.deliver_ids:
            ok = self.kb.deliver(
                fact_id,
                npc.p.npc_id,
                target,
                channel,
                round_no,
                self.ctx(),
                via=draft.note,
            )
            if ok:
                delivered.append(fact_id)
            else:
                self._counters["blocked_attempts"] += 1
        text = npc.render_answer(
            draft, question, delivered + bluff_delivered, self.rng
        )
        return delivered, text

    # ------------------------------------------------------------------ #
    def _do_evidence(self, action: EvidenceAct, round_no: int) -> None:
        spec = next(
            (
                e
                for e in self.scenario.evidence_actions
                if e.action_id == action.action_id
            ),
            None,
        )
        if spec is None or spec.action_id in self._actions_done:
            return
        self._actions_done.add(spec.action_id)
        self._counters["evidence_actions"] += 1
        # 搜查可能失手（yield_chance）。失手时标记不设、事实不解锁，
        # 转录里如实记一笔"没有发现"—— 侦探的世界和玩家一样，
        # 不确定是常态。
        if spec.yield_chance < 1.0 and self.rng.random() > spec.yield_chance:
            self.transcript.append(
                TranscriptLine(
                    round_no, "取证", f"{spec.label}——翻遍了，没有发现。", "evidence"
                )
            )
            self.asker.observe_evidence(spec.action_id, [])
            return
        self.flags.add(spec.sets_flag)
        found: list[str] = []
        for fact_id in spec.yields:
            ok = self.kb.deliver(
                fact_id,
                "world",
                self.scenario.asker_id,
                CHANNEL_EVIDENCE,
                round_no,
                self.ctx(),
                via=spec.label,
            )
            if ok:
                found.append(fact_id)
        contents = self.scenario.fact_contents()
        for fact_id in found:
            self.transcript.append(
                TranscriptLine(round_no, "取证", contents.get(fact_id, fact_id), "evidence")
            )
        self.asker.observe_evidence(spec.action_id, found)

    # ------------------------------------------------------------------ #
    def _whisper_phase(self, round_no: int) -> None:
        """NPC 之间交换传闻：能传的传（过门），传不了的记一次拦截。"""
        if self.scenario.whisper_per_round <= 0:
            return
        ids = [n.p.npc_id for n in self.npcs.values()]
        contents = self.scenario.fact_contents()
        for _ in range(self.scenario.whisper_per_round):
            if len(ids) < 2:
                return
            speaker = self.rng.choice(ids)
            listener = self.rng.choice([i for i in ids if i != speaker])
            candidates = [
                f
                for f in sorted(self.kb.known_by(speaker))
                if not self.kb.facts[f].held_secret
                and not self.kb.facts[f].voluntary_only
                and not self.kb.knows(listener, f)
                and not self._is_bluff_fact_of(speaker, f)
            ]
            if not candidates:
                continue
            fact_id = self.rng.choice(candidates)
            ok = self.kb.deliver(
                fact_id,
                speaker,
                listener,
                CHANNEL_WHISPER,
                round_no,
                self.ctx(),
                via="传闻",
            )
            if ok:
                self._counters["whispers"] += 1
                self.transcript.append(
                    TranscriptLine(
                        round_no,
                        f"{self.npcs[speaker].p.name}→{self.npcs[listener].p.name}",
                        f"（私下说）{contents.get(fact_id, fact_id)}",
                        "whisper",
                    )
                )
            else:
                self._counters["blocked_attempts"] += 1

    def _is_bluff_fact_of(self, npc_id: str, fact_id: str) -> bool:
        return fact_id in self.npcs[npc_id].p.bluffs.values()

    # ------------------------------------------------------------------ #
    def _accuse(self, action: Accuse, round_no: int) -> None:
        self.accused = action.target_npc
        if self.scenario.solution is not None:
            self.solved = action.target_npc == self.scenario.solution.get("murderer")
        names = self.scenario.npc_names()
        verdict = "指控成立。" if self.solved else "指控不成立。"
        self.transcript.append(
            TranscriptLine(
                round_no,
                self.scenario.asker_name,
                f"我指控 {names.get(action.target_npc, action.target_npc)}。{action.reason}",
                "accuse",
            )
        )
        self.transcript.append(
            TranscriptLine(round_no, "结果", verdict, "result")
        )

    # ------------------------------------------------------------------ #
    def _view(self) -> "SessionView":
        session = self

        class _View:
            def known_facts(self) -> set[str]:
                return session.kb.known_by(session.scenario.asker_id)

            def round_no(self) -> int:
                return session.round

            def asked_topics(self) -> set[tuple[str, str]]:
                return set(session._asked)

        return _View()


__all__ = [
    "Accuse",
    "AskAction",
    "Asker",
    "AskerAction",
    "EvidenceAct",
    "EvidenceAction",
    "Finish",
    "KeeperResult",
    "KeeperSession",
    "Scenario",
    "SessionView",
    "TranscriptLine",
]
