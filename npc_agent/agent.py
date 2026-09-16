"""NPC 智能体主循环 —— 把七大模块串成一条决策链。

一轮的完整链路（对应公开方案里那条"玩家说话 → ... → 记录结果"）：

    observe          从环境取观测
    state.update     刷新现场状态（谁在场、谁刚说话、冷场多久）
    dialogue.decide  该不该我说话？对谁说？
    memory.retrieve  想起相关的事
    planner.plan     定计划（或继续未完成的计划）
    tools.execute    执行（世界动作 + 语言动作，统一路径）
    reflection       失败就归因，写回记忆
"""

from __future__ import annotations

import time
from typing import Any, Optional

from .config import RuntimeConfig
from .env.base import Environment
from .llm.base import LLM, LLMUnavailable
from .modules.dialogue import AddresseeSelector, DialogueConfig, TurnManager
from .modules.memory import MemoryManager, MemoryStore
from .modules.persona import Persona
from .modules.planner import Planner
from .modules.reflection import Reflector
from .modules.state import StateTracker
from .modules.tools import ToolContext, ToolRegistry
from .types import ActionCall, ActionResult, AgentTurn, Plan, PlanStep, Utterance

# 记忆里出现这些词，说明是值得当面回引的偏好类信息
RECALL_HINTS = ("喜欢", "讨厌", "习惯", "常来", "第一次", "答应", "约定")

# 记忆内容的句子边界。必须包含「；」——
# 巩固后的摘要用「；」把多条 episodic 拼在一起，
# 不在这里切断就会拼出「…越酸越好。；小，是这个没错吧？」
_CLAUSE_BREAK = "。！？；"
# 巩固产出的 semantic 记录带的前缀，回引时要剥掉
_SUMMARY_PREFIX = "（早前对话摘要）"
_EDGE_PUNCT = "。！？，、；:： "


def extract_memory_hint(content: str, limit: int = 18) -> str:
    """从一条记忆里抽出可以直接说出口的短句。

    三步：剥掉摘要前缀 → 剥掉「某某说：」前缀 → 在第一个句子边界处切断。

    早期实现是 ``content.split("：", 1)[-1][:18].rstrip(...)``。
    按字数硬截有两个问题：一是会截到半个词，二是对巩固后的摘要无效 ——
    ``阿澈说：我特别喜欢偏酸的咖啡，越酸越好。；小满说：…`` 截出来是
    ``我特别喜欢偏酸的咖啡，越酸越好。；小``，于是 NPC 说出
    「…越好。；小，是这个没错吧？」。**按语义边界切，不按字数切。**
    """
    text = (content or "").strip()
    if text.startswith(_SUMMARY_PREFIX):
        text = text[len(_SUMMARY_PREFIX) :]
    if "：" in text:
        text = text.split("：", 1)[-1]

    cut = len(text)
    for mark in _CLAUSE_BREAK:
        pos = text.find(mark)
        if pos != -1:
            cut = min(cut, pos)
    text = text[:cut].strip(_EDGE_PUNCT)

    if len(text) > limit:
        head = text[:limit]
        # 优先在逗号/顿号处收尾，避免留下半个词
        for sep in ("，", "、"):
            pos = head.rfind(sep)
            if pos >= limit // 2:
                head = head[:pos]
                break
        text = head

    return text.strip(_EDGE_PUNCT)


class NPCAgent:
    """一个可配置、可控、可评测的游戏 NPC。"""

    def __init__(
        self,
        persona: Persona,
        env: Environment,
        scenario: dict[str, Any],
        llm: LLM,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.persona = persona
        self.env = env
        self.scenario = scenario
        self.llm = llm
        self.config = config or RuntimeConfig()
        self.id = persona.id
        self.objectives: list[dict[str, Any]] = list(scenario.get("objectives") or [])
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.env.reset()
        self.state = StateTracker(self.id, self.persona.name)
        self.memory = MemoryManager(
            MemoryStore(
                half_life=self.config.memory_half_life,
                consolidate_at=self.config.memory_consolidate_at,
                strategy=self.config.memory_strategy,
            ),
            top_k=self.config.memory_top_k,
            strategy=self.config.memory_strategy,
        )
        self.planner = Planner(
            self.persona,
            self.llm,
            world_facts=self.env.world_facts(),
            max_retries=self.config.max_plan_retries,
            max_tokens=self.config.max_tokens,
        )
        self.registry = ToolRegistry(self.env, self.config)
        self.turn_manager = TurnManager(
            AddresseeSelector(
                self.persona,
                DialogueConfig(
                    idle_ticks_before_proactive=self.config.idle_ticks_before_proactive
                ),
            )
        )
        self.reflector = Reflector(
            self.persona, self.memory, self.llm, reflect_every=self.config.reflect_every
        )
        self.active_plan: Optional[Plan] = None
        self.turn_index = 0
        self.turns: list[AgentTurn] = []
        self._retry_counts: dict[tuple, int] = {}
        self._shared_topics: set[str] = set()
        self._last_share_tick = -99
        self._attempted: set[str] = set()
        self.state.update_from_env(self.env.observe(self.id))

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #
    def step(self, utterance: Optional[Utterance] = None) -> AgentTurn:
        started = time.perf_counter()
        self.turn_index += 1

        observation = self.env.observe(self.id)
        self.state.update_from_env(observation)
        # 世界规则会变（物品被拿走、位置改变），规划器每轮刷新一次事实表
        self.planner.facts = self.env.world_facts()

        if utterance is not None:
            self.state.note_utterance(utterance)
            self.memory.observe(utterance, self.id)
        else:
            self.state.tick_silence()

        plan_pending = bool(self.active_plan and not self.active_plan.done)
        decision = self.turn_manager.next_decision(
            self.state, utterance, has_pending_plan=plan_pending
        )

        memories = self.memory.retrieve(
            self._memory_query(utterance, decision), now=self.state.tick
        )

        turn = AgentTurn(
            tick=self.state.tick,
            actor_id=self.id,
            addressed_to=decision.addressed_to,
            decision_reason=decision.reason,
            used_memories=[m.id for m in memories],
        )

        # 只在本轮之前没有计划时才新建计划。
        # 本轮刚建的计划已经考虑了玩家这句话，不该再额外插一句回应。
        started_this_turn = False
        if not plan_pending and decision.should_speak:
            self.active_plan = self._make_plan(utterance, decision, memories)
            started_this_turn = self.active_plan is not None

        ctx = ToolContext(
            actor_id=self.id,
            tick=self.state.tick,
            env=self.env,
            memory=self.memory,
            persona=self.persona,
            tracker=self.state,
        )

        if self.active_plan and not self.active_plan.done:
            next_step = self.active_plan.next_step
            if (
                not started_this_turn
                and decision.urgency >= 0.9
                and utterance is not None
            ):
                if utterance.is_question:
                    # 计划是之前就在跑的，这时被点名提问 → **回答优先于计划**。
                    # 否则就会出现"玩家在问问题，NPC 却在背教程"的经典翻车。
                    self._respond(turn, ctx, memories, utterance, decision)
                elif next_step is None or next_step.tool != "speak":
                    # 被点名但只是打招呼 —— 先应一声，再继续干活
                    self._quick_acknowledge(turn, utterance, ctx, memories)
            self._run_plan(turn, ctx, memories, utterance)
            turn.plan = self.active_plan
        elif decision.should_speak:
            self._respond(turn, ctx, memories, utterance, decision)

        if turn.say:
            self.state.npc_utterances += 1

        # --- Reflection ---
        if self.reflector.should_reflect(self.turn_index, turn.results):
            self.reflector.reflect(turn, turn.results, self.state, self.turn_index)

        # --- 人设检查 ---
        if turn.say:
            turn.persona_violations = self.persona.check(turn.say, self.state.world_flags)

        if self.active_plan and self.active_plan.done:
            finished = self.active_plan
            # 计划全部走通 → 这个目标只做一次，不再重复开场
            if finished.objective_id and not finished.failed_steps:
                self._attempted.add(finished.objective_id)
            self.active_plan = None

        turn.latency_ms = (time.perf_counter() - started) * 1000
        self.turns.append(turn)
        return turn

    # ------------------------------------------------------------------ #
    # 规划
    # ------------------------------------------------------------------ #
    def _make_plan(
        self,
        utterance: Optional[Utterance],
        decision: Any,
        memories: list[Any],
    ) -> Optional[Plan]:
        # 1) 玩家明确点单 → 意图模板优先（确定性最高，也最贴合玩家意图）
        if utterance is not None:
            request_plan = self.planner.plan_for_utterance(utterance, self.state)
            if request_plan and request_plan.steps:
                return request_plan

        # 2) 玩家在问"我该怎么用 / 教教我" → 直接走场景引导流程。
        #    新客问"这里怎么点单"，正确的回答就是那套引导动作，而不是一句客套话。
        if utterance is not None and self.planner.wants_scenario_flow(utterance.text):
            guide_plan = self.planner.plan_next_objective(
                self.objectives, self.state, self._attempted
            )
            if guide_plan and guide_plan.steps:
                return guide_plan

        # 3) 被直接点名（提问或搭话）→ 不启动场景目标，交给 _respond 去回答。
        #    这一条是"不背固定教程"的关键闸门。
        if decision.urgency >= 0.9:
            return None

        # 4) 有模型 → 让模型规划
        if self.config.use_llm_planner:
            llm_plan = self.planner.plan_with_llm(
                self.state,
                utterance,
                memories,
                self.registry.catalog(self.id),
                goal_hint=self._pending_goal_hint(),
            )
            if llm_plan and llm_plan.steps:
                return llm_plan

        # 5) 兜底：推进场景里还没完成的目标
        return self.planner.plan_next_objective(self.objectives, self.state, self._attempted)

    def _pending_goal_hint(self) -> str:
        pending = [
            f"- {o.get('goal')}" for o in self.objectives
            if self.state.objectives.get(o.get("id"), "pending") != "done"
        ]
        return "\n".join(pending)

    # ------------------------------------------------------------------ #
    # 执行计划
    # ------------------------------------------------------------------ #
    def _run_plan(
        self,
        turn: AgentTurn,
        ctx: ToolContext,
        memories: list[Any],
        utterance: Optional[Utterance],
    ) -> None:
        executed = 0
        guard = 0
        while (
            self.active_plan is not None
            and not self.active_plan.done
            and executed < self.config.max_steps_per_turn
            and guard < 12
        ):
            guard += 1
            step = self.active_plan.next_step
            if step is None:
                break

            call = self._materialize_step(step, ctx, memories, utterance)
            if call is None:
                step.status = "failed"
                step.note = "无法把这一步翻译成具体动作"
                executed += 1
                continue

            result = self.registry.execute(call, ctx)
            turn.actions.append(call)
            turn.results.append(result)

            if result.ok:
                step.status = "done"
                if call.tool == "speak":
                    turn.say = result.detail
                executed += 1
                continue

            # --- 失败：先尝试重规划 ---
            key = self._step_key(step)
            retries = self._retry_counts.get(key, 0)
            if retries < self.config.max_plan_retries:
                patched = self.planner.replan(self.active_plan, step, result.detail, self.state)
                if patched is not None:
                    self._retry_counts[key] = retries + 1
                    self.active_plan = patched
                    continue
            step.status = "failed"
            step.note = result.detail
            executed += 1

    def _materialize_step(
        self,
        step: PlanStep,
        ctx: ToolContext,
        memories: list[Any],
        utterance: Optional[Utterance],
    ) -> Optional[ActionCall]:
        """把计划步骤翻译成具体的工具调用。

        speak 步骤允许写 intent 而不是写死台词 —— 这样同一份场景配置
        可以在不同上下文里说出贴合现场的话，而不是背课文。
        """
        args = dict(step.args)
        if step.tool == "speak":
            intent = args.pop("intent", None)
            # 计划步骤可以顺带传模板参数（例如 item_name），用来生成更具体的台词
            extra = {k: args.pop(k) for k in list(args) if k not in ("text", "to")}
            if "text" not in args:
                if intent:
                    args["text"] = self._render_intent(intent, memories, utterance, extra)
                else:
                    args["text"] = self.persona.render_template("fallback")
        if step.tool == "remember" and "content" not in args:
            args["content"] = self._render_intent("acknowledge", memories, utterance)
        return ActionCall(tool=step.tool, args=args, reason=step.goal)

    @staticmethod
    def _step_key(step: PlanStep) -> tuple:
        return (step.tool, tuple(sorted((str(k), str(v)) for k, v in step.args.items())))

    # ------------------------------------------------------------------ #
    # 即时回应（没有计划时）
    # ------------------------------------------------------------------ #
    def _respond(
        self,
        turn: AgentTurn,
        ctx: ToolContext,
        memories: list[Any],
        utterance: Optional[Utterance],
        decision: Any,
    ) -> None:
        if decision.proactive:
            # 还有没做完的目标 → 推进它
            pending = [k for k, v in self.state.objectives.items() if v != "done"]
            if pending:
                intent = self.turn_manager.selector.pick_proactive_intent(self.state)
            elif self._proactive_share(turn, ctx):
                # 目标都做完了 → 主动分享店里的事（主动使用工具，而不是干聊）
                return
            else:
                # 没什么可说的就保持安静，比复读一句"嗯"更像人
                return
        elif utterance is not None and utterance.is_question:
            intent = "recall" if self._has_recallable(memories, self.state.tick) else "answer_question"
        elif self._has_recallable(memories, self.state.tick):
            intent = "recall"
        else:
            intent = "acknowledge" if utterance is not None else "fallback"

        text = self._generate_speech(intent, memories, utterance, decision)
        call = ActionCall(
            tool="speak",
            args={"text": text, "to": decision.addressed_to},
            reason=decision.reason,
        )
        result = self.registry.execute(call, ctx)
        turn.actions.append(call)
        turn.results.append(result)
        if result.ok:
            turn.say = result.detail

    def _proactive_share(self, turn: AgentTurn, ctx: ToolContext) -> bool:
        """主动透露一个还没聊过的话题。展示"主动使用工具"的能力。

        话题要先过两道筛子：人设允许聊（can_discuss）+ 世界已解锁（available_topics）。
        少任何一道，NPC 就会反复尝试说一件它其实不该说的事。
        """
        allowed = set(self.env.available_topics(self.id))
        topics = [
            t
            for t in self.persona.can_discuss
            if t in allowed and t not in self._shared_topics
        ]
        # 主动分享要有间隔，否则会变成"每冷场一次就丢一个知识点"的复读机
        if not topics or self.state.tick - self._last_share_tick < 2:
            return False
        topic = topics[0]
        self._shared_topics.add(topic)  # 试过就不再试，避免死循环
        self._last_share_tick = self.state.tick
        call = ActionCall("tell_fact", {"topic": topic}, reason="冷场，主动起个话头")
        result = self.registry.execute(call, ctx)
        turn.actions.append(call)
        turn.results.append(result)
        if not result.ok:
            return False
        text = result.state_delta.get("text") or result.detail
        speak = ActionCall("speak", {"text": text, "to": turn.addressed_to}, reason="主动分享")
        spoken = self.registry.execute(speak, ctx)
        turn.actions.append(speak)
        turn.results.append(spoken)
        if spoken.ok:
            turn.say = spoken.detail
        return True

    def _quick_acknowledge(
        self,
        turn: AgentTurn,
        utterance: Utterance,
        ctx: ToolContext,
        memories: list[Any],
    ) -> None:
        text = self.persona.render_template("acknowledge", target=utterance.speaker_name)
        call = ActionCall("speak", {"text": text, "to": utterance.speaker_id}, reason="被点名先应一声")
        result = self.registry.execute(call, ctx)
        turn.actions.append(call)
        turn.results.append(result)

    # ------------------------------------------------------------------ #
    # 台词生成
    # ------------------------------------------------------------------ #
    INTENT_HINTS = {
        "opening": "开场，招呼大家并带出今天的氛围。",
        "invite_intro": "邀请在场玩家依次做个自我介绍。",
        "probe": "就玩家刚说的话追问一个具体细节，把话头递回去。",
        "greet_new": "招呼第一次来的客人，主动提出请一杯。",
        "teach_order": "用一两句话说明这里的点单方式。",
        "announce_rules": "宣布接下来小游戏的规则，简短。",
        "ask_question": "出一个关于星空或咖啡的小问题让玩家猜。",
        "wrap_up": "收尾，并把舞台交还给玩家。",
        "answer_question": "回答玩家的问题。",
        "recall": "你想起对方之前说过的偏好或事情，主动提起来确认。",
        "acknowledge": "简短回应对方，表示你在听。",
        "fallback": "随口接一句话，保持气氛。",
    }

    def _generate_speech(
        self,
        intent: str,
        memories: list[Any],
        utterance: Optional[Utterance],
        decision: Any,
    ) -> str:
        """有模型就让模型说，没模型就用模板兜底。两条路都必须受人设约束。"""
        if self.config.use_llm_speech and self.llm.available:
            generated = self._llm_speech(intent, memories, utterance)
            if generated:
                return generated
        return self._render_intent(intent, memories, utterance)

    def _llm_speech(
        self,
        intent: str,
        memories: list[Any],
        utterance: Optional[Utterance],
    ) -> Optional[str]:
        hint = self.INTENT_HINTS.get(intent, "自然地接一句话。")
        prompt = f"""{self.persona.system_block()}

【现场】
{self.state.scene_block()}

【你想起的事】
{self.memory.context_block(memories)}

【玩家刚说】
{utterance.render() if utterance else "（没有人说话，冷场了）"}

【这一轮你要做的】
{hint}

直接输出你要说的台词。不要加引号，不要解释，不要旁白，不要写动作描写。"""
        try:
            text = self.llm.complete(
                [{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.speech_max_tokens,
            ).strip()
        except LLMUnavailable:
            return None
        return text or None

    def _render_intent(
        self,
        intent: str,
        memories: list[Any],
        utterance: Optional[Utterance],
        extra: Optional[dict[str, Any]] = None,
    ) -> str:
        tracker = self.state
        target = None
        if utterance is not None and utterance.speaker_id in tracker.players:
            target = tracker.players[utterance.speaker_id].name
        elif tracker.players:
            target = tracker.active_players()[0].name

        topic_hint = ""
        if intent in ("ask_question",):
            topic_hint = "露台朝北，能看到什么"
        elif intent == "opening":
            topic_hint = "今天露台的星星不错"

        memory_hint = ""
        for record in memories:
            if any(hint in record.content for hint in RECALL_HINTS):
                memory_hint = extract_memory_hint(record.content)
                break

        return self.persona.render_template(
            intent,
            target=target or "你",
            topic_hint=topic_hint,
            memory_hint=memory_hint or "那件事",
            answer_hint="这个我还没想过，你怎么看？",
            **(extra or {}),
        )

    @staticmethod
    def _has_recallable(memories: list[Any], now: int) -> bool:
        """判断有没有值得当面回引的记忆。

        必须排除**本轮刚写进去的**记忆 —— 否则玩家说"我第一次来"，
        NPC 下一句就是"你之前提过你第一次来"，变成尴尬的复读。
        回引的前提是"过去"确实存在。
        """
        return any(
            m.tick < now
            and m.importance >= 0.55
            and any(hint in m.content for hint in RECALL_HINTS)
            for m in memories
        )

    def _memory_query(self, utterance: Optional[Utterance], decision: Any) -> str:
        parts = []
        if utterance is not None:
            parts.append(utterance.text)
        parts.append(self.state.topic)
        parts.extend(m.name for m in self.state.players.values())
        return " ".join(p for p in parts if p)

    # ------------------------------------------------------------------ #
    def render_last_turn(self) -> str:
        if not self.turns:
            return "(还没有行动)"
        return self.turns[-1].render()
