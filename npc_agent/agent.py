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
from typing import Any, Optional, Sequence

from .config import RuntimeConfig
from .env.base import Environment
from .llm.base import LLM, LLMUnavailable
from .modules.dialogue import AddresseeSelector, DialogueConfig, TurnManager
from .modules.memory import MemoryManager, MemoryStore
from .modules.persona import Persona
from .modules.planner import Planner, render_condition
from .modules.reflection import Reflector
from .modules.repetition import find_repeat, similarity
from .modules.state import StateTracker
from .modules.tools import ToolContext, ToolRegistry
from .types import ActionCall, ActionResult, AgentTurn, Plan, PlanStep, Utterance


def _tag_plan(plan: Optional[Plan], source: str) -> Optional[Plan]:
    """给计划标上来源，然后原样返回。

    见 `Plan.source` 的注释：规划失败会**静默回落**到启发式规划器，
    不标来源的话，"模型规划的"和"回落之后启发式规划的"在任何地方
    都长得一模一样。所以这个标记不能省 —— 它是"这次到底是谁在规划"
    的唯一依据。`plan` 是 `None` 时原样返回，方便直接包住 return。
    """
    if plan is not None:
        plan.source = source
    return plan


#: 记忆里出现这些词，说明是值得当面回引的偏好类信息
RECALL_HINTS = ("喜欢", "讨厌", "习惯", "常来", "第一次", "答应", "约定")

#: 记多少条自己说过的话。查重与 prompt 都读它。
#:
#: 为什么不留全部：`said` 越长，"和很久以前那句撞了"就越容易误判 ——
#: 隔了 20 轮再说一句"嗯——"不算复读，连着两轮说才算。
#: 8 条 ≈ 最近三四轮（每轮最多两句话），正好覆盖"玩家能听出来的重复"。
SAID_WINDOW = 8

#: prompt 里最多列几条自己的近话。列太多会把注意力从玩家身上拉走。
SAID_IN_PROMPT = 4

#: 陈述句的意图阶梯。
#:
#: **复读的根因不是模型不行，是意图选择塌缩**：`_respond` 原来把
#: "玩家说了句陈述"一律映射成 `acknowledge`，而 acknowledge 的模板是
#: 一句固定的话 —— 实测 10 轮对话里「阿澈说的我记下了。」说了 7 遍。
#:
#: 给同一类输入两个以上可选意图、并按"用得最少"轮换，逐字复读就没了。
#: 顺序有意义：先 `acknowledge`（应一声，最安全），再 `probe`（追问，把话头递回去）——
#: 后者正是阿柚人设里"喜欢用问题回答问题"的那一条。
STATEMENT_LADDER = ("acknowledge", "probe")

#: 撞了复读、需要换一个意图时说哪些话是**语义上可以替换**的。
#:
#: 刻意不含 `answer_question`：把"回答"换成"应一声"是丢信息，
#: 不是去重 —— 宁可让它重复，也不能让它答非所问。
#: `unknown`（"我说不好"）在名单里：它本来就是一句没有信息量的话，
#: 换一种说法完全等价，而它恰恰是最容易被连着说两次的那一句。
NOVEL_FALLBACK_INTENTS = ("acknowledge", "probe", "unknown", "fallback")

#: 这些意图说出的话**不含新信息**（"我记下了" / "后来呢？" / "我说不好"）。
#: 它们合起来是一个**有限的池子**（每人设约 12 条），长对话必然耗尽 ——
#: 实测（离线，5 个场景）：10 轮复读率 0%，20 轮 0%~40%（合计 22%）。
#:
#: ⚠️ **试过但没用的修法，别重复踩**：连续应声 N 次之后强制改去
#: `tell_fact` 分享一个还没讲过的话题。量下来是**中性的**
#: （在那时的构建上 20 轮 18.1% → 18.1%）—— 因为话题只有 3~4 个，
#: 分享完还是回到同一个池子。
#: 模板生成器的词汇量是**有限**的，复读率的下界就是
#: `(轮数 − 词汇量) / 轮数`。要真正解决长对话，得接模型
#: （`use_llm_speech`，prompt 里已经带了【你最近说过】）。
#: 实测：同一个 20 轮对话，离线 25% → 真实模型 **0%**。

# 记忆内容的句子边界。必须包含「；」——
# 巩固后的摘要用「；」把多条 episodic 拼在一起，
# 不在这里切断就会拼出「…越酸越好。；小，是这个没错吧？」
_CLAUSE_BREAK = "。！？；"
# 巩固产出的 semantic 记录带的前缀，回引时要剥掉
_SUMMARY_PREFIX = "（早前对话摘要）"
_EDGE_PUNCT = "。！？，、；:： "


def swap_person(text: str) -> str:
    """把「我/你」整体互换 —— 用于把**别人的话**转成可以直接说出口的第二人称。

    为什么必须是**同时**互换，而不是「我→你」单向替换：

    | 玩家原话 | 单向「我→你」 | 同时互换（正确） |
    |---|---|---|
    | 我喜欢偏酸的 | 你喜欢偏酸的 ✅ | 你喜欢偏酸的 ✅ |
    | 我觉得你不错 | 你觉得你不错 ❌ | 你觉得我不错 ✅ |
    | 我们常来 | 你们常来 ✅ | 你们常来 ✅ |

    单向替换在第二种上会把自己绕进去（"你觉得你不错"）。同时互换用占位符
    走一遍，`我们`/`我的`/`你们` 这些复合词会自动跟着对。

    ⚠️ 只对**别人说的话**用。NPC 自己写的记忆（`remember` 工具）本来就是
    第一人称，互换会把「小满说的我记下了」变成「小满说的你记下了」。
    """
    if not text:
        return text
    return text.replace("我", "\x00").replace("你", "我").replace("\x00", "你")


def _split_speaker(text: str) -> tuple[str, str]:
    """把 `某某说：…` 拆成（说话人, 内容）。没有前缀时说话人为空串。

    ⚠️ **冒号前那一段结尾还有一个「说」字**（`observe` 写的是
    ``f"{speaker_name}说：{text}"``）。忘了剥它，`speaker != own_name`
    就永远成立 —— 于是 NPC **自己**说的话也被换人称，说出
    「你觉得我不错」这种错位。这个 bug 是 `test_own_words_are_not_switched`
    抓出来的（写完先跑测试，别等）。
    """
    if "：" not in text:
        return "", text
    speaker, _, body = text.partition("：")
    speaker = speaker.strip()
    if speaker.endswith("说"):
        speaker = speaker[:-1].strip()
    return speaker, body


def extract_memory_hint(content: str, limit: int = 18, own_name: str = "") -> str:
    """从一条记忆里抽出可以直接说出口的短句。

    四步：剥掉摘要前缀 → 剥掉「某某说：」前缀 → **别人说的话换人称** →
    在第一个句子边界处切断。

    早期实现是 ``content.split("：", 1)[-1][:18].rstrip(...)``。
    按字数硬截有两个问题：一是会截到半个词，二是对巩固后的摘要无效 ——
    ``阿澈说：我特别喜欢偏酸的咖啡，越酸越好。；小满说：…`` 截出来是
    ``我特别喜欢偏酸的咖啡，越酸越好。；小``，于是 NPC 说出
    「…越好。；小，是这个没错吧？」。**按语义边界切，不按字数切。**

    ## 人称那一步是被一个真实跑批逼出来的

    记忆存的是**玩家原话**（`observe` 写成 ``阿澈说：我喜欢偏酸的咖啡``）。
    剥掉「阿澈说：」之后剩下第一人称，再塞进人设模板
    ``对了，你之前提过{memory_hint}，是这个没错吧？``，就成了：

        「对了，你之前提过**我**特别喜欢偏酸的咖啡，是这个没错吧？」

    —— NPC 把玩家的话当成了自己的话。三个人设的 recall 模板全是这个形状，
    所以这是**全量**的，不是某一句的偶发。而且六维评测给这些用例全打了
    **1.000**：`recall_in_speech` 只查子串在不在，看不见人称对不对。

    修法：剥掉「某某说：」时如果说话人**不是自己**，就把这段内容的人称换过来
    （`swap_person`）。`own_name` 为空时保持原样 —— 老调用方（和纯函数测试）
    不传就不改变行为。
    """
    text = (content or "").strip()
    if text.startswith(_SUMMARY_PREFIX):
        text = text[len(_SUMMARY_PREFIX) :]

    speaker, body = _split_speaker(text)
    if speaker and own_name and speaker != own_name:
        body = swap_person(body)
    text = body

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
        *,
        reset_env: bool = True,
    ) -> None:
        self.persona = persona
        self.env = env
        self.scenario = scenario
        self.llm = llm
        self.config = config or RuntimeConfig()
        self.id = persona.id
        # 目标可以指定 owner。多 NPC 场景里每个 NPC 只领自己的那份，
        # 共享的联合目标（不写 owner）两边都会去推 —— 这是"协作"的最小机制：
        # 各干各的活，但完成条件定义在同一个世界状态上。
        self.objectives: list[dict[str, Any]] = [
            obj
            for obj in (scenario.get("objectives") or [])
            if obj.get("owner") in (None, "", self.id)
        ]
        # 自己的目标 id。state.objectives 是**整个世界的**目标表（含同伴那一份），
        # 主动推进时必须按这个集合过滤 —— 否则阿柚会跑去替小舟弹琴。
        self._objective_ids: set[str] = {
            str(obj.get("id")) for obj in self.objectives if obj.get("id")
        }
        self.reset(reset_env=reset_env)

    # ------------------------------------------------------------------ #
    def reset(self, reset_env: bool = True) -> None:
        """重置。

        ``reset_env=False`` 是多 NPC 场景必须的：多个 agent 共享同一个世界，
        如果每个 agent 都 reset 一次环境，后 reset 的会把先 reset 的
        位置、物品、世界标记全部抹掉。世界由 Cast 统一重置一次。
        """
        if reset_env:
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
                    idle_ticks_before_proactive=self.config.idle_ticks_before_proactive,
                    npc_share_ceiling=self.config.npc_share_ceiling,
                ),
            )
        )
        self.reflector = Reflector(
            self.persona, self.memory, self.llm, reflect_every=self.config.reflect_every
        )
        self.active_plan: Optional[Plan] = None
        self.turn_index = 0
        self.turns: list[AgentTurn] = []
        #: 我自己最近说过的话（**不含**玩家的话）。三处都读它：
        #: 台词 prompt（别重复）、复读闸门（撞了就换说法）、
        #: 以及"我是不是刚应过一声"。
        self.said: list[str] = []
        #: 每个意图用过几次。两个用途：多变体模板选第几个、意图阶梯按"用得最少"轮换。
        self._intent_uses: dict[str, int] = {}
        #: 最近用过的意图（按时间）。`_intent_uses` 是计数、不是顺序，
        #: 而"别连着用同一个"需要顺序 —— 两者不能合并。
        self._intent_log: list[str] = []
        #: 已经回引过的记忆 id。见 `_next_recallable`。
        self._recalled: set[str] = set()
        self._retry_counts: dict[tuple, int] = {}
        self._shared_topics: set[str] = set()
        self._last_share_tick = -99
        self._attempted: set[str] = set()
        self._observed: set[tuple] = set()
        self.state.update_from_env(self.env.observe(self.id))

    # ------------------------------------------------------------------ #
    # 只读视图
    # ------------------------------------------------------------------ #
    @property
    def planner_failures(self) -> int:
        """规划调用失败的次数。见 `Planner.failures` 的注释。"""
        return self.planner.failures

    @property
    def planner_last_error(self) -> str:
        return self.planner.last_error

    @property
    def planner_empty_plans(self) -> int:
        """模型"调用成功但计划不可用"的次数。见 `Planner.empty_plans`。"""
        return self.planner.empty_plans

    def plan_sources(self) -> dict[str, int]:
        """这个 NPC 产出过的计划**按来源**计数。

        `use_llm_planner` 开着而这里出现 `heuristic` ⇒ 那就是**静默回落**：
        模型被问过了，但它没给出可用计划，框架换成了启发式规划器。
        不记来源的话，这两种情况在数据里完全一样。
        """
        counts: dict[str, int] = {}
        for turn in self.turns:
            plan = turn.plan
            if plan is None or not plan.source:
                continue
            counts[plan.source] = counts.get(plan.source, 0) + 1
        return counts

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def observe_utterance(self, utterance: Utterance) -> bool:
        """只听不说：把一条发言记进现场状态与记忆，不触发任何回应。

        多 NPC 场景需要区分"听见"和"要回答"：
        同伴跟玩家聊天时，我也该把内容记下来（这样后面能接得上），
        但不该跟着一起开口。返回是否是新听到的。

        去重键用 (tick, 说话人, 文本)：同一个 tick 里同一个人不会说两遍同样的话，
        所以这个键足够唯一；不去重的话，调度器补听 + step 内建监听会记两次。
        """
        key = (utterance.tick, utterance.speaker_id, utterance.text)
        if key in self._observed:
            return False
        self._observed.add(key)
        self.state.note_utterance(utterance)
        self.memory.observe(utterance, self.id)
        return True

    def step(
        self,
        utterance: Optional[Utterance] = None,
        *,
        other_npc_spoke_last: bool = False,
        allow_speech: bool = True,
        count_silence: bool = True,
    ) -> AgentTurn:
        """推进一轮。

        ``other_npc_spoke_last`` / ``allow_speech`` 由多 NPC 调度器（Cast）传入：
        前者影响"要不要新建发言计划"，后者在工具层直接挡住 speak ——
        因为正在执行计划的 NPC 不会因为对话决策而停下，必须两边都拦。

        ``count_silence`` 也是给调度器用的：Cast 一个 tick 里会调用所有 agent，
        冷场计数必须一个 tick 只加一次，否则"冷场 2 轮后主动开口"会被提前触发。
        """
        started = time.perf_counter()
        self.turn_index += 1

        observation = self.env.observe(self.id)
        self.state.update_from_env(observation)
        # 世界规则会变（物品被拿走、位置改变），规划器每轮刷新一次事实表
        self.planner.facts = self.env.world_facts()

        if utterance is not None:
            self.observe_utterance(utterance)
        elif count_silence:
            self.state.tick_silence()

        plan_pending = bool(self.active_plan and not self.active_plan.done)
        decision = self.turn_manager.next_decision(
            self.state,
            utterance,
            has_pending_plan=plan_pending,
            other_npc_spoke_last=other_npc_spoke_last,
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

        ctx = ToolContext(
            actor_id=self.id,
            tick=self.state.tick,
            env=self.env,
            memory=self.memory,
            persona=self.persona,
            tracker=self.state,
            npc_share_ceiling=self.config.npc_share_ceiling,
            allow_speech=allow_speech,
        )

        # 玩家在问我（提问 / 点我的名）→ **先回答**。
        #
        # ⚠️ 回答和"干自己的活"是**两件事**，都要做：
        #   - 只回答不开计划 → NPC 变成应答机，自己的目标永远推不动
        #     （实测：duet 的 `turn_taking` 用例因此掉到 task=0.00 —— 拿铁没端出去）；
        #   - 只开计划不回答 → **答非所问**，问「这店开了多久了」答「点单很简单…」。
        # 所以这里是"先回答、再照常开计划跑计划"，
        # 计划里那句重复的话由 `_run_plan` 让开（一轮只说一句）。
        answered = False
        if (
            not plan_pending
            and decision.should_speak
            and self._player_is_asking_me(utterance)
        ):
            self._respond(turn, ctx, memories, utterance, decision)
            answered = True

        # 只在本轮之前没有计划时才新建计划。
        # 本轮刚建的计划已经考虑了玩家这句话，不该再额外插一句回应 ——
        # 除非那句回应是**回答**（上面那一步），那时计划里的开场白就该让开。
        started_this_turn = False
        if not plan_pending and decision.should_speak:
            self.active_plan = self._make_plan(utterance, decision, memories)
            started_this_turn = self.active_plan is not None

        if self.active_plan and not self.active_plan.done:
            next_step = self.active_plan.next_step
            if (
                not started_this_turn
                and not answered
                and self._player_wants_me(utterance)
            ):
                if utterance is not None and utterance.is_question:
                    # 计划是之前就在跑的，这时玩家提问 → **回答优先于计划**。
                    # 否则就会出现"玩家在问问题，NPC 却在背教程"的经典翻车。
                    self._respond(turn, ctx, memories, utterance, decision)
                elif next_step is None or next_step.tool != "speak":
                    # 玩家说了话但计划里本来没安排发言 —— 先应一声，再继续干活。
                    # 不这么做的话，NPC 在"跑自己的计划"期间是**完全听不见的**：
                    # 实测（tutorial）玩家连说三句，NPC 一声不吭去后厨拿牛奶。
                    self._quick_acknowledge(turn, utterance, ctx, memories)
            self._run_plan(turn, ctx, memories, utterance)
            turn.plan = self.active_plan
        elif decision.should_speak and not answered:
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

        self._note_spoken(turn)

        turn.latency_ms = (time.perf_counter() - started) * 1000
        self.turns.append(turn)
        return turn

    # ------------------------------------------------------------------ #
    def _note_spoken(self, turn: AgentTurn) -> None:
        """把这一轮**真的说出口**的话记进 `said`。

        ⚠️ 判据是 `(action, result)` 而不是 `turn.say`，两个理由：

        1. `_quick_acknowledge` 那一声（"哎，我在"）**不写 turn.say** ——
           只看 turn.say 就会漏掉它，于是"刚应过一声"下一轮又会应一声。
        2. `turn.say` 是"最后一句"，而一轮里可能说了两句
           （先应一声、再正式回答）。

        用 `result.detail` 而不是 `action.args["text"]`：`detail` 是
        `persona.apply_style` **裁过**的最终文本，也就是玩家真正听到的那句。
        拿没裁过的原文去查重，会把"裁完其实一模一样"的两句判成不同。
        """
        for call, result in zip(turn.actions, turn.results):
            if call.tool == "speak" and result.ok and result.detail:
                self.said.append(result.detail)
        if len(self.said) > SAID_WINDOW:
            del self.said[:-SAID_WINDOW]

    def _player_is_asking_me(self, utterance: Optional[Utterance]) -> bool:
        """玩家是不是在**问我**（提问，或者点我的名）？

        这是"别背固定教程"的闸门判据：问句的正确回应是回答，
        而不是开一个"推进场景目标"的计划（那样 NPC 会答非所问地背教程）。

        ⚠️ 这里原来写的是 `decision.urgency >= 0.9` —— 一个**代理量**，
        而 `decide()` 里"有人提问"那一支的 urgency 只有 0.6，
        "继续执行未完成的计划"那一支只有 0.55 ⇒ 问句永远过不了这个闸门。
        判据换成**直接问"是不是在问我"**，就不用再维护这个代理量的隐含假设。

        ⚠️ **下单不算提问，哪怕它写成问句**：「能给我来杯拿铁吗？」
        正确反应是「好，稍等，我这就去弄」+ 真去做那杯，
        而不是回一句泛泛的话、再把 `accept_order` 那句让掉（点单闭环就断了）。
        判据复用 `planner.is_request`，不另写一套（两套必然漂移）。
        """
        if utterance is None:
            return False
        if self.planner.is_request(utterance.text):
            return False
        return utterance.is_question or self.turn_manager.selector.mentions_npc(
            utterance.text
        )

    def _player_wants_me(self, utterance: Optional[Utterance]) -> bool:
        """玩家这句话是不是**冲我来的**？—— 计划不能压过"玩家在跟我说话"。

        ⚠️ 调用点原来写的是 `decision.urgency >= 0.9`，那是个**代理量**，
        而且它**永远不成立**：`decide()` 里"继续执行未完成的计划"那一支的
        urgency 只有 0.55，而它排在"被点名"（0.95）和"有人提问"（0.6）**前面** ——
        于是只要计划没跑完，"回答优先于计划"那段代码**一行都执行不到**（死代码）。

        实测后果（tutorial，控制台默认路径）：玩家连说
        「我想喝点酸的」「你叫什么名字」「这店开了多久了」，
        NPC 一声不吭地去后厨拿牛奶、做拿铁、递咖啡 —— **三个问题一个没答**。
        这就是"NPC 只顾说自己的、对话推不动"。

        判据改成**直接问"这句话是不是给我的"**，和 `decide()` 第 4 支
        （"这句话不是对我说的"）保持同一个语义：明确在跟别人说话就不插嘴。
        """
        if utterance is None:
            return False
        selector = self.turn_manager.selector
        if utterance.mentions and not selector.mentions_npc(utterance.text):
            return False  # 明确在跟别人说话，不插嘴
        return utterance.speaker_id in self.state.players or utterance.is_question

    def pick_intent(self, candidates: Sequence[str]) -> str:
        """从候选意图里挑一个**用得最少**的。

        复读的根因不是模型不行，是**意图选择塌缩**：见 `STATEMENT_LADDER` 的注释。

        用**计数**而不是"最近 N 条窗口"：计数天然是确定的，
        同一段对话重放必然得到同一串意图 —— 控制台每次请求都从头重放整段对话，
        靠的就是这个可复现性，不能为了去重把它换成一个随机的选择。

        ⚠️ 它**只选不消费**：自增发生在 `_bump_intent`（`_generate_speech` /
        `_quick_acknowledge` 里）。所以连着调两次会得到同一个答案 —— 这不是 bug，
        但调了不用就等于没轮换。写新调用点时要记得把选出来的意图真的说出去。
        """
        if not candidates:
            raise ValueError("候选意图不能为空")
        index, _ = min(
            enumerate(candidates),
            key=lambda pair: (self._intent_uses.get(pair[1], 0), pair[0]),
        )
        return candidates[index]

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
                return _tag_plan(request_plan, "request_template")

        # 2) 玩家在问"我该怎么用 / 教教我" → 直接走场景引导流程。
        #    新客问"这里怎么点单"，正确的回答就是那套引导动作，而不是一句客套话。
        if utterance is not None and self.planner.wants_scenario_flow(utterance.text):
            guide_plan = self.planner.plan_next_objective(
                self.objectives, self.state, self._attempted
            )
            if guide_plan and guide_plan.steps:
                return _tag_plan(guide_plan, "scenario_flow")

        # 3) 被**点名** → 不启动场景目标，只回答（`_respond` 本轮已经答过了）。
        #    这一条是"不背固定教程"的关键闸门。
        #
        #    ⚠️ 这里原来写的是 `decision.urgency >= 0.9` —— 一个**代理量**。
        #    只有"被点名"那一支会给 0.95，所以它其实等价于"被点名"；
        #    但用代理量写有两个坏处：读者看不出这条闸门管的是谁，
        #    而且**提问那一支的 0.6 会静默漏过去**，于是"玩家在问我"
        #    照样会去开一个背教程的计划（实测：问「这店开了多久了」答「点单很简单…」）。
        #    问句不在这里拦 —— 它由 `step()` 先回答、再照常开计划（见那里的注释）。
        if utterance is not None and self.turn_manager.selector.mentions_npc(
            utterance.text
        ):
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
                return _tag_plan(llm_plan, "model")

        # 5) 兜底：推进场景里还没完成的目标
        #
        #    ⚠️ 走到这里有两种情况，**必须能分辨**：
        #      (a) `use_llm_planner` 关着（`--no-planner` 对照组）—— 正常；
        #      (b) 开着但模型没给出可用计划 —— 这就是**静默回落**，
        #          实测 2026-09-19 那次 231 条跑批里有 48% 的用例发生过。
        #    靠 `Plan.source == "heuristic"` + 配置里的 `use_llm_planner` 分辨。
        return _tag_plan(
            self.planner.plan_next_objective(
                self.objectives, self.state, self._attempted
            ),
            "heuristic",
        )

    def _pending_goal_hint(self) -> str:
        """喂给规划 prompt 的"还没完成的目标"。

        ⚠️ **必须带上 `success_when`。** 只给 goal 文本的话，模型会规划出
        "听起来完成了目标"的动作，但那个动作不满足机器判定的完成条件。

        实测（duet，模型规划，4 次里 3 次失败）：小舟的 `play_song` 完成条件是
        `{flag: song_started}`，模型规划的是
        `speak → emote(play_guitar) → start_activity(song_request)` ——
        全是"像在起歌"的动作，**唯独没有人 `set_flag(song_started)`**，
        于是目标永远 pending，依赖它的联合目标也跟着挂住。

        这是**信息不对称**（启发式规划器读得到条件，模型读不到），
        和"规划 prompt 里没有配方表所以想不到 take_item"同一类。
        """
        lines: list[str] = []
        for obj in self.objectives:
            if self.state.objectives.get(obj.get("id"), "pending") == "done":
                continue
            lines.append(f"- {obj.get('goal')}")
            condition = obj.get("success_when")
            if condition:
                lines.append(
                    f"  · 完成条件（**做到这个才算完成**）：{render_condition(condition)}"
                )
        return "\n".join(lines)

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

            # 本轮话头已经给同伴了。这一步不是失败，是让位 ——
            # 所以标 skipped 直接跳过，而不是交给工具去撞一个失败，
            # 更不该触发重规划（重规划会白烧一次模型调用）。
            if step.tool == "speak" and not ctx.allow_speech:
                step.status = "skipped"
                step.note = "本轮已有另一位 NPC 开口，让出话头"
                executed += 1
                continue

            # **一轮只说一句话。**
            # 本轮已经说过了（回答了玩家的问题 / 应了一声），计划里这句就略过 ——
            # 不然"先回答、再跑计划"会变成一轮里连说两句，
            # 而且计划那句会盖掉回答（`turn.say` 是"最后一句"），
            # 界面上就只剩答非所问的那句了。
            # 标 skipped 而不是 failed：这不是做不到，是这一轮不需要它。
            if step.tool == "speak" and turn.say:
                step.status = "skipped"
                step.note = "本轮已经说过一句（回答玩家），这句略过"
                executed += 1
                continue

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
                    # 走 _generate_speech 而不是直接 _render_intent：
                    # 否则计划步骤会绕过模型、退回背课文，
                    # NPC 就变成"接玩家话时像人，干活时像复读机"。
                    args["text"] = self._generate_speech(
                        intent, memories, utterance, extra=extra
                    )
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
        recall_record = None
        if decision.proactive:
            # 只推进**自己的**目标。state.objectives 是整个世界的目标表，
            # 里面还挂着同伴那一份 —— 不过滤的话，冷场时阿柚会去"推进"小舟的
            # 弹琴目标，做出越俎代庖的动作。
            pending = [
                k
                for k, v in self.state.objectives.items()
                if v != "done" and k in self._objective_ids
            ]
            if pending:
                intent = self.turn_manager.selector.pick_proactive_intent(
                    self.state, only=self._objective_ids, used=self._recent_intents()
                )
            elif self._proactive_share(turn, ctx):
                # 目标都做完了 → 主动分享店里的事（主动使用工具，而不是干聊）
                return
            else:
                # 没什么可说的就保持安静，比复读一句"嗯"更像人
                return
        elif utterance is not None and utterance.is_question:
            recall_record = self._next_recallable(memories, self.state.tick, utterance)
            if recall_record is not None:
                intent = "recall"
            elif self._direct_answer(utterance):
                # 能直接答（问到自己 / 知识库对得上）→ 真的答（见 `_direct_answer`）
                intent = "answer_question"
            else:
                # 答不上来。**单独一个意图**，不是"answer_question 的兜底文本"——
                # 因为这一句一定会被反复用到（玩家的开放式问题是无限的），
                # 它需要自己的变体轮换和去重闸门，而 answer_question 不能换说法。
                intent = "unknown"
        else:
            recall_record = self._next_recallable(memories, self.state.tick, utterance)
            if recall_record is not None:
                intent = "recall"
            elif utterance is not None:
                # 陈述句：**不再一律 acknowledge**。见 `STATEMENT_LADDER`。
                intent = self.pick_intent(STATEMENT_LADDER)
            else:
                intent = "fallback"

        text = self._generate_speech(
            intent, memories, utterance, decision, recall_record=recall_record
        )
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
            if recall_record is not None:
                # **说出口了**才算回引过。被发言占比上限拦下的那次不算 ——
                # 否则这条记忆永远没机会被说出来（标记了却没说，等于丢掉）。
                self._recalled.add(recall_record.id)

    def _recent_intents(self, window: int = 2) -> list[str]:
        """最近说过的几句话分别用的什么意图。给"别连着用同一个"用。"""
        return self._intent_log[-window:]

    def _next_recallable(
        self, memories: list[Any], now: int, utterance: Optional[Utterance] = None
    ) -> Optional[Any]:
        """还没回引过的、值得回引的那条记忆。

        ⚠️ **没有这一步，`recall` 会把同一条记忆反复回引。**
        检索每轮都会返回同一条（它是打分最高的），`_recallable` 每轮都放行，
        于是 NPC 每隔一轮就说一遍「你上次说过第一次来吧。」
        —— 实测（duet，10 轮）这一句出现了 **4 次**，比 acknowledge 更像复读机：
        acknowledge 至少是"应一声"，而这条是**主动把同一个话题捡起来又说一遍**，
        正好就是"NPC 反复重复一个话题，对话推不动"。

        回引过一次就够了 —— 玩家已经知道你记得。
        """
        for record in self._recallable(memories, now, utterance):
            if record.id not in self._recalled:
                return record
        return None

    def _proactive_share(self, turn: AgentTurn, ctx: ToolContext) -> bool:
        """主动透露一个还没聊过的话题。展示"主动使用工具"的能力。

        话题要先过两道筛子：人设允许聊（`can_discuss`）+ 世界已解锁（`available_topics`）。
        少任何一道，NPC 就会反复尝试说一件它其实不该说的事。

        **要有间隔**，否则会变成"每冷场一次就丢一个知识点"的复读机。
        """
        allowed = set(self.env.available_topics(self.id))
        topics = [
            t
            for t in self.persona.can_discuss
            if t in allowed and t not in self._shared_topics
        ]
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
        """被点名时先应一声，再去回答。

        这里**刻意**用模板而不是模型：它是一句不含信息的语气词（"哎，我在"），
        却要在玩家提问的同一轮里抢在正式回答之前说出来。
        让模型生成它只会白白多一次推理延迟，换不来任何表达价值。
        代价是它会拉低"自由台词率"——这是指标口径问题，不是质量问题，
        所以报告里读这个数时要记得扣除这类语气词。

        ⚠️ 但它**也要走变体轮换**：这句话每被点名一次就说一遍，
        原来写死取变体 0，于是同一句"某某说的我记下了。"会连着出现好几次
        —— 恰恰是最容易被玩家一眼看穿的那一类复读（因为它最机械）。
        """
        variant = self._bump_intent("acknowledge")
        text = self.persona.render_template(
            "acknowledge", variant=variant, target=utterance.speaker_name
        )
        call = ActionCall("speak", {"text": text, "to": utterance.speaker_id}, reason="被点名先应一声")
        result = self.registry.execute(call, ctx)
        turn.actions.append(call)
        turn.results.append(result)
        # ⚠️ 必须写 `turn.say`，和 `_respond` / `_run_plan` 保持一致。
        # 不写的话这一声在**界面上完全不存在**：`_turn_payload` 会把成功的
        # `speak` 从 actions 里滤掉（理由是"已经由 say 表达了"），
        # 而 `say` 是空的 —— 两边一起把它抹掉，页面显示"（没有说话）"，
        # 可 NPC 明明说了。这正是 `_turn_payload` 注释里防的那类假话，
        # 只是从另一条路进来的。实测：tutorial 第 2 轮页面上是沉默，
        # 而 `env.utterances` 里躺着「小鹿说的我记下了。」。
        if result.ok and not turn.say:
            turn.say = result.detail

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
        "unknown": "这个问题你不知道答案，老实说不知道，不要编。",
        "recall": "你想起对方之前说过的偏好或事情，主动提起来确认。",
        "acknowledge": "简短回应对方，表示你在听。",
        "fallback": "随口接一句话，保持气氛。",
    }

    def _bump_intent(self, intent: str) -> int:
        """记一次"这个意图被用了一次"，返回**本次该用第几个变体**。

        读在自增之前，所以同一个意图的变体是轮着用的
        （第 1 次用变体 0、第 2 次用变体 1…）。

        抽成一个小函数是因为有**两条**出话路径都要记账：
        `_generate_speech`（走模型或模板）和 `_quick_acknowledge`
        （刻意只用模板，为了不为一句话多花一次推理延迟）。
        两边各写一遍自增，早晚会漏一处 —— 漏了那处的变体就永远停在第一个。
        """
        variant = self._intent_uses.get(intent, 0)
        self._intent_uses[intent] = variant + 1
        self._intent_log.append(intent)
        if len(self._intent_log) > SAID_WINDOW:
            del self._intent_log[:-SAID_WINDOW]
        return variant

    def _generate_speech(
        self,
        intent: str,
        memories: list[Any],
        utterance: Optional[Utterance],
        decision: Any = None,
        extra: Optional[dict[str, Any]] = None,
        recall_record: Any = None,
    ) -> str:
        """有模型就让模型说，没模型就用模板兜底。两条路都必须受人设约束。

        ``extra`` 是模板槽位的补充事实（例如 ``item_name=拿铁``）。
        走模型时也要把它喂进去，否则模型不知道自己在交付什么，
        会说出"你的咖啡好了"这种丢信息的话。

        ## 复读闸门

        生成完要**和 `self.said` 比一次**。命中就：
        模型路径再问一次（并明确告诉它撞了哪句），仍然撞就换一个
        **最近没用过的意图**的模板 —— 那个模板必然是新话。

        闸门放在这里而不是放在 `_respond`，是因为**所有**台词都从这里出去：
        即时回应、计划步骤里的 speak、主动分享，一条都漏不掉。
        只堵 `_respond` 的话，"接玩家话时像人、干活时像复读机"照样会发生。
        """
        # 多变体模板选第几个 —— 读在自增之前，所以同一个意图的变体是轮着用的
        variant = self._bump_intent(intent)

        text = ""
        if self.config.use_llm_speech and self.llm.available:
            text = self._llm_speech(intent, memories, utterance, extra) or ""
            clash = find_repeat(text, self.said) if text else None
            if clash:
                # 只重问一次：第二次还撞说明模型认死了这一句，
                # 再问下去只是白烧一次推理延迟（这个模型单次 7~14s）。
                text = (
                    self._llm_speech(intent, memories, utterance, extra, avoid=clash)
                    or text
                )

        if text and not find_repeat(text, self.said):
            return text

        # 模板兜底（没模型 / 模型返回空 / 模型还是撞了）
        rendered = self._render_intent(
            intent, memories, utterance, extra, recall_record=recall_record, variant=variant
        )
        if not find_repeat(rendered, self.said):
            return rendered

        # 连模板都撞了 —— 换一个**最近没用过**的意图。只在这几个之间换：
        # 它们语义上可以互相替代，换掉不会丢信息（见 NOVEL_FALLBACK_INTENTS）。
        if intent in NOVEL_FALLBACK_INTENTS:
            for alt in self._novel_intents(intent):
                # ⚠️ 试**全部**变体，不是只试"下一个"。
                # 变体是循环使用的：用到第 4 次时"下一个变体"很可能正是
                # 很久以前说过的那个 —— 只试一个的话闸门会以为自己已经尽力了。
                for variant_index in range(self.persona.template_count(alt)):
                    candidate = self._render_intent(
                        alt,
                        memories,
                        utterance,
                        extra,
                        recall_record=recall_record,
                        variant=variant_index,
                    )
                    if not find_repeat(candidate, self.said):
                        return candidate

        # 全都撞了：宁可说一句重复的，也不要突然沉默 ——
        # 沉默会让 turn.say 变空，而"这一轮到底说没说话"是评测在断言的。
        return rendered

    def _novel_intents(self, exclude: str) -> list[str]:
        """按"用得最少"排序的替代意图，排除当前这个。"""
        pool = [i for i in NOVEL_FALLBACK_INTENTS if i != exclude]
        pool.sort(key=lambda i: (self._intent_uses.get(i, 0), NOVEL_FALLBACK_INTENTS.index(i)))
        return pool

    def _said_block(self) -> str:
        """把"我自己最近说过什么"渲染成 prompt 片段。"""
        recent = self.said[-SAID_IN_PROMPT:]
        if not recent:
            return "（你还没说过话）"
        return "\n".join(f"  - {line}" for line in recent)

    def _llm_speech(
        self,
        intent: str,
        memories: list[Any],
        utterance: Optional[Utterance],
        extra: Optional[dict[str, Any]] = None,
        avoid: Optional[str] = None,
    ) -> Optional[str]:
        hint = self.INTENT_HINTS.get(intent, "自然地接一句话。")
        extra_block = ""
        if extra:
            facts = "；".join(f"{k}：{v}" for k, v in extra.items() if v)
            if facts:
                extra_block = f"\n\n【本轮的事实】\n{facts}"
        # ⚠️ 这一块是治复读的关键，而且是**必须**的：
        # 记忆库只存玩家说的话（`MemoryManager.observe` 明确跳过自己），
        # 现场状态里也没有对话历史 —— 也就是说，在加上这一段之前，
        # 模型**根本不知道自己刚才说了什么**，它只能凭运气不重复。
        # 实测（tutorial，真实模型 6 轮）它碰巧没重复，但那是运气不是设计：
        # 换一个话题更集中的场景，它就会绕回同一件事。
        said_block = f"""
【你最近说过】（**不要重复这些内容**，也不要换个说法说同一件事）
{self._said_block()}
"""
        avoid_block = ""
        if avoid:
            avoid_block = (
                f"\n\n⚠️ 你上一版写的「{avoid}」和你说过的话重复了。"
                f"请换一个**完全不同**的说法，或者追问一个具体细节。"
            )
        prompt = f"""{self.persona.system_block()}

【现场】
{self.state.scene_block()}
{said_block}
【你想起的事】
{self.memory.context_block(memories)}

【玩家刚说】
{utterance.render() if utterance else "（没有人说话，冷场了）"}

【这一轮你要做的】
{hint}{extra_block}{avoid_block}

直接输出你要说的台词。不要加引号，不要解释，不要旁白，不要写动作描写。
如果确实没有新东西可说，就追问一个**具体**的细节，而不是重复已经说过的话。"""
        try:
            text = self.llm.complete(
                [{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.speech_max_tokens,
            ).strip()
        except LLMUnavailable:
            return None
        return text or None

    def _direct_answer(self, utterance: Optional[Utterance]) -> str:
        """玩家的问题能不能**直接答**？能就返回那句答话，不能返回空串。

        三条路，按"答错了最难看"排序：

        1. **问到 NPC 自己**（`persona.self_facts`）—— 优先级最高。
           离线路径答不上这类问题时会说「这个我不太清楚」，
           而玩家问的是"你叫什么名字" —— **NPC 不知道自己的名字**，
           比复读还难看（复读只是没信息，这是人设当场崩掉）。
        2. **世界知识**（`world_facts["knowledge"]`）—— 问题里出现话题标题。
        3. 都没有 → 返回空串，由调用方改用 `unknown` 意图老实说不知道。

        ⚠️ 这里原来是一个**固定字符串**「这个我还没想过，你怎么看？」——
        于是 NPC 对**每一个**问题都用同一句话回避。
        实测（控制台默认 10 轮对话）：hosting 和 icebreaker 各出现 2 次，
        而且它比复读更糟：复读是"说了等于没说"，这是**把问题踢回给玩家**。
        玩家问了两遍「你叫什么名字」，对话还停在原地 —— 就是"推不动"。

        **只做精确匹配，不做模糊匹配**：在这种短句上模糊匹配会答出胡话 ——
        "这店开了多久了"和「咖啡屋的故事」字符重合度看着不低，
        但"我下次还来"也一样不低，于是 NPC 会开始对着闲聊背知识条目。
        **宁可少答，不能乱答。** 开放式问题的正解是接模型（`use_llm_speech`）。
        """
        question = (utterance.text if utterance is not None else "") or ""
        if not question:
            return ""

        about_self = self.persona.answer_about_self(question)
        if about_self:
            return about_self

        knowledge = (self.env.world_facts() or {}).get("knowledge") or {}
        if not knowledge:
            return ""
        allowed = set(self.env.available_topics(self.id))
        for topic, entry in knowledge.items():
            if topic not in allowed:
                continue
            # 标题命中（精确子串）或配置里写明的关键词命中。
            # 关键词表住在**世界配置**里而不是这里 —— 咖啡屋和体素世界
            # 各有各的话题，写死在 agent 里就等于把两个世界焊在一起。
            needles = [str(entry.get("title") or "").strip()]
            needles += [str(k) for k in (entry.get("keywords") or []) if str(k).strip()]
            if any(n and n in question for n in needles):
                # ⚠️ **答过就算说过了。**
                # 不标记的话，`_share_topic` 会在后面把同一条知识再"主动分享"一遍：
                # 实测（icebreaker，20 轮）「这家店开在星屿的旧灯塔下面…」
                # 先在第 8 轮作为回答说了，又在第 11 轮作为主动分享说了一遍。
                # 两个机制各自都做了去重，但它们**不共享那份"说过了"的账**。
                self._shared_topics.add(topic)
                return str(entry.get("text") or "")
        return ""

    def _render_intent(
        self,
        intent: str,
        memories: list[Any],
        utterance: Optional[Utterance],
        extra: Optional[dict[str, Any]] = None,
        recall_record: Any = None,
        variant: int | None = None,
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
        # 和 `_recallable` 用**同一份筛选结果** —— 两边各写一遍过滤条件，
        # 就会出现"判定说可以回引，但抽出来的 hint 来自另一条记忆"。
        #
        # 优先用调用方指定的那条（`recall_record`）：回引目标的挑选发生在
        # `_respond` 里（那里要判"这条回引过没有"），如果这里再挑一次，
        # 就可能挑到**另一条** —— 于是 `_recalled` 标记的是 A，说出去的是 B。
        candidates = [recall_record] if recall_record is not None else self._recallable(
            memories, self.state.tick, utterance
        )
        for record in candidates:
            if record is None:
                continue
            # 传自己的名字进去：剥掉「某某说：」之后，**别人**说的话要换人称，
            # 否则 NPC 会把玩家的第一人称当成自己的（见 extract_memory_hint）。
            memory_hint = extract_memory_hint(
                record.content, own_name=self.persona.name
            )
            break

        return self.persona.render_template(
            intent,
            variant=variant,
            target=target or "你",
            topic_hint=topic_hint,
            memory_hint=memory_hint or "那件事",
            # 见 `_direct_answer`：这里原来写死一句"这个我还没想过，你怎么看？"，
            # 于是 NPC 对每个问题都用同一句话回避。
            answer_hint=self._direct_answer(utterance),
            **(extra or {}),
        )

    @staticmethod
    def _recallable(
        memories: list[Any], now: int, utterance: Optional[Utterance] = None
    ) -> list[Any]:
        """筛出**值得当面回引**的记忆。回引和抽 hint 必须用同一份筛选结果。

        两个条件缺一个都会说出坏话：

        1. **是"过去"的事**（`m.tick < now`）。否则玩家说"我第一次来"，
           NPC 下一句就是"你之前提过你第一次来"，尴尬的复读。
        2. **不是本轮刚说的那句话本身**。

        第 2 条是后来补的，因为 tick 判定挡不住它。实测（`memory_seat_recall`）：

            玩家：阿柚，你还记得我习惯坐哪儿吗？
            NPC ：对了，你之前提过**阿柚，我还记得你习惯坐哪儿吗**，是这个没错吧？

        玩家**问句本身**含「习惯」这个线索词，于是它被当成"值得回引的记忆"，
        NPC 把问题引回来当成了过去的事 —— 而且顺手把自己的名字说成了玩家的。
        `utterance.tick` 和 `state.tick` 谁大取决于调度顺序，靠 tick 判不稳，
        所以这里直接比对**当前这句话的文本**。
        """
        skip = (utterance.text or "").strip() if utterance is not None else ""
        out = []
        for m in memories:
            if m.tick >= now:
                continue
            if m.importance < 0.55:
                continue
            if not any(hint in m.content for hint in RECALL_HINTS):
                continue
            if skip and skip in m.content:
                continue
            out.append(m)
        return out

    @classmethod
    def _has_recallable(
        cls, memories: list[Any], now: int, utterance: Optional[Utterance] = None
    ) -> bool:
        """有没有值得当面回引的记忆 —— 见 `_recallable`。"""
        return bool(cls._recallable(memories, now, utterance))

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
