"""多 NPC 协作调度器 —— Cast（导演）。

单 NPC 时，Agent 自己决定"要不要说话"就够了。
场上同时站着两个 NPC，就多出三个单 Agent 场景里不存在的问题：

    1. **谁先说**    被点名的先说；没人被点名就轮转，而且上一轮没轮到的人优先
    2. **会不会撞车** 一个 tick 里只能有一个 NPC 开口
    3. **没说话的听见了吗** 听见 ≠ 要回答，但必须记下来，否则后面接不上话

这三个问题都不能交给 NPC 自己解决 —— 每个 NPC 只看得见自己的状态，
它不知道同伴这一轮打不打算开口。所以需要一个站在 NPC 之上的"导演"：
Cast 持有**共享的 env** 和**各自独立的 Agent**，统一收发发言权。

    env    共享 —— 两个人在同一家店里，位置、物品、世界标记只有一份
    agent  独立 —— 记忆、状态、计划都是私有的，两个 NPC 不该共用一个脑子

这条界线就是"多 Agent"和"一个 Agent 分饰两角"的区别：
前者是真的有两个各怀心思的个体，只是站在同一个世界里。
"""

from __future__ import annotations

from typing import Any, Optional

from .agent import NPCAgent
from .config import RuntimeConfig, load_persona
from .env import build_env
from .env.base import Environment
from .env.mc_client import WorldClient
from .env.star_isle import DEFAULT_START
from .llm.base import LLM
from .modules.conversation import ConversationBoard
from .modules.motives import MotiveBoard, WorldEvent
from .modules.persona import Persona
from .modules.social import GossipNetwork
from .types import AgentTurn, Utterance


# --------------------------------------------------------------------------- #
# 场景配置 → 演员表
# --------------------------------------------------------------------------- #
def cast_specs(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    """从场景配置里读出演员表。两种写法都支持：

        npcs:                        # 多 NPC
          - {id: ayou, persona: ayou, start: counter}
          - {id: xiaozhou, persona: xiaozhou, start: terrace}

        npc: ayou                    # 单 NPC（老写法，一行都不用改）
        npc_start: counter

    统一成 [{id, persona, start}]，上层就不用到处写 if 了。
    """
    specs = scenario.get("npcs") or []
    default_start = scenario.get("npc_start", DEFAULT_START)
    if specs:
        out: list[dict[str, Any]] = []
        for raw in specs:
            spec = dict(raw)
            spec.setdefault("persona", spec.get("id"))
            spec.setdefault("start", default_start)
            out.append(spec)
        return out

    npc_id = scenario.get("npc")
    if not npc_id:
        return []
    return [{"id": npc_id, "persona": npc_id, "start": default_start}]


def load_cast(scenario: dict[str, Any]) -> list[Persona]:
    """按演员表把人设文件读出来。顺序与 cast_specs 一致。"""
    return [
        Persona.from_dict(load_persona(spec["persona"] or spec["id"]))
        for spec in cast_specs(scenario)
    ]


def env_cast(scenario: dict[str, Any], personas: list[Persona]) -> list[dict[str, Any]]:
    """环境要的那份演员表：{id, name, start}。

    name 从人设里取，而不是从场景配置里再写一遍 ——
    两处各写一份名字，改了一处漏一处，NPC 就会在别人的点名里叫不出自己。
    """
    specs = cast_specs(scenario)
    out = []
    for spec, persona in zip(specs, personas):
        out.append(
            {"id": persona.id, "name": persona.name, "start": spec.get("start") or DEFAULT_START}
        )
    return out


def build_cast(
    scenario: dict[str, Any],
    llm: LLM,
    config: RuntimeConfig | None = None,
    *,
    client: WorldClient | None = None,
) -> "Cast":
    """从场景配置造出一个能直接跑的剧组（含世界）。

    CLI、评测 harness、单元测试都走这一个入口 ——
    三处各写一遍构造顺序，早晚会有一处忘了把 cast 传给环境，
    结果就是"场景里配了两个 NPC，环境里只造出一个"。

    `client` 只在 minecraft 环境有意义：传了就接真实世界，
    不传就用进程内体素世界。测试与离线评测永远不传。
    """
    personas = load_cast(scenario)
    if not personas:
        raise ValueError(
            f"场景 {scenario.get('id')} 没有配置 NPC（需要 npc: 或 npcs:）"
        )
    # 走环境注册表，而不是直接 new StarIsleEnv：
    # 场景 YAML 里写 env: minecraft 就能换到体素世界，这里一行都不用改。
    env = build_env(scenario, cast=env_cast(scenario, personas), client=client)
    return Cast(scenario, env, personas, llm, config)


# --------------------------------------------------------------------------- #
# 导演
# --------------------------------------------------------------------------- #
class Cast:
    """一组共享同一个世界的 NPC，外加一个负责发言权的导演。"""

    def __init__(
        self,
        scenario: dict[str, Any],
        env: Environment,
        personas: list[Persona],
        llm: LLM,
        config: RuntimeConfig | None = None,
    ) -> None:
        if not personas:
            raise ValueError("Cast 至少需要一个 NPC")
        self.scenario = scenario
        # 场景标识：存档要靠它判断"这份档是不是同一场戏"。
        self.scenario_id = str(scenario.get("id") or scenario.get("name") or "")
        self.env = env
        self.config = config or RuntimeConfig()
        # 每个 NPC 一个独立 Agent。reset_env=False 是必须的：
        # 世界是共享的，构造期间谁都不该去重置它（下面 reset() 统一重置一次）。
        self.agents: dict[str, NPCAgent] = {
            persona.id: NPCAgent(
                persona, env, scenario, llm, self.config, reset_env=False
            )
            for persona in personas
        }
        self.order: list[str] = list(self.agents)
        # 下一轮"没人被点名时"谁先开口。初始按配置顺序。
        self._next_first: str = self.order[0]
        self.rounds = 0
        self.history: list[list[AgentTurn]] = []
        # 对话线程（谁欠谁一句话 / 哪些内容点说过了 / 连聊到第几轮）。
        # 它和"记忆"是两件事：记忆是私有的、会衰减的；这块板子是公共的、只活几轮。
        self.board = ConversationBoard(chain_limit=self.config.max_peer_exchanges)
        self.reset()

    # ------------------------------------------------------------------ #
    # 只读视图
    # ------------------------------------------------------------------ #
    @property
    def npc_ids(self) -> list[str]:
        return list(self.agents)

    @property
    def is_multi_npc(self) -> bool:
        return len(self.agents) > 1

    @property
    def lead(self) -> NPCAgent:
        """领衔 NPC（演员表第一位）。只给单 NPC 的检视类命令用。"""
        return self.agents[self.order[0]]

    def name_of(self, actor_id: str) -> str:
        agent = self.agents.get(actor_id)
        return agent.persona.name if agent else actor_id

    def memory_contents(self) -> dict[str, list[str]]:
        """每个 NPC 各自的记忆库。

        多 NPC 评测必须分开看 —— 否则"阿柚记住了客人的偏好"会被算成
        "整个剧组都记住了"，记忆指标就废了。
        """
        return {
            pid: [r.content for r in agent.memory.store.records]
            for pid, agent in self.agents.items()
        }

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """重置世界和所有 NPC。

        **环境只 reset 一次。** 让每个 agent 各自 reset 的话，
        后一个会把前一个的位置、背包、世界标记全部抹掉 ——
        它们操作的是同一个世界。这是多 NPC 最容易踩的坑，
        所以 NPCAgent.reset 才要带一个 reset_env 开关。
        """
        self.env.reset()
        for agent in self.agents.values():
            agent.reset(reset_env=False)
        self._next_first = self.order[0]
        self.rounds = 0
        self.history = []
        self.board = ConversationBoard(chain_limit=self.config.max_peer_exchanges)
        # 动机裁定台。场景没配 `motives:` 时是空的 —— 空裁定台不产生任何事件，
        # 所以其余四个场景的行为一个字都不变。
        self.motives = MotiveBoard.from_dicts(
            self.scenario.get("motives"),
            min_tick=self.config.motive_resolve_after,
        )
        #: 这一局发生过的世界事件（按时间）。报告与转写从这里取。
        self.events: list[WorldEvent] = []
        # 传闻网络：这一局里"谁说过什么"是公共事实，和各自的记忆分开。
        self.gossip = (
            GossipNetwork(threshold=self.config.gossip_importance)
            if self.config.gossip_enabled
            else None
        )
        # 重置会重建对话板，注入要跟着重做 —— 否则 agent 手里还攥着上一局的板子，
        # "这个话题说过了"会在新一局里继续生效。
        for agent in self.agents.values():
            agent.board = self.board
            agent.gossip = self.gossip
            agent.pending_motive = None
            # 编年史是**引用**而不是拷贝：新事件随时会追加进来，
            # NPC 下一轮就该能提到它。
            agent.chronicle = self.events
        # 同伴名单也是运行期的（见 `Persona.peers`）：单人场景没有这一项，
        # 多人场景每个 NPC 都要知道"场上还有谁"，否则模型会对着空气说话。
        for pid, agent in self.agents.items():
            agent.persona.peers = [
                {
                    "id": other.id,
                    "name": other.persona.name,
                    "role": other.persona.role,
                }
                for oid, other in self.agents.items()
                if oid != pid
            ]

    # ------------------------------------------------------------------ #
    # 发言权调度
    # ------------------------------------------------------------------ #
    def speaking_order(self, utterance: Optional[Utterance]) -> list[str]:
        """决定这一轮 NPC 的出场顺序。

        被点名的人排第一 —— 玩家的指令优先于调度器的公平性，
        "会听话"比"轮流发言"重要。

        其余按 ``_next_first`` 起头轮转。这个游标不是简单的 round-robin：
        它指向**上一轮没轮上的人**，所以话多的 NPC 不会把安静的 NPC 饿死。
        """
        ids = self.order
        if len(ids) <= 1:
            return list(ids)

        named = [
            pid
            for pid in (utterance.mentions if utterance is not None else [])
            if pid in self.agents
        ]
        if named:
            first = named[0]
            return [first] + [pid for pid in ids if pid != first]

        start = ids.index(self._next_first) if self._next_first in ids else 0
        return [ids[(start + i) % len(ids)] for i in range(len(ids))]

    def _speakers_this_tick(self, tick: int) -> set[str]:
        """这一轮里真的开过口的 NPC。

        以环境里的发言记录为准，而不是 turn.say ——
        工具层拦下的那次 `speak`（``allow_speech=False``）不会写 turn.say，
        只看 turn.say 会漏判，于是两个 NPC 在同一 tick 都开口。
        """
        return {
            u.speaker_id
            for u in self.env.utterances
            if u.tick == tick and u.speaker_id in self.agents
        }

    # ------------------------------------------------------------------ #
    # 一轮
    # ------------------------------------------------------------------ #
    def step(self, utterance: Optional[Utterance] = None) -> list[AgentTurn]:
        """推进一轮：所有 NPC 依次行动，然后时间前进一格。

        调用方负责把玩家发言写进环境（``env.record_player_utterance``），
        这里负责分发和收发言权，并在结束时 ``advance_tick`` ——
        一个 tick 就是一轮，全部 NPC 共享同一个 tick，
        否则"谁和谁在同一轮说话"这件事没法判定。

        ## 两种轮次

        **玩家轮**（``utterance`` 给了）：玩家说话，NPC 依次决定要不要接。

        **同伴轮**（``utterance`` 是 None，但对话板上欠着一句话）：
        把上一轮那句**同伴发言**当成这一轮的输入，交给被点名的人。
        没有这一支，NPC 之间的对话就是断的 —— 实测（duet，8 轮离线）
        "对另一个 NPC 说的"台词 **0 句**，因为同伴的话从来没有被当成"输入"，
        它只被记进了记忆（听见 ≠ 要回答）。

        两条边界：

        * 一个 tick 仍然**只有一个人**开口。同伴轮里没被点名的人一律
          ``allow_speech=False`` —— 否则两个 NPC 会同时接同一句话。
        * 玩家一开口就清空义务。玩家插话的优先级高于"NPC 还没聊完"，
          这和人打断两个人说话时的预期一致。

        返回按出场顺序排列的 turns（可能有人这一轮什么都没说）。
        """
        tick = self.env.tick
        peer_turn = False
        obligation = None

        if utterance is not None:
            # 玩家开口 → 这段 NPC 对聊结束，欠着的话作废。
            self.board.player_turn()
        elif self.board.pending is not None:
            pending = self.board.pending
            # 上一轮开的债，这一轮才还。同轮开同轮还等于自己跟自己说话。
            if pending.tick < tick and pending.target_id in self.agents:
                obligation = self.board.take(pending.target_id)
                if obligation is not None:
                    peer_turn = True
                    utterance = obligation.as_utterance()
            else:
                self.board.expire()

        order = self._order_for(utterance, obligation)
        turns: list[AgentTurn] = []
        spoke = False

        for index, pid in enumerate(order):
            agent = self.agents[pid]
            before = len(self.env.utterances)
            # 同伴轮：只有被点名的人有发言权。其余人这一轮只是"在场"。
            allowed = (not spoke) and (not peer_turn or pid == obligation.target_id)
            turn = agent.step(
                utterance,
                # 已经有人开口 → 后面的 NPC 让出话头。两道闸门都要给：
                # 对话层管"要不要新建发言计划"，工具层管"正在跑的计划里的 speak"。
                other_npc_spoke_last=spoke,
                allow_speech=allowed,
                # 一个 tick 只让第一个人记一次冷场。否则"冷场 2 轮后主动开口"
                # 会被同轮的多个人各加一次，提前触发。
                # 同伴轮不算冷场 —— 有人正在说话。
                count_silence=(index == 0 and not peer_turn),
            )
            turns.append(turn)

            new_speeches = [
                u
                for u in self.env.utterances[before:]
                if u.speaker_id == pid and u.tick == tick
            ]
            if not new_speeches:
                continue
            spoke = True
            # NPC 之间说的话，同伴也要听见。
            # Agent.step 内部的监听只处理传进来的那条玩家发言，
            # 不补这一步，"阿柚转头跟小舟说：你来弹一首" 就断在半路 ——
            # 小舟的现场状态和记忆里根本没有这句话，自然接不上。
            #
            # ⚠️ 曾经试过"只让同一个地点的人听见"（想给传闻腾出空间），
            # 结果是**既有契约被破坏**：`multi_turn_taking` 用例明确要求
            # 小舟记得阿柚说过的话（"没说话的听见了吗"）。
            # 传闻因此换了内容 —— 它传的不是原话，是**判断**（见 `social.py`）。
            for other_id, other in self.agents.items():
                if other_id == pid:
                    continue
                for spoken in new_speeches:
                    other.observe_utterance(spoken)
            for spoken in new_speeches:
                self._note_dialogue(spoken, peer_turn=peer_turn)

        self._advance_floor(order, tick)
        self.rounds += 1
        self.history.append(turns)
        self._resolve_motives(tick)
        self._spread_gossip(tick)
        # 这一轮没人接那句话 → 作废。留着会让"三轮前的一句问话"
        # 在玩家插了一句之后突然被回答。
        if self.board.pending is not None and self.board.pending.tick < tick:
            self.board.expire()
        self.env.advance_tick()
        return turns

    def _resolve_motives(self, tick: int) -> None:
        """裁定动机冲突，并把"该说一句让步"的通知交给让步方。

        事件在**一轮结束时**产生，下一轮由让步方说出口 —— 不在同一轮里
        一边裁定一边发言：那样玩家看到的是"世界刚决定完，NPC 立刻宣布"，
        像念旁白。
        """
        for event in self.motives.poll(tick, {pid: a.persona.name for pid, a in self.agents.items()}):
            self.events.append(event)
            loser = self.agents.get(event.loser)
            if loser is not None:
                loser.pending_motive = event

    def _spread_gossip(self, tick: int) -> None:
        """把这一轮新说出口的事实，投给还没听过的同伴。

        投递发生在**一轮结束时**，而不是说话人开口的瞬间 —— 否则
        同一轮里后面说话的 NPC 会在同一 tick 就"听说了"一件事，
        而玩家看到的是两个人几乎同时知道，像是共享了同一个脑子。
        隔一轮，才像"传话"。
        """
        if self.gossip is None:
            return
        for pid, agent in self.agents.items():
            for fact in self.gossip.digest(pid):
                agent.memory.receive_gossip(
                    content=f"听{fact.source_name}说，{fact.content}",
                    tick=tick,
                    source_id=fact.source_id,
                    source_name=fact.source_name,
                    importance=min(0.7, fact.importance),
                    subject=fact.subject,
                )

    # ------------------------------------------------------------------ #
    # 同伴轮的出场顺序
    # ------------------------------------------------------------------ #
    def _order_for(
        self, utterance: Optional[Utterance], obligation: Any = None
    ) -> list[str]:
        """这一轮谁先开口。

        同伴轮：被点名的人**必须**排第一（他不是"优先"，他是唯一有发言权的人）。
        玩家轮：沿用 ``speaking_order``（点名 > 轮转）。
        """
        if obligation is not None and obligation.target_id in self.agents:
            target = obligation.target_id
            return [target] + [pid for pid in self.order if pid != target]
        return self.speaking_order(utterance)

    def _note_dialogue(self, spoken: Utterance, *, peer_turn: bool = False) -> None:
        """把一句 NPC 发言记进对话板，并决定要不要开一条接话义务。

        三种情况会被认为"这句话是说给某个 NPC 的"：

        1. 产出方显式写了收件人（`addressed_to`）—— 最可靠，优先用；
        2. 文本里点了同伴的名（`mentions`）—— 次可靠；
        3. **同伴轮里**的一句问句，场上只有两个人 —— 那就是问另一个人的。

        第 3 条**只在同伴轮生效**。这一条是被一次实测逼出来的：
        阿柚在跑自己的计划时，计划里那句开场白是「第一次来吧？我请你一杯，
        想喝什么？」—— 一句对**全场**说的问句。按"2 个 NPC + 问句"判，
        小舟会以为在问自己，于是回一句「……还行。」，她自己的目标永远没开工。
        玩家看到的是：一个 NPC 在忙，另一个只会应声。
        """
        target = None
        if spoken.addressed_to in self.agents and spoken.addressed_to != spoken.speaker_id:
            target = spoken.addressed_to
        elif spoken.addressed_to:
            # 产出方**显式**指定了收件人（是某个玩家）→ 别抢。
            # 少了这一支，阿柚问玩家「还想到什么？」，小舟会当成在问自己 ——
            # 实测 12 轮里出现过一次，玩家看到的是"两个 NPC 在互相接话，
            # 而那句话明明是问我的"。
            target = None
        else:
            for pid in spoken.mentions:
                if pid in self.agents and pid != spoken.speaker_id:
                    target = pid
                    break
        if (
            target is None
            and peer_turn
            and spoken.addressed_to is None
            and len(self.agents) == 2
        ):
            # 两个 NPC 在场、说话的是 NPC、而且是问句 → 问的就是另一个人。
            others = [pid for pid in self.agents if pid != spoken.speaker_id]
            if others and spoken.is_question and spoken.speaker_id in self.agents:
                target = others[0]

        self.board.record(
            tick=spoken.tick,
            speaker_id=spoken.speaker_id,
            speaker_name=spoken.speaker_name,
            text=spoken.text,
            addressed_to=target,
        )
        if target is None:
            return
        # **只有"要一个回应"的话才开义务。**
        # 判据由产出方声明（`Utterance.expects_reply`）或它是问句；
        # 不按"说给谁"就要求回应 —— 那句「……听见了。」会变成
        # 「我记下了」→「嗯」→「听见了」的礼貌死循环（实测 10 轮里 4 轮是回声）。
        if not (spoken.expects_reply or spoken.is_question):
            return
        self.board.open(
            target_id=target,
            speaker_id=spoken.speaker_id,
            speaker_name=spoken.speaker_name,
            text=spoken.text,
            tick=spoken.tick,
            reason=f"{spoken.speaker_name}点了{self.name_of(target)}",
        )

    def _advance_floor(self, order: list[str], tick: int) -> None:
        """把下一轮的话头交给"这一轮没轮上的人"。"""
        if len(order) <= 1:
            self._next_first = order[0]
            return
        spoke = self._speakers_this_tick(tick)
        silent = [pid for pid in order if pid not in spoke]
        if spoke and silent:
            # 有人开口、有人没轮到 → 下一轮让没开口的先来，避免被饿死
            self._next_first = silent[0]
        else:
            # 全开口或全没开口 → 正常轮转
            self._next_first = order[1]

    # ------------------------------------------------------------------ #
    def transcript(self) -> list[str]:
        """整场对话的带名转写，用于报告与人工复盘。"""
        return [f"{u.speaker_name}: {u.text}" for u in self.env.utterances]
