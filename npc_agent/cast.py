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
        # 分组（见 `_parse_groups`）。没有 `groups:` 的场景退化成**一个组**，
        # 于是 duet / trio 的调度语义一个字都不变 —— 分组是加法，不是改写。
        self.groups, self._group_by_actor = self._parse_groups()
        self.group_names: dict[str, str] = {
            str(key): str(value)
            for key, value in (scenario.get("group_names") or {}).items()
        }
        #: 演员表第一位所在的组。`_next_first` 始终是这个组的游标 ——
        #: 既有的单组场景与直接改 `_next_first` 的测试因此继续成立。
        self._primary_group = self.group_of(self.order[0])
        # 下一轮"没人被点名时"谁先开口。初始按配置顺序。
        self._next_first: str = self.order[0]
        self._group_next: dict[str, str] = {
            gid: members[0] for gid, members in self.groups.items()
        }
        self.rounds = 0
        self.history: list[list[AgentTurn]] = []
        # 对话线程（谁欠谁一句话 / 哪些内容点说过了 / 连聊到第几轮）。
        # 它和"记忆"是两件事：记忆是私有的、会衰减的；这块板子是公共的、只活几轮。
        self.board = ConversationBoard(chain_limit=self.config.max_peer_exchanges)
        self.reset()

    def _parse_groups(self) -> tuple[dict[str, list[str]], dict[str, str]]:
        """把场景里的 `groups:` 读成 {组 id: [成员]} 与 {成员: 组 id}。

        两种写法：

            groups:                 # 字典（推荐）
              counter: [ayou, tang]
              terrace: [xiaozhou, luo]
            groups:                 # 列表（组名里带空格/冒号时更好写）
              - {id: counter, members: [ayou, tang]}

        没配 `groups:` → 全部人一个组 `main`。这是 duet / trio 的现状，
        也是"分组是加法"的保证：单组时所有新逻辑都退化成旧行为。

        ⚠️ 配了分组就必须**把人分全**。少一个人不是"默认随便放进某组"，
        而是配置错了 —— 那个 NPC 会永远不在任何发言队列里，所有模块指标
        都正常，只有玩家发现"第四个人从来没说过话"。所以这里直接报错。
        """
        raw = self.scenario.get("groups")
        if not raw:
            return {"main": list(self.order)}, {pid: "main" for pid in self.order}

        pairs: list[tuple[str, list[str]]] = []
        if isinstance(raw, dict):
            pairs = [(str(gid), list(members or [])) for gid, members in raw.items()]
        elif isinstance(raw, list):
            for entry in raw:
                if not isinstance(entry, dict):
                    raise ValueError(f"groups 的列表项必须是对象：{entry!r}")
                pairs.append(
                    (
                        str(entry.get("id") or ""),
                        list(entry.get("members") or entry.get("npcs") or []),
                    )
                )
        else:
            raise ValueError("groups 必须是字典或列表")

        groups: dict[str, list[str]] = {}
        group_of: dict[str, str] = {}
        for gid, members in pairs:
            if not gid:
                raise ValueError("groups 里有一个分组没有 id")
            if gid in groups:
                raise ValueError(f"分组 {gid!r} 定义了两次")
            if not members:
                raise ValueError(f"分组 {gid!r} 没有成员")
            for pid in members:
                pid = str(pid)
                if pid not in self.agents:
                    raise ValueError(f"分组 {gid!r} 里的 {pid!r} 不在演员表 {self.order}")
                if pid in group_of:
                    raise ValueError(
                        f"{pid!r} 同时出现在 {group_of[pid]!r} 和 {gid!r} 两个分组里"
                    )
                group_of[pid] = gid
                groups.setdefault(gid, []).append(pid)

        missing = [pid for pid in self.order if pid not in group_of]
        if missing:
            raise ValueError(
                f"这些 NPC 没有分组：{missing}。配了 groups 就必须把人分全 —— "
                "漏掉的人永远不会被排进任何发言队列。"
            )
        return groups, group_of

    # ------------------------------------------------------------------ #
    # 只读视图
    # ------------------------------------------------------------------ #
    @property
    def npc_ids(self) -> list[str]:
        return list(self.agents)

    @property
    def group_map(self) -> dict[str, str]:
        """{NPC id: 组 id}。评测按 (tick, 组) 判撞车时读它。"""
        return dict(self._group_by_actor)

    def group_of(self, actor_id: str) -> str:
        return self._group_by_actor.get(actor_id, "")

    def group_name(self, group_id: str) -> str:
        return self.group_names.get(group_id, group_id)

    def group_members(self, group_id: str) -> list[str]:
        return list(self.groups.get(group_id, []))

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
        self._group_next = {gid: members[0] for gid, members in self.groups.items()}
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
    def speaking_order(
        self, utterance: Optional[Utterance], *, group: Optional[str] = None
    ) -> list[str]:
        """决定这一轮 NPC 的出场顺序（默认返回全场的拼接顺序）。

        被点名的人排第一 —— 玩家的指令优先于调度器的公平性，
        "会听话"比"轮流发言"重要。

        其余按 ``_next_first`` 起头轮转。这个游标不是简单的 round-robin：
        它指向**上一轮没轮上的人**，所以话多的 NPC 不会把安静的 NPC 饿死。

        **冷场轮再按"说过几句话"排一次。** 轮转只记得上一轮谁没开口，
        而冷场主动开口要攒够 N 轮 —— 中间只要有人说过一句，计数就清零。
        实测（duet 的 `duet_named` 用例）：阿柚在第 1 轮回应玩家，
        第 3 轮又轮到它冷场开口，小舟从头到尾没说过一个字。
        活世界不该是"谁先开口谁一直开口"。

        `group=` 只看某一个组。多组场景里，每组各有一个游标 ——
        共用一个游标会让第二组永远排在后面，第三、四个人形同虚设。
        """
        if group is not None:
            return self._group_order(group, utterance)
        out: list[str] = []
        for gid in self.groups:
            out.extend(self._group_order(gid, utterance))
        return out

    def _group_order(
        self,
        group_id: str,
        utterance: Optional[Utterance],
        obligation: Any = None,
    ) -> list[str]:
        """一个组内部的出场顺序。单组时与旧的 ``speaking_order`` 完全一致。"""
        ids = self.groups.get(group_id, [])
        if len(ids) <= 1:
            return list(ids)

        if obligation is not None and obligation.target_id in ids:
            target = obligation.target_id
            return [target] + [pid for pid in ids if pid != target]

        named = [
            pid
            for pid in (utterance.mentions if utterance is not None else [])
            if pid in ids
        ]
        if named:
            first = named[0]
            return [first] + [pid for pid in ids if pid != first]

        # 主组始终读 `_next_first`（既有测试会直接改它）；其余组各读各的游标。
        cursor = (
            self._next_first
            if group_id == self._primary_group
            else self._group_next.get(group_id, ids[0])
        )
        start = ids.index(cursor) if cursor in ids else 0
        rotation = [ids[(start + i) % len(ids)] for i in range(len(ids))]
        if utterance is None:
            rank = {pid: index for index, pid in enumerate(rotation)}
            rotation.sort(
                key=lambda pid: (self.agents[pid].state.npc_utterances, rank[pid])
            )
        return rotation

    # ------------------------------------------------------------------ #
    # 频道：谁听得见、谁有发言权
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_private(utterance: Optional[Utterance]) -> bool:
        return utterance is not None and utterance.channel == "private"

    def _private_targets(self, utterance: Utterance) -> list[str]:
        """私聊的收件 NPC。

        优先用产出方声明的 `addressed_to`；没有就取 audience 里的 NPC。
        两者都没有 ⇒ 这是一条**没有收件人的悄悄话**，按公开处理会泄漏，
        所以返回空——导演会让整轮无人接话。失败方向必须是"少说一句"。
        """
        if utterance.addressed_to in self.agents:
            return [utterance.addressed_to]  # type: ignore[list-item]
        return [pid for pid in (utterance.audience or ()) if pid in self.agents]

    def _active_groups(self, utterance: Optional[Utterance]) -> set[str]:
        """这一轮哪些组可以开口（不等于一定开口）。

        * 冷场轮：所有组都能自己决定要不要主动开口；
        * 公开玩家发言：所有组都听得见，但**点名只叫得动被点的那一组** ——
          点阿柚做咖啡时，露台那组不该同时冒出一句无关的话；
        * 限范围发言（audience 非空）：只有收件人所在的组；
        * 私聊：只有收件人所在的那一组，而且组里只有他一个人有发言权。
        """
        if utterance is None:
            return set(self.groups)
        if self._is_private(utterance):
            return {
                self.group_of(pid)
                for pid in self._private_targets(utterance)
                if pid in self.agents
            }
        if utterance.audience is not None:
            return {
                self.group_of(pid)
                for pid in utterance.audience
                if pid in self.agents
            }
        named = [pid for pid in utterance.mentions if pid in self.agents]
        if named:
            return {self.group_of(pid) for pid in named}
        return set(self.groups)

    def _hears(self, actor_id: str, utterance: Optional[Utterance]) -> bool:
        """这句话是否落在某个 NPC 的收件范围里。

        `audience=None` 是公开频道，所有人都听得见（玩家公开发言、
        单组场景里的 NPC 台词）；给了 audience 就只认名单。
        """
        if utterance is None:
            return False
        if utterance.audience is None:
            return True
        return actor_id in utterance.audience

    def speakers_by_group_tick(self) -> dict[int, dict[str, list[str]]]:
        """每轮每个组里谁开过口。

        多组场景的撞车判据是 `(tick, 组)`，不是 `tick` ——
        两组在同一个 tick 各说一句是**并行对话**，不是抢话。
        """
        out: dict[int, dict[str, list[str]]] = {}
        for utterance in self.env.utterances:
            if utterance.speaker_id not in self.agents:
                continue
            gid = self.group_of(utterance.speaker_id)
            speakers = out.setdefault(utterance.tick, {}).setdefault(gid, [])
            if utterance.speaker_id not in speakers:
                speakers.append(utterance.speaker_id)
        return out

    def collisions(self) -> list[tuple[int, str, list[str]]]:
        """同一轮同一个组里有两个人开口 —— 这才是真的撞车。"""
        out: list[tuple[int, str, list[str]]] = []
        for tick, groups in sorted(self.speakers_by_group_tick().items()):
            for gid, speakers in groups.items():
                if len(speakers) > 1:
                    out.append((tick, gid, list(speakers)))
        return out

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
                    # 同伴对聊只发生在**同一个组**里。多组场景下把这句话的
                    # 收件范围收紧到那一组，别让另一组听见半截对话。
                    if len(self.groups) > 1:
                        gid = self.group_of(obligation.target_id)
                        utterance.audience = tuple(self.groups.get(gid, []))
                        utterance.channel = gid
            else:
                self.board.expire()

        private = self._is_private(utterance)
        private_targets = self._private_targets(utterance) if private else []
        private_target = private_targets[0] if private_targets else ""
        # 没有收件人的悄悄话：整轮无人接话。宁可少说一句，也不能广播。
        active_groups = self._active_groups(utterance)
        order = [
            pid
            for pid in self._order_for(utterance, obligation)
            if self.group_of(pid) in active_groups
        ]
        turns: list[AgentTurn] = []
        # 每组的发言权各自收放：A 组说过话不影响 B 组开口。
        spoke_groups: set[str] = set()

        for pid in order:
            agent = self.agents[pid]
            gid = self.group_of(pid)
            before = len(self.env.utterances)
            heard = utterance if self._hears(pid, utterance) else None
            if private:
                # 悄悄话只有收件人能答。同组的人在场，但既听不见也插不上话。
                allowed = bool(private_target) and pid == private_target
            elif peer_turn:
                # 同伴轮：只有被点名的人有发言权。其余人这一轮只是"在场"。
                allowed = obligation is not None and pid == obligation.target_id
            else:
                allowed = gid not in spoke_groups
            turn = agent.step(
                heard,
                # 已经有人开口 → **同组**后面的 NPC 让出话头。两道闸门都要给：
                # 对话层管"要不要新建发言计划"，工具层管"正在跑的计划里的 speak"。
                other_npc_spoke_last=(gid in spoke_groups),
                allow_speech=allowed,
                # 冷场是**世界**的属性：这一轮没人开口，每个人都该记一轮。
                # 曾经写成 `index == 0`（"同一轮只加一次"），但发言顺序是轮转的
                # ⇒ 每个 NPC 每 N 轮才加一次，阈值 2 在 duet 要 3 轮、trio 要 4 轮
                # 才触发，而界面上写着"冷场 2 轮"。已经有人开口就不再记
                # （这一轮不是冷场）；开口的人稍后由自己清零（见 agent.step）。
                # 分组之后这条按组计：A 组聊得热闹，不该把 B 组的冷场计数清零。
                # 同伴轮 / 私聊不算冷场 —— 有人正在说话。
                count_silence=(
                    not private
                    and not peer_turn
                    and gid not in spoke_groups
                    and heard is None
                ),
            )
            turns.append(turn)

            new_speeches = [
                u
                for u in self.env.utterances[before:]
                if u.speaker_id == pid and u.tick == tick
            ]
            if not new_speeches:
                continue
            spoke_groups.add(gid)
            for spoken in new_speeches:
                self._stamp_scope(
                    spoken, private_input=utterance if private else None
                )
            # NPC 之间说的话，同伴也要听见。
            # Agent.step 内部的监听只处理传进来的那条玩家发言，
            # 不补这一步，"阿柚转头跟小舟说：你来弹一首" 就断在半路 ——
            # 小舟的现场状态和记忆里根本没有这句话，自然接不上。
            #
            # ⚠️ 曾经试过"只让同一个地点的人听见"（想给传闻腾出空间），
            # 结果是**既有契约被破坏**：`multi_turn_taking` 用例明确要求
            # 小舟记得阿柚说过的话（"没说话的听见了吗"）。
            # 现在按**频道**过滤：公开频道（单组场景）仍然人人听得见，
            # 分组/私聊则只送到收件范围 —— 那正是分组存在的意义。
            for other_id, other in self.agents.items():
                if other_id == pid:
                    continue
                for spoken in new_speeches:
                    if self._hears(other_id, spoken):
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
                # 每转一手：细节掉一层（fidelity × 0.82），
                # 但对转述者的信任让确信度再涨一点 —— 见 `memory.receive_gossip`。
                agent.memory.receive_gossip(
                    content=fact.content,
                    tick=tick,
                    source_id=fact.source_id,
                    source_name=fact.source_name,
                    importance=min(0.7, fact.importance),
                    subject=fact.subject,
                    hop=fact.hop + 1,
                    fidelity=round(fact.fidelity * 0.82, 4),
                    confidence=min(0.98, round(fact.confidence + 0.05, 4)),
                )
                # 听到同伴的分享，关系近一点（单向：听的人对说的人）。
                self._bump_affinity(pid, fact.source_id, 1)

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

    def _stamp_scope(
        self,
        spoken: Utterance,
        *,
        private_input: Optional[Utterance] = None,
    ) -> None:
        """给 NPC 刚说出口的话盖上车道标记。

        两件事：

        1. **私聊的回复也要是私聊。** 玩家悄悄问，NPC 大声答，
           等于把悄悄话广播给全店 —— 泄漏发生在回复上，不在提问上。
        2. **多组场景里，NPC 的话只送到自己的组。** 不隔离的话，
           两组并行对话会互相串台，"分组"就只剩一个配置字段。

        单组场景**什么都不盖**：保留 `audience=None` 的公开语义，
        既有场景的观察范围一个字都不变。
        """
        if private_input is not None:
            spoken.channel = "private"
            audience = tuple(private_input.audience or ())
            if not audience:
                target = self._private_targets(private_input)
                audience = tuple(target)
            spoken.audience = audience
            return
        if len(self.groups) <= 1:
            return
        if spoken.audience is not None or spoken.channel:
            return
        gid = self.group_of(spoken.speaker_id)
        members = self.groups.get(gid, [])
        if members:
            spoken.audience = tuple(members)
            spoken.channel = gid

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

        私聊**不进公共对话板**：板子会开"下一轮该谁接"的义务，
        而悄悄话不该把第三个人卷进来。私聊的来回由玩家自己继续。
        """
        if spoken.channel == "private":
            return
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
        members = self.groups.get(self.group_of(spoken.speaker_id), [])
        if (
            target is None
            and peer_turn
            and spoken.addressed_to is None
            and len(members) == 2
        ):
            # 这个组里只有两个人、说话的是 NPC、而且是问句 → 问的就是另一个。
            # 用**组内人数**而不是全场人数：分组之后，"另一个"必须是同组的
            # 那一个；拿全场人数判，四人场景里会挑到一个听不见这句话的人。
            others = [pid for pid in members if pid != spoken.speaker_id]
            if others and spoken.is_question and spoken.speaker_id in self.agents:
                target = others[0]

        if target is not None and self.group_of(target) != self.group_of(spoken.speaker_id):
            # 跨组点名不作数：对方根本听不见，开出来的义务会让他下一轮
            # 对着空气回答一句 —— 这是分组隔离最容易漏的一处。
            target = None

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
        # 说上话，两个人的关系就近一点 —— 双向，但分量不同：
        # 被点名的人得到的是"被惦记"，点名的人得到的是"我主动了"。
        self._bump_affinity(spoken.speaker_id, target, 2)
        self._bump_affinity(target, spoken.speaker_id, 1)

    def _bump_affinity(self, actor_id: str, other_id: str, delta: int) -> None:
        """调整角色之间的好感。**只对场上真实存在的角色生效** ——
        配置里写错一个 id，静默失败比崩掉更难查，所以这里不报错但也不乱写。
        """
        actor = getattr(self.env, "actors", {}).get(actor_id)
        if actor is None or other_id not in self.env.actors:
            return
        actor.bump_affinity(other_id, delta)

    def _advance_floor(self, order: list[str], tick: int) -> None:
        """把下一轮的话头交给**每个组**里这一轮没轮上的人。

        单组时与旧实现逐字等价（`order` 就是全场顺序）。
        多组时每组各推各的游标 —— 共用一个游标会让一组永远压着另一组。
        """
        by_group: dict[str, list[str]] = {}
        for pid in order:
            if pid in self.agents:
                by_group.setdefault(self.group_of(pid), []).append(pid)
        spoke = self._speakers_this_tick(tick)
        for gid, members in by_group.items():
            if len(members) <= 1:
                nxt = members[0]
            else:
                silent = [pid for pid in members if pid not in spoke]
                # 有人开口、有人没轮到 → 下一轮让没开口的先来，避免被饿死；
                # 全开口或全没开口 → 正常轮转。
                nxt = silent[0] if spoke and silent else members[1]
            self._group_next[gid] = nxt
            if gid == self._primary_group:
                self._next_first = nxt

    # ------------------------------------------------------------------ #
    def transcript(self) -> list[str]:
        """整场对话的带名转写，用于报告与人工复盘。"""
        return [f"{u.speaker_name}: {u.text}" for u in self.env.utterances]
