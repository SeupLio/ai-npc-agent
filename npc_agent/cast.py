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
from .env.base import Environment
from .env.star_isle import DEFAULT_START, StarIsleEnv
from .llm.base import LLM
from .modules.persona import Persona
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
) -> "Cast":
    """从场景配置造出一个能直接跑的剧组（含世界）。

    CLI、评测 harness、单元测试都走这一个入口 ——
    三处各写一遍构造顺序，早晚会有一处忘了把 cast 传给环境，
    结果就是"场景里配了两个 NPC，环境里只造出一个"。
    """
    personas = load_cast(scenario)
    if not personas:
        raise ValueError(
            f"场景 {scenario.get('id')} 没有配置 NPC（需要 npc: 或 npcs:）"
        )
    env = StarIsleEnv(scenario, cast=env_cast(scenario, personas))
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
        被点名时先应的那一声（``_quick_acknowledge``）不写 turn.say，
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

        返回按出场顺序排列的 turns（可能有人这一轮什么都没说）。
        """
        tick = self.env.tick
        order = self.speaking_order(utterance)
        turns: list[AgentTurn] = []
        spoke = False

        for index, pid in enumerate(order):
            agent = self.agents[pid]
            before = len(self.env.utterances)
            turn = agent.step(
                utterance,
                # 已经有人开口 → 后面的 NPC 让出话头。两道闸门都要给：
                # 对话层管"要不要新建发言计划"，工具层管"正在跑的计划里的 speak"。
                other_npc_spoke_last=spoke,
                allow_speech=not spoke,
                # 一个 tick 只让第一个人记一次冷场。否则"冷场 2 轮后主动开口"
                # 会被同轮的多个人各加一次，提前触发。
                count_silence=(index == 0),
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
            for other_id, other in self.agents.items():
                if other_id == pid:
                    continue
                for spoken in new_speeches:
                    other.observe_utterance(spoken)

        self._advance_floor(order, tick)
        self.rounds += 1
        self.history.append(turns)
        self.env.advance_tick()
        return turns

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
