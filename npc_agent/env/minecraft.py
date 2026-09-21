"""MinecraftEnv —— 把体素世界适配成 Environment 接口。

## 这个文件存在的唯一理由

证明 `NPCAgent` 真的与环境无关。

它一个 Agent 模块都没改，一行 Agent 代码都没碰，只是把同一个
`NPCAgent` 接到一个**结构上完全不同**的世界：

    星屿咖啡屋                        Minecraft
    ──────────────────────────────    ──────────────────────────────
    位置是离散的几个房间               位置是三维坐标 + 命名地点（POI）
    背包是物品列表（有没有）           背包是 {物品: 数量}（有几个）
    拿东西 = take_item                 拿东西 = mine（挖）或 craft（合成）
    工位是隐式的（配方自带 station）   工位是显式的工作台，且配方有产出数量
    没有时间压力                       昼夜循环，夜里没光源就干不了活
    交付 = give_item                   交付 = transfer

**Agent 那边感知到的差别只有一个**：背包里的东西后面多了"×3"。
它照样规划、照样调工具、照样在被拒绝时重规划 —— 因为护栏给的理由
仍然是一句人话（"天黑了，看不清矿脉"）。

## 为什么 MinecraftEnv 不是 Environment 的"特例"

`Environment` 抽象一点没变。Minecraft 的不同之处全部被这个适配器吸收：

    observe()   把 WorldClient 的状态翻译成标准观测 schema
    dispatch()  把标准工具调用翻译成 WorldClient 操作
    tool_specs()换一套动词（mine / craft / place / transfer）

所以"换环境 = 写一个适配器"这句话在这里第一次被真正检验：
不是从零写一个环境，而是**翻译**一个已有的世界。
"""

from __future__ import annotations

from typing import Any

from ..types import ActionCall, ActionResult, Utterance, looks_like_question
from .base import Environment, ToolSpec
from .conditions import ConditionContext
from .mc_client import (
    BLOCK_NAMES,
    MC_RECIPES,
    LocalWorldClient,
    WorldClient,
    WorldClientError,
)

MAX_TRANSCRIPT = 40

#: 没写起始地点时，所有人从村口开始
DEFAULT_START = "village_square"


class MinecraftEnv(Environment):
    """体素世界的 Environment 适配器。

    只跟 `WorldClient` 说话，不关心背后是进程内模拟还是真的 Minecraft。
    """

    name = "minecraft"

    def __init__(
        self,
        scenario: dict[str, Any],
        npc_id: str = "",
        npc_name: str = "",
        *,
        cast: list[dict[str, Any]] | None = None,
        client: WorldClient | None = None,
    ) -> None:
        self.scenario = scenario
        self.cast: list[dict[str, Any]] = [dict(spec) for spec in (cast or [])]
        if not self.cast:
            self.cast = [
                {
                    "id": npc_id,
                    "name": npc_name,
                    "start": scenario.get("npc_start", "village_square"),
                }
            ]
        self.npc_id = self.cast[0]["id"]
        self.npc_name = self.cast[0]["name"]

        # 没传 client 就退回进程内世界：clone 下来就能跑，不需要装 Minecraft。
        # 这条默认值很重要 —— 它让"换环境"这件事在离线评测里也是可验证的。
        self.client: WorldClient = client or LocalWorldClient(scenario)
        self._owns_client = client is None

        # 把场景（有哪些人、有哪些地点）告诉世界。
        # 真实 Minecraft 里世界早就存在了，这一步只是把游戏内实体映射到我们的 id 上。
        self._call("configure", scenario=self._client_scenario())

        self.tick = 0
        self.events: list[str] = []
        self.objective_specs: list[dict[str, Any]] = list(scenario.get("objectives") or [])
        self.objective_state: dict[str, str] = {
            o["id"]: "pending" for o in self.objective_specs
        }
        self.settable_flags: set[str] = set(
            scenario.get("world", {}).get("settable_flags") or []
        )
        # 世界知识：和咖啡屋同一套结构（title / text / requires）。
        # 没有它，NPC 干完活之后就没话可说了 —— 现场会显得空。
        # 更重要的是：知识边界（requires 未满足就不许讲）是**安全机制**，
        # 它必须在两个世界里都成立，否则"不剧透"这条保证只在一个环境里有。
        world = scenario.get("world") or {}
        self.knowledge: dict[str, dict[str, Any]] = {
            str(topic): dict(entry or {})
            for topic, entry in (world.get("knowledge") or {}).items()
        }
        self.knowledge_unlocked: set[str] = set(world.get("knowledge_unlocked") or [])
        # 资源点 id → 该点能采到什么。用于 tool_specs 里给出合法取值，
        # 这是"把合法取值写进描述"那条经验的又一次应用。
        self._pois: dict[str, dict[str, Any]] = dict(
            (scenario.get("pois") or {})
        ) or self._state_pois()
        self._reset_world()

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def _client_scenario(self) -> dict[str, Any]:
        """把场景配置翻译成世界后端能懂的演员表。

        咖啡屋场景用 `npc:` / `npcs:` + `players:` 描述人，
        WorldClient 用一张 `actors:` 表。这一层翻译是适配器的本职：
        **场景格式是给策划写的，世界操作是给机器执行的，两者不必长得一样。**

        翻译成 `actors` 之后，`LocalWorldClient` 与真实桥进程拿到的是
        同一份输入 —— 后者只需要把 id 对上游戏内用户名。
        """
        actors: list[dict[str, Any]] = []
        for spec in self.cast:
            actors.append(
                {
                    "id": spec["id"],
                    "name": spec.get("name", spec["id"]),
                    "kind": "npc",
                    "start": spec.get("start") or DEFAULT_START,
                    "inventory": dict(spec.get("inventory") or {}),
                }
            )
        for spec in self.scenario.get("players") or []:
            actors.append(
                {
                    "id": spec["id"],
                    "name": spec.get("name", spec["id"]),
                    "kind": "player",
                    "start": spec.get("start") or DEFAULT_START,
                    "inventory": dict(spec.get("inventory") or {}),
                }
            )
        out = dict(self.scenario)
        out["actors"] = actors
        return out

    def _state_pois(self) -> dict[str, dict[str, Any]]:
        return dict(self._snapshot().get("pois") or {})

    def _reset_world(self) -> None:
        result = self._call("reset")
        if not result.get("ok"):
            self.events.append(f"世界重置失败：{result.get('reason')}")
        self.tick = 0
        # 重置后 tick 又回到 0，缓存键会撞上重置前那一格 —— 必须显式失效
        self._invalidate_utterances()

    def reset(self) -> dict[str, Any]:
        self.objective_state = {o["id"]: "pending" for o in self.objective_specs}
        self.events = []
        self._reset_world()
        return self.observe(self.npc_id)

    def close(self) -> None:
        """只关掉自己创建的 client —— 外部传进来的由调用方负责。"""
        if self._owns_client:
            self.client.close()

    # ------------------------------------------------------------------ #
    # 与 WorldClient 的薄封装
    # ------------------------------------------------------------------ #
    def _call(self, op: str, **params: Any) -> dict[str, Any]:
        """调用世界操作，并把**传输层故障**与**游戏内失败**分开。

        游戏内失败（挖不到矿）→ 返回 ok=False + 原因，交给 Reflection 学。
        传输层故障（桥进程死了）→ 抛 WorldClientError，立刻炸出来。
        把后者伪装成"NPC 没做到"，会让一次工程事故看起来像一次模型失败 ——
        那是最难查的一类问题。
        """
        try:
            return self.client.call(op, **params).to_dict()
        except WorldClientError:
            raise
        except Exception as exc:  # 世界后端不该把异常漏出来
            return {"ok": False, "reason": f"世界后端异常：{type(exc).__name__}: {exc}", "data": {}}

    def _snapshot(self) -> dict[str, Any]:
        return self.client.state()

    def advance_tick(self) -> None:
        self.tick += 1
        self._call("advance_tick", n=1)

    # ------------------------------------------------------------------ #
    # 观测：翻译成标准 schema
    # ------------------------------------------------------------------ #
    def observe(self, actor_id: str) -> dict[str, Any]:
        """把世界状态翻译成 Agent 期望的观测格式。

        字段名与星屿咖啡屋**完全一致**（tick / self / locations /
        present_actors / visible_actors / visible_items / recent_utterances /
        world_flags / objectives / activities / affinity），
        这是 Agent 一行不改就能跑在这里的原因。

        差异只体现在**内容**上：
        - self.inventory 带数量（"橡木原木×3"），因为这里数量有意义
        - visible_items 是当前位置可采的资源，不是地上散落的物品
        - 多了 time_of_day / day，让"现在是不是夜里"成为一个可推理的事实
        """
        state = self._snapshot()
        actors = state.get("actors") or {}
        me = actors.get(actor_id) or {}
        pois = state.get("pois") or {}
        my_poi = me.get("poi")

        def describe(other: dict[str, Any]) -> dict[str, Any]:
            return {
                "id": other.get("id"),
                "name": other.get("name"),
                "kind": other.get("kind"),
                "loc": other.get("poi"),
                "loc_name": (pois.get(other.get("poi") or "") or {}).get("name", other.get("poi")),
                "here": other.get("poi") == my_poi and other.get("id") != actor_id,
            }

        # 在场（能听见）不等于同位置（能递东西）—— 这条区分在体素世界里
        # 体现为"同一个 POI"，语义与咖啡屋一致。
        present = [
            describe(a) for aid, a in actors.items() if aid != actor_id
        ]
        visible = [spec for spec in present if spec["here"]]

        # 当前位置能采什么。对应咖啡屋的 visible_items（地上散落的物品）。
        resources = (state.get("resources") or {}).get(my_poi or "") or {}
        visible_items = [
            {"id": block, "name": BLOCK_NAMES.get(block, block), "loc": my_poi, "count": count}
            for block, count in resources.items()
            if count > 0
        ]
        visible_items += [
            {
                "id": entry.get("block"),
                "name": BLOCK_NAMES.get(entry.get("block") or "", entry.get("block")),
                "loc": self._poi_of_pos(pois, entry.get("pos")),
                "count": 1,
                "placed": True,
            }
            for entry in (state.get("placed") or [])
            if self._poi_of_pos(pois, entry.get("pos")) == my_poi
        ]

        return {
            "tick": state.get("tick", self.tick),
            "day": state.get("day", 0),
            "time_of_day": state.get("time_of_day", "day"),
            "self": {
                "id": me.get("id", actor_id),
                "name": me.get("name", actor_id),
                "loc": my_poi,
                "loc_name": (pois.get(my_poi or "") or {}).get("name", my_poi),
                "pos": list(me.get("pos") or []),
                # 数量写进字符串：Agent 侧只把背包当成"一串东西"来展示，
                # 不改任何代码就获得了数量信息。
                "inventory": [
                    f"{BLOCK_NAMES.get(item, item)}×{count}"
                    for item, count in sorted((me.get("inventory") or {}).items())
                ],
            },
            "locations": {pid: entry.get("name", pid) for pid, entry in pois.items()},
            "present_actors": present,
            "visible_actors": visible,
            "visible_items": visible_items,
            "recent_utterances": [
                f"{u.get('speaker_name') or u.get('speaker_id')}: {u.get('text')}"
                for u in (state.get("utterances") or [])[-MAX_TRANSCRIPT:]
            ],
            "world_flags": sorted(state.get("flags") or []),
            "objectives": self.objectives_status(),
            "activities": {},
            "affinity": {
                aid: (a.get("affinity") or {}).get(actor_id, 50)
                for aid, a in actors.items()
                if a.get("kind") == "player"
            },
        }

    @staticmethod
    def _poi_of_pos(pois: dict[str, Any], pos: Any) -> str | None:
        for poi_id, entry in pois.items():
            if list(entry.get("pos") or []) == list(pos or []):
                return poi_id
        return None

    # ------------------------------------------------------------------ #
    # 事实
    # ------------------------------------------------------------------ #
    def condition_context(self) -> ConditionContext:
        """给共享判定器提供事实。数量直接给 —— 这正是 player_has_count 的用武之地。"""
        state = self._snapshot()
        actors = state.get("actors") or {}
        utterances = state.get("utterances") or []
        speech: dict[str, int] = {}
        for entry in utterances:
            speaker = entry.get("speaker_id")
            speech[speaker] = speech.get(speaker, 0) + 1
        return ConditionContext(
            flags=set(state.get("flags") or []),
            quantities={
                aid: dict(a.get("inventory") or {}) for aid, a in actors.items()
            },
            speech_counts=speech,
            player_ids=[aid for aid, a in actors.items() if a.get("kind") == "player"],
        )

    def world_facts(self) -> dict[str, Any]:
        """暴露世界规则给离线启发式规划器。

        配方沿用星屿咖啡屋的字段名（name / needs / station），
        只是 needs 是带数量的字典、多了一个 yields。
        规划器用 recipe_needs() 把两种写法归一化，所以它不需要知道
        自己在给哪个世界做计划。

        ⚠️ **`knowledge` 必须一起暴露。** 原来漏了它，后果很具体：
        `village.yaml` 写了四条世界知识（stonemasonry / torch_light /
        cave_danger / cave_secret），`self.knowledge` 也读进来了、
        `available_topics()` 也会算，但 `world_facts()` 不往外给 ——
        于是 `agent._direct_answer()` 拿到的是空表，
        **在体素世界里 NPC 永远答不上任何一个知识问题**，
        全部落到「这个我还没想过，你怎么看？」。而星屿咖啡屋那边
        （`StarIsleEnv.world_facts`）是一直给的。
        「换环境不换行为」是这个世界存在的**理由**，这里少一个键就破了。
        """
        return {
            "locations": {
                pid: entry.get("name", pid) for pid, entry in self._pois.items()
            },
            "recipes": {k: dict(v) for k, v in MC_RECIPES.items()},
            "items": dict(BLOCK_NAMES),
            "knowledge": {k: dict(v) for k, v in self.knowledge.items()},
        }

    def available_topics(self, actor_id: str) -> list[str]:
        """当前可以安全透露的话题。

        语义与星屿咖啡屋完全一致（`requires` 未满足就不许讲），
        因为"不剧透"是**任务设计层面**的保证，不是某个世界的能力 ——
        两个世界共用同一份判定，才谈得上"换环境不换行为"。
        """
        allowed: list[str] = []
        for topic, entry in self.knowledge.items():
            requires = entry.get("requires")
            if requires and requires not in (self._snapshot().get("flags") or []):
                continue
            if not requires and topic not in self.knowledge_unlocked:
                continue
            allowed.append(topic)
        return allowed

    # ------------------------------------------------------------------ #
    # 工具清单
    # ------------------------------------------------------------------ #
    def tool_specs(self, actor_id: str) -> list[ToolSpec]:
        """Minecraft 的动词。

        和咖啡屋一样，**把合法取值直接写进描述** —— 这是最省事也最有效的
        防幻觉手段。模型看到 move_to 的例子是 move_to(forest)，
        就不会去猜一个不存在的地点。
        """
        poi_ids = list(self._pois)
        poi_hint = "、".join(poi_ids) or "（本场景没有地点）"
        block_ids = list(BLOCK_NAMES)
        block_hint = "、".join(block_ids)
        recipe_ids = list(MC_RECIPES)
        recipe_hint = "、".join(recipe_ids) or "（本场景没有配方）"
        topic_ids = list(self.knowledge)
        topic_hint = "、".join(topic_ids) or "（本场景没有话题）"
        player_ids = [a for a in self._player_ids()]
        player_hint = "、".join(player_ids) or "（本场景没有玩家）"
        station_ids = sorted({str(r.get("station")) for r in MC_RECIPES.values() if r.get("station")})
        station_hint = "、".join(station_ids) or "（没有合成台）"

        return [
            ToolSpec(
                "move_to",
                f"走到一个地点。地点只能是 {poi_hint}",
                {"location": f"地点 id，只能是 {poi_hint}"},
                [f"move_to({poi_ids[0]})"] if poi_ids else ["move_to(?)"],
            ),
            ToolSpec(
                "mine",
                "采集当前地点的一个方块。必须人在对应地点，且夜里需要附近有火把",
                {"block": f"方块 id，只能是 {block_hint}"},
                ["mine(oak_log)"],
            ),
            ToolSpec(
                "craft",
                f"在合成台合成物品。配方只能是 {recipe_hint}；工作台是 {station_hint}",
                {"item": f"配方 id，只能是 {recipe_hint}"},
                [f"craft({recipe_ids[0]})"] if recipe_ids else ["craft(?)"],
            ),
            ToolSpec(
                "place",
                "把背包里的方块放到当前地点（例如把火把插在洞口）",
                {"block": f"方块 id，只能是 {block_hint}", "target": f"地点 id，只能是 {poi_hint}"},
                [f"place(torch, {poi_ids[-1]})"] if poi_ids else ["place(torch, ?)"],
            ),
            ToolSpec(
                "transfer",
                f"把物品交给同一地点的玩家。玩家 id 只能是 {player_hint}",
                {"item": "物品 id", "player": f"玩家 id，只能是 {player_hint}"},
                [f"transfer(torch, {player_ids[0]})"] if player_ids else ["transfer(torch, ?)"],
            ),
            ToolSpec("consume", "消耗背包里的物品", {"item": "物品 id", "count": "数量"}, ["consume(coal, 1)"]),
            ToolSpec(
                "tell_fact",
                f"按知识边界透露一个话题，没事做时用它起个话头。话题只能是 {topic_hint}",
                {"topic": f"话题 id，只能是 {topic_hint}"},
                [f"tell_fact({topic_ids[0]})"] if topic_ids else ["tell_fact(?)"],
            ),
            ToolSpec("set_flag", "设置任务标记（仅限白名单）", {"key": "标记名", "value": "值"}, ["set_flag(cave_lit, 1)"]),
            ToolSpec("wait", "本回合不做任何世界动作（例如等天亮）", {}, ["wait()"]),
        ]

    def _player_ids(self) -> list[str]:
        actors = self._snapshot().get("actors") or {}
        return [aid for aid, a in actors.items() if a.get("kind") == "player"]

    # ------------------------------------------------------------------ #
    # 执行
    # ------------------------------------------------------------------ #
    def dispatch(self, actor_id: str, call: ActionCall) -> ActionResult:
        if actor_id not in (self._snapshot().get("actors") or {}):
            return ActionResult(False, call.tool, f"未知角色: {actor_id}")
        handler = getattr(self, f"_h_{call.tool}", None)
        if handler is None:
            return ActionResult(
                False, call.tool, f"没有名为 {call.tool} 的工具。可用工具见工具清单。"
            )
        try:
            return handler(actor_id, call.args or {})
        except WorldClientError:
            raise
        except Exception as exc:  # 兜底：环境永远不抛异常出去
            return ActionResult(False, call.tool, f"执行异常: {type(exc).__name__}: {exc}")

    def _ok(self, tool: str, detail: str = "", **delta: Any) -> ActionResult:
        if detail:
            self.events.append(detail)
        return ActionResult(True, tool, detail, delta)

    def _fail(self, tool: str, detail: str) -> ActionResult:
        return ActionResult(False, tool, detail)

    def _from_op(self, tool: str, result: dict[str, Any], detail: str = "") -> ActionResult:
        """把世界操作的结果翻成 ActionResult。

        `detail` 优先用世界给的原因 —— 那句话是为 NPC 写的
        （"天黑了，看不清矿脉"），比适配器能编出来的任何话都有用。
        """
        if result.get("ok"):
            return self._ok(tool, detail, **(result.get("data") or {}))
        return self._fail(tool, str(result.get("reason") or "世界拒绝了这次操作"))

    # ------------------------------------------------------------------ #
    # 各工具的护栏实现
    # ------------------------------------------------------------------ #
    def _h_move_to(self, actor_id: str, args: dict[str, Any]) -> ActionResult:
        location = str(args.get("location", "")).strip()
        result = self._call("move", actor=actor_id, target=location)
        if not result.get("ok"):
            return self._from_op("move_to", result)
        name = (self._pois.get(location) or {}).get("name", location)
        return self._from_op("move_to", result, f"移动到{name}")

    def _h_mine(self, actor_id: str, args: dict[str, Any]) -> ActionResult:
        block = str(args.get("block", "")).strip()
        result = self._call("mine", actor=actor_id, block=block)
        if not result.get("ok"):
            return self._from_op("mine", result)
        label = BLOCK_NAMES.get(block, block)
        return self._from_op("mine", result, f"采到一个{label}")

    def _h_craft(self, actor_id: str, args: dict[str, Any]) -> ActionResult:
        item = str(args.get("item", "")).strip()
        result = self._call("craft", actor=actor_id, item=item)
        if not result.get("ok"):
            return self._from_op("craft", result)
        label = (MC_RECIPES.get(item) or {}).get("name", item)
        return self._from_op("craft", result, f"合成出{label}")

    def _h_place(self, actor_id: str, args: dict[str, Any]) -> ActionResult:
        block = str(args.get("block", "")).strip()
        target = str(args.get("target", "")).strip()
        result = self._call("place", actor=actor_id, block=block, target=target)
        if not result.get("ok"):
            return self._from_op("place", result)
        label = BLOCK_NAMES.get(block, block)
        where = (self._pois.get(target) or {}).get("name", target)
        return self._from_op("place", result, f"把{label}放在{where}")

    def _h_transfer(self, actor_id: str, args: dict[str, Any]) -> ActionResult:
        item = str(args.get("item", "")).strip()
        player = str(args.get("player", "")).strip()
        count = int(args.get("count") or 1)
        result = self._call(
            "transfer", src=actor_id, dst=player, item=item, count=count
        )
        if not result.get("ok"):
            return self._from_op("transfer", result)
        label = BLOCK_NAMES.get(item, item)
        return self._from_op("transfer", result, f"把{label}交给了{player}")

    def _h_consume(self, actor_id: str, args: dict[str, Any]) -> ActionResult:
        item = str(args.get("item", "")).strip()
        count = int(args.get("count") or 1)
        result = self._call("consume", actor=actor_id, item=item, count=count)
        return self._from_op("consume", result)

    def _h_tell_fact(self, actor_id: str, args: dict[str, Any]) -> ActionResult:
        """透露一个话题。语义与咖啡屋逐字一致 —— 包括拒绝时的理由。

        "不剧透"是任务设计的保证，不该因为换了世界就失效。
        """
        topic = str(args.get("topic", "")).strip()
        entry = self.knowledge.get(topic)
        if entry is None:
            known = "、".join(self.knowledge) or "（本场景没有话题）"
            return self._fail("tell_fact", f"你不知道「{topic}」这个话题。能聊的：{known}")
        requires = entry.get("requires")
        if requires and requires not in (self._snapshot().get("flags") or []):
            return self._fail(
                "tell_fact",
                f"「{entry.get('title', topic)}」现在还不能讲 —— 需要先满足条件 {requires}，否则会剧透",
            )
        if topic not in self.knowledge_unlocked and not requires:
            return self._fail(
                "tell_fact", f"关于「{entry.get('title', topic)}」你今天还没打算聊"
            )
        text = entry.get("text", "")
        return self._ok("tell_fact", text, topic=topic, text=text)

    def _h_set_flag(self, actor_id: str, args: dict[str, Any]) -> ActionResult:
        key = str(args.get("key", "")).strip()
        if key not in self.settable_flags:
            allowed = "、".join(sorted(self.settable_flags)) or "（本场景不允许设置标记）"
            return self._fail("set_flag", f"不允许设置标记「{key}」。允许的标记：{allowed}")
        result = self._call("set_flag", flag=key)
        return self._from_op("set_flag", result, f"设置标记 {key}")

    def _h_wait(self, actor_id: str, args: dict[str, Any]) -> ActionResult:
        """等一回合。夜里没光源时，这是唯一有意义的动作。"""
        return self._ok("wait", "原地等了一会儿")

    # ------------------------------------------------------------------ #
    # 其他 Environment 接口
    # ------------------------------------------------------------------ #
    def broadcast(self, actor_id: str, text: str) -> None:
        """发言 = 在世界里发一条聊天。

        这样"谁说过什么"只有**一个**事实来源（世界后端），
        不需要在适配器里再维护一份平行日志 —— 两份日志迟早会不一致，
        而不一致的表现是"导演以为没人开口，于是两个 NPC 抢话"。
        """
        self._call("chat", actor=actor_id, text=text)
        self._invalidate_utterances()

    @property
    def utterances(self) -> list[Utterance]:
        """从世界后端的聊天记录派生出发言日志。

        导演每轮会读它 2～3 次（判断有没有人开口、取本轮新增的发言）。
        每次都往后端要一次完整状态，在真实桥上是三次往返 —— 所以按 tick 缓存。
        缓存只在会改变聊天记录的地方失效：broadcast / 注入玩家发言 / 重置 / 推进时间。
        """
        cached = getattr(self, "_utterance_cache", None)
        tick = self.tick
        if cached is not None and cached[0] == tick:
            return cached[1]
        state = self._snapshot()
        kinds = {
            aid: a.get("kind") for aid, a in (state.get("actors") or {}).items()
        }
        out = [
            Utterance(
                speaker_id=entry.get("speaker_id") or "",
                speaker_name=entry.get("speaker_name") or entry.get("speaker_id") or "",
                text=entry.get("text") or "",
                tick=int(entry.get("tick") or 0),
                role="npc" if kinds.get(entry.get("speaker_id")) == "npc" else "player",
                is_question=looks_like_question(entry.get("text") or ""),
            )
            for entry in (state.get("utterances") or [])
        ]
        self._utterance_cache = (tick, out)
        return out

    def _invalidate_utterances(self) -> None:
        self._utterance_cache = None

    def record_player_utterance(self, player_id: str, text: str) -> Utterance:
        """玩家的发言也要进世界 —— 否则 NPC 在游戏里"听不见"玩家说话。"""
        self._call("chat", actor=player_id, text=text)
        self._invalidate_utterances()
        log = self.utterances
        return log[-1] if log else super().record_player_utterance(player_id, text)

    def speaker_name(self, actor_id: str) -> str:
        actors = self._snapshot().get("actors") or {}
        return (actors.get(actor_id) or {}).get("name") or actor_id

    def snapshot(self) -> dict[str, Any]:
        """世界快照。

        键名与 StarIsleEnv 对齐（`world_flags` / `loc`），
        因为评测 harness 的断言是**跨环境共用**的 —— 它读 `snapshot["world_flags"]`，
        两个世界就必须都叫这个名字。这里曾经叫 `flags`，
        结果是体素世界的用例里"未达成标记"永远为真。
        """
        state = self._snapshot()
        actors = state.get("actors") or {}
        return {
            "env": self.name,
            "tick": state.get("tick", self.tick),
            "time_of_day": state.get("time_of_day"),
            "day": state.get("day"),
            "scenario": self.scenario.get("id"),
            "actors": {
                aid: {
                    "name": a.get("name"),
                    "kind": a.get("kind"),
                    "loc": a.get("poi"),
                    "inventory": dict(a.get("inventory") or {}),
                    "affinity": dict(a.get("affinity") or {}),
                }
                for aid, a in actors.items()
            },
            "placed": state.get("placed"),
            "resources": state.get("resources"),
            # 地点表进快照：评测要断言"火把插在洞口"，而不只是"插了火把"
            "pois": state.get("pois"),
            "world_flags": list(state.get("flags") or []),
            "objectives": dict(self.objective_state),
            "utterance_count": len(self.utterances),
        }
