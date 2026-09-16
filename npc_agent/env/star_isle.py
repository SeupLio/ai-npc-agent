"""星屿咖啡屋 —— 内置的确定性文字世界。

选它作为默认环境的原因：
1. **确定性**：同样的输入得到同样的世界状态，评测才能出可复现的数字
2. **够复杂**：有位置图、物品流转、配方合成、任务标记、知识边界、好感度
3. **零依赖**：不需要起 Minecraft 服务端，clone 下来就能跑

世界的规则全部体现在 dispatch() 的护栏里。护栏失败会返回结构化原因，
这个原因就是 Reflection 模块的输入 —— 这正是"看起来会说、实际上不会玩"
那个失败模式的修复机制。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..types import ActionCall, ActionResult, Utterance
from .base import Environment, ToolSpec

# --------------------------------------------------------------------------- #
# 世界常量（策划配置的等价物）
# --------------------------------------------------------------------------- #
LOCATIONS: dict[str, str] = {
    "counter": "吧台",
    "window": "窗边座",
    "shelf": "书架角",
    "kitchen": "后厨",
    "terrace": "露台",
    "door": "门口",
}

ITEM_NAMES: dict[str, str] = {
    "beans": "咖啡豆",
    "milk": "牛奶",
    "lemon": "柠檬",
    "apple": "苹果",
    "latte": "拿铁",
    "lemonade": "柠檬水",
    "apple_pie": "苹果派",
    "old_book": "旧书",
    "gramophone": "留声机",
}

# 配方：需要哪些材料、在哪个工位、产出什么
RECIPES: dict[str, dict[str, Any]] = {
    "latte": {"name": "拿铁", "needs": ["beans", "milk"], "station": "kitchen"},
    "lemonade": {"name": "柠檬水", "needs": ["lemon"], "station": "counter"},
    "apple_pie": {"name": "苹果派", "needs": ["apple"], "station": "kitchen"},
}

# 知识库。requires 不为空时，需要对应世界标记才允许透露 —— 这是"不剧透"的实现。
KNOWLEDGE: dict[str, dict[str, Any]] = {
    "house_story": {
        "title": "咖啡屋的故事",
        "text": "这家店开在星屿的旧灯塔下面，最早是个给守塔人歇脚的地方。",
        "requires": None,
    },
    "brewing": {
        "title": "手冲的门道",
        "text": "水温低一点，闷蒸久一点，酸味会更干净。",
        "requires": None,
    },
    "constellation": {
        "title": "露台的星空",
        "text": "露台朝北，天晴的时候能看到很清楚的星轨。",
        "requires": None,
    },
    "hidden_menu": {
        "title": "隐藏菜单",
        "text": "其实还有一杯不写在菜单上的特调，叫「灯塔余晖」，只有熟客知道。",
        "requires": "hidden_menu_unlocked",  # ← 需要解锁，防止 NPC 一上来就剧透
    },
}

DEFAULT_START = "counter"
MAX_TRANSCRIPT = 40  # 观测窗口只保留最近 N 条发言


# --------------------------------------------------------------------------- #
# 世界内的实体
# --------------------------------------------------------------------------- #
@dataclass
class Actor:
    id: str
    name: str
    kind: str  # npc | player
    loc: str
    inventory: list[str] = field(default_factory=list)
    affinity: dict[str, int] = field(default_factory=dict)  # 对其他角色的好感度 0-100
    flags: set[str] = field(default_factory=set)
    seated: bool = False

    def affinity_to(self, other_id: str) -> int:
        return self.affinity.get(other_id, 50)

    def bump_affinity(self, other_id: str, delta: int) -> None:
        current = self.affinity.get(other_id, 50)
        self.affinity[other_id] = max(0, min(100, current + delta))


@dataclass
class Item:
    id: str
    loc: str | None = None
    holder: str | None = None

    @property
    def display(self) -> str:
        return ITEM_NAMES.get(self.id, self.id)


# --------------------------------------------------------------------------- #
# 环境实现
# --------------------------------------------------------------------------- #
class StarIsleEnv(Environment):
    name = "star-isle"

    def __init__(self, scenario: dict[str, Any], npc_id: str, npc_name: str) -> None:
        self.scenario = scenario
        self.npc_id = npc_id
        self.npc_name = npc_name
        self._handlers: dict[str, Callable[[str, dict[str, Any]], ActionResult]] = {
            "move_to": self._h_move_to,
            "take_item": self._h_take_item,
            "craft_item": self._h_craft_item,
            "give_item": self._h_give_item,
            "emote": self._h_emote,
            "tell_fact": self._h_tell_fact,
            "start_activity": self._h_start_activity,
            "judge_answer": self._h_judge_answer,
            "set_flag": self._h_set_flag,
            "sit_down": self._h_sit_down,
            "wait": self._h_wait,
        }
        self.reset()

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def reset(self) -> dict[str, Any]:
        cfg = self.scenario
        self.tick = 0
        self.utterances: list[Utterance] = []
        self.events: list[str] = []

        self.actors: dict[str, Actor] = {
            self.npc_id: Actor(self.npc_id, self.npc_name, "npc", cfg.get("npc_start", DEFAULT_START))
        }
        for spec in cfg.get("players", []):
            self.actors[spec["id"]] = Actor(
                spec["id"], spec["name"], "player", spec.get("start", "door")
            )

        self.items: dict[str, Item] = {}
        for item_id, spec in (cfg.get("world", {}).get("items") or {}).items():
            spec = spec or {}
            self.items[item_id] = Item(item_id, loc=spec.get("loc"), holder=spec.get("holder"))

        world = cfg.get("world", {})
        self.world_flags: set[str] = set(world.get("flags") or [])
        self.knowledge_unlocked: set[str] = set(world.get("knowledge_unlocked") or [])
        self.settable_flags: set[str] = set(world.get("settable_flags") or [])
        self.activities: dict[str, str] = {
            a["id"]: a["name"] for a in (cfg.get("activities") or [])
        }
        self.objective_state: dict[str, str] = {
            o["id"]: "pending" for o in (cfg.get("objectives") or [])
        }
        self.objective_specs: list[dict[str, Any]] = list(cfg.get("objectives") or [])
        return self.observe(self.npc_id)

    def advance_tick(self) -> None:
        self.tick += 1

    # ------------------------------------------------------------------ #
    # 观测
    # ------------------------------------------------------------------ #
    def observe(self, actor_id: str) -> dict[str, Any]:
        actor = self.actors[actor_id]

        def describe(other: Actor) -> dict[str, Any]:
            return {
                "id": other.id,
                "name": other.name,
                "kind": other.kind,
                "loc": other.loc,
                "loc_name": LOCATIONS.get(other.loc, other.loc),
                "here": other.loc == actor.loc,
            }

        # 关键区分：**在场**（能听见、能对话）不等于 **同位置**（能递东西）。
        # 站在吧台能听见门口客人说话，但递不过去一杯咖啡。
        present = [describe(a) for a in self.actors.values() if a.id != actor_id]
        visible = [spec for spec in present if spec["here"]]

        visible_items = [
            {"id": i.id, "name": i.display, "loc": i.loc}
            for i in self.items.values()
            if i.loc == actor.loc and i.holder is None
        ]
        return {
            "tick": self.tick,
            "self": {
                "id": actor.id,
                "name": actor.name,
                "loc": actor.loc,
                "loc_name": LOCATIONS.get(actor.loc, actor.loc),
                "inventory": [ITEM_NAMES.get(i, i) for i in actor.inventory],
            },
            "locations": dict(LOCATIONS),
            "present_actors": present,
            "visible_actors": visible,
            "visible_items": visible_items,
            "recent_utterances": [u.render() for u in self.utterances[-MAX_TRANSCRIPT:]],
            "world_flags": sorted(self.world_flags),
            "objectives": self.objectives_status(),
            "activities": dict(self.activities),
            "affinity": {
                pid: actor.affinity_to(pid)
                for pid, a in self.actors.items()
                if a.kind == "player"
            },
        }

    def world_facts(self) -> dict[str, Any]:
        """把世界规则暴露给 Planner 的离线启发式路径。"""
        return {
            "locations": dict(LOCATIONS),
            "recipes": {k: dict(v) for k, v in RECIPES.items()},
            "items": dict(ITEM_NAMES),
        }

    def available_topics(self, actor_id: str) -> list[str]:
        """过滤掉未解锁的话题，避免 NPC 反复尝试说一个它其实不该说的东西。"""
        allowed = []
        for topic, entry in KNOWLEDGE.items():
            requires = entry.get("requires")
            if requires and requires not in self.world_flags:
                continue
            if not requires and topic not in self.knowledge_unlocked:
                continue
            allowed.append(topic)
        return allowed

    def objectives_status(self) -> dict[str, str]:
        """根据 success_when 条件刷新目标完成情况。

        支持多种条件类型，因为"目标完成"未必等于"某个标记被设置"：
            {flag: X}                     某个世界标记被设置
            {all_players_spoke: N}        每位玩家都至少说过 N 次话
            {player_has: {pid: [item]}}   某位玩家的背包里真的出现了某样东西

        `player_has` 是最重要的一种：它把"目标完成"定义在**世界状态**上，
        而不是"NPC 执行完了自己的步骤"。这样即使玩家自己拿到了那杯咖啡
        （比如另一个流程给的），目标也会正确地判定为完成。
        """
        for spec in self.objective_specs:
            condition = spec.get("success_when") or {}
            if self._condition_met(condition):
                self.objective_state[spec["id"]] = "done"
        return dict(self.objective_state)

    def _condition_met(self, condition: dict[str, Any]) -> bool:
        if not condition:
            return False
        if "flag" in condition:
            return condition["flag"] in self.world_flags
        if "all_players_spoke" in condition:
            needed = int(condition["all_players_spoke"])
            players = [a for a in self.actors.values() if a.kind == "player"]
            if not players:
                return False
            return all(
                sum(1 for u in self.utterances if u.speaker_id == p.id) >= needed
                for p in players
            )
        if "player_has" in condition:
            for player_id, items in (condition["player_has"] or {}).items():
                actor = self.actors.get(player_id)
                if actor is None:
                    return False
                if any(item not in actor.inventory for item in items):
                    return False
            return True
        return False

    def snapshot(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "scenario": self.scenario.get("id"),
            "actors": {
                a.id: {
                    "name": a.name,
                    "kind": a.kind,
                    "loc": a.loc,
                    "inventory": list(a.inventory),
                    "affinity": dict(a.affinity),
                    "seated": a.seated,
                    "flags": sorted(a.flags),
                }
                for a in self.actors.values()
            },
            "items": {
                i.id: {"loc": i.loc, "holder": i.holder} for i in self.items.values()
            },
            "world_flags": sorted(self.world_flags),
            "objectives": dict(self.objective_state),
            "utterance_count": len(self.utterances),
            "events": list(self.events[-20:]),
        }

    # ------------------------------------------------------------------ #
    # 发言广播
    # ------------------------------------------------------------------ #
    def broadcast(self, actor_id: str, text: str) -> None:
        actor = self.actors[actor_id]
        self.utterances.append(
            Utterance(
                speaker_id=actor.id,
                speaker_name=actor.name,
                text=text,
                tick=self.tick,
                role="npc" if actor.kind == "npc" else "player",
                is_question=text.rstrip().endswith(("?", "？")),
            )
        )

    def record_player_utterance(self, player_id: str, text: str) -> Utterance:
        """外部驱动（CLI / 评测用例）注入玩家发言。"""
        actor = self.actors[player_id]
        mentioned = [a.id for a in self.actors.values() if a.name in text and a.id != player_id]
        utterance = Utterance(
            speaker_id=actor.id,
            speaker_name=actor.name,
            text=text,
            tick=self.tick,
            role="player",
            mentions=mentioned,
            is_question=text.rstrip().endswith(("?", "？")),
        )
        self.utterances.append(utterance)
        return utterance

    # ------------------------------------------------------------------ #
    # 工具清单
    # ------------------------------------------------------------------ #
    def tool_specs(self, actor_id: str) -> list[ToolSpec]:
        specs = [
            ToolSpec("move_to", "移动到某个位置", {"location": "位置 id"}, ["move_to(counter)"]),
            ToolSpec("take_item", "拿起当前所在位置的物品", {"item": "物品 id"}, ["take_item(beans)"]),
            ToolSpec("craft_item", "在正确工位上用材料制作饮品", {"recipe": "配方 id"}, ["craft_item(latte)"]),
            ToolSpec("give_item", "把手里的物品交给同一位置的玩家", {"item": "物品 id", "player": "玩家 id"}, ["give_item(latte, player_a)"]),
            ToolSpec("sit_down", "坐下，用于引导玩家入座", {}, ["sit_down()"]),
            ToolSpec("emote", "做一个动作表情", {"name": "动作名"}, ["emote(微笑)"]),
            ToolSpec("tell_fact", "按知识边界透露一个话题（越界会被拒绝）", {"topic": "话题 id"}, ["tell_fact(brewing)"]),
            ToolSpec("start_activity", "开启一个店内活动", {"activity": "活动 id"}, ["start_activity(star_quiz)"]),
            ToolSpec("judge_answer", "判定玩家在活动中的回答是否正确", {"player": "玩家 id", "correct": "true/false"}, ["judge_answer(player_a, true)"]),
            ToolSpec("set_flag", "设置任务标记（仅限白名单，防止乱改状态）", {"key": "标记名", "value": "值"}, ["set_flag(topic_found, 1)"]),
            ToolSpec("wait", "本回合不做任何世界动作", {}, ["wait()"]),
        ]
        return specs

    # ------------------------------------------------------------------ #
    # 统一执行入口
    # ------------------------------------------------------------------ #
    def dispatch(self, actor_id: str, call: ActionCall) -> ActionResult:
        if actor_id not in self.actors:
            return ActionResult(False, call.tool, f"未知角色: {actor_id}")
        handler = self._handlers.get(call.tool)
        if handler is None:
            return ActionResult(
                False,
                call.tool,
                f"没有名为 {call.tool} 的工具。可用工具见工具清单。",
            )
        try:
            return handler(actor_id, call.args or {})
        except Exception as exc:  # 兜底：环境永远不抛异常出去
            return ActionResult(False, call.tool, f"执行异常: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    # 辅助
    # ------------------------------------------------------------------ #
    def _ok(self, tool: str, detail: str = "", **delta: Any) -> ActionResult:
        if detail:
            self.events.append(detail)
        return ActionResult(True, tool, detail, delta)

    def _fail(self, tool: str, detail: str) -> ActionResult:
        return ActionResult(False, tool, detail)

    def _item_label(self, item_id: str) -> str:
        return ITEM_NAMES.get(item_id, item_id)

    def _loc_label(self, loc_id: str) -> str:
        return LOCATIONS.get(loc_id, loc_id)

    def _player(self, player_id: str) -> Actor | None:
        actor = self.actors.get(player_id)
        return actor if actor and actor.kind == "player" else None

    # ------------------------------------------------------------------ #
    # 各工具的护栏实现
    # ------------------------------------------------------------------ #
    def _h_move_to(self, actor_id: str, args: dict[str, Any]) -> ActionResult:
        location = str(args.get("location", "")).strip()
        if location not in LOCATIONS:
            return self._fail("move_to", f"没有叫「{location}」的位置。可选: {', '.join(LOCATIONS)}")
        actor = self.actors[actor_id]
        if actor.loc == location:
            return self._ok("move_to", f"已经在{self._loc_label(location)}了")
        actor.loc = location
        return self._ok("move_to", f"移动到{self._loc_label(location)}", loc=location)

    def _h_take_item(self, actor_id: str, args: dict[str, Any]) -> ActionResult:
        item_id = str(args.get("item", "")).strip()
        item = self.items.get(item_id)
        if item is None:
            return self._fail("take_item", f"世界里没有「{item_id}」这个物品")
        actor = self.actors[actor_id]
        if item.holder == actor_id:
            return self._ok("take_item", f"{item.display}已经拿在手上了", item=item_id)
        if item.holder is not None:
            return self._fail("take_item", f"{item.display}已经在{self.actors[item.holder].name}手里了")
        if item.loc != actor.loc:
            return self._fail(
                "take_item",
                f"{item.display}在{self._loc_label(item.loc or '')}，你现在在{self._loc_label(actor.loc)}，需要先 move_to 过去",
            )
        item.loc = None
        item.holder = actor_id
        actor.inventory.append(item_id)
        return self._ok("take_item", f"拿起{item.display}", item=item_id)

    def _h_craft_item(self, actor_id: str, args: dict[str, Any]) -> ActionResult:
        recipe_id = str(args.get("recipe", "")).strip()
        recipe = RECIPES.get(recipe_id)
        if recipe is None:
            return self._fail("craft_item", f"没有「{recipe_id}」这个配方。可选: {', '.join(RECIPES)}")
        actor = self.actors[actor_id]
        if actor.loc != recipe["station"]:
            return self._fail(
                "craft_item",
                f"做{recipe['name']}需要待在{self._loc_label(recipe['station'])}，你现在在{self._loc_label(actor.loc)}",
            )
        missing = [n for n in recipe["needs"] if n not in actor.inventory]
        if missing:
            names = "、".join(self._item_label(m) for m in missing)
            return self._fail("craft_item", f"材料不够，还缺{names}")
        for need in recipe["needs"]:
            actor.inventory.remove(need)
            self.items.pop(need, None)
        actor.inventory.append(recipe_id)
        self.items[recipe_id] = Item(recipe_id, loc=None, holder=actor_id)
        return self._ok("craft_item", f"做好了{recipe['name']}", item=recipe_id)

    def _h_give_item(self, actor_id: str, args: dict[str, Any]) -> ActionResult:
        item_id = str(args.get("item", "")).strip()
        player_id = str(args.get("player", "")).strip()
        actor = self.actors[actor_id]
        if item_id not in actor.inventory:
            return self._fail("give_item", f"你手里没有{self._item_label(item_id)}")
        player = self._player(player_id)
        if player is None:
            return self._fail("give_item", f"找不到玩家 {player_id}")
        if player.loc != actor.loc:
            return self._fail(
                "give_item",
                f"{player.name}在{self._loc_label(player.loc)}，不在你身边，递不过去",
            )
        actor.inventory.remove(item_id)
        player.inventory.append(item_id)
        item = self.items.setdefault(item_id, Item(item_id))
        item.holder = player_id
        item.loc = None
        actor.bump_affinity(player_id, 4)
        return self._ok(
            "give_item", f"把{self._item_label(item_id)}递给{player.name}", item=item_id, to=player_id
        )

    def _h_sit_down(self, actor_id: str, args: dict[str, Any]) -> ActionResult:
        actor = self.actors[actor_id]
        actor.seated = True
        return self._ok("sit_down", f"{actor.name}坐下了", seated=True)

    def _h_emote(self, actor_id: str, args: dict[str, Any]) -> ActionResult:
        name = str(args.get("name", "点头")).strip() or "点头"
        return self._ok("emote", f"{self.actors[actor_id].name}{name}")

    def _h_tell_fact(self, actor_id: str, args: dict[str, Any]) -> ActionResult:
        topic = str(args.get("topic", "")).strip()
        entry = KNOWLEDGE.get(topic)
        if entry is None:
            return self._fail("tell_fact", f"你不知道「{topic}」这个话题")
        requires = entry.get("requires")
        if requires and requires not in self.world_flags:
            return self._fail(
                "tell_fact",
                f"「{entry['title']}」现在还不能讲 —— 需要先满足条件 {requires}，否则会剧透",
            )
        if topic not in self.knowledge_unlocked and not requires:
            return self._fail("tell_fact", f"关于「{entry['title']}」你今天还没打算聊")
        return self._ok("tell_fact", entry["text"], topic=topic, text=entry["text"])

    def _h_start_activity(self, actor_id: str, args: dict[str, Any]) -> ActionResult:
        activity = str(args.get("activity", "")).strip()
        if activity not in self.activities:
            return self._fail(
                "start_activity",
                f"没有「{activity}」这个活动。可选: {', '.join(self.activities) or '（本场景无活动）'}",
            )
        self.world_flags.add("activity_started")
        return self._ok(
            "start_activity", f"开启了活动：{self.activities[activity]}", activity=activity
        )

    def _h_judge_answer(self, actor_id: str, args: dict[str, Any]) -> ActionResult:
        player = self._player(str(args.get("player", "")).strip())
        if player is None:
            return self._fail("judge_answer", "找不到要判定的玩家")
        correct = args.get("correct")
        if isinstance(correct, str):
            correct = correct.strip().lower() in ("1", "true", "yes", "对", "正确")
        actor = self.actors[actor_id]
        if correct:
            actor.bump_affinity(player.id, 6)
            player.flags.add("answered_correctly")
            return self._ok("judge_answer", f"{player.name}答对了", correct=True)
        actor.bump_affinity(player.id, 1)
        return self._ok("judge_answer", f"{player.name}答错了，给点提示", correct=False)

    def _h_set_flag(self, actor_id: str, args: dict[str, Any]) -> ActionResult:
        key = str(args.get("key", "")).strip()
        value = args.get("value", "1")
        if not key:
            return self._fail("set_flag", "缺少 key")
        if key not in self.settable_flags:
            return self._fail(
                "set_flag",
                f"「{key}」不在本场景允许 NPC 修改的标记白名单内（防止智能体越权改世界状态）",
            )
        truthy = str(value).strip().lower() not in ("0", "false", "no", "")
        if truthy:
            self.world_flags.add(key)
        else:
            self.world_flags.discard(key)
        self.objectives_status()
        return self._ok("set_flag", f"标记 {key} = {truthy}", key=key, value=truthy)

    def _h_wait(self, actor_id: str, args: dict[str, Any]) -> ActionResult:
        return self._ok("wait", "静观其变")
