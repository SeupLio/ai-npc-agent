"""模块六：Planning —— 任务分解与重规划。

两条路径：
    有模型   → 把工具清单 + 现场状态 + 记忆喂给模型，让它输出 JSON 计划
    没模型   → 走场景配置里的目标步骤（objectives），以及玩家请求的意图模板

重规划（replan）是这里最有价值的部分：当某一步失败时，Planner 会读失败原因，
**插入纠错步骤**再重试，而不是原地重试同一步。

    例：take_item(beans) 失败，原因是"咖啡豆在后厨，你现在在吧台"
        → 插入 move_to(kitchen)，然后重试 take_item(beans)

这就是"看起来会说、实际上不会玩"的修复机制。
"""

from __future__ import annotations

import re
from typing import Any, Optional

from ..llm.base import LLM, LLMUnavailable
from ..types import Plan, PlanStep, Utterance
from .memory import MemoryRecord
from .persona import Persona
from .state import StateTracker

# 玩家点单的意图识别（离线路径）
#
# ⚠️ **这张表和 `env/star_isle.RECIPES` 必须点名同一批东西。**
# 认单靠这张表，造步骤靠 `world_facts()["recipes"]` —— **两张表**，
# 各自演进就会漂移，而漂移时两边都不报错：配方表多一项 ⇒ 玩家点得到的东西
# NPC 说「做不了」；识别表多一项 ⇒ NPC 接下了一个做不出来的单。
# `tests/test_planner.py::test_the_menu_has_exactly_one_source_of_truth` 钉住它。
#
# `手冲` 必须排在 `拿铁` **前面**：`拿铁` 那条含 `咖啡`，
# 而「手冲咖啡」两边的模式都命中，顺序决定谁赢。
ORDER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(手冲|pour\s*over|pour-over)", re.I), "pour_over"),
    (re.compile(r"(拿铁|咖啡|latte)", re.I), "latte"),
    (re.compile(r"(柠檬水|柠檬|lemonade)", re.I), "lemonade"),
    (re.compile(r"(苹果派|苹果|apple\s*pie)", re.I), "apple_pie"),
]

# 只有出现"请求类"措辞时才算点单。
# 否则"我特别喜欢偏酸的咖啡"会被误判成下单，NPC 就会莫名其妙去做一杯咖啡。
REQUEST_MARKERS = re.compile(
    r"(来一?[杯份个]|要一?[杯份个]|给我|帮我|麻烦|能不能给|可以给|想喝|想点|点一?[杯份个]|上一?[杯份个])"
)

# --------------------------------------------------------------------------- #
# 完成条件的**人话渲染** —— 给规划 prompt 用
# --------------------------------------------------------------------------- #
#
# 为什么需要它：`plan_with_llm` 原来只把目标的 **goal 文本**喂给模型，
# **不给 `success_when`**。于是模型规划出"听起来完成了目标"的动作，
# 但那个动作**不满足机器判定的完成条件**。
#
# 实测（duet，模型规划，4 次里 3 次失败）：
#   小舟的目标 `play_song`，完成条件是 `{flag: song_started}`。
#   模型规划出来的是 `speak → emote(play_guitar) → start_activity(song_request)`
#   —— 全都是"像在起歌"的动作，**唯独没有人去 set_flag(song_started)**。
#   于是 `play_song` 永远 pending，依赖它的联合目标 `terrace_night` 也跟着挂住。
#
# 这不是"模型不会规划"，是**信息不对称** —— 和之前"规划 prompt 里没有配方表，
# 所以模型想不到先 take_item"是同一类病：**启发式规划器读得到那份条件，
# 模型读不到**。修法也一样：把判据本身告诉它。
#
# ⚠️ 只给**判据**和**对应的工具**，不给步骤顺序 —— 顺序仍然由模型自己排。
# 一旦把 `steps` 也抄进 prompt，LLM 规划就退化成"照着剧本念"，
# 那正是这个项目要证明它不是的东西。

_CONDITION_PHRASES: dict[str, str] = {
    "flag": "世界标记 `{name}` 被置上（用 `set_flag`）",
    # ⚠️ `{items}` **不要**再加反引号：下面拼的时候每一项已经带了，
    # 再包一层会得到 ``` ``latte`` ```（Markdown 里就是"一个反引号包着的 latte"）。
    "player_has": "玩家 `{pid}` 手里**真的有** {items}（用 `give_item`）",
    "player_has_count": "玩家 `{pid}` 手里**真的有** {items}（数量版，用 `give_item`）",
    "all_flags": "这些世界标记全部置上（用 `set_flag`）：{names}",
    "any_flags": "这些世界标记里任一个置上（用 `set_flag`）：{names}",
    "all_players_spoke": "每位玩家都至少说过 {n} 次话",
}


def render_condition(condition: Any) -> str:
    """把一个 `success_when` 条件渲染成人话。

    纯函数，脱离世界可测 —— 见 `tests/test_planner.py`。
    认不出来的条件**如实说出来**，不猜：猜错会让模型去追一个不存在的目标，
    比"我不知道"更难查。
    """
    if not isinstance(condition, dict) or not condition:
        return "（没有完成条件）"

    if condition.get("all_of"):
        parts = [render_condition(c) for c in condition["all_of"]]
        return "**且**".join(f"（{p}）" if "且" in p or "或" in p else p for p in parts)
    if condition.get("any_of"):
        parts = [render_condition(c) for c in condition["any_of"]]
        return "**或**".join(f"（{p}）" if "且" in p or "或" in p else p for p in parts)

    for kind, phrase in _CONDITION_PHRASES.items():
        if kind not in condition:
            continue
        value = condition[kind]
        if kind == "flag":
            return phrase.format(name=value)
        if kind in ("all_flags", "any_flags"):
            return phrase.format(names="、".join(f"`{n}`" for n in (value or [])))
        if kind == "all_players_spoke":
            return phrase.format(n=value)
        # player_has / player_has_count
        bits = []
        for pid, items in (value or {}).items():
            if isinstance(items, dict):
                shown = "、".join(f"`{i}`×{n}" for i, n in items.items())
            else:
                shown = "、".join(f"`{i}`" for i in (items or []))
            bits.append(phrase.format(pid=pid, items=shown))
        return "，".join(bits) if bits else "（没有完成条件）"

    return f"（无法识别的完成条件：{sorted(condition)}）"


# 需要走"场景目标"而不是即时动作的意图。
# 玩家问"这里怎么点单"时，正确的回答是那套引导动作，而不是一句客套话。
SCENARIO_INTENTS = re.compile(
    r"(新手|怎么用|怎么玩|怎么点|怎么开始|怎么弄|怎么办|教我|带我|开始吧|来一局"
    r"|玩个游戏|主持|介绍一下|介绍一下自己|破冰|有什么推荐|推荐一下|有什么好玩)"
)


def recipe_needs(recipe: dict[str, Any]) -> dict[str, int]:
    """把配方的材料表统一成 {材料: 数量}。

    两种写法都要支持，因为它们表达的是两种不同的世界：

        needs: [beans, milk]             咖啡屋 —— 只要"有没有"，不问几个
        needs: {planks: 2, coal: 1}      Minecraft —— "2 块木板"和"1 块木板"不同

    咖啡屋的配方是策划手写的，写成列表更直观；Minecraft 的配方照抄原版，
    必须带数量（1 原木出 4 木板，2 木板出 4 木棍）。

    归一化放在这里而不是让每个调用点各判一次：`_plan_serve` 和 `_world_block`
    都要读这张表，两处各写一遍判断，迟早有一处漏掉 dict 分支 ——
    而漏掉的表现是"NPC 计划里少取了材料"，跑到世界那边才失败，很难查。
    """
    needs = recipe.get("needs") or {}
    if isinstance(needs, dict):
        return {str(k): int(v) for k, v in needs.items()}
    return {str(name): 1 for name in needs}


class Planner:
    def __init__(
        self,
        persona: Persona,
        llm: LLM,
        world_facts: dict[str, Any] | None = None,
        max_retries: int = 1,
        max_tokens: int = 2048,
    ) -> None:
        self.persona = persona
        self.llm = llm
        self.facts = world_facts or {}
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        #: 规划调用失败的次数与最后一次原因。
        #:
        #: 为什么要记：`plan_with_llm` 失败时会**静默回落到启发式规划**
        #: （见 `NPCAgent._decide_plan` 的兜底分支）。回落本身是对的 ——
        #: 一次调用失败不该让 NPC 卡住 —— 但它带来一个测量陷阱：
        #: `--no-planner` 和"planner 开着但一直在失败"会产生**完全一样的轨迹**，
        #: 于是那个对照实验可能在读者不知情的情况下变成自己跟自己比。
        #:
        #: 这类失败还会伪装成模型行为：预算被思维链吃光时返回的是空内容，
        #: 报错像"模型不行"，其实是配置问题（见 config.speech_max_tokens 的注释）。
        self.failures = 0
        self.last_error = ""
        #: 模型**调用成功、JSON 也解析出来了，但计划不可用**的次数
        #: （`steps` 为空，或每一步的工具名都不可用）。
        #:
        #: ⚠️ 这一类比 `failures` 更容易被漏掉：它不抛异常，所以从前
        #: **一次都没被记过** —— 于是报告里那句"解析失败 0 条"只覆盖了
        #: 抛异常的那一类，"模型给了不可用的计划"这一类是**隐形的**。
        #: 单列出来，才能把"模型不行"和"请求没回来"分开说。
        self.empty_plans = 0

    # ------------------------------------------------------------------ #
    # 场景目标 → 计划
    # ------------------------------------------------------------------ #
    def plan_objective(self, objective: dict[str, Any], tracker: StateTracker) -> Plan:
        steps = [
            PlanStep(goal=s.get("goal", ""), tool=s.get("tool", "wait"), args=dict(s.get("args") or {}))
            for s in objective.get("steps", [])
        ]
        steps = self._skip_satisfied(steps, tracker)
        return Plan(
            goal=objective.get("goal", ""),
            rationale=f"场景目标 {objective.get('id')}（优先级 {objective.get('priority', 99)}）",
            steps=steps,
            objective_id=str(objective.get("id", "")),
        )

    def plan_next_objective(
        self,
        objectives: list[dict[str, Any]],
        tracker: StateTracker,
        attempted: set[str] | None = None,
    ) -> Optional[Plan]:
        """挑一个还没完成、也还没试过的目标来做。

        `attempted` 很关键：目标的开场白只该说一次。
        没有它，NPC 会在"玩家还没开口"的每一轮重复念同一段欢迎词。

        没有 steps 的目标直接跳过：它只是一个**完成条件**，不是计划。
        多 NPC 场景的联合目标就是这种 —— "两个人都干完才算数"，
        完成与否由共享的世界状态判定（success_when: all_flags），
        没有哪一步是某一个 NPC 该单独去执行的。
        不跳过的话会得到一个空计划，还会被误标成"已尝试"而永不复查。
        """
        attempted = attempted or set()
        pending = [
            o
            for o in objectives
            if tracker.objectives.get(o.get("id"), "pending") != "done"
            and o.get("id") not in attempted
            and (o.get("steps") or [])
        ]
        if not pending:
            return None
        pending.sort(key=lambda o: o.get("priority", 99))
        return self.plan_objective(pending[0], tracker)

    # ------------------------------------------------------------------ #
    # 玩家请求 → 计划
    # ------------------------------------------------------------------ #
    def plan_for_utterance(
        self, utterance: Utterance, tracker: StateTracker
    ) -> Optional[Plan]:
        """玩家明确点单 → 生成"制作-交付"计划。

        必须先看到请求类措辞（来一杯 / 给我 / 帮我…），
        否则"我喜欢喝咖啡"这种陈述句会被误判成下单。
        """
        text = utterance.text or ""
        speaker = utterance.speaker_id

        if not self.is_request(text):
            return None

        for pattern, item_id in ORDER_PATTERNS:
            if pattern.search(text):
                return self._plan_serve(item_id, speaker, tracker)
        return None

    def is_request(self, text: str) -> bool:
        """玩家这句话是不是在**下单**（"来一杯拿铁" / "能给我做杯咖啡吗"）。

        单独开一个方法，是因为同一个判断现在有**两个**消费方：
        1. `plan_for_utterance` —— 要不要生成"制作-交付"计划；
        2. `NPCAgent._player_is_asking_me` —— 这句要不要**当问题去回答**。

        第 2 处是后加的：下单**经常写成问句**（「能给我来杯拿铁吗？」），
        光看 `is_question` 会把它当成提问，于是 NPC 回一句泛泛的话、
        而 `accept_order` 那句「好，稍等，我这就去弄」被让掉 —— 点单闭环就断了。
        两边各写一遍 `REQUEST_MARKERS.search(...)` 早晚漂移，所以收拢到这里。
        """
        return bool(REQUEST_MARKERS.search(text or ""))

    def wants_scenario_flow(self, text: str) -> bool:
        return bool(SCENARIO_INTENTS.search(text or ""))

    # ------------------------------------------------------------------ #
    def _plan_serve(self, item_id: str, player_id: str, tracker: StateTracker) -> Plan:
        """做一杯饮品并递给客人。这是"语言 → 动作 → 世界状态"最短的一条闭环。"""
        recipes = self.facts.get("recipes") or {}
        recipe = recipes.get(item_id)
        steps: list[PlanStep] = []

        if not recipe:
            return Plan(goal=f"招待{player_id}", rationale="没有这个配方", steps=[])

        station = recipe.get("station")
        player = tracker.players.get(player_id)
        player_loc = player.location if player else None
        locations = self.facts.get("locations") or {}
        station_name = locations.get(station, station or "")
        player_loc_name = locations.get(player_loc, player_loc or "")

        if station:
            steps.append(PlanStep(f"去{station_name}准备", "move_to", {"location": station}))
        for ingredient in recipe_needs(recipe):
            steps.append(PlanStep(f"取{ingredient}", "take_item", {"item": ingredient}))
        steps.append(PlanStep(f"制作{recipe.get('name', item_id)}", "craft_item", {"recipe": item_id}))
        if player_loc:
            steps.append(
                PlanStep(f"走到{player_loc_name}找客人", "move_to", {"location": player_loc})
            )
        steps.append(
            PlanStep(f"把{recipe.get('name', item_id)}递给客人", "give_item", {"item": item_id, "player": player_id})
        )
        steps = self._skip_satisfied(steps, tracker)

        # 点单闭环的两句话：先应一声，交付时说一句。
        # 没有这两句，NPC 会一声不吭地把咖啡做完塞给你 —— 技术上对，体验上很怪。
        item_name = recipe.get("name", item_id)
        steps.insert(0, PlanStep("先应一声", "speak", {"intent": "accept_order"}))
        steps.append(PlanStep("交付时招呼一声", "speak", {"intent": "deliver_order", "item_name": item_name}))

        return Plan(
            goal=f"给{player.name if player else player_id}做一杯{item_name}",
            rationale="玩家点了单，走完整的制作-交付闭环",
            steps=steps,
        )

    # ------------------------------------------------------------------ #
    # 重规划
    # ------------------------------------------------------------------ #
    def replan(
        self,
        plan: Plan,
        failed_step: PlanStep,
        reason: str,
        tracker: StateTracker,
    ) -> Optional[Plan]:
        """根据失败原因插入纠错步骤。没有可用的纠错方案时返回 None（放弃该步）。"""
        fixes = self._corrective_steps(failed_step, reason)
        if not fixes:
            return None

        new_steps: list[PlanStep] = []
        for step in plan.steps:
            if step is failed_step:
                new_steps.extend(fixes)
                new_steps.append(
                    PlanStep(
                        goal=f"{step.goal}（重试）",
                        tool=step.tool,
                        args=dict(step.args),
                        note=f"上次失败: {reason}",
                    )
                )
            else:
                new_steps.append(step)
        return Plan(
            goal=plan.goal,
            rationale=f"重规划：{reason}",
            steps=new_steps,
        )

    def _corrective_steps(self, step: PlanStep, reason: str) -> list[PlanStep]:
        """把失败原因翻译成补救动作。"""
        fixes: list[PlanStep] = []
        if "需要先 move_to" in reason or "不在你身边" in reason:
            target = self._infer_location(reason)
            if target:
                fixes.append(PlanStep(f"先移动到{target}", "move_to", {"location": target}))
        if "材料不够" in reason:
            for ingredient in re.findall(r"还缺([\u4e00-\u9fff、]+)", reason):
                for name in ingredient.split("、"): 
                    item_id = self._item_id_by_name(name)
                    if item_id:
                        fixes.append(PlanStep(f"补取{name}", "take_item", {"item": item_id}))
        if "需要待在" in reason and not fixes:
            target = self._infer_location(reason)
            if target:
                fixes.append(PlanStep(f"先移动到{target}", "move_to", {"location": target}))
        if "还不知道这个" in reason or "没有叫" in reason:
            fixes.append(PlanStep("改用已知信息回应", "speak", {"text": self.persona.unknown_topic_reply()}))
        return fixes

    def _infer_location(self, reason: str) -> Optional[str]:
        """从失败原因里推断"该去哪"。

        **必须排除「你现在在 X」里的那个 X** —— 那是出发地，不是目的地。

        早期实现取"原因串里第一个被提到的地点"。take_item 的失败原因是
        「柠檬在后厨，你现在在吧台，需要先 move_to 过去」，
        而 LOCATIONS 的字典序里「吧台」排在「后厨」前面，
        于是推断出「吧台」：NPC 原地 move_to 到自己已经站着的地方，
        再试一次 take_item 还是失败，永远拿不到东西。
        教程场景之所以没暴露这个坑，是因为它的目标步骤里本来就写好了
        move_to(kitchen)，根本走不到重规划这一步。
        """
        locations = self.facts.get("locations") or {}
        here_name = ""
        marker = "你现在在"
        pos = reason.find(marker)
        if pos != -1:
            tail = reason[pos + len(marker) :]
            for name in locations.values():
                if name and tail.startswith(name):
                    here_name = name
                    break
        for loc_id, name in locations.items():
            if name and name in reason and name != here_name:
                return loc_id
        return None

    def _item_id_by_name(self, name: str) -> Optional[str]:
        items = self.facts.get("items") or {}
        for item_id, display in items.items():
            if display == name:
                return item_id
        return None

    # ------------------------------------------------------------------ #
    def _skip_satisfied(self, steps: list[PlanStep], tracker: StateTracker) -> list[PlanStep]:
        """把**幂等且已生效**的步骤直接标记完成。

        只对 set_flag / start_activity 做跳过，因为它们的副作用是幂等的。

        move_to / take_item 刻意**不跳过** —— 位置是随时间变化的：
        计划生成时"我已经在吧台"不代表执行到那一步时还在吧台。
        （踩过的坑：提前跳过 move_to 会让"回吧台交付"这一步凭空消失，
        于是 NPC 站在后厨就想把咖啡递给门口的客人。）
        """
        for step in steps:
            if step.tool == "set_flag" and step.args.get("key") in tracker.world_flags:
                step.status = "done"
                step.note = "标记已存在"
            elif step.tool == "start_activity" and "activity_started" in tracker.world_flags:
                step.status = "done"
                step.note = "活动已开启"
        return steps

    # ------------------------------------------------------------------ #
    # LLM 路径
    # ------------------------------------------------------------------ #
    def plan_with_llm(
        self,
        tracker: StateTracker,
        utterance: Optional[Utterance],
        memories: list[MemoryRecord],
        tool_catalog: str,
        goal_hint: str = "",
    ) -> Optional[Plan]:
        if not self.llm.available:
            # 没配模型不是"失败"，是"没开这一路"。分开计数，
            # 否则 `--no-planner` 的对照组会被记成一堆失败。
            self.last_error = "未配置模型，走启发式规划"
            return None

        prompt = f"""{self.persona.system_block()}

【现场】
{tracker.scene_block()}

【想起的事】
{self._render_memories(memories)}

【玩家刚说】
{utterance.render() if utterance else "（没有人说话）"}

【可用工具】
{tool_catalog}

【世界规则】
{self._world_block()}

【本场景目标】
{goal_hint or "（无）"}

请给出一个最多 4 步的行动计划。只输出 JSON：
{{"goal": "目标", "rationale": "为什么这么做", "steps": [{{"goal": "这一步做什么", "tool": "工具名", "args": {{}}}}]}}"""
        try:
            data = self.llm.complete_json(
                [{"role": "user", "content": prompt}],
                schema_hint="goal, rationale, steps[{goal, tool, args}]",
                max_tokens=self.max_tokens,
            )
        except LLMUnavailable as exc:
            # **不要把原因吞掉。** 吞掉之后，"NPC 没做完目标"和
            # "规划调用根本没成功"在报告里长得一模一样，
            # 而这两件事的修法完全不同（一个改提示词/规划器，一个改预算）。
            self.failures += 1
            self.last_error = str(exc)
            return None

        raw_steps = data.get("steps") or []
        steps = []
        for raw in raw_steps:
            if not isinstance(raw, dict) or not raw.get("tool"):
                continue
            args = raw.get("args") or {}
            steps.append(
                PlanStep(
                    goal=str(raw.get("goal", "")),
                    tool=str(raw["tool"]),
                    args=dict(args) if isinstance(args, dict) else {},
                )
            )
        if not steps:
            # ⚠️ 这一支**从前什么都没记** —— 模型调用成功、JSON 也解析出来了，
            # 但 `steps` 为空（或每一步的工具名都不可用），于是返回 None、
            # 静默回落到启发式规划。结果是"模型给了不可用的计划"这一类
            # **从来没被计数过**，而报告里却写着"解析失败 0 条"。
            # 不抛异常不等于成功 —— 这里必须留下痕迹。
            self.empty_plans += 1
            self.last_error = (
                "模型返回了不可用的计划（steps 为空，或每一步的工具名都不可用）"
            )
            return None
        return Plan(
            goal=str(data.get("goal", "")),
            rationale=str(data.get("rationale", "由模型规划")),
            steps=steps,
        )

    @staticmethod
    def _render_memories(memories: list[MemoryRecord]) -> str:
        if not memories:
            return "（没有想起相关的事）"
        return "\n".join(f"  - {m.content}" for m in memories)

    def _world_block(self) -> str:
        """把世界规则（配方 / 工位 / 材料）喂给模型。

        不喂的话，模型只知道"存在 craft_item 这个工具"，不知道
        "拿铁要咖啡豆 + 牛奶、而且必须站在后厨"。于是它的计划是
        `craft_item(latte)` → 失败 → 再 `craft_item(latte)`，
        永远想不到中间要先 `take_item`。
        启发式规划器之所以做得到，是因为它读的就是这张 recipes 表 ——
        **两边拿到的世界知识必须是同一份**，否则"换模型"就不是换一个变量，
        而是换了一整套能力。
        """
        recipes = self.facts.get("recipes") or {}
        if not recipes:
            return "（无）"
        items = self.facts.get("items") or {}
        lines = ["制作配方（必须先备齐材料，并站到对应工位上）："]
        for recipe_id, spec in recipes.items():
            needs = "、".join(
                f"{items.get(name, name)}({name})×{count}"
                if count > 1
                else f"{items.get(name, name)}({name})"
                for name, count in recipe_needs(spec).items()
            )
            station = spec.get("station")
            yields = int(spec.get("yields") or 1)
            made = f"，产出 {yields} 个" if yields > 1 else ""
            lines.append(
                f"  - {spec.get('name', recipe_id)}({recipe_id})"
                f" ← 材料 {needs}，工位 {station}{made}"
            )
        return "\n".join(lines)
