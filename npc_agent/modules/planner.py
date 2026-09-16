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
ORDER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(拿铁|咖啡|latte)", re.I), "latte"),
    (re.compile(r"(柠檬水|柠檬|lemonade)", re.I), "lemonade"),
    (re.compile(r"(苹果派|苹果|apple\s*pie)", re.I), "apple_pie"),
]

# 只有出现"请求类"措辞时才算点单。
# 否则"我特别喜欢偏酸的咖啡"会被误判成下单，NPC 就会莫名其妙去做一杯咖啡。
REQUEST_MARKERS = re.compile(
    r"(来一?[杯份个]|要一?[杯份个]|给我|帮我|麻烦|能不能给|可以给|想喝|想点|点一?[杯份个]|上一?[杯份个])"
)

# 需要走"场景目标"而不是即时动作的意图。
# 玩家问"这里怎么点单"时，正确的回答是那套引导动作，而不是一句客套话。
SCENARIO_INTENTS = re.compile(
    r"(新手|怎么用|怎么玩|怎么点|怎么开始|怎么弄|怎么办|教我|带我|开始吧|来一局"
    r"|玩个游戏|主持|介绍一下|介绍一下自己|破冰|有什么推荐|推荐一下|有什么好玩)"
)


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

        if not REQUEST_MARKERS.search(text):
            return None

        for pattern, item_id in ORDER_PATTERNS:
            if pattern.search(text):
                return self._plan_serve(item_id, speaker, tracker)
        return None

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
        for ingredient in recipe.get("needs", []):
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
        except LLMUnavailable:
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
            needs = "、".join(f"{items.get(n, n)}({n})" for n in spec.get("needs", []))
            station = spec.get("station")
            lines.append(
                f"  - {spec.get('name', recipe_id)}({recipe_id})"
                f" ← 材料 {needs}，工位 {station}"
            )
        return "\n".join(lines)
