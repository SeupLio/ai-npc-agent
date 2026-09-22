"""模块四：Dialogue —— 多人对话的发言权与收件人判定。

这是整个项目和"套壳聊天机器人"最大的分水岭。

单人对话里，NPC 只要"回答最后一句话"就行了。
但米哈游公开的场景里，娜洛要同时面对 4 名玩家。这时候 NPC 必须回答三个问题：

    1. 现在该不该我说话？      （发言权）
    2. 我这句话是对谁说的？    （收件人）
    3. 没人说话的时候我要不要主动开口？（主动发起）

第 3 点尤其重要 —— 公开方案里专门提到 NPC 要"在活动冷下来时把不同玩家重新拉回同一件事"。
另外还有一条容易被忽略的约束：**不能抢戏**。所以这里设了发言占比上限，
超过上限且没被点名时，NPC 会主动沉默。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..types import DialogueDecision, Utterance
from .persona import Persona
from .state import StateTracker

# 指向 NPC 的称呼模式
#
# ⚠️ 这里原来有两个**死常量**：`QUESTION_MARKERS = ("?", "？")` 和
# `HOST_INTENTS = (...)`，定义了但全项目没有任何一处引用。
# 死常量比没有更糟 —— `QUESTION_MARKERS` 这个名字会让人以为
# "问句判定就在这儿"，而真正的判定在 `types.looks_like_question`
# （它必须认「你叫什么名字」这种不带问号的问句，见那里的注释）。
# 判句由 `Utterance.is_question` 带进来，本模块不再自己判。
#
# ⚠️ 2026-09-19 又清掉**第三个**：`DialogueConfig.min_urgency_to_speak = 0.30`。
# 同样"定义了但没人读"，而且更阴 —— 它长得像**可调阈值**，
# 于是读者会去调它、以为能改变发言门槛，实际上改了什么都不会发生。
# 发言门槛真正的判据是 `decide()` 的分支**顺序**（先匹配到哪一支就走哪一支），
# 不是任何 `urgency` 数值。`urgency` 只用于显示与排序。
# **同一个坑踩了两次，说明"清死常量"该是改这个文件时的固定动作。**


@dataclass
class DialogueConfig:
    idle_ticks_before_proactive: int = 2   # 冷场多少轮后主动开口
    npc_share_ceiling: float = 0.62        # 发言占比上限，超过且没被点名就闭嘴


class AddresseeSelector:
    """决定"这一轮 NPC 要不要说话、对谁说"。"""

    def __init__(self, persona: Persona, config: DialogueConfig | None = None) -> None:
        self.persona = persona
        self.config = config or DialogueConfig()

    # ------------------------------------------------------------------ #
    def mentions_npc(self, text: str) -> bool:
        if not text:
            return False
        candidates = {self.persona.name, *self.persona.aliases}
        return any(name and name in text for name in candidates)

    def pick_addressee(
        self, tracker: StateTracker, utterance: Optional[Utterance]
    ) -> Optional[str]:
        """选出这句话应该对谁说。

        优先级：被点名的人 > 刚说话的人 > 最近活跃的玩家

        被点名的可以是**同伴 NPC** —— 这样"阿柚转头跟小舟说：你来弹一首"
        才有一个明确的收件人，下一轮小舟才知道该自己接。
        """
        if utterance is not None:
            for pid in utterance.mentions:
                if pid in tracker.players or pid in tracker.peers:
                    return pid
            if utterance.speaker_id in tracker.players:
                return utterance.speaker_id
            if utterance.speaker_id in tracker.peers:
                # 同伴刚跟我说话 —— 回他，而不是回某个玩家
                return utterance.speaker_id

        active = tracker.active_players()
        if active:
            return active[0].id
        return None

    # ------------------------------------------------------------------ #
    def decide(
        self,
        tracker: StateTracker,
        utterance: Optional[Utterance],
        *,
        has_pending_plan: bool = False,
        other_npc_spoke_last: bool = False,
    ) -> DialogueDecision:
        """核心决策。返回 should_speak / addressed_to / 理由 / 紧急度。"""
        cfg = self.config

        # 0) 多 NPC 场景：另一个 NPC 刚说完，让出去
        if other_npc_spoke_last and not (utterance and self.mentions_npc(utterance.text)):
            return DialogueDecision(False, None, "另一个 NPC 刚发言，不打断", 0.0)

        # 1) 有没做完的计划 —— 继续推进。
        #
        # ⚠️ 这里原来写的是"继续推进，**但只在计划里含说话步骤时才发言**"。
        # 那句话**是假的**：本支无条件返回 `should_speak=True`，
        # 没有任何地方检查"计划里有没有 speak 步骤"。
        #
        # 真实语义是："**这一轮我有活要干**"，不是"这一轮我要开口"。
        # 开口与否由 `NPCAgent.step()` 决定：玩家在跟我说话时，这一轮的
        # 那句话由 `_respond` 产出，计划里排到 `speak` 的那一步由
        # `_run_plan` 让开（一轮只说一句）；玩家没说话时才由计划开口。
        # 把两件事混成一句注释，会让人以为"没话说就不会走这支"，
        # 从而看不出"NPC 在跑计划期间曾经完全听不见玩家"那个缺陷
        # （见 `NPCAgent._player_wants_me` 的注释）。
        if has_pending_plan:
            return DialogueDecision(
                True,
                self.pick_addressee(tracker, utterance),
                "继续执行未完成的计划",
                urgency=0.55,
            )

        # 2) 被点名 —— 最高优先级，必须回应
        if utterance is not None and self.mentions_npc(utterance.text):
            target = self.pick_addressee(tracker, utterance)
            return DialogueDecision(
                True,
                target,
                f"{utterance.speaker_name}点名了我",
                urgency=0.95,
            )

        # 3) 有人提问但没点名 —— 主持人身份或话题开放时接话
        if utterance is not None and utterance.is_question:
            share = tracker.npc_share()
            if share > cfg.npc_share_ceiling:
                return DialogueDecision(
                    False, None, f"我发言占比已达 {share:.0%}，把机会让给玩家", 0.2
                )
            return DialogueDecision(
                True,
                self.pick_addressee(tracker, utterance),
                "现场有人抛出问题，我来接",
                urgency=0.6,
            )

        # 4) 玩家在跟别人说话 —— 不插嘴
        if utterance is not None and utterance.mentions:
            return DialogueDecision(
                False, None, "这句话不是对我说的", 0.1
            )

        # 5) 冷场 —— 主动发起话题（这是"活世界"的关键动作）
        if tracker.silence_ticks >= cfg.idle_ticks_before_proactive:
            target = self.pick_addressee(tracker, utterance)
            return DialogueDecision(
                True,
                target,
                f"冷场 {tracker.silence_ticks} 轮，主动把话头捡起来",
                urgency=0.5,
                proactive=True,
            )

        # 6) 有玩家说了话但没点名 —— 轻量回应
        if utterance is not None and utterance.speaker_id in tracker.players:
            share = tracker.npc_share()
            if share > cfg.npc_share_ceiling:
                return DialogueDecision(
                    False, None, f"我发言占比已达 {share:.0%}，保持安静", 0.15
                )
            return DialogueDecision(
                True,
                self.pick_addressee(tracker, utterance),
                "接着玩家的话头往下聊",
                urgency=0.45,
            )

        return DialogueDecision(False, None, "没有需要回应的输入", 0.0)

    # ------------------------------------------------------------------ #
    def pick_proactive_intent(
        self,
        tracker: StateTracker,
        only: set[str] | None = None,
        used: list[str] | None = None,
    ) -> str:
        """冷场时该说什么。优先推进未完成的目标。

        ``only`` 用来把范围收窄到"我自己的目标" ——
        多 NPC 场景里 tracker.objectives 是整个世界的目标表，
        不过滤就会替同伴操心（见 NPCAgent._respond 的注释）。

        ``used`` 是**最近说过的话分别用的意图**。目标没做完时，
        每个冷场轮都会重算一次同一个意图，于是同一句开场白被反复捡起来 ——
        实测（duet，10 轮）「你上次说过第一次来吧。」出现了 4 次。
        有多个未完成目标时，优先挑一个**最近没说过的**；
        只有一个目标时不硬换（换掉就不是"推进目标"了，而是跑题）。
        """
        pending = [
            k
            for k, v in tracker.objectives.items()
            if v != "done" and (only is None or k in only)
        ]
        mapping = {
            "greet_all": "invite_intro",
            "find_topic": "probe",
            "welcome_drink": "greet_new",
            "teach_order": "teach_order",
            "host_round": "ask_question",
        }
        # 去重但保持顺序 —— 两个目标可能映射到同一个意图，
        # 不去重的话"优先挑没说过的那句"会在同一个意图上空转一圈。
        options: list[str] = []
        for objective in pending:
            intent = mapping.get(objective)
            if intent and intent not in options:
                options.append(intent)
        if not options:
            return "fallback"

        recent = list(used or [])
        for intent in options:
            if intent not in recent:
                return intent
        return options[0]


class TurnManager:
    """维护谁在什么时候说了话，并给出本轮决策。给 Agent 循环调用。"""

    def __init__(self, selector: AddresseeSelector) -> None:
        self.selector = selector
        self.history: list[DialogueDecision] = []

    def next_decision(
        self,
        tracker: StateTracker,
        utterance: Optional[Utterance],
        *,
        has_pending_plan: bool = False,
        other_npc_spoke_last: bool = False,
    ) -> DialogueDecision:
        decision = self.selector.decide(
            tracker,
            utterance,
            has_pending_plan=has_pending_plan,
            other_npc_spoke_last=other_npc_spoke_last,
        )
        self.history.append(decision)
        return decision
