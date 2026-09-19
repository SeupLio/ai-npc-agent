"""模块一：Persona —— 人设与边界控制。

人设不是一段"你是一个温柔的咖啡店老板"的提示词就完事了。
真正让 NPC 可控的是三层约束：

    1. 语气层  —— 说话风格、句数上限（防止 NPC 长篇大论抢玩家的戏）
    2. 知识层  —— 能聊什么、不能聊什么（不知道就说不清楚，而不是硬编）
    3. 红线层  —— 出戏词 + 剧透词（未解锁前不允许出现）

三层都是可检查的，所以"人设一致性"才能被量化成评测指标，而不是靠感觉。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# 句末边界。两条容易踩的规则：
#
# 1. **「…」不是句末**。省略号表示语气拖长，算成句末的话，
#    小舟的开场「……你好。」会被切成 ["…", "…", "你好。"] 三段，
#    sentence_max=2 一裁就只剩「……」，整句台词凭空消失。
# 2. **连续标点算一个边界**。「真的吗？！」是一个人问了一句话，
#    不是两句；所以只在前一个标点后面**不是**标点时才切。
SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?])(?![。！？!?])")


@dataclass
class Persona:
    id: str
    name: str
    role: str = ""
    aliases: list[str] = field(default_factory=list)
    traits: list[str] = field(default_factory=list)
    style: dict[str, Any] = field(default_factory=dict)
    can_discuss: list[str] = field(default_factory=list)
    locked_topics: list[str] = field(default_factory=list)
    spoiler_terms: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    relationships: dict[str, str] = field(default_factory=dict)
    #: 意图 → 台词模板。值可以是 str，也可以是 list[str]（多变体，见 `template()`）。
    templates: dict[str, Any] = field(default_factory=dict)
    #: 「问到自己」时的答话：`[{"ask": [关键词…], "reply": "…"}]`。
    #:
    #: 为什么需要它：离线启发式答不上"关于 NPC 自己"的问题，于是
    #   玩家：你叫什么名字
    #   NPC ：这个我不太清楚，别听我瞎说。      ← NPC 不知道自己的名字
    # 这比复读还难看 —— 复读只是没信息，这是**人设当场崩掉**。
    # 而这类问题恰恰是最容易被问到的（玩家第一次见到 NPC 就会问）。
    #: 关键词匹配是**精确子串**，不做模糊：模糊匹配会答出胡话，见 `_knowledge_answer`。
    self_facts: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Persona":
        boundary = data.get("knowledge_boundary") or {}
        return cls(
            id=data.get("id", "npc"),
            name=data.get("name", "NPC"),
            role=data.get("role", ""),
            aliases=list(data.get("aliases") or []),
            traits=list(data.get("traits") or []),
            style=dict(data.get("style") or {}),
            can_discuss=list(boundary.get("can_discuss") or []),
            locked_topics=list(boundary.get("locked") or []),
            spoiler_terms=list(data.get("spoiler_terms") or []),
            goals=list(data.get("goals") or []),
            relationships=dict(data.get("relationships") or {}),
            templates=dict(data.get("utterance_templates") or {}),
            self_facts=list(data.get("self_facts") or []),
        )

    # ------------------------------------------------------------------ #
    # 供 prompt 使用
    # ------------------------------------------------------------------ #
    def system_block(self) -> str:
        lines = [f"你是{self.name}，{self.role}。"]
        if self.traits:
            lines.append("性格：" + "；".join(self.traits) + "。")
        style = self.style
        if style.get("tone"):
            lines.append(f"说话风格：{style['tone']}。")
        if style.get("sentence_max"):
            lines.append(f"每次最多说 {style['sentence_max']} 句话，不要长篇大论。")
        if self.goals:
            lines.append("你在意的事：" + "；".join(self.goals) + "。")
        lines.append(
            "铁律：不跳出角色，不承认自己是程序或模型，不替玩家做决定，不剧透未解锁的内容。"
        )
        return "\n".join(lines)

    def style_hint(self) -> str:
        style = self.style
        bits = []
        if style.get("tone"):
            bits.append(str(style["tone"]))
        if style.get("tics"):
            bits.append("口头禅：" + "、".join(style["tics"]))
        return "；".join(bits)

    # ------------------------------------------------------------------ #
    # 一致性检查（评测指标的直接来源）
    # ------------------------------------------------------------------ #
    @property
    def forbidden_phrases(self) -> list[str]:
        return list(self.style.get("forbidden") or [])

    def out_of_character(self, text: str) -> list[str]:
        """返回命中的出戏词。空列表表示通过。"""
        if not text:
            return []
        return [p for p in self.forbidden_phrases if p and p in text]

    def spoiler_hits(self, text: str, unlocked_topics: set[str]) -> list[str]:
        """返回命中的剧透词（已解锁的话题不算）。"""
        if not text:
            return []
        unlocked_texts = set()
        for topic in unlocked_topics:
            unlocked_texts.add(topic)
        hits = []
        for term in self.spoiler_terms:
            if term and term in text and not any(term in u for u in unlocked_texts):
                hits.append(term)
        return hits

    def too_long(self, text: str) -> bool:
        limit = int(self.style.get("max_chars") or 0)
        return bool(limit) and len(text or "") > limit

    def sentence_count(self, text: str) -> int:
        return len([s for s in SENTENCE_SPLIT.split(text or "") if s.strip()])

    def check(self, text: str, unlocked_topics: set[str] | None = None) -> list[str]:
        """一次性返回所有违规项，供评测与 Reflection 使用。"""
        violations: list[str] = []
        ooc = self.out_of_character(text)
        if ooc:
            violations.append(f"出戏词: {', '.join(ooc)}")
        spoilers = self.spoiler_hits(text, unlocked_topics or set())
        if spoilers:
            violations.append(f"剧透: {', '.join(spoilers)}")
        if self.too_long(text):
            violations.append("过长")
        max_sentences = int(self.style.get("sentence_max") or 0)
        if max_sentences and self.sentence_count(text) > max_sentences:
            violations.append("句数超限")
        return violations

    # ------------------------------------------------------------------ #
    # 离线启发式：台词生成与风格裁剪
    # ------------------------------------------------------------------ #
    def template(self, intent: str, variant: int | None = None) -> str:
        """取某个意图的台词模板。

        模板可以是**一个字符串**，也可以是**一串候选**。
        写成候选是为了治复读：像 `acknowledge`（"我记下了"）这种高频意图，
        只有一个写法时，同一段对话里必然出现逐字重复 —— 实测过，
        10 轮对话里同一句说了 7 遍（见 `npc_agent/modules/repetition.py`）。

        `variant` 由调用方按"这个意图用过几次"给出（`NPCAgent._intent_uses`），
        所以轮换是**确定性**的：同一段对话重放必然得到同一串台词。
        这一点不能破 —— 控制台每次请求都从头重放整段对话，靠的就是它。
        """
        raw: Any = self.templates.get(intent) or self.templates.get("fallback") or "嗯——"
        if isinstance(raw, (list, tuple)):
            pool = [str(item) for item in raw if str(item).strip()]
            if not pool:
                return "嗯——"
            raw = pool[(variant or 0) % len(pool)]
        return str(raw)

    def template_count(self, intent: str) -> int:
        """这个意图有几个变体（单条模板算 1）。

        给"去重时把所有变体都试一遍"用：只试下一个变体是不够的 ——
        变体是循环使用的，用到第 4 次时"下一个"很可能正是很久以前说过的那个。
        """
        raw: Any = self.templates.get(intent)
        if isinstance(raw, (list, tuple)):
            return max(1, len([item for item in raw if str(item).strip()]))
        return 1

    def render_template(
        self, intent: str, variant: int | None = None, **kwargs: Any
    ) -> str:
        raw = self.template(intent, variant=variant)
        try:
            return raw.format(**{k: (v if v is not None else "") for k, v in kwargs.items()})
        except (KeyError, IndexError):
            return raw

    def apply_style(self, text: str) -> str:
        """把文本裁到人设允许的长度（离线模式用；有模型时由模型自己守规矩）。"""
        if not text:
            return text
        max_sentences = int(self.style.get("sentence_max") or 0)
        if max_sentences:
            parts = [s for s in SENTENCE_SPLIT.split(text) if s.strip()]
            if len(parts) > max_sentences:
                text = "".join(parts[:max_sentences])
        limit = int(self.style.get("max_chars") or 0)
        if limit and len(text) > limit:
            text = text[:limit].rstrip("，,、 ") + "。"
        return text

    def unknown_topic_reply(self) -> str:
        return "嗯——这个我还真说不好。"

    def answer_about_self(self, text: str) -> str:
        """玩家问到 NPC 自己时，它该怎么答。没有匹配返回空串。

        **精确子串匹配**，不做模糊：模糊匹配在短句上会答出胡话
        （"你叫什么名字"和"这店开了多久了"的字符重合度都很高），
        而这里的代价是人设当场崩掉 —— 宁可说"说不好"，不能答错自己的名字。
        """
        question = text or ""
        for entry in self.self_facts:
            keywords = entry.get("ask") or []
            reply = str(entry.get("reply") or "").strip()
            if reply and any(str(k) and str(k) in question for k in keywords):
                return reply
        return ""

    def greet(self) -> str:
        return self.render_template("opening", topic_hint="").strip()
