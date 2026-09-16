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
    templates: dict[str, str] = field(default_factory=dict)

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
    def template(self, intent: str) -> str:
        return self.templates.get(intent) or self.templates.get("fallback") or "嗯——"

    def render_template(self, intent: str, **kwargs: Any) -> str:
        raw = self.template(intent)
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

    def greet(self) -> str:
        return self.render_template("opening", topic_hint="").strip()
