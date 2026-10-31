"""世界观一致性 —— 让 NPC 说不出与设定矛盾的话。

## 这个模块补的是"OOC"里最难抓的一种

出戏词（"作为一个 AI"）是**表面**问题，词表就能挡；
剧透是**边界**问题，解锁状态就能挡；
而"这家店开在星屿的旧灯塔下面"被 NPC 说成"我们店开在山顶上"，
是**事实**问题 —— 它语法通顺、语气对、没出戏、没剧透，
只是**和这个世界不符**。

前两种本项目一直有护栏，第三种没有。而它恰恰是长线运营里最贵的错误：
玩家会截图、会考据、会在社区里问"设定是不是改了"。
米哈游公开方案里把"世界观与编年史"单列成一个工具，原因就在这里。

## 为什么判据是"同一句话里同时出现主体和矛盾说法"

只查矛盾词会误伤：「山顶的风景不错」是一句普通的闲聊，
它没有断言"这家店在山顶"。
只查主体会误伤：「这家店我常来」没有说任何事实。

所以判据是**两者同句共现** —— 主体（这家店的某个叫法）+ 与设定冲突的说法。
这个口径保守，会漏（换了说法的矛盾抓不到），但**不会冤枉**。
在这个位置上，漏比冤枉便宜：冤枉会让 NPC 说不出本来正确的话，
而玩家看不出那是护栏的错，只会觉得这 NPC 怪。

## 它同时是"给模型的设定书"

`canon_block()` 会被拼进台词 prompt。护栏负责**事后拦截**，
设定书负责**事前减少** —— 只有拦截没有设定书，模型会反复撞墙；
只有设定书没有拦截，模型偶尔还是会编。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

#: 句子边界。用**分句**而不是整段：一段话里前半句说店、后半句说山，
#: 那不是矛盾，是两件事。
_SENTENCE = re.compile(r"[。！？；!?;\n]+")


@dataclass(frozen=True)
class LoreEntry:
    """一条设定。

    `wrong` 是**与设定冲突的说法**，不是"不许出现的词"。
    两者的区别：后者会把「山顶的风景」也挡掉，前者只在
    「这家店……山顶」同句出现时才判定为矛盾。
    """

    id: str
    subject: tuple[str, ...]
    relation: str
    value: str
    wrong: tuple[str, ...] = ()
    note: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoreEntry":
        subject = data.get("subject") or []
        if isinstance(subject, str):
            subject = [subject]
        wrong = data.get("wrong") or []
        if isinstance(wrong, str):
            wrong = [wrong]
        return cls(
            id=str(data.get("id") or data.get("relation") or "lore"),
            subject=tuple(str(s) for s in subject if str(s).strip()),
            relation=str(data.get("relation") or ""),
            value=str(data.get("value") or ""),
            wrong=tuple(str(w) for w in wrong if str(w).strip()),
            note=str(data.get("note") or ""),
        )

    def render(self) -> str:
        return f"{self.subject[0] if self.subject else '?'}的{self.relation}是{self.value}"


@dataclass
class LoreViolation:
    entry_id: str
    subject: str
    wrong: str
    sentence: str

    def render(self) -> str:
        return f"与设定冲突：{self.subject}的{self.entry_id}不是「{self.wrong}」（原句：{self.sentence}）"


class LoreBook:
    """世界设定书 + 一致性判据。"""

    def __init__(self, entries: Iterable[LoreEntry] = ()) -> None:
        self.entries: list[LoreEntry] = list(entries)
        #: 检查过多少句、拦下多少次。报告用它证明护栏真的在跑，
        #: 而不是"写了但一次都没触发"。
        self.checked = 0
        self.hits = 0

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dicts(cls, raw: Iterable[dict[str, Any]] | None) -> "LoreBook":
        return cls(LoreEntry.from_dict(item) for item in (raw or []) if isinstance(item, dict))

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)

    # ------------------------------------------------------------------ #
    def check(self, text: str) -> list[LoreViolation]:
        """这句话有没有和设定矛盾。空设定书 / 空文本 → 没有违规。"""
        if not text or not self.entries:
            return []
        out: list[LoreViolation] = []
        for sentence in _SENTENCE.split(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            self.checked += 1
            for entry in self.entries:
                if not entry.wrong:
                    continue
                subject = next((s for s in entry.subject if s and s in sentence), None)
                if subject is None:
                    continue
                wrong = next((w for w in entry.wrong if w in sentence), None)
                if wrong is None:
                    continue
                out.append(LoreViolation(entry.id, subject, wrong, sentence))
        if out:
            self.hits += 1
        return out

    def canon_block(self, limit: int = 8) -> str:
        """拼进 prompt 的设定书。没有设定时返回空串（单 NPC 场景不受影响）。"""
        if not self.entries:
            return ""
        lines = [f"  - {entry.render()}" for entry in self.entries[:limit]]
        return "【这个世界的设定】（说错会被拦下，不要编）\n" + "\n".join(lines)

    def render(self) -> str:
        if not self.entries:
            return "（没有设定书）"
        return "\n".join(f"  [{e.id}] {e.render()}" for e in self.entries)


__all__ = ["LoreBook", "LoreEntry", "LoreViolation"]
