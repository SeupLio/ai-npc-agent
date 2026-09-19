"""复读判据 —— 判断"这句话我是不是刚说过"。

## 为什么需要一个独立模块

复读是 NPC 最容易被玩家一眼看穿的毛病：技术上都对（台词符合人设、没有出戏词、
工具也调用了），但玩家第二遍听到同一句话就知道对面是个壳子。

而它**不会被任何现有指标抓到**。六维评测看的是"这句话符不符合人设、
有没有用到记忆、有没有答到点子上" —— 每一维都只看**单句**，
没有任何一维看"这句话和前面那句是不是同一句"。所以复读可以在
六维全 1.000 的情况下发生。

## 判据为什么不能直接用 `difflib`

中文短句的差异常常全在标点上。实测（`tests/test_repetition.py` 钉住这组数）：

| A | B | 原样比 | 归一化后 |
|---|---|---|---------|
| `小鹿说的我记下了。` | `小鹿说的我记下了` | 0.94 | **1.00** |
| `嗯——我听着呢，你接着说。` | `嗯我听着呢你接着说` | 0.82 | **1.00** |
| `欢迎，随便坐。今天露台的星星不错` | `欢迎随便坐今天露台的星星不错` | 0.93 | **1.00** |

这三对都是**同一句话**，但原样比都不到 1.00。所以先归一化：
去掉全部标点与空白，再比。

归一化还有个副作用是好的：它让"换了个标点重新说一遍"也算复读 ——
玩家耳朵听到的就是同一句话，标点不改变这件事。

## 两道判据

| 判据 | 含义 | 典型 |
|---|---|---|
| `== 1.00` | **逐字复读**（归一化后完全相同） | 模板固定字符串 |
| `>= 0.80` | **换汤不换药** | 同一句加了个语气词 |

阈值 0.80 是**量的**，两边都留了余量：

    算复读：我喜欢偏酸的      vs 我喜欢偏酸的咖啡     → 0.86   （就是同一件事）
    不算：  我喜欢拿铁        vs 我喜欢柠檬水         → 0.55   （两句话）
    不算：  你好呀            vs 谢谢！               → 0.00

`0.80` 落在 `0.55` 和 `0.86` 中间，离两边都不近。
调低会误伤（把"换了个话题"算成复读），调高会漏（"加个语气词再说一遍"）。
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

#: 复读判定的默认阈值。标定过程见模块 docstring 最后一段。
DEFAULT_THRESHOLD = 0.80

#: 归一化时抹掉的字符：全部中英文标点 + 空白。
#:
#: 用「抹掉标点」而不是「统一标点」，是因为目标只是"别让标点影响判定"，
#: 而统一标点需要先决定「，」和「、」算不算同一个 —— 那是个没有答案的问题。
_PUNCT = re.compile(
    r"[\s，。！？、；：…—～·「」『』“”‘’（）()\[\]【】《》〈〉,.!?;:\-—_/\\|~`\"']+"
)


def normalize(text: str) -> str:
    """抹掉标点与空白。空输入返回空串。"""
    return _PUNCT.sub("", text or "")


def similarity(a: str, b: str) -> float:
    """两句话的相似度，0~1。归一化后比较。

    任一句归一化后为空 → 0.0（空话**不算**复读：NPC 沉默是另一回事，
    把它算成复读会让"这一轮没说话"污染复读率）。
    """
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def find_repeat(
    text: str,
    history: Iterable[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> Optional[str]:
    """`text` 和 `history` 里哪一句撞了？返回撞上的那句，没有则 None。

    返回**撞上的原句**而不是布尔值，是为了让调用方能把它放进重试提示里
    （"你刚说过「…」，换一个说法"）—— 只说"你重复了"模型不知道该改什么。
    """
    for previous in history:
        if similarity(text, previous) >= threshold:
            return previous
    return None


def is_repeat(
    text: str,
    history: Iterable[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> bool:
    return find_repeat(text, history, threshold) is not None


# --------------------------------------------------------------------------- #
# 整段对话的复读率
# --------------------------------------------------------------------------- #
@dataclass
class RepeatReport:
    """一段对话的复读情况。

    `lines` 是 `(说话人, 台词)` 的序列。**只比同一个说话人自己说过的话** ——
    两个 NPC 说同一句是"撞车"（`cast` 已经有 `collisions` 在管），
    和"复读"是两件事，混在一个数里就说不清是哪个机制坏了。
    """

    total: int = 0
    repeats: list[tuple[int, int, str, str, float]] = field(default_factory=list)
    distinct: int = 0

    @property
    def rate(self) -> float:
        """复读率 = 复读的句数 / 总句数。0 句时返回 0.0。"""
        return len(self.repeats) / self.total if self.total else 0.0

    def render(self) -> str:
        if not self.repeats:
            return f"{self.total} 句，无复读"
        lines = [f"{self.total} 句，复读 {len(self.repeats)} 句（{self.rate:.1%}）"]
        for i, j, text, prev, sim in self.repeats:
            lines.append(f"  第{i + 1}句 撞第{j + 1}句（{sim:.2f}）：{text}")
        return "\n".join(lines)


def find_repeats(
    lines: Iterable[tuple[str, str]],
    threshold: float = DEFAULT_THRESHOLD,
) -> RepeatReport:
    """数一遍整段对话里的复读。

    一句台词只记**一次**（撞上最早的那一句就停）：
    否则同一句说 7 遍会被记成 1+2+…+6 = 21 次，复读率直接爆表 ——
    这个数要能拿去和别的版本比，就不能被"同一句说了几遍"放大。
    """
    items = list(lines)
    report = RepeatReport(total=len(items), distinct=len({normalize(t) for _, t in items}))
    for i, (speaker, text) in enumerate(items):
        for j in range(i):
            prev_speaker, prev = items[j]
            if prev_speaker != speaker:
                continue
            sim = similarity(text, prev)
            if sim >= threshold:
                report.repeats.append((i, j, text, prev, sim))
                break
    return report


__all__ = [
    "DEFAULT_THRESHOLD",
    "RepeatReport",
    "find_repeat",
    "find_repeats",
    "is_repeat",
    "normalize",
    "similarity",
]
