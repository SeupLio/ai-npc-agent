"""玩家情绪感知 —— NPC 要听得出来"这句话带不带情绪"。

## 为什么这个模块必须存在

米哈游《星布谷地》公开的玩家实录里有一条被反复引用的细节：
玩家大喊一声「娜洛！」，NPC 从**语气**里听出紧迫，惊慌地问怎么了。
反过来说明一件事：只做"内容理解"的 NPC 会漏掉一大类真实输入——

    「娜洛！！出事了！！」  ← 没有任何问句结构，纯靠语气
    「你们这咖啡也太难喝了吧」 ← 陈述句，内容是投诉

这两句话如果都按 `KIND_STATEMENT` 走 `acknowledge`，玩家的体感就是
"NPC 在敷衍我"。**语气是输入的一部分，不是噪声。**

## 判据（离线、零成本、可复现）

三档情绪，优先级 `urgent > complaint > praise`（一次大喊同时带着
抱怨词时，先处理"人可能有急事"，再谈咖啡好不好喝）：

- **urgent（急切/求助）**：连续感叹号、重复感叹、求助关键词
  （救命/快/帮帮我/出事了/着火了）、或连续问号+感叹号叠加。
- **complaint（投诉/不满）**：负面评价词（难喝/太贵/差评/离谱），
  仅在**非问句**时成立 —— 「好喝吗？」是提问不是夸，反之亦然。
- **praise（夸赞）**：正面评价词（好喝/好听/喜欢/棒），同样仅在
  非问句时成立。

问句豁免的理由写在判据里：情绪词出现在问句里时，玩家在**求信息**，
把它当情绪回应就是答非所问（实测最容易翻车的场景）。
`urgent` 不豁免问句 —— 「快！还有救吗？？」里"快"本身就是信号。

## 产出怎么被消费

`detect_emotion` 返回 `"urgent" | "complaint" | "praise" | "neutral"`。
Agent 在意图裁定前调用；命中非中性情绪时把意图顶到对应档
（`respond_urgent` / `soothe` / `delight`），模板槽位里带上
`{emotion}` 提示，LLM 路径则把「玩家情绪：急切」注入提示词 ——
**离线和在线看到的是同一条判据**，这是可评测的前提。
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 关键词表（离线判据的全部事实来源；改这里 = 改行为，记得同步测试）
# ---------------------------------------------------------------------------
URGENT_MARKERS = (
    "救命",
    "帮帮我",
    "帮我",
    "快帮",
    # 「快！/ 快!」：催促+感叹的经典组合。单独一个「快」不算 ——
    # 「快说说什么事」是普通催促句；感叹号才是"现在就要"的信号。
    "快！",
    "快!",
    "快点！",
    "出事了",
    "出事",
    "着火了",
    "着火",
    "急用",
    "不好了",
    "糟了",
)

COMPLAINT_MARKERS = (
    "难喝",
    "太难喝",
    "太难吃",
    "太贵",
    "贵得",
    "差评",
    "投诉",
    "退货",
    "退款",
    "离谱",
    "敷衍",
    "什么破",
    "不咋地",
    "差劲",
)

PRAISE_MARKERS = (
    "好喝",
    "好喝到",
    "太好喝",
    "好听",
    "太好听",
    # ⚠️ 「喜欢」必须带指向才算夸。裸「喜欢」是**偏好陈述**：
    # 「我口味偏酸，酸一点的都喜欢」是玩家在说自己爱什么，
    # 不是在夸你 —— 拿 delight 接它会挤掉 acknowledge，
    # 而 acknowledge 的 {said_hint} 引用（"你说的……我记下了"）
    # 恰恰是"我在听"的痕迹（实测被 test_recall_cooldown 抓到）。
    "喜欢你们",
    "喜欢这",
    "喜欢你",
    "真棒",
    "太棒",
    "厉害",
    "绝了",
    "满分",
    "用心",
)

# 连续感叹号（中英文都算）：两个及以上才算"喊"。
# 单个感叹号是正常语气强度，触发情绪回应会显得一惊一乍。
_RE_SHOUT = re.compile(r"[!！]{2,}")
# 感叹+问号叠加：「？？！」「！?！」——错愕+急切。
_RE_PANIC_MIX = re.compile(r"[!！?？]{3,}")


def detect_emotion(text: str) -> str:
    """判读一句话的语气档位。返回四档之一。

    纯离线启发式，无状态、无副作用 —— 同一句话永远同一个答案，
    这是它可进回归测试的前提。
    """
    if not text:
        return "neutral"
    stripped = text.strip()

    # 急切：结构信号（喊）优先于内容信号（词），两个取其一就算。
    if _RE_SHOUT.search(stripped) or _RE_PANIC_MIX.search(stripped):
        return "urgent"
    if any(m in stripped for m in URGENT_MARKERS):
        return "urgent"

    # 投诉 / 夸赞只在**非问句**成立（理由见模块头注释）。
    is_question = "？" in stripped or "?" in stripped
    if not is_question:
        if any(m in stripped for m in COMPLAINT_MARKERS):
            return "complaint"
        if any(m in stripped for m in PRAISE_MARKERS):
            return "praise"

    return "neutral"


# ---------------------------------------------------------------------------
# 情绪 → 意图 / 提示词的映射（消费方读这张表，不要各写一份 if-else）
# ---------------------------------------------------------------------------
EMOTION_INTENTS = {
    "urgent": "respond_urgent",
    "complaint": "soothe",
    "praise": "delight",
}

# 提示词注入用的中文描述。命名刻意直白：这段文本会被拼进
# LLM 提示词，读它的模型和读它的人都不该需要解码。
EMOTION_LABELS = {
    "urgent": "急切（玩家可能在喊你，先关心人再谈事）",
    "complaint": "不满（玩家在投诉，先接住情绪再处理事）",
    "praise": "高兴（玩家在夸你，接住这份好意）",
}


def emotion_label(emotion: str) -> str:
    return EMOTION_LABELS.get(emotion, "")


def intent_for(emotion: str) -> str:
    """情绪档位对应的模板意图。中性返回空串（调用方按原意图走）。"""
    return EMOTION_INTENTS.get(emotion, "")


__all__ = [
    "COMPLAINT_MARKERS",
    "EMOTION_INTENTS",
    "EMOTION_LABELS",
    "PRAISE_MARKERS",
    "URGENT_MARKERS",
    "detect_emotion",
    "emotion_label",
    "intent_for",
]
