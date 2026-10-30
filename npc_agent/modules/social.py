"""共享记忆 —— NPC 之间的"传闻网络"。

## 这个模块对应的是哪一种"活"

米哈游公开的方案里有一条很容易被当成宣传话术的描述：
**"你在咖啡店跟娜洛聊过的事，可能过几天就被街市的谷多曼提起。"**
这句话在工程上有一个非常具体的含义 —— 记忆不是每个角色各自一份孤岛，
而是**会在角色之间流动**。

本框架原来的记忆是**完全私有**的：`MemoryManager` 一人一份，
`Cast.memory_contents()` 明确按 NPC 分开（那是评测要的）。
于是"信息在世界上流通"这件事在架构里根本不存在。

## 传什么，不传什么

不是所有记忆都该流动。一个 NPC 心里的事有三类：

| 类别 | 例子 | 传吗 |
|---|---|---|
| **世界事实** | 这家店开在旧灯塔下面 | ✅ 传（本来就该大家都知道） |
| **对玩家的判断** | 阿澈喜欢偏酸的 | ✅ 传（这正是"角色之间共享印象"） |
| **私下的想法** | 我觉得阿柚今天有点怪 | ❌ 不传 |

第三类不传是**产品判断**，不是技术限制：NPC 把心里话全广播出去，
玩家会发现这个世界没有"背面"，所有人都是同一个信息源。
所以只有**说出口的**事实才进网络 —— 传播的入口是 `speak`，
不是记忆写入。这条边界比任何过滤规则都可靠。

## 为什么要带来源

「听阿柚说，这家店开在旧灯塔下面」和「这家店开在旧灯塔下面」
在玩家耳朵里是两句不同的话：前者说明**这个世界里有人在传话**。
所以每条传闻都带 `source_name`，并在台词里显式说出来。
去掉来源，这个功能就退化成"复制了一条记忆"，看不出任何"活"。

## 只传一跳

默认 `max_hops=1`：阿柚说的，小舟能记住；小舟再传给别人时，
来源仍然写"听阿柚说"。不做多跳扩散，理由是**可验证性**：
两跳之后，"谁说的"和"说的还是不是原话"都没法保证，
而一条传错三次的传闻比没有传闻更伤 —— 它会让世界看起来在撒谎。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: 传闻的归一化键。和复读判据、话题账本同一套口径（去标点与空白）。
_PUNCT = re.compile(r"[\s，。！？、；：…—～·「」『』“”‘’（）()\[\]【】《》〈〉,.!?;:\-—_/\\|~`\"']+")


def fact_key(content: str) -> str:
    """一条事实的去重键。"""
    return _PUNCT.sub("", content or "")


#: 一句玩家发言里出现这些词，说明它在讲**自己的偏好**。
#: 偏好是唯一一种"值得被别人知道、但说出来时没有人会当回事"的信息 ——
#: 所以它是传闻最自然的载体。
_PREFERENCE_MARKERS = ("喜欢", "讨厌", "不爱", "习惯", "常来", "怕", "忌口", "过敏")

#: 第一人称。转成第三人称时只换这个 —— **不换「你」**：
#: 「你」在玩家的话里通常指 NPC，换掉会变成"阿澈说阿柚喜欢…"这种张冠李戴。
_FIRST_PERSON = "我"


def preference_clause(text: str) -> str:
    """从一句话里取出"讲偏好"的那一小句。没有则返回空串。

    只取**第一小句**，不做句法分析：传闻要短，长了就变成"复述原话"，
    而复述原话这件事，接收方本来就已经听见了（`observe_utterance`）。
    """
    if not any(marker in (text or "") for marker in _PREFERENCE_MARKERS):
        return ""
    for clause in re.split(r"[，。！？、；]", text or ""):
        clause = clause.strip()
        if clause and any(marker in clause for marker in _PREFERENCE_MARKERS):
            return clause
    return ""


def to_third_person(clause: str, name: str) -> str:
    """「我口味偏酸」+ 阿澈 → 「阿澈口味偏酸」。

    这一步是**传闻能不能读通**的关键。不做转换，接收方记住的是
    「听阿柚说，我口味偏酸」—— 那个"我"是谁？NPC 会把玩家的话
    当成自己的话（这个坑在 `extract_memory_hint` 里已经踩过一次）。

    替换是**全局**的，不只换句首：实测「酸一点的我都喜欢」只换句首会得到
    「阿澈酸一点的我都喜欢」—— 一句话里剩下一个"我"，读起来像另一个人。
    """
    clause = (clause or "").strip()
    if not clause:
        return ""
    return clause.replace(_FIRST_PERSON, name)


@dataclass(frozen=True)
class SharedFact:
    """一条可以被传的事实。

    `fact_id` 是去重键，不是随机 id —— 同一件事被两个人说出来时，
    玩家不该听到两遍"听阿柚说…"和"听小舟说…"。
    """

    fact_id: str
    content: str
    source_id: str
    source_name: str
    tick: int
    importance: float = 0.7
    subject: str = "world"


@dataclass
class GossipNetwork:
    """一张只记"谁说过什么"的网。它不存记忆，只负责**投递**。

    ⚠️ 它和 `MemoryStore` 的分工要写清楚：
    网络是**公共的、永久的**（这一局里谁说过什么）；
    记忆是**私有的、会衰减的**（我记住了什么）。
    把两者合并会得到一个"所有 NPC 共享一个记忆库"的怪物 ——
    那正是本项目在 `cast.py` 里明确拒绝的设计（"两个 NPC 不该共用一个脑子"）。
    """

    #: 只有重要度 ≥ 阈值的事实才传播。太低会把"嗯""好的"也传成传闻。
    threshold: float = 0.7
    #: 最多传几跳。见模块 docstring 最后一段。
    max_hops: int = 1

    _facts: dict[str, SharedFact] = field(default_factory=dict)
    #: actor_id → 已经拿到过哪些 fact_id。
    _delivered: dict[str, set[str]] = field(default_factory=dict)
    #: 投递次数。报告用它证明"传闻真的发生过"，而不是"代码写了但没跑"。
    delivered_count: int = 0
    #: 因为重要度不够被挡下的条数。它是"过滤在工作"的证据。
    filtered_count: int = 0

    # ------------------------------------------------------------------ #
    def publish(self, fact: SharedFact) -> bool:
        """发布一条事实。重复发布返回 False（同一个 fact_id 只算一次）。"""
        if fact.importance < self.threshold:
            self.filtered_count += 1
            return False
        if fact.fact_id in self._facts:
            return False
        self._facts[fact.fact_id] = fact
        # 说话人自己不需要"被告知"。
        self._delivered.setdefault(fact.source_id, set()).add(fact.fact_id)
        return True

    def digest(self, actor_id: str) -> list[SharedFact]:
        """把还没给过 `actor_id` 的事实取走，并记为已投递。

        **取走即已投递**，不重复给：同一件事在一个 NPC 的记忆里出现两次，
        会让检索时它被排到最前面，于是这个 NPC 会反复提同一件事 ——
        和"记忆被反复回引"是同一个病。
        """
        known = self._delivered.setdefault(actor_id, set())
        out = [f for fid, f in self._facts.items() if fid not in known]
        for fact in out:
            known.add(fact.fact_id)
        self.delivered_count += len(out)
        return out

    def known_by(self, actor_id: str) -> set[str]:
        return set(self._delivered.get(actor_id, ()))

    def facts(self) -> list[SharedFact]:
        return list(self._facts.values())

    def render(self) -> str:
        if not self._facts:
            return "（还没有任何事实进入网络）"
        lines = [f"网络里有 {len(self._facts)} 条事实，投递 {self.delivered_count} 次"]
        for fact in self._facts.values():
            lines.append(f"  [{fact.source_name}@t{fact.tick}] {fact.content}")
        return "\n".join(lines)


__all__ = [
    "GossipNetwork",
    "SharedFact",
    "fact_key",
    "preference_clause",
    "to_third_person",
]
