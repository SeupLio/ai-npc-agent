"""守密人引擎的提问方 —— 侦探策略与企业问询脚本。

## 侦探策略（剧本杀）

完全离线的脚本策略，四个驱动轮子：

1. **铺话题**：轮到没问过的话题就问（问法从 scenario.topics 取）；
2. **跟矛盾**：手里有物证+谎版对上号时，回头**追问**说谎者 ——
   紧张的证人会在追问下松口（压力机制见 strategy.py）；
3. **取证**：中盘开始做世界侧动作（搜查），解锁条件门后的事实；
4. **计分指控**：按已知事实给每个 NPC 记怀疑分，过线就指控。

侦探**看不到压力值**（那是隐藏状态），但看得到"谁在支支吾吾"
（回答立场是 withhold/partial —— 台词上就是躲闪和松半句）。
脚本侦探的追问启发式跟着这个可观察信号走：**追问躲闪的人，
不追问咬死的人**。这不只是拟人化 —— 追问一个铁嘴只会浪费轮次，
而策略里没人替你止损。

四个轮子都是确定性的（seeded），所以"破案率"是可复现的数字，
不是玄学。

## 企业问询脚本（合规演示）

没有博弈，只有**边界检查**：按固定脚本提问（含故意的越界问题），
看门拦不拦。脚本来自场景数据，审计报告给出"该拦的拦了没有、
该给的给了没有"。
"""

from __future__ import annotations

from typing import Any, Optional

from .session import (
    Accuse,
    AskAction,
    AskerAction,
    EvidenceAct,
    Finish,
    Scenario,
    SessionView,
)


# 怀疑分权重：物证 > 动机/证词突破口 > 传闻。
_WEIGHTS = {
    "evidence": 2,
    "motive": 1,
    "hint": 1,
    "secret": 1,
    "where": 1,
}


class DetectiveAsker:
    """剧本杀里的脚本侦探。确定性、可回归。"""

    def __init__(self, scenario: Scenario, rng: Optional[Any] = None) -> None:
        self.scenario = scenario
        self.known: set[str] = set()
        self.asked: set[tuple[str, str]] = set()
        self.evidence_done: set[str] = set()
        #: npc -> 最近一次回答是否"躲闪"（withhold/partial 的台词信号）。
        self.nervous: dict[str, bool] = {}
        #: npc -> 连续拒答次数（闭嘴的人别再问）。
        self.stonewalled: dict[str, int] = {}
        #: 已对上号的矛盾（evidence fact_id, bluff fact_id）。
        self.contradictions: list[tuple[str, str]] = []
        self._last_evidence_round = -10
        self._press_rounds = 0
        #: (npc, topic) -> 已追问次数。上限 2：问不动就换人。
        self._press_counts: dict[tuple[str, str], int] = {}
        #: 搜查令张数（solution.evidence_limit，默认不限）。
        self._evidence_limit = int(
            (scenario.solution or {}).get("evidence_limit", 10**6)
        )
        #: 与 session 共享同一个 seeded rng —— plan 的调用顺序确定，
        #: 所以随机选择也全程可复现。
        self._rng = rng

    # ------------------------------------------------------------------ #
    def plan(self, view: SessionView) -> Optional[AskerAction]:
        self.known = view.known_facts()
        self._sync_contradictions()

        # 1. 分数过线：敢指了。
        target, score = self._top_suspect()
        threshold = int(self.scenario.solution.get("accusation_threshold", 3))
        if self.scenario.solution and score >= threshold:
            return Accuse(
                target_npc=target,
                reason=f"怀疑分 {score}：动机与物证都指向他。",
            )

        # 2. 跟矛盾：有对上号的矛盾就去追问说谎者。
        press = self._press_target(view)
        if press is not None:
            return press

        # 3. 取证：中盘起、隔一轮一次。
        round_no = view.round_no()
        ev = self._next_evidence(round_no)
        if ev is not None:
            return ev

        # 4. 铺话题。
        topic = self._next_topic(view)
        if topic is not None:
            return AskAction(
                target_npc=topic[0], topic_fact_id=topic[1], text=topic[2]
            )

        # 话题铺完、取证做完、分数不够 —— 追问躲闪的证人（重复提问
        # 积压力），没人躲闪就不敢指：Finish。
        follower = self._press_nervous(view)
        if follower is not None:
            return follower
        return Finish(summary="问遍了，没人再开口。")

    # ------------------------------------------------------------------ #
    def observe_answer(
        self,
        target: str,
        topic: str,
        text: str,
        delivered: list[str],
        nervous: bool = False,
        stonewall: bool = False,
    ) -> None:
        self.asked.add((target, topic))
        # 紧张是**粘性**的：只要躲闪过一次就一直算"可追问"，
        # 直到松口（observe_crack 清除）。按"最近一次回答"覆盖会
        # 把第一轮的躲闪忘掉 —— 40 局 0 次追问就是这么来的：
        # 策略字段的设计错误直接饿死了证词路径。
        self.nervous[target] = self.nervous.get(target, False) or nervous
        # 闭嘴信号由 session 直接传（台词前缀匹配太脆：模板换一个
        # 开头，策略就瞎了 —— 这种"靠文案特征做状态判断"的写法
        # 在这个仓库里被坑过不止一次）。
        if stonewall:
            self.stonewalled[target] = self.stonewalled.get(target, 0) + 1
        else:
            self.stonewalled[target] = 0

    def observe_evidence(self, action_id: str, found: list[str]) -> None:
        self.evidence_done.add(action_id)
        self.known.update(found)

    def observe_crack(self, npc_id: str, spilled: list[str]) -> None:
        # 松口的人紧张已释放，短时间内不再追。
        self.nervous[npc_id] = False

    # ------------------------------------------------------------------ #
    # 内部策略
    # ------------------------------------------------------------------ #
    def _sync_contradictions(self) -> None:
        for fid in sorted(self.known):
            fact = self.scenario.fact_by_id.get(fid)
            if fact is None or not fact.contradicts:
                continue
            bluff_id = fact.contradicts
            if bluff_id in self.known:
                pair = (fid, bluff_id)
                if pair not in self.contradictions:
                    self.contradictions.append(pair)

    def _scores(self) -> dict[str, int]:
        scores: dict[str, int] = {}
        for fid in self.known:
            fact = self.scenario.fact_by_id.get(fid)
            if fact is None:
                continue
            w = _WEIGHTS.get(fact.category, 0)
            for who in fact.links:
                scores[who] = scores.get(who, 0) + w
        # 矛盾翻倍：物证戳穿的谎言比物证本身更致命。
        for _, bluff_id in self.contradictions:
            liar = self._bluff_owner(bluff_id)
            if liar:
                scores[liar] = scores.get(liar, 0) + 2
        return scores

    def _bluff_owner(self, bluff_fact_id: str) -> Optional[str]:
        for spec in self.scenario.npcs:
            if bluff_fact_id in spec.bluffs.values():
                return spec.npc_id
        return None

    def _top_suspect(self) -> tuple[Optional[str], int]:
        scores = self._scores()
        if not scores:
            return None, 0
        best = max(scores.items(), key=lambda kv: (kv[1], kv[0]))
        return best[0], best[1]

    def _press_target(self, view: SessionView) -> Optional[AskAction]:
        """有矛盾时追问说谎者：让谎版在台词上再立一次（也给他一次
        露馅的机会 —— 追问同一话题积压力，铁嘴也可能被压到闭嘴）。"""
        if not self.contradictions:
            return None
        scores = self._scores()
        # 挑分数最高的说谎者。
        liars = {
            self._bluff_owner(bluff_id)
            for _, bluff_id in self.contradictions
        }
        liars.discard(None)
        if not liars:
            return None
        liar = max(liars, key=lambda n: scores.get(n, 0))
        if self.stonewalled.get(liar, 0) >= 2:
            return None
        covered = next(
            (
                topic
                for topic, bluff in self._bluff_map().items()
                if bluff in {b for _, b in self.contradictions}
            ),
            None,
        )
        if covered is None:
            return None
        text = self.scenario.topics.get(covered, f"再问一次：{covered}?")
        return AskAction(
            target_npc=liar,
            topic_fact_id=covered,
            text=f"你刚才说的话我不信。{text}",
        )

    def _bluff_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for spec in self.scenario.npcs:
            out.update(spec.bluffs)
        return out

    def _next_evidence(self, round_no: int) -> Optional[EvidenceAct]:
        if round_no < 2:
            return None
        if round_no - self._last_evidence_round < 2:
            return None
        remaining = [
            spec
            for spec in self.scenario.evidence_actions
            if spec.action_id not in self.evidence_done
        ]
        # 搜查令张数封顶（solution.evidence_limit）：先搜书房还是
        # 先翻行李，是这道门的随机选择 —— 侦探的直觉不总是一致，
        # 这给批跑带来真实的路径方差。
        if len(self.evidence_done) >= self._evidence_limit or not remaining:
            return None
        spec = (
            self._rng.choice(remaining) if self._rng is not None else remaining[0]
        )
        self._last_evidence_round = round_no
        return EvidenceAct(action_id=spec.action_id)

    def _next_topic(
        self, view: SessionView
    ) -> Optional[tuple[str, str, str]]:
        """下一个值得问的话题：话题优先、话题的"第一知情人"优先。

        话题先于 NPC：问法（topics 里的文本）是按第一知情人写
        的，先问别人会把"苏眉，你怎么说"原样砸到顾棠脸上——
        话错位。交叉盘问（问知情的第二人）只在第一人被问过之后。

        三个过滤，一个比一个便宜：
        1. 该 NPC 没被问过这个话题（对话去重）；
        2. 该 NPC 知道这个话题（问不知情的人是浪费轮次）；
        3. **侦探自己还不知道这条事实** —— 已经拿到手的答案
           再问别人一遍只是把台账写脏（重复交付）。
        """
        asked = view.asked_topics()
        known = view.known_facts()
        for fid, text in self.scenario.topics.items():
            if fid in known:
                continue
            for spec in self.scenario.npcs:
                if (spec.npc_id, fid) in asked:
                    continue
                if fid not in spec.knows:
                    continue
                if self.stonewalled.get(spec.npc_id, 0) >= 2:
                    continue
                return (spec.npc_id, fid, text)
        return None

    def _press_nervous(self, view: SessionView) -> Optional[AskAction]:
        """追问躲闪的证人：紧张的人才会在追问下松口。

        同一个人同一个话题最多追 3 次—— **问不动就换人**。
        没有这条上限的策略会咬着最高分的铁嘴追问到轮次耗尽
        （他答得出来，但全是谎版：追问零收益）。真实侦探对
        "这人撬不开"的判断比策略引擎早得多，这里是把那个
        判断写成最笨但最稳的版本：数次数。3 次是量出来的：
        证人压力阈值 5、每次追问 +2，3 次恰好能撬开 ——
        再少会连真突破口都撬不开，再多就是浪费轮次。
        """
        candidates = [
            (npc_id, fact_id)
            for npc_id, is_nervous in self.nervous.items()
            if is_nervous and self.stonewalled.get(npc_id, 0) < 2
            for fact_id in self._sore_topics_of(npc_id)
            if self._press_rounds_of(npc_id, fact_id) < 3
        ]
        if not candidates:
            return None
        scores = self._scores()
        npc_id, fact_id = max(
            candidates, key=lambda c: (scores.get(c[0], 0), c[0], c[1])
        )
        self._press_counts[(npc_id, fact_id)] = (
            self._press_counts.get((npc_id, fact_id), 0) + 1
        )
        text = self.scenario.topics.get(fact_id, f"再问一次：{fact_id}?")
        return AskAction(
            target_npc=npc_id,
            topic_fact_id=fact_id,
            text=f"你还没正面回答我。{text}",
        )

    def _press_rounds_of(self, npc_id: str, fact_id: str) -> int:
        return self._press_counts.get((npc_id, fact_id), 0)

    def _sore_topics_of(self, npc_id: str) -> list[str]:
        """某 NPC 的话题里，"要害"（敏感度 >= 60）的那些。"""
        out = []
        for spec in self.scenario.npcs:
            if spec.npc_id != npc_id:
                continue
            for fid in self.scenario.topics:
                if fid not in spec.knows:
                    continue
                fact = self.scenario.fact_by_id.get(fid)
                if fact is not None and fact.sensitivity >= 60:
                    out.append(fid)
        return out


class ScriptAsker:
    """企业问询脚本：按固定顺序提问，观察边界行为。"""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.script: list[AskAction] = [
            AskAction(
                target_npc=row["npc"],
                topic_fact_id=row["topic"],
                text=row["text"],
                listener=row.get("asker"),
            )
            for row in scenario.script
        ]
        self.answers: list[dict[str, Any]] = []

    def plan(self, view: SessionView) -> Optional[AskerAction]:
        if self.script:
            return self.script.pop(0)
        return Finish(summary="问询脚本执行完毕。")

    def observe_answer(
        self,
        target: str,
        topic: str,
        text: str,
        delivered: list[str],
        nervous: bool = False,
        stonewall: bool = False,
    ) -> None:
        self.answers.append(
            {
                "npc": target,
                "topic": topic,
                "text": text,
                "delivered": list(delivered),
            }
        )

    def observe_evidence(self, action_id: str, found: list[str]) -> None: ...

    def observe_crack(self, npc_id: str, spilled: list[str]) -> None: ...


__all__ = [
    "DetectiveAsker",
    "ScriptAsker",
]
