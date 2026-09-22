"""验证 `tests/test_dialogue_coherence.py` 真的会红 —— 把根因一个个放回去。

## 为什么需要这个脚本

`tests/test_dialogue_coherence.py` 里那几条断言是**回归钉子**，钉的是
2026-09-22 用户实测到的那段双 NPC 对话。但"断言写了"和"断言会红"是两件事：
一条永远绿的断言比没有断言更糟，因为它会让人以为覆盖到了。

所以这里做**变异测试的镜像**：不是往系统里注入新缺陷，而是**把修好的根因放回去**，
看对应的断言是否变红。红不了的那条，就是装饰。

实测结论（写进测试文件的 docstring 了）：

    变异 1（回应闸门收回成 `_player_is_asking_me`）
        ⇒ 第 1 轮台词变回 `第一次来吧？我请你一杯，想喝什么？` ⇒ A3 红 ✓
    变异 4（`_unfilled_order` 恒假）
        ⇒ 第 4 轮台词变成 `你上次说过你是第一次来。` ⇒ A4 红 ✓
    A1 / A5 在两次变异下都没红 —— 它们是**不许退步**的护栏，不是回归钉子。

用法：
    python scripts/verify_dialogue_coherence.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from npc_agent.agent import NPCAgent  # noqa: E402
from npc_agent.cast import build_cast  # noqa: E402
from npc_agent.config import RuntimeConfig, load_scenario  # noqa: E402
from npc_agent.llm import build_llm  # noqa: E402
import test_dialogue_coherence as T  # noqa: E402


def replay(setup: Callable[[], Callable[[], None]]):
    """跑一遍用户报的那段对话，`setup()` 返回撤销函数。"""
    cfg = replace(RuntimeConfig(), use_llm_planner=False, use_llm_speech=False)
    cast = build_cast(load_scenario("duet"), build_llm("null"), cfg)
    undo = setup()
    rounds: list[T.Round] = []
    try:
        for speaker, text in T.REPORTED_ROUNDS:
            current = T.Round(text)
            utterance = cast.env.record_player_utterance(speaker, text)
            for turn in cast.step(utterance):
                if turn.say:
                    current.speeches.append((turn.actor_id, turn.say))
                current.memories[turn.actor_id] = list(turn.used_memories)
            rounds.append(current)
    finally:
        undo()
    return cast, rounds


def check(cast, rounds) -> tuple[dict[str, bool], list[str]]:
    """跑四条断言，返回（断言名 -> 是否通过, 全部台词）。"""
    openings = T.templates_of(cast, "greet_new")
    declines = T.templates_of(cast, "unavailable_order")
    lines = [s for r in rounds for s in r.lines]

    every_round = all(r.speeches for r in rounds)
    no_canned = not [h for r in rounds for h in T.canned_opening_hits(r.lines, openings)]
    declined = bool(T.decline_hits(rounds[-1].lines, declines))

    earlier: set[str] = set()
    chain = True
    for index, r in enumerate(rounds):
        if not r.used or (index and not (r.used & earlier)):
            chain = False
        earlier |= r.used

    return (
        {
            "A1 每轮有人接话": every_round,
            "A3 不用计划开场白顶掉回应": no_canned,
            "A4 菜单外的那杯要明说做不了": declined,
            "A5 记忆链跨轮": chain,
        },
        lines,
    )


def no_mutation() -> Callable[[], None]:
    return lambda: None


def old_reply_gate() -> Callable[[], None]:
    """症状 1 的根因：回应闸门只认「提问 / 被点名」。"""
    original = NPCAgent._player_wants_reply
    NPCAgent._player_wants_reply = lambda self, u: self._player_is_asking_me(u)

    def undo() -> None:
        NPCAgent._player_wants_reply = original

    return undo


def order_never_answered() -> Callable[[], None]:
    """症状 4 的根因：计划接不住的下单**没有任何地方接**。"""
    original = NPCAgent._unfilled_order
    NPCAgent._unfilled_order = lambda self, u: False

    def undo() -> None:
        NPCAgent._unfilled_order = original

    return undo


def main() -> int:
    failures = 0
    for title, setup, expect_red in (
        ("基线（修好的代码）", no_mutation, frozenset()),
        ("变异 1：把回应闸门收回成 `_player_is_asking_me`", old_reply_gate, {"A3 不用计划开场白顶掉回应"}),
        ("变异 4：让 `_unfilled_order` 恒为假（下单没人接）", order_never_answered, {"A4 菜单外的那杯要明说做不了"}),
    ):
        cast, rounds = replay(setup)
        result, lines = check(cast, rounds)
        print(f"\n=== {title} ===")
        for name, ok in result.items():
            mark = "PASS" if ok else "FAIL"
            print(f"   {mark}  {name}")
        for line in lines:
            print(f"      台词：{line!r}")

        red = {name for name, ok in result.items() if not ok}
        if red != expect_red:
            failures += 1
            print(f"   ⚠️ 预期变红的断言是 {sorted(expect_red)}，实际 {sorted(red)}")

    print()
    if failures:
        print(f"{failures} 处与预期不符 —— 这个脚本自己也需要修")
        return 1
    print("全部符合预期：回归钉子会红，不许退步的护栏不红")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
