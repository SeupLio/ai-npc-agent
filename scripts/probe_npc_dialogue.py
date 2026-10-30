"""量一遍「NPC 和 NPC 到底能不能聊起来」。

跑法：

    python scripts/probe_npc_dialogue.py
    python scripts/probe_npc_dialogue.py --ticks 12 --json reports/npc_dialogue.json

## 为什么要单独量这一件事

六维评测里没有任何一维看「两个 NPC 之间有没有形成对话」——
`turn_taking` 只看「一个 tick 里是不是只有一个人开口」，
它**奖励沉默**：两个 NPC 全程不说话，`turn_taking` 是满分。

所以「NPC 之间聊不起来」可以在所有指标都满分的情况下存在。
这个探针量的就是那个空档，三个数：

    npc_lines        NPC 说了多少句（分母）
    cross_addressed  其中有多少句是**对另一个 NPC** 说的
    answered_rate    被点名的那个 NPC，下一轮有没有真的接话
    repeat_rate      整段里同一人重复自己的比例（`modules/repetition.py` 的口径）

前两个数回答「有没有对话」，第三个回答「对话推不推得动」，
第四个数回答「聊起来之后是不是在复读」。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npc_agent.cast import build_cast  # noqa: E402
from npc_agent.config import RuntimeConfig, load_scenario  # noqa: E402
from npc_agent.llm import NullLLM  # noqa: E402
from npc_agent.modules.repetition import find_repeats  # noqa: E402


def run(scenario_id: str, ticks: int, scripted: bool) -> dict:
    scenario = load_scenario(scenario_id)
    config = RuntimeConfig()
    cast = build_cast(scenario, NullLLM(), config)

    lines: list[dict] = []
    for i in range(ticks):
        # 前两轮由玩家各说一句，之后**完全放手**：没有人说话，
        # 场上只剩两个 NPC。这才是"NPC 之间沟通交流"的原始条件。
        utterance = None
        if scripted and i == 0:
            utterance = cast.env.record_player_utterance(
                "player_a", "今天这里挺热闹的，你们俩平时都聊些什么？"
            )
        elif scripted and i == 1:
            utterance = cast.env.record_player_utterance(
                "player_a", "阿柚，你让小舟弹一首呗？"
            )
        turns = cast.step(utterance)
        for u in cast.env.utterances:
            if u.tick == cast.env.tick - 1:
                lines.append(
                    {
                        "tick": u.tick,
                        "speaker": u.speaker_id,
                        "name": u.speaker_name,
                        "text": u.text,
                        "role": u.role,
                        "addressed_to": u.addressed_to,
                        "is_question": u.is_question,
                        "expects_reply": u.expects_reply,
                    }
                )
        for t in turns:
            if t.say and not any(
                l["tick"] == t.tick and l["speaker"] == t.actor_id and l["text"] == t.say
                for l in lines
            ):
                lines.append(
                    {
                        "tick": t.tick,
                        "speaker": t.actor_id,
                        "name": cast.name_of(t.actor_id),
                        "text": t.say,
                        "role": "npc",
                        "addressed_to": t.addressed_to,
                        "is_question": False,
                        "expects_reply": False,
                    }
                )

    npc_ids = set(cast.npc_ids)
    npc_lines = [l for l in lines if l["speaker"] in npc_ids]
    cross = [l for l in npc_lines if l["addressed_to"] in npc_ids]
    # 只有"要一个回应"的话才该被接 —— 「……还行。」说给阿柚听，
    # 但阿柚不需要再回一句。按"说给谁"统计会把正常对话算成漏接。
    owed = [l for l in cross if l["expects_reply"] or l["is_question"]]
    answered = 0
    for line in owed:
        nxt = [
            l
            for l in npc_lines
            if l["tick"] > line["tick"] and l["speaker"] == line["addressed_to"]
        ]
        if nxt and nxt[0]["tick"] == line["tick"] + 1:
            answered += 1

    per_speaker: dict[str, list[tuple[str, str]]] = {}
    for l in npc_lines:
        per_speaker.setdefault(l["speaker"], []).append((l["speaker"], l["text"]))
    repeat_lines = 0
    for speaker_lines in per_speaker.values():
        repeat_lines += len(find_repeats(speaker_lines).repeats)

    return {
        "scenario": scenario_id,
        "ticks": ticks,
        "npc_lines": len(npc_lines),
        "cross_addressed": len(cross),
        "owed": len(owed),
        "answered": answered,
        "answered_rate": (answered / len(owed)) if owed else 0.0,
        "repeat_lines": repeat_lines,
        "repeat_rate": (repeat_lines / len(npc_lines)) if npc_lines else 0.0,
        "transcript": lines,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="duet")
    ap.add_argument("--ticks", type=int, default=8)
    ap.add_argument("--no-scripted", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    result = run(args.scenario, args.ticks, not args.no_scripted)

    for line in result["transcript"]:
        to = f" →{line['addressed_to']}" if line["addressed_to"] else ""
        print(f"[t{line['tick']}] {line['name']}{to}: {line['text']}")

    print()
    print(f"NPC 台词        {result['npc_lines']} 句")
    print(f"其中对 NPC 说的  {result['cross_addressed']} 句")
    print(
        f"要求回应        {result['owed']} 句，被接住 {result['answered']} 次"
        f"（{result['answered_rate']:.0%}）"
    )
    print(f"复读            {result['repeat_lines']} 句（{result['repeat_rate']:.0%}）")

    if args.json:
        path = ROOT / args.json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
