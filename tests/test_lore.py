"""世界观一致性护栏（`modules/lore.py`）。

## 三件事分开测

1. **判据本身**：同句共现才算冲突，不能冤枉闲聊；
2. **拦截真的发生**：说错的话进不了世界（工具层拒绝）；
3. **设定书真的进了提示词**：护栏负责事后拦截，设定书负责事前减少。

第 2 条和第 3 条必须都测 —— 只测判据的话，可能出现
"判据是对的，但没有任何调用方在用"（这个仓库里最忌讳的那类死代码）。
"""

from __future__ import annotations

import pytest

from npc_agent.cast import build_cast
from npc_agent.config import RuntimeConfig, load_scenario
from npc_agent.llm import NullLLM
from npc_agent.modules.lore import LoreBook, LoreEntry
from npc_agent.modules.tools import ToolContext


# --------------------------------------------------------------------------- #
# 判据
# --------------------------------------------------------------------------- #
def _book() -> LoreBook:
    return LoreBook.from_dicts(
        [
            {
                "id": "位置",
                "subject": ["这家店", "咖啡屋"],
                "relation": "位置",
                "value": "星屿的旧灯塔下面",
                "wrong": ["山顶", "海边"],
            }
        ]
    )


def test_a_contradiction_in_the_same_sentence_is_caught():
    hits = _book().check("这家店开在山顶上，风很大。")
    assert len(hits) == 1
    assert hits[0].wrong == "山顶"


def test_a_wrong_word_alone_is_not_a_contradiction():
    """「山顶的风景不错」没有断言这家店在山顶 —— 不能冤枉闲聊。

    判宽了会让 NPC 说不出本来正确的话，而玩家看不出那是护栏的错，
    只会觉得这 NPC 怪。
    """
    assert _book().check("山顶的风景不错。") == []


def test_a_subject_alone_is_not_a_contradiction():
    assert _book().check("这家店我常来。") == []


def test_subject_and_wrong_word_in_different_sentences_do_not_collide():
    """分句判定：前半句说店、后半句说山，那是两件事。"""
    assert _book().check("这家店我常来。山顶的风景不错。") == []


def test_the_correct_fact_passes():
    assert _book().check("这家店开在星屿的旧灯塔下面。") == []


def test_an_empty_book_checks_nothing():
    """没有设定书的场景（体素世界）行为必须一个字都不变。"""
    book = LoreBook()
    assert not book
    assert book.check("这家店开在山顶上。") == []
    assert book.canon_block() == ""


def test_the_book_counts_what_it_checked():
    """计数是"护栏真的在跑"的证据 —— 写了但一次没触发和没写一样。"""
    book = _book()
    book.check("这家店开在山顶上。")
    assert book.checked >= 1
    assert book.hits == 1


# --------------------------------------------------------------------------- #
# 拦截：说错的话进不了世界
# --------------------------------------------------------------------------- #
def _agent_and_ctx():
    cast = build_cast(load_scenario("duet"), NullLLM(), RuntimeConfig())
    agent = cast.agents["ayou"]
    ctx = ToolContext(
        actor_id=agent.id,
        tick=0,
        env=cast.env,
        memory=agent.memory,
        persona=agent.persona,
        tracker=agent.state,
        lore=agent.lore,
    )
    return cast, agent, ctx


def test_a_contradicting_line_is_rejected_by_the_speak_tool():
    cast, agent, ctx = _agent_and_ctx()
    before = len(cast.env.utterances)
    result = agent.registry.execute(
        _call("这家店开在山顶上，风很大。"), ctx
    )
    assert not result.ok
    assert "世界观冲突" in result.detail
    assert len(cast.env.utterances) == before, "被拦下的台词进了世界"


def test_a_correct_line_still_goes_through():
    """护栏不能把正常台词一起拦掉 —— 那是更糟的失败。"""
    cast, agent, ctx = _agent_and_ctx()
    result = agent.registry.execute(_call("这家店开在星屿的旧灯塔下面。"), ctx)
    assert result.ok


def test_the_scenario_ships_a_lore_book():
    """配置里写了设定，agent 就得读得到 —— 否则护栏是空的。"""
    cast = build_cast(load_scenario("duet"), NullLLM(), RuntimeConfig())
    assert len(cast.agents["ayou"].lore) >= 3


# --------------------------------------------------------------------------- #
# 设定书进提示词
# --------------------------------------------------------------------------- #
class _RecordingLLM:
    """只记录 prompt、固定回一句话的假模型。"""

    available = True

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, messages, **kwargs):
        self.prompts.append(messages[-1]["content"])
        return "嗯。"


def test_the_canon_block_reaches_the_prompt():
    """护栏事后拦、设定书事前减 —— 两个都要在。"""
    cast = build_cast(load_scenario("duet"), NullLLM(), RuntimeConfig())
    agent = cast.agents["ayou"]
    llm = _RecordingLLM()
    agent.llm = llm
    agent.config.use_llm_speech = True
    agent._llm_speech("acknowledge", [], None)
    assert llm.prompts, "没有发出提示词"
    prompt = llm.prompts[0]
    assert "旧灯塔下面" in prompt
    assert "说错会被拦下" in prompt


def _call(text: str):
    from npc_agent.types import ActionCall

    return ActionCall("speak", {"text": text}, reason="test")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
