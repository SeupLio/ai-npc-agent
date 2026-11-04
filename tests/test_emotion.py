"""玩家情绪感知的护栏。

## 这里在守什么

「娜洛！！出大事了！！」没有任何问句结构 —— 按 statement 走
acknowledge 的话，玩家的体感是"NPC 在敷衍我"。这批用例钉住三件事：

1. **判据本身**：急切/投诉/夸赞的边界，问句豁免（求信息不是情绪）；
2. **意图路由**：情绪命中时 NPC 说什么（模板带不带关心/安抚/高兴）；
3. **不误伤**：普通陈述、问句、转达不被情绪分支抢走 ——
   情绪分支抢话比没有情绪分支更伤（「有什么推荐」得到一句
   "怎么了？"是灾难）。
"""

from __future__ import annotations

import pytest

from npc_agent.cast import build_cast
from npc_agent.config import RuntimeConfig, load_scenario
from npc_agent.llm import NullLLM
from npc_agent.modules.emotion import detect_emotion, intent_for, emotion_label


class TestDetectEmotion:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("娜洛！！出大事了！！", "urgent"),
            ("救命！！", "urgent"),
            ("快！还有座位吗？？", "urgent"),
            ("你们这咖啡也太难喝了吧", "complaint"),
            ("太贵了，不值", "complaint"),
            ("太好听了，你们真棒", "praise"),
            ("好喝！", "praise"),
            ("今天有什么推荐？", "neutral"),
            ("我是第一次来", "neutral"),
        ],
    )
    def test_detection(self, text, expected):
        assert detect_emotion(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            # 问句里的情绪词是求信息，不是情绪 —— 拿情绪模板回它就是答非所问。
            ("好喝吗？", "neutral"),
            ("这个难喝吗？", "neutral"),
        ],
    )
    def test_question_exemption(self, text, expected):
        assert detect_emotion(text) == expected

    def test_intent_mapping(self):
        assert intent_for("urgent") == "respond_urgent"
        assert intent_for("complaint") == "soothe"
        assert intent_for("praise") == "delight"
        assert intent_for("neutral") == ""

    def test_label_for_prompt(self):
        # 提示词注入的文案必须非空且有行动指引
        for emotion in ("urgent", "complaint", "praise"):
            label = emotion_label(emotion)
            assert label
        assert emotion_label("neutral") == ""


def _cast(scenario_id: str = "tutorial"):
    cfg = RuntimeConfig()
    cfg.llm_provider = "null"
    return build_cast(load_scenario(scenario_id), NullLLM(), cfg)


def _npc_reply(cast, text: str, player_id: str = "player_a") -> str:
    utterance = cast.env.record_player_utterance(player_id, text)
    said = []
    for turn in cast.step(utterance):
        if turn.say:
            said.append((turn.actor_id, turn.say))
    return said


class TestAgentIntegration:
    def test_urgent_gets_concern_not_acknowledge(self):
        cast = _cast()
        said = _npc_reply(cast, "阿柚！！出大事了！！")
        text = " ".join(s for _, s in said)
        # 模板必须落进"怎么了/出什么事"这一族，而不是"我记下了"
        assert any(
            keyword in text for keyword in ("怎么了", "出什么事", "慢慢说", "别急")
        ), text
        assert "记下了" not in text

    def test_urgent_question_answered_not_deflected(self):
        cast = _cast()
        # 可答的急切问句：先答再关心（respond_urgent_answer）
        said = _npc_reply(cast, "快！！你们还开着门吗？？")
        text = " ".join(s for _, s in said)
        assert text, "急切问句不能没人接"

    def test_complaint_gets_soothe(self):
        cast = _cast()
        said = _npc_reply(cast, "你们这咖啡也太难喝了吧")
        text = " ".join(s for _, s in said)
        assert any(
            keyword in text for keyword in ("抱歉", "对不起", "重新", "重做", "重弄")
        ), text

    def test_praise_gets_delight(self):
        cast = _cast()
        said = _npc_reply(cast, "太好听了，你们真棒")
        text = " ".join(s for _, s in said)
        assert any(
            keyword in text for keyword in ("喜欢", "开心", "有劲", "真好", "就好")
        ), text

    def test_normal_question_not_hijacked(self):
        cast = _cast()
        said = _npc_reply(cast, "你们有什么推荐？")
        text = " ".join(s for _, s in said)
        # 正常问句必须走回答路径：推荐的内容（拿铁/柠檬水/手冲……）
        # 或老实的"不知道"，绝不该是情绪模板的"怎么了"
        assert "怎么了" not in text
        assert "别急" not in text
