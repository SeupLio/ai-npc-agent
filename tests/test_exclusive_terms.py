"""专属称呼约定的护栏。

## 这里在守什么

星布谷地公开实录：玩家和娜洛约了亲昵称呼，别的玩家再来用，
她不认。这条护栏钉住"记忆的分寸"的四个面：

1. **抽取**：「以后就叫我小鹿吧」能抽出「小鹿」；语气词和商量句式
   不算称呼的一部分（实测踩过的两个坑，都有用例盯着）；
2. **排他**：一个叫法在这本账本里只属于一个人 —— 第二个人来约，
   得到的是拒绝，不是悄悄覆盖；
3. **当面说**：立约和拒绝都要说出口（模板），不能只是数据库里
   多一行；
4. **跨会话**：约定跟着存档走 —— 换一个进程再进来，排他仍然成立。
"""

from __future__ import annotations

import pytest

from npc_agent.cast import build_cast
from npc_agent.config import RuntimeConfig, load_scenario
from npc_agent.llm import NullLLM
from npc_agent.modules.persistence import load as persistence_load
from npc_agent.modules.persistence import save as persistence_save
from npc_agent.modules.relationship import RelationshipBook, extract_term_request


class TestExtraction:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("以后就叫我小鹿吧", "小鹿"),
            ("阿柚，以后就叫我小鹿吧", "小鹿"),
            ("也叫我小鹿呀", "小鹿"),
            ("叫我小鹿行不行", "小鹿"),
            ("可以叫我小甜心吗", "小甜心"),
            ("叫我一声", ""),
            ("叫我过去", ""),
            ("叫我什么？", ""),
            ("给我来杯拿铁", ""),
            ("今天天气不错", ""),
            ("你们这咖啡也太难喝了吧", ""),
        ],
    )
    def test_extract(self, text, expected):
        assert extract_term_request(text) == expected


class TestBook:
    def test_agreement_is_global_per_book(self):
        book = RelationshipBook()
        assert book.make_agreement("小鹿", "player_a", 1)
        # 同一个词不能再给第二个人
        assert not book.make_agreement("小鹿", "player_b", 2)
        # 换个词没问题
        assert book.make_agreement("小满", "player_b", 2)

    def test_owner_using_own_term_is_fine(self):
        book = RelationshipBook()
        book.make_agreement("小鹿", "player_a", 1)
        violated, owner = book.check_term("小鹿", "player_a")
        assert not violated and owner == ""
        # 别人用 → 拒绝，并说得出是谁的
        violated, owner = book.check_term("小鹿", "player_b")
        assert violated and owner == "player_a"
        # 没约定过的词 → 正常
        violated, owner = book.check_term("阿澈", "player_b")
        assert not violated

    def test_empty_term_rejected(self):
        book = RelationshipBook()
        assert not book.make_agreement("", "player_a", 1)
        assert not book.make_agreement("  ", "player_a", 1)


def _cast(scenario_id: str = "duet"):
    scenario = load_scenario(scenario_id)
    cfg = RuntimeConfig()
    cfg.llm_provider = "null"
    return build_cast(scenario, NullLLM(), cfg)


def _said(cast, pid, text):
    utterance = cast.env.record_player_utterance(pid, text)
    turns = cast.step(utterance)
    return [(cast.name_of(t.actor_id), t.say or "") for t in turns if t.say]


class TestIntegration:
    def test_accept_then_refuse_face_to_face(self):
        cast = _cast()
        # 阿澈跟阿柚约「小鹿」
        said = _said(cast, "player_a", "阿柚，以后就叫我小鹿吧")
        text = " ".join(s for _, s in said)
        assert "小鹿" in text
        assert any(k in text for k in ("好啊", "记下", "只应", "成"))
        book = cast.agents["ayou"].relationships
        assert any(a.term == "小鹿" for a in book.agreements)

        # 小满冒用（点名阿柚）→ 当面拒绝，且说得出是跟谁约的
        said = _said(cast, "player_b", "阿柚，也叫我小鹿吧")
        text = " ".join(s for _, s in said)
        assert any(k in text for k in ("不行", "有人用了", "换一个", "另起")), text

    def test_owner_still_gets_normal_replies(self):
        cast = _cast()
        _said(cast, "player_a", "阿柚，以后就叫我小鹿吧")
        # 约定之后，本人正常说话不应再触发拒绝
        said = _said(cast, "player_a", "今天有什么推荐？")
        text = " ".join(s for _, s in said)
        assert "不行" not in text and "有人用了" not in text

    def test_agreement_survives_save_load(self, tmp_path):
        cast = _cast()
        _said(cast, "player_a", "阿柚，以后就叫我小鹿吧")
        path = tmp_path / "save.json"
        persistence_save(cast, path)

        fresh = _cast()
        persistence_load(fresh, path)
        book = fresh.agents["ayou"].relationships
        assert any(a.term == "小鹿" and a.player_id == "player_a" for a in book.agreements)
        # 读档后的排他仍然生效
        said = _said(fresh, "player_b", "阿柚，也叫我小鹿吧")
        text = " ".join(s for _, s in said)
        assert any(k in text for k in ("不行", "有人用了", "换一个", "另起")), text

    def test_per_npc_books_do_not_leak(self):
        cast = _cast()
        # 阿澈只跟阿柚约了 —— 小舟的账本上没有这个词
        _said(cast, "player_a", "阿柚，以后就叫我小鹿吧")
        assert not cast.agents["xiaozhou"].relationships.agreements
