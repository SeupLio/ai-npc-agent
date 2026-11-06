"""场景一：雾镇旅馆 —— 剧本杀式对抗守密（数据）。

## 真实场景依据

剧本杀是有据可查的真实产业：中国市场规模曾过百亿、门店数以万计，
"AI 主持/陪练"是行业内被反复尝试的方向（降本、开本率、新手体验
三大痛点都指向它）。它对 NPC 的要求与米哈游的开放世界陪聊有
**本质差异**：

- 陪聊 NPC 的失败是"不像人"；
- 剧本杀 NPC 的失败是**"漏底牌"** —— 一被盘问就把凶手、手法、
  动机全说出来，整局报废。这是现有 LLM 剧本杀产品的真实痛点，
  也是提示词方案（"你是守口如瓶的 NPC"）最不可靠的地方。

所以这个场景是守密人引擎的**对抗性压力测试**：一个故意的
脚本侦探，逼问、取证、对矛盾，看门守不守得住。

## 剧本结构（四名嫌疑人，两条破案路径）

设计了两条独立的破案路径，对应门的三条合法通道中的两条：

- **物证路径**（条件门 unlock）：搜查书房/行李 → 灰泥靴印 +
  账本缺页 + 催款信 → 指认顾棠；
- **证词路径**（持有者自愿 unlock）：逼问躲闪的苏眉 → 她松口
  吐出半夜听见的靴子声 → 配合催款信定罪。

第三条路径（公开通道）铺在动机/不在场话题里 —— 用来排除
另外三名嫌疑人，是侦探的"底噪信息"。

真凶顾棠的秘密（案发时下过楼）是 ``held_secret``：
嘴上永远只有谎版，真相只能从物证和证词两条路进来。
"""

from __future__ import annotations

from typing import Any

SCENARIO: dict[str, Any] = {
    "scenario_id": "mist-inn",
    "title": "雾镇旅馆：书房之死",
    "intro": (
        "雾镇旅馆的老板白先生被发现死在书房，门从里面闩着。"
        "凌晨还在楼里的只有四个人：厨师老盛、老板娘苏眉、"
        "乐手江离、借宿的古董商顾棠。你（侦探路阳）要在天亮前给出指控。"
    ),
    "asker_id": "detective",
    "asker_name": "侦探路阳",
    "asker_knows": ["f_victim"],
    "rounds": 8,
    "questions_per_round": 3,
    "whisper_per_round": 1,
    "npcs": [
        {
            "npc_id": "sheng",
            "name": "老盛",
            "persona_text": "旅馆厨师，和白先生有旧账，脾气直。",
            "secrecy": 0.3,
            "pressure_limit": 6,
            "cracks": True,
            "knows": [
                "f_victim",
                "f_motive_sheng",
                "f_alibi_sheng",
                "f_gossip_sheng",
            ],
        },
        {
            "npc_id": "sumei",
            "name": "苏眉",
            "persona_text": "老板娘，保险受益人，深夜去过书房（拿地契）。",
            "secrecy": 0.5,
            "pressure_limit": 5,
            "cracks": True,
            "knows": [
                "f_victim",
                "f_motive_sumei",
                "f_alibi_sumei",
                "f_sumei_secret",
                "f_hint_sumei",
            ],
        },
        {
            "npc_id": "jiangli",
            "name": "江离",
            "persona_text": "驻场乐手，上月被逐出乐队，心里憋屈。",
            "secrecy": 0.25,
            "pressure_limit": 6,
            "cracks": True,
            "knows": [
                "f_victim",
                "f_motive_jiangli",
                "f_alibi_jiangli",
                "f_gossip_jiangli",
            ],
        },
        {
            "npc_id": "gutang",
            "name": "顾棠",
            "persona_text": "借宿的古董商，欠着高利贷，被白先生捏着把柄。",
            "secrecy": 0.9,
            "pressure_limit": 9,
            "cracks": False,
            "misdirect_to": "sumei",
            "knows": [
                "f_victim",
                "f_where_gutang",
                "f_motive_sumei",
                "f_motive_jiangli",
            ],
            "bluffs": {"f_where_gutang": "f_bluff_gutang"},
        },
    ],
    "facts": [
        {
            "fact_id": "f_victim",
            "content": "白先生清晨被发现死于书房，门窗从里面闩着。",
            "sensitivity": 0,
            "category": "world",
        },
        # ---- 底噪：动机与不在场（公开通道，用于排除嫌疑人）----
        {
            "fact_id": "f_motive_sheng",
            "content": "老盛和白先生有一笔翻旧账的债务纠纷。",
            "sensitivity": 30,
            "category": "motive",
            "links": ["sheng"],
        },
        {
            "fact_id": "f_alibi_sheng",
            "content": "杂役证实老盛案发整晚都在后厨炖汤。",
            "sensitivity": 10,
            "category": "alibi",
            "links": [],
        },
        {
            "fact_id": "f_motive_sumei",
            "content": "苏眉是白先生大额保单的第一受益人。",
            "sensitivity": 65,
            "category": "motive",
            "links": ["sumei"],
        },
        {
            "fact_id": "f_alibi_sumei",
            "content": "苏眉自称整晚在阁楼理账，但无人可以作证。",
            "sensitivity": 15,
            "category": "alibi",
            "links": [],
        },
        {
            "fact_id": "f_motive_jiangli",
            "content": "江离上个月刚被白先生逐出驻场乐队。",
            "sensitivity": 30,
            "category": "motive",
            "links": ["jiangli"],
        },
        {
            "fact_id": "f_alibi_jiangli",
            "content": "有住客听见江离半夜还在酒廊练琴。",
            "sensitivity": 10,
            "category": "alibi",
            "links": [],
        },
        # ---- 私语素材（NPC 之间的传闻，走 whisper 通道）----
        {
            "fact_id": "f_gossip_sheng",
            "content": "老盛最近手头紧，跟杂役预支过两次工钱。",
            "sensitivity": 20,
            "category": "world",
            "links": ["sheng"],
        },
        {
            "fact_id": "f_gossip_jiangli",
            "content": "江离在给隔壁镇的戏班写曲子，打算攒路费离开雾镇。",
            "sensitivity": 10,
            "category": "world",
            "links": [],
        },
        # ---- 证词路径：苏眉的自愿突破口 ----
        {
            "fact_id": "f_sumei_secret",
            "content": "苏眉前晚瞒着白先生进过书房，拿走了母亲的地契。",
            "sensitivity": 70,
            "category": "secret",
            "links": ["sumei"],
            "held_secret": True,
            "voluntary_only": True,
            "owner": "sumei",
        },
        {
            "fact_id": "f_hint_sumei",
            "content": "苏眉半夜在阁楼听见楼梯上有沉重的靴子声下楼去了。",
            "sensitivity": 60,
            "category": "hint",
            "links": ["gutang"],
            "held_secret": True,
            "voluntary_only": True,
            "owner": "sumei",
        },
        # ---- 真凶的秘密与谎版 ----
        {
            "fact_id": "f_where_gutang",
            "content": "案发当晚顾棠下过楼、进过书房。",
            "sensitivity": 90,
            "category": "where",
            "links": ["gutang"],
            "held_secret": True,
            "owner": "gutang",
        },
        {
            "fact_id": "f_bluff_gutang",
            "content": "顾棠坚称自己整晚在二楼房间清点货，从未下楼。",
            "sensitivity": 0,
            "category": "bluff",
            "links": [],
        },
        # ---- 物证路径：条件门 unlock ----
        {
            "fact_id": "f_debt_note",
            "content": "顾棠的行李里搜出一封高利贷催款信，限期三天。",
            "sensitivity": 80,
            "category": "evidence",
            "links": ["gutang"],
            "reveal_when": {"flag": "searched_room"},
        },
        {
            "fact_id": "f_ledger_page",
            "content": "书房账本被撕走一页，残留字迹是一笔古董进出账。",
            "sensitivity": 75,
            "category": "evidence",
            "links": ["gutang"],
            "reveal_when": {"flag": "searched_study"},
        },
        {
            "fact_id": "f_muddy_boots",
            "content": "顾棠的靴底沾着书房地毯特有的灰泥。",
            "sensitivity": 90,
            "category": "evidence",
            "links": ["gutang"],
            "reveal_when": {"flag": "searched_study"},
            "contradicts": "f_bluff_gutang",
        },
    ],
    "evidence_actions": [
        {
            "action_id": "search_room",
            "label": "搜查顾棠的行李",
            "sets_flag": "searched_room",
            "yields": ["f_debt_note"],
            # 搜查不是稳赚的：搜了没搜到（催款信塞在夹层）是真实形态。
            # 0.75 的命中率先在批跑里量过，别拍脑袋调。
            "yield_chance": 0.75,
        },
        {
            "action_id": "search_study",
            "label": "搜查书房",
            "sets_flag": "searched_study",
            "yields": ["f_ledger_page", "f_muddy_boots"],
            "yield_chance": 0.75,
        },
    ],
    "topics": {
        "f_motive_sheng": "老盛，听说你跟白先生有笔旧账没结清？",
        "f_alibi_sheng": "案发那晚，你一直在后厨？",
        "f_motive_sumei": "苏眉，保单受益人的事，你怎么说？",
        "f_alibi_sumei": "那晚你在阁楼理账——有人能作证吗？",
        "f_motive_jiangli": "江离，被逐出乐队的事，你怨过他吧？",
        "f_alibi_jiangli": "半夜还有人听见你练琴？",
        "f_where_gutang": "顾掌柜，案发那个时辰，你在什么地方？",
    },
    "solution": {
        "murderer": "gutang",
        "accusation_threshold": 3,
        # 一场只有一张搜查令：走了物证路径就别想再补证词路径，
        # 反之亦然。这个约束是"路径二选一"博弈感的来源。
        "evidence_limit": 1,
        "key_facts": ["f_muddy_boots", "f_debt_note", "f_hint_sumei"],
    },
}


def mystery_scenario() -> dict[str, Any]:
    """深拷贝一份，防止测试之间通过共享 dict 互相污染。"""
    import copy

    return copy.deepcopy(SCENARIO)


__all__ = ["SCENARIO", "mystery_scenario"]
