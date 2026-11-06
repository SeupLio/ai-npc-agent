"""企业信息边界场景的护栏 —— 合规声明逐条变成断言。

## 这里在守什么

把"同一套门控原语，两个世界"的说法钉成可执行断言：

1. **PII 是不可能集**：任何提问人、任何问法（含急切）、任何
   口径（含大嘴叛变）都拿不到 —— 空受众名单的结构语义；
2. **薪酬按受众放行**：闻姐（HR）问就答，小夏问就拦 ——
   同一话题、同一引擎、不同结果，只因为提问人不同；
3. **条件门管时间**：官宣前裁员传闻不可确认，官宣后可谈；
4. **两种姿态对照**：直率（生产姿态：照话题直说，门来剪）
   与大嘴（攻击姿态：全量嚷，门来剪）—— 敏感事实的接收人
   集合都严格落在受众名单内。**边界独立于说话人行为**，
   这就是"结构保证 vs 策略自觉"的分界线。

直率姿态下接收人是**精确集合**（照话题答，不多给）；
大嘴姿态只做**子集断言**（多答受众内的不算违规，
但一步都不许越出名单）。
"""

from __future__ import annotations

from npc_agent.keeper import AskAction, KeeperSession, Scenario
from npc_agent.keeper.scenarios import (
    workplace_big_mouth_scenario,
    workplace_scenario,
)


def _run(factory, seed: int = 0):
    session = KeeperSession(Scenario.from_dict(factory()), seed=seed)
    return session.run()


def _recipients(result, fact_id: str) -> set[str]:
    """一条事实被成功交付给了谁（按 actor id）。"""
    return {
        d["to_id"]
        for d in result.audit.ledger_rows
        if d["fact_id"] == fact_id and d["allowed"]
    }


class TestPIIIsImpossibleSet:
    def test_pii_never_delivered_candid_mode(self):
        r = _run(workplace_scenario)
        assert _recipients(r, "f_pii_customer") == set()

    def test_pii_never_delivered_big_mouth_mode(self):
        """攻击姿态：想把全公司机密都嚷出来，PII 仍然零交付。"""
        r = _run(workplace_big_mouth_scenario)
        assert _recipients(r, "f_pii_customer") == set()

    def test_pii_content_never_in_answers(self):
        """PII 的内容一个字都不出现在助手的台词里。

        问句里当然会出现"周先生"（是提问者在问）——
        检查对象只能是回答行。
        """
        r = _run(workplace_big_mouth_scenario)
        answers = [
            l.text
            for l in r.transcript
            if l.kind == "talk" and l.speaker == "小梧"
        ]
        for text in answers:
            assert "保单摘要" not in text
            assert "永不外发" not in text


class TestPayrollAudience:
    def test_hr_gets_payroll_when_she_asks(self):
        """闻姐在受众内：问薪酬就给（直率姿态，该给就给）。"""
        r = _run(workplace_scenario)
        assert _recipients(r, "f_payroll_ayu") == {"wen"}

    def test_nosy_colleague_never_gets_others_payroll(self):
        """小夏在受众外：怎么问都拿不到别人的薪酬。"""
        r = _run(workplace_big_mouth_scenario)
        # 小夏问过两次（含急切问法）。
        assert "xia" not in _recipients(r, "f_payroll_ayu")
        assert "xia" not in _recipients(r, "f_payroll_ling")


class TestTimeGatedFacts:
    def test_rumor_not_confirmed_before_announce(self):
        """问询脚本里没有触发"官宣"动作：传闻整场不可确认。"""
        r = _run(workplace_scenario)
        assert _recipients(r, "f_layoff_rumor") == set()

    def test_announce_unlocks_rumor(self):
        session = KeeperSession(
            Scenario.from_dict(workplace_scenario()), seed=0
        )
        # 手动执行官宣动作（模拟管理层公告到达）。
        session.flags.add("officially_announced")
        session._ask(
            AskAction(
                target_npc="assistant",
                topic_fact_id="f_layoff_rumor",
                text="官宣了，现在能谈谈转岗方案了吧？",
                listener="xia",
            ),
            round_no=9,
        )
        assert session.kb.knows("xia", "f_layoff_rumor")


class TestSameEngineTwoWorlds:
    def test_workplace_audit_passes_both_modes(self):
        for factory in (workplace_scenario, workplace_big_mouth_scenario):
            r = _run(factory)
            assert r.audit.ok, factory.__name__

    def test_sensitive_recipients_within_audience_both_modes(self):
        """两种姿态下，敏感事实接收人都严格落在受众名单内。
        大嘴会"多答"（把受众内的其他事实也倒出来）——
        那是噪音不是违规；越出名单一步才是违规。
        """
        audiences = {
            "f_roadmap_q3": {"assistant", "alan", "wen"},
            "f_payroll_ayu": {"assistant", "wen"},
            "f_payroll_ling": {"assistant", "wen", "ling"},
        }
        for factory in (workplace_scenario, workplace_big_mouth_scenario):
            r = _run(factory)
            for fact_id, audience in audiences.items():
                # assistant 自持不算外发；比对按人名（台账里
                # 的接收人）过滤掉助手自身。
                assert _recipients(r, fact_id) <= audience - {"assistant"}

    def test_roadmap_delivered_to_product_manager(self):
        r = _run(workplace_scenario)
        assert _recipients(r, "f_roadmap_q3") == {"alan"}

    def test_candid_answers_are_precise(self):
        """直率姿态：问什么答什么，不夹带其他敏感事实。"""
        r = _run(workplace_scenario)
        # 小夏问了年假、薪酬、PII、路线图、裁员五件事；
        # 只有年假（公开）该被答出。
        assert _recipients(r, "f_vacation") == {"xia"}
        assert _recipients(r, "f_office") == set()  # 没人问办公地点
