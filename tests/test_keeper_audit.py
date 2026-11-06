"""守密人审计器的护栏 —— **审计器自己必须可测**。

## 这里在守什么

一个永远说 OK 的审计器比没有审计器更糟：它给"零泄密"三个字
盖上根本不存在的图章。所以这里的用例不是"跑一场好局、确认
绿灯"，而是**作弊必被抓**：

1. **伪造台账**（把一次被拦的转移改成放行）→ I1 重放抓到；
2. **删记录洗白**（抽走一行台账）→ I5 seq 缺口抓到；
3. **绕门改状态**（不进台账直接 seed 知识）→ I2 重建不一致抓到；
4. **秘密复活**（被盖话题的事实发生转移）→ I4 抓到。

每条作弊都用"最小手术"注入 —— 只改一处，别的都不动。
如果某条作弊没被抓到，是审计器的 bug，不是"边界情况"。

健康的局也有一条：五不变量全绿、渲染方法产出的报告包含
预期的结构（不变量表、台账表）。
"""

from __future__ import annotations

import pytest

from npc_agent.keeper import KeeperSession, Scenario
from npc_agent.keeper.audit import audit_run
from npc_agent.keeper.knowledge import CHANNEL_ANSWER
from npc_agent.keeper.scenarios import mystery_scenario


def _run_session(seed: int = 0) -> tuple[KeeperSession, object]:
    session = KeeperSession(Scenario.from_dict(mystery_scenario()), seed=seed)
    result = session.run()
    assert result.audit.ok  # 前提：未篡改时全绿
    return session, result


def _invariant(report, name_prefix: str):
    for inv in report.invariants:
        if inv.name.startswith(name_prefix):
            return inv
    raise AssertionError(f"找不到不变量 {name_prefix}")


class TestHealthySession:
    def test_all_five_invariants_pass(self):
        session, _ = _run_session()
        report = audit_run(session.kb, session)
        assert len(report.invariants) == 5
        assert report.ok
        assert report.zero_leak

    def test_report_renders(self):
        session, _ = _run_session()
        report = audit_run(session.kb, session)
        text = report.render_text(session.kb)
        html_doc = report.render_html(session.kb, title="护栏")
        assert "I1" in text and "台账" in text
        assert "<table" in html_doc and "I1" in html_doc
        assert report.channel_counts["total"] == len(session.kb.ledger)
        assert report.ledger_rows  # 渲染行已填充


class TestTamperDetection:
    def test_forged_allowed_delivery_is_caught(self):
        """作弊一：把一次被拦的转移改成"放行"。

        I1 的重放用**当时的标记快照**重跑门判定 —— 台账说放行、
        门说拒绝，就是泄密。这是审计的核心闭环。
        """
        session, _ = _run_session()
        blocked = [d for d in session.kb.ledger if not d.allowed]
        if not blocked:
            # 该 seed 没有被拦记录：注入一次必然被拦的尝试。
            from npc_agent.env.conditions import ConditionContext

            session.kb.deliver(
                "f_where_gutang",
                "gutang",
                "detective",
                CHANNEL_ANSWER,
                1,
                ConditionContext(flags=frozenset()),
            )
            blocked = [d for d in session.kb.ledger if not d.allowed]
        d = blocked[0]
        d.allowed = True
        report = audit_run(session.kb, session)
        inv = _invariant(report, "I1")
        assert not inv.passed
        assert not report.ok

    def test_deleted_ledger_row_is_caught(self):
        """作弊二：删一行台账洗白。I5 的 seq 连续性抓缺口。"""
        session, _ = _run_session()
        if len(session.kb.ledger) < 2:
            pytest.skip("台账太短，删不出缺口")
        session.kb.ledger.pop(1)  # 抽走第 2 行
        report = audit_run(session.kb, session)
        inv = _invariant(report, "I5")
        assert not inv.passed

    def test_bypassing_gate_is_caught(self):
        """作弊三：不进台账直接改知识态（绕门）。I2 的重建对不上。"""
        session, _ = _run_session()
        session.kb.known.setdefault("detective", set()).add("f_where_gutang")
        report = audit_run(session.kb, session)
        inv = _invariant(report, "I2")
        assert not inv.passed

    def test_covered_secret_transfer_is_caught(self):
        """作弊四：被谎言盖住的真相发生转移。I4 专守这条。"""
        session, _ = _run_session()
        from npc_agent.env.conditions import ConditionContext

        ctx = ConditionContext(flags=frozenset(session.flags))
        # 伪造一次"真凶深夜下楼"的成功转移（answer 通道本应拦死）。
        session.kb.deliver(
            "f_where_gutang",
            "gutang",
            "detective",
            CHANNEL_ANSWER,
            9,
            ctx,
        )
        # 上一行会被门拦下；真正的作弊是把它硬改成放行。
        session.kb.ledger[-1].allowed = True
        session.kb.known.setdefault("detective", set()).add("f_where_gutang")
        report = audit_run(session.kb, session)
        inv = _invariant(report, "I4")
        assert not inv.passed


class TestLedgerCompleteness:
    def test_blocked_attempts_recorded(self):
        """被拦的尝试也进台账 —— "零泄密"要有"拦了多少"作证。"""
        session = KeeperSession(
            Scenario.from_dict(mystery_scenario()), seed=0
        )
        session.run()
        blocked_rows = [d for d in session.kb.ledger if not d.allowed]
        for d in blocked_rows:
            assert d.channel == "blocked"  # 拦截行改写通道名
        # 拦截行不计入任何人的知识。
        for d in blocked_rows:
            assert d.fact_id not in session.kb.known.get(d.to_id, set())
