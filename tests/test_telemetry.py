"""成本与延迟遥测的护栏。

## 这里在守什么

米哈游公开方案把「千万级并发与成本」列为工程三关。遥测要守住三件事：

1. **真实计量**：LLM 响应里的 usage 必须被累计 —— 包括解析失败的那次
   （失败重试同样烧钱，漏记会让成本画像系统性偏低）；
2. **两种延迟口径不混**：NPC 轮延迟（机器够不够）≠ 玩家口径延迟
   （体验好不好），取平均数会骗人，长尾才是体感；
3. **离线地板价**：NullLLM 路径 token 恒为 0 —— 这个 0 是有用的读数
   （延迟里多少是框架开销、多少是模型的），不能被污染。
"""

from __future__ import annotations

import pytest

from npc_agent.llm.openai_compat import OpenAICompatLLM
from npc_agent.modules.telemetry import (
    CostProfile,
    LatencyStats,
    SessionTelemetry,
    TokenUsage,
)


class TestTokenUsage:
    def test_accumulates(self):
        usage = TokenUsage()
        usage.add(100, 50)
        usage.add(30, 20)
        assert usage.calls == 2
        assert usage.prompt_tokens == 130
        assert usage.completion_tokens == 70
        assert usage.total_tokens == 200

    def test_zero_cost_without_calls(self):
        assert TokenUsage().total_tokens == 0


class TestLatencyStats:
    def test_percentiles_on_sorted_samples(self):
        stats = LatencyStats()
        for ms in (10, 20, 30, 40, 100):
            stats.observe(ms)
        assert stats.count == 5
        assert stats.mean == pytest.approx(40.0)
        assert stats.p50 == 30
        assert stats.p95 == 100
        assert stats.peak == 100

    def test_empty_samples_are_zero_not_crash(self):
        stats = LatencyStats()
        assert stats.p50 == 0.0
        assert stats.p95 == 0.0
        assert stats.peak == 0.0
        assert "暂无" in stats.render()


class TestCostProfile:
    def test_cost_of(self):
        profile = CostProfile(price_per_m_input=2.0, price_per_m_output=8.0)
        usage = TokenUsage()
        usage.add(1_000_000, 1_000_000)
        assert profile.cost_of(usage) == pytest.approx(10.0)

    def test_from_env_defaults_and_override(self, monkeypatch):
        monkeypatch.delenv("NPC_AGENT_PRICE_PER_M_INPUT", raising=False)
        monkeypatch.delenv("NPC_AGENT_PRICE_PER_M_OUTPUT", raising=False)
        monkeypatch.delenv("NPC_AGENT_COST_CURRENCY", raising=False)
        profile = CostProfile.from_env()
        assert profile.price_per_m_input == 2.0
        assert profile.currency == "元"

        monkeypatch.setenv("NPC_AGENT_PRICE_PER_M_INPUT", "1.5")
        monkeypatch.setenv("NPC_AGENT_COST_CURRENCY", "USD")
        assert CostProfile.from_env().price_per_m_input == 1.5
        assert CostProfile.from_env().currency == "USD"

    def test_invalid_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("NPC_AGENT_PRICE_PER_M_OUTPUT", "not-a-number")
        assert CostProfile.from_env().price_per_m_output == 8.0


class TestSessionTelemetry:
    def test_two_latency_kinds_and_round_extrapolation(self):
        telemetry = SessionTelemetry(
            CostProfile(price_per_m_input=2.0, price_per_m_output=8.0)
        )
        telemetry.note_turn(50.0, spoken=True)
        telemetry.note_turn(70.0, spoken=False)
        telemetry.note_round(150.0)
        telemetry.note_llm(500, 500)

        snap = telemetry.snapshot()
        assert snap["rounds"] == 1
        assert snap["turns"] == 2
        assert snap["spoken_turns"] == 1
        # 成本外推：一轮 0.005 元 → 每 100 轮 0.5 元
        assert snap["cost_per_100_rounds"] == pytest.approx(0.5)
        # 两种延迟口径分开
        assert snap["latency_p50_ms"] == 50.0
        assert snap["round_latency_p50_ms"] == 150.0

    def test_adopt_llm_usage_replaces_not_accumulates(self):
        telemetry = SessionTelemetry()
        telemetry.note_llm(100, 100)
        llm = OpenAICompatLLM()
        llm.usage.add(500, 250)
        llm.usage.add(100, 50)
        telemetry.adopt_llm_usage(llm)
        telemetry.adopt_llm_usage(llm)  # 再采用一次也不重复计费
        assert telemetry.usage.total_tokens == 900
        # 没有 usage 属性的 LLM（离线路径）→ 保持零读数
        telemetry.adopt_llm_usage(object())
        assert telemetry.usage.total_tokens == 900

    def test_offline_floor_render(self):
        telemetry = SessionTelemetry()
        telemetry.note_round(1.0)
        assert "成本 0（离线路径不调模型）" in telemetry.render()


class TestOpenAICompatUsage:
    def test_note_usage_ignores_malformed(self):
        llm = OpenAICompatLLM()
        llm._note_usage({})
        llm._note_usage({"usage": None})
        llm._note_usage({"usage": "garbage"})
        llm._note_usage({"usage": {"prompt_tokens": "x", "completion_tokens": "y"}})
        assert llm.usage.calls == 0

    def test_note_usage_counts_zero_token_response(self):
        # 有些网关 200 但 usage 全 0 —— 一次调用必须记一次
        llm = OpenAICompatLLM()
        llm._note_usage({"usage": {"prompt_tokens": 0, "completion_tokens": 0}})
        assert llm.usage.calls == 1
        assert llm.usage.total_tokens == 0
