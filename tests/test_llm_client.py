"""OpenAI 兼容客户端的响应处理测试。

这里的核心问题是**推理模型**（Qwen3.x / Kimi / DeepSeek-V4）会先输出
``reasoning_content``，思维链和正式回答共用 ``max_tokens`` 预算。
于是出现两种坏情况：

1. 预算被思维链吃光 → ``content`` 是空字符串，``finish_reason=length``；
2. 预算在回答说到一半时用完 → ``content`` 有内容但被**截断**，
   ``finish_reason`` 同样是 ``length``。

第 2 种最阴险：不报错，直接把"是啊，阳光都"这种半句话播给玩家。
两种都必须显式报错，让上层退回完整模板。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from npc_agent.config import RuntimeConfig
from npc_agent.llm.base import LLMUnavailable
from npc_agent.llm.openai_compat import OpenAICompatLLM


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _client(monkeypatch, payload: dict[str, Any]) -> OpenAICompatLLM:
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: _FakeResponse(payload)
    )
    return OpenAICompatLLM(model="test-model", base_url="http://x/v1", api_key="k")


def _body(content: str, finish: str, reasoning: str = "") -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": finish,
                "message": {"content": content, "reasoning_content": reasoning},
            }
        ]
    }


# --------------------------------------------------------------------------- #
def test_normal_response_is_returned(monkeypatch):
    llm = _client(monkeypatch, _body("欢迎光临。", "stop"))
    assert llm.complete([{"role": "user", "content": "hi"}]) == "欢迎光临。"


def test_empty_content_with_length_raises(monkeypatch):
    """预算被思维链吃光 —— 必须报错，不能把空台词当成"模型说了空话"。"""
    llm = _client(monkeypatch, _body("", "length", reasoning="想" * 400))
    with pytest.raises(LLMUnavailable) as excinfo:
        llm.complete([{"role": "user", "content": "hi"}])
    assert "空内容" in str(excinfo.value)
    assert "max_tokens" in str(excinfo.value)


def test_truncated_content_with_length_raises(monkeypatch):
    """回归测试：截断的半句话曾被当成正常回答直接播出去。"""
    llm = _client(monkeypatch, _body("是啊，阳光都", "length"))
    with pytest.raises(LLMUnavailable) as excinfo:
        llm.complete([{"role": "user", "content": "hi"}])
    assert "截断" in str(excinfo.value)


def test_length_with_content_is_not_silently_accepted(monkeypatch):
    """哪怕内容看起来"完整"，只要 finish_reason=length 就不能信。"""
    llm = _client(monkeypatch, _body("这是一句看起来完整的话。", "length"))
    with pytest.raises(LLMUnavailable):
        llm.complete([{"role": "user", "content": "hi"}])


#: 实测：kimi-k2.7-code 在**台词**调用里的思维链长度（字符）。
#: 来源是 228 条真实跑批里的 7 条失败记录 —— 客户端在
#: `finish_reason=length` 且内容为空时，会把思维链长度写进错误信息。
OBSERVED_SPEECH_COT_CHARS = (3394, 3526, 3753, 3773, 3840, 3905)


def test_the_speech_budget_is_sized_for_a_reasoning_models_cot() -> None:
    """台词的预算必须容得下思维链 + 正式回答。

    这条测试锁的是一个**实测数字**，不是审美。台词预算给 1024 时，
    150 条里 7 条返回空内容 → 框架退回模板台词 → 那 7 条的分数
    衡量的就不是模型了，而报告里只会显示"通过率 95%"。

    下限直接跟着观测到的思维链长度走，而不是拍一个好看的整数 ——
    这样下次有人想把预算调小，测试会指出它是拿什么换来的。
    """
    longest = max(OBSERVED_SPEECH_COT_CHARS)
    budget = RuntimeConfig().speech_max_tokens
    assert budget >= longest, (
        f"台词预算 {budget} 小于实测最长思维链 {longest} 字："
        "推理模型会把预算吃光、返回空内容，用例静默退回模板台词"
    )
    assert budget >= 4096, (
        "4096 是同端点裁判能正常处理 4043 字思维链的实测值，"
        "台词调用没有理由给得比它更紧"
    )


def test_the_planner_budget_is_not_smaller_than_the_speech_budget() -> None:
    """规划和台词是两次独立调用，各吃各的预算。

    规划的思维链不会比台词短（它要想完整个计划），所以规划预算
    没有理由比台词预算小 —— 真小了就会出现"台词正常、规划静默退回启发式"
    这种最难查的组合。
    """
    config = RuntimeConfig()
    assert config.max_tokens >= config.speech_max_tokens


def test_available_requires_model(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FakeResponse({}))
    assert OpenAICompatLLM(model="m").available is True
    assert OpenAICompatLLM(model="").available is False


def test_unavailable_client_raises_without_calling(monkeypatch):
    llm = OpenAICompatLLM(model="")
    with pytest.raises(LLMUnavailable):
        llm.complete([{"role": "user", "content": "hi"}])


def test_malformed_body_raises(monkeypatch):
    llm = _client(monkeypatch, {"unexpected": True})
    with pytest.raises(LLMUnavailable) as excinfo:
        llm.complete([{"role": "user", "content": "hi"}])
    assert "响应结构异常" in str(excinfo.value)
