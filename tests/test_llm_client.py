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
