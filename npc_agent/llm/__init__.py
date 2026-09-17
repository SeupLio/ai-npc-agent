"""LLM 抽象层。

关键设计：模型是**可选的**。
- 有 API key / 本地 vLLM 时 → OpenAICompatLLM，走真实模型
- 什么都没有时       → NullLLM，各模块自动回退到内置启发式策略

这样保证仓库 clone 下来就能跑通（CI / 评测回归 / Demo 录屏），
也让"换模型"变成一行配置，而不是一次重构。
"""

from .base import LLM, LLMUnavailable, extract_json
from .null import NullLLM
from .openai_compat import OpenAICompatLLM
from .scripted import ScriptedLLM

__all__ = [
    "LLM",
    "LLMUnavailable",
    "extract_json",
    "NullLLM",
    "OpenAICompatLLM",
    "ScriptedLLM",
    "build_llm",
]


def build_llm(provider: str = "null", **kwargs) -> LLM:
    """按 provider 名构造 LLM 实例。

    `timeout` 只在显式给出时透传 —— 默认值留在 `OpenAICompatLLM` 里，
    免得两处默认值各写一遍、改一处漏一处。
    """
    provider = (provider or "null").lower()
    if provider in ("null", "none", "offline"):
        return NullLLM()
    if provider in ("openai", "openai-compat", "compat", "vllm", "ollama", "deepseek"):
        timeout = kwargs.get("timeout")
        extra = {"timeout": float(timeout)} if timeout else {}
        return OpenAICompatLLM(
            model=kwargs.get("model", ""),
            base_url=kwargs.get("base_url", ""),
            api_key=kwargs.get("api_key", ""),
            **extra,
        )
    if provider == "scripted":
        return ScriptedLLM(kwargs.get("responses", []))
    raise ValueError(f"未知的 LLM provider: {provider}")
