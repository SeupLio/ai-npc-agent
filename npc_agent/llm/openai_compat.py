"""任意 OpenAI 兼容端点。

一套代码同时支持：
    OpenAI / DeepSeek / 通义千问(DashScope) / 月之暗面 / 智谱
    以及本地推理：vLLM / SGLang / Ollama / LM Studio

只要服务端实现 /chat/completions 即可。用 stdlib urllib，核心零依赖。

环境变量：
    NPC_AGENT_PROVIDER=openai-compat
    NPC_AGENT_BASE_URL=http://localhost:8000/v1
    NPC_AGENT_API_KEY=sk-xxx
    NPC_AGENT_MODEL=Qwen3-8B-Instruct
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Iterator

from .base import LLM, LLMUnavailable

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAICompatLLM(LLM):
    name = "openai-compat"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        # 读超时。**别调小。**
        #
        # 实测（kimi-k2.7-code）：平均 27s/次调用、规划约 35s。60s 会砍掉尾巴 ——
        # village 场景 6 次规划调用里 2 次 `TimeoutError`，而思维链长度是 0
        # （响应根本没回来）。框架把超时当成"规划失败"、静默回落启发式，
        # 于是**我们等得不够久**被记成了"模型规划得不好"。
        #
        # 这个默认值和 `RuntimeConfig.llm_timeout` 必须一致，有护栏钉住。
        timeout: float = 180.0,
    ) -> None:
        self.model = model or os.getenv("NPC_AGENT_MODEL", "")
        self.base_url = (
            base_url or os.getenv("NPC_AGENT_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.api_key = api_key or os.getenv("NPC_AGENT_API_KEY", "")
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.model)

    # ------------------------------------------------------------------ #
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, messages, temperature, max_tokens, stream=False) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        if not self.available:
            raise LLMUnavailable("未配置 NPC_AGENT_MODEL，无法调用模型")

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(self._payload(messages, temperature, max_tokens)).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # 4xx/5xx
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise LLMUnavailable(f"HTTP {exc.code}: {detail}") from exc
        except Exception as exc:  # 网络/超时
            raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc

        try:
            choice = body["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError) as exc:
            raise LLMUnavailable(f"响应结构异常: {body}") from exc

        content = (message.get("content") or "").strip()
        finish = choice.get("finish_reason")

        # 被 max_tokens 截断的**半句话**比模板台词更糟：
        # 玩家会看到"是啊，阳光都"这种说到一半就没了的台词。
        # 宁可显式报错让上层退回完整模板，也不要把残句播出去。
        if content and finish == "length":
            raise LLMUnavailable(
                f"模型输出被 max_tokens 截断（已生成 {len(content)} 字，"
                f"可能是思维链吃掉了预算）：{content[:40]!r}"
            )

        if content:
            return content

        # 推理模型（Qwen3.x / Kimi / DeepSeek-V4 等）会先输出 reasoning_content，
        # 如果 max_tokens 给小了，思维链会把预算吃光，content 返回空字符串。
        # 这时必须显式报错让上层回退，而不是把空台词当成"模型说了空话"。
        reasoning = message.get("reasoning_content") or ""
        raise LLMUnavailable(
            f"模型返回空内容（finish_reason={finish}，思维链 {len(reasoning)} 字）。"
            f"推理模型需要更大的 max_tokens。"
        )

    # ------------------------------------------------------------------ #
    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Iterator[str]:
        """SSE 流式输出。用于把 NPC 的台词逐字推给客户端。"""
        if not self.available:
            raise LLMUnavailable("未配置 NPC_AGENT_MODEL，无法调用模型")

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(
                self._payload(messages, temperature, max_tokens, stream=True)
            ).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                piece = delta.get("content")
                if piece:
                    yield piece
