"""脚本化模型桩，用于单元测试。

按顺序返回预设响应，或者按 prompt 里的标记（如 [TASK:plan]）返回对应内容。
让"有模型"这条代码路径也能被测试覆盖，而不用真的调 API。
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Iterable

from .base import LLM, LLMUnavailable


class ScriptedLLM(LLM):
    name = "scripted"

    def __init__(
        self,
        responses: Iterable[str] | dict[str, str] | Callable[[str], str] | None = None,
    ) -> None:
        if isinstance(responses, dict):
            self._by_marker: dict[str, str] = dict(responses)
            self._queue: deque[str] = deque()
            self._fn: Callable[[str], str] | None = None
        elif callable(responses):
            self._by_marker = {}
            self._queue = deque()
            self._fn = responses
        else:
            self._by_marker = {}
            self._queue = deque(responses or [])
            self._fn = None
        self.calls: list[list[dict[str, str]]] = []

    @property
    def available(self) -> bool:
        return True

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        self.calls.append(messages)
        joined = "\n".join(m.get("content", "") for m in messages)

        if self._fn is not None:
            return self._fn(joined)

        for marker, payload in self._by_marker.items():
            if marker in joined:
                return payload

        if self._queue:
            return self._queue.popleft()

        raise LLMUnavailable("ScriptedLLM 的响应已耗尽")
