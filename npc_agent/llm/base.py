"""LLM 基类与容错 JSON 解析。"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any


class LLMUnavailable(RuntimeError):
    """模型不可用。调用方应当回退到启发式路径，而不是崩溃。"""


class LLM(ABC):
    """所有模型后端的最小接口。"""

    name: str = "llm"

    @property
    def available(self) -> bool:
        return True

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """给定 chat messages，返回文本。"""

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        schema_hint: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """要求模型输出 JSON 并容错解析。解析失败返回空 dict，由调用方回退。"""
        if schema_hint:
            messages = [
                *messages,
                {"role": "system", "content": f"只输出 JSON，不要任何解释。字段：{schema_hint}"},
            ]
        raw = self.complete(messages, **kwargs)
        return extract_json(raw)


def extract_json(text: str) -> dict[str, Any]:
    """从模型输出里尽力抠出第一个完整 JSON 对象。

    处理三种常见情况：裸 JSON、```json 围栏、前后带解释文字。
    用括号配对而不是正则，避免嵌套对象被截断。
    """
    if not text:
        return {}

    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fenced:
        text = fenced.group(1)

    start = text.find("{")
    if start == -1:
        return {}

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return {}
                return parsed if isinstance(parsed, dict) else {}
    return {}
