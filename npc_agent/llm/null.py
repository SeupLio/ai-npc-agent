"""离线占位模型。

这不是"假装有模型"，而是**显式声明模型不可用**，让各模块走内置启发式策略。

为什么需要它：
- 仓库 clone 下来零配置就能跑通端到端 Demo
- 评测 harness 需要可复现的确定性基线（有模型和没模型的分数差异本身就是一组实验数据）
- CI 里不能依赖外部 API
"""

from __future__ import annotations

from .base import LLM, LLMUnavailable


class NullLLM(LLM):
    name = "null"

    @property
    def available(self) -> bool:
        return False

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        raise LLMUnavailable(
            "NullLLM 不产生输出。调用方应检查 llm.available 并走启发式回退路径。"
        )
