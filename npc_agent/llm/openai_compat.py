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
import time
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
        # 传输层重试次数（含首次）。**这不是"再试一次就好"的乐观，是有数的**：
        # 超时/预算修完之后，整臂 112/231 条（48%）至少回落一次，其中 **109 条是
        # 传输层故障** —— 请求压根没到模型那儿，重发是合理的。
        transient_attempts: int = 3,
        # 退避基数（秒）：第 n 次重试前等 `backoff * 2**(n-1)` ⇒ 1s、2s。
        # 端点真的挂掉时一次调用最坏 3×180s —— 那时本来也什么都拿不到。
        transient_backoff: float = 1.0,
        # 解析层重试次数（含首次）。**和上面那个是两回事**，所以是独立参数。
        #
        # 从前解析层（`finish_reason=length` / "空内容 + 长思维链"）**完全不在
        # 重试覆盖范围里** —— 它们是在 HTTP 200 返回**之后**由解析层抛的，
        # 而 `_attempt()` 只包住 `_post_json()`。那不是"我们决定不重试"，
        # 是"这一层结构上没被包住"。
        #
        # 现在包的**依据是量出来的**（`reports/_analyze_parse_retry.py`，0 次新调用）：
        #
        #   * 真实跑批（`eval_model_planner.json`，**生产预算 16384**）里
        #     解析层失败 **28** 条 = 空内容 25 + 被截断的半句话 3。
        #     ⇒ 附八"把预算从 4096 提到 16384 就从根上修好了"**并不成立**。
        #   * 那 25 条空内容失败的思维链长度：中位数 **16686**、均值 16503，
        #     **76%（19/25）落在离 16384 只差 400 字以内**。
        #     ⇒ 是"差一点点"，不是"差得远"。重试一次（思维链长度本身在抖）
        #     就有很大机会落回预算内。
        #   * 附九「二·补」已量出它是**抖动**而非必然：同 prompt 同预算各跑 10 次，
        #     64→失败 3、96→1、128→1。
        #
        # 成本：只在**已经失败之后**才重试，所以期望调用数是 `1 + p`
        # （p = 该预算下的单次失败率）。按实测 p=10% 算 ⇒ 期望多花 **10%** 的 token，
        # 换回 **90%** 的失败。而配额耗尽的 429 **不在此列**（见下面的分类）。
        parse_attempts: int = 2,
        # 重试时把预算放大这个倍数 —— 因为病因就是"预算差一点点"。
        # 原样重发等于赌思维链这次恰好更短；放大预算直接把病因挪开。
        # 上限由 `parse_budget_cap` 兜住，避免极端情况把预算翻上天。
        parse_budget_factor: float = 1.5,
        parse_budget_cap: int = 65536,
    ) -> None:
        self.model = model or os.getenv("NPC_AGENT_MODEL", "")
        self.base_url = (
            base_url or os.getenv("NPC_AGENT_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.api_key = api_key or os.getenv("NPC_AGENT_API_KEY", "")
        self.timeout = timeout
        self.transient_attempts = max(1, transient_attempts)
        self.transient_backoff = transient_backoff
        self.parse_attempts = max(1, parse_attempts)
        self.parse_budget_factor = max(1.0, parse_budget_factor)
        self.parse_budget_cap = max(1, parse_budget_cap)
        #: 解析层重试的次数与理由，供探针/报告读取（**别从外面猜**）。
        self.parse_retries = 0
        self.parse_retry_reasons: list[str] = []

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

    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_transient_http(code: int) -> bool:
        """**5xx 重试，429 和其余 4xx 不重试。**

        5xx 是服务端自己出问题，重发合理。429 看着像"过一会儿就好"，
        但实测那批 429 全是 `apikey_quota_exhausted`（**按 key 计的额度用光**）——
        退避 1~2 秒不会让它恢复，重试只是把失败推迟几秒。
        其余 4xx 是请求本身有问题（模型名写错、key 无效），重发纯属浪费。
        """
        return 500 <= code < 600

    def _attempt(self, call):
        """跑 `call`，**只对传输层故障重试**；重试次数用尽才抛 `LLMUnavailable`。

        抛出的消息里保留**最后一次的原始原因**（`TimeoutError: ...` 等），
        因为下游的回落分类器是按特征串认类的（`is_transport_error`）——
        把原因换成"重试失败"会让它掉进"其他"那一栏。
        """
        last = ""
        for attempt in range(self.transient_attempts):
            if attempt:
                time.sleep(self.transient_backoff * (2 ** (attempt - 1)))
            try:
                return call()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
                if not self._is_transient_http(exc.code):
                    raise LLMUnavailable(f"HTTP {exc.code}: {detail}") from exc
                last = f"HTTP {exc.code}: {detail}"
            except Exception as exc:  # 超时 / 连接断开 / 响应体被截断
                last = f"{type(exc).__name__}: {exc}"
        raise LLMUnavailable(f"{last}（共尝试 {self.transient_attempts} 次）")

    def _post_json(self, request) -> dict:
        def once() -> dict:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))

        return self._attempt(once)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        if not self.available:
            raise LLMUnavailable("未配置 NPC_AGENT_MODEL，无法调用模型")

        # ⚠️ **两层重试，包的东西不一样，别混：**
        #
        #   ① 传输层（`_post_json` → `_attempt`）：请求压根没到模型 / 没回来。
        #      重发同样的请求是合理的，而且在**同一层**里循环。
        #   ② 解析层（这个循环）：HTTP 200 回来了，但内容不可用
        #      （`finish_reason=length` / 空内容 + 长思维链）。
        #      **从前它整个在重试覆盖范围之外** —— 不是"决定不重试"，
        #      是抛出的位置在 `_attempt()` 包住的那段之外。
        #
        # ②之所以值得做，依据是量出来的（见 `parse_attempts` 的注释）：
        # 病因是"预算差一点点"（76% 的失败思维链离预算不到 400 字），
        # 而思维链长度本身在抖 ⇒ **重试时同时把预算放大**，直接挪开病因。
        budget = max_tokens
        last: LLMUnavailable | None = None
        for attempt in range(self.parse_attempts):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(
                    self._payload(messages, temperature, budget)
                ).encode("utf-8"),
                headers=self._headers(),
                method="POST",
            )
            # 传输层故障由 `_post_json` 自己重试；它抛出来时不该再进解析层重试
            # （重试的是"解析不出可用内容"，不是"请求失败"）。
            body = self._post_json(request)

            try:
                content = self._extract_content(body)
            except LLMUnavailable as exc:
                last = exc
                if attempt + 1 < self.parse_attempts:
                    self.parse_retries += 1
                    self.parse_retry_reasons.append(str(exc))
                    budget = min(
                        int(budget * self.parse_budget_factor), self.parse_budget_cap
                    )
                    continue
                raise

            return content

        # 循环走完还没 return ⇒ 最后一次也没成功（理论上上一轮的 raise 已经抛出）
        raise last or LLMUnavailable("解析层重试耗尽")

    @staticmethod
    def _extract_content(body: dict) -> str:
        """从响应体里取出可用内容；取不出就抛 `LLMUnavailable`（**带上原因**）。

        抽成一个方法，是为了让"哪些失败属于解析层"有**唯一**的定义 ——
        重试循环和调用方都读它，不再各判一套。
        """
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
        # 只重试**建立连接**那一步：一旦开始吐字，重试就会把已经发出去的内容
        # 再发一遍（客户端会看到重复的半句话）。
        response = self._attempt(
            lambda: urllib.request.urlopen(request, timeout=self.timeout)
        )
        with response:
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
