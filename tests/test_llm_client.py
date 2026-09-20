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


#: 实测"够用"的规划预算。**这个数是量出来的，不是推出来的。**
#:
#: village 场景、读超时 180s：预算 4096 时 6 次规划调用有 2 次
#: `finish_reason=length`（思维链 16590 / 17276 字被吃光）；
#: 提到 16384 之后这一类清零。
MEASURED_SUFFICIENT_PLANNER_BUDGET = 16384


def test_the_planner_budget_is_at_least_the_measured_sufficient_value() -> None:
    """规划预算必须 ≥ 实测"够用"的那个值。

    ⚠️ 这里**不用**"预算 ≥ 实测思维链字数"那种写法。那条对台词成立
    （中文散文大约 1 字/token），对规划**不成立** —— 规划思维链里大量是
    JSON 和 ASCII，实测约 4 字/token（17276 字只吃掉了约 4096 token）。
    拿字数当 token 的下界会得出"16384 < 17276，所以不够"这种**错的**结论。

    所以锁的是直接测出来的那个够用值。**而且这个实验必须在超时放宽之后做**：
    读超时 60s 时失败全是 `TimeoutError`（思维链长度 0），
    预算问题被整个盖住 —— 见 `docs/ENGINEERING.md` 附八。
    """
    assert RuntimeConfig().max_tokens >= MEASURED_SUFFICIENT_PLANNER_BUDGET


#: 实测的**单次调用耗时**（秒），kimi-k2.7-code。
#:
#: 27s 是平均调用、35s 是规划调用。旧默认超时 60s 就贴着这两个数 ——
#: 平均值贴着上限，尾巴必然被砍。
OBSERVED_CALL_SECONDS = (27.0, 35.0)


def test_the_llm_timeout_is_not_below_the_observed_call_latency() -> None:
    """读超时必须容得下**实测的单次调用耗时**，不能贴着平均值。

    这条锁的是实测数字，不是审美。旧默认值 60s 和实测值贴得太近：
    village 场景 12 轮里 6 次规划调用有 2 次 `TimeoutError`，
    而失败原文里思维链长度是 **0** —— 响应根本没回来。

    ⚠️ 和预算那条是**两个独立的假设**，所以各有各的护栏：
    实测把预算从 4096 提到 16384，失败数反而从 2 涨到 3
    ⇒ 卡住的是等待时间，不是 token。混成一条就会把
    "我们等得不够久"记成"模型规划得不好"。
    """
    timeout = RuntimeConfig().llm_timeout
    slowest = max(OBSERVED_CALL_SECONDS)
    assert timeout >= slowest * 3, (
        f"读超时 {timeout}s 只有实测单次耗时 {slowest}s 的 {timeout / slowest:.1f} 倍："
        "平均值附近就开始丢调用，而丢掉的调用会被静默记成「规划失败」"
    )


def test_the_two_timeout_defaults_agree() -> None:
    """客户端默认超时和配置默认超时**必须是同一个数**。

    这是"两个真相"那一类：两处各写一份默认值、都不报错，
    改了其中一处之后，走 `build_llm()` 的路径和走 `RuntimeConfig` 的路径
    会**等不一样久** —— 而症状只是"有时候会回落"，没人查得出来。
    """
    assert RuntimeConfig().llm_timeout == OpenAICompatLLM().timeout


def test_the_eval_path_hands_the_configured_timeout_to_the_client() -> None:
    """配置里的超时要**真的走到**客户端 —— 否则那个字段是个摆设。

    ⚠️ 这条查的是**接线**，不是数字。
    `test_the_two_timeout_defaults_agree` 只证明两个默认值相等，
    完全不能证明评测路径把配置传下去了 —— 中间那一段可以压根不传，
    而症状是"配置写了 180s、实际还是 60s"，从任何报告里都看不出来。
    """
    from npc_agent.eval.runner import _default_llm_factory

    config = RuntimeConfig(llm_provider="openai-compat", model="m")
    llm = _default_llm_factory(config)()
    assert getattr(llm, "timeout", None) == config.llm_timeout, (
        "评测路径没把 `llm_timeout` 传给客户端 —— 配置项失效了"
    )


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


# --------------------------------------------------------------------------- #
# 读超时的透传
#
# 判分的 prompt 比台词长得多，实测约 7% 的调用会超过默认的 60s ——
# 而超时会变成一条**永久缺失的判决**（`Verdict.unjudged` 是合法返回值，
# 不重试、不报错）。所以超时必须可调。
# --------------------------------------------------------------------------- #


def test_build_llm_uses_the_client_default_timeout_when_unspecified():
    """不传就落到**客户端那个默认值**上，而不是某个写死的数。

    ⚠️ 2026-09-20：这里从前写死 `== 60.0`。默认超时从 60s 提到 180s 时
    它红了 —— 但红的是"数字变了"，不是"行为错了"。改成对着
    `OpenAICompatLLM()` 的默认值断言：这样它锁的是**"不传就用默认值"**这个性质。
    数字本身该是多少，由 `test_the_llm_timeout_is_not_below_the_observed_call_latency`
    那条按实测值管 —— 两条各管一件事，改数字时只有一条会红。

    显式给 0 也算"没给"，不能变成 0 秒超时（那等于每次都失败）。
    """
    from npc_agent.llm import build_llm

    default = OpenAICompatLLM().timeout
    assert build_llm("openai-compat", model="m").timeout == default
    assert build_llm("openai-compat", model="m", timeout=0).timeout == default
    assert build_llm("openai-compat", model="m", timeout=None).timeout == default


def test_build_llm_passes_an_explicit_timeout_through():
    from npc_agent.llm import build_llm

    assert build_llm("openai-compat", model="m", timeout=180).timeout == 180.0
    assert build_llm("openai-compat", model="m", timeout="180").timeout == 180.0


def test_a_longer_timeout_is_actually_handed_to_urlopen(monkeypatch):
    """光把字段存下来不算数 —— 要真的传到 `urlopen` 上。"""
    seen: dict = {}

    def fake_urlopen(request, timeout=None):  # noqa: ANN001
        seen["timeout"] = timeout
        return _FakeResponse(
            {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    from npc_agent.llm import build_llm

    llm = build_llm("openai-compat", model="m", api_key="k", timeout=180)
    llm.complete([{"role": "user", "content": "hi"}])
    assert seen["timeout"] == 180.0

