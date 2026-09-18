"""探针的失败归因要有牙。

`probe_model.py` 存在的唯一理由，是回答"现在能不能用、不能用是为什么"。
它以前把一切异常都压成一行 `HTTP Error 429`，把响应体里的
`apikey_quota_exhausted` 和限流响应头全扔了 —— 而这恰恰是唯一的证据。

这一组测试钉住三件事：
1. 429 要认出是"额度打满"，而不是笼统的"请求失败"；
2. 没有 `Retry-After` 时要**明说没有**，因为"没有重置时间"本身就是结论；
3. 鉴权失败和路径错误要和额度问题分开，否则用户会去改错的东西。
"""

from __future__ import annotations

import importlib.util
import io
import json
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

_PROBE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "probe_model.py"


def _load_probe():
    """按路径加载 scripts/probe_model.py（scripts/ 不是包，不能直接 import）。"""
    spec = importlib.util.spec_from_file_location("_probe_model_under_test", _PROBE_PATH)
    assert spec and spec.loader, f"加载不了探针脚本：{_PROBE_PATH}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


def _http_error(code: int, body: dict | str, headers: dict | None = None) -> urllib.error.HTTPError:
    """构造一个带响应体和响应头的 HTTPError，形状和真实的一致。"""
    raw = body if isinstance(body, str) else json.dumps(body)
    hdrs = Message()
    for key, value in (headers or {}).items():
        hdrs[key] = value
    return urllib.error.HTTPError(
        url="https://example.invalid/v1/chat/completions",
        code=code,
        msg="Too Many Requests" if code == 429 else "Error",
        hdrs=hdrs,
        fp=io.BytesIO(raw.encode("utf-8")),
    )


QUOTA_BODY = {
    "error": {
        "message": "ApiKey已触发限额",
        "type": "quota_error",
        "code": "apikey_quota_exhausted",
    }
}


def test_quota_error_is_named_not_generic(capsys):
    """429 + apikey_quota_exhausted 必须被认成额度问题。"""
    verdict = probe.report_http_error(_http_error(429, QUOTA_BODY))
    assert "额度" in verdict, f"没认出额度问题：{verdict}"

    out = capsys.readouterr().out
    assert "apikey_quota_exhausted" in out, "响应体没打出来，证据被吞了"
    assert "429" in out


def test_missing_ratelimit_headers_are_reported_as_missing(capsys):
    """没有重置时间要说"没有"，而不是静默什么都不说。"""
    probe.report_http_error(_http_error(429, QUOTA_BODY))
    out = capsys.readouterr().out
    assert "无" in out and "Retry-After" in out, (
        "网关不给重置时间，这本身是结论，必须显式说出来；"
        f"实际输出：{out!r}"
    )


def test_retry_after_is_surfaced_when_present(capsys):
    """一旦网关哪天开始给 Retry-After，探针要照抄出来 —— 不能写死"没有"。"""
    probe.report_http_error(_http_error(429, QUOTA_BODY, {"Retry-After": "3600"}))
    out = capsys.readouterr().out
    assert "3600" in out, f"有 Retry-After 却没打印：{out!r}"


@pytest.mark.parametrize(
    "code, needle",
    [
        (401, "鉴权"),
        (403, "鉴权"),
        (404, "不存在"),
    ],
)
def test_failure_causes_are_distinguished(code, needle):
    """鉴权/路径问题不能和额度问题混为一谈，否则会去改错的东西。"""
    verdict = probe.report_http_error(_http_error(code, {"error": {"message": "nope"}}))
    assert needle in verdict, f"HTTP {code} 归因错了：{verdict}"


def test_non_429_does_not_claim_quota_is_exhausted():
    """500 之类的服务端错误不能甩锅给"额度"。"""
    verdict = probe.report_http_error(_http_error(500, {"error": {"message": "boom"}}))
    assert "额度已打满" not in verdict, f"500 被误判成额度问题：{verdict}"


def test_undecodable_body_does_not_crash_the_probe():
    """读不出响应体时也不能再崩一次 —— 探针崩了就什么证据都没有。"""
    exc = _http_error(429, "<not json>")
    verdict = probe.report_http_error(exc)
    assert isinstance(verdict, str) and verdict
