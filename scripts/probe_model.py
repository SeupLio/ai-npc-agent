"""单个模型的连通性 + 延迟探针。

设计原则：
- 一次只打一个模型，避免批量串行把外层工具超时打爆；
- 每步都打印时间戳，即使被 SIGTERM 也能从 stdout 看到卡在哪；
- 强制 flush，保证后台运行时输出不丢；
- **失败时把 HTTP 状态、响应体、限流相关响应头都打出来。**
  探针的用途就是回答"现在能不能用、不能用是为什么"，
  把 `429` 压成一行 `HTTP Error 429` 等于把唯一的证据扔掉。

用法：
    python scripts/probe_model.py <model> [--max-tokens 2048]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
import json

BASE_URL = os.environ.get("NPC_AGENT_BASE_URL", "")
API_KEY = os.environ.get("NPC_AGENT_API_KEY", "")

# 网关如果在这些头里给出重置时间，就照抄出来；给不出也要明说"给不出"。
RATELIMIT_HEADERS = (
    "retry-after",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
)

PERSONA_PROMPT = """你叫阿柚，是"星屿咖啡屋"的店员，性格爽朗但话不多。
说话规则：每次最多 2 句、不超过 90 字；不能说"作为 AI""我是语言模型"这类出戏的话。
当前场景：一位玩家第一次走进店里，站在吧台前。
请用阿柚的口吻说一句招呼语，并顺便问一句他是不是第一次来。
只输出台词本身，不要加引号、不要加解释。"""


def post(path: str, payload: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def report_http_error(exc: urllib.error.HTTPError) -> str:
    """把 HTTPError 拆开打印，返回一句人类可读的归因。"""
    body = ""
    try:
        body = exc.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - 读不到就算了，别因为读 body 再崩一次
        body = "<读不到响应体>"

    print(f"  HTTP 状态     = {exc.code} {exc.reason}", flush=True)
    print(f"  响应体        = {body[:500]}", flush=True)

    # 限流相关的头全列出来；一个都没有也要明说，因为"没有"本身就是结论。
    present = {k: v for k, v in exc.headers.items() if k.lower() in RATELIMIT_HEADERS}
    if present:
        print("  限流响应头    =", flush=True)
        for k, v in present.items():
            print(f"    {k}: {v}", flush=True)
    else:
        print("  限流响应头    = 无（没有 Retry-After，也没有任何 x-ratelimit-*）", flush=True)

    # 归因：把网关的 error.code 认出来，别让调用方自己去猜。
    code = ""
    try:
        code = (json.loads(body).get("error") or {}).get("code", "")
    except Exception:  # noqa: BLE001
        pass

    if exc.code == 429 or code == "apikey_quota_exhausted":
        return ("额度已打满（quota exhausted）—— 这是按 key 计的额度，不是模型的问题；"
                "网关没有给出重置时间，所以只能隔一段时间再探一次。")
    if exc.code in (401, 403):
        return "鉴权失败 —— 检查 API key 是否有效、是否有该模型权限。"
    if exc.code == 404:
        return "路径或模型不存在 —— 检查 BASE_URL 是否带 /v1、模型名是否正确。"
    return f"请求失败（HTTP {exc.code}），按上面的响应体判断。"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    if not BASE_URL:
        print("缺少 NPC_AGENT_BASE_URL（例：https://your-endpoint/v1）", flush=True)
        return 2

    if not API_KEY:
        print("缺少 NPC_AGENT_API_KEY", flush=True)
        return 2

    print(f"[{time.strftime('%H:%M:%S')}] model={args.model} max_tokens={args.max_tokens}", flush=True)

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": PERSONA_PROMPT}],
        "max_tokens": args.max_tokens,
        "temperature": 0.7,
    }

    t0 = time.time()
    try:
        body = post("/chat/completions", payload, timeout=args.timeout)
    except urllib.error.HTTPError as exc:
        dt = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] FAILED after {dt:.1f}s", flush=True)
        verdict = report_http_error(exc)
        print(f"  归因          = {verdict}", flush=True)
        return 1
    except Exception as exc:  # noqa: BLE001 - 网络层/解析层，和 HTTP 错误分开报
        dt = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] FAILED after {dt:.1f}s: "
              f"{type(exc).__name__}: {exc}", flush=True)
        print("  归因          = 连接层失败（不是 HTTP 错误码）—— 检查网络 / BASE_URL 是否可达。",
              flush=True)
        return 1

    dt = time.time() - t0
    choice = body["choices"][0]
    msg = choice.get("message", {})
    content = (msg.get("content") or "").strip()
    reasoning = msg.get("reasoning_content") or ""
    usage = body.get("usage", {})

    print(f"[{time.strftime('%H:%M:%S')}] ok in {dt:.1f}s", flush=True)
    print(f"  finish_reason = {choice.get('finish_reason')}", flush=True)
    print(f"  usage         = {usage}", flush=True)
    print(f"  reasoning_len = {len(reasoning)}", flush=True)
    print(f"  content_len   = {len(content)}", flush=True)
    print(f"  content       = {content[:400]}", flush=True)
    return 0 if content else 3


if __name__ == "__main__":
    sys.exit(main())
