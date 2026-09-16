"""单个模型的连通性 + 延迟探针。

设计原则：
- 一次只打一个模型，避免批量串行把外层工具超时打爆；
- 每步都打印时间戳，即使被 SIGTERM 也能从 stdout 看到卡在哪；
- 强制 flush，保证后台运行时输出不丢。

用法：
    python scripts/probe_model.py <model> [--max-tokens 2048]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
import json

BASE_URL = os.environ.get("NPC_AGENT_BASE_URL", "https://ai.ctaigw.cn/v1")
API_KEY = os.environ.get("NPC_AGENT_API_KEY", "")

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

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
    except Exception as exc:  # noqa: BLE001
        print(f"[{time.strftime('%H:%M:%S')}] FAILED after {time.time()-t0:.1f}s: {type(exc).__name__}: {exc}", flush=True)
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
