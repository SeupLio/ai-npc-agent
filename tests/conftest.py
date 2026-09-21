"""让 pytest 能找到 npc_agent 包（不依赖安装），并关掉重试退避的真实等待。

## 为什么要有那个 autouse fixture

`LLMJudge` 的重试退避是 `JUDGE_BACKOFF=3.0`、`JUDGE_MAX_RETRIES=2`
⇒ 重试耗尽一次要真的等 **9 秒**。测试里那些"喂坏输出 / 让模型抛异常"的用例
**每条都会走完重试** —— 实测 7 条这样，合计 **63 秒**；
再加一条 27 秒的，一个 158 秒的套件里约 **90 秒**纯粹在 `time.sleep`。

原来靠"每个测试自己传 `sleep=_no_sleep`"来避免，而实测有 **7 处忘了**。
这类"靠人记得"的约定迟早会漏，所以改成**一个地方关掉**：
`judge.DEFAULT_SLEEP` 是唯一的默认等待函数，这里把它换成 no-op。

⚠️ 为什么不直接 patch `time.sleep`：`test_judge.py` 里有一条用例
**故意用 `time.sleep` 模拟模型延迟**来测并发加速比。全局 patch 会把它的
语义一起改掉 —— 那种"顺手关掉太多"的修法会静默削弱别的测试。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _no_real_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试里不许真的等重试退避（3s / 6s）。

    只换 `judge.DEFAULT_SLEEP` —— 显式传了 `sleep=` 的用例不受影响，
    别处（比如模拟延迟）的 `time.sleep` 也不受影响。
    """
    from npc_agent.eval import judge as judge_module

    monkeypatch.setattr(judge_module, "DEFAULT_SLEEP", lambda _seconds: None)
