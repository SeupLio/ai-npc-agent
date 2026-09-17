"""`scripts/wait_for_batch.py` 的回归测试。

这个脚本本身是"监控工具"，而监控工具**判错**的代价特别高：
它说"跑死了"，人就去 `--resume` 重跑，而原进程可能还活着 ——
两个进程会同时往同一个检查点文件里写。所以它自己也得有测试。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "wait_for_batch.py"


def _load():
    spec = importlib.util.spec_from_file_location("wait_for_batch", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


wfb = _load()


# --------------------------------------------------------------------------
# 进程探活
# --------------------------------------------------------------------------


def test_a_live_process_is_reported_alive():
    assert wfb._pid_alive(os.getpid()) is True


def test_a_missing_pid_is_reported_dead():
    # 先起一个进程再把它收掉，拿到的 pid 是"确实用过、现在没了"的，
    # 比随手写 999999 更接近真实场景。
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    pid = proc.pid
    proc.wait()
    assert wfb._pid_alive(pid) is False


def test_probing_liveness_must_not_kill_the_probed_process():
    """探活必须是只读的。

    这不是假想的风险：在 POSIX 上 `os.kill(pid, 0)` 是无副作用的探针，
    很多人会顺手这么写。但在 Windows 上 CPython 的 `os.kill` 对非控制台
    信号走的是 `TerminateProcess` —— 同一行代码会把跑批直接杀掉，
    而且杀完还"证明"了进程已经不在。

    这条测试起一个真实进程，探活之后再确认它还活着。
    """
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        time.sleep(0.2)
        assert wfb._pid_alive(proc.pid) is True
        # 关键断言：探完之后它还得活着
        assert proc.poll() is None, "探活把被探测的进程弄死了"
    finally:
        proc.kill()
        proc.wait()


def test_probing_liveness_does_not_shell_out():
    """探活不该靠解析命令输出。

    早先的实现跑 `tasklist` 再解析它的文本，在中文 Windows 上因为
    输出是 GBK 而读取用 utf-8，读取线程直接抛 UnicodeDecodeError，
    `stdout` 变成 `None`，最后报出来的是
    `TypeError: argument of type 'NoneType' is not iterable` ——
    错误信息和真正的原因（编码）毫无关系，白白浪费一轮排查。

    进程存活是内核状态，本来就该直接问内核。这条测试把"不要退回到
    解析子进程输出"钉死：任何起子进程的实现都会在这里炸掉。
    """
    import subprocess as sp

    original_run, original_popen = sp.run, sp.Popen

    def boom(*args, **kwargs):
        raise AssertionError("探活不应该去起子进程 —— 直接用系统调用问内核")

    sp.run, sp.Popen = boom, boom
    try:
        assert wfb._pid_alive(os.getpid()) is True
        assert wfb._pid_alive(999_999_999) is False
    finally:
        sp.run, sp.Popen = original_run, original_popen


# --------------------------------------------------------------------------
# 时长格式化
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds,expected",
    [(0, "0s"), (45, "45s"), (59, "59s"), (60, "1m00s"),
     (200, "3m20s"), (3900, "1h05m")],
)
def test_duration_formatting(seconds, expected):
    assert wfb._fmt(seconds) == expected


# --------------------------------------------------------------------------
# 主循环的退出码
# --------------------------------------------------------------------------


def _write_checkpoint(path: Path, *, total: int, done: int, runs=None) -> None:
    path.write_text(
        json.dumps({"total": total, "done": done, "runs": runs or []},
                   ensure_ascii=False),
        encoding="utf-8",
    )


def test_exit_zero_when_the_batch_is_complete(tmp_path):
    ckpt = tmp_path / "ckpt.json"
    _write_checkpoint(ckpt, total=3, done=3)
    assert wfb.main(["--checkpoint", str(ckpt), "--interval", "0.01"]) == 0


def test_exit_zero_when_the_product_json_landed(tmp_path):
    """产物落盘比数检查点更可靠 —— 检查点是过程中写的，产物是收尾写的。"""
    ckpt = tmp_path / "ckpt.json"
    product = tmp_path / "eval_model.json"
    _write_checkpoint(ckpt, total=3, done=2)   # 还没数完
    product.write_text("{}", encoding="utf-8")
    assert wfb.main([
        "--checkpoint", str(ckpt), "--expect-json", str(product),
        "--interval", "0.01",
    ]) == 0


def test_exit_two_when_the_process_is_gone_and_work_remains(tmp_path):
    ckpt = tmp_path / "ckpt.json"
    _write_checkpoint(ckpt, total=3, done=1)
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    dead_pid = proc.pid
    proc.wait()
    assert wfb.main([
        "--checkpoint", str(ckpt), "--pid", str(dead_pid),
        "--interval", "0.01", "--timeout", "5",
    ]) == 2


def test_a_slow_case_is_not_mistaken_for_a_dead_process(tmp_path):
    """**这是这个脚本存在的理由。**

    检查点每完成一条用例才写一次，而单条用例最慢实测 1066s。
    所以"检查点很久没动"和"进程死了"是两件完全不同的事。

    这里造一个"检查点不动、但进程活着"的局面：给一个必然超时的
    `--timeout`，正确实现应该报 3（还在跑），而不是 2（死了）。
    """
    ckpt = tmp_path / "ckpt.json"
    _write_checkpoint(ckpt, total=3, done=1)
    # 用自己这个进程当"活着的跑批进程"
    assert wfb.main([
        "--checkpoint", str(ckpt), "--pid", str(os.getpid()),
        "--interval", "0.01", "--timeout", "0.05",
    ]) == 3


def test_silence_fallback_only_fires_after_the_generous_window(tmp_path):
    """没给 PID 时的兜底判据必须**足够宽松**。

    最慢单条实测 1066s（约 18 分钟）。默认静默窗口 2400s（40 分钟），
    留了一倍余量。这里用两个极端窗口验证兜底的两端行为 ——
    窗口给足时绝不误报，窗口被显式压到不可能满足时才判死。
    """
    ckpt = tmp_path / "ckpt.json"
    _write_checkpoint(ckpt, total=3, done=1)
    # 窗口给足 → 走超时分支（3），不误报死亡
    assert wfb.main([
        "--checkpoint", str(ckpt), "--interval", "0.01",
        "--timeout", "0.05", "--max-silence", "3600",
    ]) == 3
    # 窗口压到不可能满足（负数）→ 兜底立即判定为死。
    # 用负数而不是 0：刚写完检查点的那一刻 `silence` 恰好是 0，
    # 拿 0 当阈值会让这条断言依赖时钟精度。
    assert wfb.main([
        "--checkpoint", str(ckpt), "--interval", "0.01",
        "--timeout", "5", "--max-silence", "-1",
    ]) == 2


def test_a_broken_probe_does_not_get_reported_as_a_dead_batch(tmp_path):
    """探测手段失效 ≠ 任务失败。

    如果 `OpenProcess` 之类的手段本身不可用，脚本必须继续等，
    而不是把工具故障说成"跑批崩了"。
    """
    ckpt = tmp_path / "ckpt.json"
    _write_checkpoint(ckpt, total=3, done=1)
    original = wfb._pid_alive
    wfb._pid_alive = lambda pid: True   # 模拟"探测手段说它活着/无法判断"
    try:
        assert wfb.main([
            "--checkpoint", str(ckpt), "--pid", "424242",
            "--interval", "0.01", "--timeout", "0.05",
        ]) == 3
    finally:
        wfb._pid_alive = original
