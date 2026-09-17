"""等一批跑完 —— 并且分得清「跑完了」「跑死了」「还在跑」。

## 为什么需要这个脚本

跑批的检查点**每完成一条用例才写一次**。而单条用例最慢可以跑到 18 分钟
（`memory_survives_consolidation` 那三条实测 657s / 1017s / 1066s）。
于是「检查点 5 分钟没动」完全可能只是那条慢用例还在跑。

我第一版监控脚本就是拿「检查点陈旧」当死亡判据，结果把一条**正在正常
运行**的跑批判成了崩溃，差一点就去 `--resume` 重跑 —— 而那时原进程
还活着，两个进程会同时往同一个检查点文件里写。

判据应该是「进程还活着吗」，不是「文件有没有动」。

## 退出码

    0  跑完了（done == total，或产物 JSON 已落盘）
    2  进程没了但没跑完 —— 需要 `--resume`
    3  超时（还在跑，只是没等到）

## 用法

    python scripts/wait_for_batch.py \
        --checkpoint reports/batch_model_checkpoint.json \
        --expect-json reports/eval_model.json \
        --pid 16900 --pid 15284 \
        --interval 20 --timeout 3600
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _read_checkpoint(path: Path) -> dict:
    """读检查点。写方用的是 os.replace 原子替换，所以读到的要么是旧的要么是新的。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _pid_alive(pid: int) -> bool:
    """判断进程是否存活。不引入 psutil —— 不想为一个监控脚本多一个依赖。

    Windows 走 `OpenProcess` + `GetExitCodeProcess`，不解析任何命令输出。
    **不解析输出这点很关键**：中文 Windows 的 `tasklist` 输出是 GBK，
    用 `text=True`（默认 utf-8）读会让读取线程抛 UnicodeDecodeError，
    `stdout` 变成 `None`，最后报一个 `TypeError: NoneType is not iterable`
    —— 错误信息和真正的原因（编码）毫无关系，白白浪费一轮排查。
    进程存活是内核状态，本来就不该靠解析本地化文本去问。

    另外**绝不能用 `os.kill(pid, 0)` 探活**：POSIX 上这是无副作用的探针，
    但在 Windows 上 CPython 的 `os.kill` 对非控制台信号走的是
    `TerminateProcess` —— 拿它探活会把跑批直接杀掉。

    为什么不能只 `OpenProcess` 成功就算活着：进程**终止之后**，
    只要还有别的进程持有它的句柄（`subprocess.Popen` 就会一直持有到
    对象被回收），内核里那个进程对象就还在，`OpenProcess` 依然成功。
    只看 `OpenProcess` 会把一条早就死掉的跑批报成"还在跑"，
    于是监控脚本会一直等到超时，而不是干脆地告诉你"去 --resume"。
    所以必须再问一次退出码。
    """
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            ERROR_ACCESS_DENIED = 5

            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if not handle:
                # 打开失败有两种截然不同的原因，不能一律当成"死了"：
                #   5   ACCESS_DENIED → 进程存在，只是不归我查（别人的进程）
                #   87  INVALID_PARAMETER → 确实没有这个 PID
                if ctypes.get_last_error() == ERROR_ACCESS_DENIED:
                    return True
                return False
            try:
                code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    # 问不出退出码就别断言它死了 —— 探测手段失效不能
                    # 伪装成任务故障。
                    return True
                # 已知怪癖：真有一个进程以退出码 259 退出的话，这里会
                # 把它看成"还在跑"。这个代价可以接受 —— 宁可多等，
                # 也不要把活着的跑批判死。
                return code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except (OSError, AttributeError):
            # 探测手段本身失效时**不能**当作"进程死了"。
            return True

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds // 60:.0f}m{seconds % 60:02.0f}s"
    return f"{seconds // 3600:.0f}h{(seconds % 3600) // 60:02.0f}m"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="等一批跑完，并区分跑完/跑死/还在跑")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument(
        "--expect-json",
        type=Path,
        default=None,
        help="正常结束时才会出现的产物；出现即视为完成（比数检查点更可靠）",
    )
    ap.add_argument("--pid", action="append", default=[], type=int,
                    help="跑批进程的 PID，可重复。给了就按进程存活判断")
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--max-silence", type=float, default=2400.0,
                    help="没给 --pid 时的兜底：检查点静默超过这么久才算死。"
                         "默认 40 分钟 —— 最慢单条实测 18 分钟，留了一倍余量")
    args = ap.parse_args(argv)

    started = time.time()
    last_done = -1
    last_change = time.time()
    pids = list(args.pid)

    if not pids:
        print("[warn] 没给 --pid，只能靠检查点静默时间兜底判断。"
              "这个判据曾经把活着的跑批误判成死亡 —— 能拿到 PID 就尽量给。",
              flush=True)

    while True:
        payload = _read_checkpoint(args.checkpoint)
        total = int(payload.get("total") or 0)
        done = int(payload.get("done") or 0)
        runs = payload.get("runs") or []
        calls = sum(int(r.get("llm_calls") or 0) for r in runs)
        degraded = sum(1 for r in runs if r.get("degraded"))
        failed = sum(1 for r in runs if not r.get("ok"))

        if done != last_done:
            last_change = time.time()
            last_done = done
            print(
                f"{time.strftime('%H:%M:%S')} {done}/{total} "
                f"calls={calls} degraded={degraded} failed={failed}",
                flush=True,
            )

        # 产物落盘 = 真的正常结束了。这比数检查点可靠：检查点是过程中写的，
        # 产物是收尾写的。
        if args.expect_json and args.expect_json.exists():
            print(f"OK: 产物已落盘 {args.expect_json}", flush=True)
            return 0
        if total and done >= total:
            print(f"OK: {done}/{total} 跑完了", flush=True)
            return 0

        elapsed = time.time() - started
        if elapsed > args.timeout:
            print(f"TIMEOUT: {done}/{total}，{_fmt(elapsed)} 内没等到", flush=True)
            return 3

        if pids:
            alive = [p for p in pids if _pid_alive(p)]
            if not alive:
                print(f"DEAD: 进程都没了，{done}/{total} 没跑完 —— 需要 --resume",
                      flush=True)
                return 2
        else:
            silence = time.time() - last_change
            if silence > args.max_silence:
                print(f"DEAD(猜测): 检查点静默 {_fmt(silence)}，{done}/{total}",
                      flush=True)
                return 2

        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
