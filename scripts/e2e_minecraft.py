"""端到端：真 Node 桥 <-> 真 Minecraft 服务端。

和 tests/test_minecraft_env.py 里的 `--dry-run` 测试的区别：
dry-run 验证的是**协议**，这一条验证的是**协议 + mineflayer + 服务端**三段接得上。

跑法（服务端要先起来）：
    python scripts/e2e_minecraft.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from npc_agent.env.mc_client import MineflayerClient  # noqa: E402

NODE = sys.executable.replace("python.exe", "")  # 占位，下面用真实 node 路径
NODE = r"C:/Users/10718/.workbuddy-ai/binaries/node/versions/22.22.2-2/node.exe"
BRIDGE = str(ROOT / "scripts" / "mineflayer_bridge.js")

SCENARIO = {
    "actors": [
        {"id": "ayan", "name": "阿岩", "kind": "npc", "start": "camp"},
    ],
    # 超平坦世界的表面在 y = -60（基岩 -64 / 泥土 / 草方块 -61，站立面 -60）。
    # 写 y=4 的话路径规划器会找不到路 —— 它不会飞。
    "pois": {
        "camp": {"name": "营地", "pos": [0, -60, 0]},
        "forest": {"name": "北边林子", "pos": [12, -60, 6]},
    },
    "resources": {"forest": {"oak_log": 3}},
}


def main() -> int:
    cmd = [
        NODE, BRIDGE,
        "--host", "127.0.0.1",
        "--port", "25565",
        "--username", "npc_agent",
        "--version", "1.21.11",
    ]
    print("启动桥：", " ".join(cmd))
    with MineflayerClient(cmd, cwd=str(ROOT), timeout=60.0) as client:
        # 1) configure
        res = client.configure(SCENARIO)
        print(f"[1] configure ok={res.ok} reason={res.reason!r}")
        if not res.ok:
            return 1

        # 2) 等 bot 真的连上（桥的 requireBot 会区分"没连上"和"操作失败"）
        ready = False
        for i in range(30):
            probe = client.call("move", actor="ayan", target="camp")
            if probe.ok:
                ready = True
                print(f"[2] bot 已连上（第 {i + 1} 次探测成功）")
                break
            if "还没连上" not in probe.reason:
                print(f"[2] 意外的失败：{probe.reason!r}")
                return 1
            time.sleep(2)
        if not ready:
            print("[2] 60 秒内 bot 没连上服务端")
            return 1

        # 3) state 能读到真实世界镜像
        state = client.state()
        print(f"[3] state tick={state.get('tick')} dry_run={state.get('dry_run')} "
              f"actors={list((state.get('actors') or {}).keys())}")

        # 4) 一次真实移动 + 取物
        moved = client.call("move", actor="ayan", target="forest")
        print(f"[4] move->forest ok={moved.ok} reason={moved.reason!r}")

        took = client.call("mine", actor="ayan", block="oak_log")
        print(f"[5] mine ok={took.ok} reason={took.reason!r} data={took.data}")

        state = client.state()
        actor = (state.get("actors") or {}).get("ayan") or {}
        print(f"[6] 移动后 pos={actor.get('pos')} poi={actor.get('poi')} "
              f"inventory={actor.get('inventory')}")

        # 5) 说一句话（走真实 chat）
        spoke = client.call("chat", actor="ayan", text="我到林子了。")
        print(f"[7] chat ok={spoke.ok} reason={spoke.reason!r}")

        print("\n=== 端到端结论 ===")
        print("  协议通：     是（configure/state/call 全部有响应）")
        print("  mineflayer： 是（bot 连上并 spawn）")
        print("  服务端：     1.21.11 @ 127.0.0.1:25565")
        print(f"  真实动作：   move ok={moved.ok} / mine ok={took.ok} / chat ok={spoke.ok}")
        return 0 if (moved.ok and took.ok and spoke.ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
