"""端到端：真 Node 桥 <-> 真 Minecraft 服务端。

## 这个文件和 `test_minecraft_env.py` 里的 `--dry-run` 测试有什么区别

`--dry-run` 那条验证的是**协议**：请求/响应成对、id 不错位、坏输入不带走桥。
它不需要服务端，所以永远能跑，也因此**证明不了最后一段接得上**。

这一条验证的是**协议 + mineflayer + 真实服务端**三段。它需要：

1. 装了 node 依赖（`npm i mineflayer mineflayer-pathfinder vec3`）
2. 一个真在跑的服务端（1.21.11，超平坦，`online-mode=false`）

这两样都不是默认环境里有的，所以**默认跳过**。要跑就设
`NPC_AGENT_MC_E2E=1`：

    NPC_AGENT_MC_E2E=1 python -m pytest tests/test_minecraft_e2e.py -q

跳过而不是失败，是因为"没起服务端"**不是代码的问题**。
但如果起了服务端还跳过，那就说明判据写错了 —— 所以判据里
`NPC_AGENT_MC_E2E` 和端口探测是**两个独立条件**，各自的跳过原因分开写。
"""

from __future__ import annotations

import os
import shutil
import socket
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from npc_agent.env.mc_client import MineflayerClient  # noqa: E402

MC_HOST = os.environ.get("NPC_AGENT_MC_HOST", "127.0.0.1")
MC_PORT = int(os.environ.get("NPC_AGENT_MC_PORT", "25565"))
MC_VERSION = os.environ.get("NPC_AGENT_MC_VERSION", "1.21.11")

# 超平坦世界：基岩 -64 / 泥土 / 草方块 -61，**站立面是 y = -60**。
# 写成 y=4 会让 mineflayer-pathfinder 找不到路（它不会飞）——
# 报的是 "Took to long to decide path to goal!"，看起来像桥坏了，
# 其实只是坐标写在了半空里。
GROUND_Y = -60


def _node() -> str:
    node = shutil.which("node")
    if node:
        return node
    managed = Path(
        r"C:/Users/10718/.workbuddy-ai/binaries/node/versions/22.22.2-2/node.exe"
    )
    if managed.exists():
        return str(managed)
    pytest.skip("找不到 node")


def _port_open() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((MC_HOST, MC_PORT)) == 0


@pytest.fixture
def real_server() -> str:
    """只有在明确要求、并且服务端真的在监听时才放行。"""
    if os.environ.get("NPC_AGENT_MC_E2E") != "1":
        pytest.skip("没设 NPC_AGENT_MC_E2E=1，跳过真实服务端测试")
    if not _port_open():
        pytest.skip(f"{MC_HOST}:{MC_PORT} 没有服务端在监听")
    if not (ROOT / "node_modules" / "mineflayer").exists():
        pytest.skip("没装 mineflayer（npm i mineflayer mineflayer-pathfinder vec3）")
    return _node()


def test_bridge_drives_a_real_minecraft_server(real_server: str) -> None:
    """走完 move -> mine -> chat，并检查**世界真的变了**。

    断言落在真实状态上（坐标、背包），不落在桥自己的记账上 ——
    记账标记可能设了但事情没做，反过来也一样。
    """
    cmd = [
        real_server,
        str(ROOT / "scripts" / "mineflayer_bridge.js"),
        "--host", MC_HOST,
        "--port", str(MC_PORT),
        "--username", "npc_agent_test",
        "--version", MC_VERSION,
    ]
    scenario = {
        "actors": [{"id": "ayan", "name": "阿岩", "kind": "npc", "start": "camp"}],
        "pois": {
            "camp": {"name": "营地", "pos": [0, GROUND_Y, 0]},
            "forest": {"name": "北边林子", "pos": [12, GROUND_Y, 6]},
        },
        "resources": {"forest": {"oak_log": 3}},
    }

    with MineflayerClient(cmd, cwd=str(ROOT), timeout=60.0) as client:
        assert client.configure(scenario).ok

        # 等 bot spawn。桥用 requireBot() 区分"还没连上"和"操作失败"，
        # 所以这里能安全地轮询 —— 前者不算失败。
        for _ in range(30):
            probe = client.call("move", actor="ayan", target="camp")
            if probe.ok:
                break
            assert "还没连上" in probe.reason, f"连上了但移动失败：{probe.reason}"
            time.sleep(2)
        else:
            pytest.fail("60 秒内 bot 没连上服务端")

        assert client.state()["dry_run"] is False

        moved = client.call("move", actor="ayan", target="forest")
        assert moved.ok, moved.reason

        mined = client.call("mine", actor="ayan", block="oak_log")
        assert mined.ok, mined.reason

        # 断言真实状态：坐标真的到了林子，背包里真的有木头
        actor = client.state()["actors"]["ayan"]
        assert actor["poi"] == "forest"
        assert actor["pos"] == [12, GROUND_Y, 6]
        assert actor["inventory"].get("oak_log", 0) >= 1

        assert client.call("chat", actor="ayan", text="我到林子了。").ok
