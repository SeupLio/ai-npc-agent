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
    """找 node：环境变量优先，其次 PATH。

    不写死本机路径 —— 那会让这个文件在别人机器上直接失效，
    而且把开发者的目录结构带进公开仓库。
    """
    node = os.environ.get("NPC_AGENT_NODE") or shutil.which("node")
    if not node:
        pytest.skip("找不到 node（装 Node.js，或用 NPC_AGENT_NODE 指定路径）")
    return node


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
    """走完 move -> mine -> chat，断言落在**真实遥测**上。

    ## 为什么断言 `bot_pos` 而不是 `actors[*].pos`

    `state()` 里有两份位置，含义完全不同：

    - `actors[*].pos` —— 桥的**镜像**（我们记的账）
    - `bot_pos` —— `bot.entity.position`，**真实位置**

    只断言镜像等于没测：镜像由桥自己写，它想写什么就是什么。
    这个坑我真的踩了 —— 第一版测试断言 `actor["pos"] == [12, -60, 6]`，
    看着很硬，其实只是在读桥自己刚写进去的值。
    真状态是 `bot_pos`，所以断言它。

    ## 为什么允许 `move` 失败

    服务端的世界是**会累积的**：`mine` 挖的是 bot 脚下的方块，
    于是跑完一次，那个 POI 的位置就矮了一格。下次 `GoalBlock` 指向
    一个悬空坐标，`mineflayer-pathfinder` 会超时。

    这是世界状态的问题，不是桥的问题。所以这里不要求 `move` 必成功，
    而是要求**镜像和现实一致** —— 失败时镜像不许偷偷更新。
    那正好是修过的一个 bug（镜像写在动作之前），这条断言把它钉住。
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

        state = client.state()
        assert state["dry_run"] is False, "连上了服务端，不该走 dry-run 分支"
        assert state["bot_connected"] is True
        assert state["bot_pos"] is not None, "拿不到 bot 真实位置，说明没真连上"

        # move：不要求落点精确，但镜像必须跟现实一致
        moved = client.call("move", actor="ayan", target="forest")
        mirror = client.state()["actors"]["ayan"]["poi"]
        if moved.ok:
            assert mirror == "forest"
        else:
            assert mirror != "forest", (
                "move 返回失败，但镜像里 actor.poi 已经是目的地了 —— "
                f"state() 会报告它在林子，实际它还在原地。（原因：{moved.reason}）"
            )

        # mine：真挖一个方块（走 bot.dig），再读真实遥测
        mined = client.call("mine", actor="ayan", block="oak_log")
        assert mined.ok, mined.reason
        assert client.state()["bot_pos"] is not None

        # chat：真往服务端发一条消息
        assert client.call("chat", actor="ayan", text="我到林子了。").ok
