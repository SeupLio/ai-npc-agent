"""WorldClient —— Minecraft 世界操作的传输层抽象。

## 为什么要再抽一层

`Environment` 是"世界"，`WorldClient` 是"怎么跟世界说话"。分开的理由有两个：

1. **可测性**：`MinecraftEnv` 的全部逻辑（观测怎么组装、工具怎么映射、
   护栏怎么给理由）都能在没有 Minecraft 服务端的情况下单测。
   跑一次真实服务端要几十秒，跑一次 LocalWorldClient 只要几毫秒。

2. **同一个契约，两种传输**：`LocalWorldClient` 在进程内执行，
   `MineflayerClient` 把同样的操作编码成 JSON 行，通过 stdio 发给一个
   Node 桥进程。**上层不知道下面跑的是哪一个** —— 这不是文档里的一句
   承诺，而是代码结构本身：两个后端都只实现 `call()`，
   所有类型化的方法（move / mine / craft / ...）都写在基类上，
   物理上只存在一份。

## 操作契约

每个操作都是 `(op, params)` → `{ok, reason, data}`，全部 JSON 可序列化。
这是唯一需要两个后端对齐的东西 —— 也就是 tests/test_minecraft_env.py 里
对两个后端跑同一套 contract conformance 用例的原因。

    reset()                                    回到初始状态
    configure(scenario)                        声明场景（演员 / 地点 / 资源点）
    state()                                    完整世界状态
    move(actor, target)                        移动到 POI
    mine(actor, block)                         采集方块
    craft(actor, item)                         在工作台合成
    place(actor, block, target)                放置方块
    consume(actor, item, count)                消耗物品
    transfer(src, dst, item, count)            物品转移（交付）
    chat(actor, text)                          说话
    set_flag(flag)                             设置世界标记
    advance_tick(n)                            时间推进

## 失败也是数据

操作失败返回 `ok=False` + **一句人话的原因**，不抛异常。
这句话会一路流到 Reflection 模块 —— "天黑了，看不清矿脉，需要先做个火把"
就是 NPC 下一轮该做的事。护栏的价值不在于拦住它，而在于告诉它为什么被拦。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

# --------------------------------------------------------------------------- #
# 方块与物品
# --------------------------------------------------------------------------- #
BLOCK_NAMES: dict[str, str] = {
    "oak_log": "橡木原木",
    "planks": "木板",
    "stick": "木棍",
    "coal": "煤炭",
    "torch": "火把",
    "cobblestone": "圆石",
    "stone": "石头",
    "air": "空气",
}

#: 合成配方。数值照抄 Minecraft：1 原木出 4 木板，2 木板出 4 木棍，
#: 1 煤炭 + 1 木棍出 4 火把。
#: 保留这种"进 1 出 4"的不对称是有意的 —— 它是数量语义存在的理由。
#: 如果配方都是 1:1，背包就退化成一个集合，"数量"这条能力就测不出来。
MC_RECIPES: dict[str, dict[str, Any]] = {
    "planks": {"name": "木板", "needs": {"oak_log": 1}, "yields": 4, "station": "workshop"},
    "stick": {"name": "木棍", "needs": {"planks": 2}, "yields": 4, "station": "workshop"},
    "torch": {"name": "火把", "needs": {"coal": 1, "stick": 1}, "yields": 4, "station": "workshop"},
}

#: 白天持续的 tick 数。一个昼夜 = 12 tick，其中后 4 tick 是夜里。
DAY_TICKS = 12
NIGHT_START = 8

#: 火把照亮周围多少格。夜里在这个半径内才能干活。
TORCH_LIGHT_RADIUS = 8

DEFAULT_POIS: dict[str, dict[str, Any]] = {
    "village_square": {"name": "村口广场", "pos": [0, 0, 0]},
    "forest": {"name": "北边林子", "pos": [6, 0, 2]},
    "workshop": {"name": "木工台", "pos": [2, 0, -3]},
    "cave_mouth": {"name": "洞口", "pos": [10, 0, -6]},
}

DEFAULT_RESOURCES: dict[str, dict[str, int]] = {
    "forest": {"oak_log": 8},
    "cave_mouth": {"coal": 4, "cobblestone": 12},
}


# --------------------------------------------------------------------------- #
# 操作结果
# --------------------------------------------------------------------------- #
@dataclass
class OpResult:
    """一次世界操作的结果。JSON 可序列化 —— 它要能穿过 stdio。"""

    ok: bool
    reason: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reason": self.reason, "data": self.data}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "OpResult":
        return cls(
            ok=bool(raw.get("ok")),
            reason=str(raw.get("reason") or ""),
            data=dict(raw.get("data") or {}),
        )

    @classmethod
    def fail(cls, reason: str) -> "OpResult":
        return cls(False, reason)


# --------------------------------------------------------------------------- #
# 传输层基类
# --------------------------------------------------------------------------- #
class WorldClient(ABC):
    """世界操作的传输接口。

    子类只需要实现 `call()`。所有类型化方法都在这里，
    因此两个后端在**行为层面**不可能漂移 —— 它们跑的是同一段代码。
    """

    # ------------------------------------------------------------------ #
    # 唯一需要子类实现的东西
    # ------------------------------------------------------------------ #
    @abstractmethod
    def call(self, op: str, **params: Any) -> OpResult:
        """执行一次操作。**不允许抛异常**，失败返回 ok=False + 原因。"""

    def close(self) -> None:
        """释放资源。进程内后端不需要做什么。"""

    def __enter__(self) -> "WorldClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # 类型化包装（两个后端共用这一份）
    # ------------------------------------------------------------------ #
    def reset(self) -> OpResult:
        return self.call("reset")

    def configure(self, scenario: dict[str, Any]) -> OpResult:
        """声明场景（有哪些人、哪些地点、哪些资源点）。

        真实 Minecraft 里世界早就存在了，这一步只是把游戏内实体
        映射到我们的 actor id 上；进程内后端则用它重建世界。
        """
        return self.call("configure", scenario=scenario)

    def state(self) -> dict[str, Any]:
        result = self.call("state")
        return dict(result.data or {})

    def move(self, actor: str, target: str) -> OpResult:
        return self.call("move", actor=actor, target=target)

    def mine(self, actor: str, block: str) -> OpResult:
        return self.call("mine", actor=actor, block=block)

    def craft(self, actor: str, item: str) -> OpResult:
        return self.call("craft", actor=actor, item=item)

    def place(self, actor: str, block: str, target: str) -> OpResult:
        return self.call("place", actor=actor, block=block, target=target)

    def consume(self, actor: str, item: str, count: int = 1) -> OpResult:
        return self.call("consume", actor=actor, item=item, count=count)

    def transfer(self, src: str, dst: str, item: str, count: int = 1) -> OpResult:
        return self.call("transfer", src=src, dst=dst, item=item, count=count)

    def chat(self, actor: str, text: str) -> OpResult:
        return self.call("chat", actor=actor, text=text)

    def set_flag(self, flag: str) -> OpResult:
        return self.call("set_flag", flag=flag)

    def advance_tick(self, n: int = 1) -> OpResult:
        return self.call("advance_tick", n=n)


# --------------------------------------------------------------------------- #
# 后端一：进程内体素世界（零依赖，离线可跑）
# --------------------------------------------------------------------------- #
@dataclass
class McActor:
    id: str
    name: str
    kind: str  # npc | player
    pos: list[int]
    inventory: dict[str, int] = field(default_factory=dict)
    affinity: dict[str, int] = field(default_factory=dict)

    def affinity_to(self, other_id: str) -> int:
        return self.affinity.get(other_id, 50)

    def bump_affinity(self, other_id: str, delta: int) -> None:
        self.affinity[other_id] = max(0, min(100, self.affinity_to(other_id) + delta))


class LocalWorldClient(WorldClient):
    """进程内的体素世界。

    它的职责不是"模拟 Minecraft"，而是**把 Minecraft 的规则形状做对**：
    有坐标、有数量、有昼夜、有合成台、有光源约束。
    形状对了，Agent 面对的问题结构就对了；像素级还原没有意义。

    所有规则都写在 `call()` 的分支里，失败一律返回原因字符串。
    """

    def __init__(self, scenario: dict[str, Any] | None = None) -> None:
        # 世界的搭建逻辑只有一份 —— 就在 _op_configure 里。
        # 构造和后续的 configure 走同一条路，避免"启动时和重置后
        # 世界长得不一样"这种只在重跑时才暴露的问题。
        self.scenario: dict[str, Any] = {}
        self.pois: dict[str, dict[str, Any]] = {}
        self.resource_table: dict[str, dict[str, int]] = {}
        self._op_configure(scenario)

    # ------------------------------------------------------------------ #
    # 状态
    # ------------------------------------------------------------------ #
    def reset(self) -> OpResult:
        self.tick = 0
        self.flags: set[str] = set()
        #: 采集点剩余资源。会随采集减少 —— 资源不是取之不尽的，
        #: 这让"先把木头砍够"这类规划有了意义。
        self.resources: dict[str, dict[str, int]] = {
            key: dict(value) for key, value in self.resource_table.items()
        }
        #: 被放下的方块 {(x, y, z): block}
        self.placed: dict[tuple[int, int, int], str] = {}
        self.utterances: list[dict[str, Any]] = []
        self.actors: dict[str, McActor] = {}
        for spec in self.scenario.get("actors") or []:
            self.actors[spec["id"]] = McActor(
                id=spec["id"],
                name=spec.get("name", spec["id"]),
                kind=spec.get("kind", "npc"),
                pos=list(self._poi_pos(spec.get("start", "village_square"))),
                inventory={k: int(v) for k, v in (spec.get("inventory") or {}).items()},
                affinity=dict(spec.get("affinity") or {}),
            )
        return OpResult(True)

    def _poi_pos(self, poi_id: str) -> list[int]:
        entry = self.pois.get(poi_id)
        if entry is None:
            return [0, 0, 0]
        return list(entry.get("pos") or [0, 0, 0])

    def _poi_at(self, pos: Sequence[int]) -> str | None:
        for poi_id, entry in self.pois.items():
            if list(entry.get("pos") or []) == list(pos):
                return poi_id
        return None

    @property
    def day(self) -> int:
        return self.tick // DAY_TICKS

    def time_of_day(self) -> str:
        return "night" if self.tick % DAY_TICKS >= NIGHT_START else "day"

    def is_night(self) -> bool:
        return self.time_of_day() == "night"

    def lit_positions(self) -> list[list[int]]:
        return [list(pos) for pos, block in self.placed.items() if block == "torch"]

    def _has_light(self, actor: McActor) -> bool:
        """夜里干活需要光源。

        白天不需要 —— 所以火把的作用不是装饰，而是**让夜里也能施工**。
        这条规则让"什么时候做火把"变成一个真的决策，而不是清单上的第三步。
        """
        if not self.is_night():
            return True
        ax, ay, az = actor.pos
        for pos in self.lit_positions():
            bx, by, bz = pos
            if max(abs(ax - bx), abs(ay - by), abs(az - bz)) <= TORCH_LIGHT_RADIUS:
                return True
        return False

    def state(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "day": self.day,
            "time_of_day": self.time_of_day(),
            "pois": {k: dict(v) for k, v in self.pois.items()},
            "actors": {
                a.id: {
                    "id": a.id,
                    "name": a.name,
                    "kind": a.kind,
                    "pos": list(a.pos),
                    "poi": self._poi_at(a.pos),
                    "inventory": dict(a.inventory),
                    "affinity": dict(a.affinity),
                }
                for a in self.actors.values()
            },
            "resources": {k: dict(v) for k, v in self.resources.items()},
            "placed": [{"pos": list(p), "block": b} for p, b in self.placed.items()],
            "flags": sorted(self.flags),
            "utterances": [dict(u) for u in self.utterances],
        }

    # ------------------------------------------------------------------ #
    # 操作分发
    # ------------------------------------------------------------------ #
    def call(self, op: str, **params: Any) -> OpResult:
        handler = getattr(self, f"_op_{op}", None)
        if handler is None:
            return OpResult.fail(f"未知操作：{op}")
        return handler(**params)

    def _actor(self, actor_id: str) -> McActor | None:
        return self.actors.get(actor_id)

    def _op_reset(self, **_kw: Any) -> OpResult:
        return self.reset()

    def _op_configure(self, scenario: dict[str, Any] | None = None, **_kw: Any) -> OpResult:
        """按场景重建世界。幂等 —— 重跑同一份配置得到同一个世界。"""
        self.scenario = dict(scenario or {})
        self.pois = {
            key: dict(value)
            for key, value in (self.scenario.get("pois") or DEFAULT_POIS).items()
        }
        self.resource_table = {
            key: dict(value)
            for key, value in (self.scenario.get("resources") or DEFAULT_RESOURCES).items()
        }
        return self.reset()

    def _op_state(self, **_kw: Any) -> OpResult:
        return OpResult(True, data=self.state())

    def _op_move(self, actor: str, target: str, **_kw: Any) -> OpResult:
        who = self._actor(actor)
        if who is None:
            return OpResult.fail(f"世界上没有 {actor} 这个人")
        if target not in self.pois:
            known = "、".join(f"{k}（{v['name']}）" for k, v in self.pois.items())
            return OpResult.fail(f"没有叫「{target}」的地方。能去的地方：{known}")
        who.pos = self._poi_pos(target)
        return OpResult(True, data={"poi": target, "pos": list(who.pos)})

    def _op_mine(self, actor: str, block: str, **_kw: Any) -> OpResult:
        who = self._actor(actor)
        if who is None:
            return OpResult.fail(f"世界上没有 {actor} 这个人")
        here = self._poi_at(who.pos)
        if here is None:
            return OpResult.fail("这里没有可以采集的东西，先 move_to 到一个采集点")
        available = self.resources.get(here) or {}
        if available.get(block, 0) <= 0:
            have = "、".join(f"{k}×{v}" for k, v in available.items() if v > 0)
            if have:
                return OpResult.fail(f"{self.pois[here]['name']}这里没有{BLOCK_NAMES.get(block, block)}，只有：{have}")
            return OpResult.fail(f"{self.pois[here]['name']}这里已经被采空了")
        # 光照护栏：夜里没有光源就采不了。这条是 Reflection 最该学到的东西。
        if not self._has_light(who):
            return OpResult.fail(
                "天黑了，看不清矿脉。需要先做个火把（torch）放在附近，或者等天亮"
            )
        available[block] -= 1
        who.inventory[block] = who.inventory.get(block, 0) + 1
        return OpResult(
            True,
            data={"block": block, "count": who.inventory[block], "poi": here},
        )

    def _op_craft(self, actor: str, item: str, **_kw: Any) -> OpResult:
        who = self._actor(actor)
        if who is None:
            return OpResult.fail(f"世界上没有 {actor} 这个人")
        recipe = MC_RECIPES.get(item)
        if recipe is None:
            known = "、".join(f"{k}（{v['name']}）" for k, v in MC_RECIPES.items())
            return OpResult.fail(f"没有「{item}」的配方。能做的：{known}")
        here = self._poi_at(who.pos)
        if here != recipe["station"]:
            station_name = self.pois.get(recipe["station"], {}).get("name", recipe["station"])
            return OpResult.fail(
                f"{recipe['name']}要在{station_name}做，你现在在"
                f"{self.pois.get(here, {}).get('name', '别的地方')}，需要先 move_to({recipe['station']})"
            )
        missing = [
            f"{BLOCK_NAMES.get(mat, mat)}×{need - who.inventory.get(mat, 0)}"
            for mat, need in recipe["needs"].items()
            if who.inventory.get(mat, 0) < need
        ]
        if missing:
            need_text = "、".join(
                f"{BLOCK_NAMES.get(m, m)}×{n}" for m, n in recipe["needs"].items()
            )
            return OpResult.fail(
                f"材料不够，还缺：{'、'.join(missing)}。合成{recipe['name']}需要：{need_text}"
            )
        for mat, need in recipe["needs"].items():
            who.inventory[mat] -= need
            if who.inventory[mat] <= 0:
                del who.inventory[mat]
        who.inventory[item] = who.inventory.get(item, 0) + recipe["yields"]
        return OpResult(True, data={"item": item, "count": who.inventory[item]})

    def _op_place(self, actor: str, block: str, target: str, **_kw: Any) -> OpResult:
        who = self._actor(actor)
        if who is None:
            return OpResult.fail(f"世界上没有 {actor} 这个人")
        if who.inventory.get(block, 0) <= 0:
            have = "、".join(f"{BLOCK_NAMES.get(k, k)}×{v}" for k, v in who.inventory.items())
            return OpResult.fail(f"背包里没有{BLOCK_NAMES.get(block, block)}，现在有：{have or '（空）'}")
        here = self._poi_at(who.pos)
        if target not in self.pois:
            return OpResult.fail(f"没有叫「{target}」的地方")
        if here != target:
            return OpResult.fail(
                f"你不在{self.pois[target]['name']}，放不了东西。需要先 move_to({target})"
            )
        pos = tuple(self._poi_pos(target))
        if pos in self.placed:
            return OpResult.fail(f"{self.pois[target]['name']}这里已经有一个{BLOCK_NAMES.get(self.placed[pos], self.placed[pos])}了")
        self.placed[pos] = block
        who.inventory[block] -= 1
        if who.inventory[block] <= 0:
            del who.inventory[block]
        return OpResult(True, data={"block": block, "at": target})

    def _op_consume(self, actor: str, item: str, count: int = 1, **_kw: Any) -> OpResult:
        who = self._actor(actor)
        if who is None:
            return OpResult.fail(f"世界上没有 {actor} 这个人")
        if who.inventory.get(item, 0) < count:
            return OpResult.fail(
                f"{BLOCK_NAMES.get(item, item)}不够，需要 {count} 个，"
                f"只有 {who.inventory.get(item, 0)} 个"
            )
        who.inventory[item] -= count
        if who.inventory[item] <= 0:
            del who.inventory[item]
        return OpResult(True, data={"item": item, "count": who.inventory.get(item, 0)})

    def _op_transfer(
        self, src: str, dst: str, item: str, count: int = 1, **_kw: Any
    ) -> OpResult:
        giver = self._actor(src)
        taker = self._actor(dst)
        if giver is None or taker is None:
            return OpResult.fail("转交双方必须都在世界上")
        if giver.inventory.get(item, 0) < count:
            return OpResult.fail(
                f"{giver.name}手里没有足够的{BLOCK_NAMES.get(item, item)}"
            )
        if list(giver.pos) != list(taker.pos):
            return OpResult.fail(
                f"{taker.name}不在附近，得先 move_to 过去才能把东西交给他"
            )
        giver.inventory[item] -= count
        if giver.inventory[item] <= 0:
            del giver.inventory[item]
        taker.inventory[item] = taker.inventory.get(item, 0) + count
        return OpResult(True, data={"item": item, "to": dst, "count": count})

    def _op_chat(self, actor: str, text: str, **_kw: Any) -> OpResult:
        who = self._actor(actor)
        if who is None:
            return OpResult.fail(f"世界上没有 {actor} 这个人")
        self.utterances.append(
            {
                "tick": self.tick,
                "speaker_id": actor,
                "speaker_name": who.name,
                "text": text,
                "poi": self._poi_at(who.pos),
            }
        )
        return OpResult(True)

    def _op_set_flag(self, flag: str, **_kw: Any) -> OpResult:
        self.flags.add(flag)
        return OpResult(True, data={"flags": sorted(self.flags)})

    def _op_advance_tick(self, n: int = 1, **_kw: Any) -> OpResult:
        self.tick += max(1, int(n))
        return OpResult(True, data={"tick": self.tick, "time_of_day": self.time_of_day()})


# --------------------------------------------------------------------------- #
# 后端二：真实的 Mineflayer 桥（stdio JSON 行）
# --------------------------------------------------------------------------- #
class WorldClientError(RuntimeError):
    """桥进程本身出问题了 —— 这和"操作失败"是两回事，必须区分。

    操作失败（挖不到矿）是**游戏内的事实**，要交给 Reflection 去学；
    桥断了是**工程故障**，应该立刻炸出来，而不是被当成 NPC 的错。
    """


class MineflayerClient(WorldClient):
    """通过 stdio 上的 JSON 行与 Node 桥通信。

    协议极简：一行一个 JSON 请求，一行一个 JSON 响应。
        → {"op": "move", "actor": "ayou", "target": "forest", "id": 1}
        ← {"ok": true, "reason": "", "data": {...}, "id": 1}

    为什么用 stdio 而不是 WebSocket/HTTP：桥是 Agent 的子进程，
    生命周期完全跟着 Agent 走，不需要端口、不需要鉴权、不需要处理
    "上一次没关干净"。对一个演示项目来说这是最不容易出错的形态。
    """

    def __init__(
        self,
        command: Sequence[str] | None = None,
        *,
        cwd: str | None = None,
        timeout: float = 30.0,
        node: str | None = None,
    ) -> None:
        self.timeout = timeout
        self._seq = 0
        self._stderr_tail: list[str] = []
        cmd = list(command) if command else self._default_command(node)
        if not cmd:
            raise WorldClientError(
                "找不到可用的 node，也没有显式指定桥命令。"
                "装上 Node.js，或者用 LocalWorldClient 跑离线模式。"
            )
        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,  # 行缓冲：不 flush 的话请求会卡在管道里
            )
        except OSError as exc:
            raise WorldClientError(f"启动桥进程失败：{exc}") from exc

    @staticmethod
    def _default_command(node: str | None) -> list[str]:
        bridge = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "scripts",
            "mineflayer_bridge.js",
        )
        exe = node or shutil.which("node")
        if not exe:
            return []
        return [exe, bridge]

    # ------------------------------------------------------------------ #
    def call(self, op: str, **params: Any) -> OpResult:
        self._seq += 1
        request = {"op": op, "id": self._seq, **params}
        line = json.dumps(request, ensure_ascii=False)
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise WorldClientError(f"桥进程已退出，无法发送 {op}：{exc}") from exc
        response = self._read_line()
        if response.get("id") != request["id"]:
            # 串包意味着协议已经错位，继续跑下去只会把错误归因到 NPC 身上
            raise WorldClientError(
                f"响应 id 不匹配：期望 {request['id']}，收到 {response.get('id')}"
            )
        return OpResult.from_dict(response)

    def _read_line(self) -> dict[str, Any]:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            raise WorldClientError(
                "桥进程没有响应就退出了。" + self._stderr_hint()
            )
        line = line.strip()
        if not line:
            raise WorldClientError("桥返回了空行")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WorldClientError(f"桥返回的不是 JSON：{line[:200]}") from exc
        if not isinstance(payload, dict):
            raise WorldClientError(f"桥返回的不是对象：{line[:200]}")
        return payload

    def _stderr_hint(self) -> str:
        if self.proc.stderr is None:
            return ""
        try:
            err = self.proc.stderr.read()
        except (OSError, ValueError):
            return ""
        if err:
            return f"\n桥的 stderr：\n{err.strip()[-500:]}"
        return ""

    def close(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
