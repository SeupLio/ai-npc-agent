"""环境层：把"世界"从"智能体"里解耦出来。

## 环境注册表

场景 YAML 用 `env:` 字段选世界，默认 `star-isle`。
已有场景一行都不用改 —— 默认值就是它们原来跑的那个环境。

    # configs/scenarios/village.yaml
    env: minecraft

注册表让"加一个世界"变成一个纯粹的**加法**：写一个 Environment 子类，
在 `ENVIRONMENTS` 里登记一行，然后任何场景都能通过配置切过去。
Agent 侧不需要知道有几个世界存在。

`build_env()` 是唯一的构造入口 —— CLI、评测 harness、测试都走它，
这样"某个场景到底跑在哪个世界上"只有一处答案。
"""

from __future__ import annotations

from typing import Any, Callable

from .base import Environment, ToolSpec
from .conditions import ConditionContext, condition_met, refresh_objectives
from .mc_client import (
    BLOCK_NAMES,
    MC_RECIPES,
    LocalWorldClient,
    MineflayerClient,
    OpResult,
    WorldClient,
    WorldClientError,
)
from .minecraft import MinecraftEnv
from .star_isle import LOCATIONS, KNOWLEDGE, RECIPES, StarIsleEnv

DEFAULT_ENV = "star-isle"


def _build_star_isle(
    scenario: dict[str, Any],
    npc_id: str,
    npc_name: str,
    cast: list[dict[str, Any]] | None,
) -> Environment:
    return StarIsleEnv(scenario, npc_id, npc_name, cast=cast)


def _build_minecraft(
    scenario: dict[str, Any],
    npc_id: str,
    npc_name: str,
    cast: list[dict[str, Any]] | None,
) -> Environment:
    # 离线优先：默认用进程内体素世界。想接真实服务端时，
    # 显式传 client=MineflayerClient() 即可 —— 上面的代码完全一样。
    return MinecraftEnv(scenario, npc_id, npc_name, cast=cast)


#: 环境名 → 构造函数。加世界只需要在这里加一行。
ENVIRONMENTS: dict[str, Callable[..., Environment]] = {
    "star-isle": _build_star_isle,
    "minecraft": _build_minecraft,
}

#: 环境名 → 给人看的名字。CLI / 报告里用它，避免把展示文案硬编码在界面上。
ENV_LABELS: dict[str, str] = {
    "star-isle": "星屿咖啡屋",
    "minecraft": "Minecraft 体素世界",
}


def env_label(name: str) -> str:
    return ENV_LABELS.get(name, name)


def env_name_of(scenario: dict[str, Any]) -> str:
    """场景跑在哪个世界上。没写就是默认环境。"""
    return str(scenario.get("env") or DEFAULT_ENV)


def build_env(
    scenario: dict[str, Any],
    npc_id: str = "",
    npc_name: str = "",
    *,
    cast: list[dict[str, Any]] | None = None,
    client: WorldClient | None = None,
) -> Environment:
    """按场景配置构造环境。

    未知的环境名**直接报错并列出可用取值**，而不是静默退回默认环境：
    静默退回会让一份写错配置的场景跑在错误的世界里，
    然后所有评测数字都是错的 —— 而错误信息只是"目标没完成"。
    """
    name = env_name_of(scenario)
    builder = ENVIRONMENTS.get(name)
    if builder is None:
        known = "、".join(sorted(ENVIRONMENTS))
        raise ValueError(f"未知环境「{name}」。可用环境：{known}")
    if name == "minecraft":
        return MinecraftEnv(
            scenario, npc_id, npc_name, cast=cast, client=client
        )
    if client is not None:
        raise ValueError(f"环境「{name}」不接受外部 WorldClient")
    return builder(scenario, npc_id, npc_name, cast)


__all__ = [
    "Environment",
    "ToolSpec",
    "StarIsleEnv",
    "MinecraftEnv",
    "LOCATIONS",
    "RECIPES",
    "KNOWLEDGE",
    "BLOCK_NAMES",
    "MC_RECIPES",
    "ConditionContext",
    "condition_met",
    "refresh_objectives",
    "WorldClient",
    "WorldClientError",
    "LocalWorldClient",
    "MineflayerClient",
    "OpResult",
    "ENVIRONMENTS",
    "DEFAULT_ENV",
    "ENV_LABELS",
    "env_label",
    "env_name_of",
    "build_env",
]
