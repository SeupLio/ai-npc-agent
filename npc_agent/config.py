"""运行时配置与 YAML 加载。

所有可调参数集中在这里，方便做消融实验（例如调 memory_top_k 看记忆召回率变化）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


@dataclass
class RuntimeConfig:
    """一次运行的全部可调参数。"""

    # --- LLM ---
    llm_provider: str = "null"          # null | openai-compat | scripted
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048            # 推理模型会先输出思维链，预算要给够
    # 单句台词的预算。别按"一句话 90 字"去估 —— 推理模型的思维链和正式回答
    # 共用这个预算，思维链通常就要 300~500 token。给 512 会让台词在说到
    # 一半时被截断（"是啊，阳光都"），所以这里按"思维链 + 完整台词"来给。
    speech_max_tokens: int = 1024
    use_llm_planner: bool = True      # 关掉可做消融：只用启发式规划
    use_llm_speech: bool = True       # 关掉可做消融：只用模板台词

    # --- Memory ---
    memory_top_k: int = 6               # 每轮注入 prompt 的记忆条数
    memory_consolidate_at: int = 24     # episodic 超过多少条触发巩固
    memory_half_life: float = 40.0      # 时间衰减半衰期（tick）
    memory_strategy: str = "hybrid"     # hybrid | recency | lexical | importance | none

    # --- Planning ---
    max_steps_per_turn: int = 3         # 单轮最多执行几个工具，防止失控
    max_plan_retries: int = 1           # 单步失败后的重规划次数

    # --- Reflection ---
    reflect_every: int = 6              # 每 N 轮强制反思一次

    # --- Dialogue ---
    idle_ticks_before_proactive: int = 2  # 冷场多少轮后主动发起话题

    verbose: bool = False

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        """从环境变量构造，便于在没有配置文件的情况下快速切模型。"""
        return cls(
            llm_provider=os.getenv("NPC_AGENT_PROVIDER", "null"),
            base_url=os.getenv("NPC_AGENT_BASE_URL", ""),
            api_key=os.getenv("NPC_AGENT_API_KEY", ""),
            model=os.getenv("NPC_AGENT_MODEL", ""),
            temperature=float(os.getenv("NPC_AGENT_TEMPERATURE", "0.7")),
            memory_strategy=os.getenv("NPC_AGENT_MEMORY_STRATEGY", "hybrid"),
            verbose=os.getenv("NPC_AGENT_VERBOSE", "").lower() in ("1", "true", "yes"),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("api_key"):
            data["api_key"] = "***"
        return data


def load_yaml(path: str | Path) -> dict[str, Any]:
    """加载 YAML。相对路径按 configs/ 解析。"""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = CONFIG_DIR / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"找不到配置文件: {candidate}")
    with candidate.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_persona(persona_id: str) -> dict[str, Any]:
    return load_yaml(Path("personas") / f"{persona_id}.yaml")


def load_scenario(scenario_id: str) -> dict[str, Any]:
    return load_yaml(Path("scenarios") / f"{scenario_id}.yaml")


def list_scenarios() -> list[str]:
    folder = CONFIG_DIR / "scenarios"
    if not folder.exists():
        return []
    return sorted(p.stem for p in folder.glob("*.yaml"))
