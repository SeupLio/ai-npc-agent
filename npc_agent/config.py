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
    # 规划调用的预算。推理模型会先输出思维链，预算要给够 ——
    # 规划和台词是两次独立的调用，各吃各的预算，不能共用一个数。
    max_tokens: int = 4096
    # 单句台词的预算。别按"一句话 90 字"去估 —— 推理模型的思维链和正式回答
    # **共用**这个预算。给 512 会让台词说到一半被截断（"是啊，阳光都"）；
    # 给 1024 会让整条调用返回**空内容**。
    #
    # 实测（kimi-k2.7-code，228 条跑批）：思维链长度 3394~3905 字，
    # 1024 的预算下 150 条里有 7 条返回空内容（finish_reason=length）→
    # 框架退回模板台词 → 那 7 条的分数衡量的就不是模型了。
    # 同一端点上裁判用 4096 能正常处理 4043 字的思维链，所以这里也用 4096。
    # 注意这是**上限不是目标**：模型不写那么长就不会多花钱。
    speech_max_tokens: int = 4096
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
    # 发言占比上限：超过且没被点名就让出话头。
    # 之前 dialogue.py 和 tools.py 各自硬编码了一份 0.62，改一处漏一处，
    # 现在收拢成一个配置项。
    npc_share_ceiling: float = 0.62

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
