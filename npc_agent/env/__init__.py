"""环境层：把"世界"从"智能体"里解耦出来。"""

from .base import Environment, ToolSpec
from .star_isle import LOCATIONS, KNOWLEDGE, RECIPES, StarIsleEnv

__all__ = ["Environment", "ToolSpec", "StarIsleEnv", "LOCATIONS", "RECIPES", "KNOWLEDGE"]
