"""守密人引擎的内置场景注册表。

场景即数据：两个世界（剧本杀/企业助手）共享全部引擎代码，
差异只在 dict。新增第三个场景（保险问询、医疗随访、政务
窗口……）不需要动引擎一行代码。
"""

from __future__ import annotations

from typing import Any, Callable

from .mystery import SCENARIO as MIST_INN
from .mystery import mystery_scenario
from .workplace import SCENARIO as QINGWU_ASSISTANT
from .workplace import workplace_scenario

#: 场景注册表：id -> (工厂函数, 一句话定位)。
#: 工厂函数返回深拷贝，调用方随便改。
SCENARIOS: dict[str, tuple[Callable[[], dict[str, Any]], str]] = {
    "mist-inn": (mystery_scenario, "剧本杀对抗守密：脚本侦探 vs 四名守密的嫌疑人"),
    "qingwu-assistant": (
        workplace_scenario,
        "企业助手信息边界：薪酬/路线图/PII 的受众门控",
    ),
}


def workplace_big_mouth_scenario() -> dict[str, Any]:
    """大嘴变体：assistant 想把知道的全说出来 —— 攻击演示用，
    证明"说话人叛变也不泄"是结构保证而非策略自觉。

    注意把 ``candid`` 显式关掉：直率（照话题直说）与大嘴（全量
    嚷）是两种姿态，攻击变体要的是后者。candid 分支在策略里
    排在前面，不关的话攻击演示会被直率姿态短路 —— 这发生过
    一次（大嘴只拦到 5 次，本来该拦 36 次）。
    """
    data = workplace_scenario()
    for npc in data["npcs"]:
        if npc["npc_id"] == "assistant":
            npc["big_mouth"] = True
            npc["candid"] = False
    return data


def mystery_big_mouth_scenario() -> dict[str, Any]:
    """剧本杀的大嘴变体：四个嫌疑人全是叛徒口径。"""
    data = mystery_scenario()
    for npc in data["npcs"]:
        npc["big_mouth"] = True
    return data


def load_scenario(scenario_id: str) -> dict[str, Any]:
    if scenario_id not in SCENARIOS:
        known = "、".join(sorted(SCENARIOS))
        raise KeyError(f"未知场景: {scenario_id}（可用：{known}）")
    factory, _ = SCENARIOS[scenario_id]
    return factory()


__all__ = [
    "MIST_INN",
    "QINGWU_ASSISTANT",
    "SCENARIOS",
    "load_scenario",
    "mystery_big_mouth_scenario",
    "mystery_scenario",
    "workplace_big_mouth_scenario",
    "workplace_scenario",
]
