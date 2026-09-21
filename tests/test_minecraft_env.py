"""Minecraft 适配器的测试。

这个文件里最重要的不是"能挖到矿"，而是几条**结构性**断言：

1. 同一个 NPCAgent 类，不改一行代码，在两个结构完全不同的世界上都能跑完任务。
   这是"环境无关"从一句主张变成一条可执行断言的唯一方式。

2. WorldClient 的两个后端跑同一套契约用例。
   契约只写了一遍，所以"两个后端行为一致"不需要靠人工比对来保证。

3. 护栏失败一律返回**一句人话**，而不是抛异常 —— 那句话是 Reflection 的输入。
"""

from __future__ import annotations

import itertools
import json
import sys
import textwrap

import pytest

from npc_agent.agent import NPCAgent
from npc_agent.cast import build_cast
from npc_agent.config import load_persona, load_scenario
from npc_agent.env import ENVIRONMENTS, build_env, env_name_of
from npc_agent.env.base import Environment
from npc_agent.env.mc_client import (
    DAY_TICKS,
    MC_RECIPES,
    LocalWorldClient,
    MineflayerClient,
    WorldClient,
    WorldClientError,
)
from npc_agent.env.minecraft import MinecraftEnv
from npc_agent.env.star_isle import StarIsleEnv
from npc_agent.llm import NullLLM
from npc_agent.modules.persona import Persona
from npc_agent.types import ActionCall

# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #
TWO_ACTORS = {
    "actors": [
        {"id": "ayan", "name": "阿岩", "kind": "npc", "start": "forest"},
        {"id": "player_a", "name": "小鹿", "kind": "player", "start": "village_square"},
    ]
}


@pytest.fixture
def client() -> LocalWorldClient:
    return LocalWorldClient(TWO_ACTORS)


def village_client() -> LocalWorldClient:
    """按 village 场景配好演员与地理的进程内世界。

    直接 LocalWorldClient(load_scenario("village")) 是不行的：
    场景配置里的人是 `npc:` + `players:`，而世界后端要的是 `actors:`。
    这个翻译由 MinecraftEnv 负责 —— 所以这里从环境拿 client，
    走的是和真实运行完全一样的构造路径。
    """
    return MinecraftEnv(load_scenario("village"), "ayan", "阿岩").client


@pytest.fixture
def village() -> dict:
    return load_scenario("village")


@pytest.fixture
def mc_env(village: dict) -> MinecraftEnv:
    return MinecraftEnv(village, "ayan", "阿岩")


def call(tool: str, **args) -> ActionCall:
    return ActionCall(tool, args)


def ayan_persona() -> Persona:
    return Persona.from_dict(load_persona("ayan"))


# --------------------------------------------------------------------------- #
# WorldClient 契约一致性
# --------------------------------------------------------------------------- #
class ContractClient(WorldClient):
    """把操作转发给一个 LocalWorldClient，但强制走 JSON 序列化。

    这是给 MineflayerClient 做的"等价替身"：如果 MinecraftEnv 不小心依赖了
    LocalWorldClient 的某个 Python 对象（比如直接把 tuple 当 key 用），
    经过一次 JSON 往返就会露馅 —— 而真实的桥一定经过这次往返。
    """

    def __init__(self, inner: LocalWorldClient) -> None:
        self.inner = inner

    def call(self, op: str, **params):
        from npc_agent.env.mc_client import OpResult

        payload = json.loads(json.dumps({"op": op, **params}, ensure_ascii=False))
        op_name = payload.pop("op")
        result = self.inner.call(op_name, **payload)
        return OpResult.from_dict(json.loads(json.dumps(result.to_dict(), ensure_ascii=False)))


@pytest.mark.parametrize("transport", ["local", "json-roundtrip"])
def test_contract_is_transport_independent(transport: str) -> None:
    """同一套操作序列，两种传输，得到同一个世界状态。

    "同一个契约、两种传输"如果只写在文档里，它就会慢慢变成谎话。
    这条测试把它变成可执行的：任何一端加了字段却忘了另一端，这里就红。
    """
    inner = village_client()
    world: WorldClient = inner if transport == "local" else ContractClient(inner)

    world.reset()
    world.move("ayan", "forest")
    assert world.mine("ayan", "oak_log").ok
    world.move("ayan", "workshop")
    assert world.craft("ayan", "planks").ok
    assert world.craft("ayan", "stick").ok
    world.move("ayan", "cave_mouth")
    assert world.mine("ayan", "coal").ok
    world.move("ayan", "workshop")
    assert world.craft("ayan", "torch").ok
    world.move("ayan", "cave_mouth")
    assert world.place("ayan", "torch", "cave_mouth").ok
    world.set_flag("cave_lit")

    state = world.state()
    assert state["actors"]["ayan"]["inventory"].get("torch") == 3
    assert any(e["block"] == "torch" for e in state["placed"])
    assert "cave_lit" in state["flags"]


def test_every_op_returns_json_serializable_data(client: LocalWorldClient) -> None:
    """每个操作的返回都必须能过一遍 JSON —— 它要穿过 stdio。"""
    ops = [
        ("reset", {}),
        ("state", {}),
        ("move", {"actor": "ayan", "target": "workshop"}),
        ("mine", {"actor": "ayan", "block": "oak_log"}),
        ("craft", {"actor": "ayan", "item": "planks"}),
        ("place", {"actor": "ayan", "block": "torch", "target": "workshop"}),
        ("consume", {"actor": "ayan", "item": "oak_log", "count": 1}),
        ("transfer", {"src": "ayan", "dst": "player_a", "item": "oak_log", "count": 1}),
        ("chat", {"actor": "ayan", "text": "行。"}),
        ("set_flag", {"flag": "x"}),
        ("advance_tick", {"n": 1}),
    ]
    for op, params in ops:
        payload = client.call(op, **params).to_dict()
        json.dumps(payload, ensure_ascii=False)  # 不抛就算过


def test_unknown_op_fails_gracefully(client: LocalWorldClient) -> None:
    result = client.call("teleport_to_moon")
    assert result.ok is False
    assert "未知操作" in result.reason


# --------------------------------------------------------------------------- #
# 世界规则
# --------------------------------------------------------------------------- #
def test_mining_works_during_the_day(client: LocalWorldClient) -> None:
    client.move("ayan", "forest")
    result = client.mine("ayan", "oak_log")
    assert result.ok
    assert result.data["count"] == 1


def test_mining_at_night_requires_light(client: LocalWorldClient) -> None:
    """昼夜循环不是装饰 —— 它让"什么时候做火把"变成真的决策。"""
    client.move("ayan", "forest")
    client.advance_tick(DAY_TICKS - DAY_TICKS // 3)  # 推进到夜里
    assert client.is_night()

    blocked = client.mine("ayan", "oak_log")
    assert blocked.ok is False
    assert "天黑" in blocked.reason
    # 理由必须告诉它怎么办，否则 Reflection 学不到东西
    assert "火把" in blocked.reason


def test_a_nearby_torch_lifts_the_night_restriction(client: LocalWorldClient) -> None:
    """火把的作用是让夜里也能干活 —— 这正是任务里要做火把的原因。"""
    client.move("ayan", "forest")
    client.advance_tick(DAY_TICKS - DAY_TICKS // 3)
    assert client.mine("ayan", "oak_log").ok is False

    client.actors["ayan"].inventory["torch"] = 1
    assert client.place("ayan", "torch", "forest").ok
    assert client.mine("ayan", "oak_log").ok is True


def test_day_and_night_alternate(client: LocalWorldClient) -> None:
    seen = set()
    for _ in range(DAY_TICKS * 2):
        seen.add(client.time_of_day())
        client.advance_tick()
    assert seen == {"day", "night"}


def test_resources_are_finite(client: LocalWorldClient) -> None:
    client.move("ayan", "forest")
    mined = sum(1 for _ in range(50) if client.mine("ayan", "oak_log").ok)
    assert mined == 8  # 场景里配的 oak_log: 8
    assert "采空" in client.mine("ayan", "oak_log").reason


def test_craft_requires_the_right_station(client: LocalWorldClient) -> None:
    client.actors["ayan"].inventory["oak_log"] = 2
    client.move("ayan", "forest")
    result = client.craft("ayan", "planks")
    assert result.ok is False
    assert "木工台" in result.reason
    assert "move_to(workshop)" in result.reason


def test_craft_reports_the_missing_materials(client: LocalWorldClient) -> None:
    client.move("ayan", "workshop")
    result = client.craft("ayan", "torch")
    assert result.ok is False
    assert "煤炭" in result.reason
    assert "木棍" in result.reason


def test_recipes_are_authentic_minecraft(client: LocalWorldClient) -> None:
    """1 原木 → 4 木板，2 木板 → 4 木棍。

    保留这种"进 1 出 4"的不对称是有意的：如果配方都是 1:1，
    背包就退化成一个集合，"数量"这条能力就永远测不出来。
    """
    assert MC_RECIPES["planks"]["yields"] == 4
    assert MC_RECIPES["stick"]["needs"] == {"planks": 2}
    assert MC_RECIPES["torch"]["needs"] == {"coal": 1, "stick": 1}


def test_quantities_actually_accumulate(client: LocalWorldClient) -> None:
    client.actors["ayan"].inventory["oak_log"] = 1
    client.move("ayan", "workshop")
    client.craft("ayan", "planks")
    assert client.actors["ayan"].inventory["planks"] == 4
    client.craft("ayan", "stick")
    # 用了 2 块木板，还剩 2 块
    assert client.actors["ayan"].inventory["planks"] == 2
    assert client.actors["ayan"].inventory["stick"] == 4


def test_place_needs_the_item_and_the_right_spot(client: LocalWorldClient) -> None:
    client.move("ayan", "forest")
    assert client.place("ayan", "torch", "forest").ok is False  # 背包里没有

    client.actors["ayan"].inventory["torch"] = 1
    assert client.place("ayan", "torch", "cave_mouth").ok is False  # 人不在那
    assert client.place("ayan", "torch", "forest").ok is True


def test_place_is_not_idempotent(client: LocalWorldClient) -> None:
    """同一个地点已经有火把了，再插一支应该被拒绝并说明原因。

    不是"无所谓地成功" —— 那样 NPC 会以为自己的动作生效了，
    而世界里什么都没变，之后所有推理都建立在假前提上。
    """
    client.actors["ayan"].inventory["torch"] = 2
    client.move("ayan", "forest")
    assert client.place("ayan", "torch", "forest").ok is True
    second = client.place("ayan", "torch", "forest")
    assert second.ok is False
    assert "已经有一个火把" in second.reason


def test_transfer_requires_colocation(client: LocalWorldClient) -> None:
    client.actors["ayan"].inventory["torch"] = 1
    far = client.transfer("ayan", "player_a", "torch")
    assert far.ok is False
    assert "不在附近" in far.reason

    client.move("ayan", "village_square")
    assert client.transfer("ayan", "player_a", "torch").ok is True


def test_guardrail_messages_are_human_readable(client: LocalWorldClient) -> None:
    """护栏的价值不在于拦住 NPC，而在于告诉它为什么被拦。

    被拦之后那句话会直接进 Reflection —— 所以它必须是一句人话，
    而且**要指出下一步该做什么**。只说"不行"的护栏等于没有护栏。
    """
    client.move("ayan", "forest")
    cases = [
        (client.move("ayan", "moon"), "能去的地方"),        # 列出合法取值
        (client.craft("ayan", "diamond_sword"), "能做的"),  # 列出合法取值
        (client.mine("ayan", "cobblestone"), "只有"),       # 说清楚这里有什么
    ]
    for result, hint in cases:
        assert result.ok is False
        assert hint in result.reason
        assert len(result.reason) > 8
        assert "Traceback" not in result.reason
        assert "Exception" not in result.reason

    # 站错工位时必须给出 move_to 的下一步
    client.actors["ayan"].inventory["oak_log"] = 1
    wrong_station = client.craft("ayan", "planks")
    assert "move_to(workshop)" in wrong_station.reason


# --------------------------------------------------------------------------- #
# MinecraftEnv：翻译
# --------------------------------------------------------------------------- #
def test_observation_uses_the_standard_schema(mc_env: MinecraftEnv) -> None:
    """字段名必须和星屿咖啡屋完全一致 —— 这是 Agent 一行不改的前提。"""
    obs = mc_env.observe("ayan")
    for key in [
        "tick", "self", "locations", "present_actors", "visible_actors",
        "visible_items", "recent_utterances", "world_flags", "objectives",
        "activities", "affinity",
    ]:
        assert key in obs, f"观测缺少 {key}"

    assert set(obs["self"]) >= {"id", "name", "loc", "loc_name", "inventory"}
    # 多出来的是体素世界特有的事实，不影响 Agent 读已知字段
    assert obs["time_of_day"] in {"day", "night"}


def test_inventory_carries_quantities(mc_env: MinecraftEnv) -> None:
    mc_env.client.actors["ayan"].inventory.update({"oak_log": 3, "torch": 1})
    inventory = mc_env.observe("ayan")["self"]["inventory"]
    assert "橡木原木×3" in inventory
    assert "火把×1" in inventory


def test_tools_use_minecraft_verbs(mc_env: MinecraftEnv) -> None:
    names = {spec.name for spec in mc_env.tool_specs("ayan")}
    assert {"move_to", "mine", "craft", "place", "transfer"} <= names
    # 咖啡屋的动词不该出现在这里
    assert "take_item" not in names
    assert "give_item" not in names


def test_tool_descriptions_list_valid_values(mc_env: MinecraftEnv) -> None:
    """把合法取值写进描述，是最省事也最有效的防幻觉手段。"""
    specs = {spec.name: spec for spec in mc_env.tool_specs("ayan")}
    move_desc = specs["move_to"].description + str(specs["move_to"].params)
    for poi in ["forest", "workshop", "cave_mouth", "village_square"]:
        assert poi in move_desc
    assert "player_a" in specs["transfer"].description + str(specs["transfer"].params)


def test_set_flag_is_whitelisted(mc_env: MinecraftEnv) -> None:
    ok = mc_env.dispatch("ayan", call("set_flag", key="cave_lit", value="1"))
    assert ok.ok
    bad = mc_env.dispatch("ayan", call("set_flag", key="become_admin", value="1"))
    assert bad.ok is False
    assert "不允许设置标记" in bad.detail


# --------------------------------------------------------------------------- #
# 知识边界：安全机制必须两个世界都成立
# --------------------------------------------------------------------------- #
def test_available_topics_respects_unlock_conditions(mc_env: MinecraftEnv) -> None:
    """requires 未满足的话题不能出现在可聊清单里 —— 这是"不剧透"的实现。"""
    before = mc_env.available_topics("ayan")
    assert "stonemasonry" in before
    assert "cave_secret" not in before  # 需要 cave_lit

    mc_env.dispatch("ayan", call("set_flag", key="cave_lit", value="1"))
    assert "cave_secret" in mc_env.available_topics("ayan")


def test_tell_fact_refuses_a_locked_topic_with_a_reason(mc_env: MinecraftEnv) -> None:
    """拒绝必须说明"为什么现在不能说"，而不是干巴巴的"不行"。"""
    blocked = mc_env.dispatch("ayan", call("tell_fact", topic="cave_secret"))
    assert blocked.ok is False
    assert "剧透" in blocked.detail
    assert "cave_lit" in blocked.detail

    mc_env.dispatch("ayan", call("set_flag", key="cave_lit", value="1"))
    allowed = mc_env.dispatch("ayan", call("tell_fact", topic="cave_secret"))
    assert allowed.ok is True
    assert "矿脉" in allowed.detail


def test_tell_fact_rejects_unknown_topics(mc_env: MinecraftEnv) -> None:
    result = mc_env.dispatch("ayan", call("tell_fact", topic="quantum_physics"))
    assert result.ok is False
    assert "能聊的" in result.detail


def test_knowledge_boundary_matches_across_worlds() -> None:
    """同一个 requires 条件，两个世界给出同一种拒绝。

    "不剧透"是任务设计层面的保证，不是某个世界的能力 ——
    如果只有咖啡屋守这条规矩，那换环境就等于把安全机制丢了。
    """
    star = StarIsleEnv(load_scenario("icebreaker"), "ayou", "阿柚")
    voxel = MinecraftEnv(load_scenario("village"), "ayan", "阿岩")

    # 咖啡屋的 hidden_menu 需要 hidden_menu_unlocked；体素世界的 cave_secret 需要 cave_lit
    assert "hidden_menu" not in star.available_topics("ayou")
    assert "cave_secret" not in voxel.available_topics("ayan")

    star_blocked = star.dispatch("ayou", ActionCall("tell_fact", {"topic": "hidden_menu"}))
    voxel_blocked = voxel.dispatch("ayan", ActionCall("tell_fact", {"topic": "cave_secret"}))
    assert star_blocked.ok is False and voxel_blocked.ok is False
    # 两边都说了"剧透"这件事，也都指出了缺哪个条件
    for detail, condition in [
        (star_blocked.detail, "hidden_menu_unlocked"),
        (voxel_blocked.detail, "cave_lit"),
    ]:
        assert "剧透" in detail
        assert condition in detail


def test_dispatch_never_raises(mc_env: MinecraftEnv) -> None:
    for action in [
        call("mine"),
        call("mine", block=""),
        call("place"),
        call("craft", item=""),
        call("transfer", item="x", player="nobody"),
        call("no_such_tool"),
        call("move_to", location=""),
    ]:
        result = mc_env.dispatch("ayan", action)
        assert result.ok is False
        assert result.detail


def test_objective_completes_on_world_state(mc_env: MinecraftEnv) -> None:
    assert mc_env.objectives_status()["light_the_cave"] == "pending"
    mc_env.dispatch("ayan", call("set_flag", key="cave_lit", value="1"))
    assert mc_env.objectives_status()["light_the_cave"] == "done"


def test_condition_context_exposes_quantities(mc_env: MinecraftEnv) -> None:
    mc_env.client.actors["ayan"].inventory["oak_log"] = 5
    ctx = mc_env.condition_context()
    assert ctx.count_item("ayan", "oak_log") == 5
    assert ctx.count_item("ayan", "torch") == 0
    assert "player_a" in ctx.player_ids


def test_reset_restores_the_world(mc_env: MinecraftEnv) -> None:
    mc_env.dispatch("ayan", call("set_flag", key="cave_lit", value="1"))
    mc_env.reset()
    assert mc_env.objectives_status()["light_the_cave"] == "pending"
    assert mc_env.snapshot()["world_flags"] == []


def test_snapshot_uses_the_same_keys_as_the_coffee_shop() -> None:
    """快照的键名必须跨环境一致 —— 评测 harness 的断言是共用的。

    这里曾经叫 `flags`，而 harness 读的是 `world_flags`，
    结果是体素世界的用例里"未达成标记"永远为真、静默通过。
    """
    from npc_agent.env.star_isle import StarIsleEnv

    star = StarIsleEnv(load_scenario("tutorial"), "ayou", "阿柚").snapshot()
    voxel = MinecraftEnv(load_scenario("village"), "ayan", "阿岩").snapshot()
    for key in ["tick", "scenario", "actors", "world_flags", "objectives"]:
        assert key in star, f"咖啡屋快照缺少 {key}"
        assert key in voxel, f"体素世界快照缺少 {key}"
    # actors 里的字段名也要对得上：harness 读 actor["inventory"] / actor["loc"]
    assert {"name", "kind", "loc", "inventory", "affinity"} <= set(
        next(iter(voxel["actors"].values()))
    )


# --------------------------------------------------------------------------- #
# 环境注册表
# --------------------------------------------------------------------------- #
def test_scenarios_default_to_star_isle() -> None:
    """已有场景一行都不用改 —— 默认值就是它们原来跑的那个环境。"""
    assert env_name_of({"id": "tutorial"}) == "star-isle"
    assert isinstance(build_env(load_scenario("tutorial")), StarIsleEnv)


def test_village_selects_minecraft(village: dict) -> None:
    assert env_name_of(village) == "minecraft"
    assert isinstance(build_env(village), MinecraftEnv)


def test_unknown_environment_fails_loudly() -> None:
    """静默退回默认环境会让一份写错的配置跑在错误的世界里，
    然后所有评测数字都是错的 —— 而错误信息只是"目标没完成"。"""
    with pytest.raises(ValueError) as exc:
        build_env({"id": "x", "env": "nope"})
    assert "nope" in str(exc.value)
    assert "minecraft" in str(exc.value)


def test_registry_contains_both_worlds() -> None:
    assert set(ENVIRONMENTS) >= {"star-isle", "minecraft"}


def test_every_registered_env_is_an_environment() -> None:
    for name, builder in ENVIRONMENTS.items():
        env = builder(load_scenario("tutorial"), "ayou", "阿柚", None)
        assert isinstance(env, Environment), name


# --------------------------------------------------------------------------- #
# 最重要的一条：同一个 Agent，两个世界
# --------------------------------------------------------------------------- #
def test_observation_schemas_match_across_worlds() -> None:
    """两个世界的观测 schema 必须一致 —— 这是 Agent 一行不改的前提。

    只比**键名**，不比内容：内容本来就该不同（那是两个世界的事实），
    键名相同才意味着 Agent 不需要写 if。
    """
    from npc_agent.env.star_isle import StarIsleEnv

    star = StarIsleEnv(load_scenario("tutorial"), "ayou", "阿柚")
    voxel = MinecraftEnv(load_scenario("village"), "ayan", "阿岩")
    star_obs = star.observe("ayou")
    voxel_obs = voxel.observe("ayan")

    # 子集关系，不是相等关系：更丰富的世界**可以多加事实**
    # （Minecraft 多给了坐标 pos 和 time_of_day），但绝不能少给 ——
    # 少一个键就意味着 Agent 得写 if，抽象就漏了。
    assert set(star_obs) <= set(voxel_obs)
    assert set(star_obs["self"]) <= set(voxel_obs["self"])
    for key in ["locations", "present_actors", "visible_actors", "visible_items",
                "recent_utterances", "world_flags", "objectives", "affinity"]:
        assert key in star_obs and key in voxel_obs
    # 多出来的那些是体素世界独有的事实，不是契约的一部分
    assert "pos" in voxel_obs["self"] and "pos" not in star_obs["self"]


def test_the_same_agent_class_runs_on_both_worlds() -> None:
    """环境无关的可执行断言 —— 这个适配器存在的全部理由。

    同一份人设、同一个 NPCAgent 类、同一行构造代码、同一句输入，
    在咖啡屋和体素世界里给出**同一个决策**；而它们读到的世界确实不同。
    两件事缺一不可：

      只证明"决策相同" → 可能两边其实都没读世界，测试是空的
      只证明"世界不同" → 没证明 Agent 能适应，等于没说

    所以下面两段断言是配对出现的。
    """
    persona = ayan_persona()
    observed: dict[str, tuple] = {}
    for scenario_id in ["tutorial", "village"]:
        scenario = load_scenario(scenario_id)
        env = build_env(scenario, cast=[{"id": persona.id, "name": persona.name}])
        # 同一个类，同一行构造代码，两个结构完全不同的世界
        agent = NPCAgent(persona, env, scenario, NullLLM())
        turn = agent.step(
            env.record_player_utterance("player_a", "阿岩，能帮我个忙吗？")
        )
        observed[scenario_id] = (turn, agent, env)

    star_turn, star_agent, star_env = observed["tutorial"]
    voxel_turn, voxel_agent, voxel_env = observed["village"]

    # ① 同一个决策
    assert star_turn.say == voxel_turn.say
    assert star_turn.decision_reason == voxel_turn.decision_reason

    # ② 但读到的世界确实不同 —— 否则①可能只是"两边都没读世界"
    assert isinstance(star_env, StarIsleEnv)
    assert isinstance(voxel_env, MinecraftEnv)
    assert star_agent.state.location == "counter"        # 咖啡屋的吧台
    assert voxel_agent.state.location == "village_square"  # 体素世界的村口
    assert star_agent.state.location_name == "吧台"
    assert voxel_agent.state.location_name == "村口广场"
    # 两边都把背包读成了一个列表（Minecraft 侧是带数量的字符串）
    assert isinstance(star_agent.state.inventory, list)
    assert isinstance(voxel_agent.state.inventory, list)


def test_agent_reaches_the_goal_in_the_voxel_world() -> None:
    """离线路径在体素世界里真的能把任务做完。

    走的是 objectives.steps —— 也就是说，任务是用**同一套工具词汇**
    描述给两个世界的，规划器不需要为 Minecraft 写任何特例。
    """
    scenario = load_scenario("village")
    cast = build_cast(scenario, NullLLM())
    env = cast.env

    cast.step(env.record_player_utterance("player_a", "阿岩，天快黑了，洞口得点个火把。"))
    for _ in range(16):
        cast.step()

    assert env.objectives_status()["light_the_cave"] == "done"
    placed = env.snapshot()["placed"]
    assert any(e["block"] == "torch" for e in placed)
    # 真的在洞口，而不是随便一个地方
    assert [10, 0, -6] in [e["pos"] for e in placed]


def test_agent_uses_quantity_arithmetic_to_finish() -> None:
    """完成任务的过程中，数量语义必须真的被用到。

    1 原木 → 4 木板 → 4 木棍 → 4 火把，插掉 1 支还剩 3 支。
    如果背包退化成集合，这些数字就对不上。
    """
    scenario = load_scenario("village")
    cast = build_cast(scenario, NullLLM())
    env = cast.env
    cast.step(env.record_player_utterance("player_a", "阿岩，洞口得点个火把。"))
    for _ in range(16):
        cast.step()

    inventory = env.snapshot()["actors"]["ayan"]["inventory"]
    assert inventory.get("torch") == 3


def test_a_joint_objective_can_span_both_worlds_conceptually() -> None:
    """数量条件与组合条件在体素世界里同样成立。

    这里不跑完整场景，只验证判定器接上真实世界事实后行为正确 ——
    这正是把条件判定抽成共享模块的收益。
    """
    scenario = load_scenario("village")
    env = MinecraftEnv(scenario, "ayan", "阿岩")
    env.client.actors["ayan"].inventory["torch"] = 2
    env.client.actors["player_a"].inventory["torch"] = 1
    ctx = env.condition_context()

    from npc_agent.env.conditions import condition_met

    assert condition_met({"player_has_count": {"player_a": {"torch": 1}}}, ctx) is True
    assert condition_met({"player_has_count": {"player_a": {"torch": 2}}}, ctx) is False
    assert condition_met(
        {"all_of": [{"flag": "cave_lit"}, {"player_has_count": {"ayan": {"torch": 2}}}]},
        ctx,
    ) is False  # 标记还没设
    env.client.set_flag("cave_lit")
    assert condition_met(
        {"all_of": [{"flag": "cave_lit"}, {"player_has_count": {"ayan": {"torch": 2}}}]},
        env.condition_context(),
    ) is True


# --------------------------------------------------------------------------- #
# MineflayerClient：桥
# --------------------------------------------------------------------------- #
STUB_BRIDGE = textwrap.dedent(
    '''
    """最小可用的桥替身：把请求原样回声，并记录收到过什么。

    真实桥（scripts/mineflayer_bridge.js）会把这些操作翻译成 mineflayer
    的 API 调用；这里只验证**协议层**是否正确。
    """
    import json, sys

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        op = req.get("op")
        if op == "state":
            data = {"actors": {"ayan": {"id": "ayan", "kind": "npc", "inventory": {"oak_log": 2}}}}
        elif op == "boom":
            # 模拟桥自己崩了：直接退出，不回响应
            sys.exit(3)
        else:
            data = {"echo": op, "params": {k: v for k, v in req.items() if k not in ("op", "id")}}
        sys.stdout.write(json.dumps({"ok": True, "reason": "", "data": data, "id": req.get("id")}) + "\\n")
        sys.stdout.flush()
    '''
)


@pytest.fixture
def stub_bridge(tmp_path) -> str:
    path = tmp_path / "stub_bridge.py"
    path.write_text(STUB_BRIDGE, encoding="utf-8")
    return str(path)


def test_mineflayer_client_round_trips_an_op(stub_bridge: str) -> None:
    with MineflayerClient([sys.executable, stub_bridge]) as client:
        result = client.move("ayan", "forest")
        assert result.ok
        assert result.data["echo"] == "move"
        assert result.data["params"]["target"] == "forest"


def test_mineflayer_client_reads_state(stub_bridge: str) -> None:
    with MineflayerClient([sys.executable, stub_bridge]) as client:
        state = client.state()
        assert state["actors"]["ayan"]["inventory"] == {"oak_log": 2}


def test_mineflayer_client_ids_do_not_get_confused(stub_bridge: str) -> None:
    """响应必须和请求的 id 对上。

    串包意味着协议已经错位，继续跑下去只会把错误归因到 NPC 身上 ——
    那是最难查的一类问题，所以宁可直接炸。
    """
    with MineflayerClient([sys.executable, stub_bridge]) as client:
        for _ in range(5):
            assert client.call("chat", actor="ayan", text="行。").ok


def test_bridge_death_raises_instead_of_looking_like_a_game_failure(stub_bridge: str) -> None:
    """传输层故障与游戏内失败必须分开。

    把"桥死了"伪装成"NPC 没做到"，会让一次工程事故看起来像一次模型失败。
    """
    client = MineflayerClient([sys.executable, stub_bridge])
    try:
        with pytest.raises(WorldClientError):
            client.call("boom")
    finally:
        client.close()


def test_missing_node_is_reported_clearly() -> None:
    with pytest.raises(WorldClientError) as exc:
        MineflayerClient(["definitely-not-a-real-binary-xyz"])
    assert "启动桥进程失败" in str(exc.value)


def test_env_propagates_transport_failure_but_swallows_game_failure(
    village: dict, stub_bridge: str
) -> None:
    """MinecraftEnv 必须把两类失败区别对待。"""
    client = MineflayerClient([sys.executable, stub_bridge])
    try:
        env = MinecraftEnv(village, "ayan", "阿岩", client=client)
        # 桥说 ok，所以这里成功
        assert env.dispatch("ayan", call("move_to", location="forest")).ok
        # 桥崩了 → 必须抛出去
        env._call("boom")
    except WorldClientError:
        pass
    else:
        pytest.fail("传输层故障被吞掉了")
    finally:
        client.close()


def test_env_does_not_own_an_injected_client(village: dict, stub_bridge: str) -> None:
    """外部传进来的 client 由调用方负责关 —— 否则会出现"谁关了它"的幽灵 bug。"""
    client = MineflayerClient([sys.executable, stub_bridge])
    env = MinecraftEnv(village, "ayan", "阿岩", client=client)
    env.close()
    assert client.proc.poll() is None  # 还活着
    client.close()


# --------------------------------------------------------------------------- #
# 真实的 Node 桥（dry-run）
# --------------------------------------------------------------------------- #
def _node_exe() -> str:
    import shutil

    node = shutil.which("node")
    if not node:
        pytest.skip("没有 node，跳过真实桥的协议测试")
    return node


@pytest.fixture
def real_bridge() -> str:
    from pathlib import Path

    bridge = Path(__file__).resolve().parent.parent / "scripts" / "mineflayer_bridge.js"
    if not bridge.exists():
        pytest.skip("找不到 mineflayer_bridge.js")
    return str(bridge)


def test_real_bridge_speaks_the_protocol(real_bridge: str) -> None:
    """真的起一个 Node 进程，走真的 stdio，跑真的 JSON。

    前面用 Python 替身验证的是"适配器不依赖 Python 对象"；
    这一条验证的是"协议在真正的进程边界上成立"。
    两者都需要 —— 前者快，后者才是交付时真正会跑的东西。
    """
    with MineflayerClient([_node_exe(), real_bridge, "--dry-run"]) as client:
        assert client.configure(
            {
                "actors": [{"id": "ayan", "name": "阿岩", "kind": "npc", "start": "forest"}],
                "pois": {"forest": {"name": "北边林子", "pos": [6, 0, 2]}},
                "resources": {"forest": {"oak_log": 3}},
            }
        ).ok
        state = client.state()
        assert state["dry_run"] is True
        assert state["actors"]["ayan"]["name"] == "阿岩"
        assert state["pois"]["forest"]["name"] == "北边林子"


def test_real_bridge_handles_unknown_op_and_bad_json(real_bridge: str) -> None:
    """协议层的两种坏输入都不能带走桥。"""
    with MineflayerClient([_node_exe(), real_bridge, "--dry-run"]) as client:
        unknown = client.call("teleport")
        assert unknown.ok is False
        assert "未知操作" in unknown.reason
        # 桥还活着，能继续服务
        assert client.state()["tick"] == 0


def test_real_bridge_keeps_request_response_order(real_bridge: str) -> None:
    """响应顺序必须等于请求顺序。

    上层是同步一问一答的，如果桥把两个 async 操作并发跑，
    响应的 id 就会错位 —— 而错位意味着把 A 的结果当成 B 的结果，
    这种错误不会报错，只会让 NPC 做出莫名其妙的决定。
    """
    with MineflayerClient([_node_exe(), real_bridge, "--dry-run"]) as client:
        client.configure(
            {"actors": [{"id": "ayan", "name": "阿岩", "kind": "npc", "start": "forest"}]}
        )
        for i in range(10):
            result = client.call("chat", actor="ayan", text=f"第{i}句")
            assert result.ok, result.reason


def test_real_bridge_matches_the_offline_world_guardrail_wording(real_bridge: str) -> None:
    """两个世界必须给**同一句话**。

    否则模型在离线后端上学到的"天黑了要火把"到了真实后端就不成立了，
    而这正是最容易被忽略的一类不一致。
    """
    scenario = {
        "actors": [{"id": "ayan", "name": "阿岩", "kind": "npc", "start": "forest"}],
        "pois": {"forest": {"name": "北边林子", "pos": [6, 0, 2]}},
        "resources": {"forest": {"oak_log": 3}},
    }
    offline = LocalWorldClient(dict(scenario))
    offline.move("ayan", "forest")
    offline.advance_tick(DAY_TICKS - DAY_TICKS // 3)

    with MineflayerClient([_node_exe(), real_bridge, "--dry-run"]) as client:
        client.configure(dict(scenario))
        client.move("ayan", "forest")
        client.advance_tick(DAY_TICKS - DAY_TICKS // 3)
        bridge_result = client.mine("ayan", "oak_log")

    offline_result = offline.mine("ayan", "oak_log")
    assert offline_result.ok is False
    assert bridge_result.ok is False
    assert offline_result.reason == bridge_result.reason


# --------------------------------------------------------------------------- #
# 场景配置
# --------------------------------------------------------------------------- #
def test_village_scenario_is_well_formed(village: dict) -> None:
    assert village["env"] == "minecraft"
    assert village["npc"] == "ayan"
    assert set(village["pois"]) >= {"forest", "workshop", "cave_mouth"}
    objective = village["objectives"][0]
    tools = {step["tool"] for step in objective["steps"]}
    assert {"mine", "craft", "place"} <= tools


def test_village_persona_exists_and_fits_the_world() -> None:
    persona = load_persona("ayan")
    assert persona["id"] == "ayan"
    assert persona["name"] == "阿岩"
    assert persona["style"]["sentence_max"] >= 1
    assert persona["utterance_templates"]["fallback"]


# --------------------------------------------------------------------------- #
# 两个真相：`available_topics` 与 `tell_fact` 必须**全称一致**
# --------------------------------------------------------------------------- #
def _require_flags() -> list[str]:
    """所有 `requires` 条件（用来穷举世界状态）。"""
    from npc_agent.env.star_isle import KNOWLEDGE

    return sorted({e["requires"] for e in KNOWLEDGE.values() if e.get("requires")})


def _all_topics() -> list[str]:
    from npc_agent.env.star_isle import KNOWLEDGE

    return sorted(KNOWLEDGE)


def _make_env(world: str, actor: str, scenario: str):  # noqa: ANN202
    if world == "star":
        return StarIsleEnv(load_scenario(scenario), actor, "阿柚")
    return MinecraftEnv(load_scenario(scenario), actor, "阿岩")


#: 必须用**真的锁了话题**的场景。`icebreaker` 里 `knowledge_unlocked` 恰好
#: 覆盖了所有"无 requires"的话题 ⇒ 那条检查在它上面**恒为空转**，
#: 拿它跑这条护栏会得到"永远绿"的假象（实测：把检查删掉照样全绿）。
#: `village` 锁住 3 个、`tutorial` 锁住 1 个。
@pytest.mark.parametrize(
    ("world", "scenario"), [("star", "village"), ("voxel", "village")]
)
def test_every_topic_available_topics_offers_is_one_tell_fact_accepts(
    world: str, scenario: str
) -> None:
    """`available_topics` 说能聊 ⇒ `tell_fact` 必须说能讲；反之亦然。

    **为什么这条值得单独钉住**：`_proactive_share()` 会先把候选话题记进
    `_shared_topics`（**永久**黑名单）**再**执行 `tell_fact`，失败也不撤销。
    只要上面那个全称性质成立，它就永远选不到会失败的话题 ——
    那条顺序 bug 因此是**无害的**。

    但这个性质来自**两份各自实现的检查**（`StarIsleEnv.available_topics`
    和 `StarIsleEnv._h_tell_fact`）。任何一处单独改动都会让它们分家，
    而实测分家之后：**25 个话题被永久拉黑、分数仍是 235/235** ——
    现有测试一条都抓不到。所以这条护栏是那个「无害」结论的**唯一**依据。

    穷举：把 `requires` 条件的每一种组合都摆一遍，逐话题比对两处判定。
    """
    actor = "ayou" if world == "star" else "ayan"
    flags = _require_flags()
    topics = _all_topics()

    # ⚠️ **前提必须先量**：这个场景里真的存在"被锁住的话题"吗？
    # 没有的话，那条检查恒为空转，整条护栏就只是一句永远成立的话。
    probe = _make_env(world, actor, scenario)
    locked = [t for t in topics if t not in probe.available_topics(actor)]
    assert locked, (
        f"场景 {scenario!r} 里没有任何被锁住的话题 —— "
        f"这条护栏在该场景上恒为空转，换一个真的会锁话题的场景"
    )

    divergent: list[str] = []
    checked = 0
    for r in range(len(flags) + 1):
        for combo in itertools.combinations(flags, r):
            env = _make_env(world, actor, scenario)
            for f in combo:
                env.dispatch(actor, ActionCall("set_flag", {"key": f, "value": "1"}))

            offered = set(env.available_topics(actor))
            for topic in topics:
                checked += 1
                res = env.dispatch(actor, ActionCall("tell_fact", {"topic": topic}))
                if (topic in offered) != bool(res.ok):
                    divergent.append(
                        f"[flags={','.join(combo) or 'none'}] topic={topic!r} "
                        f"available_topics={topic in offered} "
                        f"tell_fact.ok={res.ok} detail={res.detail!r}"
                    )

    # 前提：真的检查了足够多的格子（否则循环写错也会「全绿」）
    assert checked >= 8, f"只检查了 {checked} 格，穷举没跑起来"
    assert not divergent, (
        "available_topics 与 tell_fact 对同一个话题给出了相反判定 —— "
        "`_proactive_share()` 会把失败的话题永久拉黑，且分数看不出来：\n  "
        + "\n  ".join(divergent[:8])
    )
