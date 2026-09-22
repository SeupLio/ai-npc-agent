"""`Planner` 的纯函数部分。

现在只有一件事：**把 `success_when` 渲染成人话**（`render_condition`）。

## 为什么它值得单独一个文件

`plan_with_llm` 原来只把目标的 **goal 文本**喂给模型，**不给 `success_when`**。
于是模型规划出"听起来完成了目标"的动作，但那个动作**不满足机器判定的完成条件**。

实测（duet，模型规划，4 次里 3 次失败）：小舟的 `play_song` 完成条件是
`{flag: song_started}`，模型规划的是
`speak → emote(play_guitar) → start_activity(song_request)` ——
全是"像在起歌"的动作，**唯独没有人 `set_flag(song_started)`**，
于是目标永远 pending，依赖它的联合目标 `terrace_night` 也跟着挂住。

这和"规划 prompt 里没有配方表，所以模型想不到先 `take_item`"是**同一类病**：
启发式规划器读得到那份条件，模型读不到。**判据本身必须进 prompt。**

## 这里最要紧的一条护栏

`test_every_condition_kind_renders` —— 条件词汇表（`CONDITION_KINDS`）里
每加一种，渲染器都必须跟得上。跟不上时渲染器会输出"无法识别的完成条件"，
而**模型会因此少知道一条判据** —— 它不会报错，只会规划出一个不达标却很像样的计划。
"""

from __future__ import annotations

import pytest

from npc_agent.env.conditions import CONDITION_KINDS
from npc_agent.env.star_isle import RECIPES
from npc_agent.modules.planner import ORDER_PATTERNS
from npc_agent.llm.null import NullLLM
from npc_agent.modules.persona import Persona
from npc_agent.modules.planner import Planner, render_condition

#: 每种条件给一个**合成样例**，用来验证渲染器认得它。
SAMPLES: dict[str, dict] = {
    "flag": {"flag": "song_started"},
    "all_flags": {"all_flags": ["a", "b"]},
    "any_flags": {"any_flags": ["a", "b"]},
    "all_players_spoke": {"all_players_spoke": 2},
    "player_has": {"player_has": {"player_a": ["latte"]}},
    "player_has_count": {"player_has_count": {"player_a": {"oak_log": 3}}},
    "all_of": {"all_of": [{"flag": "a"}, {"flag": "b"}]},
    "any_of": {"any_of": [{"flag": "a"}, {"flag": "b"}]},
}

_UNRECOGNISED = "无法识别"


@pytest.mark.parametrize("kind", sorted(CONDITION_KINDS))
def test_every_condition_kind_renders(kind: str) -> None:
    """条件词汇表里的每一种，渲染器都必须认得。

    **这是这个文件里最要紧的一条。** 加了新条件类型却忘了让渲染器跟上，
    表现不是报错，而是**模型少知道一条判据** —— 它会规划出一个
    "听起来对、判定不通过"的计划，然后卡在那里。
    """
    assert kind in SAMPLES, (
        f"`CONDITION_KINDS` 里新加了 `{kind}`，但这里没有样例。\n"
        "  请给一个合成样例 —— 渲染器不认得它的话，规划 prompt 里就会少一条判据，\n"
        "  而模型会安静地规划出一个不达标的计划（不会报错）。"
    )
    rendered = render_condition(SAMPLES[kind])
    assert _UNRECOGNISED not in rendered, (
        f"`{kind}` 渲染不出来：{rendered}\n  见 `planner._CONDITION_PHRASES`。"
    )
    assert rendered.strip(), f"`{kind}` 渲染成了空字符串"


def test_unknown_condition_says_so_instead_of_guessing() -> None:
    """认不出来就**如实说**，不许猜。

    猜错会让模型去追一个不存在的目标，比"我不知道"难查得多 ——
    而且它是静默的。
    """
    out = render_condition({"mystery_kind": 1})
    assert _UNRECOGNISED in out
    assert "mystery_kind" in out, "至少要把认不出的键名报出来，否则没法排查"


@pytest.mark.parametrize("empty", [None, {}, []])
def test_empty_condition_is_explicit(empty) -> None:
    """空条件要说"没有完成条件"，而不是渲染成空串。

    空串塞进 prompt 会变成一行什么都没有的标题，读者（模型）只能自己猜。
    """
    assert render_condition(empty) == "（没有完成条件）"


def test_no_nested_backticks() -> None:
    """不许出现 ``` ``latte`` ``` —— 那是渲染器自己拼坏的。

    `player_has` 的模板里如果给 `{items}` 再加一层反引号，而每一项自己
    已经带了反引号，就会拼出双反引号。在 Markdown 里那会显示成
    "一个反引号包着的 latte"，读起来像打字错误 —— 而 prompt 是给模型读的，
    这种噪声会稀释真正要传达的判据。
    """
    out = render_condition({"player_has": {"player_a": ["latte"]}})
    assert "``" not in out, f"渲染出双反引号：{out}"


def test_nested_all_of_renders_both_branches() -> None:
    """嵌套的 `all_of` 两个分支都要出现 —— 少一个就等于少一条判据。"""
    out = render_condition(
        {"all_of": [{"player_has": {"player_a": ["latte"]}}, {"flag": "song_started"}]}
    )
    assert "latte" in out and "song_started" in out
    assert "且" in out


def test_any_of_is_distinguishable_from_all_of() -> None:
    """`any_of` 不能渲染得像 `all_of` —— 一个是"任一"，一个是"全部"。

    弄混会让模型去做**多余的工作**（以为全都要做），或者**少做**
    （以为做一个就行）。
    """
    both = render_condition({"all_of": [{"flag": "a"}, {"flag": "b"}]})
    either = render_condition({"any_of": [{"flag": "a"}, {"flag": "b"}]})
    assert both != either
    assert "且" in both and "或" in either


def test_the_real_scenarios_render_without_falling_back() -> None:
    """所有场景里真实的 `success_when` 都要能渲染。

    合成样例过了、真实配置渲染不出来，是很容易漏的一种情况
    （真实配置里会出现 `player_has_count` 这种只在 Minecraft 用的写法）。
    """
    from npc_agent.config import list_scenarios, load_scenario

    for scenario_id in list_scenarios():
        scenario = load_scenario(scenario_id)
        for obj in scenario.get("objectives") or []:
            condition = obj.get("success_when")
            if not condition:
                continue
            rendered = render_condition(condition)
            assert _UNRECOGNISED not in rendered, (
                f"{scenario_id} 的目标 {obj.get('id')} 渲染不出来：{rendered}"
            )


# --------------------------------------------------------------------------- #
# `empty_plans`：一个**从来不抛异常**、因而从来没被计数过**的失败类别
# --------------------------------------------------------------------------- #
class _ScriptedLLM(NullLLM):
    """`available` 为真、`complete` 返回写死文本。

    用来制造"调用成功、JSON 也解析出来了，但计划不可用"这种情况 ——
    它**不抛异常**，所以走不到 `failures` 那条分支里。
    """

    name = "scripted"

    def __init__(self, payload: str) -> None:
        self.payload = payload

    @property
    def available(self) -> bool:
        return True

    def complete(self, messages, *, temperature=0.7, max_tokens=512):  # type: ignore[override]
        return self.payload


class _StubTracker:
    """`plan_with_llm` 只朝 tracker 要一段现场描述，那就只给它这一段。"""

    def scene_block(self) -> str:
        return "（测试现场）"


def _planner_with(payload: str) -> Planner:
    persona = Persona.from_dict({"id": "x", "name": "小舟"})
    return Planner(persona, _ScriptedLLM(payload))


_USABLE = (
    '{"goal": "招呼客人", "rationale": "先开口", "steps": '
    '[{"goal": "打招呼", "tool": "speak", "args": {"text": "欢迎光临"}}]}'
)


def test_an_empty_plan_is_recorded_not_silently_dropped() -> None:
    """⚠️ 补的是一个**隐形**的失败类别。

    `steps` 为空（或每步都缺 `tool`）时 `plan_with_llm` 从前直接 `return None`，
    **什么都不记** —— 于是报告里那句"解析失败 0 条"只覆盖了抛异常的那一类，
    而"模型给了不可用的计划"这一类连一个计数器都没有。
    **不抛异常不等于成功。**
    """
    planner = _planner_with('{"goal": "招呼客人", "steps": []}')
    assert planner.plan_with_llm(_StubTracker(), None, [], "工具表") is None
    assert planner.empty_plans == 1
    assert planner.failures == 0, "这不是调用失败，是计划不可用 —— 两类必须分开"
    assert planner.last_error, "至少要留下一句人话说明为什么没用它"


def test_a_plan_whose_steps_have_no_tools_is_also_empty() -> None:
    """每一步都缺 `tool` 和 `steps` 为空是同一件事：**这个计划执行不了**。"""
    planner = _planner_with('{"goal": "招呼客人", "steps": [{"goal": "打招呼"}]}')
    assert planner.plan_with_llm(_StubTracker(), None, [], "工具表") is None
    assert planner.empty_plans == 1


def test_a_blank_response_lands_in_the_same_bucket_as_an_unusable_plan() -> None:
    """空输出（抠不出 JSON）和"JSON 能解析但 `steps` 不可用"**落在同一个计数器**。

    这是**故意的**，不是没区分：从规划器的角度看，两件事是同一件事 ——
    "模型被问过，但没给出可用计划"。想分开得有原始文本，
    而 `complete_json` 只回一个 dict，抠不出"到底是空串还是散文"。

    ⚠️ 因此 `empty_plans > 0` **不能**当作"模型不会规划"的证据：
    思维链吃穿预算时返回的就是空内容（本项目实测 3/231 条 `max_tokens` 截断），
    那**落在这里**，而它是预算问题。和 `failures`（传输层）分开报，
    已经足够回答"请求没回来还是模型没给东西"。
    """
    planner = _planner_with("")
    assert planner.plan_with_llm(_StubTracker(), None, [], "工具表") is None
    assert planner.empty_plans == 1
    assert planner.failures == 0, "抠不出 JSON 不是传输层故障"


def test_a_usable_plan_counts_nothing() -> None:
    """反向：计划可用时两个计数器都不许动。

    只会涨的计数器等于没有计数器 —— 它必须能区分"没发生"和"发生了"。
    """
    planner = _planner_with(_USABLE)
    plan = planner.plan_with_llm(_StubTracker(), None, [], "工具表")
    assert plan is not None and plan.steps
    assert planner.empty_plans == 0 and planner.failures == 0


def test_the_model_being_absent_is_not_counted_as_a_failure() -> None:
    """没配模型是"没开这一路"，不是失败。

    这条一旦写反，`--no-planner` 的对照组会被记成一片红 ——
    本项目已经栽过一次同类错误（对照组被记成全是回落）。
    """
    planner = Planner(Persona.from_dict({"id": "x", "name": "小舟"}), NullLLM())
    assert planner.plan_with_llm(_StubTracker(), None, [], "工具表") is None
    assert planner.empty_plans == 0 and planner.failures == 0


# --------------------------------------------------------------------------- #
# 菜单：**两张表**必须点名同一批东西
# --------------------------------------------------------------------------- #
def menu_mismatch(
    recognised: set[str], recipes: set[str]
) -> tuple[list[str], list[str]]:
    """返回（点得到但做不出的, 做得出来但点不到的）。"""
    return sorted(recognised - recipes), sorted(recipes - recognised)


def test_the_menu_has_exactly_one_source_of_truth() -> None:
    """认单的表和造步骤的表必须一致。

    `plan_for_utterance` 认单靠 `ORDER_PATTERNS`（写死在代码里），
    `_plan_serve` 造步骤靠 `world_facts()["recipes"]`（世界配置）。
    **两张表**，各自演进 ⇒ 漂移是时间问题，而漂移时两边都不报错：

      - 配方表多一项、识别表没有 ⇒ 玩家点得到的东西，NPC 说「做不了」
      - 识别表多一项、配方表没有 ⇒ NPC 接下一个做不出来的单

    实测（2026-09-22）：`手冲` 正是第一种。`configs/personas/ayou.yaml`
    的 `self_facts` 说「手冲还算拿得出手」、`unavailable_order` 还拿它当替代品
    推荐，`KNOWLEDGE` 里有一条「手冲的门道」，**唯独 `RECIPES` 里没有它**。
    于是「帮我做一杯手冲」得到「我们这儿做不了，换一杯？」——
    而 NPC 下一句可能就是「手冲要不要试试？」。

    这是附十五那条「不变量靠两份实现碰巧一致维持」的同一个形状：
    今天两张表**恰好**一样，所以没人发现它们是两张表。
    """
    recognised = {item_id for _, item_id in ORDER_PATTERNS}
    recipes = set(RECIPES)

    assert recognised, "识别表是空的 —— 这条护栏在空转"
    assert recipes, "配方表是空的 —— 这条护栏在空转"

    unreachable, unorderable = menu_mismatch(recognised, recipes)
    assert not unreachable and not unorderable, (
        f"菜单有两张表，它们对不上："
        f"点得到但做不出 {unreachable}；做得出来但点不到 {unorderable}。"
        "补的时候**两张一起改**，并确认别名不冲突（`手冲` 必须排在 `拿铁` 前面）。"
    )


def test_the_menu_mismatch_judgement_can_actually_fail() -> None:
    """反向测试：判据必须真的分得出两种不一致，也放得过一致。"""
    assert menu_mismatch({"latte"}, {"latte"}) == ([], [])
    assert menu_mismatch({"latte", "ghost"}, {"latte"}) == (["ghost"], [])
    assert menu_mismatch({"latte"}, {"latte", "pour_over"}) == ([], ["pour_over"])
    assert menu_mismatch(set(), {"latte"}) == ([], ["latte"])
