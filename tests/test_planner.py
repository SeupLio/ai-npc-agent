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
from npc_agent.modules.planner import render_condition

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
