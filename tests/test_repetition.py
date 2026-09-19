"""复读 + "对话推不动"：判据标定、意图不塌缩、模板变体不撞、记忆只回引一次。

## 这一组测试在防什么

用户报的是"控制台里 NPC 反复重复一个话题，**导致对话无法推进**"。
拆开量出来是**两类**问题，根因都不是模型不行：

**(A) 复读 —— 四处设计塌缩**

| 根因 | 症状 | 修法 |
|---|---|---|
| 所有陈述句都落到 `acknowledge` | 10 轮里同一句说 7 遍 | 意图按"用得最少"轮换 |
| 模板只有一个写法 | 同一意图 ⇒ 逐字相同的输出 | 高频模板给多变体 |
| `recall` 反复回引同一条记忆 | "你上次说过第一次来吧" 说 4 次 | 回引过就不再回引 |
| 问句只认问号 | 「你叫什么名字」被当成陈述句 | `looks_like_question` |

改之前合计复读率 **62%**（44 句里 27 句是复读），改之后 **0%**。

**(B) 推不动 —— 计划压过了玩家**（这是更直接的原因）

`decide()` 的优先级里"继续执行未完成的计划"排在"被点名""有人提问"**前面**，
而 `step()` 里那段"回答优先于计划"的闸门写的是 `decision.urgency >= 0.9` ——
计划那一支只给 0.55，所以那段代码**从来没执行过**（死代码）。
后果：NPC 在跑自己的计划期间**完全听不见**，玩家连问三个问题一个没答。
修法见 `NPCAgent.step`（先回答、再照常跑计划）与 `_player_is_asking_me`。

## 为什么这组测试非写不可

**六维评测一条都抓不到复读**：它每一维都只看**单句**
（这句符不符合人设 / 有没有用到记忆 / 有没有答到点子上），
没有任何一维看"这句和前面那句是不是同一句"。
所以复读可以在六维全 1.000 的情况下发生 —— 实测就是这样
（修完复读与"听不见"两件事之后，离线基线仍然是 231/231、六维全 1.000）。

⚠️ **代价要一起看**：让 NPC 每轮都回应之后，离线长对话的复读率从 18% 升到 22%
（village 40%）—— 多出来的正是那些低信息量的应声，而离线模板池是固定的。
这不是"变差了"，是**同一件事现在算得更准**（以前静默的轮次把分母缩小了）。
详见 `MAX_REPETITION_RATE_LONG` 那里的表。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npc_agent.config import CONFIG_DIR, RuntimeConfig, list_scenarios, load_scenario
from npc_agent.modules.persona import Persona
from npc_agent.modules.repetition import (
    DEFAULT_THRESHOLD,
    find_repeat,
    find_repeats,
    is_repeat,
    normalize,
    similarity,
)
from npc_agent.studio import run_chat
from npc_agent.types import looks_like_question


# --------------------------------------------------------------------------- #
# 1. 判据本身的标定
# --------------------------------------------------------------------------- #
#: (A, B, 是不是同一句话)。**这三列是量的，不是拍的** ——
#: 阈值 0.80 就落在这组数的空档里。
CALIBRATION: list[tuple[str, str, bool]] = [
    # 同一句话，只是标点/空白不同 —— 玩家耳朵里是同一句
    ("小鹿说的我记下了。", "小鹿说的我记下了", True),
    ("嗯——我听着呢，你接着说。", "嗯我听着呢你接着说", True),
    ("欢迎，随便坐。今天露台的星星不错", "欢迎随便坐今天露台的星星不错", True),
    # 换汤不换药：同一件事多说了一个词
    ("我喜欢偏酸的", "我喜欢偏酸的咖啡", True),
    # 真的换了一句话 —— 这两句不能被判成复读
    ("我喜欢拿铁", "我喜欢柠檬水", False),
    ("你好呀", "谢谢！", False),
]


@pytest.mark.parametrize("a,b,same", CALIBRATION)
def test_calibration_pairs_land_on_the_right_side_of_the_threshold(
    a: str, b: str, same: bool
) -> None:
    got = similarity(a, b) >= DEFAULT_THRESHOLD
    assert got is same, f"{a!r} vs {b!r}：相似度 {similarity(a, b):.2f}，判定 {got}，期望 {same}"


def test_the_threshold_has_margin_on_both_sides() -> None:
    """阈值两边都要有余量 —— 贴着边界的阈值等于没有阈值。

    这条比上面那条更重要：上面只验"落在哪边"，这条验"离边界多远"。
    把阈值改成 0.86 或 0.55 上面仍然全绿，这条会红。
    """
    flagged = [similarity(a, b) for a, b, same in CALIBRATION if same]
    clean = [similarity(a, b) for a, b, same in CALIBRATION if not same]
    assert min(flagged) - DEFAULT_THRESHOLD >= 0.05, f"最像复读的那对只高出 {min(flagged):.2f}"
    assert DEFAULT_THRESHOLD - max(clean) >= 0.05, f"最不像的那对只低了 {max(clean):.2f}"


def test_normalize_strips_punctuation_only() -> None:
    assert normalize("嗯——我听着呢，你接着说。") == "嗯我听着呢你接着说"
    assert normalize("") == ""
    # 不能把实词也抹掉
    assert normalize("阿柚") == "阿柚"


def test_empty_lines_are_not_repeats() -> None:
    """沉默不算复读。

    算进去的话，"这一轮没说话"会污染复读率 —— 而"沉默"和"复读"
    是两个完全不同的毛病，修法也不同。
    """
    assert similarity("", "") == 0.0
    assert similarity("嗯——", "") == 0.0
    assert not is_repeat("", ["嗯——", "行。"])
    assert not is_repeat("嗯——", [])


def test_find_repeat_returns_the_line_it_collided_with() -> None:
    """返回**撞上的原句**，不是布尔值。

    调用方要把它放进重试提示里（"你刚说过「…」，换一个说法"）——
    只说"你重复了"，模型不知道该改什么。
    """
    assert find_repeat("小鹿说的我记下了", ["行。", "小鹿说的我记下了。"]) == "小鹿说的我记下了。"
    assert find_repeat("全新的一句话", ["行。", "小鹿说的我记下了。"]) is None


# --------------------------------------------------------------------------- #
# 2. 反向测试：判据必须能红
# --------------------------------------------------------------------------- #
def test_the_repeat_judge_can_actually_fail() -> None:
    """把判据换成"永远返回没重复"，上面那些断言必须跟着失效。

    不用真去改代码，而是**直接构造**一个坏掉的判据，
    验证我们的断言方式确实能识别出它 —— 否则上面那组测试可能只是
    "永远为真"的摆设。
    """
    always_clean = lambda text, history: None  # noqa: E731
    assert always_clean("小鹿说的我记下了。", ["小鹿说的我记下了。"]) is None
    # 真判据在同样输入上必须给出相反答案
    assert find_repeat("小鹿说的我记下了。", ["小鹿说的我记下了。"]) is not None


def test_repeat_report_counts_each_line_once() -> None:
    """同一句说 7 遍只记 1 次复读，不是 6 次。

    否则复读率会被"同一句说了几遍"放大：7 遍会记成 1+2+…+6 = 21 次，
    这个数就没法拿去和别的版本比了。
    """
    lines = [("阿柚", "我记下了。")] * 7
    report = find_repeats(lines)
    assert report.total == 7
    assert len(report.repeats) == 6, "第 1 句不重复，第 2~7 句各记一次"
    assert report.rate == pytest.approx(6 / 7)
    assert report.distinct == 1


def test_repeat_report_ignores_other_speakers() -> None:
    """两个 NPC 说同一句是"撞车"，不是"复读"。

    `Cast` 已经有 `collisions` 在管撞车。两个混在一个数里，
    就说不清是哪个机制坏了。
    """
    report = find_repeats([("阿柚", "你好。"), ("小舟", "你好。")])
    assert report.repeats == []


# --------------------------------------------------------------------------- #
# 3. 意图不再塌缩
# --------------------------------------------------------------------------- #
def test_the_statement_ladder_alternates_in_the_real_path() -> None:
    """意图阶梯按"用得最少"轮换，而不是永远选第一个。

    这是复读的主根因：原来所有陈述句都映射到 `acknowledge`。

    ⚠️ 必须走**真实路径**（`pick_intent` 选 + `_generate_speech` 消费），
    不能只连着调 `pick_intent`：它**只选不消费**，自增在 `_generate_speech`
    里的 `_bump_intent`。只调 `pick_intent` 会得到四次同样的答案 ——
    那不是 bug，是契约，但它测不出轮换。
    """
    from npc_agent.agent import STATEMENT_LADDER

    agent = _offline_agent("tutorial")
    texts = []
    for _ in range(4):
        intent = agent.pick_intent(STATEMENT_LADDER)
        texts.append(agent._generate_speech(intent, [], None))

    assert len(set(texts)) >= 2, f"连续 4 轮只有 {len(set(texts))} 种说法：{texts}"
    # 前两轮必须不同 —— 阶梯的第一级和第二级都要被用到
    assert texts[0] != texts[1], f"前两轮说了同一句：{texts[0]!r}"
    # 确定性：同样的输入重放一遍，必须逐字相同
    replay = []
    other = _offline_agent("tutorial")
    for _ in range(4):
        replay.append(other._generate_speech(other.pick_intent(STATEMENT_LADDER), [], None))
    assert replay == texts, "同一段输入重放没有得到同样的台词 —— 可复现性破了"


def test_pick_intent_does_not_consume_by_itself() -> None:
    """钉住契约：`pick_intent` 只选不消费。

    这条测试的价值是**在重构时报警**：如果哪天有人把自增挪进 `pick_intent`，
    这里会红 —— 而那时 `_generate_speech` 里的自增会让变体索引一次跳两格，
    症状是"变体只用到了奇数号"，很难从现象倒推回来。
    """
    from npc_agent.agent import STATEMENT_LADDER

    agent = _offline_agent("tutorial")
    first = agent.pick_intent(STATEMENT_LADDER)
    assert agent.pick_intent(STATEMENT_LADDER) == first, "pick_intent 不该自己消费"


def test_pick_intent_refuses_empty_candidates() -> None:
    agent = _offline_agent("tutorial")
    with pytest.raises(ValueError):
        agent.pick_intent([])


# --------------------------------------------------------------------------- #
# 4. 记忆只回引一次
# --------------------------------------------------------------------------- #
def test_the_same_memory_is_only_recalled_once() -> None:
    """回引过一次就够了 —— 玩家已经知道你记得。

    修之前 `recall` 每轮都回引检索到的第一条，于是同一句
    「你上次说过第一次来吧。」在 10 轮里出现了 4 次。
    它比 acknowledge 更像复读机：那是**主动把同一个话题捡起来又说一遍**。
    """
    agent = _offline_agent("icebreaker")
    memory = _record("阿澈说：我特别喜欢偏酸的咖啡，越酸越好。", tick=1, importance=0.9)
    agent.state.tick = 10

    first = agent._next_recallable([memory], now=10)
    assert first is not None, "第一次该能回引"
    agent._recalled.add(first.id)
    assert agent._next_recallable([memory], now=10) is None, "回引过之后不该再回引"


def test_recall_is_not_burned_when_the_line_was_blocked() -> None:
    """被发言占比上限拦下的那次不算"回引过"。

    否则这条记忆就永远没机会被说出来了 —— 标记了却没说，等于丢掉。
    """
    agent = _offline_agent("icebreaker")
    memory = _record("阿澈说：我特别喜欢偏酸的咖啡。", tick=1, importance=0.9)
    agent.state.tick = 10
    record = agent._next_recallable([memory], now=10)
    assert record is not None
    # 模拟"挑中了但没说出来"：不调 _respond，直接看 _recalled 仍为空
    assert agent._recalled == set(), "_recalled 不该在挑选阶段就被写"


# --------------------------------------------------------------------------- #
# 5. 模板变体
# --------------------------------------------------------------------------- #
def test_a_personas_variants_never_collide_with_each_other() -> None:
    """同一个人设的**所有**变体，两两不能是同一句话。

    ⚠️ 这是加模板时最容易踩的坑：小舟话少，`acknowledge` 和 `fallback`
    很容易都被写成「嗯。」—— 那时**轮换也救不了**，归一化之后仍然是 1.00。
    这条测试遍历所有人设的所有变体，自动挡住这种写法。
    """
    collisions: list[str] = []
    for persona in _all_personas():
        for intent, raw in persona.templates.items():
            if not isinstance(raw, (list, tuple)):
                continue
            pool = [str(item) for item in raw]
            for i in range(len(pool)):
                for j in range(i + 1, len(pool)):
                    if similarity(pool[i], pool[j]) >= DEFAULT_THRESHOLD:
                        collisions.append(
                            f"{persona.id}.{intent}: 「{pool[i]}」≈「{pool[j]}」"
                        )
    assert not collisions, "同一意图的变体互相撞（轮换救不了）：\n  " + "\n  ".join(collisions)


def test_variants_across_intents_are_also_distinct() -> None:
    """**不同**意图之间也不能撞。

    复读闸门是按"换一个意图"来救场的；如果两个意图的模板本来就是同一句，
    换了等于没换。这条只查"低频意图之间"——
    `answer_question` / `unknown` 会因问题不同而内容不同，不在此列。
    """
    #: 这些意图的模板是固定文本，互相之间不能撞
    FIXED = ("acknowledge", "probe", "fallback", "unknown", "opening", "wrap_up")
    collisions: list[str] = []
    for persona in _all_personas():
        lines: list[tuple[str, str]] = []
        for intent in FIXED:
            for variant in range(persona.template_count(intent)):
                lines.append((intent, persona.render_template(intent, variant=variant, target="小鹿")))
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                (ia, la), (ib, lb) = lines[i], lines[j]
                if la and lb and similarity(la, lb) >= DEFAULT_THRESHOLD:
                    collisions.append(f"{persona.id}: {ia}「{la}」≈ {ib}「{lb}」")
    assert not collisions, "不同意图的模板撞了（换意图救不了）：\n  " + "\n  ".join(collisions)


def test_template_variants_rotate_deterministically() -> None:
    """变体按索引轮换，且**循环**。确定性是这个项目的底线 ——
    控制台每次请求都从头重放整段对话，靠的就是它。"""
    persona = _all_personas()[0]
    intent = "acknowledge"
    count = persona.template_count(intent)
    assert count >= 2, f"{persona.id}.{intent} 只有一个变体，复读挡不住"
    rendered = [persona.render_template(intent, variant=v, target="小鹿") for v in range(count * 2)]
    assert rendered[:count] == rendered[count:], "轮换不循环"
    assert len(set(rendered[:count])) == count, "变体里有重复"


def test_a_single_string_template_still_works() -> None:
    """向后兼容：模板写成**一个字符串**时行为不变。

    三个人设里低频意图（`greet_new` / `teach_order` …）都还是单条 ——
    它们本来就说一次，给变体是多余的。
    """
    persona = _all_personas()[0]
    assert persona.template_count("greet_new") == 1
    first = persona.render_template("greet_new")
    assert persona.render_template("greet_new", variant=5) == first, "单条模板不受 variant 影响"


# --------------------------------------------------------------------------- #
# 6. 问句判据
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        # 带问号
        ("这里有什么好喝的？", True),
        ("阿柚，你还记得我的口味吗？", True),
        # **不带问号但确实是问句** —— 只看问号会全部漏掉
        ("你叫什么名字", True),
        ("这店开了多久了", True),
        ("你平时都在这儿吗", True),
        # 陈述句 / 请求 / 寒暄
        ("我想喝点酸的", False),
        ("那给我来一杯吧", False),   # 「吧」是请求不是提问
        ("谢谢！", False),
        ("你好呀", False),
        ("那明天见", False),
        # 已知误判，**故意钉在这里**：见 looks_like_question 的注释，
        # 误判的代价（把陈述当提问回应）远小于漏判（玩家问了没人答）
        ("我什么都不知道", True),
    ],
)
def test_question_detection(text: str, expected: bool) -> None:
    assert looks_like_question(text) is expected


def test_question_detection_beats_the_old_marker_only_rule() -> None:
    """反向：只认问号的旧判据会漏掉这两句 —— 这正是"对话推不动"的成因。

    修之前「你叫什么名字」被当成陈述句，NPC 只应一声「我记下了」，
    玩家问了两遍都没得到回答。
    """
    old_rule = lambda t: t.rstrip().endswith(("?", "？"))  # noqa: E731
    for text in ("你叫什么名字", "这店开了多久了"):
        assert not old_rule(text), "这两句本来就没有问号"
        assert looks_like_question(text), "新判据该认出来"


# --------------------------------------------------------------------------- #
# 6b. "问我"的判据：`urgency >= 0.9` 是个坏代理量
# --------------------------------------------------------------------------- #


def _duet_or_tutorial():
    """搭一个单 NPC 场景的 (agent, env)，用来驱动真实的 `step()`。"""
    from npc_agent.studio import _make_llm, build_cast

    cfg = RuntimeConfig()
    cfg.llm_provider = "null"
    cast = build_cast(load_scenario("tutorial"), _make_llm(cfg), cfg)
    return list(cast.agents.values())[0], cast.env


def test_a_question_is_answered_even_while_a_plan_is_running() -> None:
    """⭐ 计划**不能**压过"玩家在问我"。

    修之前 `step()` 里那段"回答优先于计划"是**死代码**：
    它的闸门是 `decision.urgency >= 0.9`，而 `decide()` 里
    "继续执行未完成的计划"那一支的 urgency 只有 0.55 ——
    只要计划没跑完，那段代码一行都执行不到。

    实测（修前，tutorial）：玩家连说
    「我想喝点酸的」「你叫什么名字」「这店开了多久了」，
    NPC 一声不吭去后厨拿牛奶、做拿铁 —— 三个问题一个没答。
    """
    agent, env = _duet_or_tutorial()
    # 第一句让 NPC 开起计划（它会去后厨备饮）
    agent.step(env.record_player_utterance("player_a", "你好呀"))
    assert agent.active_plan is not None, "前提不成立：这句本该开出一个计划"

    turn = agent.step(env.record_player_utterance("player_a", "你叫什么名字"))
    assert turn.say, "计划在跑，NPC 就对玩家的提问完全不回应了"
    assert "阿柚" in turn.say, f"没有真的回答「你叫什么名字」：{turn.say!r}"


def test_an_order_written_as_a_question_is_not_treated_as_a_question() -> None:
    """下单**经常写成问句**：「能给我来杯拿铁吗？」

    它不该被当成"提问"去回答 —— 正确反应是应一声「好，稍等」再去真做那杯。
    判据复用 `planner.is_request`（`REQUEST_MARKERS`），不另写一套。
    """
    agent, env = _duet_or_tutorial()
    turn = agent.step(env.record_player_utterance("player_a", "阿柚，能给我来杯拿铁吗？"))
    assert turn.say, "点单没人应"
    assert "稍等" in turn.say, f"点单被当成提问去回答了：{turn.say!r}"


def test_the_acknowledge_line_is_visible_in_the_turn() -> None:
    """即时应声必须写进 `turn.say`。

    不写的话这一声在**界面上完全不存在**：`studio._turn_payload` 会把成功的
    `speak` 从 actions 里滤掉（理由是"已经由 say 表达了"），而 `say` 是空的 ——
    两边一起把它抹掉，页面显示"（没有说话）"，可 NPC 明明说了。
    """
    agent, env = _duet_or_tutorial()
    agent.step(env.record_player_utterance("player_a", "你好呀"))  # 先开一个计划
    turn = agent.step(env.record_player_utterance("player_a", "我想喝点酸的"))
    assert turn.say, "应了声却没写进 turn.say（界面上会显示成沉默）"
    spoke = [a for a in turn.actions if a.tool == "speak"]
    assert spoke, "这一轮根本没有 speak 动作"


# --------------------------------------------------------------------------- #
# 7. 端到端：整段对话不许出现复读
# --------------------------------------------------------------------------- #
#: 固定的 10 轮对话。刻意**以陈述句为主** ——
#: 那正是修之前最薄弱的输入（非问句一律落到 acknowledge）。
CONVERSATION = [
    "你好呀",
    "这里有什么好喝的？",
    "我想喝点酸的",
    "你平时都在这儿吗",
    "那给我来一杯吧",
    "谢谢！",
    "你叫什么名字",
    "这店开了多久了",
    "我下次还来",
    "那明天见",
]

#: 再往后 10 轮。用来量**离线后端的词汇量上限**（见下面那条测试）。
CONVERSATION_LONG = CONVERSATION + [
    "今天人真多",
    "你们这儿有座位吗",
    "我最喜欢靠窗的位置",
    "外面下雨了",
    "你推荐什么",
    "我朋友一会儿也来",
    "他不太喝咖啡",
    "那就来两杯吧",
    "麻烦你了",
    "改天再聊",
]

#: 10 轮的复读率上限。
#:
#: 修之前合计 **62%**（44 句里 27 句复读），修之后 **0%**。
#: 这里写 **0** 而不是留个余量，是因为 0 就是实测值，
#: 而且 10 轮正是控制台演示的真实长度。以后新增场景若把某条路径的
#: 意图又写塌缩了，这条会立刻红。
MAX_REPETITION_RATE = 0.0

#: 20 轮的复读率上限 —— 这是一个**已知局限**，不是目标。
#:
#: 离线启发式后端的词汇量是有限的：每人设约 10 条"不含新信息"的模板
#: （"我记下了" / "后来呢？" / "我说不好"），加上 3~4 个知识话题。
#: 20 轮必然把池子用光，复读率的下界就是 `(轮数 − 词汇量) / 轮数`。
#: 实测（离线，5 个场景）：
#:
#: | 场景 | 10 轮 | 20 轮 | 20 轮去重后 |
#: |---|---|---|---|
#: | duet | 0.0% | 0.0% | 20 |
#: | hosting | 0.0% | 20.0% | 16 |
#: | icebreaker | 0.0% | 25.0% | 15 |
#: | tutorial | 0.0% | 25.0% | 15 |
#: | village | 0.0% | 40.0% | 12 |
#: | **合计** | **0.0%** | **22.0%** | — |
#:
#: ⚠️ **上限从 35% 提到 45%，是因为修了另一件事，不是因为复读变松了。**
#: 上一版 NPC 在"跑自己的计划"期间是**听不见的**（20 轮里只开口 16~18 次），
#: 现在它每轮都回应（20 轮 20 句）—— 多出来的正是那些**低信息量的应声**。
#: 离线模板池是**固定的**（每人设约 12 条低信息台词），
#: 于是复读率恰好撞上那个下界：`(轮数 − 词汇量) / 轮数`
#: = (20 − 12) / 20 = **40%**，village 实测正是 40.0%。
#: 换句话说这不是"变差了"，是**同一件事现在算得更准了**：
#: 以前那些静默的轮次把复读率的分母缩小了。
#:
#: 卡 45% 是**防"变得更糟"**，不假装这个数是好的。
#: 真正的解法是接模型（`use_llm_speech`）—— 那时 prompt 里带着
#: 【你最近说过】，模型能自己换话题，不受模板池大小限制
#: （实测同一个 20 轮对话：离线 25% → 真实模型 0%）。
MAX_REPETITION_RATE_LONG = 0.45


def _replay(scenario_id: str, turns: int = 10) -> list[tuple[str, str]]:
    cfg = RuntimeConfig()          # 离线启发式，不联网
    cfg.llm_provider = "null"
    lines = CONVERSATION_LONG[:turns]
    payload = {
        "scenario": scenario_id,
        "events": [{"kind": "say", "speaker": "player_a", "text": t} for t in lines],
    }
    out = run_chat(cfg, payload)
    return [
        (turn["name"], turn["say"])
        for event in out["events"]
        for turn in event["turns"]
        if turn["say"]
    ]


@pytest.mark.parametrize("scenario_id", list_scenarios())
def test_a_long_conversation_does_not_repeat_itself(scenario_id: str) -> None:
    """端到端：一整段对话里，NPC 不许复读。

    这是唯一一条能抓住"复读"的测试 —— 六维评测一条都抓不到，
    因为它每一维都只看单句。见本文件开头的说明。
    """
    lines = _replay(scenario_id, turns=10)
    assert lines, f"{scenario_id} 一句话都没说，测不出复读（是别的坏了）"
    report = find_repeats(lines)
    assert report.rate <= MAX_REPETITION_RATE, (
        f"{scenario_id} 复读率 {report.rate:.1%} 超上限 {MAX_REPETITION_RATE:.0%}\n"
        + report.render()
    )


@pytest.mark.parametrize("scenario_id", list_scenarios())
def test_the_npc_answers_every_turn(scenario_id: str) -> None:
    """**玩家每说一句，NPC 都必须有回应。**

    这条比"复读率"更接近用户报的那个毛病。原话是
    "NPC 会反复重复一个话题，**导致对话无法推进**" ——
    "推不动"的根子不是重复，是**NPC 在跑自己的计划时完全听不见**：

    实测（修前，tutorial）：玩家连说
    「我想喝点酸的」「你叫什么名字」「这店开了多久了」，
    NPC 一声不吭去后厨拿牛奶、做拿铁、递咖啡 —— 三个问题一个没答，
    页面上连着三行"（没有说话）"。

    修法见 `NPCAgent.step` 里"先回答、再照常跑计划"那一段。
    这里把"每轮都开口"钉死：以后谁再把静默加回来，这条会红。
    """
    turns = 10
    lines = _replay(scenario_id, turns=turns)
    assert len(lines) == turns, (
        f"{scenario_id} 10 轮里只开口 {len(lines)} 次 —— "
        f"有 {turns - len(lines)} 轮玩家说了话、NPC 没回应（对话会卡住）\n"
        + "\n".join(f"  {name}：{text}" for name, text in lines)
    )


def test_the_report_and_the_test_use_the_same_conversation() -> None:
    """`scripts/measure_repetition.py` 和这里必须量**同一段对话**。

    两边各写一份语料，漂移的表现是"报告说 29%、测试说 22%"——
    两个数都"有出处"，读者无从判断哪个是真的，只能两个都不信。
    这类"同一事实抄两份"在这个项目里翻过好几次车，所以钉住。

    用**按路径加载**的方式读脚本：`scripts/` 不是包，不能直接 import。
    """
    import importlib.util

    path = ROOT / "scripts" / "measure_repetition.py"
    assert path.exists(), f"{path} 不在"
    spec = importlib.util.spec_from_file_location("_measure_repetition_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert list(module.CONVERSATION) == list(CONVERSATION_LONG), (
        "报告脚本和测试用的对话语料不一致 —— 两边的复读率会各说各话。\n"
        f"  脚本 {len(module.CONVERSATION)} 句：{module.CONVERSATION[:3]}…\n"
        f"  测试 {len(CONVERSATION_LONG)} 句：{CONVERSATION_LONG[:3]}…"
    )


@pytest.mark.parametrize("scenario_id", list_scenarios())
def test_the_repetition_rate_stays_bounded_past_the_vocabulary(scenario_id: str) -> None:
    """20 轮：复读率**有上限**，但**不为零** —— 这是离线后端的已知局限。

    写这条测试不是为了说"20 轮没问题"，恰恰相反：它把"这个后端
    撑不住长对话"这件事**钉成一个可复现的数字**，
    免得以后有人看到 10 轮 0% 就以为离线模式能一直聊下去。
    """
    report = find_repeats(_replay(scenario_id, turns=20))
    assert report.rate <= MAX_REPETITION_RATE_LONG, (
        f"{scenario_id} 20 轮复读率 {report.rate:.1%} 超上限 {MAX_REPETITION_RATE_LONG:.0%}"
        f" —— 比记录的局限更糟了\n" + report.render()
    )


def test_the_offline_vocabulary_limit_is_what_we_say_it_is() -> None:
    """把"10 轮 0%、20 轮不为零"这个对比钉住。

    单独一条测试，是因为这两句话是**同时**成立的，
    而任何一条单独的测试都只能看到其中一半：
    只看 10 轮会以为离线模式没有复读问题，
    只看 20 轮会以为修没修一样。
    """
    short = _replay("icebreaker", turns=10)
    long = _replay("icebreaker", turns=20)
    short_rate = find_repeats(short).rate
    long_rate = find_repeats(long).rate

    assert short_rate == 0.0, f"10 轮就有复读（{short_rate:.1%}）—— 主线修复回退了"
    assert long_rate > 0.0, (
        "20 轮居然 0 复读 —— 要么词汇量真的变大了（好事，请更新这里的注释和数字），"
        "要么复读判据坏了"
    )


def test_the_conversation_guard_can_actually_fail() -> None:
    """反向：拿修之前的真实输出跑一遍，这条护栏必须红。

    下面这段是**修之前实测的原始输出**（tutorial，离线）——
    不是构造的。如果护栏在它上面还是绿的，那它就是摆设。
    """
    before = [
        ("阿柚", "第一次来吧？我请你一杯，想喝什么？"),
        ("阿柚", "点单很简单：跟我说想喝什么，我做好了端过来。"),
        ("阿柚", "小鹿说的我记下了。"),
        ("阿柚", "小鹿说的我记下了。"),
        ("阿柚", "小鹿说的我记下了。"),
        ("阿柚", "小鹿说的我记下了。"),
        ("阿柚", "小鹿说的我记下了。"),
    ]
    report = find_repeats(before)
    assert report.rate > MAX_REPETITION_RATE, "护栏在修之前的输出上是绿的 —— 它抓不到复读"
    # 7 句里第 3~7 句是同一句 → 后 4 句各记一次复读（第一句不算，它是首次出现）
    assert report.rate == pytest.approx(4 / 7)
    assert report.distinct == 3


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _all_personas() -> list[Persona]:
    out = []
    for path in sorted((CONFIG_DIR / "personas").glob("*.yaml")):
        import yaml

        out.append(Persona.from_dict(yaml.safe_load(path.read_text(encoding="utf-8"))))
    return out


def _offline_agent(scenario_id: str):
    from npc_agent.cast import build_cast
    from npc_agent.llm import build_llm

    cfg = RuntimeConfig()
    cfg.llm_provider = "null"
    cast = build_cast(load_scenario(scenario_id), build_llm("null"), cfg)
    return cast.lead


def _record(content: str, tick: int = 1, importance: float = 0.9):
    from npc_agent.types import MemoryRecord

    return MemoryRecord(
        id="m0001", kind="episodic", content=content, tick=tick, importance=importance
    )
