"""LLM-as-judge 的测试。

这个文件的重点是三条铁律各自的回归测试：
    1. 判不了就说判不了 —— 绝不能变成 0 分混进均值
    2. 未经校准的裁判不算证据 —— kappa 要算对
    3. 位置偏见要能被测出来 —— 不是声明一下就算缓解了
"""

from __future__ import annotations

import json
import re
import time

import pytest

from npc_agent.eval import judge as J
from npc_agent.llm import ScriptedLLM
from npc_agent.llm.null import NullLLM
from npc_agent.modules.persona import Persona


# --------------------------------------------------------------------------- #
# Cohen's kappa
# --------------------------------------------------------------------------- #
def test_kappa_is_one_for_perfect_agreement() -> None:
    assert J.cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_kappa_is_zero_for_a_judge_that_always_says_pass() -> None:
    """一个永远输出 1 的裁判，在 80% 正例的集合上一致率就是 80%。

    一致率会骗人 —— 这就是为什么必须看 kappa。kappa 把这个底扣掉，
    于是"什么都没判出来"会老老实实显示成 0。
    """
    labels = [1, 1, 1, 1, 0]
    preds = [1, 1, 1, 1, 1]
    agreement = sum(1 for a, b in zip(labels, preds) if a == b) / len(labels)
    assert agreement == pytest.approx(0.8)
    assert J.cohen_kappa(labels, preds) == pytest.approx(0.0)
    assert "不能用" in J.kappa_reading(0.0)


def test_kappa_is_negative_for_systematically_wrong_judgements() -> None:
    assert J.cohen_kappa([1, 0], [0, 1]) == pytest.approx(-1.0)
    assert "比瞎猜还差" in J.kappa_reading(-1.0)


def test_kappa_handles_the_degenerate_constant_case() -> None:
    """两边都是同一个常数：一致就是完全一致，不一致就是完全不一致。

    期望一致率会算成 1.0，除零。这里必须显式处理，否则校准一跑就崩。
    """
    assert J.cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0
    assert J.cohen_kappa([1, 1, 1], [0, 0, 0]) == 0.0


def test_kappa_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError):
        J.cohen_kappa([1, 0], [1])
    with pytest.raises(ValueError):
        J.cohen_kappa([], [])


# --------------------------------------------------------------------------- #
# 铁律一：判不了就说判不了
# --------------------------------------------------------------------------- #
def test_unavailable_model_yields_unjudged_not_zero() -> None:
    """**这是整个文件最重要的一条。**

    模型不可用时返回 0 分，会让"没测"和"答得很差"在报告里长得一模一样，
    然后把所有均值往下拽一点 —— 而且没人看得出是哪来的。
    """
    judge = J.LLMJudge(NullLLM())
    verdict = judge.judge("in_character", persona="你是阿柚", reply="嗯——坐吧。")
    assert verdict.judged is False
    assert verdict.score is None
    assert verdict.score != 0
    assert "模型不可用" in verdict.error


def test_empty_reply_is_unjudged_not_a_failure() -> None:
    """沉默可能是正确的（发言占比到顶就该闭嘴），不该记成 0 分。"""
    judge = J.LLMJudge(ScriptedLLM(['{"score": 1, "reason": "好"}']))
    verdict = judge.judge("in_character", reply="   ")
    assert verdict.judged is False
    assert verdict.score is None


def test_unknown_rubric_is_rejected_loudly() -> None:
    judge = J.LLMJudge(ScriptedLLM([]))
    with pytest.raises(KeyError, match="未知的评判标准"):
        judge.judge("no_such_rubric", reply="随便一句")


@pytest.mark.parametrize(
    "raw, error_fragment",
    [
        ("我看这条挺好的，给 1 分。", "没有 score 字段"),          # 没按格式输出
        ('{"reason": "说不出理由"}', "没有 score 字段"),            # 缺字段
        ('{"score": 0.7, "reason": "还行吧"}', "越界"),             # 让它给 0/1，它给 0.7
        ('{"score": "好", "reason": "不错"}', "不是数字"),           # 类型不对
        ('{"score": 1, "reason": ""}', "没有给出理由"),             # 没理由 = 没法审计
        ("", "没有 score 字段"),                                    # 空输出
    ],
)
def test_bad_judge_output_becomes_unjudged_never_a_guessed_score(
    raw: str, error_fragment: str
) -> None:
    """任何一步不对就判"没判"。宁可少一个数据点，也不要一个编出来的分数 ——

    编出来的分数没法被审计，也没法被校准发现，而且它会一直留在均值里。
    """
    judge = J.LLMJudge(ScriptedLLM([raw]))
    verdict = judge.judge("responsive", player="来杯拿铁", reply="好，稍等。")
    assert verdict.judged is False
    assert verdict.score is None
    assert error_fragment in verdict.error


def test_model_exception_becomes_unjudged_not_a_crash() -> None:
    """网络/鉴权/超时都算"没判"，不该让整场跑批崩掉。"""

    def boom(_text: str) -> str:
        raise RuntimeError("connection reset")

    judge = J.LLMJudge(ScriptedLLM(boom))
    verdict = judge.judge("grounded", reply="黑着呢，得先做个火把。")
    assert verdict.judged is False
    assert verdict.score is None
    assert "模型调用异常" in verdict.error


def test_a_valid_verdict_is_parsed() -> None:
    judge = J.LLMJudge(ScriptedLLM(['{"score": 1, "reason": "短句，有口头禅"}']))
    verdict = judge.judge("in_character", reply="嗯——第一次来啊？")
    assert verdict.judged is True
    assert verdict.score == 1.0
    assert verdict.reason == "短句，有口头禅"


def test_string_score_is_accepted() -> None:
    """模型把 1 写成 "1" 是常见输出，没必要为此丢掉一个数据点。"""
    judge = J.LLMJudge(ScriptedLLM(['{"score": "1", "reason": "符合"}']))
    assert judge.judge("in_character", reply="嗯——坐吧。").score == 1.0


def test_judge_prompt_carries_the_context_it_needs() -> None:
    """裁判看不到人设和现场就会开始猜 —— 那样的分数没有意义。"""
    llm = ScriptedLLM(['{"score": 1, "reason": "ok"}'])
    judge = J.LLMJudge(llm)
    judge.judge(
        "grounded",
        persona="你是阿柚，星屿咖啡屋的店主",
        scene="你在吧台，后厨有咖啡豆",
        player="来杯拿铁",
        reply="好，我去后厨拿豆子。",
    )
    prompt = "\n".join(m.get("content", "") for m in llm.calls[0])
    assert "星屿咖啡屋" in prompt
    assert "后厨有咖啡豆" in prompt
    assert "来杯拿铁" in prompt
    assert "事实一致" in prompt


# --------------------------------------------------------------------------- #
# 铁律二：校准
# --------------------------------------------------------------------------- #
def _oracle_judge() -> J.LLMJudge:
    """一个照着人工标签回答的"完美裁判"，用来验证校准数学。

    查找键是 **(标准, 台词, 玩家那句话)** 三元组，而不是只看台词。
    校准集里有两处刻意保留的重复台词：

      「说起来，露台今晚风小，坐那儿挺好。」
          —— 在「角色口吻」上通过（就是阿柚的语气），
             在「是否回应」上不通过（玩家在点单，她在推露台）。

      「嗯——这个我还真说不好。」
          —— 在「是否回应」上，问隐藏菜单时通过（确实不该讲），
             问营业时间时不通过（店主该知道，这是拿话术躲问题）。

    **同一句台词在不同标准、不同上下文里对错相反。** 这正是指望裁判
    看上下文的原因，也是校准集必须带 rubric / player / scene 的原因。
    """

    def fn(text: str) -> str:
        for item in J.load_calibration():
            if item["reply"] not in text:
                continue
            if J.RUBRICS[item["rubric"]].name not in text:
                continue
            if item.get("player") and item["player"] not in text:
                continue
            return json.dumps(
                {"score": item["label"], "reason": "照着标签答"}, ensure_ascii=False
            )
        return '{"score": 0, "reason": "没找到"}'

    return J.LLMJudge(ScriptedLLM(fn))


def test_the_same_reply_can_be_right_in_one_context_and_wrong_in_another() -> None:
    """校准集里必须存在这种样本 —— 否则它证明不了裁判在看上下文。"""
    items = J.load_calibration()
    by_reply: dict[str, list[dict]] = {}
    for item in items:
        by_reply.setdefault(item["reply"], []).append(item)

    same_rubric_conflicts = [
        (reply, group)
        for reply, group in by_reply.items()
        if len(group) > 1
        and len({g["rubric"] for g in group}) == 1
        and len({g["label"] for g in group}) > 1
    ]
    cross_rubric_conflicts = [
        (reply, group)
        for reply, group in by_reply.items()
        if len({g["rubric"] for g in group}) > 1
        and len({g["label"] for g in group}) > 1
    ]
    assert same_rubric_conflicts, "没有「同一标准下同句不同判」的样本"
    assert cross_rubric_conflicts, "没有「同一句话在不同标准下对错相反」的样本"

    # 同标准下同句不同判 → 必须靠 player 那句话区分
    for _reply, group in same_rubric_conflicts:
        assert all(g.get("player") for g in group), (
            "同一标准下同句不同判的样本必须带上玩家那句话，否则裁判无从区分"
        )


def test_calibration_of_a_perfect_judge_gives_kappa_one() -> None:
    report = J.calibrate(_oracle_judge())
    assert report.total == 24
    assert report.judged == 24
    assert report.unjudged == 0
    assert set(report.per_rubric) == set(J.DEFAULT_RUBRICS)
    for stats in report.per_rubric.values():
        assert stats["agreement"] == 1.0
        assert stats["kappa"] == 1.0
    assert not report.disagreements


def test_calibration_of_a_yes_man_judge_exposes_it() -> None:
    """一个永远说"通过"的裁判，一致率等于正例占比，但 kappa 会把它的底揭出来。"""
    judge = J.LLMJudge(ScriptedLLM(lambda _t: '{"score": 1, "reason": "看着还行"}'))
    report = J.calibrate(judge)
    assert report.judged == 24

    items = J.load_calibration()
    for key, stats in report.per_rubric.items():
        positives = sum(1 for i in items if i["rubric"] == key and i["label"] == 1)
        total = sum(1 for i in items if i["rubric"] == key)
        # 永远输出 1 → 一致率就是正例占比。这就是"一致率会骗人"的具体样子。
        assert stats["agreement"] == pytest.approx(positives / total)
        assert stats["kappa"] == pytest.approx(0.0)
        assert "不能用" in stats["reading"]
    # 每一处错判都要留档，而且带上人工标签和裁判理由 —— 那才是能改标准的地方
    mismatches = [d for d in report.disagreements if d["kind"] == "mismatch"]
    assert len(mismatches) == sum(1 for i in items if i["label"] == 0)
    assert all("judge_reason" in d and "human_label" in d for d in mismatches)


def test_unjudged_items_are_counted_separately_and_never_scored() -> None:
    """没判成的条目要单独计数，不能以 0 分的形式混进一致率。"""
    report = J.calibrate(J.LLMJudge(NullLLM()))
    assert report.judged == 0
    assert report.unjudged == report.total
    assert report.per_rubric == {}
    assert all(d["kind"] == "unjudged" for d in report.disagreements)
    assert "有 24 条没判成" in J.render_calibration(report)


def test_calibration_report_is_machine_readable() -> None:
    payload = J.calibrate(_oracle_judge()).to_dict()
    assert set(payload) == {"total", "judged", "unjudged", "per_rubric", "disagreements"}
    assert payload["per_rubric"]["in_character"]["confusion"]["真阳性"] > 0


# --------------------------------------------------------------------------- #
# 铁律三：位置偏见
# --------------------------------------------------------------------------- #
def _prefers(marker: str):
    """一个"只按内容判断"的裁判：谁的文本里含 marker 就选谁。"""

    def fn(text: str) -> str:
        first = text[text.find("【第一条】") : text.find("【第二条】")]
        return json.dumps(
            {"choice": "前者" if marker in first else "后者", "reason": "更贴合人设"},
            ensure_ascii=False,
        )

    return fn


def test_pairwise_picks_the_consistent_winner() -> None:
    judge = J.LLMJudge(ScriptedLLM(_prefers("AAA")))
    result = judge.compare_pairwise(
        persona="你是阿柚", scene="", player="来杯拿铁", reply_a="AAA", reply_b="BBB"
    )
    assert result["judged"] is True
    assert result["consistent"] is True
    assert result["winner"] == "a"


def test_pairwise_detects_position_bias_and_declares_a_tie() -> None:
    """一个总是选"先出现的那个"的裁判，交换顺序后结论会自相矛盾。

    这不是理论担忧 —— 这条测试就是在**实测**它。
    发现不一致就判平局，而不是取两次的平均（那等于把偏见平摊进结果）。
    """
    judge = J.LLMJudge(ScriptedLLM(lambda _t: '{"choice": "前者", "reason": "先看到的更好"}'))
    result = judge.compare_pairwise(
        persona="你是阿柚", scene="", player="来杯拿铁", reply_a="AAA", reply_b="BBB"
    )
    assert result["judged"] is True
    assert result["consistent"] is False
    assert result["winner"] is None
    assert "位置偏见" in result["note"]
    assert result["forward"] == "前者" and result["backward"] == "前者"


def test_pairwise_handles_a_genuine_tie() -> None:
    judge = J.LLMJudge(ScriptedLLM(lambda _t: '{"choice": "平局", "reason": "差不多"}'))
    result = judge.compare_pairwise(persona="", scene="", player="", reply_a="A", reply_b="B")
    assert result["judged"] is True
    assert result["winner"] is None
    assert result["consistent"] is True


def test_pairwise_needs_both_sides() -> None:
    judge = J.LLMJudge(ScriptedLLM(['{"choice": "前者", "reason": "x"}']))
    assert judge.compare_pairwise(
        persona="", scene="", player="", reply_a="A", reply_b="  "
    )["judged"] is False


# --------------------------------------------------------------------------- #
# 从转写里还原对话对
# --------------------------------------------------------------------------- #
def test_dialogue_pairs_reconstructs_player_and_npc_turns() -> None:
    """裁判必须知道 NPC 在回应什么，"是否回应"这个维度才判得动。

    转写是唯一保留完整对话顺序的产物：报告里的 speeches 只有 NPC 那半边。
    """
    transcript = [
        "玩家[阿澈] 阿柚，能给我来杯拿铁吗？",
        "  [ok] 阿柚 speak(text=好，稍等，我这就去弄。)",
        "  [ok] 阿柚 move_to(location=kitchen)",
        "阿柚: 好，稍等，我这就去弄。",
        "  [ok] 阿柚 take_item(item=beans)",
        "阿柚: 拿铁好了，趁热。",
        "玩家[小满] 我也要一杯。",
        "阿柚: 好，稍等，我这就去弄。",
    ]
    pairs = J.dialogue_pairs(transcript)
    assert len(pairs) == 3
    assert pairs[0]["player"] == "阿柚，能给我来杯拿铁吗？"
    assert pairs[0]["reply"] == "好，稍等，我这就去弄。"
    # 连续两句都算对这一轮的回应
    assert pairs[1]["player"] == "阿柚，能给我来杯拿铁吗？"
    assert pairs[1]["reply"] == "拿铁好了，趁热。"
    # 换玩家之后要跟着换
    assert pairs[2]["player"] == "我也要一杯。"


def test_dialogue_pairs_ignores_tool_lines() -> None:
    """工具调用行不是台词，不能进裁判的输入 —— 那会让它去评判参数。"""
    pairs = J.dialogue_pairs(["  [ok] 阿柚 move_to(location=kitchen)"])
    assert pairs == []


# --------------------------------------------------------------------------- #
# 标准本身
# --------------------------------------------------------------------------- #
def test_every_rubric_states_both_sides_of_the_decision() -> None:
    """只写"这条好不好"是没用的：裁判会按自己的偏好自由发挥。

    pass_when / fail_when 必须都是人能读懂、能照着判的条件，
    否则 kappa 低下来也看不出低在哪一条上。
    """
    for key, rubric in J.RUBRICS.items():
        assert rubric.pass_when and rubric.fail_when
        assert len(rubric.pass_when) > 20 and len(rubric.fail_when) > 20
        assert rubric.question.endswith("？")
        assert "persona" in rubric.needs and "reply" in rubric.needs
        assert rubric.name in rubric.prompt_block()


def test_default_rubrics_exist() -> None:
    assert set(J.DEFAULT_RUBRICS) <= set(J.RUBRICS)


def test_judge_temperature_is_zero() -> None:
    """判分要的是稳定，不是创造力。"""
    assert J.JUDGE_TEMPERATURE == 0.0


# --------------------------------------------------------------------------- #
# 校准集本身
# --------------------------------------------------------------------------- #
def test_calibration_set_is_balanced_and_covers_every_rubric() -> None:
    """校准集不能全是明显对/明显错的样本 —— 那样一致率必然虚高。"""
    items = J.load_calibration()
    assert len(items) >= 20
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids))
    for rubric in J.DEFAULT_RUBRICS:
        labels = [i["label"] for i in items if i["rubric"] == rubric]
        assert len(labels) >= 6, f"{rubric} 的样本太少，kappa 没有意义"
        assert 0 in labels and 1 in labels, f"{rubric} 只有单边标签，kappa 恒为 0"
    for item in items:
        assert item["label"] in (0, 1)
        assert item["reply"].strip()
        assert item["rubric"] in J.RUBRICS


def test_calibration_set_keeps_borderline_cases() -> None:
    """刻意保留边界样本，并在 note 里标出来。

    全是"明显对/明显错"的校准集只能证明裁判会做送分题。
    """
    items = J.load_calibration()
    borderline = [i for i in items if "边界" in (i.get("note") or "")]
    assert len(borderline) >= 2


# --------------------------------------------------------------------------- #
# 校准集必须先过框架自己的确定性检查
# --------------------------------------------------------------------------- #
_SENT_MAX_RE = re.compile(r"最多说\s*(\d+)\s*句")


def test_no_pass_label_contradicts_the_frameworks_own_sentence_check() -> None:
    """标了「通过」的样本，不得违反人设块里写明的句数上限。

    这条测试来自一次真实事故。拿真实模型跑校准，裁判把 `cal_ic_01` 判成 0，
    我标的是 1。看起来是"裁判错了"，但回头用框架自己的 `Persona.check()`
    复核，框架也返回 `['句数超限']` —— **错的是标注，不是裁判**。

    这件事的教训不是"某条标错了"，而是：**校准集是标准答案，标准答案自己
    出错比裁判出错更糟** —— 它会把一个正确的裁判记成错的，让 kappa 无端掉下来，
    于是"裁判不可靠"这个结论本身就是假的。所以校准集必须先过一遍框架里
    已经存在的确定性检查，再拿去当标准答案。

    只查单向：标 1 的不能违反硬约束。标 0 的可以因为语气等主观理由不合格，
    确定性检查通过也正常。
    """
    persona = Persona(id="_cal", name="_cal")
    checked = 0
    for item in J.load_calibration():
        match = _SENT_MAX_RE.search(item.get("persona") or "")
        if not match:
            continue
        checked += 1
        limit = int(match.group(1))
        count = persona.sentence_count(item["reply"])
        if item["label"] == 1:
            assert count <= limit, (
                f"{item['id']} 标了「通过」，但台词有 {count} 句、人设上限 {limit} 句。"
                "标准答案和人设卡冲突 —— 先改正标注，不要改裁判。"
            )
    assert checked >= 2, "校准集里没有人设块声明句数上限，这条测试就白跑了"


def test_calibration_labels_do_not_contradict_the_persona_module() -> None:
    """上一条测试用的是自己写的句数正则；这一条用框架真正在用的检查器。

    两者都跑，是因为"我写的正则"和"框架实际执行的约束"可能不是一回事 ——
    只测前者，等于用我的理解去校准我的理解。
    """
    from npc_agent.config import load_persona
    from npc_agent.modules.persona import Persona as P

    ayou = P.from_dict(load_persona("ayou"))
    checked = 0
    for item in J.load_calibration():
        blob = f"{item.get('persona') or ''}{item.get('scene') or ''}"
        if "阿柚" not in blob:
            continue
        # 只对明确写了句数约束的样本做断言，避免把无关样本卷进来
        if not _SENT_MAX_RE.search(item.get("persona") or ""):
            continue
        checked += 1
        issues = ayou.check(item["reply"])
        if item["label"] == 1:
            assert not issues, f"{item['id']} 标「通过」，但框架的人设检查报 {issues}"
    assert checked >= 1


# --------------------------------------------------------------------------- #
# 把校准发现的两个失效模式钉进 rubric 文本
# --------------------------------------------------------------------------- #
def test_in_character_rubric_says_persona_constraints_are_rules() -> None:
    """校准发现：裁判把「最多说 2 句」当成了风格描述而不是硬约束。

    这条测试锁住措辞。谁哪天觉得这句啰嗦删掉了，测试会拦下来 ——
    因为删掉之后 kappa 会掉，而掉的原因不会自己写进 git log。
    """
    fail_when = J.RUBRICS["in_character"].fail_when
    assert "规则" in fail_when
    assert "最多说" in fail_when


def test_responsive_rubric_names_evasive_ignorance() -> None:
    """校准发现：裁判把「拿话术躲问题」当成了「回应了」。

    具体是 `cal_rs_06`：现场牌子上写着营业时间，NPC 却说「这个我还真说不好」。
    裁判判了通过，理由是"虽然不知道，但属于针对问题的回应" —— 它读懂了字面，
    没去核对现场。同一条台词在 `cal_rs_02`（问隐藏菜单）判通过是对的，
    所以区分依据只能是现场有没有答案。rubric 必须把这条说出来。
    """
    fail_when = J.RUBRICS["responsive"].fail_when
    assert "现场" in fail_when
    assert "说不好" in fail_when or "不知道" in fail_when


def test_the_hard_calibration_case_is_still_in_the_set() -> None:
    """`cal_rs_06` 是这个校准集里最难的一条，不能被悄悄删掉。

    它和 `cal_rs_02` 共用同一句台词却标签相反，是唯一能测出
    "裁判到底有没有在看现场"的样本。删了它，kappa 会变好看，但会变得没有意义。
    """
    items = {i["id"]: i for i in J.load_calibration()}
    assert "cal_rs_06" in items
    assert items["cal_rs_06"]["label"] == 0
    assert "营业时间" in items["cal_rs_06"]["scene"]
    twin = items["cal_rs_02"]
    assert twin["reply"] == items["cal_rs_06"]["reply"]
    assert twin["label"] == 1, "同句反标的那一对被改掉了，这条测试就失去意义"


# --------------------------------------------------------------------------- #
# 并发判分
# --------------------------------------------------------------------------- #
_TRANSCRIPT = [
    "玩家[阿澈] 来杯拿铁",
    "  [ok] 阿柚 move_to(location=kitchen)",
    "阿柚: 好，稍等。",
]


def _fake_results(n: int) -> list[dict]:
    return [
        {
            "case_id": f"case_{i:02d}",
            "category": "task",
            "scenario": "tutorial",
            "transcript": list(_TRANSCRIPT),
        }
        for i in range(n)
    ]


def _scripted_judge(payload: str = '{"score": 1, "reason": "还行"}') -> J.LLMJudge:
    return J.LLMJudge(ScriptedLLM(lambda _p: payload), name="fake-judge")


def _run_judge(results, judge, *, concurrency):
    return J.judge_report_cases(
        results,
        judge=judge,
        persona_of=lambda _sid, _spk="": "你是阿柚",
        scene_of=lambda _sid: "你在吧台",
        concurrency=concurrency,
    )


def test_parallel_judging_keeps_the_report_order() -> None:
    """判分结果的顺序必须和跑批报告一致。

    顺序错了，"哪条用例人设崩了"就得人工去对 —— 等于没判。
    """
    results = _fake_results(6)
    judgements, stats = _run_judge(results, _scripted_judge(), concurrency=4)

    assert [j.index for j in judgements] == list(range(6))
    assert [j.case_id for j in judgements] == [r["case_id"] for r in results]
    assert stats["cases"] == 6


def test_parallel_judging_agrees_with_serial() -> None:
    """并发 1 和并发 4 必须判出一样的结果。"""
    results = _fake_results(4)
    serial, _ = _run_judge(results, _scripted_judge(), concurrency=1)
    parallel, _ = _run_judge(results, _scripted_judge(), concurrency=4)

    assert [j.to_dict() for j in serial] == [j.to_dict() for j in parallel]


def test_a_case_without_dialogue_is_empty_not_an_error() -> None:
    """没有台词的用例不是"判分炸了"。

    把它记成错误，报告里就会出现一批假的失败，
    而真正的问题（这条用例本来就没让 NPC 开口）反而看不见了。
    """
    results = _fake_results(3)
    results[1]["transcript"] = []
    judgements, stats = _run_judge(results, _scripted_judge(), concurrency=2)

    assert judgements[1].ok, "空转写被记成了错误"
    assert judgements[1].pairs == []
    assert stats["cases_without_dialogue"] == 1
    assert stats["cases_failed"] == 0


def test_a_judging_crash_does_not_take_down_the_whole_report() -> None:
    """单条用例判分炸了，不能让整批判分作废。"""

    class Exploding(J.LLMJudge):
        def judge_pairs(self, pairs, **kwargs):  # type: ignore[override]
            raise RuntimeError("boom")

    judgements, stats = _run_judge(_fake_results(3), Exploding(ScriptedLLM([])), concurrency=2)

    assert all(not j.ok for j in judgements)
    assert all("RuntimeError" in j.error for j in judgements)
    assert stats["cases_failed"] == 3
    assert "不是「判了 0 分」" in stats["verdict"]


def test_coverage_verdict_does_not_call_unjudged_a_zero() -> None:
    """未判超 10% 要说"通过率建立在少数判决上"，而不是把它当 0 分。"""
    assert "全部 100 条判决都拿到了分数" in J._coverage_verdict(10, 0, 100, 0)
    assert "没有变成 0 分" in J._coverage_verdict(10, 0, 100, 3)
    assert "占比超过 10%" in J._coverage_verdict(10, 0, 100, 20)
    assert "一条台词都没判到" in J._coverage_verdict(10, 0, 0, 0)
    assert J._coverage_verdict(0, 0, 0, 0) == "没有用例可判"


def test_unjudged_verdicts_are_counted_but_never_scored() -> None:
    """模型输出不合格式 → 未判；它不该出现在通过率的分子或分母里。"""
    results = _fake_results(2)
    judgements, stats = _run_judge(
        results, _scripted_judge('{"reason": "忘了给分数"}'), concurrency=2
    )

    verdicts = [v for j in judgements for v in j.verdicts()]
    assert verdicts, "一条判决都没有"
    assert all(not v["judged"] for v in verdicts)
    assert stats["unjudged"] == len(verdicts)
    assert "没有用例" not in stats["verdict"]


def test_judge_output_budget_is_sized_for_a_reasoning_models_cot() -> None:
    """裁判的输出预算必须容得下思维链 —— 这是实测踩出来的坑。

    拿 `speech_max_tokens`（当时是 1024）去跑裁判，校准集里出现了一条
    **空内容**：`finish_reason=length`，模型是推理模型，光思维链就 4043 字，
    预算被 CoT 吃光，正式回答一个字都没剩下。

    这个失败特别隐蔽：空内容会被正确判成"未判"而不是 0 分，
    所以报告不报错，只显示"未判 1 条"。但如果三成判决都因为预算不够
    没判成，通过率就建立在少数样本上了 —— 而报告看上去依然正常。

    **曾经这里写的是 `JUDGE_MAX_TOKENS > speech_max_tokens * 2`。**
    那条断言已经退役：它是个代理指标（"别把台词预算直接拿去跑裁判"），
    而台词预算后来也被提到 4096 之后，`2×` 就变成了一个
    和真实约束无关的数字。真实的约束只有一条：
    **两个预算都必须容得下推理模型的思维链**，而裁判的 prompt 更长
    （人设 + 现场 + 玩家 + 台词），所以它不能比台词预算小。
    """
    from npc_agent.config import RuntimeConfig

    speech = RuntimeConfig().speech_max_tokens
    assert J.JUDGE_MAX_TOKENS >= 4096, (
        "4096 是同端点能正常处理 4043 字思维链的实测值"
    )
    assert J.JUDGE_MAX_TOKENS >= speech, (
        "裁判的 prompt 比台词长，思维链不会更短，预算不该比台词小"
    )


def test_responsive_rubric_defines_ignorance_in_both_directions() -> None:
    """「不知道」什么时候算合格、什么时候算敷衍，两个方向都要写清楚。

    这条测试来自校准的**第二轮**：第一轮裁判把"拿话术躲问题"当成回应了，
    于是我在 fail_when 里加了"现场有答案却说不知道 = 不合格"。
    结果第二轮**矫枉过正** —— 同一条台词在问隐藏菜单时被判成了不合格，
    而那里的"我不能说"恰恰是正确回答（人设规定未解锁内容不能讲）。

    只写一个方向的边界，裁判就会倒向另一侧。两个方向都写，它才有依据去分。
    """
    rubric = J.RUBRICS["responsive"]
    assert "算通过" in rubric.pass_when, "没有说清楚「诚实的不知道」什么时候合格"
    assert "算不通过" in rubric.fail_when, "没有说清楚「躲问题的不知道」什么时候不合格"
    assert "人设" in rubric.pass_when and "现场" in rubric.fail_when


# --------------------------------------------------------------------------- #
# 调用失败的重试
#
# 判分是几小时的长作业，而模型客户端有 60s 读超时。一次抖动就会永久丢掉
# 一条判决 —— 因为 `judge()` 把异常转成 `Verdict.unjudged` **返回**，
# 而"未判"是合法返回值，不重试、不报错、不会有人发现。
# --------------------------------------------------------------------------- #


class _FlakyLLM:
    """前 `fail_times` 次调用抛异常，之后正常返回。"""

    name = "flaky"

    def __init__(self, fail_times: int, payload: str = '{"score": 1, "reason": "还行"}'):
        self.remaining = fail_times
        self.payload = payload
        self.attempts = 0
        self.exc = TimeoutError("The read operation timed out")

    @property
    def available(self) -> bool:
        return True

    def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.attempts += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise self.exc
        return self.payload


def _no_sleep(_seconds: float) -> None:
    """测试里不要真的等 3 秒、6 秒。"""


def test_a_transient_timeout_is_retried_instead_of_losing_the_verdict() -> None:
    """一次 60s 读超时不该变成一条永久缺失的判决。"""
    llm = _FlakyLLM(fail_times=1)
    judge = J.LLMJudge(llm, sleep=_no_sleep)
    verdict = judge.judge("in_character", reply="好嘞，稍等")

    assert verdict.judged is True, "重试之后应该拿到判决，而不是记成未判"
    assert verdict.score == 1.0
    assert llm.attempts == 2
    assert judge.retries == 1
    assert judge.calls == 1, "只有成功那次算一次调用"


def test_retrying_gives_up_eventually_and_reports_the_real_error() -> None:
    """一直失败要放弃，而且错误信息要留真话 —— 不能变成"没有台词可判"。"""
    llm = _FlakyLLM(fail_times=99)
    judge = J.LLMJudge(llm, max_retries=2, sleep=_no_sleep)
    verdict = judge.judge("in_character", reply="好嘞，稍等")

    assert verdict.judged is False
    assert "TimeoutError" in verdict.error, "要把真正的失败原因带出来"
    assert llm.attempts == 3, "1 次 + 2 次重试"
    assert judge.retries == 2


def test_the_sleep_between_retries_grows_exponentially() -> None:
    """退避要指数增长，否则端点抖一下会被我们连打三拳。"""
    waits: list[float] = []
    judge = J.LLMJudge(_FlakyLLM(fail_times=99), backoff=3.0, sleep=waits.append)
    judge.judge("in_character", reply="好嘞，稍等")
    assert waits == [3.0, 6.0]


def test_a_missing_reply_is_not_retried() -> None:
    """**"没有台词可判"不该重试。**

    它是确定性的：台词是空的，重试一万次也还是空的。
    把"重试"和"没判"混在一起，会让一份几千次调用的作业白烧一大截。
    """
    llm = _FlakyLLM(fail_times=0)
    judge = J.LLMJudge(llm, sleep=_no_sleep)
    verdict = judge.judge("in_character", reply="   ")

    assert verdict.judged is False
    assert llm.attempts == 0, "根本不该发起调用，更不该重试"
    assert "没有台词可判" in verdict.error


def test_an_unavailable_model_is_not_retried() -> None:
    """模型压根没配（NullLLM）也是确定性的，重试没意义。"""
    judge = J.LLMJudge(NullLLM(), sleep=_no_sleep)
    verdict = judge.judge("in_character", reply="好嘞，稍等")

    assert verdict.judged is False
    assert judge.retries == 0
    assert judge.calls == 0


def test_a_bad_output_format_is_retried_because_truncation_is_random() -> None:
    """解析失败**要**重试 —— 这条推翻了它自己的前身。

    原来这里叫 `test_a_bad_output_format_is_not_retried`，理由是
    "它通常是确定性的 —— 思维链把预算吃光就会稳定地返回空内容"。
    **实测把这个理由推翻了**：长跑里出现过一条思维链 14440 字，
    是之前实测最长值（4043）的 3.5 倍。思维链长度在不同调用之间波动极大，
    所以"被预算截断"是**随机事件**，不是这句话的属性 —— 重试大概率能过。

    代价对比：命中率 1/792，重试上限是 2 次额外调用。便宜得多。
    真正该同时做的仍然是把预算调够（见
    `test_judge_output_budget_is_sized_for_a_reasoning_models_cot`），
    但那是**下次开跑前**的事 —— 改了 `max_tokens` 会作废检查点，
    不能等跑完 35% 才发现。
    """
    llm = _FlakyLLM(fail_times=0, payload="我觉得这句话挺好的，给 1 分吧。")
    judge = J.LLMJudge(llm, sleep=_no_sleep)
    verdict = judge.judge("in_character", reply="好嘞，稍等")

    assert verdict.judged is False, "重试到底仍然解析不了，就该老实记未判"
    assert llm.attempts == 3, "初次 + 2 次重试"
    assert judge.parse_retries == 2
    # 报出来的必须是**解析失败**的原因，不能被重试过程中的其它错误盖掉
    assert "没有 score 字段" in verdict.error


def test_the_original_parse_error_survives_a_later_network_blip() -> None:
    """重试会把原始原因盖掉 —— 这里钉住"不许盖"。

    第一次返回了没格式的内容（模型答了，但答得没法解析），
    第二三次撞上网络抖动。如果直接报最后一次的错误，读者会去查网络，
    而真正的问题是输出格式。**解析失败说明模型回应了，它更接近真相。**
    """
    class _ParseThenBoom:
        name = "parse-then-boom"

        def __init__(self) -> None:
            self.attempts = 0

        @property
        def available(self) -> bool:
            return True

        def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003
            self.attempts += 1
            if self.attempts == 1:
                return "我觉得挺好的。"
            raise TimeoutError("The read operation timed out")

    judge = J.LLMJudge(_ParseThenBoom(), sleep=_no_sleep)
    verdict = judge.judge("in_character", reply="好嘞，稍等")
    assert verdict.judged is False
    assert "没有 score 字段" in verdict.error
    assert "timed out" not in verdict.error


def test_judge_retries_are_reported_separately_from_calls() -> None:
    """重试次数要出现在统计里：它回答"这次判分有多不稳"。"""
    judge = J.LLMJudge(_FlakyLLM(fail_times=1), sleep=_no_sleep)
    results = _fake_results(2)
    _, stats = J.judge_report_cases(
        results,
        judge=judge,
        persona_of=lambda _s, _spk="": "你是阿柚",
        scene_of=lambda _s: "你在吧台",
        concurrency=1,
    )
    assert stats["judge_retries"] >= 1
    assert "judge_retries" in stats



# --------------------------------------------------------------------------- #
# 解析失败也要重试：思维链长度是随机的，所以"被预算截断"不是这句话的属性
# --------------------------------------------------------------------------- #
class _ParseThenOK:
    """前 N 次返回**解析不了**的内容，之后正常返回。

    模拟的正是实测遇到的那个故障：推理模型的思维链长度在不同调用之间
    波动极大（实测 4043 → 14440 字），偶尔会把预算吃光 → 空内容。
    """

    name = "parse-flaky"

    def __init__(self, bad_times: int, bad: str = "", payload: str = '{"score": 1, "reason": "还行"}'):
        self.remaining = bad_times
        self.bad = bad
        self.payload = payload
        self.attempts = 0

    @property
    def available(self) -> bool:
        return True

    def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.attempts += 1
        if self.remaining > 0:
            self.remaining -= 1
            return self.bad
        return self.payload


def test_an_empty_response_is_retried_not_accepted_as_unjudged() -> None:
    """**这条推翻了代码里原本的假设。**

    原来的注释写着"解析失败通常是确定性的（思维链把预算吃光 → 空内容），
    重试只是白烧钱"。实测证明它是错的：跑批里出现过一条思维链 14440 字，
    是之前实测最长值（4043）的 3.5 倍。思维链长度是**随机**的，
    所以"被预算截断"是随机事件 —— 重试大概率就能过。
    """
    llm = _ParseThenOK(bad_times=1, bad="")   # 第一次空内容
    judge = J.LLMJudge(llm, sleep=_no_sleep)
    verdict = judge.judge("grounded", reply="我这儿还有一袋豆子。")
    assert verdict.judged is True
    assert verdict.score == 1
    assert llm.attempts == 2, "应该重试了一次"
    assert judge.parse_retries == 1
    assert judge.retries == 1


def test_a_response_without_a_score_field_is_retried() -> None:
    llm = _ParseThenOK(bad_times=1, bad="我觉得这句挺好的，没什么问题。")
    judge = J.LLMJudge(llm, sleep=_no_sleep)
    assert judge.judge("responsive", reply="嗯——到十点。").judged is True
    assert judge.parse_retries == 1


def test_retrying_parse_failures_is_bounded_and_still_gives_up() -> None:
    """重试是有上限的 —— 一直解析不了就要老实记"未判"，不能无限烧钱。"""
    llm = _ParseThenOK(bad_times=99, bad="")
    judge = J.LLMJudge(llm, max_retries=2, sleep=_no_sleep)
    verdict = judge.judge("grounded", reply="我这儿还有一袋豆子。")
    assert verdict.judged is False
    assert llm.attempts == 3, "初次 + 2 次重试"
    assert judge.parse_retries == 2


def test_call_failures_and_parse_failures_are_counted_separately() -> None:
    """两类重试分开计数：只有解析重试才说明"预算被思维链吃穿了"。

    合并成一个 `retries` 就看不出这一点了 —— 而这两件事的处置办法
    完全不同（一个查网络，一个加预算）。
    """
    call_llm = _FlakyLLM(fail_times=1)
    call_judge = J.LLMJudge(call_llm, sleep=_no_sleep)
    call_judge.judge("grounded", reply="有豆子。")
    assert (call_judge.retries, call_judge.parse_retries) == (1, 0)

    parse_judge = J.LLMJudge(_ParseThenOK(bad_times=1), sleep=_no_sleep)
    parse_judge.judge("grounded", reply="有豆子。")
    assert (parse_judge.retries, parse_judge.parse_retries) == (1, 1)


def test_a_missing_reply_is_still_not_retried() -> None:
    """没有台词可判 = 真确定性，**不能**重试。

    它和"解析失败"长得像（都是 unjudged），但性质完全不同：
    这句台词本来就是空的，重试一万次还是空的。
    """
    llm = _ParseThenOK(bad_times=0)
    judge = J.LLMJudge(llm, sleep=_no_sleep)
    verdict = judge.judge("grounded", reply="   ")
    assert verdict.judged is False
    assert "没有台词可判" in verdict.error
    assert llm.attempts == 0, "连模型都不该调用"
    assert judge.retries == 0


# --------------------------------------------------------------------------- #
# 判分的上下文：喂错了材料，裁判会"讲道理地"惩罚正确的行为
#
# 这一组是被真实事故逼出来的。两个 bug 都是**我们的**，不是模型的：
#   1. 现场块漏了演员表 → 32.5% 的「事实一致」判 0 是把同伴名字当编造；
#   2. 人设块只取第一个 NPC → duet 里小舟的「角色口吻」失败率 90%（阿柚 39%）。
# 共同点：裁判按它拿到的材料判得没错，是材料错了。
# 这和本项目早先踩过的"安全子串黑名单把正确的拒绝判成泄露"是同一个病。
# --------------------------------------------------------------------------- #
def _duet_scenario() -> dict:
    from npc_agent.config import load_scenario

    return load_scenario("duet")


def test_the_persona_card_follows_the_speaker_not_the_first_npc() -> None:
    """**这是上面那两个 bug 里更严重的那个。**

    duet 里站着阿柚和小舟，人设完全不同（一个温和调侃带口头禅、
    一个话少）。原来整条用例只取第一张卡，于是小舟的台词拿阿柚的卡去判 ——
    实测「角色口吻」失败率 90%，而阿柚自己只有 39%。
    """
    from npc_agent.cli import _persona_block

    scenario = _duet_scenario()
    ayou = _persona_block(scenario, "阿柚")
    xiaozhou = _persona_block(scenario, "小舟")

    assert ayou and xiaozhou
    assert ayou != xiaozhou, "两个发言者拿到的卡必须不同"
    assert "阿柚" in ayou
    assert "小舟" in xiaozhou
    assert "小舟" not in ayou, "小舟的卡不该出现在阿柚的判分材料里"


def test_the_persona_card_matches_by_actor_id_too() -> None:
    """转写里 speaker 是显示名，但演员表用的是 id —— 两种都要能匹配上。"""
    from npc_agent.cli import _persona_block

    scenario = _duet_scenario()
    assert _persona_block(scenario, "xiaozhou") == _persona_block(scenario, "小舟")


def test_an_unknown_speaker_falls_back_instead_of_going_blank() -> None:
    """认不出来的发言者要退回第一张卡，不能返回空串。

    返回空串会让裁判"没有人设可依"地瞎判 —— 比拿错卡更难发现，
    因为报告里看不出异常。
    """
    from npc_agent.cli import _persona_block

    scenario = _duet_scenario()
    assert _persona_block(scenario, "查无此人") == _persona_block(scenario, "")


def test_the_scene_block_lists_the_cast_and_the_players() -> None:
    """现场块必须给出**场上有哪些人**。

    不给的后果实测过：209 条 grounded 判 0 里 68 条（32.5%）是把
    NPC 正常提到同伴名字判成「编造现场不存在的人物」。
    """
    from npc_agent.cli import _scene_block

    block = _scene_block(_duet_scenario(), "duet")
    for who in ("阿柚", "小舟", "阿澈", "小满"):
        assert who in block, f"现场块里应该有 {who}"
    assert "不算编造" in block, "要明说提到他们不算编造，否则裁判还是会按字面判"


def test_the_scene_block_does_not_leak_what_anyone_said() -> None:
    """只给名字和身份，**不给**"他当时说了什么"。

    事后判分拿不到当时的对话状态；把静态配置里的东西伪装成"玩家说过"
    会让裁判判出一个根本不存在的现场。
    """
    from npc_agent.cli import _scene_block

    block = _scene_block(_duet_scenario(), "duet")
    assert "说过" not in block
    assert "台词" not in block


# --------------------------------------------------------------------------- #
# 上下文变了，检查点就必须作废
# --------------------------------------------------------------------------- #
def test_the_prompt_digest_changes_when_the_context_changes() -> None:
    """指纹要覆盖"我们喂了什么"，否则改好上下文再 --resume 会静默拼接。"""
    assert J.prompt_digest({"scene:duet": "A"}) != J.prompt_digest({"scene:duet": "B"})
    assert J.prompt_digest({"scene:duet": "A"}) == J.prompt_digest({"scene:duet": "A"})
    # 键的顺序不该影响结果
    assert J.prompt_digest({"a": "1", "b": "2"}) == J.prompt_digest({"b": "2", "a": "1"})


def test_the_prompt_digest_covers_the_rubric_text_not_just_its_name() -> None:
    """`rubrics` 字段只记标准**名字**，改标准**文本**它看不出来。

    这是另一个静默路径：把 rubric 措辞改好、再 --resume，
    新旧判决混成一列，而它们用的不是同一套标准。
    """
    import dataclasses

    before = J.prompt_digest({})
    original = J.RUBRICS["grounded"]
    J.RUBRICS["grounded"] = dataclasses.replace(original, pass_when="改过的标准")
    try:
        assert J.prompt_digest({}) != before
    finally:
        J.RUBRICS["grounded"] = original


def test_a_changed_prompt_context_refuses_to_resume() -> None:
    """**这条是那两个 bug 的护栏。** 上下文变了就必须拒绝恢复。"""
    judge = J.LLMJudge(ScriptedLLM(['{"score": 1, "reason": "ok"}']), name="m")
    results = [{"case_id": "c1", "scenario": "duet", "speeches": ["你好"]}]

    old_fp = J.judge_fingerprint(judge, results, prompt_context={"scene:duet": "旧的现场块"})
    payload = {
        "fingerprint": old_fp,
        "total": 1,
        "done": 1,
        "judgements": [{"case_id": "c1", "pairs": [
            {"pair": {"reply": "你好"}, "verdicts": [
                {"rubric": "grounded", "judged": True, "score": 1.0, "reason": "ok"}]}]}],
    }
    # 同上下文 -> 可以复用
    usable, note = J.plan_judge_resume(payload, old_fp)
    assert len(usable) == 1, note

    # 补了演员表（上下文变了）-> 必须拒绝，并且说清是哪一项变了
    new_fp = J.judge_fingerprint(judge, results, prompt_context={"scene:duet": "补了演员表的现场块"})
    usable, note = J.plan_judge_resume(payload, new_fp)
    assert usable == {}
    assert "prompt_digest" in note


def test_a_refusal_must_always_name_what_changed() -> None:
    """**护栏必须说得出理由。**

    原来的解释逻辑硬编码了一张字段表。将来加字段时，那张表会漏，
    于是出现「拒绝恢复（…）：」后面**什么都没有**的消息 ——
    而一个说不出理由的护栏，最后一定会被人 `--force` 掉。
    这里用一个不在任何硬编码表里的字段来钉住：取并集，就一定能列出来。
    """
    judge = J.LLMJudge(ScriptedLLM(['{"score": 1, "reason": "ok"}']), name="m")
    results = [{"case_id": "c1", "scenario": "duet", "speeches": ["你好"]}]

    recorded = J.judge_fingerprint(judge, results, prompt_context={"scene:duet": "x"})
    payload = {
        "fingerprint": recorded,
        "total": 1,
        "done": 1,
        "judgements": [{"case_id": "c1", "pairs": [
            {"pair": {"reply": "你好"}, "verdicts": [
                {"rubric": "grounded", "judged": True, "score": 1.0, "reason": "ok"}]}]}],
    }

    # 模拟"将来新增了一个影响结果的字段" —— 它不在 JUDGE_RESUME_CRITICAL_FIELDS 里
    assert "future_knob" not in J.JUDGE_RESUME_CRITICAL_FIELDS
    new_fp = dict(recorded, future_knob="开了")
    usable, note = J.plan_judge_resume(payload, new_fp)
    assert usable == {}
    assert "future_knob" in note, f"拒绝了却没说为什么：{note!r}"

    # 反向：老指纹少了新字段（真实场景 —— 加了 prompt_digest 之后，
    # 旧检查点里根本没有这个键），也必须列出来而不是沉默
    trimmed = {k: v for k, v in recorded.items() if k != "report_digest"}
    usable, note = J.plan_judge_resume(payload, dict(trimmed, prompt_digest="新的"))
    assert usable == {}
    assert "prompt_digest" in note and "report_digest" in note, note


def test_judge_pairs_asks_for_the_persona_of_each_speaker() -> None:
    """`judge_pairs` 要按发言者取卡，而不是整串用一张。"""
    asked: list[str] = []

    def persona_for(speaker: str) -> str:
        asked.append(speaker)
        return f"你是{speaker}"

    judge = J.LLMJudge(ScriptedLLM(lambda _p: '{"score": 1, "reason": "ok"}'))
    pairs = [
        {"player": "在吗", "speaker": "阿柚", "reply": "嗯——在。"},
        {"player": "在吗", "speaker": "小舟", "reply": "嗯。"},
    ]
    judge.judge_pairs(pairs, persona_of=persona_for, scene="")
    assert asked == ["阿柚", "小舟"]


# --------------------------------------------------------------------------- #
# 校准也要并行：它是一笔每次判分都要重付的固定税
# --------------------------------------------------------------------------- #
def _reply_keyed_judge(
    scores: dict[str, int], latency: dict[str, float] | None = None
) -> J.LLMJudge:
    """按**台词**决定判决的裁判，可以给每条台词配一个人为延迟。

    延迟是用来**故意打乱完成顺序**的：只有让先发的请求后回来，
    才测得出"按完成顺序累积"这个 bug。用真实模型是测不出来的 ——
    它回来的顺序本身就不可控。
    """
    latency = latency or {}

    def fn(prompt: str) -> str:
        for reply, score in scores.items():
            if reply in prompt:
                if latency.get(reply):
                    time.sleep(latency[reply])
                return json.dumps({"score": score, "reason": "脚本"})
        return json.dumps({"score": 0, "reason": "没匹配上"})

    return J.LLMJudge(ScriptedLLM(fn))


def _fake_calibration_items(n: int = 4) -> list[dict]:
    return [
        {"id": f"f{i}", "rubric": "grounded", "reply": f"台词{i}", "label": 1}
        for i in range(n)
    ]


def test_parallel_calibration_lands_results_by_index_not_completion_order() -> None:
    """**这条是并行校准唯一真正的风险点。**

    校准报告里的 `disagreements` 是给人读的列表。如果按**完成顺序**追加，
    同一份校准集跑两次会得到两个顺序不同的报告 —— 本项目有一条
    "同一输入跑两次结果必须一致"的确定性断言，那会直接违反它。

    这里让第 0 条最慢：并发跑的话它**最后**回来。
    """
    items = _fake_calibration_items(4)
    scores = {f"台词{i}": (0 if i in (0, 2) else 1) for i in range(4)}
    latency = {f"台词{i}": (0.35 if i == 0 else 0.01) for i in range(4)}

    report = J.calibrate(_reply_keyed_judge(scores, latency), items, concurrency=4)
    assert [d["id"] for d in report.disagreements] == ["f0", "f2"], (
        "判决必须按 items 下标落位，不能按完成顺序追加"
    )


def test_parallel_calibration_gives_the_same_report_as_serial() -> None:
    items = _fake_calibration_items(6)
    scores = {f"台词{i}": (0 if i % 3 == 0 else 1) for i in range(6)}

    serial = J.calibrate(_reply_keyed_judge(scores), items, concurrency=1).to_dict()
    parallel = J.calibrate(_reply_keyed_judge(scores), items, concurrency=4).to_dict()
    assert serial == parallel


def test_calibration_at_concurrency_one_does_not_build_a_thread_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """串行基线必须真的是串行的，否则"并行没改变结果"就没法验证。"""

    def _boom(*_a: object, **_k: object) -> object:
        raise AssertionError("并发 1 不该建线程池")

    monkeypatch.setattr(J, "ThreadPoolExecutor", _boom)
    report = J.calibrate(_reply_keyed_judge({"台词0": 1}), _fake_calibration_items(1),
                         concurrency=1)
    assert report.judged == 1


def test_a_single_item_does_not_build_a_thread_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """一条样本开线程池纯属浪费 —— 单条时走串行路径。"""

    def _boom(*_a: object, **_k: object) -> object:
        raise AssertionError("单条样本不该建线程池")

    monkeypatch.setattr(J, "ThreadPoolExecutor", _boom)
    report = J.calibrate(_reply_keyed_judge({"台词0": 1}), _fake_calibration_items(1),
                         concurrency=8)
    assert report.judged == 1


def test_parallel_calibration_never_turns_unjudged_into_zero() -> None:
    """铁律一在并行路径上同样成立 —— 未判绝不能变成 0 分。"""
    report = J.calibrate(J.LLMJudge(NullLLM()), _fake_calibration_items(5), concurrency=4)
    assert report.judged == 0
    assert report.unjudged == 5
    assert report.per_rubric == {}
    assert {d["kind"] for d in report.disagreements} == {"unjudged"}


# --------------------------------------------------------------------------- #
# 铁律三：裁判必须看得见**这一对之前**的对话
#
# 判据不是"分数变高了"，而是**裁判拿到的材料里有没有能区分两种历史的那个信息**。
# 前三条断言都是确定性的、不需要模型的 —— 它们证明缺陷真实存在。
# --------------------------------------------------------------------------- #
def _history_pairs() -> list[dict[str, str]]:
    """一段五轮的对话。第 i 对判得对不对，取决于它前面那几轮。"""
    return [
        {"speaker": "阿柚", "player": "店里的豆子是哪儿来的？", "reply": "都是我自己烘的。"},
        {"speaker": "阿柚", "player": "那挺费工夫的吧。", "reply": "嗯，凌晨四点就得起来。"},
        {"speaker": "阿柚", "player": "再来一杯。", "reply": "好，稍等。"},
        {"speaker": "阿柚", "player": "不要糖。", "reply": "记下了。"},
        {"speaker": "阿柚", "player": "谢谢。", "reply": "不客气。"},
    ]


def _prompts_of(pairs: list[dict[str, str]], turns: int) -> list[str]:
    """把 `judge_pairs` 真正发出去的 prompt 逐条取回来（走完整条路径，不走 `_build_prompt`）。"""
    llm = ScriptedLLM(lambda _text: '{"score": 1, "reason": "ok"}')
    judge = J.LLMJudge(llm, rubrics=["responsive"], history_turns=turns)
    judge.judge_pairs(pairs)
    return ["\n".join(m.get("content", "") for m in call) for call in llm.calls]


def test_the_history_block_never_contains_the_pair_being_judged() -> None:
    """**把答案递给裁判**是这里最容易犯的错 —— 历史里绝不能有当前这一对。

    带上待评台词之后，「是否回应」会退化成"把上一句抄一遍"，
    「角色口吻」会变成"照着刚才那句的风格再判一次"。
    """
    pairs = _history_pairs()
    for index in range(len(pairs)):
        block = J.history_block(pairs, index, J.JUDGE_HISTORY_TURNS)
        assert pairs[index]["reply"] not in block, f"第 {index} 对的历史漏进了它自己的台词"
        assert pairs[index]["player"] not in block, f"第 {index} 对的历史漏进了它自己的玩家话"

    # 反向：证明上面那两条不是永真式 —— 窗口挪一格，更早的那一对**应该**出现。
    assert pairs[0]["reply"] in J.history_block(pairs, 2, J.JUDGE_HISTORY_TURNS)


def test_the_history_block_is_the_preceding_turns_in_order() -> None:
    pairs = _history_pairs()
    assert J.history_block(pairs, 0, 4) == "", "第一对没有历史"
    assert J.history_block(pairs, 1, 4) == "玩家：店里的豆子是哪儿来的？\n阿柚：都是我自己烘的。"
    # 窗口 1：只带紧挨着的那一对
    assert J.history_block(pairs, 3, 1) == "玩家：再来一杯。\n阿柚：好，稍等。"
    # 窗口大于已发生的轮数时不补空，全带上
    block = J.history_block(pairs, 3, 4)
    assert "店里的豆子是哪儿来的？" in block
    assert "不要糖。" not in block
    # turns=0 一律空串
    assert J.history_block(pairs, 3, 0) == ""


def test_history_labels_each_line_with_its_own_speaker() -> None:
    """多 NPC 场景里，历史里每一句必须标**说话人自己**的名字。"""
    pairs = [
        {"speaker": "阿柚", "player": "今天谁看店？", "reply": "我看前厅。"},
        {"speaker": "老周", "player": "那后厨呢？", "reply": "后厨归我。"},
        {"speaker": "阿柚", "player": "知道了。", "reply": "嗯。"},
    ]
    block = J.history_block(pairs, 2, 4)
    assert "阿柚：我看前厅。" in block
    assert "老周：后厨归我。" in block
    assert block.index("阿柚：我看前厅。") < block.index("老周：后厨归我。")


def test_a_pair_without_a_speaker_falls_back_to_npc() -> None:
    pairs = [{"player": "在吗？", "reply": "在。"}, {"player": "好。", "reply": "嗯。"}]
    assert J.history_block(pairs, 1, 4) == "玩家：在吗？\nNPC：在。"


def test_without_history_two_different_pasts_look_identical_to_the_judge() -> None:
    """**这就是那个缺陷本身**，而且是确定性的、不需要模型、不需要额度。

    同一句回复、同一句玩家话，前面发生的事完全不同 —— 不带历史时裁判拿到的
    prompt **逐字节相同**，所以它**不可能**判出区别：它会按拿到的材料判得没错，
    然后判错正确的那一次。
    """
    tail = {"speaker": "阿柚", "player": "不要糖。", "reply": "好，我记下了。"}
    latte = [{"speaker": "阿柚", "player": "来杯拿铁。", "reply": "好，稍等。"}, tail]
    chitchat = [{"speaker": "阿柚", "player": "你叫什么名字？", "reply": "我叫阿柚。"}, tail]

    assert _prompts_of(latte, 0)[-1] == _prompts_of(chitchat, 0)[-1], (
        "不带历史时两条 prompt 竟然不同 —— 说明有别的东西泄漏进来了"
    )
    assert (
        _prompts_of(latte, J.JUDGE_HISTORY_TURNS)[-1]
        != _prompts_of(chitchat, J.JUDGE_HISTORY_TURNS)[-1]
    ), "带上历史之后 prompt 仍然相同 —— 历史根本没被用上"


def test_history_turns_zero_reproduces_the_old_behaviour() -> None:
    prompts = _prompts_of(_history_pairs(), 0)
    assert all("【之前的对话】" not in p for p in prompts)
    # 默认值下：第一对没有历史，从第二对起都有。
    prompts4 = _prompts_of(_history_pairs(), J.JUDGE_HISTORY_TURNS)
    assert "【之前的对话】" not in prompts4[0]
    assert all("【之前的对话】" in p for p in prompts4[1:])


def test_history_is_placed_before_the_pair_under_judgment() -> None:
    """顺序就是时间顺序 —— 历史排在「玩家刚说」之前，别让裁判以为它发生在后面。"""
    prompt = _prompts_of(_history_pairs(), J.JUDGE_HISTORY_TURNS)[3]
    assert prompt.index("【之前的对话】") < prompt.index("【玩家刚说】")


def test_history_turns_changes_the_fingerprint() -> None:
    llm = ScriptedLLM(lambda _text: '{"score": 1, "reason": "ok"}')
    results: list[dict] = []
    fp0 = J.judge_fingerprint(J.LLMJudge(llm, history_turns=0), results)
    fp4 = J.judge_fingerprint(J.LLMJudge(llm, history_turns=4), results)
    assert fp0["history_turns"] == 0
    assert fp4["history_turns"] == 4
    assert fp0 != fp4, "指纹没变 ⇒ `--resume` 会把两种判法拼成一份报告"
    assert "history_turns" in J.JUDGE_RESUME_CRITICAL_FIELDS


def test_history_turns_defaults_to_the_constant_and_is_clamped() -> None:
    llm = ScriptedLLM(lambda _text: '{"score": 1, "reason": "ok"}')
    assert J.LLMJudge(llm).history_turns == J.JUDGE_HISTORY_TURNS
    assert J.LLMJudge(llm, history_turns=-5).history_turns == 0


def test_the_calibration_set_cannot_see_dialogue_history() -> None:
    """校准集里没有历史 ⇒ **已经发布的 kappa 数字不受这次改动影响**。

    这是好事，同时也是一条限制：**校准检测不到这个改动**，
    所以它不能被拿来当"改动有效"的证据 —— 那要另做探针
    （`scripts/probe_judge_history.py`）。
    """
    assert all("history" not in item for item in J.load_calibration())
    llm = ScriptedLLM(lambda _text: '{"score": 1, "reason": "ok"}')
    judge = J.LLMJudge(llm, history_turns=J.JUDGE_HISTORY_TURNS)
    J.calibrate(judge, J.load_calibration()[:4], concurrency=1)
    prompts = ["\n".join(m.get("content", "") for m in call) for call in llm.calls]
    assert prompts, "一次 prompt 都没抓到 —— 这条断言什么都没验证"
    assert all("【之前的对话】" not in p for p in prompts)
