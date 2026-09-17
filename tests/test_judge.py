"""LLM-as-judge 的测试。

这个文件的重点是三条铁律各自的回归测试：
    1. 判不了就说判不了 —— 绝不能变成 0 分混进均值
    2. 未经校准的裁判不算证据 —— kappa 要算对
    3. 位置偏见要能被测出来 —— 不是声明一下就算缓解了
"""

from __future__ import annotations

import json
import re

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
        persona_of=lambda _sid: "你是阿柚",
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


def test_a_bad_output_format_is_not_retried() -> None:
    """解析失败不重试。

    它通常是确定性的 —— 思维链把预算吃光就会稳定地返回空内容，
    重试只是白烧钱。真正该做的是把预算调够（见
    `test_judge_output_budget_is_sized_for_a_reasoning_models_cot`）。
    """
    llm = _FlakyLLM(fail_times=0, payload="我觉得这句话挺好的，给 1 分吧。")
    judge = J.LLMJudge(llm, sleep=_no_sleep)
    verdict = judge.judge("in_character", reply="好嘞，稍等")

    assert verdict.judged is False
    assert llm.attempts == 1, "解析失败只调一次，不重试"
    assert judge.retries == 0


def test_judge_retries_are_reported_separately_from_calls() -> None:
    """重试次数要出现在统计里：它回答"这次判分有多不稳"。"""
    judge = J.LLMJudge(_FlakyLLM(fail_times=1), sleep=_no_sleep)
    results = _fake_results(2)
    _, stats = J.judge_report_cases(
        results,
        judge=judge,
        persona_of=lambda _s: "你是阿柚",
        scene_of=lambda _s: "你在吧台",
        concurrency=1,
    )
    assert stats["judge_retries"] >= 1
    assert "judge_retries" in stats

