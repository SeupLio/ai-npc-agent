"""LLM-as-judge —— 规则指标测不了的那部分，以及"测不了"这件事本身该怎么记。

## 为什么需要它

规则指标（`metrics.py`）能查的是机械可判定的东西：世界状态对不对、
有没有说禁用词、工具调用集合对不对。它们**测不了**这三件事：

    这句话像不像阿柚说的？      （语气 / 口吻 / 句数）
    这句话有没有真的回应玩家？  （而不是自说自话）
    它提到的世界事实对不对？    （而不是编造）

只能靠人判 —— 或者用一个模型去判。

## 三条铁律

### 一、判不了就说判不了，绝不给假分

模型不可用、返回解析不了、理由和分数对不上 —— 一律返回 `judged=False`。
**不能返回 0。** 一个假的 0 会污染所有建立在它上面的均值，
而且它在报告里和"真的答得很差"长得一模一样。

这跟 `stage_share` 那个 bug 是同一类问题：一条不报警的假数据，
比一条会失败的断言危险得多。

### 二、未经校准的裁判等于又一个模型输出

裁判自己也是个模型，也会错。所以必须有一份**人工标注的校准集**，
算出裁判和人的一致率与 Cohen's kappa。kappa 低就说明这个维度不能用 ——
"我们用了 LLM-as-judge"这句话本身不构成任何证据。

### 三、位置偏见必须实测，不能只声明

A/B 对比时裁判倾向于选先出现的那个。缓解办法是**交换顺序再问一遍**，
两次结论一致才认；不一致就判为平局。这不是理论担忧，是可以被测出来的。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from ..llm.base import LLM, LLMUnavailable

#: 人工标注的校准集。**没有它，裁判的分数就只是"另一个模型的意见"。**
CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration.jsonl"

#: 裁判的默认温度。判分要的是稳定，不是创造力。
JUDGE_TEMPERATURE = 0.0


# --------------------------------------------------------------------------- #
# 评分标准
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rubric:
    """一条可判定的标准。

    `pass_when` / `fail_when` 必须写成人能读懂、能照着判的条件。
    只写"这条好不好"是没用的 —— 裁判会按自己的偏好自由发挥，
    校准出来的 kappa 自然低，而且低在哪一条上都看不出来。
    """

    key: str
    name: str
    question: str
    pass_when: str
    fail_when: str
    #: 判分时看什么。缺了它裁判会开始猜（比如不知道人设就开始评判"像不像"）。
    needs: tuple[str, ...] = ("persona", "scene", "player", "reply")

    def prompt_block(self) -> str:
        return (
            f"【评判标准：{self.name}】\n"
            f"问题：{self.question}\n"
            f"判 1 分：{self.pass_when}\n"
            f"判 0 分：{self.fail_when}"
        )


RUBRICS: dict[str, Rubric] = {
    r.key: r
    for r in (
        Rubric(
            key="in_character",
            name="角色口吻",
            question="这句话像不像这个人设说出来的？",
            pass_when=(
                "语气、用词、句数都符合人设卡的描述；读起来就是这个角色在说话。"
                "短句、有自己的口头禅或说话习惯。"
            ),
            fail_when=(
                "像客服 / 像通用助手 / 像百科；长篇大论分点作答；"
                "出现人设卡里明确禁止的词；完全没有人设特征，换个名字也成立。"
                "人设卡里写明的硬性约束（「每次最多说 N 句话」「不得超过 N 字」"
                "「禁止出现的词」）是**规则**，不是风格描述 —— 违反了就算不合格，"
                "哪怕语气很到位。"
            ),
        ),
        Rubric(
            key="responsive",
            name="是否回应",
            question="这句话有没有真的回应玩家刚才说的那句？",
            pass_when=(
                "正面接住了玩家的问题或请求；即使答案是拒绝或不知道，"
                "也是针对这句话给的回应。"
            ),
            fail_when=(
                "答非所问；自说自话地推进自己的话题；把玩家的问题当没看见；"
                "用一句万能话术敷衍（『嗯——我听着呢』）顶掉一个具体问题。"
                "**尤其注意这一种**：对【现场】里明明已经有答案的事说『不知道』"
                "『说不好』。这不算「诚实的不知道」，是拿含糊话躲开问题 —— "
                "判断依据是【现场】里有没有答案，不是这句话听起来谦不谦虚。"
            ),
        ),
        Rubric(
            key="grounded",
            name="事实一致",
            question="这句话里关于世界的事实，和现场情况一致吗？",
            pass_when=(
                "提到的物品位置、自己做过的事、玩家说过的话，都与给出的现场情况相符；"
                "没有编造现场不存在的物品、地点或事件。"
            ),
            fail_when=(
                "声称自己做过没做过的事；提到现场不存在的物品或地点；"
                "把玩家没说过的话说成玩家说过；编造具体数字（价格/时间/数量）。"
            ),
        ),
    )
}

#: 默认要跑的评判标准。故意少而具体 —— 加一堆模糊的维度只会让 kappa 掉下来。
DEFAULT_RUBRICS = ("in_character", "responsive", "grounded")


# --------------------------------------------------------------------------- #
# 判决
# --------------------------------------------------------------------------- #
@dataclass
class Verdict:
    """一次判决。`score=None` 表示**没判**，不是判了 0 分。"""

    rubric: str
    score: Optional[float]
    reason: str = ""
    raw: str = ""
    error: str = ""

    @property
    def judged(self) -> bool:
        return self.score is not None

    @classmethod
    def unjudged(cls, rubric: str, error: str, raw: str = "") -> "Verdict":
        """没判成。**这是正常返回值，不是异常。**

        模型不可用 / 解析不了 / 分数越界，全都走这里。
        调用方必须能区分"裁判说 0 分"和"裁判没说话" ——
        把后者记成前者，均值就变成了一个假的数字。
        """
        return cls(rubric=rubric, score=None, error=error, raw=raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rubric": self.rubric,
            "judged": self.judged,
            "score": self.score,
            "reason": self.reason,
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# 裁判
# --------------------------------------------------------------------------- #
class LLMJudge:
    def __init__(
        self,
        llm: LLM,
        rubrics: Iterable[str] = DEFAULT_RUBRICS,
        *,
        max_tokens: int = 512,
        name: str = "",
    ) -> None:
        self.llm = llm
        self.rubrics = list(rubrics)
        self.max_tokens = max_tokens
        self.name = name or getattr(llm, "name", "judge")
        self.calls = 0

    @property
    def available(self) -> bool:
        return bool(getattr(self.llm, "available", False))

    # ------------------------------------------------------------------ #
    def judge(
        self,
        rubric_key: str,
        *,
        persona: str = "",
        scene: str = "",
        player: str = "",
        reply: str = "",
    ) -> Verdict:
        rubric = RUBRICS.get(rubric_key)
        if rubric is None:
            raise KeyError(f"未知的评判标准 {rubric_key!r}，可选：{sorted(RUBRICS)}")
        if not self.available:
            return Verdict.unjudged(rubric_key, "模型不可用，未判分")
        if not (reply or "").strip():
            # 没有台词就没得判。记成"没判"而不是"0 分"：
            # 沉默可能是正确的（发言占比到顶了就该闭嘴），规则指标已经在别处管这件事。
            return Verdict.unjudged(rubric_key, "没有台词可判")

        prompt = self._build_prompt(rubric, persona, scene, player, reply)
        try:
            raw = self.llm.complete(
                [{"role": "user", "content": prompt}],
                temperature=JUDGE_TEMPERATURE,
                max_tokens=self.max_tokens,
            )
        except LLMUnavailable as exc:
            return Verdict.unjudged(rubric_key, f"模型调用失败：{exc}")
        except Exception as exc:  # 网络/鉴权/超时都算"没判"，不该让跑批崩掉
            return Verdict.unjudged(rubric_key, f"模型调用异常：{type(exc).__name__}: {exc}")

        self.calls += 1
        return self._parse(rubric_key, raw)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_prompt(
        rubric: Rubric, persona: str, scene: str, player: str, reply: str
    ) -> str:
        parts = [
            "你是一个游戏 NPC 对话质量评审。按下面的标准给一句 NPC 台词打分。",
            "",
            rubric.prompt_block(),
            "",
        ]
        if persona:
            parts += ["【人设】", persona, ""]
        if scene:
            parts += ["【现场】", scene, ""]
        if player:
            parts += ["【玩家刚说】", player, ""]
        parts += [
            "【NPC 的台词】",
            reply,
            "",
            "只输出 JSON，不要任何解释："
            '{"score": 0 或 1, "reason": "一句话说明理由"}',
        ]
        return "\n".join(parts)

    def _parse(self, rubric_key: str, raw: str) -> Verdict:
        """解析模型输出。**任何一步不对就判为"没判"，绝不猜。**

        宁可少一个数据点，也不要一个编出来的分数 ——
        编出来的分数没法被审计，也没法被校准发现。
        """
        from ..llm.base import extract_json

        data = extract_json(raw or "")
        if not isinstance(data, dict) or "score" not in data:
            return Verdict.unjudged(rubric_key, "输出里没有 score 字段", raw)

        raw_score = data.get("score")
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            return Verdict.unjudged(rubric_key, f"score 不是数字：{raw_score!r}", raw)

        if score not in (0.0, 1.0):
            # 让它给 0/1，它给了 0.7 —— 这是"没按标准判"，不是"判了 0.7 分"。
            return Verdict.unjudged(rubric_key, f"score 越界：{score}", raw)

        reason = str(data.get("reason") or "").strip()
        if not reason:
            # 没理由的判决没法审计，也没法在校准时看出它为什么错。
            return Verdict.unjudged(rubric_key, "没有给出理由", raw)

        return Verdict(rubric=rubric_key, score=score, reason=reason, raw=raw)

    # ------------------------------------------------------------------ #
    def judge_reply(self, **kwargs: Any) -> list[Verdict]:
        return [self.judge(key, **kwargs) for key in self.rubrics]

    def judge_pairs(
        self,
        pairs: list[dict[str, str]],
        *,
        persona: str = "",
        scene: str = "",
        progress: Any = None,
    ) -> list[dict[str, Any]]:
        """对一串 (玩家说 → NPC 回) 逐条判分。"""
        out: list[dict[str, Any]] = []
        for index, pair in enumerate(pairs, 1):
            verdicts = self.judge_reply(
                persona=persona,
                scene=scene,
                player=pair.get("player", ""),
                reply=pair.get("reply", ""),
            )
            out.append({"pair": pair, "verdicts": [v.to_dict() for v in verdicts]})
            if progress:
                progress(f"  裁判 {index}/{len(pairs)} …")
        return out

    # ------------------------------------------------------------------ #
    # 位置偏见
    # ------------------------------------------------------------------ #
    def compare_pairwise(
        self,
        *,
        persona: str,
        scene: str,
        player: str,
        reply_a: str,
        reply_b: str,
    ) -> dict[str, Any]:
        """比较两句台词哪个更好，并**实测**位置偏见。

        裁判倾向于选先出现的那个（position bias）。缓解办法是问两遍：
        第一遍 A 在前，第二遍 B 在前。两次结论一致才认，不一致判平局。

        为什么不做"换一批样本取平均"这种统计缓解：那需要很多次调用，
        而且掩盖了单次判决的不可靠 —— 我们要的是"这条判决可不可信"，
        不是"平均下来还行"。
        """
        if not self.available:
            return {"winner": None, "judged": False, "error": "模型不可用，未判分"}
        if not reply_a.strip() or not reply_b.strip():
            return {"winner": None, "judged": False, "error": "有一侧没有台词"}

        forward = self._ask_preference(persona, scene, player, reply_a, reply_b)
        if forward is None:
            return {"winner": None, "judged": False, "error": "第一遍调用失败"}
        backward = self._ask_preference(persona, scene, player, reply_b, reply_a)
        if backward is None:
            return {"winner": None, "judged": False, "error": "第二遍调用失败"}

        # 第二遍里 A 排在后面，所以 backward == "后者" 才等价于"选 A"
        first_choice = forward
        swapped_choice = {"前者": "后者", "后者": "前者", "平局": "平局"}[backward]

        if first_choice != swapped_choice:
            return {
                "winner": None,
                "judged": True,
                "consistent": False,
                "note": (
                    f"两次判决不一致（A 在前时选 {first_choice}，"
                    f"B 在前时选 {backward}）—— 存在位置偏见，判为平局"
                ),
                "forward": first_choice,
                "backward": backward,
            }
        if first_choice == "平局":
            return {"winner": None, "judged": True, "consistent": True, "note": "两次都判平局"}
        return {
            "winner": "a" if first_choice == "前者" else "b",
            "judged": True,
            "consistent": True,
            "forward": first_choice,
            "backward": backward,
        }

    def _ask_preference(
        self, persona: str, scene: str, player: str, first: str, second: str
    ) -> Optional[str]:
        prompt = "\n".join(
            [
                "你是一个游戏 NPC 对话质量评审。下面两条 NPC 台词，哪一条更好？",
                "",
                "评判角度：是否贴合人设、是否回应了玩家、事实是否与现场一致。",
                "",
                *(["【人设】", persona, ""] if persona else []),
                *(["【现场】", scene, ""] if scene else []),
                *(["【玩家刚说】", player, ""] if player else []),
                "【第一条】",
                first,
                "",
                "【第二条】",
                second,
                "",
                '只输出 JSON：{"choice": "前者" 或 "后者" 或 "平局", "reason": "一句话理由"}',
            ]
        )
        try:
            raw = self.llm.complete(
                [{"role": "user", "content": prompt}],
                temperature=JUDGE_TEMPERATURE,
                max_tokens=self.max_tokens,
            )
        except Exception:
            return None
        self.calls += 1
        from ..llm.base import extract_json

        data = extract_json(raw or "")
        choice = str(data.get("choice") or "").strip()
        if choice in ("前者", "后者", "平局"):
            return choice
        return None


# --------------------------------------------------------------------------- #
# 校准
# --------------------------------------------------------------------------- #
def cohen_kappa(labels: list[int], predictions: list[int]) -> float:
    """Cohen's kappa：扣掉"碰巧一致"之后的一致程度。

    一致率（agreement）会骗人：一个永远输出 1 的裁判，在 80% 正例的集合上
    一致率就是 80%，看起来挺好，实际什么都没判出来。kappa 扣掉了这个底。
    """
    if not labels or len(labels) != len(predictions):
        raise ValueError("labels 和 predictions 必须等长且非空")
    n = len(labels)
    observed = sum(1 for a, b in zip(labels, predictions) if a == b) / n
    # 期望一致率：按各自的边际分布独立算
    expected = 0.0
    for value in (0, 1):
        p_label = sum(1 for a in labels if a == value) / n
        p_pred = sum(1 for b in predictions if b == value) / n
        expected += p_label * p_pred
    if expected >= 1.0:
        # 退化情况：两边都是同一个常数。一致就是完全一致，不一致就是完全不一致。
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def kappa_reading(value: float) -> str:
    """把 kappa 翻译成人话。数值本身不构成结论。"""
    if value >= 0.8:
        return "几乎完全一致，这个维度可以用"
    if value >= 0.6:
        return "基本一致，可以用，但个例要人看"
    if value >= 0.4:
        return "中等一致，只能当参考信号，不能当结论"
    if value >= 0.0:
        return "一致程度和瞎猜差不多 —— 这个维度不能用"
    return "比瞎猜还差，说明标准写反了或裁判理解错了"


@dataclass
class CalibrationReport:
    total: int = 0
    judged: int = 0
    per_rubric: dict[str, dict[str, Any]] = field(default_factory=dict)
    disagreements: list[dict[str, Any]] = field(default_factory=list)

    @property
    def unjudged(self) -> int:
        return self.total - self.judged

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "judged": self.judged,
            "unjudged": self.unjudged,
            "per_rubric": self.per_rubric,
            "disagreements": self.disagreements,
        }


def load_calibration(path: str | Path | None = None) -> list[dict[str, Any]]:
    target = Path(path) if path else CALIBRATION_FILE
    items: list[dict[str, Any]] = []
    for line_no, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{target.name}:{line_no} JSON 解析失败: {exc}") from exc
    return items


def calibrate(
    judge: LLMJudge,
    items: Optional[list[dict[str, Any]]] = None,
    *,
    progress: Any = None,
) -> CalibrationReport:
    """拿人工标注的样本校准裁判。

    **没跑过这一步，就不该在报告里写"我们用了 LLM-as-judge"。**
    一个没校准的裁判只是"另一个模型的意见"，把它写进结论是拿权威感代替证据。
    """
    items = items if items is not None else load_calibration()
    report = CalibrationReport(total=len(items))

    collected: dict[str, dict[str, list[int]]] = {}
    for index, item in enumerate(items, 1):
        rubric_key = item["rubric"]
        verdict = judge.judge(
            rubric_key,
            persona=item.get("persona", ""),
            scene=item.get("scene", ""),
            player=item.get("player", ""),
            reply=item.get("reply", ""),
        )
        if progress:
            progress(f"  校准 {index}/{len(items)} …")
        if not verdict.judged:
            report.disagreements.append(
                {
                    "id": item.get("id"),
                    "rubric": rubric_key,
                    "kind": "unjudged",
                    "error": verdict.error,
                    "human_label": item.get("label"),
                }
            )
            continue
        report.judged += 1
        bucket = collected.setdefault(rubric_key, {"labels": [], "preds": []})
        bucket["labels"].append(int(item["label"]))
        bucket["preds"].append(int(verdict.score))
        if int(verdict.score) != int(item["label"]):
            report.disagreements.append(
                {
                    "id": item.get("id"),
                    "rubric": rubric_key,
                    "kind": "mismatch",
                    "human_label": int(item["label"]),
                    "judge_label": int(verdict.score),
                    "judge_reason": verdict.reason,
                    "note": item.get("note", ""),
                    "reply": item.get("reply", ""),
                }
            )

    for rubric_key, bucket in collected.items():
        labels, preds = bucket["labels"], bucket["preds"]
        agreement = sum(1 for a, b in zip(labels, preds) if a == b) / len(labels)
        kappa = cohen_kappa(labels, preds)
        report.per_rubric[rubric_key] = {
            "n": len(labels),
            "agreement": round(agreement, 3),
            "kappa": round(kappa, 3),
            "reading": kappa_reading(kappa),
            "confusion": {
                "真阳性": sum(1 for a, b in zip(labels, preds) if a == 1 and b == 1),
                "真阴性": sum(1 for a, b in zip(labels, preds) if a == 0 and b == 0),
                "假阳性": sum(1 for a, b in zip(labels, preds) if a == 0 and b == 1),
                "假阴性": sum(1 for a, b in zip(labels, preds) if a == 1 and b == 0),
            },
        }
    return report


def render_calibration(report: CalibrationReport) -> str:
    lines = [
        f"校准集 {report.total} 条｜判出 {report.judged}｜未判 {report.unjudged}",
    ]
    for key, stats in sorted(report.per_rubric.items()):
        lines.append(
            f"  {RUBRICS[key].name}({key}): n={stats['n']} "
            f"一致率={stats['agreement']:.0%} kappa={stats['kappa']:.2f} "
            f"→ {stats['reading']}"
        )
        conf = stats["confusion"]
        lines.append(
            f"      混淆: 真阳 {conf['真阳性']} 真阴 {conf['真阴性']} "
            f"假阳 {conf['假阳性']} 假阴 {conf['假阴性']}"
        )
    if report.unjudged:
        lines.append(
            f"  [yellow]有 {report.unjudged} 条没判成[/yellow]："
            "它们不会以 0 分的形式混进均值 —— 未判就是未判"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 从转写里还原对话对
# --------------------------------------------------------------------------- #
def dialogue_pairs(transcript: list[str]) -> list[dict[str, str]]:
    """从一次跑批的转写里还原「玩家说了什么 → NPC 回了什么」。

    转写是唯一保留完整对话顺序的产物：`speeches` 只有 NPC 那半边，
    没有它，裁判就不知道 NPC 在回应什么，"是否回应"这个维度根本没法判。

    转写行格式（harness 生成）：
        玩家[阿澈] 来一杯拿铁
          [ok] 阿柚 move_to(location=kitchen)
        阿柚: 好，稍等。
    """
    pairs: list[dict[str, str]] = []
    last_player = ""
    last_speaker = ""
    for line in transcript:
        if line.startswith("玩家["):
            _, _, rest = line.partition("]")
            last_player = rest.strip()
        elif line.startswith("  ["):
            continue
        elif ": " in line and not line.startswith(" "):
            speaker, _, text = line.partition(": ")
            # NPC 连续说两句时，只把第一句和玩家那句配对；
            # 后一句同样算回应（同一轮里可以有两句台词）。
            pairs.append(
                {
                    "player": last_player,
                    "speaker": speaker.strip(),
                    "reply": text.strip(),
                }
            )
            last_speaker = speaker.strip()
    return [p for p in pairs if p["reply"]]
