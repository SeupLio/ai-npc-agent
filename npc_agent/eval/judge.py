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

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from ..llm.base import LLM, LLMUnavailable

#: 人工标注的校准集。**没有它，裁判的分数就只是"另一个模型的意见"。**
CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration.jsonl"

#: 留出集：写好标签时**没见过裁判输出**，之后也不再回头改 rubric。
#:
#: 为什么要有第二份：`calibration.jsonl` 那 24 条被用来改过三次 rubric，
#: 所以在它上面算出来的 kappa 里有一部分是**拟合**，不是泛化能力。
#: 往同一份里继续加样本也没用 —— 新样本马上又会参与调 rubric。
#: 唯一能拿到干净数字的办法是：先写好、封存、不碰，再一次性算。
HOLDOUT_FILE = Path(__file__).resolve().parent / "calibration_holdout.jsonl"

#: 留出集的封条。记下文件摘要 + 评分标准摘要 + 标签分布。
HOLDOUT_SEAL_FILE = Path(__file__).resolve().parent / "holdout_seal.json"

#: 裁判的默认温度。判分要的是稳定，不是创造力。
JUDGE_TEMPERATURE = 0.0

#: 判分调用失败时的重试次数与退避基数。
#:
#: 判分是**几小时**的长作业（228 条 ≈ 2900 次调用），而模型客户端有 60s
#: 读超时 —— 一次网络抖动就会永久丢掉一条判决：`judge()` 把异常转成
#: `Verdict.unjudged` 返回，而"未判"是**合法返回值**，不会重试、不会报错。
#: 跑批那边早就有重试了，判分一直漏着。
#:
#: 只对**调用失败**重试，不对解析失败重试：解析失败通常是确定性的
#: （思维链把预算吃光 → 空内容），重试只是白烧钱。
JUDGE_MAX_RETRIES = 2
JUDGE_BACKOFF = 3.0


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
                "**「不知道」什么时候算合格**：如果【人设】说这件事不能讲"
                "（未解锁的内容、不该由 NPC 做主的事），或者【现场】里确实没有"
                "相关信息，那么说『不知道 / 说不好 / 这个我做不了主』就是**正确**的回答，"
                "算通过。"
            ),
            fail_when=(
                "答非所问；自说自话地推进自己的话题；把玩家的问题当没看见；"
                "用一句万能话术敷衍（『嗯——我听着呢』）顶掉一个具体问题。"
                "**「不知道」什么时候算不合格**：如果答案就在【现场】里写着、"
                "或者属于这个角色本该知道的事（店主知道自己的营业时间、"
                "知道后厨有什么），却用『这个我还真说不好』含糊过去，"
                "那就是拿话术躲问题，算不通过。"
                "判断依据**只是**【现场】和【人设】里有没有答案 —— "
                "和这句话听起来谦不谦虚无关。"
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

#: 裁判的输出预算。**必须比台词预算大得多，这不是随手写大的。**
#:
#: 实测踩到的坑：拿 `speech_max_tokens`（1024）去跑裁判，校准集里有一条
#: 返回了空内容 —— `finish_reason=length`，因为模型是推理模型，
#: 光思维链就写了 4043 字，预算全被 CoT 吃光，正式回答一个字都没剩下。
#:
#: 这个失败很隐蔽：空内容会被判成"未判"（这是对的，不是 0 分），
#: 于是报告不会报错，只会显示"未判 1 条"。一两条无所谓，
#: 但如果 30% 的判决都因为预算不够而没判成，通过率就建立在少数样本上了。
#: 所以预算要给足，而且要给到能容纳 CoT 的量级。
JUDGE_MAX_TOKENS = 4096


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
        max_tokens: int = JUDGE_MAX_TOKENS,
        name: str = "",
        max_retries: int = JUDGE_MAX_RETRIES,
        backoff: float = JUDGE_BACKOFF,
        sleep: Any = None,
    ) -> None:
        self.llm = llm
        self.rubrics = list(rubrics)
        self.max_tokens = max_tokens
        self.name = name or getattr(llm, "name", "judge")
        self.max_retries = max(0, int(max_retries))
        self.backoff = backoff
        # 注入 sleep 是为了让测试不用真的等 3 秒、6 秒
        self._sleep = sleep or time.sleep
        self.calls = 0
        #: 重试次数。单独计数，因为它回答的是"这次判分有多不稳"——
        #: 和 `calls`（花了多少资源）是两个问题。
        self.retries = 0
        #: 其中因**解析失败**（空内容 / 没有 score 字段）而重试的次数。
        #: 单独拆出来是因为它对应一个具体故障：**思维链把预算吃光**。
        #: 这个数只要不是 0，就说明裁判预算该加或者该换更稳的模型 ——
        #: 合并进 `retries` 就看不出这一点了。
        self.parse_retries = 0

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

        # 两种情况都要重试，**但原因不同，所以要分开计数**：
        #
        #   调用失败（超时/连接断开）—— 显然是瞬时的，早就该重试。
        #   解析失败（空内容 / 没有 score 字段）—— 这个曾经被判为"确定性失败，
        #       重试只是白烧钱"，**实测证明那个判断是错的**：
        #       长跑里出现过一条思维链 **14440 字**，是之前实测最长值（4043）的
        #       3.5 倍，预算被吃光 → 返回空内容。思维链长度在不同调用之间
        #       波动极大，所以"被预算截断"是**随机事件**，不是这句话的属性。
        #       实测代价：跑批里 1/792 条命中，而重试的成本上限是
        #       0.25% × 2 次额外调用 —— 便宜得多。
        #
        # 不重试的只有一种：进来之前就返回的那些（模型不可用 / 没有台词可判）。
        # 它们是真的确定性，重试一万次结果一样。
        last_error = ""
        #: 只要**任何一次**尝试是解析失败，就优先报它。
        #:
        #: 为什么不能直接报最后一次的错误：重试会把原始原因盖掉。
        #: 比如模型第一次返回了"没有 score 字段"（模型答了，但格式不对），
        #: 第二三次恰好撞上网络抖动 —— 最后报出来的是"连接断开"，
        #: 而真正的问题是输出格式。解析失败说明**模型回应了**，
        #: 它比"连不上"更接近真相，所以优先。
        parse_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                raw = self.llm.complete(
                    [{"role": "user", "content": prompt}],
                    temperature=JUDGE_TEMPERATURE,
                    max_tokens=self.max_tokens,
                )
            except LLMUnavailable as exc:
                last_error = f"模型调用失败：{exc}"
                kind = "call"
            except Exception as exc:  # 网络/鉴权/超时都算"没判"，不该让跑批崩掉
                last_error = f"模型调用异常：{type(exc).__name__}: {exc}"
                kind = "call"
            else:
                self.calls += 1
                verdict = self._parse(rubric_key, raw)
                if verdict.judged:
                    return verdict
                # 解析失败：记下原因，走下一轮
                last_error = verdict.error or "解析失败"
                parse_error = last_error
                kind = "parse"
            if attempt < self.max_retries:
                self.retries += 1
                if kind == "parse":
                    self.parse_retries += 1
                self._sleep(self.backoff * (2 ** attempt))
        return Verdict.unjudged(rubric_key, parse_error or last_error)

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
        persona_of: Any = None,
        scene: str = "",
        progress: Any = None,
    ) -> list[dict[str, Any]]:
        """对一串 (玩家说 → NPC 回) 逐条判分。

        `persona_of(speaker)` 优先于 `persona`：多 NPC 场景里每一句台词
        属于不同的人，必须拿**说话人自己的**人设卡去判。
        只有一个 NPC 时调用方传 `persona` 就行。
        """
        out: list[dict[str, Any]] = []
        for index, pair in enumerate(pairs, 1):
            block = persona
            if persona_of is not None:
                block = persona_of(pair.get("speaker", "")) or persona
            verdicts = self.judge_reply(
                persona=block,
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


# --------------------------------------------------------------------------- #
# 留出集的封条
# --------------------------------------------------------------------------- #
def rubric_digest() -> str:
    """把评分标准的**文本**哈希一遍。

    改 rubric 会让留出集失效 —— 因为标签是照着**当时的**标准写的。
    比如把「句数是硬约束」改成「句数是风格建议」，那些原本标 0 的样本
    就不再是"正确答案"了，可它们看上去还在那里，算出来的 kappa 会变成
    一个没人能解释的数。所以这个摘要必须进封条。
    """
    blob = json.dumps(
        [
            [key, RUBRICS[key].name, RUBRICS[key].question,
             RUBRICS[key].pass_when, RUBRICS[key].fail_when]
            for key in sorted(RUBRICS)
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def items_digest(items: Iterable[dict[str, Any]]) -> str:
    """对**解析后的样本**取摘要，而不是对文件字节。

    这样修一个错别字、补一句注释不会把留出集作废 —— 摘要盯的是
    「标签和输入有没有变」，那才是会让 kappa 变味的唯一原因。
    """
    blob = json.dumps(
        [[i.get("id"), i.get("rubric"), i.get("persona"), i.get("scene"),
          i.get("player"), i.get("reply"), i.get("label")] for i in items],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def seal_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    per_rubric: dict[str, dict[str, int]] = {}
    for item in items:
        bucket = per_rubric.setdefault(str(item.get("rubric")), {"n": 0, "pass": 0, "fail": 0})
        bucket["n"] += 1
        bucket["pass" if int(item.get("label", 0)) == 1 else "fail"] += 1
    return {
        "items": len(items),
        "holdout_digest": items_digest(items),
        "rubric_digest": rubric_digest(),
        "label_balance": {
            "pass": sum(1 for i in items if int(i.get("label", 0)) == 1),
            "fail": sum(1 for i in items if int(i.get("label", 0)) == 0),
        },
        "per_rubric": per_rubric,
    }


def build_seal(
    items: list[dict[str, Any]] | None = None,
    *,
    note: str = "",
    created: str = "",
) -> dict[str, Any]:
    """给留出集打封条。`created` 由调用方传（这里不取系统时间，测试才好写）。"""
    items = items if items is not None else load_calibration(HOLDOUT_FILE)
    seal = {
        "protocol": (
            "标签写好时没见过裁判输出，之后不再回头改 rubric。"
            "跑批时若 holdout_digest 或 rubric_digest 对不上，"
            "这份留出集就不再能当作留出结果引用。"
        ),
        "created": created,
        "note": note,
    }
    seal.update(seal_summary(items))
    return seal


def load_seal(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path else HOLDOUT_SEAL_FILE
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def verify_seal(
    seal: dict[str, Any],
    items: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """检查留出集还能不能当留出集用。返回问题列表，空列表 = 通过。

    分成两种**性质不同**的问题，因为补救办法不一样：

      - `rubric_changed`：标签是照旧标准写的，现在标准变了。
        重封条**没用** —— 需要重新标注，或者承认这份只能当开发集。
      - `holdout_changed`：样本本身被改过。
        重封条同样没用 —— 改过就说明可能看过裁判输出再回头调了标签，
        这个可能性一旦存在，数字就不再可信。只能新攒一份。

    换句话说：**封条不是"确认一下"，是"一旦破了就不可修复"。**
    把它做成可以随手重置的按钮，等于没做。
    """
    items = items if items is not None else load_calibration(HOLDOUT_FILE)
    problems: list[dict[str, str]] = []
    if not seal:
        problems.append({
            "kind": "no_seal",
            "detail": "没有封条文件，无法判断这份样本是不是「没见过裁判输出」就写好的。",
        })
        return problems

    if seal.get("rubric_digest") != rubric_digest():
        problems.append({
            "kind": "rubric_changed",
            "detail": (
                f"评分标准变过（封条 {seal.get('rubric_digest')} → 现在 {rubric_digest()}）。"
                "标签是照旧标准写的，这份留出集对当前 rubric 已经不再是留出集，"
                "只能当开发集用；重新封条不能修复它。"
            ),
        })
    if seal.get("holdout_digest") != items_digest(items):
        problems.append({
            "kind": "holdout_changed",
            "detail": (
                f"留出集内容变过（封条 {seal.get('holdout_digest')} → 现在 {items_digest(items)}）。"
                "样本被改过之后，「改标签前有没有看过裁判输出」就无法自证了，"
                "这个数字不能再当留出结果。重新封条也不能修复。"
            ),
        })
    if int(seal.get("items", -1)) != len(items):
        problems.append({
            "kind": "count_changed",
            "detail": f"条数变过（封条 {seal.get('items')} → 现在 {len(items)}）。",
        })
    return problems


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


def run_holdout(
    judge: LLMJudge,
    *,
    items: Optional[list[dict[str, Any]]] = None,
    seal: Optional[dict[str, Any]] = None,
    progress: Any = None,
) -> dict[str, Any]:
    """跑留出集，返回一块可以直接进 payload 的结果。

    **封条破了也照跑。** 不跑就等于把诊断信息也一起扔掉；
    关键是结果必须被标成 `quotable: False`，并且把破在哪里原样带上 ——
    这样数字不会丢，也不可能被当成"泛化能力"引用。
    """
    items = items if items is not None else load_calibration(HOLDOUT_FILE)
    seal = seal if seal is not None else load_seal()
    problems = verify_seal(seal, items)
    report = calibrate(judge, items, progress=progress)
    block: dict[str, Any] = report.to_dict()
    block["quotable"] = not problems
    block["problems"] = problems
    block["seal"] = seal_summary(items)
    block["sealed_rubric_digest"] = seal.get("rubric_digest", "")
    return block


def contrast_rows(
    dev: CalibrationReport | dict[str, Any], holdout: dict[str, Any]
) -> list[dict[str, Any]]:
    """把「开发集」和「留出集」的 kappa 摆在一起。

    两个数的差就是**拟合的量**。只有一个数的时候，你没法判断它有多少是
    真本事 —— 这正是要另起一份留出集的原因。

    `dev` 既收 `CalibrationReport` 也收它的 `to_dict()` 形态：
    CLI 手里是前者，HTML 报告手里是后者，两边都该用同一份算法，
    否则两个地方会慢慢算出不一样的差值。
    """
    dev_per = dev.per_rubric if isinstance(dev, CalibrationReport) else (dev.get("per_rubric") or {})
    per_holdout = holdout.get("per_rubric") or {}
    rows: list[dict[str, Any]] = []
    for key in sorted(set(dev_per) | set(per_holdout)):
        d = dev_per.get(key) or {}
        h = per_holdout.get(key) or {}
        dev_kappa = d.get("kappa")
        hold_kappa = h.get("kappa")
        gap = (
            round(float(dev_kappa) - float(hold_kappa), 3)
            if dev_kappa is not None and hold_kappa is not None
            else None
        )
        rows.append({
            "rubric": key,
            "name": RUBRICS[key].name if key in RUBRICS else key,
            "dev_n": d.get("n"),
            "dev_kappa": dev_kappa,
            "holdout_n": h.get("n"),
            "holdout_kappa": hold_kappa,
            "gap": gap,
            "holdout_reading": h.get("reading", ""),
            "quotable": bool(holdout.get("quotable")),
        })
    return rows


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


# --------------------------------------------------------------------------- #
# 对一份跑批报告逐条判分（并发）
# --------------------------------------------------------------------------- #
#: 判分默认并发。和跑批一样保守 —— 端点是同一个。
DEFAULT_JUDGE_CONCURRENCY = 4


@dataclass
class CaseJudgement:
    """一条用例的判分结果。"""

    index: int
    case_id: str
    category: str = ""
    scenario: str = ""
    pairs: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def reusable(self) -> bool:
        """能不能在 `--resume` 时复用这条判分结果。

        **不能只看 `ok`。** 判分失败走的是 `Verdict.unjudged` —— 它
        **不抛异常**，所以 `error` 是空字符串，`ok` 为真，但整条用例
        一条判决都没拿到分数。把这种结果当成"判完了"复用，等于把
        "裁判当时连不上"永久写进报告：下次 `--resume` 会跳过它，
        报告里那几条永远缺判决，而没有任何地方提示要去重跑。

        所以规则是：没有对话可判（确定性结果）可以复用；
        有对话但**一条判决都没拿到**，必须重判。
        """
        if self.error:
            return False
        if not self.pairs:
            # 本来就没有对话可判 —— 这是确定性的，复用不会丢信息。
            return True
        return any(v.get("judged") for v in self.verdicts())

    def verdicts(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for entry in self.pairs:
            out.extend(entry.get("verdicts") or [])
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "scenario": self.scenario,
            "error": self.error,
            "pairs": self.pairs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaseJudgement":
        """从检查点还原一条判分结果。

        `index` 由调用方按下标重新赋值 —— 和跑批那边一样，
        用例集可能被 `--limit-cases` 改过，恢复出来的报告顺序
        必须跟着当前的用例列表走，而不是跟着上次的。
        """
        return cls(
            index=0,
            case_id=str(data.get("case_id") or ""),
            category=str(data.get("category") or ""),
            scenario=str(data.get("scenario") or ""),
            pairs=list(data.get("pairs") or []),
            error=str(data.get("error") or ""),
        )


def report_digest(results: list[dict[str, Any]]) -> str:
    """把"被判的是什么内容"压成一个摘要。

    为什么需要它：检查点是按 `case_id` 复用的，而**同一个 `case_id`
    在不同批次里的台词是不一样的**（换模型、换预算、换世界都会变）。
    拿 A 批的判决去补 B 批，报告会显示"这条判过了"，实际判的是 A 批的台词 ——
    这比没判更糟，因为它看起来有数据。

    配置指纹管的是"裁判怎么判"，摘要管的是"判的是哪份台词"，两者缺一不可。
    """
    blob = json.dumps(
        [
            [r.get("case_id"), r.get("speeches") or r.get("transcript") or []]
            for r in results
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


#: 判分侧影响结果的配置字段。和跑批那边同理：**并发数不在里面** ——
#: 它只影响判多久，不影响判出什么。
JUDGE_RESUME_CRITICAL_FIELDS = ("judge", "rubrics", "max_tokens", "temperature")


def prompt_digest(context: dict[str, str]) -> str:
    """裁判 prompt 的**上下文**指纹（人设块 + 现场块）。

    ## 为什么必须有它 —— 这条是被真实事故逼出来的

    原来的指纹只覆盖两件事：
      - "裁判怎么判"：模型名 / 标准**名字** / 预算 / 温度
      - "判的是哪份台词"：`report_digest`

    它**不覆盖我们喂给裁判的上下文**。于是出现了一个静默的破坏路径：
    改好现场块（比如补上演员表）之后再 `--resume`，新旧两批判决会被
    拼在一起 —— 而它们是在两套不同的 prompt 下判的，合起来的数字
    没人能解释。更糟的是它**不会报错**，看起来只是"判完了"。

    实测这次真踩了两个上下文 bug（都是我们的，不是模型的）：
      1. 现场块漏了演员表 → 32.5% 的「事实一致」判 0 是把同伴名字当编造；
      2. 人设块只取第一个 NPC → duet 里小舟的「角色口吻」失败率 90%
         （阿柚 39%），因为它拿的是阿柚的卡。
    两个 bug 都要求重判，所以指纹必须能识别"上下文变了"。

    `rubrics` 字段只记标准的**名字**，标准**文本**改了它看不出来 ——
    所以这里把 `rubric_digest()` 也一并算进来。
    """
    blob = json.dumps(
        {"rubric_text": rubric_digest(), "context": context},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def judge_fingerprint(
    judge: "LLMJudge",
    results: list[dict[str, Any]],
    *,
    prompt_context: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    return {
        "judge": judge.name,
        "rubrics": sorted(judge.rubrics),
        "max_tokens": judge.max_tokens,
        "temperature": JUDGE_TEMPERATURE,
        "report_digest": report_digest(results),
        # 上下文指纹。没给就留空串 —— 空串和历史检查点里的"没有这个键"
        # 不相等，所以**加了这一项之后旧检查点一律拒绝恢复**，这是对的：
        # 旧判决是在没被记录下来的上下文下判的。
        "prompt_digest": prompt_digest(prompt_context or {}),
    }


def plan_judge_resume(
    payload: dict[str, Any], fingerprint: dict[str, Any]
) -> tuple[dict[str, CaseJudgement], str]:
    """从判分检查点里挑出可以复用的结果。

    返回 `(可复用结果 by case_id, 一句人话说明)`。
    指纹对不上就返回空 + 原因 —— **不静默复用**。
    """
    entries = payload.get("judgements") or []
    if not entries:
        return {}, "判分检查点里没有已完成的结果"

    recorded = payload.get("fingerprint")
    if isinstance(recorded, dict) and recorded != fingerprint:
        # 逐字段解释差在哪。**取并集而不是硬编码字段表** —— 硬编码的表
        # 在新增字段时会漏报，于是出现"拒绝恢复，但后面什么都不列"的消息，
        # 而一个说不出理由的护栏最后一定会被人绕过。
        keys = sorted(set(recorded) | set(fingerprint))
        diff = [
            f"{k}: {recorded.get(k)!r} → {fingerprint.get(k)!r}"
            for k in keys
            if recorded.get(k) != fingerprint.get(k)
        ]
        detail = "；".join(diff) or "（指纹整体不等，但没有单字段差异 —— 请检查指纹结构本身）"
        return {}, (
            "判分检查点记录的是另一套配置（或另一份台词 / 另一套上下文），拒绝恢复"
            "（否则报告会把两次判分混成一列）：" + detail
        )

    usable: dict[str, CaseJudgement] = {}
    skipped = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        judgement = CaseJudgement.from_dict(entry)
        if not judgement.case_id:
            continue
        if judgement.reusable:
            usable[judgement.case_id] = judgement
        else:
            skipped += 1
    note = f"判分检查点里有 {len(usable)} 条可复用结果"
    if skipped:
        note += f"（另有 {skipped} 条一条判决都没拿到，会重判）"
    return usable, note


def judge_report_cases(
    results: list[dict[str, Any]],
    *,
    judge: "LLMJudge",
    persona_of: Any,
    scene_of: Any,
    concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
    on_done: Any = None,
    resume: Optional[dict[str, CaseJudgement]] = None,
    checkpoint: Any = None,
) -> tuple[list[CaseJudgement], dict[str, Any]]:
    """对一份跑批报告里的每条用例判分。

    `persona_of(scenario_id)` / `scene_of(scenario_id)` 由调用方提供 ——
    判分模块不该知道场景配置长什么样，否则换一个世界就要改判分逻辑。

    ## 为什么并发

    228 条用例 × 每条约 3.3 轮对话 × 3 条标准 ≈ 2300 次模型调用。
    按实测单次 15~50 秒算，串行是**十几小时**量级。不并行就等于不会有人跑，
    于是"我们用了 LLM-as-judge"就永远停留在声明阶段。

    ## 为什么可以共享一个 judge 实例

    `judge_reply` 是无状态的（不缓存、不累加跨条状态），
    模型客户端每次调用新开连接（`urllib.request.urlopen`），
    不共享可变连接对象。所以并发调用互不干扰。

    结果**按用例下标落位**再返回：判分报告的顺序必须和跑批报告一致，
    否则"哪条用例人设崩了"要人工去对，等于没判。

    单条用例内部仍然串行（一次对话的几轮之间有上下文关系，
    并行判会让同一条用例的判决来自不同的时间点，反而不好归因）。

    ## 为什么也要检查点

    跑批那边早就有了，判分这边一直漏着 —— 而判分**比跑批更长**
    （跑批 227 条约 50 分钟，判分 228 条约 4 小时以上）。
    一个四小时、两千多次调用、没有任何断点续跑的作业，
    崩一次就是全丢。检查点和跑批共用 `runner.Checkpoint`
    （加锁 + 临时文件 + `os.replace` 原子替换）。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(results)
    slots: list[Optional[CaseJudgement]] = [None] * total
    done = 0

    def work(index: int) -> CaseJudgement:
        result = results[index]
        scenario_id = result.get("scenario") or "tutorial"
        case_id = str(result.get("case_id") or f"case_{index}")
        pairs = dialogue_pairs(result.get("transcript") or [])
        if not pairs:
            # 没有对话 = 没东西可判。记成空，不记成错误 ——
            # "这条用例本来就没有台词"和"判分炸了"是两回事。
            return CaseJudgement(
                index=index,
                case_id=case_id,
                category=result.get("category") or "",
                scenario=scenario_id,
            )
        try:
            # **人设要按发言者取，不能整条用例用同一张卡。**
            #
            # 原来这里是 `persona_of(scenario_id)`，一个场景一张卡 —— 而
            # 多 NPC 场景（duet）里两个人共用一张卡，第二个 NPC 的台词
            # 是拿第一个 NPC 的人设去判的。实测代价：
            #   duet 里 小舟 的「角色口吻」失败率 **90%**（36/40），
            #   阿柚 是 39% —— 因为小舟在用阿柚的卡（要「嗯——」、要温和调侃）。
            # 这是**我们的 bug**，不是模型的问题：裁判按它拿到的材料判得没错。
            verdicts = judge.judge_pairs(
                pairs,
                persona_of=lambda speaker: persona_of(scenario_id, speaker),
                scene=scene_of(scenario_id),
            )
        except Exception as exc:  # 兜底：判分炸了不能让整批报告作废
            return CaseJudgement(
                index=index,
                case_id=case_id,
                category=result.get("category") or "",
                scenario=scenario_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        return CaseJudgement(
            index=index,
            case_id=case_id,
            category=result.get("category") or "",
            scenario=scenario_id,
            pairs=verdicts,
        )

    # 先落位可复用的结果，再把剩下的排进待判队列。
    # 下标**重新映射**到当前的用例列表上（`--limit-cases` 会改这个列表）。
    reused = 0
    pending: list[int] = []
    for index, result in enumerate(results):
        case_id = str(result.get("case_id") or f"case_{index}")
        previous = (resume or {}).get(case_id)
        if previous is not None and previous.reusable:
            previous.index = index
            slots[index] = previous
            reused += 1
        else:
            pending.append(index)

    if concurrency <= 1 or total <= 1:
        for index in pending:
            judgement = work(index)
            slots[index] = judgement
            done += 1
            if on_done:
                on_done(judgement, done, len(pending))
            if checkpoint:
                checkpoint.save()
    elif pending:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(work, i): i for i in pending}
            for future in as_completed(futures):
                index = futures[future]
                slots[index] = future.result()
                done += 1
                if on_done:
                    on_done(slots[index], done, len(pending))
                if checkpoint:
                    checkpoint.save()

    judgements = [j for j in slots if j is not None]
    judgements.sort(key=lambda j: j.index)
    stats = judge_coverage(judgements, concurrency=concurrency)
    stats["reused"] = reused
    stats["executed"] = len(pending)
    # 重试次数要报出来：它回答"这次判分有多不稳"，
    # 和 `unjudged`（最后真没判成的）是两个问题。
    stats["judge_retries"] = judge.retries
    # 其中解析失败导致的。这个数不是 0 = 裁判预算被思维链吃穿过，
    # 是"该加预算 / 该换模型"的直接证据，所以单独报。
    stats["judge_parse_retries"] = judge.parse_retries
    return judgements, stats


def judge_coverage(
    judgements: list[CaseJudgement],
    *,
    concurrency: int = 1,
    reused: int = 0,
    executed: Optional[int] = None,
) -> dict[str, Any]:
    """判分的覆盖情况。**"判了几条"和"炸了几条"必须分开报。**

    `reused` / `executed` 也要报出来：一次 `--resume` 只判了 1 条用例
    和一次从头判 228 条，报告上看起来都是"228 条判完了"。
    不写清楚，"这次实际判了多少"就无从判断 —— 而它决定了这次
    到底烧了多少模型调用、有多少判决是这一轮新产生的。
    """
    failed = [j for j in judgements if not j.ok]
    empty = [j for j in judgements if j.ok and not j.pairs]
    judged = sum(len(j.verdicts()) for j in judgements)
    unjudged = sum(
        1 for j in judgements for v in j.verdicts() if not v.get("judged")
    )
    return {
        "cases": len(judgements),
        "cases_failed": len(failed),
        "cases_without_dialogue": len(empty),
        "failed_ids": [j.case_id for j in failed],
        "verdicts": judged,
        "unjudged": unjudged,
        "concurrency": concurrency,
        "reused": reused,
        "executed": len(judgements) - reused if executed is None else executed,
        "verdict": _coverage_verdict(len(judgements), len(failed), judged, unjudged),
    }


def _coverage_verdict(total: int, failed: int, verdicts: int, unjudged: int) -> str:
    if not total:
        return "没有用例可判"
    if failed:
        return (
            f"有 {failed}/{total} 条用例判分时炸了（不是「判了 0 分」）。"
            "这些用例的台词没有被评估，报告里的通过率是**剩下的那些**算出来的"
        )
    if not verdicts:
        return "一条台词都没判到 —— 检查转写是否为空"
    if unjudged / verdicts > 0.10:
        return (
            f"有 {unjudged}/{verdicts} 条判决没拿到分数（模型超时/输出不合格式）。"
            "占比超过 10%，通过率建立在剩下的少数判决上，读的时候要留意"
        )
    if unjudged:
        return (
            f"有 {unjudged}/{verdicts} 条判决没拿到分数。"
            "它们没有变成 0 分混进均值，但样本变少了"
        )
    return f"全部 {verdicts} 条判决都拿到了分数"
