"""探针：给裁判补上对话历史，**到底改不改判决、改得对不对**。

## 为什么不能拿校准集来量这件事

`calibrate()` 调的是 `judge.judge(...)`，**不传 `history`** ——
校准集里每一条也只有 `player` 一句话、没有历史。所以：

| | turns=0 | turns=4 |
|---|---|---|
| 校准集 56 条的 prompt | 一样 | **一样** |

**校准集检测不到这次改动**（这一条有护栏：`tests/test_judge.py::
test_the_calibration_set_cannot_see_dialogue_history`）。
好消息是已经发布的 kappa 数字不受影响；坏消息是**它证明不了这次改动有用**。
所以必须另做探针 —— 就是这个脚本。

## 缺陷是什么（这一段不需要模型，是确定性的）

裁判原来只看【玩家刚说】+【NPC 的台词】。同一句回复、同一句玩家话，
**前面发生过什么完全不影响裁判拿到的 prompt** ⇒ 两段不同的历史在它眼里
逐字节相同 ⇒ 它**按拿到的材料判得完全正确，然后判错正确的那一次**。

`grounded`（事实一致）的 `fail_when` 里明写着两条**只能靠历史发现**的错：

- 「**声称自己做过没做过的事**」
- 「**把玩家没说过的话说成玩家说过**」

不带历史时，裁判**没有任何办法**发现这两条 —— 判据在它看不见的地方。

## 这个探针怎么设计（关键：**两个方向都要测**）

只测"带上历史之后分数涨了"是不够的 —— 那可能只是"给了更多上下文 ⇒ 更宽松"。
所以四个手写用例分两个方向：

| 用例 | 期望 | 不带历史时会发生什么 |
|---|---|---|
| H1 自相矛盾（关门时间） | **不通过** | 只看当前这一对，看不出冲突 ⇒ **误判成通过** |
| H2 自相矛盾（自己的口味） | **不通过** | 同上 ⇒ **误判成通过** |
| H3 指代（"那现在呢？"） | **通过** | 缺少指代对象 ⇒ **误判成答非所问** |
| H4 指代（"那他到了吗？"） | **通过** | 同上 ⇒ **误判成答非所问** |

两个方向都动，才说明这是"补上了判据"，而不是"分数整体被抬高了"。

## ⚠️ 写用例时最容易毁掉实验的一件事

**别把判据写进 `persona` / `scene`。** 比如 H1 里如果 `scene` 写了
"店晚上十点关门"，那么**不带历史也能判对** —— 实验当场失效，
而且失效得很安静（结果会显示"两臂都对"，看起来像"改动没用"）。
判据只能活在**对话历史**里，这正是要被测的东西。

用法：
    python -u scripts/probe_judge_history.py
    PROBE_TURNS=0,4 PROBE_TRANSCRIPTS=2 python -u scripts/probe_judge_history.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from npc_agent.eval.judge import (  # noqa: E402
    JUDGE_HISTORY_TURNS,
    LLMJudge,
    dialogue_pairs,
)
from npc_agent.llm import build_llm  # noqa: E402

#: 两臂：旧行为（0）和默认值（4）。
#:
#: ## ⚠️ 允许写重复值 —— 这是**测噪声底**的办法
#:
#: `PROBE_TURNS=4,4` 会让两"臂"变成**同一种配置**。于是脚本报出来的
#: "翻转率"就是**同一个 prompt 跑两次的差异** = 模型自身的不确定性。
#: 这个数必须先量出来，否则真实转写上的翻转率根本没法解释：
#: 一个 16.7% 的翻转率，如果噪声底也是 16.7%，那它什么都没说明。
TURNS = [int(x) for x in (os.environ.get("PROBE_TURNS") or f"0,{JUDGE_HISTORY_TURNS}").split(",")]

#: 从真实跑批报告里再取几条用例，量"判决翻转率"。
N_TRANSCRIPTS = int(os.environ.get("PROBE_TRANSCRIPTS") or "2")

#: 并发数。`judge_pairs()` 内部是串行的，所以并发放在**用例 × 臂**这一层。
#: 串行跑这个探针是 30 分钟量级，并发 6 之后是几分钟。
CONCURRENCY = int(os.environ.get("PROBE_CONCURRENCY") or "6")

REPORT = ROOT / "reports" / "eval_model.json"

# --------------------------------------------------------------------------- #
# 人设与现场
#
# ⚠️ 刻意**不写**任何会在下面用例里成为判据的事实（关门时间、她的口味……）。
#    写进去就等于把答案提前给了裁判，两臂都会判对，实验失效且看不出来。
# --------------------------------------------------------------------------- #
PERSONA = "你是阿柚，星屿咖啡屋的店主。说话简短、口语化，每次最多两句话。"
SCENE = "星屿咖啡屋，下午，吧台后面。"


def _case(
    case_id: str,
    rubric: str,
    history: list[tuple[str, str]],
    final: tuple[str, str],
    label: int,
    why: str,
) -> dict:
    pairs = [
        {"speaker": "阿柚", "player": p, "reply": r} for p, r in history
    ] + [{"speaker": "阿柚", "player": final[0], "reply": final[1]}]
    return {
        "id": case_id,
        "rubric": rubric,
        "pairs": pairs,
        "label": label,
        "why": why,
    }


#: 期望 `label`：1 = 应当通过，0 = 应当不通过。
CASES = [
    _case(
        "H1-自相矛盾-关门时间",
        "grounded",
        [("你们几点关门？", "晚上十点。")],
        ("那我十点半来行吗？", "行，十点半我还在。"),
        0,
        "她两轮前说十点关门，这里说十点半还在。冲突的**另一半只存在于历史里** —— "
        "不带历史时，玩家问十点半、她说十点半还在，读起来完全自洽。",
    ),
    _case(
        "H2-自相矛盾-自己的口味",
        "grounded",
        [("你平时自己喝什么？", "我只喝意式，喝不惯手冲。")],
        ("那给我推荐一杯。", "我推荐手冲，我每天都喝手冲。"),
        0,
        "她刚说自己只喝意式，这里说每天都喝手冲。不带历史时看不到前一句。",
    ),
    _case(
        "H3-指代-那现在呢",
        "responsive",
        [("今天有手冲吗？", "没有，豆子用完了。")],
        ("那现在呢？", "现在有了，我朋友刚送来一包。"),
        1,
        "「那现在呢？」的指代对象（有没有手冲）只在历史里；"
        "不带历史时「现在有了」无所指，容易被判成答非所问。",
    ),
    _case(
        "H4-指代-那他到了吗",
        "responsive",
        [("老周今天来吗？", "他一般下午到。")],
        ("那他到了吗？", "到了，在后厨磨豆子呢。"),
        1,
        "「他」是谁只在历史里。不带历史时代词悬空。",
    ),
]


def _score_of(verdicts: list[dict], index: int, rubric: str) -> int | None:
    """取第 `index` 对里**指定标准**的判决：1 = 通过，0 = 不通过，None = 未判。

    ## ⚠️ 必须按**标准名**取，不能按下标取

    第一版写成 `verdicts[index]["verdicts"][0]`，也就是"永远取第一个标准"。
    但 `judge_pairs` 是按 `rubrics` 的顺序返回的 —— 这里传的是
    `["grounded", "responsive"]`，于是**所有用例都被拿「事实一致」判了**，
    包括两条本来为「是否回应」写的用例。

    这个错**不会报错**：四个用例照样各出一个分数、表格照样排整齐，
    只是 H3/H4 的数字根本不是它们要测的那个标准。
    看出来是因为我把**判决理由**一起打了出来 —— 理由里写着
    「编造现场不存在的地点」，那是「事实一致」的话术。
    **只看数字是看不出来的。**
    """
    try:
        for v in verdicts[index]["verdicts"]:
            if v.get("rubric") != rubric:
                continue
            if not v.get("judged"):
                return None
            return 1 if float(v.get("score") or 0) >= 1.0 else 0
    except (IndexError, KeyError):
        return None
    return None


def _judge_case(judge: LLMJudge, case: dict, turns: int) -> tuple[int | None, str]:
    """判**最后一对**（也就是设计好的那一对），带上 `turns` 轮历史。"""
    out = judge.judge_pairs(
        case["pairs"],
        persona=PERSONA,
        scene=SCENE,
        history_turns=turns,
    )
    last = len(case["pairs"]) - 1
    score = _score_of(out, last, case["rubric"])
    reason = ""
    try:
        for v in out[last]["verdicts"]:
            if v.get("rubric") == case["rubric"]:
                reason = v.get("reason") or ""
    except (IndexError, KeyError):
        pass
    return score, reason


def _run_tasks(tasks: list, workers: int) -> list:
    """并发跑一批无参调用，**按提交顺序**返回结果。

    `judge.judge_pairs()` 是无状态的（不缓存、不累加跨条状态，每次调用新开连接），
    所以并发互不干扰 —— 论证同 `judge.py::judge_report_cases`。
    ⚠️ 必须 `ex.map`（保序），不能 `as_completed` —— 后者按完成顺序给结果，
    两臂的结果会错位配对，而错位**不会报错**，只会给出一个看起来正常的错误结论。
    """
    if workers <= 1:
        return [task() for task in tasks]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda fn: fn(), tasks))


def run_hand_cases(judge: LLMJudge, workers: int = CONCURRENCY) -> dict:
    print("=" * 78)
    print("第一部分：四个手写用例（判据只存在于对话历史里）")
    print("=" * 78)
    tasks = []
    keys: list[tuple[str, int]] = []
    for case in CASES:
        for pos, turns in enumerate(TURNS):
            # ⚠️ 键用**位置**（`pos`）而不是 `turns`：`PROBE_TURNS=4,4` 时
            # 两臂的 turns 相同，用 turns 当键会让后者覆盖前者，
            # 于是"噪声底"被算成 0 —— 一个漂亮但完全虚假的 0。
            keys.append((case["id"], pos))
            tasks.append(lambda c=case, t=turns: _judge_case(judge, c, t))
    results = _run_tasks(tasks, workers)
    by_key = dict(zip(keys, results))

    table: dict[str, dict[int, int | None]] = {}
    for case in CASES:
        print(f"\n--- {case['id']}（标准：{case['rubric']}；期望："
              f"{'通过' if case['label'] else '不通过'}）---")
        print(f"  为什么它能区分两臂：{case['why']}")
        table[case["id"]] = {}
        for pos, turns in enumerate(TURNS):
            score, reason = by_key[(case["id"], pos)]
            table[case["id"]][pos] = score
            got = "未判" if score is None else ("通过" if score else "不通过")
            ok = "✅" if score == case["label"] else "❌"
            print(f"  第{pos + 1}臂(turns={turns}): {got} {ok}  理由：{reason[:110]}")

    print("\n" + "-" * 78)
    print(f"{'用例':<26}{'期望':<8}" + "".join(
        f"{'臂' + str(p + 1) + '(t=' + str(t) + ')':<16}"
        for p, t in enumerate(TURNS)
    ))
    for case in CASES:
        want = "通过" if case["label"] else "不通过"
        cells = []
        for pos, _turns in enumerate(TURNS):
            s = table[case["id"]][pos]
            mark = "未判" if s is None else ("通过" if s else "不通过")
            cells.append(f"{mark}{'✅' if s == case['label'] else '❌'}")
        print(f"{case['id']:<26}{want:<8}" + "".join(f"{c:<16}" for c in cells))

    print("\n按臂统计判对数：")
    for pos, turns in enumerate(TURNS):
        correct = sum(
            1 for case in CASES if table[case["id"]][pos] == case["label"]
        )
        print(f"  第{pos + 1}臂(turns={turns}): {correct}/{len(CASES)}")
    return table


def run_transcripts(judge: LLMJudge, workers: int = CONCURRENCY) -> dict:
    """真实转写上的**翻转率**：同一条用例，两臂判出来一样吗。"""
    if not REPORT.exists():
        print(f"\n[跳过] 找不到 {REPORT.name}，不做真实转写那部分")
        return {}
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    # ⚠️ 跑批报告里装用例的键是 **`results`**（不是 `cases`），
    #    转写字段是 **`transcript`**（`speeches` 只有 NPC 那半边，没顺序）。
    #    这两个名字我第一版都猜错了 —— 猜错的后果是"扫到 0 条、静默跳过"，
    #    看起来像"报告里没有转写"，而不是像"我读错键了"。
    cases = [c for c in payload.get("results") or [] if c.get("transcript")]
    cases = cases[:N_TRANSCRIPTS]
    print("\n" + "=" * 78)
    print(f"第二部分：真实转写上的翻转率（{len(cases)} 条用例，标准 {judge.rubrics}）")
    print("=" * 78)
    tasks = []
    keys: list[tuple[int, int]] = []

    def _arm(pairs: list[dict[str, str]], turns: int) -> dict[str, list[int | None]]:
        """判完一整条用例，**按标准分开**返回每一对的判决（`None` = 未判）。

        ⚠️ `judge_pairs(...)` 必须**只调一次**。把它写进列表推导式里
        （`[_score_of(judge.judge_pairs(...), k) for k in range(len(p))]`）
        会让它按对数重复调用 —— 实测把模型调用数悄悄放大成 5 倍，
        而判决结果完全一样，**只有调用次数看得出来**。

        ⚠️ 标准要**分开统计**：把两条标准混在一个翻转率里，
        就等于说"改动让判决动了 12%"，但没人知道动的是哪一条 ——
        而归因正是这个探针存在的理由。
        """
        out = judge.judge_pairs(
            pairs, persona=PERSONA, scene=SCENE, history_turns=turns
        )
        return {
            r: [_score_of(out, k, r) for k in range(len(out))] for r in judge.rubrics
        }

    for i, case in enumerate(cases):
        pairs = dialogue_pairs(case["transcript"])
        for pos, turns in enumerate(TURNS):
            # 键用位置，理由同 `run_hand_cases`：`PROBE_TURNS=4,4` 时
            # turns 会重复，用 turns 当键会把噪声底算成 0。
            keys.append((i, pos))
            tasks.append(lambda p=pairs, t=turns: _arm(p, t))
    results = _run_tasks(tasks, workers)
    by_key = dict(zip(keys, results))

    stats = {
        "pairs": 0,
        "flips": 0,
        "unjudged": 0,
        "by_rubric": {
            r: {"pairs": 0, "flips": 0, "unjudged": 0} for r in judge.rubrics
        },
    }
    for i, case in enumerate(cases):
        pairs = dialogue_pairs(case["transcript"])
        if len(pairs) < 2:
            continue
        first, last = by_key[(i, 0)], by_key[(i, len(TURNS) - 1)]
        for rubric in judge.rubrics:
            flips = comparable = unjudged = 0
            for k in range(len(pairs)):
                a, b = first[rubric][k], last[rubric][k]
                if a is None or b is None:
                    unjudged += 1
                    continue
                comparable += 1
                if a != b:
                    flips += 1
            stats["pairs"] += comparable
            stats["flips"] += flips
            stats["unjudged"] += unjudged
            bucket = stats["by_rubric"][rubric]
            bucket["pairs"] += comparable
            bucket["flips"] += flips
            bucket["unjudged"] += unjudged
            print(f"  {case.get('case_id')} [{rubric}]: {comparable} 对可比，"
                  f"{flips} 对翻转，未判 {unjudged}")
            print(f"      第1臂(turns={TURNS[0]}): {first[rubric]}")
            print(f"      末臂(turns={TURNS[-1]}): {last[rubric]}")
    if stats["pairs"]:
        print("\n合计（按标准分开）：")
        for rubric, bucket in stats["by_rubric"].items():
            rate = bucket["flips"] / bucket["pairs"] if bucket["pairs"] else 0.0
            print(f"  {rubric}: {bucket['pairs']} 对可比，{bucket['flips']} 对翻转 "
                  f"= {rate:.1%}，未判 {bucket['unjudged']}")
        total_rate = stats["flips"] / stats["pairs"]
        print(f"  合计：{stats['pairs']} 对可比，{stats['flips']} 对翻转 "
              f"= {total_rate:.1%}；未判 {stats['unjudged']} 对")
    return stats


def main() -> int:
    model = os.environ.get("NPC_AGENT_MODEL") or "kimi-k2.7-code"
    print(f"模型：{model}｜两臂：turns={TURNS}｜默认值 JUDGE_HISTORY_TURNS={JUDGE_HISTORY_TURNS}")
    llm = build_llm(
        os.environ.get("NPC_AGENT_PROVIDER") or "openai-compat",
        model=model,
        base_url=os.environ.get("NPC_AGENT_BASE_URL") or "",
        api_key=os.environ.get("NPC_AGENT_API_KEY") or "",
        timeout=180.0,
        retries=3,
    )
    if not llm.available:
        print("模型不可用 —— 探针不做任何推断。")
        return 1

    # 一次只放一个标准，判决才归因得清。
    judge = LLMJudge(llm, rubrics=["grounded", "responsive"], max_tokens=8192)
    run_hand_cases(judge, CONCURRENCY)
    run_transcripts(judge, CONCURRENCY)

    print("\n" + "=" * 78)
    print("这个探针能说明什么 / 不能说明什么")
    print("=" * 78)
    if len(set(TURNS)) < len(TURNS):
        print(f"⚠️ 两臂是**同一种配置**（turns={TURNS}）—— 这是一次**噪声底**测量：")
        print("   报出来的翻转率就是同一个 prompt 跑两次的差异。")
        print("   拿它和两臂不同的翻转率比，才知道真实效应有没有超出噪声。")
    print("能：在判据只存在于对话历史里的用例上，两臂判决是否不同、方向对不对。")
    print("不能：它**不**证明整体判分质量变好了 —— 4 个手写用例不是统计样本；")
    print("      翻转率也只说明「改了多少」，不说明「改对多少」。")
    print("      要把「改对多少」说成结论，得给真实转写做人工标注。")
    print("⚠️ 还有一个更基础的坑：**下标 0 那一对两臂的 prompt 逐字节相同**")
    print("   （没有历史可带）。所以下标 0 上的翻转**不可能**是历史造成的 ——")
    print("   那是模型自己的不确定性。测噪声底就是在量这件事有多大。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
