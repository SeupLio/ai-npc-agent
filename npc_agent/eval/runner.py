"""并行跑批 —— 把"两小时"变成"十几分钟"，并且**如实报告它变成了多少**。

## 为什么并行不是"加个线程池"就完了

串行跑 213 条用例、每条 10 轮、每轮一次模型调用，就是两千多次调用。
按实测单次 14 秒算，是**八小时**量级（用户给的估算是两小时，同一量级）。
不并行，这个评测就跑不动，于是也就没人跑，于是"评测"这件事就名存实亡。

但并行会引入四类新问题，每一类都会让结果**看起来正常、其实不可信**：

### 一、顺序错乱

线程池是"谁先跑完谁先回来"。如果按完成顺序往报告里塞结果，
同一份用例集跑两次会得到两份顺序不同的报告，`--compare` 逐条对比直接失效，
diff 里全是无关的错位。所以结果**按用例下标落位**，报告顺序与串行完全一致。

### 二、失败被静默降级（这一类最危险）

`LLMUnavailable` 在框架里是**设计好的回退信号**：模型不可用时各模块走启发式。
这在离线跑批里是对的，但在**模型跑批里是灾难** —— 端点中途挂了，
后半程的台词全部退回模板，而报告只会显示"通过率 91%"。

所以每个 case 外面套一层计数器：这次用例期间模型失败了几次。
失败过的用例标记为 `degraded`，单独统计，并且明确写出
"这些用例的分数不代表模型能力"。同时**重跑该用例**（指数退避）。

### 三、检查点被写坏

"边跑边落盘"要防的不只是进程被杀，还有"被杀在写文件的中途" ——
那会留下一个截断的 JSON，下次启动直接解析失败，比没写还糟。
所以写入走"临时文件 + 原子替换"。

### 四、并发数被当成免费午餐

端点是会限流的。并发调太高，`429` 变多，重试变多，
总耗时反而更长 —— 而且失败率上升会让 `degraded` 比例升高。
所以并发数是**可配置**的，而且报告里要给出实测加速比，
让"并行到底有没有用"变成一个数字，而不是一句声明。
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..config import RuntimeConfig
from ..llm.base import LLM, LLMUnavailable
from .harness import CaseResult, EvalHarness

#: 默认并发。刻意保守：端点通常有并发上限，调太高会变成重试风暴。
DEFAULT_CONCURRENCY = 4

#: 默认重试次数（不含首次）。
DEFAULT_MAX_RETRIES = 2

#: 退避基数（秒）。第 n 次重试等 backoff * 2^(n-1)。
DEFAULT_BACKOFF = 3.0


# --------------------------------------------------------------------------- #
# 给模型套一层计数器
# --------------------------------------------------------------------------- #
class MeteredLLM(LLM):
    """包装一个真实模型，记录"调了几次、失败了几次"。

    存在的唯一理由：`LLMUnavailable` 会被框架吞掉并回退到模板，
    于是"端点挂了"在报告里长得和"模型答得不错"一模一样。
    没有这层计数，模型跑批的正确性就没法验证。
    """

    def __init__(self, inner: LLM, label: str = "") -> None:
        self.inner = inner
        self.label = label or getattr(inner, "name", "llm")
        self.calls = 0
        self.failures = 0
        self.last_error = ""

    @property
    def name(self) -> str:  # type: ignore[override]
        return self.label

    @property
    def available(self) -> bool:
        return bool(getattr(self.inner, "available", False))

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        self.calls += 1
        try:
            return self.inner.complete(
                messages, temperature=temperature, max_tokens=max_tokens
            )
        except LLMUnavailable as exc:
            # 只记 LLMUnavailable：那是"传输/端点"这一类的失败。
            # 其他异常是代码 bug，应该立刻炸出来，不该被算成"模型抖动"。
            self.failures += 1
            self.last_error = str(exc)
            raise


def metered_factory(inner_factory: Callable[..., LLM], label: str = "") -> Callable[[], LLM]:
    """构造"每条用例一个新的计数器 + 新的模型客户端"的工厂。

    每条用例一份，是为了让计数天然线程安全 —— 不需要锁，
    也不会把 A 用例的失败算到 B 用例头上。
    """

    def make() -> LLM:
        return MeteredLLM(inner_factory(), label=label)

    return make


# --------------------------------------------------------------------------- #
# 单条用例的结果
# --------------------------------------------------------------------------- #
@dataclass
class CaseRun:
    index: int
    case_id: str
    result: Optional[CaseResult] = None
    attempts: int = 1
    duration: float = 0.0
    error: str = ""
    llm_calls: int = 0
    llm_failures: int = 0
    llm_last_error: str = ""

    @property
    def ok(self) -> bool:
        """用例真的跑完了（不是基础设施故障）。"""
        return self.result is not None

    @property
    def degraded(self) -> bool:
        """模型调用失败过 —— 这条用例可能走了模板回退，分数不代表模型能力。

        单独标记而不是丢掉：丢掉就变成了隐藏信息，
        而且"哪一类用例最容易触发回退"本身就是有用的信号。
        """
        return self.llm_failures > 0

    def to_dict(self, *, include_result: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "case_id": self.case_id,
            "index": self.index,
            "ok": self.ok,
            "degraded": self.degraded,
            "attempts": self.attempts,
            "duration": round(self.duration, 2),
            "llm_calls": self.llm_calls,
            "llm_failures": self.llm_failures,
            "error": self.error,
            "llm_last_error": self.llm_last_error[:200],
        }
        # 只有检查点需要带上完整结果 —— 恢复的时候没有它就只能知道
        # "这条跑过了"，却拿不出分数、转写和失败原因，等于白存。
        if include_result and self.result is not None:
            payload["result"] = self.result.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaseRun":
        """从检查点还原一条结果。"""
        result = None
        if isinstance(data.get("result"), dict):
            try:
                result = CaseResult.from_dict(data["result"])
            except Exception:
                # 结果还原不了就当成"没跑过"，重跑一遍。
                # 宁可多花一次调用的钱，也不要一份读不出来的报告。
                result = None
        return cls(
            index=int(data.get("index") or 0),
            case_id=str(data.get("case_id") or ""),
            result=result,
            attempts=int(data.get("attempts") or 1),
            duration=float(data.get("duration") or 0.0),
            error=str(data.get("error") or ""),
            llm_calls=int(data.get("llm_calls") or 0),
            llm_failures=int(data.get("llm_failures") or 0),
            llm_last_error=str(data.get("llm_last_error") or ""),
        )


@dataclass
class BatchStats:
    """跑批的实测数字。**加速比是量出来的，不是声明出来的。**"""

    wall: float = 0.0
    serial_estimate: float = 0.0
    concurrency: int = 1
    cases: int = 0
    retried: int = 0
    degraded: int = 0
    failed: int = 0
    #: 从检查点恢复、没有重新跑的条数。单独报出来，否则"跑了 228 条"
    #: 和"跑了 5 条 + 恢复了 223 条"看起来是一样的。
    reused: int = 0
    #: 这次**实际执行**的条数（= cases - reused）。
    #: 加速比的分子分母都只跟它有关：复用的那些既没花时间、也没被串行重跑过。
    executed: int = 0
    #: 模型调用总数。**这个数字为 0 时，"通过率 100%" 衡量的不是模型** ——
    #: 所以它必须出现在报告里，而不是留给读者去猜。
    llm_calls: int = 0
    llm_failures: int = 0

    @property
    def speedup(self) -> float:
        """实测加速比 = 串行估计 / 实际墙钟。

        串行估计用**这次实际执行的那些用例**的耗时求和 —— 那是串行跑这一批
        真正要花的时间（重试也计入，因为串行同样要重试）。这个数字可以和
        并发数对照：明显低于并发数，说明瓶颈在端点的排队或限流上，不在本地。

        **一条都没实际执行时返回 0，而不是返回一个巨大的数字。**
        全部复用检查点时墙钟接近 0，`串行估计 / 墙钟` 会算出几千倍这种
        "看起来很厉害、其实什么都没跑"的数。宁可报 0 并说明"没有实际跑批"。
        """
        if self.executed <= 0 or self.wall <= 0:
            return 0.0
        return round(self.serial_estimate / self.wall, 2)

    @property
    def efficiency(self) -> float:
        """并行效率 = 加速比 / 并发数。低于 0.5 说明并发开得太高（或端点限流）。"""
        if self.executed <= 0 or not self.concurrency:
            return 0.0
        return round(self.speedup / self.concurrency, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_sec": round(self.wall, 1),
            "serial_estimate_sec": round(self.serial_estimate, 1),
            "speedup": self.speedup,
            "concurrency": self.concurrency,
            "efficiency": self.efficiency,
            "cases": self.cases,
            "reused_cases": self.reused,
            "executed_cases": self.executed,
            "retried_cases": self.retried,
            "degraded_cases": self.degraded,
            "failed_cases": self.failed,
            "llm_calls": self.llm_calls,
            "llm_failures": self.llm_failures,
        }


# --------------------------------------------------------------------------- #
# 检查点
# --------------------------------------------------------------------------- #
class Checkpoint:
    """线程安全的检查点写入。

    两件事：
      1. **加锁**：并发下多个线程同时落盘会互相覆盖。
      2. **原子替换**：写临时文件再 `os.replace`。
         "边跑边落盘"要防的不只是进程被杀，还有"被杀在写文件的中途" ——
         那会留下一个截断的 JSON，下次启动解析失败，比没写还糟。
    """

    def __init__(self, path: str | Path, build_payload: Callable[[], dict[str, Any]]) -> None:
        self.path = Path(path)
        self._build = build_payload
        self._lock = threading.Lock()
        self.writes = 0

    def save(self) -> None:
        with self._lock:
            payload = self._build()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(tmp, self.path)
            self.writes += 1


# --------------------------------------------------------------------------- #
# 中断后恢复
# --------------------------------------------------------------------------- #
#: 检查点里要记下的"影响结果的配置"。
#:
#: 恢复时必须逐项对上，否则拒绝 —— 把 `--no-planner` 跑出来的半批和
#: planner-on 跑出来的另半批拼起来，报告会把两个不同的变量混成一列，
#: 而且从数字上完全看不出来。宁可重跑，也不要一份悄悄混了两个配置的报告。
#: 指纹里必须包含**所有会改变模型行为的配置**。
#:
#: 漏掉一个的后果是静默的：恢复会把两批不同配置的结果拼成一份报告，
#: 而报告里只会写"228 条"，不会写"其中 150 条跑在 1024 的台词预算下"。
#:
#: 这条清单是被一次真实事故补全的：台词预算从 1024 提到 4096 之后，
#: 如果指纹里没有 `speech_max_tokens`，`--resume` 就会拿旧的
#: 1024 结果去补新配置的批 —— 而那 7 条返回空内容、退回模板的用例
#: 会被当成"已经跑完了"。
RESUME_CRITICAL_FIELDS = (
    "llm_provider",
    "model",
    "base_url",
    "temperature",
    "max_tokens",
    "speech_max_tokens",
    # 超时也改结果：等 60s 和等 180s，回落率完全不同（见 `llm_timeout` 的注释）。
    # 不进指纹的话，`--resume` 会把"60s 跑出来的半批"和"180s 跑出来的另半批"
    # 拼成一份报告，而报告只会写"跑了 N 条"。
    "llm_timeout",
    # 重试次数同理：重试 3 次和重试 1 次的**回落率不同**，
    # 拼在一起那份报告就说不清自己是用哪种配置跑出来的。
    "llm_retries",
    "memory_strategy",
    "memory_top_k",
    "memory_consolidate_at",
    "memory_half_life",
    "use_llm_planner",
    "use_llm_speech",
)


def config_fingerprint(config: RuntimeConfig) -> dict[str, Any]:
    """把影响结果的配置摘出来，用于恢复时比对。

    刻意**不含**并发数、重试次数、退避基数：它们影响"跑多久"和"跑不跑得完"，
    不影响"跑出什么"。把并发数也放进去会导致"上次并发 4、这次想开 6"
    就必须整批重跑，那是把工程参数和实验变量混为一谈。
    """
    return {field: getattr(config, field, None) for field in RESUME_CRITICAL_FIELDS}


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """读检查点。**读不到/读坏了都返回空 dict，不抛异常。**

    检查点的职责是"尽量救回一点"，不是"必须存在"。一个截断的检查点
    如果让整条命令直接崩掉，那它就从保险变成了新的故障源。
    """
    target = Path(path)
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def plan_resume(
    payload: dict[str, Any], config: RuntimeConfig
) -> tuple[dict[str, CaseRun], str]:
    """从检查点里挑出可以复用的结果。

    返回 `(可复用结果 by case_id, 一句人话说明)`。
    配置指纹对不上就返回空 + 说明原因 —— **不静默复用**。
    """
    runs = payload.get("runs") or []
    if not runs:
        return {}, "检查点里没有已完成的结果"

    recorded = payload.get("config")
    current = config_fingerprint(config)
    if isinstance(recorded, dict) and recorded != current:
        diff = [
            f"{k}: {recorded.get(k)!r} → {current.get(k)!r}"
            for k in RESUME_CRITICAL_FIELDS
            if recorded.get(k) != current.get(k)
        ]
        return {}, (
            "检查点记录的是另一套配置，拒绝恢复（否则报告会把两个配置混成一列）："
            + "；".join(diff)
        )

    usable: dict[str, CaseRun] = {}
    for item in runs:
        if not isinstance(item, dict):
            continue
        run = CaseRun.from_dict(item)
        # 只复用**真的跑完了**的。基础设施故障的用例必须重跑 ——
        # 它失败的原因是当时的环境，不是模型，换一次跑很可能就过了。
        if run.case_id and run.ok:
            usable[run.case_id] = run
    return usable, f"检查点里有 {len(usable)} 条可复用结果"


# --------------------------------------------------------------------------- #
# 并行执行
# --------------------------------------------------------------------------- #
def run_cases(
    cases: list[dict[str, Any]],
    config: RuntimeConfig,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    llm_factory: Optional[Callable[[], LLM]] = None,
    cases_dir: Optional[Path] = None,
    on_done: Optional[Callable[[CaseRun, int, int], None]] = None,
    checkpoint: Optional[Checkpoint] = None,
    resume: Optional[dict[str, CaseRun]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[CaseRun], BatchStats]:
    """跑一批用例。

    返回 `(runs, stats)`，其中 `runs` **按用例下标排序** ——
    与串行结果逐条对齐，`--compare` 才有意义。

    并发 = 1 时走串行路径：不建线程池，行为与改造前完全一致。
    保留这条路径不是为了"兼容"，而是为了让并行结果**可以被串行验证** ——
    没有串行基线，就没法证明并行没有改变结果。

    `resume` 给出"已经跑过的结果"，这些用例不再重跑，直接按下标落位。
    下标会**重新映射**（用例集可能被 `--limit` 或类别筛选改过），
    否则恢复出来的报告顺序会和用例集对不上。
    """
    total = len(cases)
    slots: list[Optional[CaseRun]] = [None] * total
    lock = threading.Lock()
    done = 0
    started = time.time()

    # 先落位可复用的结果，再把剩下的排进待跑队列
    reused = 0
    pending: list[int] = []
    for index, case in enumerate(cases):
        case_id = str(case.get("id", f"case_{index}"))
        previous = (resume or {}).get(case_id)
        if previous is not None and previous.ok:
            previous.index = index
            slots[index] = previous
            reused += 1
        else:
            pending.append(index)

    def work(index: int) -> CaseRun:
        case = cases[index]
        case_id = str(case.get("id", f"case_{index}"))
        best = CaseRun(index=index, case_id=case_id)
        # 累加器：重试花掉的时间必须计入串行估计，否则加速比是虚报的
        # （串行跑一遍同样要重试，把重试时间排除在外等于假装重试不要钱）。
        spent = 0.0
        calls = 0
        for attempt in range(1, max_retries + 2):
            run = _run_one(case, index, case_id, config, llm_factory, cases_dir)
            spent += run.duration
            calls += run.llm_calls
            run.attempts = attempt
            run.duration = spent
            run.llm_calls = calls
            # 注意：llm_failures 刻意**只取最后一次尝试**的值。
            # 它回答的问题是"这份结果可不可信"，而重试成功的用例结果就是可信的。
            # 调用次数（calls）才是累计值 —— 它回答的是"这次跑批花了多少资源"。
            best = run
            if run.llm_failures == 0 or not run.ok:
                # 模型没失败 → 结果可信，不用重跑。
                # 基础设施故障（run.ok 为假）→ 重跑整个用例也没用，交给上层报错。
                break
            if attempt <= max_retries:
                # 指数退避：3s、6s、12s。不加抖动是因为并发数很低（≤ 8），
                # 撞车概率小，而抖动会让"同一批用例跑两次"不再逐条可比。
                sleep(backoff * (2 ** (attempt - 1)))
        return best

    if concurrency <= 1:
        for index in pending:
            run = work(index)
            slots[index] = run
            done += 1
            if on_done:
                on_done(run, done, len(pending))
            if checkpoint:
                checkpoint.save()
    elif pending:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(work, i): i for i in pending}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    run = future.result()
                except Exception as exc:  # 兜底：worker 里没接住的异常
                    run = CaseRun(
                        index=index,
                        case_id=str(cases[index].get("id", f"case_{index}")),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                slots[index] = run
                with lock:
                    done += 1
                    if on_done:
                        on_done(run, done, len(pending))
                    if checkpoint:
                        checkpoint.save()

    runs = [r for r in slots if r is not None]
    runs.sort(key=lambda r: r.index)

    # 串行估计只算**这次真的执行了**的用例。复用的那些这次没花时间，
    # 串行跑一遍也不会再花时间 —— 把它们算进去，"加速比"就变成了
    # "复用省下多少"的冒牌货，而不是"并发快了多少"。
    executed_set = set(pending)
    stats = BatchStats(
        wall=time.time() - started,
        # 串行估计 = 每条用例的实际耗时求和（`work()` 里已经把每次重试的耗时累加了）。
        # 重试的**工作**必须计入，否则是虚报加速比 —— 串行跑一遍同样要重试。
        #
        # 但重试之间的**退避睡眠**刻意不计入：两条路径都要等同样长的时间，
        # 把它算进串行、却算进并行的墙钟，就是在拿睡眠时间冒充加速。
        # 少算一点是保守方向：加速比只会被低估，不会被吹高。
        serial_estimate=sum(r.duration for r in runs if r.index in executed_set),
        concurrency=max(1, concurrency),
        cases=len(runs),
        reused=reused,
        executed=len(pending),
        retried=sum(1 for r in runs if r.attempts > 1),
        degraded=sum(1 for r in runs if r.degraded),
        failed=sum(1 for r in runs if not r.ok),
        llm_calls=sum(r.llm_calls for r in runs),
        llm_failures=sum(r.llm_failures for r in runs),
    )
    return runs, stats


def _run_one(
    case: dict[str, Any],
    index: int,
    case_id: str,
    config: RuntimeConfig,
    llm_factory: Optional[Callable[[], LLM]],
    cases_dir: Optional[Path],
) -> CaseRun:
    """跑一条用例。异常分两类，处理方式刻意不同。

    **传输/端点失败**（`LLMUnavailable`）→ 由 MeteredLLM 计数，
    用例照常返回结果，但被标记 `degraded`，外层会重跑它。

    **其他异常**（`KeyError` / `TypeError` / 断言炸了）→ 立刻记为基础设施故障，
    不重跑、不降级、不混进通过率。这类是代码 bug，重跑一百次也是同一个 bug，
    而且把它算成"用例失败"会让评测报告替 bug 背锅。
    """
    started = time.time()
    meter: Optional[MeteredLLM] = None

    def factory() -> LLM:
        nonlocal meter
        meter = MeteredLLM((llm_factory or _default_llm_factory(config))(), label=config.model or config.llm_provider)
        return meter

    harness = EvalHarness(config, cases_dir, llm_factory=factory)
    try:
        result = harness.run_case(case)
    except Exception as exc:
        return CaseRun(
            index=index,
            case_id=case_id,
            duration=time.time() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    return CaseRun(
        index=index,
        case_id=case_id,
        result=result,
        duration=time.time() - started,
        llm_calls=meter.calls if meter else 0,
        llm_failures=meter.failures if meter else 0,
        llm_last_error=meter.last_error if meter else "",
    )


def _default_llm_factory(config: RuntimeConfig) -> Callable[[], LLM]:
    from ..llm import build_llm

    return lambda: build_llm(
        config.llm_provider,
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=config.llm_timeout,
        retries=config.llm_retries,
    )


# --------------------------------------------------------------------------- #
# 报告辅助
# --------------------------------------------------------------------------- #
def degraded_summary(runs: list[CaseRun]) -> dict[str, Any]:
    """把"哪些用例被降级了"单独说清楚。

    这一段存在的意义：模型跑批里最糟的失败方式不是"跑挂了"，
    而是"端点中途挂了，后半程悄悄退回模板，报告显示 91%"。
    不把降级单独列出来，那个 91% 就是假的。
    """
    degraded = [r for r in runs if r.degraded]
    failed = [r for r in runs if not r.ok]
    calls = sum(r.llm_calls for r in runs)
    return {
        "total": len(runs),
        "degraded": len(degraded),
        "failed": len(failed),
        "llm_calls": calls,
        "llm_failures": sum(r.llm_failures for r in runs),
        "degraded_ids": [r.case_id for r in degraded],
        "failed_ids": [r.case_id for r in failed],
        "verdict": _degraded_verdict(len(runs), len(degraded), len(failed), calls),
    }


def _degraded_verdict(total: int, degraded: int, failed: int, calls: int = 0) -> str:
    if not total:
        return "没有用例"
    if failed:
        return f"有 {failed} 条用例没跑完（基础设施故障，不是模型答错）—— 这次跑批不完整"
    if calls == 0:
        # 离线跑批（没配模型）会走到这里。必须说清楚"没有模型参与" ——
        # 否则报告会写"全部用例都拿到了模型输出"，而实际上一次都没调用过。
        # 那是最尴尬的一种不诚实：分数是真的，但它衡量的东西和读者以为的不是一回事。
        return (
            "这次一次模型都没调用（走的是离线启发式路径）。"
            "这份分数衡量的是**框架的确定性逻辑**，不能当作模型能力 —— "
            "要测模型请加 --provider / --model。"
        )
    if degraded / total > 0.10:
        return (
            f"有 {degraded}/{total} 条用例期间模型调用失败过，已重试仍失败。"
            "占比超过 10%，**这次跑批的分数不能当作模型能力** —— 先解决端点稳定性"
        )
    if degraded:
        return (
            f"有 {degraded}/{total} 条用例期间模型调用失败过并回退到模板。"
            "占比不高，但读分数时要记得这几条不代表模型能力"
        )
    return "全部用例都拿到了模型输出，没有回退 —— 这份分数可以当结论用"


def render_batch_stats(stats: BatchStats) -> str:
    if stats.llm_calls:
        model_part = f"模型调用 {stats.llm_calls} 次"
        if stats.llm_failures:
            model_part += f"（失败 {stats.llm_failures}）"
    else:
        # 不写"调用 0 次"，写清楚这意味着什么 —— 否则读者会以为模型参与了
        model_part = "未调用模型（离线路径）"

    if stats.executed:
        perf_part = (
            f"墙钟 {stats.wall:.0f}s｜串行估计 {stats.serial_estimate:.0f}s｜"
            f"并发 {stats.concurrency} → 实测加速 {stats.speedup}×"
            f"（效率 {stats.efficiency:.0%}）"
        )
    else:
        # 全部复用检查点：墙钟接近 0，算出来的"加速比"会是几千倍这种
        # 什么都没跑出来的假数字。直接说清楚没有实际跑批。
        perf_part = f"没有实际跑批（{stats.reused} 条全部复用检查点）"

    return (
        f"跑批：{stats.cases} 条｜{perf_part}｜"
        + model_part
        + (f"｜复用 {stats.reused} 条" if stats.reused and stats.executed else "")
        + (f"｜重跑 {stats.retried} 条" if stats.retried else "")
        + (f"｜降级 {stats.degraded} 条" if stats.degraded else "")
        + (f"｜故障 {stats.failed} 条" if stats.failed else "")
    )
