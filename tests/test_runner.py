"""并行跑批的测试。

这个文件按 `runner.py` 模块 docstring 里列的**四类新问题**组织，
每一类都对应一组测试 —— 因为那四类问题的共同点是
"结果看起来正常、其实不可信"，光靠跑一遍是发现不了的：

    一、顺序错乱            → 结果必须按下标落位，与串行逐条对齐
    二、失败被静默降级      → 端点挂了不能长得像"模型答得不错"
    三、检查点被写坏        → 不留截断文件
    四、并发被当成免费午餐  → 加速比要量出来，不是声明出来

额外一组：**并行不改变答案**。没有这条，前面四组全都没意义 ——
如果并行本身就会让结果不一样，那"顺序对齐"对齐的是一份错的答案。
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from npc_agent.config import RuntimeConfig
from npc_agent.eval import runner as R
from npc_agent.eval.harness import EvalHarness
from npc_agent.llm import LLM, LLMUnavailable


# --------------------------------------------------------------------------- #
# 测试用的假模型
# --------------------------------------------------------------------------- #
class FakeLLM(LLM):
    """可控的假模型：能慢、能坏、能只坏前几次。

    `fail_first` 是**每个实例**前几次调用失败。因为 `run_cases` 给每条用例
    都造一个新实例，所以它天然表示"这条用例的前几次调用失败"。

    **`available` 和"会不会失败"是两件事**，这里刻意分开。
    第一版把 `fail_forever` 也当成 `available=False`，结果框架看到
    "模型不可用"就根本不调用它，直接走启发式 —— 于是测出来的现象是
    "零次调用、零次失败、用例照常通过"，正是这个模块要抓的那种
    "失败长得像成功"。端点会挂不等于端点没配：`available` 说的是
    "配好了没有"，`fail_*` 说的是"调了会不会炸"。
    """

    name = "fake"

    def __init__(
        self,
        *,
        reply: str = "好，稍等，我这就去弄。",
        delay: float = 0.0,
        fail_first: int = 0,
        fail_forever: bool = False,
        explode: str = "",
        available: bool = True,
    ) -> None:
        self.reply = reply
        self.delay = delay
        self.fail_first = fail_first
        self.fail_forever = fail_forever
        self.explode = explode
        self._available = available
        self.calls = 0
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._available

    def complete(self, messages, *, temperature: float = 0.7, max_tokens: int = 512) -> str:
        with self._lock:
            self.calls += 1
            n = self.calls
        if self.delay:
            time.sleep(self.delay)
        if self.explode:
            # 非 LLMUnavailable 的异常 = 代码 bug，不该被当成"模型抖动"
            raise ValueError(self.explode)
        if self.fail_forever or n <= self.fail_first:
            raise LLMUnavailable("fake endpoint down")
        return self.reply


def _factory(**kwargs):
    def make() -> LLM:
        return FakeLLM(**kwargs)

    return make


def _cases(n: int = 4) -> list[dict]:
    """从真实用例集里取 n 条 —— 用真用例，避免测试和 harness 脱节。

    取**全部**类别而不只是手写的那几条：`generated.jsonl` 是跑批的主力，
    只用 3 条手写用例测并行，测的是一个不存在的规模。
    """
    harness = EvalHarness(RuntimeConfig())
    cases = harness.load_cases()
    assert len(cases) >= n, "用例集太小，跑不了这组测试"
    return cases[:n]


def _run(cases, *, concurrency: int, factory, **kwargs):
    return R.run_cases(
        cases,
        RuntimeConfig(),
        concurrency=concurrency,
        max_retries=kwargs.pop("max_retries", 0),
        backoff=kwargs.pop("backoff", 0.0),
        llm_factory=factory,
        sleep=kwargs.pop("sleep", lambda _s: None),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# 一、顺序错乱
# --------------------------------------------------------------------------- #
def test_results_are_ordered_by_index_not_by_completion() -> None:
    """并发下完成顺序是乱的，但报告顺序必须与串行完全一致。

    做法是让**后面的用例先跑完**：给前面的用例加延迟。
    如果实现按完成顺序收集，这里就会看到下标乱序。
    """
    cases = _cases(6)

    def factory() -> LLM:
        # 用一个全局计数让"第几条用例"越靠后越快，制造完成顺序倒挂
        with _counter_lock:
            _counter["n"] += 1
            n = _counter["n"]
        return FakeLLM(reply="好，稍等，我这就去弄。", delay=0.06 * (6 - n))

    _counter.clear()
    _counter["n"] = 0
    runs, _ = _run(cases, concurrency=6, factory=factory)

    assert [r.index for r in runs] == list(range(len(cases))), (
        "结果没有按下标落位 —— 报告顺序会随线程调度漂移，--compare 直接失效"
    )


_counter: dict[str, int] = {}
_counter_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# 并行不改变答案
# --------------------------------------------------------------------------- #
def test_parallel_agrees_with_serial_case_by_case() -> None:
    """同一批用例，并发 1 和并发 4 必须给出**逐条相同**的结论。

    这是整个并行改造的前提：并发只该改变花多久，不该改变算出什么。
    不测这条，后面"顺序对齐""加速比"全都是在给一份错的答案做包装。
    """
    cases = _cases(6)
    factory = _factory()

    serial_runs, serial_stats = _run(cases, concurrency=1, factory=factory)
    par_runs, par_stats = _run(cases, concurrency=4, factory=factory)

    assert serial_stats.concurrency == 1
    assert par_stats.concurrency == 4
    assert [r.case_id for r in serial_runs] == [r.case_id for r in par_runs]

    for s, p in zip(serial_runs, par_runs):
        assert s.ok and p.ok, f"{s.case_id} 没跑完"
        assert s.result.passed == p.result.passed, f"{s.case_id} 结论不一致"
        assert s.result.metrics.as_dict() == p.result.metrics.as_dict(), (
            f"{s.case_id} 各维度分数不一致"
        )


def test_concurrency_one_does_not_build_a_thread_pool(monkeypatch) -> None:
    """并发 1 走串行路径。

    保留串行路径不是为了兼容，是为了让并行结果**可以被串行验证**。
    所以这里直接断言"串行时根本没建线程池" —— 否则所谓串行基线
    可能自己也在并发跑，那它就不是基线了。
    """

    def boom(*_a, **_kw):
        raise AssertionError("concurrency=1 不应该建线程池")

    monkeypatch.setattr(R, "ThreadPoolExecutor", boom)
    cases = _cases(3)
    runs, stats = _run(cases, concurrency=1, factory=_factory())
    assert len(runs) == 3
    assert stats.concurrency == 1


# --------------------------------------------------------------------------- #
# 二、失败被静默降级
# --------------------------------------------------------------------------- #
def test_a_model_that_always_fails_is_marked_degraded_not_silently_passed() -> None:
    """端点全程挂掉时，用例仍然会"跑完"（框架回退到模板），但必须被标成降级。

    这是最危险的一类失败：`LLMUnavailable` 是设计好的回退信号，
    所以"端点挂了"和"模型答得不错"在报告里长得一模一样。
    """
    cases = _cases(3)
    runs, stats = _run(cases, concurrency=3, factory=_factory(fail_forever=True))

    assert all(r.ok for r in runs), "回退路径应该让用例照常跑完"
    assert all(r.degraded for r in runs), "全程失败却没标降级 —— 报告会说谎"
    assert all(r.llm_failures > 0 for r in runs)
    assert stats.degraded == 3

    summary = R.degraded_summary(runs)
    assert summary["degraded"] == 3
    assert summary["llm_failures"] > 0
    assert "不能当作模型能力" in summary["verdict"], (
        "100% 降级却给出了温和结论 —— 这份分数会被当成真的"
    )


def test_degraded_verdict_uses_a_ten_percent_threshold() -> None:
    """10% 是分界线：超过就不能拿分数当结论，不到就只提示。

    边界值单独测，因为"恰好 10%"最容易被 `>` 和 `>=` 写反 ——
    写反了不会报错，只会让一份不可信的报告看起来可信。
    """
    assert "占比不高" in R._degraded_verdict(10, 1, 0, calls=100)  # 10% 不算超过
    assert "不能当作模型能力" in R._degraded_verdict(10, 2, 0, calls=100)  # 20% 超过
    assert "全部用例都拿到了模型输出" in R._degraded_verdict(10, 0, 0, calls=100)


def test_an_offline_run_does_not_claim_it_got_model_output() -> None:
    """离线跑批（一次模型都没调用）不能说"拿到了模型输出"。

    这条是实测发现的：228 条离线用例全绿，报告结尾写着
    "全部用例都拿到了模型输出，没有回退" —— 而实际上一次都没调用过。
    分数是真的，但它衡量的东西和这句话暗示的不是一回事。
    """
    cases = _cases(3)
    runs, stats = _run(cases, concurrency=2, factory=_factory(available=False))

    assert stats.llm_calls == 0
    verdict = R.degraded_summary(runs)["verdict"]
    assert "一次模型都没调用" in verdict
    assert "拿到了模型输出" not in verdict
    assert "框架的确定性逻辑" in verdict

    # 报告正文里也要看得见，不能只在终端
    assert stats.to_dict()["llm_calls"] == 0
    assert "未调用模型" in R.render_batch_stats(stats)


def test_infrastructure_failure_is_reported_separately_from_a_wrong_answer() -> None:
    """代码 bug 炸出来的用例，不能被算成"模型答错了"。

    算进去的话，评测报告就替 bug 背了锅 —— 而且这个锅会挂在模型头上，
    下一次调模型参数时就会去修一个根本不存在的模型问题。
    """
    cases = _cases(3)
    runs, stats = _run(
        cases, concurrency=1, factory=_factory(explode="boom"), max_retries=2
    )

    assert all(not r.ok for r in runs)
    assert all(r.result is None for r in runs)
    assert all("ValueError" in r.error for r in runs)
    assert stats.failed == 3
    assert stats.degraded == 0, "代码 bug 不该被算成端点降级"

    summary = R.degraded_summary(runs)
    assert len(summary["failed_ids"]) == 3
    assert "不是模型答错" in summary["verdict"]


def test_metered_llm_only_counts_unavailable_failures() -> None:
    """`MeteredLLM` 只该把 `LLMUnavailable` 记成失败，其他异常必须照常抛。

    把 `ValueError` 也记成"模型抖动"，会让真正的代码 bug 混进重试逻辑里，
    然后被重试掩盖掉 —— 重试一百次还是同一个 bug。
    """
    meter = R.MeteredLLM(FakeLLM(fail_forever=True), label="t")
    with pytest.raises(LLMUnavailable):
        meter.complete([])
    assert meter.calls == 1 and meter.failures == 1

    boom = R.MeteredLLM(FakeLLM(explode="bug"), label="t")
    with pytest.raises(ValueError):
        boom.complete([])
    assert boom.calls == 1 and boom.failures == 0, "代码 bug 被记成了模型失败"


def test_metered_llm_passes_through_the_inner_name_and_availability() -> None:
    meter = R.MeteredLLM(FakeLLM(), label="kimi-k2.7-code")
    assert meter.name == "kimi-k2.7-code"
    assert meter.available is True
    assert R.MeteredLLM(FakeLLM(available=False)).available is False


def test_a_configured_but_failing_endpoint_still_gets_called() -> None:
    """端点会挂 ≠ 端点没配。这两件事混淆会让降级检测整个失效。

    这条测试的来历是我自己写错的一个测试：第一版把 `fail_forever` 也当成
    `available=False`，于是框架认为"模型没配"直接走启发式，一次都没调用 ——
    用例全部通过、零次失败、零次降级，报告干干净净。

    这正是 `runner.py` 要防的那一类失败：**"端点挂了"长得和"模型答得不错"
    一模一样**。区别在于真实世界里没人会替你把这个 `available` 设错，
    所以只能靠测试把这个区分钉住。
    """
    assert FakeLLM(fail_forever=True).available is True
    assert FakeLLM(fail_first=1).available is True

    runs, stats = _run(
        _cases(2), concurrency=1, factory=_factory(fail_forever=True), max_retries=0
    )
    assert all(r.llm_calls > 0 for r in runs), (
        "配置好的模型一次都没被调用 —— 降级检测根本没有机会生效"
    )
    assert all(r.degraded for r in runs)


def test_an_unconfigured_endpoint_is_skipped_without_calls() -> None:
    """反过来：真没配模型时（`available=False`），框架走启发式、不调用它。

    这是**正确**行为，不该被算成降级 —— 离线跑批全靠它。
    和上一条合起来才说明白：`degraded` 的含义是"配了但调用失败"，
    不是"这次跑的是启发式"。
    """
    runs, stats = _run(
        _cases(2), concurrency=1, factory=_factory(available=False), max_retries=2
    )
    assert all(r.ok for r in runs)
    assert all(r.llm_calls == 0 for r in runs)
    assert all(not r.degraded for r in runs), "离线跑批被误判成降级"
    assert stats.retried == 0


# --------------------------------------------------------------------------- #
# 重试
# --------------------------------------------------------------------------- #
def test_a_case_that_fails_then_succeeds_is_retried_and_ends_up_trustworthy() -> None:
    """重试成功的用例，最终**不算**降级 —— 它的结果是可信的。

    同时必须留下 `attempts > 1` 的痕迹：结果可信，但"它被重试过"这件事
    本身是有用的信号（端点不稳时这个数字会先涨）。

    注意这里用的是**跨实例**的抖动工厂，而不是 `fail_first`：
    每次重试都会造一个新的模型实例，所以 `fail_first=1` 会让**每一次**
    尝试都失败，永远重试不完。真实世界里的"抖一下就好了"是
    "第一个连接失败、重连成功"，对应的正是"第一个实例坏、后面的实例好"。
    """
    calls = {"n": 0}
    lock = threading.Lock()

    def factory() -> LLM:
        with lock:
            calls["n"] += 1
            first = calls["n"] == 1
        return FakeLLM(fail_forever=first)

    runs, stats = _run(_cases(1), concurrency=1, factory=factory, max_retries=2)

    assert all(r.attempts == 2 for r in runs), "失败了一次却没有重试"
    assert all(r.ok for r in runs)
    assert all(not r.degraded for r in runs), "重试成功了还算降级，降级比例就失去意义"
    assert stats.retried == 1
    assert stats.degraded == 0
    # 第一次尝试确实调用过模型并且失败了 —— 否则"重试成功"是编的
    assert runs[0].llm_calls > 0
    assert runs[0].llm_failures == 0, "最后一次尝试没有失败，才叫重试成功"


def test_retry_gives_up_after_max_retries() -> None:
    cases = _cases(2)
    runs, stats = _run(
        cases, concurrency=1, factory=_factory(fail_forever=True), max_retries=3
    )
    assert all(r.attempts == 4 for r in runs), "尝试次数应为 1 次首发 + 3 次重试"
    assert all(r.degraded for r in runs)
    assert stats.retried == 2


def test_backoff_grows_exponentially_between_retries() -> None:
    """退避必须是 3s → 6s → 12s，而不是每次都等 3s。

    端点限流时，固定间隔重试等于持续给已经过载的端点加压。
    """
    slept: list[float] = []
    cases = _cases(1)
    _run(
        cases,
        concurrency=1,
        factory=_factory(fail_forever=True),
        max_retries=3,
        backoff=3.0,
        sleep=slept.append,
    )
    assert slept == [3.0, 6.0, 12.0]


def test_no_sleep_happens_when_the_case_succeeds() -> None:
    slept: list[float] = []
    _run(
        _cases(1),
        concurrency=1,
        factory=_factory(),
        max_retries=3,
        backoff=3.0,
        sleep=slept.append,
    )
    assert slept == [], "成功用例不该白白等待"


def test_retry_time_is_counted_in_the_serial_estimate() -> None:
    """重试花掉的时间必须计入串行估计，否则加速比是虚报的。

    串行跑一遍同样要重试 —— 把重试时间排除在外，等于假装重试不要钱。
    """
    cases = _cases(1)
    slow = 0.05

    runs_once, stats_once = _run(
        cases, concurrency=1, factory=_factory(delay=slow), max_retries=0
    )
    runs_retry, stats_retry = _run(
        cases,
        concurrency=1,
        factory=_factory(fail_forever=True, delay=slow),
        max_retries=2,
    )

    assert runs_once[0].attempts == 1
    assert runs_retry[0].attempts == 3
    assert stats_retry.serial_estimate > stats_once.serial_estimate * 2, (
        "重试了 3 次，串行估计却几乎没涨 —— 加速比会因此虚高"
    )


def test_llm_call_counts_accumulate_across_attempts() -> None:
    """调用次数是**累计**的（它回答"花了多少资源"），

    而 `llm_failures` 只取最后一次尝试（它回答"这份结果可不可信"）。
    两个数字语义不同，混成一个就会同时说错两件事。
    """
    cases = _cases(1)
    runs, _ = _run(
        cases, concurrency=1, factory=_factory(fail_forever=True), max_retries=2
    )
    run = runs[0]
    assert run.attempts == 3
    # 调用次数跨尝试累加（10 次/条 × 3 次），失败次数只取最后一次尝试。
    # 这就是两个数字语义不同的地方：calls 回答"花了多少资源"，
    # failures 回答"这份结果可不可信"。断言写成 calls == failures
    # 就等于把两个语义混成一个 —— 那正是这段代码要避免的。
    assert run.llm_calls == run.llm_failures * run.attempts
    assert run.llm_failures > 0
    assert run.llm_last_error


# --------------------------------------------------------------------------- #
# 三、检查点
# --------------------------------------------------------------------------- #
def test_checkpoint_file_is_always_valid_json(tmp_path) -> None:
    """每次落盘都必须是一个完整 JSON —— 哪怕写到一半被打断。

    截断的 JSON 比没有检查点更糟：下次启动直接解析失败，
    而"跑了两小时"的进度就锁在那个坏文件里了。
    """
    path = tmp_path / "ckpt.json"
    state = {"done": 0}

    def payload() -> dict:
        return {"done": state["done"], "runs": list(range(state["done"]))}

    ckpt = R.Checkpoint(path, payload)
    for i in range(1, 6):
        state["done"] = i
        ckpt.save()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["done"] == i
        assert len(data["runs"]) == i

    assert not path.with_suffix(path.suffix + ".tmp").exists(), "临时文件没清掉"
    assert ckpt.writes == 5


def test_checkpoint_survives_concurrent_writes(tmp_path) -> None:
    """并发下多个线程同时落盘不能互相写坏。

    没有锁的话，两个线程会同时写同一个临时文件，`os.replace` 之后再读
    就可能拿到混合内容 —— 而这是**最难复现**的一类 bug。
    """
    path = tmp_path / "ckpt.json"
    counter = {"n": 0}
    lock = threading.Lock()

    def payload() -> dict:
        with lock:
            counter["n"] += 1
            n = counter["n"]
        return {"n": n, "pad": "x" * 200}

    ckpt = R.Checkpoint(path, payload)

    def hammer() -> None:
        for _ in range(5):
            ckpt.save()

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert ckpt.writes == 40
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["pad"] == "x" * 200, "文件被写成了半截"
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_checkpoint_is_called_once_per_case(tmp_path) -> None:
    cases = _cases(3)
    path = tmp_path / "ckpt.json"
    state: dict = {"runs": []}
    ckpt = R.Checkpoint(path, lambda: {"runs": state["runs"]})

    def on_done(run, done, total) -> None:
        state["runs"].append(run.to_dict())

    _run(cases, concurrency=3, factory=_factory(), on_done=on_done, checkpoint=ckpt)
    assert ckpt.writes == 3
    assert json.loads(path.read_text(encoding="utf-8"))["runs"]


# --------------------------------------------------------------------------- #
# 四、加速比要量出来
# --------------------------------------------------------------------------- #
def test_speedup_is_measured_not_asserted() -> None:
    """加速比必须是量出来的数字，而且要能和并发数对照。

    这条测试只断言"并行确实比串行快"，并检查口径自洽 ——
    不断言一个具体倍率，因为那会把测试变成对机器性能的断言。
    """
    cases = _cases(6)
    delay = 0.06

    _runs_s, stats_serial = _run(
        cases, concurrency=1, factory=_factory(delay=delay)
    )
    _runs_p, stats_par = _run(cases, concurrency=3, factory=_factory(delay=delay))

    assert stats_par.wall < stats_serial.wall, "并发 3 没有比串行快，并行改造没生效"
    assert stats_par.speedup > 1.5, f"实测加速比只有 {stats_par.speedup}×"
    assert 0 < stats_par.efficiency <= 1.0
    assert stats_par.concurrency == 3

    # 口径自洽：串行估计 = 每条用例耗时之和
    assert stats_par.serial_estimate == pytest.approx(
        sum(r.duration for r in _runs_p), abs=0.05
    )


def test_speedup_never_claims_more_than_the_concurrency() -> None:
    """加速比不可能超过并发数 —— 超过了说明分子分母口径不一致。"""
    cases = _cases(4)
    runs, stats = _run(cases, concurrency=2, factory=_factory(delay=0.02))
    assert stats.speedup <= 2.0 + 0.05, (
        f"加速比 {stats.speedup}× 超过了并发数 2 —— 串行估计把睡眠时间也算进去了"
    )


def test_batch_stats_are_machine_readable() -> None:
    cases = _cases(2)
    _, stats = _run(cases, concurrency=2, factory=_factory(fail_first=0))
    payload = stats.to_dict()
    for key in (
        "wall_sec",
        "serial_estimate_sec",
        "speedup",
        "concurrency",
        "efficiency",
        "cases",
        "retried_cases",
        "degraded_cases",
        "failed_cases",
    ):
        assert key in payload, f"报告里缺 {key}，读报告的人就看不到这个数字"
    assert payload["cases"] == 2
    assert isinstance(R.render_batch_stats(stats), str)


def test_empty_batch_is_handled() -> None:
    """零条用例不该炸，也不该给出一个编出来的加速比。"""
    runs, stats = R.run_cases([], RuntimeConfig(), concurrency=4)
    assert runs == []
    assert stats.cases == 0
    assert stats.speedup == 0.0
    assert R.degraded_summary([])["verdict"] == "没有用例"


def test_the_cli_checkpoint_is_actually_populated_during_the_run(tmp_path, monkeypatch) -> None:
    """`eval --checkpoint` 写出来的文件必须有内容，而且是**边跑边长**的。

    这条测试针对一个真实的接线 bug：`run_cases` 的检查点本身是好的
    （上面几条测试都在测它），但 CLI 侧等 `run_cases` 返回之后才把结果
    喂给检查点，于是整个跑批期间写出来的始终是 `{"done": 0, "runs": []}`。
    跑三小时被杀，打开检查点发现什么都没存。

    `run_cases` 的单测发现不了这个 —— 它测的是"检查点被调用了"，
    而不是"喂进去的东西是真的"。所以必须从 CLI 入口跑一遍。
    """
    from npc_agent import cli as C

    ckpt = tmp_path / "ckpt.json"
    report = tmp_path / "report.json"
    monkeypatch.setenv("NPC_AGENT_PROVIDER", "null")
    code = C.main(
        [
            "eval",
            "--limit",
            "4",
            "--concurrency",
            "2",
            "--checkpoint",
            str(ckpt),
            "--json",
            str(report),
        ]
    )
    assert code == 0, "离线跑批应该干净通过"

    assert ckpt.exists(), "检查点文件根本没生成"
    data = json.loads(ckpt.read_text(encoding="utf-8"))
    assert data["total"] == 4
    assert data["done"] == 4, f"检查点只记了 {data['done']}/4 条 —— 接线漏了"
    assert len(data["runs"]) == 4, "检查点里没有逐条结果，被杀之后什么都恢复不了"
    assert all(r["case_id"] for r in data["runs"])
    assert not ckpt.with_suffix(ckpt.suffix + ".tmp").exists()


def test_the_cli_checkpoint_is_actually_populated_during_the_run(tmp_path, monkeypatch) -> None:
    """`eval --checkpoint` 写出来的文件必须有内容，而且是**边跑边长**的。

    这条测试针对一个真实的接线 bug：`run_cases` 的检查点本身是好的
    （上面几条测试都在测它），但 CLI 侧等 `run_cases` 返回之后才把结果
    喂给检查点，于是整个跑批期间写出来的始终是 `{"done": 0, "runs": []}`。
    跑三小时被杀，打开检查点发现什么都没存。

    `run_cases` 的单测发现不了这个 —— 它测的是"检查点被调用了"，
    而不是"喂进去的东西是真的"。所以必须从 CLI 入口跑一遍。
    """
    from npc_agent import cli as C

    ckpt = tmp_path / "ckpt.json"
    report = tmp_path / "report.json"
    monkeypatch.setenv("NPC_AGENT_PROVIDER", "null")
    code = C.main(
        [
            "eval",
            "--limit",
            "4",
            "--concurrency",
            "2",
            "--checkpoint",
            str(ckpt),
            "--json",
            str(report),
        ]
    )
    assert code == 0, "离线跑批应该干净通过"

    assert ckpt.exists(), "检查点文件根本没生成"
    data = json.loads(ckpt.read_text(encoding="utf-8"))
    assert data["total"] == 4
    assert data["done"] == 4, f"检查点只记了 {data['done']}/4 条 —— 接线漏了"
    assert len(data["runs"]) == 4, "检查点里没有逐条结果，被杀之后什么都恢复不了"
    assert all(r["case_id"] for r in data["runs"])
    assert not ckpt.with_suffix(ckpt.suffix + ".tmp").exists()


# --------------------------------------------------------------------------- #
# 中断后恢复
# --------------------------------------------------------------------------- #
def test_checkpoint_round_trip_preserves_the_result() -> None:
    """检查点必须能还原出完整结果 —— 分数、转写、台词、失败原因。

    只记"这条跑过了"是不够的：恢复的时候拿不出分数和转写，
    等于白存 —— 而报告和判分都要靠它们。
    """
    cases = _cases(2)
    runs, _ = _run(cases, concurrency=2, factory=_factory())
    assert all(r.ok for r in runs)

    for original in runs:
        restored = R.CaseRun.from_dict(original.to_dict(include_result=True))
        assert restored.case_id == original.case_id
        assert restored.ok
        assert restored.result is not None
        assert restored.result.metrics.as_dict() == original.result.metrics.as_dict()
        assert restored.result.transcript == original.result.transcript
        assert restored.result.speeches == original.result.speeches
        assert restored.result.notes == original.result.notes

    # 不带 include_result 时不该塞进结果（进度快照不需要那么重）
    assert "result" not in runs[0].to_dict()


def test_resume_reuses_completed_cases_and_says_so() -> None:
    """恢复时已完成的用例不再重跑，而且"复用了多少条"要出现在报告里。

    不报出来的话，"跑了 228 条"和"跑了 5 条 + 恢复了 223 条"看起来一模一样。
    """
    cases = _cases(4)
    first, _ = _run(cases, concurrency=2, factory=_factory())
    resume_map = {r.case_id: r for r in first}

    # 这次用一个**会立刻失败**的模型：如果它真去重跑了，结果会变成降级
    second, stats = _run(
        cases,
        concurrency=2,
        factory=_factory(fail_forever=True),
        resume=resume_map,
    )

    assert stats.reused == 4
    assert stats.executed == 0
    assert stats.degraded == 0, "复用的用例被重跑了（重跑的话会因模型失败而降级）"
    assert [r.case_id for r in second] == [r.case_id for r in first]
    assert [r.result.metrics.as_dict() for r in second] == [
        r.result.metrics.as_dict() for r in first
    ]


def test_a_partially_reused_batch_only_runs_the_remainder() -> None:
    cases = _cases(4)
    first, _ = _run(cases[:2], concurrency=2, factory=_factory())
    resume_map = {r.case_id: r for r in first}

    runs, stats = _run(cases, concurrency=2, factory=_factory(), resume=resume_map)
    assert stats.reused == 2
    assert stats.executed == 2
    assert len(runs) == 4
    assert [r.index for r in runs] == [0, 1, 2, 3], "恢复后的下标没有重新映射"


def test_resume_never_reports_a_fake_speedup() -> None:
    """全部复用检查点时不能报出一个巨大的加速比。

    实测踩到过：全部复用 → 墙钟接近 0 → `串行估计 / 墙钟` 算出 **9611×**。
    这是一个"看起来很厉害、其实什么都没跑"的数字，正是这个模块要防的那类。
    """
    cases = _cases(3)
    first, _ = _run(cases, concurrency=2, factory=_factory())
    resume_map = {r.case_id: r for r in first}

    _, stats = _run(cases, concurrency=2, factory=_factory(), resume=resume_map)
    assert stats.speedup == 0.0
    assert stats.efficiency == 0.0
    assert "没有实际跑批" in R.render_batch_stats(stats)
    assert "加速" not in R.render_batch_stats(stats)

    # 复用来的耗时也不能算进串行估计
    assert stats.serial_estimate == 0.0


def test_reused_durations_are_excluded_from_the_serial_estimate() -> None:
    """复用的那些用例这次没花时间，串行跑一遍也不会再花时间。

    把它们算进串行估计，"加速比"就变成了"复用省下多少"的冒牌货，
    而不是"并发快了多少"。
    """
    cases = _cases(4)
    first, _ = _run(cases[:2], concurrency=2, factory=_factory(delay=0.05))
    resume_map = {r.case_id: r for r in first}

    _, stats = _run(
        cases, concurrency=1, factory=_factory(delay=0.05), resume=resume_map
    )
    assert stats.reused == 2
    assert stats.executed == 2
    # 只跑了 2 条，所以串行估计应该接近"2 条 × 各自耗时"，
    # 而不是"4 条之和"。这里用一个宽区间锁住量级，避免测试依赖具体机器速度。
    assert 0 < stats.serial_estimate < sum(
        r.duration for r in _run(cases, concurrency=1, factory=_factory(delay=0.05))[0]
    ), "串行估计里混进了复用条目的耗时"


def test_resume_refuses_when_the_config_changed() -> None:
    """配置对不上必须拒绝恢复 —— 这是恢复功能里最重要的一条。

    把 `--no-planner` 跑出来的半批和 planner-on 跑出来的另半批拼起来，
    报告会把两个不同的变量混成一列，而且从数字上完全看不出来。
    宁可整批重跑，也不要一份悄悄混了两个配置的报告。
    """
    config_a = RuntimeConfig()
    config_b = RuntimeConfig()
    config_b.use_llm_planner = not config_a.use_llm_planner

    # 用一份**真跑出来的**结果做载荷。手工拼一个 {"ok": True} 是不行的：
    # 没有 result 的记录恢复不出分数和转写，`plan_resume` 会（正确地）拒绝它。
    done, _ = _run(_cases(2), concurrency=2, factory=_factory())
    payload = {
        "config": R.config_fingerprint(config_a),
        "runs": [r.to_dict(include_result=True) for r in done],
    }
    usable, why = R.plan_resume(payload, config_a)
    assert usable and "可复用" in why

    usable, why = R.plan_resume(payload, config_b)
    assert usable == {}, "配置变了却仍然复用了结果"
    assert "另一套配置" in why
    assert "use_llm_planner" in why, "没有说清楚是哪一项配置对不上"

    # 并发数不属于"影响结果"的配置，改它不该导致整批重跑
    assert "concurrency" not in R.RESUME_CRITICAL_FIELDS


def test_resume_skips_cases_that_did_not_finish() -> None:
    """上次没跑完的用例（基础设施故障）必须重跑，不能当成"已完成"。

    它失败的原因是当时的环境，不是模型 —— 换一次跑很可能就过了。
    把它当成完成，就等于把一次事故永久固化进报告。
    """
    done, _ = _run(_cases(1), concurrency=1, factory=_factory())
    good = done[0].to_dict(include_result=True)
    crashed = dict(good, case_id="crashed", index=1, ok=False, error="ValueError: boom")
    crashed.pop("result", None)

    payload = {
        "config": R.config_fingerprint(RuntimeConfig()),
        "runs": [good, crashed],
    }
    usable, _ = R.plan_resume(payload, RuntimeConfig())
    assert good["case_id"] in usable
    assert "crashed" not in usable


def test_a_corrupt_checkpoint_is_ignored_not_fatal(tmp_path) -> None:
    """检查点读坏了就当成没有，不能把整条命令搞崩。

    检查点的职责是"尽量救回一点"，不是"必须存在"。
    一个截断的检查点如果让命令直接崩掉，它就从保险变成了新的故障源。
    """
    assert R.load_checkpoint(tmp_path / "nope.json") == {}

    bad = tmp_path / "bad.json"
    bad.write_text('{"done": 3, "runs": [{"case_id": "x"', encoding="utf-8")
    assert R.load_checkpoint(bad) == {}

    weird = tmp_path / "weird.json"
    weird.write_text("[1, 2, 3]", encoding="utf-8")
    assert R.load_checkpoint(weird) == {}

    # 结果字段坏掉的单条记录 → 当成"没跑过"，而不是抛异常
    restored = R.CaseRun.from_dict({"case_id": "x", "ok": True, "result": "不是字典"})
    assert restored.result is None


def test_the_cli_resume_round_trip(tmp_path, monkeypatch) -> None:
    """从 CLI 入口跑一遍完整的"跑一半 → 恢复"。"""
    from npc_agent import cli as C

    monkeypatch.setenv("NPC_AGENT_PROVIDER", "null")
    ckpt = tmp_path / "ck.json"

    assert C.main(["eval", "--limit", "3", "--concurrency", "2",
                   "--checkpoint", str(ckpt), "--json", str(tmp_path / "a.json")]) == 0
    first = json.loads(ckpt.read_text(encoding="utf-8"))
    assert first["done"] == 3
    assert first["config"]["llm_provider"] == "null"

    # 换成"更大的 limit"，前 3 条应复用、后 2 条新跑
    assert C.main(["eval", "--limit", "5", "--concurrency", "2", "--resume",
                   "--checkpoint", str(ckpt), "--json", str(tmp_path / "b.json")]) == 0
    second = json.loads(ckpt.read_text(encoding="utf-8"))
    assert second["done"] == 5, "复用来的结果没有写回检查点，第二次中断就丢了"
    assert len(second["runs"]) == 5
    assert all("result" in r for r in second["runs"]), "检查点里没有完整结果，恢复不出来"

    report = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
    assert report["summary"]["total"] == 5
    assert report["batch"]["stats"]["reused_cases"] == 3
    assert report["batch"]["stats"]["executed_cases"] == 2


def test_resume_without_a_checkpoint_path_is_an_error(monkeypatch) -> None:
    """`--resume` 不给 `--checkpoint` 是自相矛盾的，要说清楚而不是静默全跑。"""
    from npc_agent import cli as C

    monkeypatch.setenv("NPC_AGENT_PROVIDER", "null")
    assert C.main(["eval", "--limit", "2", "--resume"]) == 2
