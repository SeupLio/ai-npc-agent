"""本地自测控制台：一条命令，一个网页，不用模型也能玩。

    python -m npc_agent.cli studio          # 起服务并打开浏览器
    python -m npc_agent.cli studio --no-open --port 9000

## 为什么要有它

这个项目原来的入口全是命令行：`demo` / `chat` / `eval` / `sensitivity` …
每个都要记参数，而且 `chat` 只能对着终端打字、`eval` 只印一张表、
`sensitivity` 只告诉你"全被抓到"。**能证明结论的东西都在，但不好看。**

一个求职项目的自测入口应该满足三件事：

1. **一条命令**，不用记参数；
2. **看得见** —— 对话、世界状态、记忆、六维分数同屏；
3. **能自己动手推翻结论** —— 点一下注入缺陷，亲眼看评测掉分。

所以这里做的是一个**离线**控制台：默认走启发式路径，不联网、不调模型。
配了 `NPC_AGENT_PROVIDER=openai-compat` 就自动切成真实模型（`/api/meta` 会报）。

## 设计取舍

- **只用标准库**（`http.server`）。加一个 Web 框架就为了这点路由，
  不值得让 `pip install` 多一个依赖 —— 而且这个项目对"离线可复现"很执着。
- **无状态**：`/api/chat` 每次请求都**从头重放**整段对话。
  客户端只发事件列表，服务端不存会话。这样刷新页面不会错乱，
  而且同一段对话**必然复现同样的输出**（可复现是这个项目的底线）。
- **变异/评测串行**：`sensitivity` 的注入是改**类属性**，全局生效。
  两个请求并发跑会让"哪个缺陷导致掉分"说不清 —— 所以用一把锁串起来。
- **报告列表以文件系统为准**，不以清单为准（见 `eval/report_index.py`）。
"""

from __future__ import annotations

import json
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .cast import Cast, build_cast
from .config import RuntimeConfig, list_scenarios, load_scenario
from .env import env_label
from .eval.harness import EvalHarness, EvalReport
from .eval.report_index import DOCS_DIR, index as report_index
from .eval.sensitivity import MUTANTS, run_sensitivity
from .llm import build_llm
from .modules.repetition import find_repeats
from .studio_ui import PAGE

#: 一次对话最多重放多少个事件。前端每发一句话就把**整段**历史重发一遍，
#: 所以这个上限同时也是"单次请求的计算量上限"。
MAX_EVENTS = 80

#: 变异/评测的并发闸门。见模块 docstring 最后一条。
_RUN_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# 组装
# --------------------------------------------------------------------------- #
def _config(overrides: dict[str, Any] | None = None) -> RuntimeConfig:
    cfg = RuntimeConfig.from_env()
    for key, value in (overrides or {}).items():
        if value not in (None, ""):
            setattr(cfg, key, value)
    return cfg


def _make_llm(cfg: RuntimeConfig):
    return build_llm(
        cfg.llm_provider,
        model=cfg.model,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        timeout=cfg.llm_timeout,
    )


def _scenario_summaries(cfg: RuntimeConfig) -> list[dict[str, Any]]:
    """场景列表。为了拿到 NPC 名字真的造一次剧组 —— 便宜且不会和 demo 走岔。"""
    # 延迟导入：`cli` 只在 `cmd_studio` 里导入本模块，所以这里不成环。
    # 例子台词是 demo 的脚本，**不另抄一份** —— 抄了就会和 demo 漂移。
    from .cli import DEMO_SCRIPTS

    out: list[dict[str, Any]] = []
    for scenario_id in list_scenarios():
        scenario = load_scenario(scenario_id)
        cast = build_cast(scenario, _make_llm(cfg), cfg)
        # 脚本里 `None` 表示"这一轮没人说话"，所以要先跳过它再解包。
        sample = next(
            (line[1] for line in (DEMO_SCRIPTS.get(scenario_id) or []) if line),
            "",
        )
        out.append(
            {
                "id": scenario_id,
                "name": scenario.get("name") or scenario_id,
                "description": scenario.get("description") or "",
                "world_label": env_label(cast.env.name),
                "npcs": [a.persona.name for a in cast.agents.values()],
                "players": [
                    {"id": p.get("id"), "name": p.get("name") or p.get("id")}
                    for p in (scenario.get("players") or [])
                ],
                "sample": sample,
            }
        )
    return out


def _case_categories(cfg: RuntimeConfig) -> list[dict[str, Any]]:
    """按类别数一遍用例。**用 harness 自己加载**，不另写一套 glob。"""
    harness = EvalHarness(cfg)
    counts: dict[str, int] = {}
    for case in harness.load_cases():
        key = str(case.get("category") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return [
        {"id": key, "total": total}
        for key, total in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def _commands() -> list[str]:
    return [
        "python -m npc_agent.cli studio                      # 本控制台",
        "python -m npc_agent.cli demo --scenario duet        # 双 NPC 演示",
        "python -m npc_agent.cli demo --scenario village     # Minecraft 体素世界",
        "python -m npc_agent.cli eval                        # 离线六维基线",
        "python -m npc_agent.cli sensitivity --html docs/sensitivity.html",
        "python -m npc_agent.cli ablate                      # 记忆策略消融",
        "python -m npc_agent.cli worlds --html docs/worlds.html",
        "python scripts/regen_docs.py                        # 重生成离线报告",
        "python -m pytest tests                              # 全套测试",
    ]


def build_meta(cfg: RuntimeConfig) -> dict[str, Any]:
    llm = _make_llm(cfg)
    categories = _case_categories(cfg)
    return {
        "llm_available": bool(getattr(llm, "available", False)),
        "provider": cfg.llm_provider,
        "model": cfg.model or "（未配置）",
        "scenarios": _scenario_summaries(cfg),
        "categories": categories,
        "case_total": sum(c["total"] for c in categories),
        "mutants": [
            {
                "id": m.id,
                "kind": m.kind,
                "targets": list(m.targets),
                "description": m.description,
            }
            for m in MUTANTS
        ],
        "mutant_total": len(MUTANTS),
        "reports": report_index(),
        "commands": _commands(),
        # 界面上要解释"为什么按一下空转什么都没发生"。
        # 这个阈值**不能写死在页面里** —— 它来自配置，改了配置页面就会说错话。
        "idle_ticks_before_proactive": int(cfg.idle_ticks_before_proactive),
    }


# --------------------------------------------------------------------------- #
# 对话
# --------------------------------------------------------------------------- #
def _turn_payload(turn: Any, cast: Cast) -> dict[str, Any]:
    # ⚠️ 这里**不发** `turn.acted`。它曾经在响应里，而页面拿它判断
    # "这一轮该怎么显示" —— 但 `acted = say or actions`，一个只说了一句话的
    # 回合 `acted=True` 而 `actions` 为空（成功的 speak 会被下面滤掉），
    # 于是界面显示"在忙自己的事"却一条动作都列不出来。
    # 结局由**可见动作数**决定，`acted` 是回答另一个问题的字段 ——
    # 唯一消费方（页面）已经不用它了，留着只会再次引人用错。
    return {
        "name": cast.name_of(turn.actor_id),
        "say": turn.say,
        "decision_reason": turn.decision_reason,
        # 这个计划是谁产出的（见 `types.PLAN_SOURCES`）。
        # 规划失败会**静默回落**到启发式规划器，不显示来源的话，
        # "模型规划的"和"回落之后的"在页面上长得一模一样。
        "plan_source": getattr(turn.plan, "source", "") if turn.plan else "",
        "actions": [
            {
                "tool": action.tool,
                "render": action.render(),
                "ok": result.ok,
                "detail": result.detail,
            }
            for action, result in zip(turn.actions, turn.results)
            # 成功的 `speak` 已经由 `say` 表达了，再列一遍只会让日志变吵；
            # 但**失败的** speak 必须留着 —— 那时 `say` 是空的，
            # 把它一起过滤掉就等于把"想说但被拦下了"整条信息丢掉，
            # 界面上只剩"没说话，但在忙自己的事"，而那是一句不实的话
            # （它其实什么都没做成）。
            if action.tool != "speak" or not result.ok
        ],
        "used_memories": [str(m) for m in (turn.used_memories or [])][:6],
        "violations": list(turn.persona_violations or []),
    }


def _memory_payload(cast: Cast, per_agent: int = 6) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for agent in cast.agents.values():
        stats = agent.memory.store.stats()
        records = list(getattr(agent.memory.store, "records", []))[-per_agent:]
        out[agent.persona.name] = {
            "episodic": stats.episodic,
            "semantic": stats.semantic,
            "reflection": stats.reflection,
            "consolidated": stats.consolidated,
            "recent": [f"[{r.kind}] {r.content}" for r in reversed(records)],
        }
    return out


def run_chat(cfg: RuntimeConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """重放一整段对话，返回每个事件产生的回合。

    **每次请求都从头重放**：客户端把全部事件发过来，服务端不存会话。
    好处是同一段输入必然得到同一段输出（可复现），刷新页面也不会错乱。
    """
    scenario_id = str(payload.get("scenario") or "tutorial")
    events = list(payload.get("events") or [])[:MAX_EVENTS]

    scenario = load_scenario(scenario_id)
    cast = build_cast(scenario, _make_llm(cfg), cfg)
    env = cast.env
    player_ids = [p.get("id") for p in (scenario.get("players") or [])]
    fallback_speaker = player_ids[0] if player_ids else "player_a"

    rendered: list[dict[str, Any]] = []
    for event in events:
        kind = str(event.get("kind") or "idle")
        utterance = None
        entry: dict[str, Any] = {"kind": kind}
        if kind == "say":
            text = str(event.get("text") or "").strip()
            if not text:
                continue
            speaker = str(event.get("speaker") or fallback_speaker)
            if player_ids and speaker not in player_ids:
                speaker = fallback_speaker
            utterance = env.record_player_utterance(speaker, text)
            entry["speaker_id"] = speaker
            entry["speaker_name"] = utterance.speaker_name
            entry["text"] = text
        entry["turns"] = [
            _turn_payload(turn, cast) for turn in cast.step(utterance)
        ]
        rendered.append(entry)

    snapshot = env.snapshot()
    npc_ids = set(cast.npc_ids)
    collisions = sum(
        1
        for _, speakers in env.speakers_by_tick().items()
        if len({s for s in speakers if s in npc_ids}) > 1
    )
    # 复读统计。**必须在控制台里看得见**：这个毛病是用户在这里发现的，
    # 而它最阴的地方是"每一句单看都没问题"—— 六维评测一条都抓不到
    # （每一维都只看单句）。不给一个数，就只能靠人一句句读。
    spoken = [
        (turn["name"], turn["say"])
        for event in rendered
        for turn in event["turns"]
        if turn["say"]
    ]
    repeat = find_repeats(spoken)

    # 计划来源统计。**必须在控制台里看得见**：规划调用失败会静默回落到
    # 启发式规划器，而两条路径产出的轨迹完全一样 —— 不显示来源的话，
    # "接上模型规划有没有用"这件事在页面上根本看不出来。
    # 实测（2026-09-19，231 条跑批）：48% 的用例至少回落过一次。
    plan_sources: dict[str, int] = {}
    for event in rendered:
        for turn in event["turns"]:
            source = turn.get("plan_source") or ""
            if source:
                plan_sources[source] = plan_sources.get(source, 0) + 1
    llm_enabled = bool(cfg.use_llm_planner)
    # ⚠️ 模型**可用**才谈得上"回落"。没配模型是"没开这一路"，不是失败 ——
    # 否则离线控制台一打开就会报一堆回落，而这正是"对照组被记成一片红"
    # 的老毛病换了个入口（`use_llm_planner` 默认是 True）。
    # 见 `Planner.plan_with_llm` 里那句"没配模型不是失败，是没开这一路"。
    llm_available = any(a.planner.llm.available for a in cast.agents.values())
    # 这两个是**原始输入**（配置怎么说 / 模型在不在），下面是**结论**。
    # ⚠️ 结论只算一遍，页面直接读它 —— 让页面自己写 `enabled && available`
    # 就是同一条规则的两份实现，哪天口径改了必然只有一边跟上（本项目有前科）。
    llm_active = bool(llm_enabled and llm_available)
    # 模型开着且真的可用，而计划来自启发式 ⇒ 模型被问过，但没给出可用计划。
    silent_fallbacks = plan_sources.get("heuristic", 0) if llm_active else 0

    return {
        "scenario": scenario_id,
        "world_label": env_label(env.name),
        "events": rendered,
        "snapshot": {
            "world_flags": list(snapshot.get("world_flags") or []),
            "objectives": dict(snapshot.get("objectives") or {}),
        },
        "memories": _memory_payload(cast),
        "speech": {
            cast.name_of(pid): count
            for pid, count in (env.speech_counts() or {}).items()
        },
        "collisions": collisions,
        "planner": {
            "llm_enabled": llm_enabled,
            "llm_available": llm_available,
            #: 模型规划这一路**真的活着**（配置开着 **且** 模型配了）。
            #: 判"是不是回落"只看这一个 —— 页面别再自己与一遍。
            "llm_active": llm_active,
            "sources": plan_sources,
            "silent_fallbacks": silent_fallbacks,
        },
        "repetition": {
            "total": repeat.total,
            "repeats": len(repeat.repeats),
            "rate": repeat.rate,
            "distinct": repeat.distinct,
            "examples": [
                {
                    "at": i + 1,
                    "collides_with": j + 1,
                    "text": text,
                    "similarity": sim,
                }
                for i, j, text, _prev, sim in repeat.repeats[:5]
            ],
        },
    }


# --------------------------------------------------------------------------- #
# 评测 / 变异
# --------------------------------------------------------------------------- #
def run_eval(cfg: RuntimeConfig, payload: dict[str, Any]) -> dict[str, Any]:
    import time

    category = str(payload.get("category") or "").strip()
    limit = max(0, int(payload.get("limit") or 0))

    started = time.perf_counter()
    with _RUN_LOCK:
        harness = EvalHarness(cfg)
        cases = harness.load_cases([category] if category else None)
        if limit:
            cases = cases[:limit]
        report = EvalReport()
        for case in cases:
            report.results.append(harness.run_case(case))
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    rows = []
    for result in report.results:
        data = result.to_dict()
        rows.append(
            {
                "id": data["case_id"],
                "category": data["category"],
                "description": data["description"],
                "passed": data["passed"],
                "scores": data["scores"],
                "reason": "；".join(data.get("notes") or []) or "—",
            }
        )
    return {
        "summary": report.to_dict()["summary"],
        "cases": rows,
        "elapsed_ms": elapsed_ms,
    }


#: 目标维度掉分小于这个数，就提示"覆盖面薄"。
#:
#: 这不是随口定的阈值：`0.007` 那次就是靠它被发现的 —— 一个本该最敏感的缺陷
#: 只让分数动了 0.007，而 `caught` 仍然是 `True`。
#: **「抓住了」是个布尔值，它不告诉你掉了多少。**
WEAK_DELTA = 0.05


def run_mutant(cfg: RuntimeConfig, payload: dict[str, Any]) -> dict[str, Any]:
    mutant_id = str(payload.get("mutant") or "").strip()
    category = str(payload.get("category") or "").strip()
    limit = max(0, int(payload.get("limit") or 0))

    mutants = tuple(m for m in MUTANTS if m.id == mutant_id)
    if not mutants:
        raise ValueError(
            f"未知的变异 {mutant_id!r}；可选：{[m.id for m in MUTANTS]}"
        )
    mutant = mutants[0]

    # 一把锁：注入改的是**类属性**，并发跑会让"哪个缺陷导致掉分"说不清。
    with _RUN_LOCK:
        available = len(EvalHarness(cfg).load_cases([category] if category else None))
        report = run_sensitivity(
            categories=[category] if category else None,
            limit=limit,
            config=cfg,
            mutants=(mutant,),
        )

    outcome = report.outcomes[0]
    # ⚠️ `targets` 在 `mutant` 上，不在 `outcome` 上。写错过一次，
    # 症状是 AttributeError —— 比"静默算错"好，但还是别写错。
    #
    # `weak` 只在**抓到**的前提下才算数：没抓到的时候"目标维度没动"是必然的，
    # 再说一遍"覆盖面薄"会把两个不同的结论搅在一起。
    weak = bool(
        outcome.caught
        and mutant.targets
        and abs(outcome.deltas.get(mutant.targets[0], 0.0)) < WEAK_DELTA
    )
    return {
        "id": mutant.id,
        "kind": mutant.kind,
        "targets": list(mutant.targets),
        "description": mutant.description,
        "total": report.total_cases,
        "baseline_pass": report.baseline_pass_rate,
        "mutant_pass": outcome.pass_rate,
        "baseline_means": report.baseline_means,
        "mutant_means": outcome.metric_means,
        "deltas": outcome.deltas,
        "caught": outcome.caught,
        "target_hit": outcome.target_hit,
        "weak": weak,
        # 跑了子集就不能报"抓没抓到" —— 前 80 条里没有回忆用例，
        # `retrieval_disabled` 会显示"没抓到"，那是**取样造成的**，不是评测的结论。
        "partial": report.total_cases < available,
    }


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "npc-agent-studio"

    #: 由 `serve_studio` / `start_in_thread` 注入。
    #:
    #: ⚠️ 这里**必须是真正的赋值**，不能只写类型标注。
    #: 只写 `config: RuntimeConfig` 是**注解**，类上根本没有这个属性 ——
    #: 而 `log_message` 在**每一次**响应里都会被调到（`send_response` 内部调它），
    #: 于是每个请求都以 `AttributeError` 收场，客户端只看到
    #: `RemoteDisconnected`（服务端把连接关了，什么都没回）。
    #: 这个坑很值得记：**注解不是默认值**，而"错误发生在日志函数里"会让
    #: 真正的病因完全看不见。
    config: RuntimeConfig = RuntimeConfig()

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        # 用 `getattr` 兜底：日志函数**永远不该**成为请求失败的原因。
        cfg = getattr(self, "config", None)
        if cfg is not None and getattr(cfg, "verbose", False):
            super().log_message(fmt, *args)

    # -- 工具 -- #
    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, message: str, status: int = 400) -> None:
        self._json({"error": message}, status)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"请求体不是合法 JSON：{exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("请求体必须是一个 JSON 对象")
        return data

    # -- 路由 -- #
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/meta":
            try:
                self._json(build_meta(self.config))
            except Exception as exc:  # noqa: BLE001 — 报给页面，别让连接挂掉
                self._error(f"初始化失败：{exc}", 500)
            return
        if path.startswith("/report/"):
            self._serve_report(path[len("/report/") :])
            return
        self._error("没有这个路由", 404)

    def _serve_report(self, name: str) -> None:
        """只服务 `docs/` 下的 .html，且**必须**落在 docs/ 里面。

        白名单不是洁癖：这个服务会把目录内容回给浏览器，
        拼路径时不校验就等于把 `../` 交给调用方。
        """
        if not name.endswith(".html") or "/" in name or "\\" in name or name.startswith("."):
            self._error("报告名不合法", 400)
            return
        target = (DOCS_DIR / name).resolve()
        if DOCS_DIR.resolve() not in target.parents or not target.is_file():
            self._error(f"找不到报告 {name}", 404)
            return
        self._send(200, target.read_bytes(), "text/html; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        routes = {
            "/api/chat": run_chat,
            "/api/eval": run_eval,
            "/api/mutant": run_mutant,
        }
        handler = routes.get(path)
        if handler is None:
            self._error("没有这个路由", 404)
            return
        try:
            payload = self._read_json()
        except ValueError as exc:
            self._error(str(exc))
            return
        try:
            self._json(handler(self.config, payload))
        except ValueError as exc:
            # 参数错是**用户**的问题，用 400 说清楚。
            self._error(str(exc), 400)
        except Exception as exc:  # noqa: BLE001 — 500 也要回 JSON，否则前端只能看到 HTML
            self._error(f"{type(exc).__name__}: {exc}", 500)


# --------------------------------------------------------------------------- #
def _pick_port(host: str, port: int, tries: int = 20) -> ThreadingHTTPServer:
    last: OSError | None = None
    for offset in range(tries):
        try:
            return ThreadingHTTPServer((host, port + offset), _Handler)
        except OSError as exc:
            last = exc
    raise OSError(f"{host}:{port}~{port + tries - 1} 都占着：{last}")


def start_in_thread(
    host: str = "127.0.0.1", port: int = 0, **overrides: Any
) -> tuple[ThreadingHTTPServer, str]:
    """起一个后台服务，返回 `(httpd, base_url)`。给测试和 `--port 0` 用。

    `port=0` 让系统挑一个空闲端口 —— 测试里**必须**这样，
    写死端口在并行跑测试时会互相抢。
    """
    cfg = _config(overrides)
    _Handler.config = cfg
    httpd = _pick_port(host, port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://{host}:{httpd.server_address[1]}"


def serve_studio(
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    **overrides: Any,
) -> int:
    """起控制台。返回进程退出码（`serve_forever` 被 Ctrl-C 打断时返回 0）。"""
    cfg = _config(overrides)
    _Handler.config = cfg
    # 端口被占就顺延。绑 0 交给系统挑也可以（`--port 0`），那时不打顺延日志。
    httpd = _pick_port(host, port)
    actual = httpd.server_address[1]
    url = f"http://{host}:{actual}/"

    mode = "在线模型" if _make_llm(cfg).available else "离线启发式（不调模型）"
    print(f"NPC Agent Studio  →  {url}")
    print(f"  推理模式：{mode}")
    print(f"  报告目录：{DOCS_DIR}")
    if actual != port:
        print(f"  （{port} 被占用，顺延到 {actual}）")
    print("  Ctrl-C 退出")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 — 无头环境里打不开浏览器不该让服务起不来
            print("  （打不开浏览器，手动访问上面的地址即可）")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        httpd.server_close()
    return 0


def free_port(host: str = "127.0.0.1") -> int:
    """要一个空闲端口。测试和 `--port 0` 都用它。"""
    with socket.socket() as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


__all__ = [
    "MAX_EVENTS",
    "WEAK_DELTA",
    "build_meta",
    "free_port",
    "run_chat",
    "run_eval",
    "run_mutant",
    "serve_studio",
    "start_in_thread",
]
