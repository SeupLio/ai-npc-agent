"""游戏接入层 —— 让 Unity / Godot / Web 客户端真正接得上。

## 为什么这个模块必须存在

此前的所有入口（demo / chat / studio）都是**给人看的**：终端、控制台网页。
但真实游戏在引擎里 —— Unity 是 C#，Godot 是 GDScript，它们不跑 Python。
一个 NPC 框架如果不能被引擎进程调用，就永远停在"技术演示"，
这是从"可评测的框架"到"可上线的组件"之间唯一的一道墙。

推倒这堵墙的最小做法：把 Cast 包成一个**面向游戏客户端的 HTTP 服务**。
选 HTTP + JSON 而不是 gRPC / WebSocket：

- 每个引擎都有 HTTP 客户端，零 SDK 成本；
- 请求/响应模式（POST /say → 拿到本轮 NPC 台词与动作）覆盖对话主循环；
- 轮询（GET /events?since=）覆盖"别人的回合我也要看到"；
- 不需要长连接的服务端复杂度 —— 这一层留给游戏引擎自己的网络栈。

## 会话模型

    POST /v1/sessions {scenario}            一个会话 = 一个世界副本
    POST /v1/sessions/{sid}/say             玩家发言（支持点名/分组/私聊）
    POST /v1/sessions/{sid}/idle            冷场推一格（世界自己转）
    GET  /v1/sessions/{sid}                 NPC 状态/关系/编年史/成本
    GET  /v1/sessions/{sid}/events?since=N  事件流（增量）
    POST /v1/sessions/{sid}/save /load      跨会话记忆（按存档槽）
    DELETE /v1/sessions/{sid}               收摊

**每个会话独立一个 Cast、独立一把锁**：会话之间互不可见（A 服的玩家
不该听见 B 服的悄悄话）；会话内串行（一轮没跑完，下一轮排队 ——
发言权轮转本来就不支持并发发言）。

## 内容热更新（42 天版本节奏的现实答案）

`POST /v1/sessions` 每次都**重新读场景 YAML**。改了人设/设定书/场景，
新会话立刻生效，老会话继续用开服时的内容 —— 这正是内容团队要的
语义："更新不影响进行中的对局"。不需要专门的 reload 端点，
也不需要停服。

## 成本可见（工程三关之一）

每个 /say 和 /idle 的响应都带 `telemetry`：真实 token 用量、
p50/p95 延迟、按单价折算的会话成本、每百轮推算。接游戏的团队
不需要读代码就知道这套 NPC 一天要烧多少钱。

## 边界（写清楚，别让接入方猜）

- **无鉴权**：默认绑定 127.0.0.1。要暴露给引擎服务器，
  自己挂反向代理 + 鉴权，这是网关层的事，不该 NPC 服务管。
- **单进程**：会话存内存，进程重启即消失；跨进程用 /save /load。
- **玩家槽位**：游戏客户端把自家的 user id 映射到场景的
  player_a / player_b 槽位上（建会话时可改名）。引擎侧的账号体系
  引擎自己管，NPC 只关心"这场里有几个人、谁在说话"。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from .cast import Cast, build_cast
from .config import RuntimeConfig, list_scenarios, load_scenario
from .llm import build_llm
from .modules.persistence import load as persistence_load
from .modules.persistence import save as persistence_save
from .modules.telemetry import CostProfile, SessionTelemetry

#: 事件日志每会话的软上限。超了就从头裁 —— 事件流是给客户端增量拉取的，
#: 拉过的旧事件没有第二次消费价值，留着只是内存泄漏的种子。
MAX_EVENTS = 2000

#: 单轮最长执行时间（秒）。离线路径毫秒级；LLM 路径一次调用可达数十秒
#: （`openai_compat` 的读超时是 180s）。护在这里是为了**别的会话**：
#: ThreadingHTTPServer 一个请求一个线程，一个卡死的会话不该把
#: 整个进程的线程池拖穿。
ROUND_TIMEOUT_SECONDS = 600.0


def _make_llm(cfg: RuntimeConfig):
    """每个会话一个 LLM 实例 —— 用量归因的前提。

    共享实例的话 `llm.usage` 是全局累计的，A 会话烧的 token 会算进
    B 会话的成本里；成本画像对上线决策是硬数字，归因错了比没有更糟。
    """
    return build_llm(
        cfg.llm_provider,
        model=cfg.model,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        timeout=cfg.llm_timeout,
        retries=cfg.llm_retries,
        parse_retries=cfg.llm_parse_retries,
    )


class GameSession:
    """一个会话：独立世界 + 独立锁 + 独立遥测。"""

    def __init__(
        self,
        scenario_id: str,
        cfg: RuntimeConfig,
        save_dir: Path,
        player_names: Optional[dict[str, str]] = None,
        owner: str = "",
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.scenario_id = scenario_id
        self.cfg = cfg
        self.save_dir = save_dir
        self.created_at = time.time()
        self.lock = threading.Lock()
        self.event_seq = 0
        self.events: list[dict[str, Any]] = []

        # 每次建会话重新读 YAML —— 内容热更新就靠这一行（见模块注释）。
        scenario = load_scenario(scenario_id)
        self.llm = _make_llm(cfg)
        self.cast: Cast = build_cast(scenario, self.llm, cfg)
        self.telemetry = SessionTelemetry(CostProfile.from_env())
        self.player_slots: list[str] = [
            str(p.get("id")) for p in (scenario.get("players") or [])
        ]
        # 存档归属：引擎侧的用户标识（user_42 之类）。不传就退到第一个
        # 玩家槽位 —— 存档路径跟着**人**走，不跟会话走（见 `_save_path`）。
        self.owner = owner or (self.player_slots[0] if self.player_slots else "anon")

        # 把引擎侧的玩家名映射到场景槽位。改名发生在**任何交互之前**，
        # 之后所有 Utterance.speaker_name 都走新名字 —— 不需要通知
        # 每个模块"玩家改名了"，它们本来就该读环境，而不是缓存名字。
        for slot, name in (player_names or {}).items():
            actor = self.cast.env.actors.get(slot)
            if actor is not None and name:
                actor.name = str(name)

        self._log("session_open", {"scenario": scenario_id})

    # ------------------------------------------------------------------ #
    # 轮次执行
    # ------------------------------------------------------------------ #
    def player_say(
        self,
        player_id: str,
        text: str,
        target: str = "",
        channel: str = "public",
    ) -> dict[str, Any]:
        """玩家发言 → 跑一轮 → 返回这一轮的 NPC 回合与遥测。"""
        env = self.cast.env
        if player_id not in self.player_slots:
            raise ValueError(
                f"玩家槽位 {player_id} 不存在（这场是：{self.player_slots}）"
            )
        text = str(text or "").strip()
        if not text:
            raise ValueError("text 不能为空")

        kwargs: dict[str, Any] = {}
        if channel == "private" and target:
            # 收件人必须是场上真实存在的演员：写错的私聊宁可整轮没人接，
            # 也不能退化成公开（见 studio 的同一判据 —— 两处必须一致，
            # 不然控制台和服务端对同一句话给出不同行为）。
            if target not in env.actors:
                raise ValueError(f"私聊收件人 {target} 不在场上")
            kwargs["audience"] = (player_id, target)
            kwargs["channel"] = "private"
            kwargs["addressed_to"] = target
        elif target:
            if target not in env.actors:
                raise ValueError(f"点名对象 {target} 不在场上")
            kwargs["addressed_to"] = target

        started = time.perf_counter()
        utterance = env.record_player_utterance(player_id, text, **kwargs)
        turns, spoken_by = self._step(utterance)
        elapsed = (time.perf_counter() - started) * 1000.0

        self._note_round(turns, elapsed)
        entry = {
            "kind": "player",
            "player_id": player_id,
            "text": text,
            "channel": kwargs.get("channel", "public"),
            "turns": [self._turn_payload(t, spoken_by.get(t.actor_id)) for t in turns],
        }
        self._log("player_said", entry)
        return self._round_response(entry, turns)

    def idle(self) -> dict[str, Any]:
        """不说话，推世界一格：NPC 主动行为（补答/私语/搭话）发生在这里。"""
        started = time.perf_counter()
        turns, spoken_by = self._step()
        elapsed = (time.perf_counter() - started) * 1000.0
        self._note_round(turns, elapsed)
        entry = {"kind": "idle", "turns": [self._turn_payload(t, spoken_by.get(t.actor_id)) for t in turns]}
        self._log("idle", entry)
        return self._round_response(entry, turns)

    def _step(self, utterance: Any = None) -> tuple[list[Any], dict[str, Any]]:
        """跑一轮，并按说话人配对"真正落进世界的那句话"。

        频道信息必须来自世界里的 utterance，而不是从 ActionCall 猜 ——
        分组场景里 NPC 台词的 channel 是导演在 `_stamp_scope` 盖的章，
        私语是 speak 工具盖的章，回合对象本身两样都不知道。
        （studio 用同一招，两处行为必须一致。）
        """
        before = len(self.cast.env.utterances)
        turns = self.cast.step(utterance)
        spoken_by = {
            line.speaker_id: line for line in self.cast.env.utterances[before:]
        }
        return turns, spoken_by

    # ------------------------------------------------------------------ #
    # 状态与存档
    # ------------------------------------------------------------------ #
    def state(self) -> dict[str, Any]:
        env = self.cast.env
        snapshot = env.snapshot()
        return {
            "session_id": self.id,
            "scenario": self.scenario_id,
            "tick": env.tick,
            "rounds": self.cast.rounds,
            "players": [
                {
                    "id": slot,
                    "name": env.actors[slot].name,
                    "affinity": {
                        npc_id: env.actors[slot].affinity_to(npc_id)
                        for npc_id in self.cast.agents
                    },
                }
                for slot in self.player_slots
            ],
            "npcs": [self._npc_payload(a) for a in self.cast.agents.values()],
            "world": {
                "flags": list(snapshot.get("world_flags") or []),
                "objects": snapshot.get("objects") or {},
            },
            "telemetry": self.telemetry.snapshot(),
        }

    def save(self, slot: str = "default") -> dict[str, Any]:
        path = self._save_path(slot)
        persistence_save(self.cast, path)
        self._log("saved", {"slot": slot, "path": str(path)})
        return {"slot": slot, "path": str(path)}

    def load(self, slot: str = "default") -> dict[str, Any]:
        path = self._save_path(slot)
        if not path.exists():
            raise FileNotFoundError(f"存档槽 {slot} 不存在（{path}）")
        persistence_load(self.cast, path)
        self._log("loaded", {"slot": slot})
        return {"slot": slot, "npcs": [self._npc_payload(a) for a in self.cast.agents.values()]}

    def events_since(self, since: int) -> list[dict[str, Any]]:
        return [e for e in self.events if e["seq"] > since]

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    def _save_path(self, slot: str) -> Path:
        # 路径跟着**人**走（scenario + owner + slot），不跟会话走。
        # 曾用会话 id：新会话永远读不到旧档，"跨会话记忆"当场变假。
        safe = "".join(c for c in str(slot) if c.isalnum() or c in "-_") or "default"
        owner = "".join(c for c in str(self.owner) if c.isalnum() or c in "-_") or "anon"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        return self.save_dir / f"{self.scenario_id}_{owner}_{safe}.json"

    def _note_round(self, turns: list[Any], elapsed_ms: float) -> None:
        self.telemetry.note_round(elapsed_ms)
        for turn in turns:
            self.telemetry.note_turn(turn.latency_ms, spoken=bool(turn.say))
        # 累计值替换语义（见 telemetry.adopt_llm_usage）：每轮之后同步一次。
        self.telemetry.adopt_llm_usage(self.llm)

    def _round_response(self, entry: dict[str, Any], turns: list[Any]) -> dict[str, Any]:
        return {
            "session_id": self.id,
            "entry": entry,
            "turns": entry["turns"],
            "telemetry": self.telemetry.snapshot(),
            "any_reply": any(bool(t.say) for t in turns),
        }

    def _npc_payload(self, agent: Any) -> dict[str, Any]:
        relationships = {}
        for pid, bond in agent.relationships.bonds.items():
            name = (
                agent.state.players[pid].name
                if pid in agent.state.players
                else pid
            )
            relationships[name] = {
                "stage": bond.stage,
                "score": bond.score,
                "titles": list(bond.titles),
                "terms": list(bond.terms),
            }
        return {
            "id": agent.id,
            "name": agent.persona.name,
            "role": agent.persona.role,
            "intent_log": list(getattr(agent, "_intent_log", [])[-12:]),
            "relationships": relationships,
            "memory_stats": agent.memory.store.stats().__dict__
            if hasattr(agent.memory.store.stats(), "__dict__")
            else {},
        }

    def _turn_payload(self, turn: Any, spoken: Any = None) -> dict[str, Any]:
        """一个 NPC 回合 → 引擎能直接播的载荷。

        和 studio 的 `_turn_payload` 刻意保持同构（字段名一致），
        但**不再复用**：studio 面向调试页面（要 violations、used_memories），
        这里面向引擎（要 addressed_to、channel、latency）。两边消费方
        不同，硬塞进一个函数会让两边都背着对方用不到的字段。
        """
        channel = getattr(spoken, "channel", "") or ""
        return {
            "actor_id": turn.actor_id,
            "name": self.cast.name_of(turn.actor_id),
            "say": turn.say,
            "addressed_to": turn.addressed_to or "",
            "channel": channel,
            "actions": [
                {"tool": action.tool, "args": dict(action.args), "ok": result.ok}
                for action, result in zip(turn.actions, turn.results)
            ],
            "decision_reason": turn.decision_reason,
            "latency_ms": round(turn.latency_ms, 1),
        }

    def _log(self, type_: str, entry: dict[str, Any]) -> None:
        self.event_seq += 1
        self.events.append({"seq": self.event_seq, "type": type_, "t": time.time(), **entry})
        if len(self.events) > MAX_EVENTS:
            del self.events[:-MAX_EVENTS]


class GameServer:
    """会话注册表 + HTTP 服务。"""

    def __init__(
        self,
        cfg: RuntimeConfig | None = None,
        save_dir: str | Path = "saves",
    ) -> None:
        self.cfg = cfg or RuntimeConfig.from_env()
        self.save_dir = Path(save_dir)
        self.sessions: dict[str, GameSession] = {}
        self._registry_lock = threading.Lock()
        self._httpd: Optional[ThreadingHTTPServer] = None

    # ------------------------------------------------------------------ #
    def create_session(
        self,
        scenario_id: str = "tutorial",
        player_names: Optional[dict[str, str]] = None,
        owner: str = "",
    ) -> GameSession:
        if scenario_id not in list_scenarios():
            raise ValueError(
                f"未知场景 {scenario_id}（可选：{', '.join(list_scenarios())}）"
            )
        session = GameSession(
            scenario_id, self.cfg, self.save_dir, player_names, owner=owner
        )
        with self._registry_lock:
            self.sessions[session.id] = session
        return session

    def close(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None

    # ------------------------------------------------------------------ #
    def serve(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            """一个路由方法一个端点，`_reply` 统一 JSON 出口。

            错误也走 JSON（`{"error": ...}`）而不是 HTML 错误页 ——
            引擎侧的 HTTP 栈拿到 HTML 会当成不可解析的 500，
            JSON 至少能让它把原因打进日志。
            """

            def log_message(self, fmt, *args):  # noqa: N802
                # 默认实现往 stderr 印访问日志，服务化之后那是日志系统的地盘。
                pass

            # ---------------- GET ---------------- #
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                parts = [p for p in parsed.path.split("/") if p]
                query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                try:
                    if parts == ["v1", "health"]:
                        self._reply(
                            200,
                            {
                                "ok": True,
                                "sessions": len(server.sessions),
                                "provider": server.cfg.llm_provider,
                                "scenarios": list_scenarios(),
                            },
                        )
                    elif parts == ["v1", "scenarios"]:
                        self._reply(200, self._scenarios())
                    elif len(parts) == 3 and parts[:2] == ["v1", "sessions"]:
                        session = server._session(parts[2])
                        self._reply(200, session.state())
                    elif len(parts) == 4 and parts[:2] == ["v1", "sessions"] and parts[3] == "events":
                        session = server._session(parts[2])
                        since = int(query.get("since", 0))
                        self._reply(200, {"events": session.events_since(since)})
                    else:
                        self._reply(404, {"error": f"未知路径 {self.path}"})
                except Exception as exc:
                    self._error(exc)

            # ---------------- POST / DELETE ---------------- #
            def do_POST(self) -> None:  # noqa: N802
                parts = [p for p in urlparse(self.path).path.split("/") if p]
                body = self._read_body()
                try:
                    if parts == ["v1", "sessions"]:
                        session = server.create_session(
                            str(body.get("scenario") or "tutorial"),
                            body.get("players") or None,
                            str(body.get("owner") or ""),
                        )
                        self._reply(201, session.state())
                    elif len(parts) == 4 and parts[:2] == ["v1", "sessions"]:
                        session = server._session(parts[2])
                        with session.lock:
                            action = parts[3]
                            if action == "say":
                                out = session.player_say(
                                    str(body.get("player_id") or ""),
                                    str(body.get("text") or ""),
                                    str(body.get("target") or ""),
                                    str(body.get("channel") or "public"),
                                )
                            elif action == "idle":
                                out = session.idle()
                            elif action == "save":
                                out = session.save(str(body.get("slot") or "default"))
                            elif action == "load":
                                out = session.load(str(body.get("slot") or "default"))
                            else:
                                self._reply(404, {"error": f"未知操作 {action}"})
                                return
                        self._reply(200, out)
                    else:
                        self._reply(404, {"error": f"未知路径 {self.path}"})
                except Exception as exc:
                    self._error(exc)

            def do_DELETE(self) -> None:  # noqa: N802
                parts = [p for p in urlparse(self.path).path.split("/") if p]
                try:
                    if len(parts) == 3 and parts[:2] == ["v1", "sessions"]:
                        sid = parts[2]
                        with server._registry_lock:
                            session = server.sessions.pop(sid, None)
                        if session is None:
                            raise KeyError(f"会话 {sid} 不存在")
                        with session.lock:
                            session._log("session_closed", {})
                        self._reply(200, {"closed": sid})
                    else:
                        self._reply(404, {"error": f"未知路径 {self.path}"})
                except Exception as exc:
                    self._error(exc)

            # ---------------- 工具 ---------------- #
            def _scenarios(self) -> list[dict[str, Any]]:
                out = []
                for sid in list_scenarios():
                    scenario = load_scenario(sid)
                    out.append(
                        {
                            "id": sid,
                            "name": scenario.get("name") or sid,
                            "description": scenario.get("description") or "",
                            "players": [
                                {"id": p.get("id"), "name": p.get("name") or p.get("id")}
                                for p in (scenario.get("players") or [])
                            ],
                        }
                    )
                return out

            def _read_body(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0:
                    return {}
                raw = self.rfile.read(length)
                try:
                    data = json.loads(raw.decode("utf-8"))
                    return data if isinstance(data, dict) else {}
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return {}

            def _reply(self, code: int, payload: dict[str, Any]) -> None:
                data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _error(self, exc: Exception) -> None:
                code = 404 if isinstance(exc, (KeyError, FileNotFoundError)) else 400
                self._reply(code, {"error": f"{type(exc).__name__}: {exc}"})

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._httpd.serve_forever()

    # ------------------------------------------------------------------ #
    def _session(self, sid: str) -> GameSession:
        with self._registry_lock:
            session = self.sessions.get(sid)
        if session is None:
            raise KeyError(f"会话 {sid} 不存在")
        return session


def serve(host: str = "127.0.0.1", port: int = 8080, save_dir: str | Path = "saves") -> None:
    """CLI 入口：`python -m npc_agent.cli serve`。"""
    server = GameServer(save_dir=save_dir)
    print(f"ai-npc-agent 游戏服务已启动: http://{host}:{port}")
    print("  文档: docs/GAME_SERVER.md    健康检查: GET /v1/health")
    try:
        server.serve(host, port)
    except KeyboardInterrupt:
        print("\n已停止")


__all__ = ["GameServer", "GameSession", "serve"]
