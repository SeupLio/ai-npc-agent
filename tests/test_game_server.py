"""游戏接入层的护栏。

## 这里在守什么

这是"框架"变"服务"的一层 —— Unity / Godot / Web 客户端从这里接入。
契约破了，游戏端的表现是"整个 NPC 系统坏了"，不是"某个功能怪"：

1. **会话隔离**：A 服的玩家不该听见 B 服的悄悄话（各自独立的 Cast）；
2. **错误方向安全**：私聊收件人写错必须报 400，不能退化成公开广播；
3. **存档跟着人走**：换一个会话（玩家断线重连）必须还能读回记忆 ——
   路径跟着 owner 走，不跟会话 id 走；
4. **成本可见**：每次 /say 的响应里都带遥测，接游戏的团队不需要
   另起一套监控才能算账。
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from npc_agent.game_server import GameServer


@pytest.fixture()
def server(tmp_path):
    server = GameServer(save_dir=tmp_path / "saves")
    yield server
    server.close()


@pytest.fixture()
def http_server(server):
    # 端口 0 = 系统挑一个空闲的，测试之间互不抢
    thread = threading.Thread(
        target=lambda: server.serve("127.0.0.1", 0), daemon=True
    )
    # serve(port=0) 会在 ThreadingHTTPServer 里拿到实际端口
    thread.start()
    for _ in range(50):
        if server._httpd is not None:
            break
        time.sleep(0.05)
    port = server._httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"
    yield server, base
    server.close()


def _req(base, method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"{base}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


# --------------------------------------------------------------------------- #
# 核心逻辑（不起 HTTP）
# --------------------------------------------------------------------------- #
class TestSessionLogic:
    def test_player_rename_propagates_to_utterances(self, server):
        session = server.create_session(
            "duet", player_names={"player_a": "铁柱"}, owner="user_42"
        )
        out = session.player_say("player_a", "阿柚！！出大事了！！")
        text = " ".join(t["say"] or "" for t in out["turns"])
        assert "铁柱" in text
        assert "player_a" not in text  # 槽位 id 不该漏进台词

    def test_unknown_player_slot_rejected(self, server):
        session = server.create_session("duet")
        with pytest.raises(ValueError, match="玩家槽位"):
            session.player_say("player_z", "hi")

    def test_empty_text_rejected(self, server):
        session = server.create_session("duet")
        with pytest.raises(ValueError, match="text"):
            session.player_say("player_a", "  ")

    def test_private_with_wrong_target_rejected_not_broadcast(self, server):
        session = server.create_session("duet")
        with pytest.raises(ValueError, match="私聊收件人"):
            session.player_say(
                "player_a", "悄悄话", target="ghost", channel="private"
            )

    def test_private_reply_stays_private(self, server):
        session = server.create_session("duet", owner="user_42")
        out = session.player_say(
            "player_a", "偷偷问一句，你们打烊是几点？",
            target="ayou", channel="private",
        )
        spoken = [t for t in out["turns"] if t["say"]]
        assert spoken
        assert all(t["channel"] == "private" for t in spoken)

    def test_sessions_are_isolated_worlds(self, server):
        a = server.create_session("duet", owner="user_a")
        b = server.create_session("duet", owner="user_b")
        a.player_say("player_a", "以后就叫我小鹿吧")
        # A 世界的约定不影响 B 世界 —— B 里这个词还没被占
        assert not b.cast.agents["ayou"].relationships.agreements

    def test_save_follows_owner_not_session(self, server, tmp_path):
        first = server.create_session("duet", owner="user_42")
        first.player_say("player_a", "阿柚，以后就叫我小鹿吧")
        saved = first.save("main")

        # 玩家"断线重连"：新会话，同一个 owner
        second = server.create_session("duet", owner="user_42")
        second.load("main")
        reply = second.player_say("player_b", "阿柚，也叫我小鹿吧")
        text = " ".join(t["say"] or "" for t in reply["turns"])
        # 约定跨会话仍然排他
        assert "小鹿" in text
        assert "不行" in text or "有人用了" in text
        # 存档路径可预测（owner 维度），且不含会话 id
        assert "user_42" in str(saved["path"])
        assert first.id not in str(saved["path"])

    def test_telemetry_in_every_round(self, server):
        session = server.create_session("duet")
        out = session.player_say("player_a", "你们有什么推荐？")
        assert out["telemetry"]["rounds"] == 1
        assert out["telemetry"]["turns"] >= 1
        idle = session.idle()
        assert idle["telemetry"]["rounds"] == 2

    def test_events_stream_increments(self, server):
        session = server.create_session("duet")
        session.player_say("player_a", "hello")
        session.player_say("player_a", "again")
        events = session.events_since(0)
        types = [e["type"] for e in events]
        assert types.count("player_said") == 2
        # 增量拉取：只要第二条
        tail = session.events_since(events[-2]["seq"])
        assert len(tail) == 1


# --------------------------------------------------------------------------- #
# HTTP 层
# --------------------------------------------------------------------------- #
class TestHTTPEndpoints:
    def test_health_and_scenarios(self, http_server):
        _, base = http_server
        code, health = _req(base, "GET", "/v1/health")
        assert code == 200 and health["ok"]
        assert "duet" in health["scenarios"]

        code, scenarios = _req(base, "GET", "/v1/scenarios")
        assert code == 200
        ids = [s["id"] for s in scenarios]
        assert "tutorial" in ids and "quartet" in ids

    def test_full_round_trip(self, http_server):
        _, base = http_server
        code, state = _req(base, "POST", "/v1/sessions", {"scenario": "duet"})
        assert code == 201
        sid = state["session_id"]
        assert {n["name"] for n in state["npcs"]} == {"阿柚", "小舟"}

        code, out = _req(
            base, "POST", f"/v1/sessions/{sid}/say",
            {"player_id": "player_a", "text": "阿柚！！出大事了！！"},
        )
        assert code == 200
        assert out["telemetry"]["rounds"] == 1

        code, idle = _req(base, "POST", f"/v1/sessions/{sid}/idle", {})
        assert code == 200

        code, events = _req(base, "GET", f"/v1/sessions/{sid}/events?since=0")
        assert code == 200
        assert {e["type"] for e in events["events"]} >= {"session_open", "player_said"}

    def test_unknown_session_is_404(self, http_server):
        _, base = http_server
        code, _ = _req(base, "GET", "/v1/sessions/nope")
        assert code == 404
        code, _ = _req(base, "POST", "/v1/sessions/nope/say", {"player_id": "player_a", "text": "x"})
        assert code == 404

    def test_bad_input_is_400_not_500(self, http_server):
        _, base = http_server
        _, state = _req(base, "POST", "/v1/sessions", {"scenario": "duet"})
        sid = state["session_id"]
        code, _ = _req(
            base, "POST", f"/v1/sessions/{sid}/say",
            {"player_id": "player_z", "text": "hi"},
        )
        assert code == 400
        # 私聊收件人写错：必须失败，而不是广播
        code, _ = _req(
            base, "POST", f"/v1/sessions/{sid}/say",
            {"player_id": "player_a", "text": "hi", "target": "ghost", "channel": "private"},
        )
        assert code == 400

    def test_unknown_route_is_404(self, http_server):
        _, base = http_server
        code, _ = _req(base, "GET", "/v1/whatever")
        assert code == 404

    def test_delete_closes_session(self, http_server):
        _, base = http_server
        _, state = _req(base, "POST", "/v1/sessions", {"scenario": "duet"})
        sid = state["session_id"]
        code, _ = _req(base, "DELETE", f"/v1/sessions/{sid}")
        assert code == 200
        code, _ = _req(base, "GET", f"/v1/sessions/{sid}")
        assert code == 404

    def test_save_load_over_http(self, http_server):
        server, base = http_server
        _, state = _req(base, "POST", "/v1/sessions", {"scenario": "duet", "owner": "u1"})
        sid = state["session_id"]
        _req(base, "POST", f"/v1/sessions/{sid}/say",
             {"player_id": "player_a", "text": "阿柚，以后就叫我小鹿吧"})
        code, saved = _req(base, "POST", f"/v1/sessions/{sid}/save", {"slot": "main"})
        assert code == 200 and saved["path"]

        # 新会话（重连）读档
        _, state2 = _req(base, "POST", "/v1/sessions", {"scenario": "duet", "owner": "u1"})
        sid2 = state2["session_id"]
        code, _ = _req(base, "POST", f"/v1/sessions/{sid2}/load", {"slot": "main"})
        assert code == 200
