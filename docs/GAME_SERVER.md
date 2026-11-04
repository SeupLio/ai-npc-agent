# 游戏接入指南（GameServer HTTP API）

> 这一页回答一个问题：**Unity / Godot / Web 客户端怎么把这套 NPC 接进自己的游戏。**
> 框架本体是 Python，你的游戏引擎不是 —— 这层 HTTP API 就是给两边接的桥。

```bash
python -m npc_agent.cli serve --host 0.0.0.0 --port 8080
```

不配任何模型也能跑（离线启发式路径，零成本零延迟起步）。
接真实模型：

```bash
NPC_AGENT_PROVIDER=openai-compat \
NPC_AGENT_BASE_URL=https://api.deepseek.com/v1 \
NPC_AGENT_API_KEY=sk-xxx \
NPC_AGENT_MODEL=deepseek-chat \
python -m npc_agent.cli serve --port 8080
```

---

## 一、接入的最短路径（5 分钟）

游戏里 NPC 对话的主循环只有一句话：**玩家说话 → POST /say → 播出返回的台词**。

```text
1. POST /v1/sessions          开一个会话（= 一个世界副本）
2. POST /v1/sessions/{id}/say 玩家说话，拿 NPC 回合（台词+动作+延迟）
3. GET  /v1/sessions/{id}     要状态（关系/好感/成本）就随时查
4. POST /v1/sessions/{id}/save 玩家下线时存档
5. POST /v1/sessions/{id}/load 玩家回线时读档（NPC 还记得他）
```

### Unity (C#) 最小样例

```csharp
// 建会话（开场时调一次）
var http = new HttpClient();
var open = await http.PostAsync("http://localhost:8080/v1/sessions",
    new StringContent(@"{""scenario"":""tutorial"",""owner"":""" + userId + @"""}",
    Encoding.UTF8, "application/json"));
var session = JsonNode.Parse(await open.Content.ReadAsStringAsync());
string sid = (string)session["session_id"];

// 玩家说话（每帧对话调用）
var say = await http.PostAsync($"http://localhost:8080/v1/sessions/{sid}/say",
    new StringContent(@"{""player_id"":""player_a"",""text"":""你们这儿有什么推荐？""}",
    Encoding.UTF8, "application/json"));
var result = JsonNode.Parse(await say.Content.ReadAsStringAsync());
foreach (var turn in (JsonArray)result["turns"]) {
    if ((string)turn["say"] != null)
        Debug.Log($"{turn["name"]}: {turn["say"]}");   // 播台词
    foreach (var action in (JsonArray)turn["actions"])
        Debug.Log($"动作: {action["tool"]}");          // 走行为树/动画
}
```

### Godot (GDScript) 最小样例

```gdscript
func talk(text: String) -> void:
    var body := JSON.stringify({"player_id": "player_a", "text": text})
    var resp := await _http.request(
        "http://localhost:8080/v1/sessions/%s/say" % session_id,
        ["Content-Type: application/json"],
        HTTPClient.METHOD_POST, body)
    var data = JSON.parse_string(_http.get_request_body_string() if false else body)
```

（GDScript 侧用 `HTTPRequest` 节点 POST 同样的 JSON 即可，字段一致。）

---

## 二、端点契约

所有请求/响应都是 JSON。错误统一返回 4xx + `{"error": "..."}`。

### 会话

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/v1/health` | 存活检查，带 provider 与场景清单 |
| `GET` | `/v1/scenarios` | 可用场景（NPC、玩家槽位、分组） |
| `POST` | `/v1/sessions` | 建会话，body 见下 |
| `GET` | `/v1/sessions/{sid}` | 状态快照（含 telemetry） |
| `DELETE` | `/v1/sessions/{sid}` | 收摊 |

`POST /v1/sessions` 请求体：

```json
{
  "scenario": "duet",
  "owner": "user_42",
  "players": {"player_a": "铁柱", "player_b": "小满"}
}
```

- `scenario`：场景 id（见 `/v1/scenarios`），默认 `tutorial`；
- `owner`：**你游戏侧的用户 id** —— 存档按它落盘，换会话（断线重连）记忆不丢；
- `players`：把场景的玩家槽位改成你游戏里的玩家名。NPC 喊的就是这个名字。

### 对话

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/v1/sessions/{sid}/say` | 玩家发言 |
| `POST` | `/v1/sessions/{sid}/idle` | 冷场推一格（NPC 主动行为发生地） |
| `GET` | `/v1/sessions/{sid}/events?since=N` | 增量事件流 |

`POST /say` 请求体：

```json
{
  "player_id": "player_a",
  "text": "阿柚，偷偷问下打烊几点？",
  "target": "ayou",
  "channel": "private"
}
```

- `channel`：`public`（默认，全场听见）/ `private`（小窗，**必须带 target**）；
- `target`：点名的对象（NPC id 或玩家 id）。

响应（节选）：

```json
{
  "entry": {"kind": "player", "turns": [...]},
  "turns": [
    {
      "actor_id": "ayou",
      "name": "阿柚",
      "say": "（压低声音）……",
      "addressed_to": "player_a",
      "channel": "private",
      "actions": [{"tool": "speak", "args": {...}, "ok": true}],
      "decision_reason": "主动私语：把只告诉小鹿的事说了",
      "latency_ms": 0.8
    }
  ],
  "telemetry": {
    "rounds": 3,
    "round_latency_p95_ms": 1.2,
    "llm_calls": 0,
    "total_tokens": 0,
    "cost_per_100_rounds": 0.0
  }
}
```

`/say` 与 `/idle` 的响应里**永远带 telemetry** —— 接游戏的团队不用另起
监控就能算成本账（`cli cost` 是它的离线画像版）。

### 存档

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/v1/sessions/{sid}/save` | body `{"slot": "main"}` |
| `POST` | `/v1/sessions/{sid}/load` | body `{"slot": "main"}` |

存档路径 = `{scenario}_{owner}_{slot}.json`（默认目录 `saves/`，
`--save-dir` 改）——**跟着人走，不跟会话走**。断线重连、换进程重启
都读得回来：记忆、关系、称号、专属称呼约定、已私语过的秘密。

---

## 三、三条设计决定（接之前值得知道）

1. **一个会话 = 一个世界副本。** 会话之间互不可见（A 服玩家的悄悄话
   不会泄进 B 服），会话内串行（一轮没跑完下一轮排队 —— 发言权轮转
   本来就不支持并发发言）。多玩家共处一个世界：`/say` 带 `player_id`
   切换发言者即可。

2. **内容热更新是"新会话生效"。** 每次建会话都重新读 YAML —— 改人设、
   改设定书、加 NPC，`POST /v1/sessions` 一个新会话就是新内容，
   跑着的老会话继续用老内容（不会半场换剧本）。这就是 42 天版本
   节奏的答案：发版不用停服。

3. **错误方向安全。** 私聊收件人写错返回 400，整轮没人接话 ——
   绝不退化成公开广播。宁可少说一句，不能多说一句。

## 四、边界（现在不提供的东西）

- **鉴权**：默认无。对外网开放请在前面挂网关（token / 限流 / 审计），
  这层刻意不做 —— 每个游戏的账号体系不一样，框架猜不了。
- **长连接**：没有 WebSocket / SSE。请求-响应 + 轮询覆盖对话主循环；
  长连接的复杂度留给游戏引擎自己的网络栈。
- **多进程横向扩展**：单进程内存会话。会话状态已在 Cast 内可序列化
  （存档机制就是证明），但"多 worker 共享一个会话"还没做 ——
  单会话本来也串行，先把单机做对。
