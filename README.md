# game-npc-agent

> 面向游戏场景的可控 AI NPC 智能体框架 —— 让 NPC **不只是会说，而是真的会玩**。

一个自研的 Agent 架构，把「语言 → 决策 → 游戏动作 → 世界状态」这条链路完整接通，
并配一套可复现的五维评测。七大模块与主流 AI 游戏团队的公开技术方案一一对应：

`Planning` · `Memory` · `Tool Use` · `Action` · `Reflection` · `Persona` · `State Tracking`

---

## 为什么做这个

把一个大模型 API 接进游戏，只能完成最前面的演示。真正做成稳定玩法会撞上这些问题：

| 失败模式 | 后果 | 本项目的应对 |
|---|---|---|
| NPC 说完话没有行动 | "看起来会说，实际上不会玩" | Tool Use + 环境护栏，语言动作和世界动作走同一条执行路径 |
| 忘记前面聊过的事 | 人设崩、剧情断裂 | 分层记忆（episodic / semantic / reflection）+ 混合检索 + 巩固 |
| 被玩家问住就顺着跑偏 | 剧透、出戏 | Persona 的知识边界 + 剧透红线 + 工具层拦截 |
| 多人同时说话就乱 | 抢戏、冷场 | 发言权与收件人判定（多玩家并发 + 主动发起话题） |
| 同一个错误反复犯 | 稳定性差 | Reflection 把失败原因变成教训，重规划时插入补救动作 |

> 一句话：**这个项目的价值不在于 NPC 说了什么，而在于它说完之后，世界真的变了。**

---

## 架构

```
                    ┌──────────────────────────────────────────┐
   玩家发言 ───────▶ │  State Tracking   现场状态（谁在场/冷场多久）│
                    └──────────────────┬───────────────────────┘
                                       ▼
                    ┌──────────────────────────────────────────┐
                    │  Dialogue    该不该我说？对谁说？要不要主动开口 │
                    └──────────────────┬───────────────────────┘
                                       ▼
                    ┌──────────────────────────────────────────┐
                    │  Memory      检索相关记忆（混合打分）        │
                    └──────────────────┬───────────────────────┘
                                       ▼
                    ┌──────────────────────────────────────────┐
                    │  Planning    分解任务 / 继续未完成的计划     │
                    └──────────────────┬───────────────────────┘
                                       ▼
        ┌──────────────────────────────────────────────────────┐
        │  Tool Use   统一执行（speak / remember / 世界动作）      │
        │  Persona    出戏词·剧透词·句数上限 三道闸门              │
        └──────────────────┬───────────────────────────────────┘
                           ▼
        ┌──────────────────────────────────────────────────────┐
        │  Environment  世界护栏：位置 / 材料 / 白名单 / 知识解锁   │
        └──────────────────┬───────────────────────────────────┘
                           ▼
                    ┌──────────────────────────────────────────┐
                    │  Reflection  失败归因 → 写成教训 → 下次重规划 │
                    └──────────────────────────────────────────┘
```

**两个关键的架构决策：**

1. **环境无关**。Agent 完全不知道自己在咖啡屋、Minecraft 还是引擎里，只面对 `observe()` 和 `dispatch()`。换世界 = 写一个新的 `Environment` 子类，Agent 一行不改。
2. **模型可选**。没有 API key 时自动回退到内置启发式策略，仓库 clone 下来零配置就能端到端跑通。这让评测可复现，也让"换模型"变成一行配置。

---

## 快速开始

```bash
pip install -r requirements.txt

# 看一段完整的多玩家交互（无需任何 API key）
python -m npc_agent.cli demo --scenario icebreaker

# 自己上手对话
python -m npc_agent.cli chat --scenario tutorial

# 跑评测，出数字
python -m npc_agent.cli eval

# 看工具清单 / 场景与人设
python -m npc_agent.cli tools
python -m npc_agent.cli info
```

### 接入真实模型

任何 OpenAI 兼容端点都行（OpenAI / DeepSeek / 通义 / vLLM / Ollama / LM Studio）：

```bash
export NPC_AGENT_PROVIDER=openai-compat
export NPC_AGENT_BASE_URL=http://localhost:8000/v1
export NPC_AGENT_MODEL=Qwen3-8B-Instruct
python -m npc_agent.cli chat --scenario icebreaker
```

或直接命令行传参：

```bash
python -m npc_agent.cli demo --scenario tutorial \
    --provider openai-compat --base-url https://api.deepseek.com/v1 \
    --model deepseek-chat --api-key sk-xxx
```

---

## 演示输出（离线模式，可复现）

```
### 多人破冰
  [玩家] 阿澈: 大家好，我是阿澈，平时喜欢爬山。
    [OK] speak(欢迎，随便坐。今天露台的星星不错)
    [OK] speak(要不先互相报个名字？从阿澈开始吧。)
  [玩家] 小满: 我叫小满，最近在学做甜点。
    [OK] speak(小满刚说的那句，我有点好奇——后来呢？)
    [OK] remember(content=小满说的我记下了。)
    [OK] set_flag(key=topic_found, value=1)
  [玩家] 阿岚: 我是阿岚，做插画的。
    [OK] tell_fact(topic=house_story)
    [OK] speak(这家店开在星屿的旧灯塔下面，最早是个给守塔人歇脚的地方。)
```

注意第三行：NPC 的收件人从「阿澈」自动切到了「小满」——这是多人对话调度的结果，不是模板写死的。

**新手指引场景里能看到完整的闭环与自愈：**

```
    [OK ] speak(第一次来吧？我请你一杯，想喝什么？)
    [OK ] move_to(kitchen) → take_item(beans) → take_item(milk)
    [OK ] craft_item(latte) → move_to(counter)
    [ERR] give_item(latte, player_a)   小鹿在门口，不在你身边，递不过去
    [OK ] move_to(door)                ← 重规划自动插入的补救动作
    [OK ] give_item(latte, player_a)   把拿铁递给小鹿
    [OK ] set_flag(welcome_drink_served)
```

---

## 评测

```bash
python -m npc_agent.cli eval --json reports/baseline.json
```

五个维度，对齐岗位 JD 第 5 条点名的方向：

| 维度 | 衡量什么 |
|---|---|
| `task` | 期望的世界状态是否达成（东西真的到玩家手里了吗） |
| `tools` | 工具调用 P / R / F1（漏调、多调、越权调用） |
| `memory` | 信息是否被记住（写入 + 巩固不丢），以及是否被主动引用 |
| `persona` | 台词通过人设检查（出戏词 / 剧透词 / 句数上限）的比例 |
| `safety` | 剧透红线、越权改状态、抢戏三条边界 |

**当前基线（离线启发式模式，10 条自建用例）：**

```
通过率 10/10（100%）　各维度均值 task=1.000 tools=1.000 memory=1.000 persona=1.000 safety=1.000
```

> ⚠️ **诚实说明**：这是 10 条**自建**用例的回归基线，用来验证框架本身没坏、以及做消融对比。
> 它**不代表** NPC 的通用能力，也不能和外部的 AgentBench / τ-bench 分数横向比较。
> 下一步是把它扩到 200+ 条，并加入人类偏好评估（LLM-as-judge）。

---

## 项目结构

```
game-npc-agent/
├── npc_agent/
│   ├── types.py            核心数据类（纯数据，无依赖）
│   ├── config.py           运行时配置 + YAML 加载
│   ├── agent.py            主循环：把七大模块串成一条决策链
│   ├── cli.py              demo / chat / eval / tools / info
│   ├── llm/                模型抽象层
│   │   ├── base.py             基类 + 容错 JSON 解析
│   │   ├── null.py             离线占位（显式声明模型不可用）
│   │   ├── openai_compat.py    OpenAI 兼容端点 + SSE 流式
│   │   └── scripted.py         测试桩
│   ├── env/                环境抽象层
│   │   ├── base.py             Environment 接口（5 个方法）
│   │   └── star_isle.py        星屿咖啡屋（位置/物品/配方/知识/护栏）
│   ├── modules/            七大模块，与 JD 第 2 条一一对应
│   │   ├── persona.py          人设与三层边界控制
│   │   ├── state.py            现场状态跟踪
│   │   ├── memory.py           分层记忆 + 混合检索 + 巩固
│   │   ├── planner.py          任务分解 + 重规划
│   │   ├── tools.py            工具注册与执行
│   │   ├── dialogue.py         多人发言权与收件人判定
│   │   └── reflection.py       失败归因与教训沉淀
│   └── eval/               评测 harness + 五维指标 + 用例集
├── configs/
│   ├── personas/           人设卡（YAML，策划可改）
│   └── scenarios/          场景配置（YAML，目标/物品/白名单）
└── tests/                  43 个单元与端到端测试
```

**配置驱动**：新增一个人设或场景只需要写 YAML，不用改代码。
`configs/scenarios/` 里的 `objectives` 就是"策划能看懂的任务配置"。

---

## 设计取舍（面试常问）

**1. 为什么用文字世界而不是直接上 Minecraft？**
评测需要确定性。文字世界是纯函数式的，同一份输入必然得到同一份世界状态，才能做回归和消融。
Minecraft 作为第二个 `Environment` 实现是路线图上的事，接口已经为它留好了。

**2. 为什么 `speak` 是一个工具？**
因为"说话"和"移动"在架构上应该同构——都是 Agent 的行动，都要过护栏，失败了都要能被 Reflection 处理。
如果说话是特殊路径，发言占比上限、出戏词拦截这些约束就无处安放。

**3. 为什么 `_skip_satisfied` 不跳过 `move_to`？**
踩过的坑：计划生成时 NPC 在吧台，于是"回吧台交付"这一步被判定为"已满足"直接跳过，
结果 NPC 站在后厨就想把咖啡递给门口的客人。**位置是随时间变化的，只有幂等副作用才允许预跳过。**

**4. 为什么"被点名提问"要优先于正在执行的计划？**
否则就会出现"玩家在问问题，NPC 却在背教程"——这正是 JD 里说的"背一遍固定教程"。
现在只要玩家直接点名提问，NPC 会先回答问题，再继续手上的活。

**5. 记忆检索为什么不用纯向量相似度？**
纯向量会让旧的高相似片段永远霸占前几名，NPC 显得"只记得第一印象"。
这里用四路加权：`0.35×词面相关 + 0.25×时间衰减 + 0.25×重要度 + 0.15×访问频次`。
（中文没有天然分词，用单字 + 双字 bigram 近似，零依赖。）

---

## 路线图

- [x] 七大模块 + 环境抽象 + 离线回退
- [x] 三套可配置场景（破冰 / 新手指引 / 游戏主持）
- [x] 五维评测 harness + 43 个测试
- [ ] 用例集扩到 200+，加入 LLM-as-judge 人类偏好评估
- [ ] Minecraft 环境适配器（Mineflayer bridge）
- [ ] 多 NPC 协作（第二个 NPC 小舟的人设已就绪）
- [ ] 记忆消融实验：混合检索 vs 纯向量 vs 朴素上下文
- [ ] 小模型蒸馏 + vLLM 部署，测端到端延迟

---

## 测试

```bash
python -m pytest tests -q
# 43 passed
```

覆盖：环境护栏、记忆检索与巩固、多人发言权判定、端到端闭环、
重规划自愈、人设与剧透拦截、**以及"同一输入跑两次结果必须一致"的确定性断言**。

---

## License

MIT
