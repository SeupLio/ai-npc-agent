# 交接文档 · ai-npc-agent（多 NPC 游戏对话框架）

> 写于 2026-10-31。**这是一份交接时的状态快照**：§4 / §5 列的四块改动已在同日
> 完成验证、回归、文档修订和推送。保留本文是为了让后来的人看到"接手时工作区
> 是什么样、哪些坑是那时候埋下的"，**不是当前待办清单**。

---

## 0. 这份文档的边界

- **交接时**已推送到 GitHub 的内容：见 §3（当时最新提交 `cc50354`）。
- **交接时未提交、仍在本地工作区**的内容：见 §4（Tasks 14/15/16/17）。
  这四块已在 2026-10-31 完成并提交（详见 `ENGINEERING.md` 附二十六）。
- 交接时用户（产品方）说过"不需要再推进项目编码"；后续会话按新指令
  完成了收尾。本文保留当时的语境，不作为当前约束。
- 交接时工作区**没有经过完整 pytest 回归**。收尾后：**1177 passed, 1 skipped**
  （收集 1178），活世界基准 **17/17**、六个注入全部抓到。

---

## 1. 项目一句话

`E:/WB AI/game-npc-agent` 是一个 Python 框架，模拟"一间咖啡屋里的多个 NPC 自己接话、互相记得你、因各自动机产生小冲突"的涌现叙事。灵感来自米哈游公开的两份 NPC 设计方法论，但工程化成了可测、可离线复现（NullLLM，零模型调用）的系统。

核心闭环：**导演（Cast）+ 每个角色一个 NPCAgent + 共享世界状态 + 发言权轮转**。
评测哲学：**离线启发式路径（NullLLM）做确定性回归；真 LLM 路径只在需要"像人"时才开**。所有"产品级"能力都有测试钉住，不在提示词里"请求"而在工具层拦截。

---

## 2. 已经验证并交付的能力（历史，仅供参考）

按时间顺序（git log，最新在上）：

| 提交 | 内容 |
|---|---|
| `cc50354` | 跨会话记忆：关掉再进来，她还记得你（persistence，含 tick 续算、编年史引用陷阱） |
| `552f8f3` | 活世界的四块空白：设定一致性 / 关系与称号 / 动机涌现事件 / 一对多注意力 |
| `af93493` | NPC 之间真的能对话：接话义务 + 话题账本 + 印象流动 |
| `1149ab4` | README 重写 + 评测分辨率实验 |
| `60fe509` | 删实测出来的死代码 + 护栏 |
| `1a20208` | 双 NPC 答非所问修复（四个根因 + 两个自造回归） |

评测基线（**已提交状态**下的数字，注意：含未提交新增用例后数字会变，见 §5）：
- 六维综合能力评测：235/235 通过
- 对话基准：18/18 通过
- 活世界基准（living_world_bench）：**交接时** 15/15，含 5 个故障注入
  （收尾后 **17/17**，6 个注入；新增「多 NPC 同场」2 条与第三人调度注入）
- 完整 pytest 套件：**交接时** 1150 passed, 1 skipped（ENGINEERING.md:2097）
  （收尾后 **1177 passed, 1 skipped**，收集 1178）

---

## 3. 已提交但本次未涉及 / 已稳定的模块

以下模块在 `552f8f3` 与 `cc50354` 已提交并推送，本次未改动逻辑（仅被 §4 的新增能力调用）：
- `npc_agent/modules/lore.py` —— 同句"主语+错误说法"才判违规；`canon_block()` 注入提示词。
- `npc_agent/modules/relationship.py` —— 好感→阶段→称号；称号只口头宣布一次。阶段阈值：56=面熟 / 68=常客 / 85=老友（初始好感 50）。
- `npc_agent/modules/attention.py` —— 未答问题记账 + 优先级排序（`score()` = 被提到+100 / 提问+40 / 等待+5每 tick 封顶 30 / 好感±10）。
- `npc_agent/modules/motives.py` —— 注意：`poll` 在 §4.4 被重写，是**唯一一处改动既有模块逻辑**的地方。
- `npc_agent/modules/persistence.py` —— 注意：`_restore_events` 在 §4.4 被小改（resolved 键从"冲突对"改为"资源"）。

---

## 4. 交接时未提交的工作（已于 2026-10-31 完成并提交，保留原始说明）

`git status` 现状：`agent.py / cast.py / env/star_isle.py / modules/{conversation,state,motives,persistence,state}.py / studio.py / studio_ui.py` 已修改；新增 `configs/personas/luo.yaml`、`configs/scenarios/trio.yaml`、`tests/test_peers.py`、`tests/test_recall_cooldown.py`。

### 4.1 Task 14 · 同伴好感（NPC↔NPC）

**为什么做**：`_ask_peer`（冷场时跟同伴搭话）原本按字典顺序挑人。两人场景看不出问题；三人以上第三个人永远不被搭话，玩家只会觉得"那个人被孤立了"。好感在这里不是"模拟社交"，是给挑人一个**非任意的依据**。

**改了什么**（精确 diff）：
- `env/star_isle.py` `observe()`：`affinity` 现在包含**所有非自身角色**（含同伴），不只是玩家。⚠️ 循环变量**不能叫 `actor`** —— 会遮住外层"正在观察的那个角色"，导致 `actor.affinity_to(pid)` 变成"每个人对自己的好感"（全是 50，且不报错）。这是埋过的坑，已用注释钉住。
- `modules/state.py` `update_from_env()`：好感按 kind 分流，同伴好感写进 `self.peers`（不只 `self.players`）。少了这一支，`_ask_peer` 的"按关系挑人"永远读到 50。
- `cast.py`：新增 `_bump_affinity(actor_id, other_id, delta)`（只对场上真实存在的角色生效，写错 id 静默失败而非崩）；在**转达/点名**时调用 —— 点名方+2、被点名方+1；**听到同伴分享**时听者对说的人+1（单向）。
- `agent.py` `_ask_peer()`：`peers.sort(key=lambda pid: (-affinity, pid))`，按关系挑最近的；同好感按 id 稳定排序（结果可复现）。
- 新增 `tests/test_peers.py`（7 条）：观察携带同伴好感、状态分流、说话/听新闻拉近距离、好感有界、挑更近的同伴、都问过后闭嘴。

### 4.2 Task 15 · Studio 把"看不见的机制"亮出来

**为什么做**：项目里"好感一直在算、没人读"曾躺了两轮。调参的人看不到关系/称号/欠着的回答/编年史，等于这些机制不存在。

**改了什么**：
- `studio.py` `_memory_payload()`：每个 agent 额外返回 `bonds`（对每个人的阶段/分数/最新称号）和 `debts`（欠着的回答文本）。
- `studio.py` 新增 `_chronicle_payload()`（世界发生过的事件，不属于任何 NPC，不该因某人失忆而消失）和 `_peer_affinity_payload()`（NPC 互相对彼此的好感，页面能看见"谁跟谁更近"）。
- `run_chat()` 的返回 dict 增加 `chronicle` 与 `peer_affinity` 两个键。
- `studio_ui.py`：记忆卡片增加"关系·称号·欠着的回答"子块；新增"编年史"卡片渲染编年史 + 同伴好感表。

### 4.3 Task 16 · 跨 NPC 回引冷却（修"两个人轮流复读"）

**问题（实测）**：读档后的第二次进店，阿柚说"你之前提过你口味偏酸"，下一轮小舟又说"你上次说过你口味偏酸"。两人记忆里各有一条（一条亲耳听、一条听同伴说），按记录 id 去重拦不住，玩家感受是"轮流复读"。

**做法**：用**时间窗冷却**而非语义去重（语义去重要另一套模型且会误判；冷却失败方式安全——顶多少回引一次）。
- `conversation.py` `TopicLedger` 新增 `is_recent(point, now, window)`：与 `is_spent`（一局一次）是两回事。两者都要：只用 `is_spent` 则小舟永远不能再提这个人；只用 `is_recent` 则隔十轮又能讲店史。
- `agent.py` 新增常量 `RECALL_COOLDOWN = 4`（⚠️ 为什么是 4 不是 10：太短挡不住轮流复读，太长让"记得你"整段只发生一次；4 轮≈玩家能听出重复的距离）。在 `_avoid_spent_point` 里：`recall`/`heard_from` 意图若 `is_recent(recall_guest, speaker_id)` 为真 → 降级为 `acknowledge`。冷却计时从"真的说出口"开始（被拦下的那次不计时）。
- 新增 `tests/test_recall_cooldown.py`（5 条）：时间窗内/外、未标记、spent 与 recent 区别、端到端两人不轮流复读、冷却过期后还能再提。

### 4.4 Task 17 · 三人同场 + 动机裁定"按资源分组"

**为什么做**：两人时很多规则"看起来对"只是被"只有一个人可挑"掩盖了。三人场景是"多 NPC 是不是真的多"的对照实验。

**新增配置**：
- `configs/personas/luo.yaml` —— 第三个 NPC 洛（调琴师），带齐所有人设意图模板（`welcome_back`/`grant_title`/`yield_to_peer`/`recall_event_win`/`recall_event_lose`）。
- `configs/scenarios/trio.yaml` —— 三 NPC（ayou/xiaozhou/luo）+ 两玩家，三人抢同一资源"营业时间"，含 lore 块。场景 id=`trio`，会被 `list_scenarios()`（config.py:240 自动扫 `configs/scenarios/`）自动发现，**无需改代码**。

**改了什么（核心，且是唯一改动既有模块逻辑的地方）**：
- `motives.py` `MotiveBoard.poll()` **重写**：原来枚举所有 (a,b) 冲突对 → 三人抢同一资源会产出 **3 条事件、2 句让步台词**（实测 trio t3 一口气三条："小舟要…阿柚让步""洛要…阿柚让步""小舟要…洛让步"）。改为**按资源分组**：每个资源只裁定一次，赢家一个，其余人全进 `losers`，`loser` 记让步方里最强的那个（由他代表说那句台词）。
- `WorldEvent` 增加 `losers: list[str]` 和 `resource: str` 两个字段。
- `resolved` 类型从 `set[tuple[str,str,str]]` 改为 `set[str]`（按资源去重）。
- `poll` 新增 `names` 参数，事件文本用显示名（不再出现 `xiaozhou要…` 这种内部 id）。
- `persistence.py` `_restore_events()` 同步改为把 `event.resource` 加回 `board.resolved`（读档不恢复则下一轮再裁定，让步方把同一句说第二遍）。

---

## 5. 交接收尾时做的验证（已完成）

> 未提交代码**没有经过完整 pytest 回归**。尤其 Task 17 改了 `poll` 的内部契约（resolved 类型、事件生成方式），可能让既有 `test_motives.py` / `test_living_world_bench.py` 的预期落空。

**第一步：跑受影响模块的离线测试**（这些全用 NullLLM，快、不触网）：

```bash
cd "E:/WB AI/game-npc-agent"
python -m pytest tests/test_motives.py tests/test_peers.py tests/test_recall_cooldown.py tests/test_persistence.py tests/test_living_world_bench.py -q
```

**第二步：跑完整套件确认没回归**：

```bash
python -m pytest tests -q
```
（交接时已提交基线是 1150 passed / 1 skipped；收尾后为 **1177 passed / 1 skipped**，
收集 1178。以实跑为准，别照抄旧数字。）

**第三步：三人场景冒烟**（确认第三个人真的说话、动机只裁定一次）：

```bash
python -m npc_agent.cli demo --scenario trio --ticks 12 --save trio_save.json
python -m npc_agent.cli demo --scenario trio --load trio_save.json   # 应自动先问"你还记得我吗？"
```
重点看控制台：t3 附近**只应有一条**"动机冲突"事件，`losers` 含两个让步方；洛（第三个人）要有有效发言，而不是全程背景板。

**第四步（已完成）**：活世界基准已补「多 NPC 同场」2 条用例
（`trio_everyone_speaks`、`trio_clash_resolved_once`），并新增故障注入
「掐掉第三人调度」。现在基准是 **6 个维度、17 条**，注入 **6/6** 被抓到；
`report_index.py`、README 与离线报告已同步。

---

## 6. 提交与推送的正确姿势（绕过 github.com 的 000 报错）

> 本机 `git push` 到 `github.com` 会返回 `000` / 空回复（只有 `api.github.com` 可达）。已验证可用的绕过方式：**Git Data API 脚本**。

- 脚本：`C:/Users/10718/AppData/Local/Temp/push_via_api.py`（blob→tree→commit→ref PATCH）。
- 运行前提：
  - 环境变量 `GH_TOKEN`（从凭据填充拿，不是 repo 里的东西，别提交）。
  - `MSYS_NO_PATHCONV=1`（否则 Git Bash 会把 Windows 路径改坏）。
  - **commit 的 author date 与 committer date 必须分别取**（`git show -s --format=%aI` 和 `%cI`）再分别写入，否则 commit SHA 会分叉、推送后 hash 对不上。
- 建议的提交粒度（一次一个逻辑，别混）：
  1. `feat(peers): NPC 之间也有好感，冷场挑人按关系而非字典序`
  2. `feat(studio): 页面亮出关系/称号/欠答/编年史/同伴好感`
  3. `fix(recall): 跨 NPC 回引冷却，修两个人轮流复读`
  4. `feat(trio): 三人同场场景 + 动机裁定按资源分组（每资源只裁一次）`
- 提交信息里把"为什么"写清楚（这个项目要求每条改动都能回答"解决了哪个实测痛点"）。

---

## 7. 文档收尾（已完成）

- README 局限 #5 与 TODO 已修订：多 NPC 支持到 3 个（全场轮转），
  四人以上 / 分组与私聊列进未来工作。
- `scripts/sync_doc_counts.py` 已重跑，README / ENGINEERING / PROJECT_STORY
  的测试计数同步到 1178 收集 / 1177 通过 / 1 跳过。
- `docs/BOUNDARY.md`、`docs/TECHNICAL.md`、`docs/PROJECT_STORY.md`
  已按 trio / 同伴好感 / 回引冷却 / 存档好感修正边界与叙述。

---

## 8. 设计决策备忘（"为什么这么写"——避免接手人误改）

1. **动机裁定按资源分组，不是按冲突对**：语义是"一件事裁定一次，赢家一个，其余让步"。按对枚举会让三人抢同一资源产出 3 事件，玩家看到"为同一件事吵三次"。
2. **回引用冷却而非语义去重**：失败方式安全（顶多少回引一次），不需要额外模型。
3. **同伴好感不是社交模拟，是给挑人非任意依据**：字典序在两人场景掩盖了"第三个人被孤立"。
4. **resolved 用资源键而非冲突对键**：读档恢复时必须和 `poll` 的分组口径一致，否则会二次裁定。
5. **所有"结果可复现"的地方都做稳定排序**（按 id 兜底）：同一段对话重放必须得到同一个故事，评测才能比。

---

## 9. 已知坑 & 排错速查

- **变量命名遮罩**：`star_isle.observe()` 的循环变量曾叫 `actor`，遮住外层"正在观察的角色"，导致同伴好感全是 50 且不报错。改名叫 `a` 并显式用 `actor_id`。
- **CRLF 警告**：`persistence.py` / `state.py` 工作副本有 LF→CRLF 提示，提交后 Git 会规范化，不影响逻辑；若 CI 卡换行，检查 `.gitattributes`。
- **GitHub 推送 000**：见 §6，别再试裸 `git push`，直接用 API 脚本。
- **故障注入用 monkeypatch 而非配置开关**：living_world_bench 的 `DEFECTS` 通过替换方法（如 `agent.relationships.sync = lambda...`、`cast.motives = MotiveBoard()`）模拟"模块被关掉"，保证"关掉就退化"能被测到。新增能力若可关，建议同样补一条缺陷用例。

---

## 10. 一句话交接

> 活世界四块（设定/关系/动机/注意力）和跨会话记忆已上线并验证；
> **同伴好感、Studio 可视化、跨 NPC 回引冷却、三人场景**四块也在 2026-10-31
> 完成回归（1177 passed / 1 skipped）、补上两人/三人基准用例、修订文档并推送。
> 当前已知边界：多 NPC 支持到 3 个，仍是全场轮转，四人以上没实现。
> 别动已推送的 `cc50354` 之前的逻辑。

本地记忆日志（`.workbuddy-ai/memory/2026-10-30.md`、`2026-10-31.md`）有更细的迭代过程，已被 .gitignore 排除，不在仓库里，但本机可读。
