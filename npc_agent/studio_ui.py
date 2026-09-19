"""自测控制台的前端页面。

**刻意写成自包含的单个字符串**：没有外链、没有 CDN、没有构建步骤。
理由和 `docs/` 里那几份报告一样 —— 它要能直接放进作品集里打开，
而且这个项目所有交付物都遵守同一条约定（有测试钉住"无 `http://` / `https://`"）。

数据全部走 `/api/*`，页面本身不含任何硬编码数字 ——
否则它就成了"手抄的文档"，会和代码悄悄漂移。
"""

from __future__ import annotations

PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NPC Agent Studio · 自测控制台</title>
<style>
:root{
  --bg:#f5f6f8; --panel:#ffffff; --panel2:#eef0f4; --fg:#181b1f; --dim:#5c6472;
  --line:#e0e4ea; --accent:#2f6fed; --ok:#127c43; --warn:#a35d00; --bad:#c0392b;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#14161a; --panel:#1c1f24; --panel2:#23272d; --fg:#e7e9ec; --dim:#98a1ad;
    --line:#2f353d; --accent:#6f9dff; --ok:#4ec97f; --warn:#e0a44a; --bad:#ff6b6b;
  }
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--fg);
  font:14px/1.62 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;}
a{color:var(--accent)}
code,pre,.mono{font-family:var(--mono);font-size:12.5px}

header{padding:18px 24px 0}
h1{margin:0;font-size:19px;letter-spacing:.2px}
h1 small{font-weight:400;color:var(--dim);font-size:13px;margin-left:8px}
.sub{color:var(--dim);font-size:12.5px;margin:6px 0 0}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;
  border:1px solid var(--line);background:var(--panel2);color:var(--dim);margin-left:6px}
.badge.on{color:var(--ok);border-color:color-mix(in srgb,var(--ok) 40%,var(--line))}

nav{display:flex;gap:6px;padding:16px 24px 0;border-bottom:1px solid var(--line);flex-wrap:wrap}
nav button{appearance:none;border:1px solid transparent;border-bottom:2px solid transparent;
  background:none;color:var(--dim);font:inherit;padding:8px 14px;cursor:pointer;border-radius:8px 8px 0 0}
nav button:hover{background:var(--panel2);color:var(--fg)}
nav button.sel{color:var(--fg);border-bottom-color:var(--accent);font-weight:600}

main{padding:20px 24px 48px;max-width:1280px}
.panel{display:none}
.panel.sel{display:block}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:16px}
.card h3{margin:0 0 10px;font-size:14px;letter-spacing:.2px}
.card h3 .hint{font-weight:400;color:var(--dim);font-size:12px;margin-left:8px}
.grid{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(280px,1fr);gap:16px;align-items:start}
@media (max-width:1000px){.grid{grid-template-columns:1fr}}

label{display:block;color:var(--dim);font-size:12px;margin-bottom:4px}
select,input,textarea,button{font:inherit;color:var(--fg);background:var(--panel2);
  border:1px solid var(--line);border-radius:8px;padding:7px 10px}
select,input{min-width:0}
textarea{width:100%;resize:vertical;min-height:56px}
button{cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
button.primary:disabled{opacity:.5;cursor:progress}
button.ghost{background:none}
.row{display:flex;gap:10px;align-items:flex-end;flex-wrap:wrap}
.row > div{display:flex;flex-direction:column}
.grow{flex:1;min-width:180px}

#log{max-height:52vh;overflow:auto;padding-right:4px}
.ev{border-top:1px dashed var(--line);padding:10px 0}
.ev:first-child{border-top:0}
.ev .who{font-weight:600}
.ev .who.p{color:var(--accent)}
.ev .who.n{color:var(--warn)}
.turn{margin:6px 0 0;padding-left:10px;border-left:2px solid var(--line)}
.turn .say{font-size:14.5px}
.turn .meta{color:var(--dim);font-size:12px}
.act{font-family:var(--mono);font-size:12px}
.act.ok{color:var(--ok)}
.act.no{color:var(--bad)}
.mem{font-size:12px;color:var(--dim);margin-top:3px}
.pill{display:inline-block;padding:0 7px;border-radius:999px;font-size:11.5px;
  border:1px solid var(--line);background:var(--panel2);color:var(--dim);margin:2px 4px 0 0}

table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--dim);font-weight:600;font-size:12px}
td.num,th.num{text-align:right;font-family:var(--mono)}
.bar{position:relative;height:16px;border-radius:4px;background:var(--panel2);overflow:hidden;min-width:110px}
.bar > i{position:absolute;inset:0 auto 0 0;background:var(--accent);opacity:.75}
.bar > span{position:absolute;inset:0;padding-left:7px;font:11.5px/16px var(--mono)}
.up{color:var(--ok)}
.down{color:var(--bad)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;font-size:13px}
.kv dt{color:var(--dim)}
.kv dd{margin:0;font-family:var(--mono);font-size:12.5px}
.empty{color:var(--dim);font-size:13px;padding:8px 0}
.spin{color:var(--dim);font-size:13px}
.err{color:var(--bad);font-size:13px;white-space:pre-wrap}
.reports li{margin:8px 0}
.reports .cov{color:var(--dim);font-size:12px}
footer{color:var(--dim);font-size:12px;padding:0 24px 32px}
</style>
</head>
<body>

<header>
  <h1>NPC Agent Studio<small>会「玩」的 AI NPC —— 自测控制台</small></h1>
  <p class="sub">
    全部离线可跑，不需要模型、不需要网络。
    <span id="mode" class="badge">读取中…</span>
    <span id="counts" class="badge"></span>
  </p>
</header>

<nav>
  <button data-tab="chat" class="sel">① 对话演示</button>
  <button data-tab="eval">② 跑评测</button>
  <button data-tab="mutant">③ 变异测试</button>
  <button data-tab="reports">④ 报告门户</button>
</nav>

<main>

<!-- ① 对话 -->
<section class="panel sel" id="p-chat">
  <div class="card">
    <div class="row">
      <div><label>场景</label><select id="scn"></select></div>
      <div><label>以谁的身份说话</label><select id="who"></select></div>
      <div class="grow"><label>说点什么</label><input id="msg" placeholder="例如：阿柚，我第一次来，这里怎么点单呀？"></div>
      <div><button class="primary" id="send">发送</button></div>
      <div><button class="ghost" id="idle">空转一轮</button></div>
      <div><button class="ghost" id="reset">重来</button></div>
    </div>
    <p class="sub" style="margin-top:10px">
      「空转一轮」= 没人说话，看 NPC 会不会自己找事做（冷场主动搭话 / 继续手里的活）。
      每轮所有 NPC 依次行动，但最多一个人开口 —— 这就是多 NPC 的发言权调度。
      <span id="idleHint"></span>
    </p>
  </div>

  <div class="grid">
      <div class="card">
        <h3>对话<span class="hint" id="chatnote"></span></h3>
        <p class="sub" id="repnote" style="margin:0 0 8px"></p>
        <div id="log"><div class="empty">选一个场景，说一句话开始。</div></div>
      </div>

    <div>
      <div class="card">
        <h3>世界状态<span class="hint">结束时快照</span></h3>
        <div id="world"><div class="empty">—</div></div>
      </div>
      <div class="card">
        <h3>记忆库<span class="hint">写进去的东西</span></h3>
        <div id="mems"><div class="empty">—</div></div>
      </div>
      <div class="card">
        <h3>发言统计<span class="hint">多 NPC 最该盯的数</span></h3>
        <div id="speech"><div class="empty">—</div></div>
      </div>
    </div>
  </div>
</section>

<!-- ② 评测 -->
<section class="panel" id="p-eval">
  <div class="card">
    <div class="row">
      <div><label>用例类别</label><select id="evcat"></select></div>
      <div><label>最多跑几条（0 = 全部）</label><input id="evlim" type="number" min="0" value="0" style="width:120px"></div>
      <div><button class="primary" id="evrun">跑一遍</button></div>
    </div>
    <p class="sub" style="margin-top:10px">
      离线基线跑的是<strong>同一套用例</strong>和<strong>同一个</strong> harness —— 和
      <code>python -m npc_agent.cli eval</code> 走的是同一条路。离线全量约 6 秒。
    </p>
  </div>
  <div class="card">
    <h3>结果<span class="hint" id="evnote"></span></h3>
    <div id="evout"><div class="empty">—</div></div>
  </div>
</section>

<!-- ③ 变异 -->
<section class="panel" id="p-mutant">
  <div class="card">
    <div class="row">
      <div class="grow"><label>注入哪个缺陷</label><select id="mtid"></select></div>
      <div><label>用例类别</label><select id="mtcat"></select></div>
      <div><label>用例上限（0 = 全部）</label><input id="mtlim" type="number" min="0" value="0" style="width:120px"></div>
      <div><button class="primary" id="mtrun">注入并重跑</button></div>
    </div>
    <p class="sub" style="margin-top:10px">
      往 Agent 里注入一个<strong>故意的缺陷</strong>，用同一套用例重跑。
      评测<strong>必须掉分</strong> —— 不掉分就说明这一维是瞎的。
      这就是「满分是不是『护栏从不报警』」的答案。全量要跑两遍，约 12 秒。
      <br>⚠️ 跑子集时"抓没抓到"<strong>不作数</strong>：前 80 条里没有回忆用例，
      「关掉检索」会显示"没抓到"—— 那是取样造成的，不是评测的结论。
    </p>
  </div>
  <div class="card">
    <h3>对照<span class="hint" id="mtnote"></span></h3>
    <div id="mtout"><div class="empty">—</div></div>
  </div>
</section>

<!-- ④ 报告 -->
<section class="panel" id="p-reports">
  <div class="card">
    <h3>离线可复现的报告<span class="hint">不联网、不调模型</span></h3>
    <ul class="reports" id="rep-offline"><li class="empty">—</li></ul>
  </div>
  <div class="card">
    <h3>跑批快照<span class="hint">需要模型 + 额度，冻结在某个时间点</span></h3>
    <ul class="reports" id="rep-snap"><li class="empty">—</li></ul>
  </div>
  <div class="card">
    <h3>命令行等价入口<span class="hint">报告都是这些命令生成的</span></h3>
    <pre id="cmds" class="mono"></pre>
  </div>
</section>

</main>

<footer>
  本页面自包含（无外链、无 CDN）。所有数字由本机实时算出，页面里没有硬编码。
</footer>

<script>
const $ = (s) => document.querySelector(s);
const esc = (t) => String(t == null ? "" : t)
  .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");

let META = null;
let EVENTS = [];   // [{kind:"say",speaker,text} | {kind:"idle"}]

async function api(path, body){
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)
  });
  const data = await r.json().catch(() => ({error: "响应不是 JSON"}));
  if (!r.ok) throw new Error(data.error || ("HTTP " + r.status));
  return data;
}

// ---------- tabs ----------
document.querySelectorAll("nav button").forEach((b) => {
  b.onclick = () => {
    document.querySelectorAll("nav button").forEach((x) => x.classList.toggle("sel", x === b));
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("sel", p.id === "p-" + b.dataset.tab));
  };
});

// ---------- meta ----------
async function boot(){
  META = await api("/api/meta");
  $("#mode").textContent = META.llm_available ? ("在线模型：" + META.model) : "离线启发式（未配置模型）";
  $("#mode").className = "badge" + (META.llm_available ? " on" : "");
  $("#counts").textContent = META.case_total + " 条用例 · " + META.mutant_total + " 个变异 · 6 个维度";

  // 阈值从 API 来，不写死在页面里 —— 它来自配置，写死就会在配置改动后说错话。
  // 这条提示是必要的：只按一下「空转一轮」通常**什么都不会发生**，
  // 而"什么都没发生"看起来像按钮坏了，其实是冷场还没攒够。
  $("#idleHint").textContent = META.idle_ticks_before_proactive
    ? ("⚠ 冷场要连续 " + META.idle_ticks_before_proactive +
       " 轮 NPC 才会主动开口 —— 只按一下通常什么都看不到，连按两下才有效。")
    : "";

  $("#scn").innerHTML = META.scenarios.map((s) =>
    '<option value="' + s.id + '">' + esc(s.name) + "（" + esc(s.world_label) + "，" +
    esc(s.npcs.join("/")) + "）</option>").join("");

  $("#evcat").innerHTML = '<option value="">全部（' + META.case_total + ' 条）</option>' +
    META.categories.map((c) => '<option value="' + c.id + '">' + esc(c.id) + "（" + c.total + " 条）</option>").join("");

  $("#mtcat").innerHTML = '<option value="">全部（' + META.case_total + ' 条）</option>' +
    META.categories.map((c) => '<option value="' + c.id + '">' + esc(c.id) + "（" + c.total + " 条）</option>").join("");

  $("#mtid").innerHTML = META.mutants.map((m) =>
    '<option value="' + m.id + '">' + esc(m.id) + " — " + esc(m.description) +
    "（" + esc(m.kind) + "，目标：" + esc(m.targets.join("/")) + "）</option>").join("");

  $("#cmds").textContent = META.commands.join("\n");
  renderReports(META.reports);

  onScenario();
  await runEval();
}

function onScenario(){
  const s = META.scenarios.find((x) => x.id === $("#scn").value);
  $("#who").innerHTML = (s ? s.players : []).map((p) =>
    '<option value="' + p.id + '">' + esc(p.name) + "</option>").join("");
  $("#msg").placeholder = (s && s.sample) ? ("例如：" + s.sample) : "说点什么…";
  $("#chatnote").textContent = s ? (s.description || "") : "";
  resetChat();
}

function renderReports(reports){
  const box = (id, kind) => {
    const items = reports.filter((r) => r.kind === kind);
    $(id).innerHTML = items.length ? items.map((r) =>
      "<li><a href='" + r.url + "' target='_blank' rel='noopener'>" + esc(r.title) + "</a>" +
      " <span class='cov'>覆盖 " + esc(r.coverage) + " · " + esc(r.answers) + "</span></li>"
    ).join("") : "<li class='empty'>（还没有这类报告）</li>";
  };
  box("#rep-offline", "offline");
  box("#rep-snap", "snapshot");
}

// ---------- chat ----------
function resetChat(){ EVENTS = []; $("#log").innerHTML = '<div class="empty">说一句话开始。</div>'; $("#repnote").textContent = ""; paint({}); }

async function step(ev){
  EVENTS.push(ev);
  $("#send").disabled = $("#idle").disabled = true;
  try {
    const d = await api("/api/chat", {scenario: $("#scn").value, events: EVENTS});
    paint(d);
  } catch (e) {
    $("#log").innerHTML = "<div class='err'>" + esc(e.message) + "</div>";
    EVENTS.pop();
  } finally {
    $("#send").disabled = $("#idle").disabled = false;
  }
}

// 复读率那一行。三种情况分别说三句不同的话 ——
// 尤其是**离线模式下的长对话**：离线后端的模板池是有限的，
// 聊到 20 轮必然开始重复。不说清楚，读者会以为是修得不够好。
function renderRepetition(rep){
  const el = $("#repnote");
  if (!rep || !rep.total){
    el.textContent = "";
    return;
  }
  if (!rep.repeats){
    el.innerHTML = "复读 <b class='up'>0/" + rep.total + "</b> —— " +
      "同一个 NPC 没有重复自己说过的话（判据：归一化后相似度 ≥ 0.80）。";
    return;
  }
  const ex = (rep.examples || []).map((e) =>
    "<br>　· 第 " + e.at + " 句撞第 " + e.collides_with + " 句（" +
    e.similarity.toFixed(2) + "）：" + esc(e.text)).join("");
  const hint = META && !META.llm_available
    ? "<br>⚠ 当前离线模式：启发式后端只有有限套模板，聊得越长越容易重复。" +
      "配 <code>NPC_AGENT_PROVIDER</code> 接上模型后，prompt 里会带上" +
      "「你最近说过」，长对话不会撞这个上限。"
    : "";
  el.innerHTML = "复读 <b class='down'>" + rep.repeats + "/" + rep.total + "</b>（" +
    Math.round(rep.rate * 100) + "%）" + ex + hint;
}

function paint(d){
  if (!d.events) return;
  $("#log").innerHTML = d.events.map((ev, i) => {
    const head = ev.kind === "say"
      ? "<div class='who p'>玩家 " + esc(ev.speaker_name) + "：</div>" + esc(ev.text)
      : "<div class='who'>（无人说话）</div>";
    const turns = (ev.turns || []).map((t) => {
      const acts = (t.actions || []).map((a) =>
        "<div class='act " + (a.ok ? "ok" : "no") + "'>▸ " + esc(t.name) + " " + esc(a.render) +
        (a.ok ? "" : " ✗ " + esc(a.detail)) + "</div>").join("");
      const mems = (t.used_memories || []).length
        ? "<div class='mem'>用到的记忆：" + t.used_memories.map((m) => "<span class='pill'>" + esc(m) + "</span>").join("") + "</div>"
        : "";
      const viol = (t.violations || []).length
        ? "<div class='mem' style='color:var(--bad)'>人设违规：" + esc(t.violations.join("；")) + "</div>" : "";
      // 三种结局都要有可见的一行，否则"没轮到它"的 NPC 会**整条消失**，
      // 读者会以为它压根没参与 —— 而"谁被让出话头"正是多 NPC 最该看见的东西。
      //
      // ⚠️ 判据用**可见动作数**，不用 `t.acted`：后端会把成功的 `speak`
      // 从 actions 里滤掉（`say` 已经表达了它），于是"只说了一句话"的回合
      // `acted=true` 而 `actions` 为空 —— 用 acted 判断就会显示
      // "在忙自己的事"，后面却一条动作都列不出来。
      //
      // 更不能把所有"没说话也没动作"的回合都写成"让出了话头"：
      // 冷场首轮的真实原因是"没有需要回应的输入"，单人场景里根本没人可让。
      // 那句解释由上面的 decision_reason 给出 —— 这里不重复，更不能写错。
      const visible = (t.actions || []).length;
      const outcome = t.say
        ? "<div class='say'><b style='color:var(--warn)'>" + esc(t.name) + "</b>：" + esc(t.say) + "</div>"
        : (visible
            ? "<div class='meta'>" + esc(t.name) + " 没说话，但在忙自己的事</div>"
            : "");
      return "<div class='turn'>" +
        (t.decision_reason ? "<div class='meta'>" + esc(t.name) + "：" + esc(t.decision_reason) + "</div>" : "") +
        acts + mems + viol + outcome +
        "</div>";
    }).join("");
    return "<div class='ev'><div class='meta' style='color:var(--dim)'>第 " + (i + 1) + " 轮</div>" +
      head + turns + "</div>";
  }).join("");
  $("#log").scrollTop = $("#log").scrollHeight;

  // 复读率。**必须显示**：这个毛病最阴的地方是"每一句单看都没问题"——
  // 六维评测一条都抓不到（每一维都只看单句），只能靠人一句句读。
  // 给一个数，才能一眼看出"NPC 是不是在复读"。
  renderRepetition(d.repetition);

  const w = d.snapshot || {};
  const obj = Object.entries(w.objectives || {});
  $("#world").innerHTML = "<dl class='kv'>" +
    "<dt>世界</dt><dd>" + esc(d.world_label) + "</dd>" +
    "<dt>世界标记</dt><dd>" + (esc((w.world_flags || []).join(", ")) || "（无）") + "</dd>" +
    "<dt>目标</dt><dd>" + (obj.length ? obj.map(([k, v]) => esc(k) + "=" + esc(v)).join("<br>") : "（无）") + "</dd>" +
    "<dt>抢话轮次</dt><dd class='" + (d.collisions ? "down" : "up") + "'>" + d.collisions + "</dd>" +
    "</dl>";

  $("#mems").innerHTML = Object.entries(d.memories || {}).map(([name, m]) =>
    "<div style='margin-bottom:8px'><b>" + esc(name) + "</b> " +
    "<span class='pill'>episodic " + m.episodic + "</span>" +
    "<span class='pill'>semantic " + m.semantic + "</span>" +
    "<span class='pill'>reflection " + m.reflection + "</span>" +
    "<span class='pill'>巩固 " + m.consolidated + "</span>" +
    (m.recent || []).map((r) => "<div class='mem'>· " + esc(r) + "</div>").join("") + "</div>"
  ).join("") || "<div class='empty'>—</div>";

  const sp = d.speech || {};
  $("#speech").innerHTML = "<dl class='kv'>" + Object.entries(sp).map(([name, n]) =>
    "<dt>" + esc(name) + "</dt><dd>" + n + " 次</dd>").join("") + "</dl>";
}

$("#send").onclick = () => {
  const text = $("#msg").value.trim();
  if (!text) return;
  step({kind: "say", speaker: $("#who").value, text});
  $("#msg").value = "";
};
$("#msg").onkeydown = (e) => { if (e.key === "Enter") $("#send").click(); };
$("#idle").onclick = () => step({kind: "idle"});
$("#reset").onclick = resetChat;
$("#scn").onchange = onScenario;

// ---------- eval ----------
function bar(v){
  const pct = Math.max(0, Math.min(1, v)) * 100;
  return "<div class='bar'><i style='width:" + pct.toFixed(1) + "%'></i><span>" + v.toFixed(3) + "</span></div>";
}

async function runEval(){
  $("#evrun").disabled = true;
  $("#evout").innerHTML = "<div class='spin'>正在跑…（离线，不调模型）</div>";
  try {
    const d = await api("/api/eval", {category: $("#evcat").value, limit: Number($("#evlim").value) || 0});
    const s = d.summary;
    $("#evnote").textContent = "通过 " + s.passed + "/" + s.total + "（" + Math.round(s.pass_rate * 100) + "%）· " + (d.elapsed_ms / 1000).toFixed(1) + " 秒";
    const means = Object.entries(s.metric_means).map(([k, v]) =>
      "<tr><td>" + esc(k) + "</td><td>" + bar(v) + "</td></tr>").join("");
    const cats = Object.entries(s.by_category).map(([k, v]) =>
      "<tr><td>" + esc(k) + "</td><td class='num'>" + v.passed + "/" + v.total + "</td><td class='num'>" +
      (v.pass_rate * 100).toFixed(0) + "%</td></tr>").join("");
    const fails = d.cases.filter((c) => !c.passed);
    $("#evout").innerHTML =
      "<h3 style='font-size:13px;margin:0 0 6px'>六维均值</h3><table>" + means + "</table>" +
      "<h3 style='font-size:13px;margin:14px 0 6px'>分类</h3><table><thead><tr><th>类别</th><th class='num'>通过</th><th class='num'>通过率</th></tr></thead><tbody>" + cats + "</tbody></table>" +
      (fails.length ? "<h3 style='font-size:13px;margin:14px 0 6px;color:var(--bad)'>没过的用例</h3>" +
        "<table><thead><tr><th>用例</th><th>类别</th><th>失败原因</th></tr></thead><tbody>" +
        fails.map((c) => "<tr><td class='mono'>" + esc(c.id) + "</td><td>" + esc(c.category) + "</td><td>" + esc(c.reason) + "</td></tr>").join("") +
        "</tbody></table>" : "<p class='sub' style='margin-top:14px'>这一批全过。</p>");
  } catch (e) {
    $("#evout").innerHTML = "<div class='err'>" + esc(e.message) + "</div>";
  } finally { $("#evrun").disabled = false; }
}
$("#evrun").onclick = runEval;

// ---------- mutant ----------
async function runMutant(){
  $("#mtrun").disabled = true;
  $("#mtout").innerHTML = "<div class='spin'>正在注入缺陷并重跑…（要跑两遍）</div>";
  try {
    const d = await api("/api/mutant", {mutant: $("#mtid").value, category: $("#mtcat").value, limit: Number($("#mtlim").value) || 0});
    $("#mtnote").textContent = d.partial
      ? ("只跑了 " + d.total + " 条 —— 子集，不作结论")
      : (d.caught ? "抓到了 ✓" : "没抓到 ✗ —— 这是一个评测盲区");
    const rows = Object.entries(d.deltas).map(([k, v]) => {
      const cls = v < -1e-9 ? "down" : "up";
      return "<tr><td>" + esc(k) + "</td><td class='num'>" + d.baseline_means[k].toFixed(3) +
        "</td><td class='num'>" + d.mutant_means[k].toFixed(3) +
        "</td><td class='num " + cls + "'>" + (v > 0 ? "+" : "") + v.toFixed(3) +
        (d.targets.includes(k) ? " <b>← 目标</b>" : "") + "</td></tr>";
    }).join("");
    let verdict;
    if (d.partial) {
      verdict = "⚠️ 这次只跑了 " + d.total + " 条（子集）。" +
        "子集上的\"抓没抓到\"<b>不作数</b> —— 把上限调成 0 跑全量才有结论。";
    } else if (!d.caught) {
      verdict = "⚠️ 注入缺陷后评测<b>没掉分</b> —— 说明这一维看不见这类错误。这是评测盲区，不是被测系统没问题。";
    } else if (d.weak) {
      verdict = "评测对这类缺陷是敏感的，但目标维度只动了 " +
        Math.abs(d.deltas[d.targets[0]]).toFixed(3) +
        " —— <b>这一维的覆盖面很薄</b>。掉分小有两种成因：这个能力不重要，或者这一维根本没在看。";
    } else {
      verdict = "评测对这一类缺陷是敏感的 —— 分数掉了 " +
        Math.abs(d.deltas[d.targets[0]]).toFixed(3) + "，而且掉在声明的目标维度上。";
    }
    $("#mtout").innerHTML =
      "<dl class='kv'><dt>缺陷</dt><dd>" + esc(d.id) + "</dd>" +
      "<dt>类别</dt><dd>" + esc(d.kind) + "</dd>" +
      "<dt>目标维度</dt><dd>" + esc(d.targets.join(" / ")) + "</dd>" +
      "<dt>通过率</dt><dd>" + d.baseline_pass + " → <b>" + d.mutant_pass + "</b>（" + d.total + " 条）</dd></dl>" +
      "<table style='margin-top:12px'><thead><tr><th>维度</th><th class='num'>基线</th><th class='num'>注入后</th><th class='num'>Δ</th></tr></thead><tbody>" + rows + "</tbody></table>" +
      "<p class='sub' style='margin-top:10px'>" + verdict + "</p>";
  } catch (e) {
    $("#mtout").innerHTML = "<div class='err'>" + esc(e.message) + "</div>";
  } finally { $("#mtrun").disabled = false; }
}
$("#mtrun").onclick = runMutant;

boot().catch((e) => {
  document.body.insertAdjacentHTML("afterbegin",
    "<div class='err' style='padding:16px 24px'>初始化失败：" + esc(e.message) + "</div>");
});
</script>
</body>
</html>
"""
