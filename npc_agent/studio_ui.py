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
<meta name="color-scheme" content="dark light">
<title>NPC Agent Studio · 活世界实验台</title>
<style>
:root{
  --bg:#0b0d12; --panel:#12161f; --panel2:#181e29; --ink:#ece5d8; --dim:#98a1b2;
  --line:#28303f; --soft:rgba(255,255,255,.025); --soft2:rgba(255,255,255,.06);
  --amber:#e7b978; --amber2:#f3d6a6; --teal:#74d6c2; --violet:#b2a5ff; --rose:#ef8f8f;
  --ok:#6fd6a0; --warn:#e7b978; --bad:#ef8f8f;
  --shadow:0 20px 44px -30px #000;
  --cmd:rgba(0,0,0,.18);
  --sans:system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
  --serif:ui-serif,Georgia,"Songti SC","Noto Serif SC",serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
}
@media (prefers-color-scheme: light){
  :root{
    --bg:#f5f1e8; --panel:#fffdf8; --panel2:#f1ece1; --ink:#201d18; --dim:#6f685c;
    --line:#ded6c7; --soft:rgba(32,29,24,.025); --soft2:rgba(32,29,24,.06);
    --amber:#a9711f; --amber2:#8a5a14; --teal:#177f70; --violet:#6152b8; --rose:#b34b4b;
    --ok:#1f7f4d; --warn:#a9711f; --bad:#b34b4b;
    --shadow:0 20px 44px -32px rgba(60,48,30,.35);
    --cmd:rgba(32,29,24,.045);
  }
}
*{box-sizing:border-box}
html,body{min-height:100%}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.62 var(--sans);-webkit-font-smoothing:antialiased}
a{color:var(--amber)}
code,pre,.mono{font-family:var(--mono);font-size:12.5px}
::selection{background:rgba(231,185,120,.28)}

.sky{position:fixed;inset:0;z-index:-1;pointer-events:none;background:
  radial-gradient(1100px 620px at 8% -8%, rgba(231,185,120,.16), transparent 62%),
  radial-gradient(900px 520px at 102% 4%, rgba(116,214,194,.10), transparent 58%),
  radial-gradient(1000px 700px at 50% 118%, rgba(178,165,255,.08), transparent 60%),
  linear-gradient(180deg,#0b0d12 0%,#0d1016 55%,#0a0c11 100%)}
.sky::after{content:"";position:absolute;inset:0;
  background-image:linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px),
    linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px);
  background-size:64px 64px;
  -webkit-mask-image:linear-gradient(180deg,rgba(0,0,0,.7),transparent 72%);
  mask-image:linear-gradient(180deg,rgba(0,0,0,.7),transparent 72%)}
@media (prefers-color-scheme: light){
  .sky{background:
    radial-gradient(1100px 620px at 8% -8%, rgba(169,113,31,.13), transparent 62%),
    radial-gradient(900px 520px at 102% 4%, rgba(23,127,112,.10), transparent 58%),
    linear-gradient(180deg,#f7f3ea 0%,#f2ede2 60%,#efe9dd 100%)}
  .sky::after{background-image:linear-gradient(rgba(32,29,24,.03) 1px,transparent 1px),
    linear-gradient(90deg,rgba(32,29,24,.03) 1px,transparent 1px)}
}

.masthead{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;
  padding:28px 32px 14px;max-width:1440px;margin:0 auto}
.brand{display:flex;gap:16px;align-items:center;min-width:0}
.sigil{width:52px;height:52px;flex:0 0 auto;border-radius:16px;display:grid;place-items:center;
  font-size:22px;color:var(--amber2);border:1px solid rgba(231,185,120,.35);
  background:radial-gradient(circle at 30% 25%,rgba(231,185,120,.30),rgba(231,185,120,.06) 65%),#141924;
  box-shadow:0 0 32px rgba(231,185,120,.12),inset 0 0 18px rgba(231,185,120,.06)}
@media (prefers-color-scheme: light){.sigil{background:radial-gradient(circle at 30% 25%,rgba(169,113,31,.22),rgba(169,113,31,.05) 65%),#fffaf0}}
.eyebrow{font:600 10.5px/1 var(--mono);letter-spacing:.22em;color:var(--amber);text-transform:uppercase}
h1{margin:6px 0 4px;font:600 26px/1.1 var(--serif);letter-spacing:.02em}
.tagline{margin:0;color:var(--dim);font-size:13px}
.mode{display:flex;gap:10px;align-items:center;padding:10px 14px;border:1px solid var(--line);
  border-radius:999px;background:linear-gradient(180deg,var(--panel),var(--panel2));flex:0 0 auto}
.lamp{width:8px;height:8px;border-radius:50%;background:var(--amber);
  box-shadow:0 0 0 4px rgba(231,185,120,.12),0 0 14px var(--amber);animation:breathe 3.2s ease-in-out infinite}
.mode.on .lamp{background:var(--ok);box-shadow:0 0 0 4px rgba(111,214,160,.12),0 0 14px var(--ok)}
.mode b{display:block;font-size:12.5px;line-height:1.35}
.mode span{display:block;color:var(--dim);font-size:11.5px;line-height:1.35}

.ticker{display:flex;gap:22px;flex-wrap:wrap;align-items:center;
  padding:0 32px 14px;max-width:1440px;margin:0 auto;border-bottom:1px solid var(--line)}
.tk{font-size:12px;color:var(--dim)}
.tk i{font-style:normal;margin-right:7px;letter-spacing:.08em}
.tk b{color:var(--ink);font-family:var(--mono);font-weight:600}
.tk.right{margin-left:auto;font-size:11.5px}

nav{position:sticky;top:0;z-index:20;display:flex;gap:8px;padding:12px 32px;
  max-width:1440px;margin:0 auto;overflow-x:auto;
  background:linear-gradient(180deg,rgba(11,13,18,.92),rgba(11,13,18,.72));
  -webkit-backdrop-filter:blur(14px);backdrop-filter:blur(14px);border-bottom:1px solid var(--line)}
@media (prefers-color-scheme: light){nav{background:linear-gradient(180deg,rgba(245,241,232,.94),rgba(245,241,232,.78))}}
nav button{appearance:none;border:1px solid transparent;background:none;color:var(--dim);
  font:inherit;font-size:12.5px;padding:9px 15px;border-radius:999px;cursor:pointer;
  display:flex;gap:8px;align-items:center;white-space:nowrap;transition:.15s}
nav button b{font-family:var(--mono);font-size:10.5px;color:var(--dim);font-weight:500}
nav button:hover{color:var(--ink);background:var(--panel2)}
nav button.sel{color:var(--ink);border-color:rgba(231,185,120,.38);
  background:linear-gradient(180deg,rgba(231,185,120,.16),rgba(231,185,120,.05));
  box-shadow:inset 0 1px 0 rgba(255,255,255,.04)}
nav button.sel b{color:var(--amber)}

main{padding:22px 32px 56px;max-width:1440px;margin:0 auto}
.panel{display:none}
.panel.sel{display:block;animation:rise .28s ease both}
.card{position:relative;background:linear-gradient(180deg,var(--panel),var(--panel2));
  border:1px solid var(--line);border-radius:16px;padding:16px 18px;margin-bottom:14px;
  box-shadow:var(--shadow),inset 0 1px 0 rgba(255,255,255,.03)}
.card-head{display:flex;justify-content:space-between;gap:12px;align-items:baseline;margin-bottom:10px}
.card-head h3{margin:0;font-size:14px;font-weight:600;letter-spacing:.01em}
.card-head .hint{margin:0;font-size:11px}
.hint{color:var(--dim);font-size:11.5px;margin:10px 0 0;line-height:1.6}
.hint.warn{color:var(--warn)}
.muted{color:var(--dim);font-size:11.5px}
.err{color:var(--bad);font-size:12px;white-space:pre-wrap;line-height:1.6}
.up{color:var(--ok)}.down{color:var(--bad)}

label{display:block;color:var(--dim);font-size:10.5px;letter-spacing:.06em;margin-bottom:5px}
select,input,textarea,button{font:inherit;color:var(--ink);background:var(--panel2);
  border:1px solid var(--line);border-radius:10px;padding:8px 11px}
select,input{min-width:0}
textarea{width:100%;resize:vertical;min-height:56px}
select:focus,input:focus,textarea:focus{outline:none;border-color:rgba(231,185,120,.55);
  box-shadow:0 0 0 3px rgba(231,185,120,.10)}
button{cursor:pointer}
button.primary{background:linear-gradient(180deg,#ecc689,#d9a55e);border-color:#e3b675;
  color:#22180c;font-weight:700;box-shadow:0 8px 24px -12px rgba(231,185,120,.65)}
button.primary:hover{filter:brightness(1.06)}
button.primary:disabled{opacity:.55;cursor:progress}
button.ghost{background:var(--soft);border-color:var(--line);color:var(--ink)}
button.ghost:hover{border-color:rgba(231,185,120,.38);color:var(--amber)}

.world-stage{padding:20px}
.world-top{display:flex;gap:24px;justify-content:space-between;align-items:flex-start;flex-wrap:wrap}
.world-title{min-width:0}
.world-title h2{font:600 25px/1.2 var(--serif);margin:6px 0;letter-spacing:.01em}
.world-title p{margin:0;color:var(--dim);font-size:12.5px;max-width:660px}
.world-pick{display:flex;gap:10px;align-items:flex-end;flex-wrap:wrap}
.world-badge{display:inline-flex;align-items:center;gap:6px;padding:7px 11px;border-radius:999px;
  border:1px solid rgba(116,214,194,.3);background:rgba(116,214,194,.07);
  color:var(--teal);font:11.5px var(--mono)}
.world-badge::before{content:"";width:6px;height:6px;border-radius:50%;background:var(--teal);
  box-shadow:0 0 10px var(--teal)}

.roster{display:grid;grid-template-columns:repeat(auto-fit,minmax(228px,1fr));gap:10px;margin:18px 0 4px}
.npc{position:relative;display:flex;gap:11px;align-items:flex-start;padding:12px;
  border:1px solid var(--line);border-radius:14px;overflow:hidden;
  background:linear-gradient(135deg,var(--soft),transparent 60%),var(--panel2)}
.npc::after{content:"";position:absolute;right:-30px;top:-30px;width:90px;height:90px;border-radius:50%;
  background:radial-gradient(circle,hsl(var(--h) 60% 60% / .18),transparent 70%);pointer-events:none}
.npc-main{min-width:0;flex:1}
.ava{width:40px;height:40px;flex:0 0 auto;border-radius:13px;display:grid;place-items:center;
  font:600 17px/1 var(--serif);border:1px solid hsl(var(--h) 55% 60% / .35);
  background:hsl(var(--h) 55% 60% / .12);color:hsl(var(--h) 75% 76%)}
.ava.player{--h:214;background:rgba(231,185,120,.12);border-color:rgba(231,185,120,.35);color:var(--amber2)}
.npc-id b{display:block;font-size:14px}
.npc-id span{display:block;color:var(--dim);font-size:11.5px;margin-top:2px}
.npc-stats{display:flex;gap:10px;flex-wrap:wrap;margin-top:7px;font-size:10.5px;color:var(--dim)}
.npc-stats b{font-family:var(--mono);color:var(--ink);font-weight:600}
.npc-goal{margin-top:5px;color:var(--dim);font-size:11px;line-height:1.5;
  display:-webkit-box;-webkit-line-clamp:1;-webkit-box-orient:vertical;overflow:hidden}
.npc-bar{height:3px;border-radius:99px;background:var(--soft2);margin-top:8px;overflow:hidden}
.npc-bar i{display:block;height:100%;border-radius:99px;
  background:linear-gradient(90deg,var(--teal),var(--amber));transition:width .4s ease}
.npc-know{margin-top:5px;color:var(--dim);font:10px var(--mono)}

.composer{display:flex;gap:10px;align-items:flex-end;flex-wrap:wrap;margin-top:14px;
  padding-top:14px;border-top:1px dashed var(--line)}
.composer .ava{width:34px;height:34px;border-radius:11px;font-size:14px}
.composer-who,.composer-say,.composer-scope{display:flex;flex-direction:column}
.composer-say{flex:1;min-width:220px}
.composer-say input{width:100%}
.composer-scope select{max-width:210px}
.composer-actions{display:flex;gap:8px;flex-wrap:wrap}

.scope-tag{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:999px;
  font:10.5px/1.5 var(--mono);color:var(--dim);border:1px solid var(--line);background:var(--soft)}
.scope-tag.private{color:var(--violet);border-color:rgba(178,165,255,.34);background:rgba(178,165,255,.08)}
.scope-tag.group{color:var(--teal);border-color:rgba(116,214,194,.34);background:rgba(116,214,194,.08)}
.scope-tag.bad{color:var(--bad);border-color:rgba(239,143,143,.32);background:rgba(239,143,143,.07)}
.group-badge{display:inline-flex;align-items:center;margin-left:6px;padding:1px 7px;border-radius:999px;
  font:9.5px/1.6 var(--mono);font-weight:500;color:var(--teal);
  border:1px solid rgba(116,214,194,.3);background:rgba(116,214,194,.07);vertical-align:1px}
.bubble-player.private{border-color:rgba(178,165,255,.4);
  background:linear-gradient(180deg,rgba(178,165,255,.15),rgba(178,165,255,.05))}
.bubble-player.group{border-color:rgba(116,214,194,.38);
  background:linear-gradient(180deg,rgba(116,214,194,.13),rgba(116,214,194,.04))}
.bubble-player .scope-tag{display:inline-flex;margin:0 0 3px}

.columns{display:grid;grid-template-columns:minmax(0,1.58fr) minmax(336px,.92fr);
  gap:14px;align-items:start;margin-top:14px}
@media (max-width:1080px){.columns{grid-template-columns:1fr}}
.side{min-width:0}
.side .card{overflow:hidden}

.log{max-height:62vh;overflow:auto;padding-right:6px;scroll-behavior:smooth}
.beat{padding:14px 0;border-top:1px dashed var(--line);animation:rise .3s ease both}
.beat:first-child{border-top:0;padding-top:2px}
.beat-head{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin-bottom:10px}
.turn-no{font:600 10px/1 var(--mono);letter-spacing:.12em;color:var(--dim);
  background:var(--soft2);border:1px solid var(--line);padding:4px 8px;border-radius:999px}
.bubble-player{margin-left:auto;max-width:80%;padding:9px 13px;font-size:13.5px;line-height:1.55;
  border-radius:14px 14px 4px 14px;color:var(--amber2);
  background:linear-gradient(180deg,rgba(231,185,120,.16),rgba(231,185,120,.07));
  border:1px solid rgba(231,185,120,.32)}
.bubble-player i{display:block;font:10.5px var(--mono);font-style:normal;color:var(--dim);margin-bottom:3px}
.idle-tag{font:11px var(--mono);color:var(--dim);border:1px dashed var(--line);
  border-radius:999px;padding:3px 9px}
.turn{display:flex;gap:10px;margin:9px 0 0;animation:rise .3s ease both}
.turn .ava{width:34px;height:34px;border-radius:11px;font-size:14px;margin-top:2px}
.turn-main{flex:1;min-width:0;padding:10px 12px;border:1px solid var(--line);
  border-radius:4px 14px 14px 14px;background:var(--soft)}
.turn-name{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:12px}
.turn-name b{font-size:12.5px}
.decision{margin-top:5px;color:var(--dim);font-size:11px;line-height:1.55;
  border-left:2px solid var(--line);padding-left:8px}
.say{margin-top:6px;font-size:14.5px;line-height:1.65;color:var(--ink)}
.acts{display:flex;flex-direction:column;gap:4px;margin-top:7px}
.act{display:inline-flex;gap:7px;align-items:baseline;width:fit-content;max-width:100%;
  padding:3px 8px;border:1px solid transparent;border-radius:8px;
  font-family:var(--mono);font-size:11px;line-height:1.5}
.act.ok{color:var(--ok);border-color:rgba(111,214,160,.22);background:rgba(111,214,160,.06)}
.act.yield{color:var(--dim);border-color:var(--line);background:var(--soft)}
.act.no{color:var(--bad);border-color:rgba(239,143,143,.26);background:rgba(239,143,143,.06)}
.pills{display:flex;flex-wrap:wrap;gap:5px;margin-top:7px;align-items:center}
.pill{display:inline-flex;gap:5px;align-items:center;padding:2px 8px;border-radius:999px;
  font-size:10.5px;line-height:1.5;color:var(--dim);border:1px solid var(--line);background:var(--soft)}
.pill.warn{color:var(--warn);border-color:rgba(231,185,120,.32);background:rgba(231,185,120,.07)}
.pill.bad{color:var(--bad);border-color:rgba(239,143,143,.32);background:rgba(239,143,143,.07)}
.pill.ok{color:var(--ok);border-color:rgba(111,214,160,.32);background:rgba(111,214,160,.07)}
.rep{margin:0 0 8px;font-size:11.5px;color:var(--dim);line-height:1.6}
.rep .pill{margin-right:4px}
.rep-ex{margin-top:5px;font-size:10.5px;color:var(--dim);line-height:1.6}

.empty{display:flex;flex-direction:column;align-items:center;gap:9px;padding:34px 16px;
  color:var(--dim);font-size:12px;text-align:center;line-height:1.7}
.empty b{color:var(--ink);font-size:13px}
.empty code{padding:6px 10px;border:1px dashed var(--line);border-radius:9px;
  background:var(--soft);color:var(--amber2);max-width:100%;white-space:pre-wrap}
.constellation{width:120px;height:72px;overflow:visible;margin-bottom:2px}
.constellation circle{fill:var(--amber);opacity:.8}
.constellation .halo{fill:none;stroke:rgba(231,185,120,.25);stroke-width:1;stroke-dasharray:3 6}
.constellation path{stroke:var(--line);stroke-width:1;stroke-dasharray:3 5;fill:none}

.graph svg{display:block;width:100%;height:auto;overflow:visible}
.graph .stage-ring{fill:none;stroke:var(--line);stroke-dasharray:3 7}
.graph .edge{stroke-linecap:round}
.graph .edge.player{stroke:rgba(231,185,120,.4)}
.graph .edge.peer{stroke:rgba(116,214,194,.36);stroke-dasharray:4 4}
.graph text{font-family:var(--sans);paint-order:stroke;stroke:var(--panel);stroke-width:3px;stroke-linejoin:round}
.graph .node-glyph{font:600 15px var(--serif);text-anchor:middle}
.graph .node-name{font-size:10.5px;fill:var(--dim);text-anchor:middle}
.graph .edge-label{font:500 9.5px var(--mono);fill:var(--dim);text-anchor:middle}
.graph .player-name{font:600 11px var(--mono);fill:var(--amber2);text-anchor:middle}
.graph-key{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;color:var(--dim);font-size:10.5px}
.graph-key i{display:inline-block;width:14px;height:0;border-top:2px solid;margin-right:5px;vertical-align:middle}
.graph-key .k-player{border-color:rgba(231,185,120,.65)}
.graph-key .k-peer{border-color:rgba(116,214,194,.65);border-top-style:dashed}
.peer-list{display:grid;gap:4px;margin-top:9px;font-size:10.5px;color:var(--dim)}
.peer-list b{color:var(--ink);font-family:var(--mono);font-weight:500}

.stat-row{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px}
.stat{padding:9px 11px;border:1px solid var(--line);border-radius:12px;background:var(--soft)}
.stat span{display:block;color:var(--dim);font-size:10px;letter-spacing:.06em}
.stat b{display:block;font:600 19px/1.25 var(--mono);margin-top:2px}
.stat b.small{font-size:11.5px;line-height:1.5;word-break:break-word}
.subhead{display:flex;justify-content:space-between;align-items:baseline;gap:10px;
  color:var(--dim);font-size:10px;letter-spacing:.12em;text-transform:uppercase;margin:12px 0 5px}
.obj{display:flex;justify-content:space-between;gap:12px;padding:6px 0;
  border-bottom:1px solid var(--soft2);font-size:11.5px}
.obj-id{color:var(--ink);font-family:var(--mono);font-size:11px}
.obj-val{color:var(--dim);font-family:var(--mono);font-size:10.5px;text-align:right;word-break:break-all}
.planner-bar{display:flex;height:8px;border-radius:99px;overflow:hidden;background:var(--soft2);margin-top:6px}
.planner-bar i{display:block;height:100%}
.planner-row{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;
  font-size:10.5px;color:var(--dim);margin-top:5px}

.mem-npc{border:1px solid var(--line);border-radius:12px;margin-bottom:8px;
  background:var(--soft);overflow:hidden}
.mem-npc summary{list-style:none;display:flex;gap:8px;align-items:center;padding:9px 10px;cursor:pointer}
.mem-npc summary::-webkit-details-marker{display:none}
.mem-npc summary .ava{width:28px;height:28px;border-radius:9px;font-size:12px}
.mem-npc summary b{font-size:12.5px}
.mem-tally{margin-left:auto;color:var(--dim);font:10px var(--mono)}
.mem-body{padding:0 10px 10px;border-top:1px dashed var(--line)}
.meters{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin:9px 0}
.meter{font-size:10px;color:var(--dim)}
.meter .track{height:5px;border-radius:99px;background:var(--soft2);overflow:hidden;margin:4px 0 3px}
.meter .track i{display:block;height:100%;border-radius:99px}
.meter.fid .track i{background:linear-gradient(90deg,var(--teal),#a8e7dc)}
.meter.conf .track i{background:linear-gradient(90deg,var(--amber),var(--amber2))}
.meter b{font-family:var(--mono);color:var(--ink);font-weight:500}
.mem-line{color:var(--dim);font-size:10.5px;line-height:1.6;margin:5px 0}
.mem-rec{padding:7px 0;border-top:1px solid var(--soft2);font-size:11.5px}
.mem-rec .content{line-height:1.55;word-break:break-word}
.mem-rec .gone{text-decoration:line-through;opacity:.6}
.timeline{list-style:none;margin:0;padding:0 0 0 12px;border-left:1px solid var(--line)}
.timeline li{position:relative;padding:0 0 10px 11px;color:var(--dim);font-size:11.5px;line-height:1.55}
.timeline li::before{content:"";position:absolute;left:-4px;top:5px;width:7px;height:7px;border-radius:50%;
  background:var(--amber);box-shadow:0 0 10px rgba(231,185,120,.5)}
.timeline li:last-child{padding-bottom:0}
.speech-row{display:grid;grid-template-columns:62px 1fr 38px;gap:8px;align-items:center;margin:7px 0;font-size:11.5px}
.speech-row > span{color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.speech-row em{font-style:normal;font-family:var(--mono);color:var(--ink);text-align:right}
.speech-track{height:6px;border-radius:99px;background:var(--soft2);overflow:hidden}
.speech-track i{display:block;height:100%;border-radius:99px;
  background:linear-gradient(90deg,hsl(var(--h) 55% 55%),hsl(var(--h) 75% 70%));transition:width .4s ease}

.eval-hero{display:flex;gap:24px;align-items:center;flex-wrap:wrap;padding:4px 0 12px}
.ring{--p:0%;width:128px;height:128px;flex:0 0 auto;border-radius:50%;position:relative;display:grid;place-items:center;
  background:conic-gradient(var(--teal) var(--p),var(--soft2) 0)}
.ring::before{content:"";position:absolute;inset:9px;border-radius:50%;background:var(--panel)}
.ring-in{position:relative;text-align:center;z-index:1}
.ring-in b{display:block;font:600 26px/1 var(--mono);color:var(--ink)}
.ring-in span{display:block;color:var(--dim);font-size:10px;letter-spacing:.1em;margin-top:4px}
.eval-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(108px,1fr));gap:9px;flex:1;min-width:200px}
.eval-stat{padding:10px 12px;border:1px solid var(--line);border-radius:12px;background:var(--soft)}
.eval-stat span{display:block;color:var(--dim);font-size:10px;letter-spacing:.05em}
.eval-stat b{display:block;font:600 17px/1.35 var(--mono);margin-top:2px}
.eval-stat b.small{font-size:11.5px;line-height:1.5;word-break:break-word}
.metric-row{display:grid;grid-template-columns:84px 1fr 54px;gap:10px;align-items:center;margin:7px 0;font-size:11.5px}
.metric-row label{color:var(--dim);font-family:var(--mono);font-size:11px;margin:0}
.metric-row b{font-family:var(--mono);font-size:11px;text-align:right;color:var(--ink);font-weight:500}
.bar{position:relative;height:8px;border-radius:99px;background:var(--soft2);overflow:hidden}
.bar > i{position:absolute;inset:0 auto 0 0;border-radius:99px;
  background:linear-gradient(90deg,var(--teal),var(--amber));transition:width .4s ease}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--soft2);vertical-align:top}
th{color:var(--dim);font-weight:500;font-size:10px;letter-spacing:.08em;text-transform:uppercase}
td.num,th.num{text-align:right;font-family:var(--mono)}
.table-wrap{overflow-x:auto}
.details{margin-top:12px;border:1px solid rgba(239,143,143,.28);border-radius:12px;
  padding:10px 12px;background:rgba(239,143,143,.04)}
.details summary{cursor:pointer;color:var(--bad);font-size:12px;font-weight:600}

.verdict{display:flex;gap:12px;align-items:flex-start;padding:13px 15px;border-radius:14px;
  border:1px solid var(--line);background:var(--soft);margin-bottom:13px}
.verdict .v-ico{font-size:20px;line-height:1.2;flex:0 0 auto}
.verdict b{display:block;font-size:13.5px;margin-bottom:3px}
.verdict p{margin:0;color:var(--dim);font-size:11.5px;line-height:1.65}
.verdict.ok{border-color:rgba(111,214,160,.36);
  background:linear-gradient(135deg,rgba(111,214,160,.10),transparent 62%)}
.verdict.ok .v-ico{color:var(--ok)}
.verdict.warn{border-color:rgba(231,185,120,.36);
  background:linear-gradient(135deg,rgba(231,185,120,.10),transparent 62%)}
.verdict.warn .v-ico{color:var(--warn)}
.verdict.bad{border-color:rgba(239,143,143,.36);
  background:linear-gradient(135deg,rgba(239,143,143,.10),transparent 62%)}
.verdict.bad .v-ico{color:var(--bad)}
.defect{display:flex;gap:7px;flex-wrap:wrap;align-items:center;margin:2px 0 12px}
.delta.up{color:var(--ok)}.delta.down{color:var(--bad)}

.rep-list{display:grid;gap:8px}
.rep-card{display:flex;gap:11px;align-items:center;padding:11px 12px;border:1px solid var(--line);
  border-radius:12px;background:var(--soft);text-decoration:none;color:inherit;transition:.15s}
.rep-card:hover{border-color:rgba(231,185,120,.42);transform:translateY(-1px)}
.rep-card.unknown{border-style:dashed}
.rep-ico{width:34px;height:34px;flex:0 0 auto;border-radius:10px;display:grid;place-items:center;
  border:1px solid var(--line);color:var(--amber);background:rgba(231,185,120,.07);font-size:15px}
.rep-main{flex:1;min-width:0}
.rep-main b{display:block;font-size:12.5px}
.rep-main span{display:block;color:var(--dim);font-size:11px;margin-top:2px;line-height:1.5}
.rep-cov{font-family:var(--mono);font-size:10px;color:var(--dim);border:1px solid var(--line);
  border-radius:999px;padding:3px 8px;white-space:nowrap}
.cmd{margin:0;padding:14px;border:1px solid var(--line);border-radius:12px;background:var(--cmd);
  color:var(--dim);font:11.5px/1.9 var(--mono);overflow:auto;white-space:pre}
footer{color:var(--dim);font-size:11px;padding:0 32px 40px;max-width:1440px;margin:0 auto}

@keyframes breathe{0%,100%{opacity:.62;transform:scale(.9)}50%{opacity:1;transform:scale(1.06)}}
@keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important;scroll-behavior:auto!important}}
@media (max-width:720px){
  .masthead{padding:20px 16px 12px;align-items:flex-start;flex-direction:column}
  .ticker{padding:0 16px 12px;gap:14px}
  .tk.right{margin-left:0}
  nav{padding:10px 16px}
  main{padding:16px 16px 40px}
  .world-stage{padding:15px}
  .world-title h2{font-size:21px}
  .bubble-player{max-width:100%}
  .col{grid-template-columns:1fr}
}
</style>
</head>
<body>
<div class="sky" aria-hidden="true"></div>

<header class="masthead">
  <div class="brand">
    <div class="sigil" aria-hidden="true">✦</div>
    <div class="brand-text">
      <div class="eyebrow">Living World Lab · 星屿</div>
      <h1>NPC Agent Studio</h1>
      <p class="tagline">会「玩」的 AI NPC —— 从一句话，到世界真的变了。</p>
    </div>
  </div>
  <div class="mode" id="modeBox">
    <span class="lamp" aria-hidden="true"></span>
    <div><b id="mode">读取中…</b><span id="modelLine">连接本机服务</span></div>
  </div>
</header>

<div class="ticker">
  <span class="tk"><i>六维回归</i><b id="counts">—</b></span>
  <span class="tk"><i>世界基准</i><b id="worldCount">—</b></span>
  <span class="tk"><i>故障注入</i><b id="mutantCount">—</b></span>
  <span class="tk right">离线可跑 · 同一段输入必然复现同一段输出</span>
</div>

<nav>
  <button data-tab="chat" class="sel"><b>01</b>世界剧场</button>
  <button data-tab="eval"><b>02</b>评测工坊</button>
  <button data-tab="mutant"><b>03</b>变异试验</button>
  <button data-tab="reports"><b>04</b>报告档案</button>
</nav>

<main>

<!-- 01 世界剧场 -->
<section class="panel sel" id="p-chat">
  <div class="card world-stage">
    <div class="world-top">
      <div class="world-title">
        <div class="eyebrow">当前世界</div>
        <h2 id="scnName">—</h2>
        <p id="chatnote">选一个场景，让 NPC 开始自己的日子。</p>
      </div>
      <div class="world-pick">
        <div><label for="scn">场景</label><select id="scn"></select></div>
        <span class="world-badge" id="worldLabel">—</span>
      </div>
    </div>
    <div class="roster" id="roster"></div>
    <div class="composer">
      <span class="ava player" id="whoAva">?</span>
      <div class="composer-who"><label for="who">以谁的身份说话</label><select id="who"></select></div>
      <div class="composer-scope"><label for="scope">说给谁听</label><select id="scope"></select></div>
      <div class="composer-say"><label for="msg">说点什么</label><input id="msg" placeholder="例如：我第一次来，这里怎么点单呀？"></div>
      <div class="composer-actions">
        <button class="primary" id="send">发送</button>
        <button class="ghost" id="idle" title="没人说话，看 NPC 会不会自己找事做">空转一轮</button>
        <button class="ghost" id="seed" title="填入一条场景示例台词">示例</button>
        <button class="ghost" id="reset">重开</button>
      </div>
    </div>
    <p class="hint" id="idleHint"></p>
  </div>

  <div class="columns">
    <div class="card theater">
      <div class="card-head">
        <div><h3>世界剧场</h3><span class="hint" id="theaterHint">每轮所有 NPC 依次行动，但最多一个人开口</span></div>
        <div class="hint">发言权由 Cast 统一收放</div>
      </div>
      <p class="rep" id="repnote"></p>
      <div class="log" id="log">
        <div class="empty">
          <svg class="constellation" viewBox="0 0 120 72" aria-hidden="true">
            <circle class="halo" cx="60" cy="36" r="30"></circle>
            <path d="M22 48 L58 20 L96 46"></path>
            <circle cx="22" cy="48" r="2.6"></circle>
            <circle cx="58" cy="20" r="3.4"></circle>
            <circle cx="96" cy="46" r="2.6"></circle>
            <circle cx="58" cy="56" r="1.5"></circle>
          </svg>
          <b>世界还没开始</b>
          <span>选一个场景，说一句话。NPC 会各自决定、各自行动，<br>同一分组里每轮最多一个人开口，不同分组可以同时说话。</span>
          <code>今天这里挺热闹的。</code>
        </div>
      </div>
    </div>

    <aside class="side">
      <div class="card">
        <div class="card-head"><h3>关系星图</h3><span class="hint">玩家好感 · NPC 默契</span></div>
        <div id="graph" class="graph"><div class="muted">世界还没开始。</div></div>
        <div class="graph-key">
          <span><i class="k-player"></i>玩家 ↔ NPC 好感</span>
          <span><i class="k-peer"></i>NPC ↔ NPC 默契</span>
        </div>
        <div id="peerlist" class="peer-list"></div>
      </div>

      <div class="card">
        <div class="card-head"><h3>世界脉搏</h3><span class="hint">结束时快照</span></div>
        <div id="world"><div class="muted">还没有脉搏。</div></div>
      </div>

      <div class="card">
        <div class="card-head"><h3>记忆代谢</h3><span class="hint">来源 · 保真 · 确信 · 遗忘</span></div>
        <div id="mems"><div class="muted">记忆还是空的。</div></div>
      </div>

      <div class="card">
        <div class="card-head"><h3>编年史</h3><span class="hint">这个世界发生过什么</span></div>
        <div id="chronicle"><div class="muted">还没发生过什么。</div></div>
      </div>

      <div class="card">
        <div class="card-head"><h3>发言配平</h3><span class="hint">多 NPC 最该盯的数</span></div>
        <div id="speech"><div class="muted">还没人开口。</div></div>
      </div>
    </aside>
  </div>
</section>

<!-- 02 评测 -->
<section class="panel" id="p-eval">
  <div class="card">
    <div class="card-head"><h3>评测工坊</h3><span class="hint">和命令行同一条路：EvalHarness</span></div>
    <div class="composer" style="margin-top:0;padding-top:0;border-top:0">
      <div><label>用例类别</label><select id="evcat"></select></div>
      <div><label>最多跑几条（0 = 全部）</label><input id="evlim" type="number" min="0" value="0" style="width:140px"></div>
      <div class="composer-actions"><button class="primary" id="evrun">跑一遍</button></div>
      <span class="hint" style="margin:0">离线全量约 6 秒；不与命令行各算一份，数字必须逐字一致。</span>
    </div>
  </div>
  <div class="card">
    <div class="card-head"><h3>结果</h3><span class="hint" id="evnote"></span></div>
    <div id="evout"><div class="empty"><b>还没跑</b><span>点「跑一遍」，看六维基线和分类通过率。</span></div></div>
  </div>
</section>

<!-- 03 变异 -->
<section class="panel" id="p-mutant">
  <div class="card">
    <div class="card-head"><h3>变异试验</h3><span class="hint">往 Agent 里注入一个故意的缺陷</span></div>
    <div class="composer" style="margin-top:0;padding-top:0;border-top:0">
      <div class="composer-say"><label>注入哪个缺陷</label><select id="mtid"></select></div>
      <div><label>用例类别</label><select id="mtcat"></select></div>
      <div><label>用例上限（0 = 全部）</label><input id="mtlim" type="number" min="0" value="0" style="width:140px"></div>
      <div class="composer-actions"><button class="primary" id="mtrun">注入并重跑</button></div>
    </div>
    <p class="hint">评测<strong>必须掉分</strong> —— 不掉分就说明这一维是瞎的。这就是「满分是不是『护栏从不报警』」的答案。
      全量要跑两遍，约 12 秒。<br>
      ⚠ 跑子集时"抓没抓到"<strong>不作数</strong>：前 80 条里没有回忆用例，「关掉检索」会显示"没抓到" —— 那是取样造成的，不是评测的结论。</p>
  </div>
  <div class="card">
    <div class="card-head"><h3>对照</h3><span class="hint" id="mtnote"></span></div>
    <div id="mtout"><div class="empty"><b>挑一个缺陷</b><span>看看评测到底抓不抓得住它。</span></div></div>
  </div>
</section>

<!-- 04 报告 -->
<section class="panel" id="p-reports">
  <div class="card">
    <div class="card-head"><h3>离线可复现的报告</h3><span class="hint">不联网、不调模型</span></div>
    <div class="rep-list" id="rep-offline"><div class="muted">（还没有这类报告）</div></div>
  </div>
  <div class="card">
    <div class="card-head"><h3>跑批快照</h3><span class="hint">需要模型 + 额度，冻结在某个时间点</span></div>
    <div class="rep-list" id="rep-snap"><div class="muted">（还没有这类报告）</div></div>
  </div>
  <div class="card">
    <div class="card-head"><h3>命令行等价入口</h3><span class="hint">报告都是这些命令生成的</span></div>
    <pre id="cmds" class="cmd"></pre>
  </div>
</section>

</main>

<footer>本页面自包含（无外链、无 CDN）· 所有数字由本机实时算出，页面里没有硬编码。</footer>

<script>
const $ = (s) => document.querySelector(s);
const esc = (t) => String(t == null ? "" : t)
  .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");

let META = null;
let CURRENT = null;
let EVENTS = [];
let HUES = {};

const HUE_LIST = [36, 168, 268, 204, 320];
const SRC_LABEL = {
  model: "模型规划", heuristic: "启发式规划",
  request_template: "点单模板", scenario_flow: "场景引导"
};

// 星图的节点位置。场景最多三个人、两个玩家 —— 固定坐标比现算三角函数少一层依赖。
const NPC_POS = [[64,74],[160,46],[256,74],[98,116],[222,116]];
const PLR_POS = [[98,186],[222,186]];

async function api(path, body){
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)
  });
  const data = await r.json().catch(() => ({error: "响应不是 JSON"}));
  if (!r.ok) throw new Error(data.error || ("HTTP " + r.status));
  return data;
}

function hueOf(name){
  return HUES[name] || 36;
}

function ava(name, extra){
  return "<span class='ava" + (extra || "") + "' style='--h:" + hueOf(name) + "'>" +
    esc(String(name || "?")[0]) + "</span>";
}

function emptyBlock(title, sub){
  return "<div class='empty'><b>" + esc(title) + "</b><span>" + esc(sub) + "</span></div>";
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
  const online = !!META.llm_available;
  $("#mode").textContent = online ? ("在线模型 · " + META.model) : "离线启发式 · 未配置模型";
  $("#modelLine").textContent = online ? ("provider：" + META.provider) : "不需要网络，同一段输入必然复现";
  $("#modeBox").classList.toggle("on", online);
  $("#counts").textContent = META.case_total + " 条 · " + META.dimensions + " 维";
  $("#worldCount").textContent = META.world_case_total + " 条 · " + META.world_dimensions + " 维";
  $("#mutantCount").textContent = META.mutant_total + " 个缺陷";

  // 阈值从 API 来，不写死在页面里 —— 它来自配置，写死就会在配置改动后说错话。
  // 只按一下「空转一轮」通常什么都不会发生，不解释清楚会被当成按钮坏了。
  const idle = META.idle_ticks_before_proactive;
  $("#idleHint").textContent = idle
    ? ("冷场要连续 " + idle + " 轮，NPC 才会主动开口；如果它手上还有没做完的活，会先做完。"
       + "所以只按一下通常什么都看不到，多按几下才有效。")
    : "";
  $("#idleHint").className = "hint" + (idle ? " warn" : "");

  $("#scn").innerHTML = META.scenarios.map((s) =>
    "<option value='" + esc(s.id) + "'>" + esc(s.name) + " · " + esc(s.world_label) + "</option>").join("");

  const catOptions = '<option value="">全部（' + META.case_total + ' 条）</option>' +
    META.categories.map((c) => "<option value='" + esc(c.id) + "'>" + esc(c.id) + "（" + c.total + " 条）</option>").join("");
  $("#evcat").innerHTML = catOptions;
  $("#mtcat").innerHTML = catOptions;

  $("#mtid").innerHTML = META.mutants.map((m) =>
    "<option value='" + esc(m.id) + "'>" + esc(m.id) + " — " + esc(m.description) +
    "（" + esc(m.kind) + "，目标：" + esc(m.targets.join("/")) + "）</option>").join("");

  $("#cmds").textContent = META.commands.join("\n");
  renderReports(META.reports);

  onScenario();
  runEval();   // 后台跑 —— 页面先可用，结果回来再填
}

function currentPlayer(){
  const players = (CURRENT && CURRENT.players) || [];
  return players.find((p) => p.id === $("#who").value) || players[0] || {name: "?"};
}

function syncWho(){
  $("#whoAva").textContent = String(currentPlayer().name || "?")[0];
}

function onScenario(){
  CURRENT = META.scenarios.find((x) => x.id === $("#scn").value) || META.scenarios[0];
  HUES = {};
  (CURRENT.roster || []).forEach((p, i) => { HUES[p.name] = HUE_LIST[i % HUE_LIST.length]; });

  const groups = CURRENT.groups || [];
  const multi = groups.length > 1;

  $("#who").innerHTML = (CURRENT.players || []).map((p) =>
    "<option value='" + esc(p.id) + "'>" + esc(p.name) + "</option>").join("");
  // 说话范围：默认公开；多组场景多出「对某组说」；私聊对每个 NPC 一档。
  // 私聊的选项 id 带 `npc:` 前缀 —— 和组 id 共用同一个 select，前缀是唯一区分。
  let scope = "<option value=''>公开 · 所有人</option>";
  if (multi) {
    scope += groups.map((g) =>
      "<option value='group:" + esc(g.id) + "'>对组 · " + esc(g.name) + "</option>").join("");
  }
  scope += (CURRENT.roster || []).map((p) =>
    "<option value='npc:" + esc(p.id) + "'>私聊 · " + esc(p.name) + "</option>").join("");
  $("#scope").innerHTML = scope;

  $("#scnName").textContent = CURRENT.name;
  $("#worldLabel").textContent = CURRENT.world_label;
  $("#chatnote").textContent = (CURRENT.description || "") + (multi
    ? " 场上有 " + groups.length + " 个分组（" + groups.map((g) => g.name).join(" / ") +
      "）：同组轮流，不同组可以同时说话。"
    : "");
  $("#theaterHint").textContent = multi
    ? "每个分组每轮最多一个人开口，不同分组可以同时说话"
    : "每轮所有 NPC 依次行动，但最多一个人开口";
  syncWho();
  resetChat();
}

// select 里读回的「这一句说给谁」。返回 [组, NPC]，两个都空就是公开。
// 用数组而不是对象：页面的字段护栏会把对象属性名当成 API 字段查，
// 而 `npc` / `target` 只存在于**请求**里，响应里没有。
function scopeTarget(){
  if (!CURRENT) return [null, null];
  const v = $("#scope").value || "";
  return [
    v ? (CURRENT.groups || []).find((g) => v === "group:" + g.id) : null,
    v ? (CURRENT.roster || []).find((p) => v === "npc:" + p.id) : null,
  ];
}

// 输入框提示随范围变 —— 不说清楚，用户会以为私聊的话全场都听见了。
function syncScope(){
  if (!CURRENT) return;
  const [group, npc] = scopeTarget();
  $("#msg").placeholder = npc
    ? ("只对 " + npc.name + " 一个人说 —— 其他角色听不见")
    : (group
      ? ("只对「" + group.name + "」说 —— 另一组听不见")
      : (CURRENT.sample ? ("例如：" + CURRENT.sample) : "说点什么…"));
}

function renderReports(reports){
  const box = (id, kind) => {
    const items = reports.filter((r) => r.kind === kind);
    $(id).innerHTML = items.length ? items.map((r) =>
      "<a class='rep-card" + (r.known ? "" : " unknown") + "' href='" + esc(r.url) + "' target='_blank' rel='noopener'>" +
      "<span class='rep-ico'>" + (kind === "snapshot" ? "◷" : "◈") + "</span>" +
      "<span class='rep-main'><b>" + esc(r.title || r.file) + "</b><span>" + esc(r.answers || "（没有说明）") + "</span></span>" +
      "<span class='rep-cov'>" + esc(r.coverage) + "</span></a>").join("")
      : "<div class='muted'>（还没有这类报告）</div>";
  };
  box("#rep-offline", "offline");
  box("#rep-snap", "snapshot");
}

// ---------- chat ----------
function resetChat(){
  EVENTS = [];
  $("#msg").value = "";
  if ($("#scope")) $("#scope").value = "";
  $("#repnote").innerHTML = "";
  renderRoster({});
  renderGraph({});
  $("#peerlist").innerHTML = "";
  $("#log").innerHTML = "<div class='empty'><svg class='constellation' viewBox='0 0 120 72' aria-hidden='true'>" +
    "<circle class='halo' cx='60' cy='36' r='30'></circle>" +
    "<path d='M22 48 L58 20 L96 46'></path>" +
    "<circle cx='22' cy='48' r='2.6'></circle><circle cx='58' cy='20' r='3.4'></circle>" +
    "<circle cx='96' cy='46' r='2.6'></circle><circle cx='58' cy='56' r='1.5'></circle></svg>" +
    "<b>世界还没开始</b><span>说一句话，NPC 的决定、动作与世界状态会一起出现。<br>同一分组每轮最多一个人开口，不同分组可以同时说话。</span>" +
    "<code>" + esc((CURRENT && CURRENT.sample) || "今天这里挺热闹的。") + "</code></div>";
  $("#world").innerHTML = emptyBlock("还没有脉搏", "说一句话，目标与标记会在这里更新。");
  $("#mems").innerHTML = emptyBlock("记忆还是空的", "NPC 会记住你说过的话 —— 以及后来从谁那里听说了什么。");
  $("#chronicle").innerHTML = emptyBlock("还没发生过什么", "世界事件会在这里按时间排开。");
  $("#speech").innerHTML = emptyBlock("还没人开口", "发言配平是多 NPC 场景最该盯的数。");
  syncScope();
}

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
// 尤其是离线模式下的长对话：离线后端的模板池是有限的，
// 聊到 20 轮必然开始重复。不说清楚，读者会以为是修得不够好。
function renderRepetition(rep){
  const el = $("#repnote");
  if (!rep || !rep.total){
    el.innerHTML = "";
    return;
  }
  if (!rep.repeats){
    el.innerHTML = "<span class='pill ok'>复读 0/" + rep.total + "</span>" +
      "同一个 NPC 没有重复自己说过的话（判据：归一化后相似度 ≥ 0.80）。";
    return;
  }
  const ex = (rep.examples || []).map((e) =>
    "第 " + e.at + " 句撞第 " + e.collides_with + " 句（" + e.similarity.toFixed(2) + "）：" + esc(e.text)).join("<br>");
  const hint = META && !META.llm_available
    ? "<br>⚠ 当前离线模式：启发式后端只有有限套模板，聊得越长越容易重复。" +
      "配 <code>NPC_AGENT_PROVIDER</code> 接上模型后，prompt 里会带上「你最近说过」。"
    : "";
  el.innerHTML = "<span class='pill bad'>复读 " + rep.repeats + "/" + rep.total + "</span>（" +
    Math.round(rep.rate * 100) + "%）" +
    "<div class='rep-ex'>" + ex + hint + "</div>";
}

function renderRoster(d){
  if (!CURRENT) return;
  const mem = d.memories || {};
  const sp = d.speech || {};
  $("#roster").innerHTML = (CURRENT.roster || []).map((p) => {
    const m = mem[p.name] || {};
    const k = m.knowledge || {};
    const spoken = sp[p.name] || 0;
    const conf = Math.round((k.avg_confidence || 0) * 100);
    const fid = Math.round((k.avg_fidelity || 0) * 100);
    const total = (m.episodic || 0) + (m.semantic || 0) + (m.reflection || 0);
    const bonds = Object.entries(m.bonds || {}).map(([who, b]) => esc(who) + " " + b.score).join(" · ") || "尚未建立";
    return "<div class='npc' style='--h:" + hueOf(p.name) + "'>" + ava(p.name) +
      "<div class='npc-main'>" +
        "<div class='npc-id'><b>" + esc(p.name) +
          (p.group_name ? "<span class='group-badge'>" + esc(p.group_name) + "</span>" : "") +
          "</b><span>" + esc(p.role || "NPC") + "</span></div>" +
        "<div class='npc-stats'><span>发言 <b>" + spoken + "</b></span><span>记忆 <b>" + total + "</b></span><span>好感 <b>" + bonds + "</b></span></div>" +
        "<div class='npc-goal'>" + esc(p.goal || (p.traits && p.traits[0]) || "") + "</div>" +
        "<div class='npc-bar'><i style='width:" + conf + "%'></i></div>" +
        "<div class='npc-know'>确信 " + conf + "% · 保真 " + fid + "%" + (k.inversions ? " · 倒挂 " + k.inversions : "") + "</div>" +
      "</div></div>";
  }).join("");
}

function renderGraph(d){
  if (!CURRENT) return;
  const roster = CURRENT.roster || [];
  const players = CURRENT.players || [];
  if (!roster.length){
    $("#graph").innerHTML = emptyBlock("没有角色", "这个场景还没有配置 NPC。");
    return;
  }
  const mem = d.memories || {};
  const peers = d.peer_affinity || {};
  const NIDX = {};
  const PIDX = {};
  roster.forEach((p, i) => { NIDX[p.name] = i; });
  players.forEach((p, i) => { PIDX[p.name] = i; });

  let edges = "";
  let labels = "";
  roster.forEach((p, i) => {
    const np = NPC_POS[i] || [160, 96];
    const m = mem[p.name] || {};
    Object.entries(m.bonds || {}).forEach(([who, b]) => {
      const j = PIDX[who];
      const pp = PLR_POS[j];
      if (pp === undefined) return;
      const v = b.score || 0;
      const w = 1 + Math.min(3, Math.abs(v) / 30);
      const op = 0.22 + Math.min(0.58, Math.abs(v) / 150);
      edges += "<line class='edge player' x1='" + np[0] + "' y1='" + np[1] + "' x2='" + pp[0] + "' y2='" + pp[1] +
        "' stroke-width='" + w.toFixed(1) + "' opacity='" + op.toFixed(2) + "'></line>";
      labels += "<text class='edge-label' x='" + ((np[0] + pp[0]) / 2) + "' y='" +
        ((np[1] + pp[1]) / 2 - 3) + "'>" + esc(String(v)) + "</text>";
    });
  });
  roster.forEach((p, i) => {
    const np = NPC_POS[i] || [160, 96];
    Object.entries(peers[p.name] || {}).forEach(([who, v]) => {
      const j = NIDX[who];
      if (j === undefined || j <= i) return;
      const np2 = NPC_POS[j] || [160, 96];
      const w = 1 + Math.min(2.5, Math.abs(v) / 30);
      edges += "<line class='edge peer' x1='" + np[0] + "' y1='" + np[1] + "' x2='" + np2[0] + "' y2='" + np2[1] +
        "' stroke-width='" + w.toFixed(1) + "' opacity='0.6'></line>";
      labels += "<text class='edge-label' x='" + ((np[0] + np2[0]) / 2) + "' y='" +
        ((np[1] + np2[1]) / 2 - 4) + "'>" + esc(String(v)) + "</text>";
    });
  });

  const nodes = roster.map((p, i) => {
    const np = NPC_POS[i] || [160, 96];
    const h = hueOf(p.name);
    return "<g><circle cx='" + np[0] + "' cy='" + np[1] + "' r='22' fill='hsl(" + h + " 60% 60% / .14)'></circle>" +
      "<circle cx='" + np[0] + "' cy='" + np[1] + "' r='22' fill='none' stroke='hsl(" + h + " 60% 62% / .55)' stroke-width='1.2'></circle>" +
      "<text class='node-glyph' x='" + np[0] + "' y='" + (np[1] + 5) + "' fill='hsl(" + h + " 70% 66%)'>" + esc(String(p.name)[0]) + "</text>" +
      "<text class='node-name' x='" + np[0] + "' y='" + (np[1] + 37) + "'>" + esc(p.name) + "</text></g>";
  }).join("");

  const playerNodes = players.map((p, j) => {
    const pp = PLR_POS[j];
    if (!pp) return "";
    return "<g><rect x='" + (pp[0] - 27) + "' y='" + (pp[1] - 14) + "' width='54' height='28' rx='9'" +
      " fill='rgba(231,185,120,.10)' stroke='rgba(231,185,120,.38)'></rect>" +
      "<text class='player-name' x='" + pp[0] + "' y='" + (pp[1] + 4) + "'>" + esc(p.name) + "</text></g>";
  }).join("");

  $("#graph").innerHTML = "<svg viewBox='0 0 320 216' role='img' aria-label='关系星图'>" +
    "<ellipse class='stage-ring' cx='160' cy='108' rx='142' ry='96'></ellipse>" +
    edges + labels + nodes + playerNodes + "</svg>";

  const peerRows = Object.entries(peers).map(([name, row]) => {
    const inner = Object.entries(row).map(([who, v]) => esc(who) + " <b>" + v + "</b>").join(" · ");
    return inner ? "<div>" + esc(name) + " → " + inner + "</div>" : "";
  }).join("");
  $("#peerlist").innerHTML = peerRows || "<span class='muted'>NPC 之间的默契会在这里出现。</span>";
}

function renderWorld(d){
  const w = d.snapshot || {};
  const obj = Object.entries(w.objectives || {});
  const flags = w.world_flags || [];
  const PLANNER = d.planner || {};
  const srcEntries = Object.entries(PLANNER.sources || {});
  let srcTotal = 0;
  srcEntries.forEach(([k, v]) => { srcTotal += v; });
  const seg = srcEntries.map(([k, v]) => {
    const width = srcTotal ? (v / srcTotal * 100) : 0;
    const color = k === "model" ? "var(--violet)" : (k === "heuristic" ? "var(--teal)" : "var(--amber)");
    return "<i style='width:" + width.toFixed(1) + "%;background:" + color + "' title='" +
      esc(SRC_LABEL[k] || k) + " ×" + v + "'></i>";
  }).join("");

  $("#world").innerHTML =
    "<div class='stat-row'>" +
      "<div class='stat'><span>同组抢话</span><b class='" + (d.collisions ? "down" : "up") + "'>" + d.collisions + "</b></div>" +
      "<div class='stat'><span>静默回落</span><b class='" + (PLANNER.silent_fallbacks ? "down" : "up") + "'>" +
        (PLANNER.silent_fallbacks || 0) + "</b></div>" +
    "</div>" +
    "<div class='stat-row'>" +
      "<div class='stat'><span>世界</span><b class='small'>" + esc(d.world_label || "—") + "</b></div>" +
      "<div class='stat'><span>规划构成</span><b class='small'>" +
        (srcEntries.length ? srcEntries.map(([k, v]) => esc(SRC_LABEL[k] || k) + " ×" + v).join(" · ") : "—") + "</b></div>" +
    "</div>" +
    "<div class='subhead'><span>目标</span><span>" + obj.length + "</span></div>" +
    (obj.length
      ? obj.map(([k, v]) => "<div class='obj'><span class='obj-id'>" + esc(k) + "</span><span class='obj-val'>" + esc(v) + "</span></div>").join("")
      : "<div class='muted'>还没有目标状态</div>") +
    "<div class='subhead'><span>世界标记</span><span>" + flags.length + "</span></div>" +
    (flags.length
      ? "<div class='pills'>" + flags.map((f) => "<span class='pill ok'>" + esc(f) + "</span>").join("") + "</div>"
      : "<div class='muted'>还没有标记</div>") +
    "<div class='subhead'><span>规划来源</span><span>" + (PLANNER.llm_active ? "模型这一路活着" : "离线启发式") + "</span></div>" +
    "<div class='planner-bar'>" + (seg || "<i style='width:100%;background:var(--line)'></i>") + "</div>" +
    (srcTotal
      ? "<div class='planner-row'>" + srcEntries.map(([k, v]) =>
          "<span>" + esc(SRC_LABEL[k] || k) + " " + Math.round(v / srcTotal * 100) + "%</span>").join("") + "</div>"
      : "");
}

function renderMemories(d){
  const entries = Object.entries(d.memories || {});
  if (!entries.length){
    $("#mems").innerHTML = emptyBlock("记忆还是空的", "NPC 会记住你说过的话 —— 以及后来从谁那里听说了什么。");
    return;
  }
  $("#mems").innerHTML = entries.map(([name, m]) => {
    const k = m.knowledge || {};
    const fid = Math.round((k.avg_fidelity || 0) * 100);
    const conf = Math.round((k.avg_confidence || 0) * 100);
    const bonds = Object.entries(m.bonds || {}).map(([who, b]) =>
      "<span class='pill'>" + esc(who) + " · " + esc(b.stage) + " " + b.score +
      (b.titles && b.titles.length ? " · " + esc(b.titles[b.titles.length - 1]) : "") + "</span>").join("");
    const debts = (m.debts || []).length
      ? "<div class='mem-line'>欠着的回答：" + m.debts.map(esc).join(" / ") + "</div>" : "";
    const rows = (m.recent_meta || []).map((r) => {
      let tags = "<span class='pill'>" + esc(r.state) + "</span>";
      if (r.origin === "told") {
        tags += "<span class='pill'>听 " + esc(r.source || "别人") + " 说 · " + r.hop + " 跳</span>" +
          "<span class='pill'>保真 " + Math.round(r.fidelity * 100) + "%</span>" +
          "<span class='pill'>确信 " + Math.round(r.confidence * 100) + "%</span>";
      }
      if (r.inversion) tags += "<span class='pill warn'>越传越确信</span>";
      if (r.forgotten) tags += "<span class='pill bad'>已遗忘</span>";
      if (r.privacy === "secret") tags += "<span class='pill bad'>私密</span>";
      if (r.links) tags += "<span class='pill'>关联 " + r.links + "</span>";
      return "<div class='mem-rec'><div class='pills'>" + tags + "</div>" +
        "<div class='content'>[" + esc(r.kind) + "] " +
        (r.forgotten ? "<span class='gone'>" + esc(r.content) + "</span>" : esc(r.content)) + "</div></div>";
    }).join("");
    return "<details class='mem-npc' open><summary>" + ava(name) + "<b>" + esc(name) + "</b>" +
      "<span class='mem-tally'>E" + (m.episodic || 0) + " · S" + (m.semantic || 0) + " · R" + (m.reflection || 0) + "</span></summary>" +
      "<div class='mem-body'>" +
        "<div class='meters'>" +
          "<div class='meter fid'><span>平均保真</span><div class='track'><i style='width:" + fid + "%'></i></div><b>" + fid + "%</b></div>" +
          "<div class='meter conf'><span>平均确信</span><div class='track'><i style='width:" + conf + "%'></i></div><b>" + conf + "%</b></div>" +
        "</div>" +
        "<div class='mem-line'>启用 " + (k.active || 0) + " · 转述 " + (k.told || 0) +
          " · 遗忘 " + (k.forgotten || 0) + " · 倒挂 " + (k.inversions || 0) + "</div>" +
        (bonds ? "<div class='pills'>" + bonds + "</div>" : "") + debts +
        (rows || "<div class='muted'>还没有记忆</div>") +
      "</div></details>";
  }).join("");
}

function renderChronicle(d){
  const chron = d.chronicle || [];
  $("#chronicle").innerHTML = chron.length
    ? "<ol class='timeline'>" + chron.map((e) => "<li>" + esc(e) + "</li>").join("") + "</ol>"
    : emptyBlock("还没发生过什么", "世界事件会在这里按时间排开。");
}

function renderSpeech(d){
  const entries = Object.entries(d.speech || {});
  if (!entries.length){
    $("#speech").innerHTML = emptyBlock("还没人开口", "发言配平是多 NPC 场景最该盯的数。");
    return;
  }
  let sum = 0;
  let maxN = 1;
  entries.forEach(([name, v]) => { sum += v; if (v > maxN) maxN = v; });
  $("#speech").innerHTML = entries.map(([name, n]) =>
    "<div class='speech-row'><span>" + esc(name) + "</span>" +
    "<div class='speech-track' style='--h:" + hueOf(name) + "'><i style='width:" + (n / maxN * 100).toFixed(1) + "%'></i></div>" +
    "<em>" + n + "</em></div>").join("") +
    "<div class='mem-line'>共 " + sum + " 次发言" +
      (d.collisions ? " · 同组抢话 " + d.collisions + " 轮" : " · 没有抢话") + "</div>";
}

function paint(d){
  if (!d.events) return;
  // 规划来源。规划调用失败时框架会静默回落到启发式规划器，
  // 两条路径产出的轨迹完全一样 —— 不把来源画出来，
  // "接上模型规划到底有没有生效"在页面上根本看不出来。
  const PLANNER = d.planner || {};
  // "回落"只有在模型真的可用时才成立：没配模型是"没开这一路"，不是失败。
  // 判据由后端算好（llm_active），页面照读，不自己再与一遍。
  const llmOn = !!PLANNER.llm_active;

  $("#log").innerHTML = d.events.map((ev, i) => {
    // 说话范围标签由后端发的事件字段决定，页面不猜。收件人写错时后端
    // 让这一轮没人接话（不广播），页面在这里把原因原样印出来。
    const scopeCls = ev.scope === "private" ? "private" : (ev.scope === "group" ? "group" : "");
    const scopeBadge = ev.scope_error
      ? "<span class='scope-tag bad'>" + esc(ev.scope_error) + "</span>"
      : (ev.scope === "private"
        ? "<span class='scope-tag private'>私聊 → " + esc(ev.target_name || "?") + "</span>"
        : (ev.scope === "group"
          ? "<span class='scope-tag group'>对组 · " + esc(ev.group_name || "?") + "</span>"
          : ""));
    const head = ev.kind === "say"
      ? "<span class='turn-no'>第 " + (i + 1) + " 轮</span>" + scopeBadge +
        "<span class='bubble-player " + scopeCls + "'><i>玩家 · " + esc(ev.speaker_name) + "</i>" + esc(ev.text) + "</span>"
      : "<span class='turn-no'>第 " + (i + 1) + " 轮</span><span class='idle-tag'>无人说话 · 世界自转</span>";
    const turns = (ev.turns || []).map((t) => {
      const acts = (t.actions || []).map((a) => {
        // 颜色与符号都从后端发来的 `mark` 取 —— 页面不自己判 `ok`。
        // yield（按策略主动让出话头）是正确行为，不许印成红色的 ✗。
        const cls = a.mark === "ok" ? "ok" : (a.mark === "yield" ? "yield" : "no");
        const tail = a.mark === "ok" ? "" : " " + (a.mark === "yield" ? "↷ " : "✗ ") + esc(a.detail);
        return "<div class='act " + cls + "'>▸ " + esc(a.render) + tail + "</div>";
      }).join("");
      const mems = (t.used_memories || []).length
        ? "<div class='pills'><span class='muted'>用到的记忆</span>" +
          t.used_memories.map((m) => "<span class='pill'>" + esc(m) + "</span>").join("") + "</div>"
        : "";
      const viol = (t.violations || []).length
        ? "<div class='pills'><span class='pill bad'>人设违规 · " + esc(t.violations.join("；")) + "</span></div>" : "";
      // 计划来源标签。heuristic 出现在这里而模型开着 ⇒ 静默回落。
      const src = t.plan_source || "";
      const fellBack = src === "heuristic" && llmOn;
      const srcBadge = src
        ? "<span class='pill" + (fellBack ? " bad" : "") + "'>" + esc(SRC_LABEL[src] || src) +
          (fellBack ? " · 回落" : "") + "</span>"
        : "";
      // 三种结局都要有可见的一行，否则"没轮到它"的 NPC 会整条消失。
      // 判据用可见动作数，不用 t.acted：成功的 speak 会被后端从 actions 里滤掉，
      // 于是"只说了一句话"的回合 acted=true 而 actions 为空。
      const visible = (t.actions || []).length;
      const outcome = t.say
        ? "<div class='say'>" + esc(t.say) + "</div>"
        : (visible ? "<div class='decision'>" + esc(t.name) + " 没说话，但在忙自己的事</div>" : "");
      // 频道标签来自**真正落进世界的那条发言**（后端按 speaker 配好）。
      // 私聊回复也是私聊 —— 玩家悄悄问、NPC 大声答曾经是一个真实的泄漏。
      const chBadge = t.channel_label
        ? "<span class='scope-tag " + (t.channel === "private" ? "private" : "group") + "'>" +
          esc(t.channel_label) + "</span>"
        : "";
      return "<div class='turn'>" + ava(t.name) + "<div class='turn-main'>" +
        "<div class='turn-name'><b>" + esc(t.name) + "</b>" + chBadge + srcBadge + "</div>" +
        (t.decision_reason ? "<div class='decision'>" + esc(t.decision_reason) + "</div>" : "") +
        acts + mems + viol + outcome +
        "</div></div>";
    }).join("");
    return "<div class='beat" + (ev.kind === "say" ? "" : " idle") + "'>" +
      "<div class='beat-head'>" + head + "</div>" + turns + "</div>";
  }).join("");
  $("#log").scrollTop = $("#log").scrollHeight;

  renderRepetition(d.repetition);
  renderRoster(d);
  renderGraph(d);
  renderWorld(d);
  renderMemories(d);
  renderChronicle(d);
  renderSpeech(d);
}

$("#send").onclick = () => {
  const text = $("#msg").value.trim();
  if (!text) return;
  const [group, npc] = scopeTarget();
  // 范围字段直接写进对象字面量：`ev.target = ...` 里的 `.target` 会被
  // 页面字段护栏当成"读了响应里的 target"，而它只存在于请求里。
  step({
    kind: "say", speaker: $("#who").value, text,
    group: group ? group.id : "", target: npc ? npc.id : "",
  });
  $("#msg").value = "";
};
$("#msg").onkeydown = (e) => { if (e.key === "Enter") $("#send").click(); };
$("#idle").onclick = () => step({kind: "idle"});
$("#seed").onclick = () => { if (CURRENT && CURRENT.sample) $("#msg").value = CURRENT.sample; };
$("#reset").onclick = resetChat;
$("#scn").onchange = onScenario;
$("#who").onchange = syncWho;
$("#scope").onchange = syncScope;

// ---------- eval ----------
function bar(v){
  const pct = Math.max(0, Math.min(1, v)) * 100;
  return "<div class='bar'><i style='width:" + pct.toFixed(1) + "%'></i></div>";
}

async function runEval(){
  $("#evrun").disabled = true;
  $("#evout").innerHTML = "<div class='empty'><b>正在跑…</b><span>离线，不调模型。</span></div>";
  try {
    const d = await api("/api/eval", {category: $("#evcat").value, limit: Number($("#evlim").value) || 0});
    const s = d.summary;
    const pct = Math.round(s.pass_rate * 1000) / 10;
    $("#evnote").textContent = "通过 " + s.passed + "/" + s.total + " · " + pct + "% · " + (d.elapsed_ms / 1000).toFixed(1) + " 秒";
    const means = Object.entries(s.metric_means).map(([k, v]) =>
      "<div class='metric-row'><label>" + esc(k) + "</label>" + bar(v) + "<b>" + v.toFixed(3) + "</b></div>").join("");
    const cats = Object.entries(s.by_category).map(([k, v]) =>
      "<tr><td>" + esc(k) + "</td><td class='num'>" + v.passed + "/" + v.total + "</td><td class='num'>" +
      (v.pass_rate * 100).toFixed(0) + "%</td></tr>").join("");
    const fails = d.cases.filter((c) => !c.passed);
    $("#evout").innerHTML =
      "<div class='eval-hero'>" +
        "<div class='ring' style='--p:" + pct + "%'><div class='ring-in'><b>" + pct + "%</b><span>通过率</span></div></div>" +
        "<div class='eval-stats'>" +
          "<div class='eval-stat'><span>通过</span><b>" + s.passed + " / " + s.total + "</b></div>" +
          "<div class='eval-stat'><span>耗时</span><b>" + (d.elapsed_ms / 1000).toFixed(1) + " s</b></div>" +
          "<div class='eval-stat'><span>类别</span><b>" + Object.entries(s.by_category).length + "</b></div>" +
        "</div>" +
      "</div>" +
      "<div class='subhead'><span>维度均值</span><span>1.000 = 全对</span></div>" + means +
      "<div class='subhead'><span>分类</span><span></span></div>" +
      "<div class='table-wrap'><table><thead><tr><th>类别</th><th class='num'>通过</th><th class='num'>通过率</th></tr></thead><tbody>" +
        cats + "</tbody></table></div>" +
      (fails.length
        ? "<details class='details' open><summary>没过的用例 " + fails.length + " 条</summary><div class='table-wrap'><table><thead><tr><th>用例</th><th>类别</th><th>失败原因</th></tr></thead><tbody>" +
          fails.map((c) => "<tr><td class='mono'>" + esc(c.id) + "</td><td>" + esc(c.category) + "</td><td>" + esc(c.reason) + "</td></tr>").join("") +
          "</tbody></table></div></details>"
        : "<p class='muted' style='margin-top:12px'>这一批全过。</p>");
  } catch (e) {
    $("#evout").innerHTML = "<div class='err'>" + esc(e.message) + "</div>";
  } finally { $("#evrun").disabled = false; }
}
$("#evrun").onclick = runEval;

// ---------- mutant ----------
async function runMutant(){
  $("#mtrun").disabled = true;
  $("#mtout").innerHTML = "<div class='empty'><b>正在注入缺陷并重跑…</b><span>要跑两遍。</span></div>";
  try {
    const d = await api("/api/mutant", {mutant: $("#mtid").value, category: $("#mtcat").value, limit: Number($("#mtlim").value) || 0});
    $("#mtnote").textContent = d.partial
      ? ("只跑了 " + d.total + " 条 —— 子集，不作结论")
      : (d.caught ? "抓到了" : "没抓到 —— 这是一个评测盲区");
    const status = d.partial ? "warn" : (d.caught ? (d.weak ? "warn" : "ok") : "bad");
    const icon = d.partial ? "◷" : (d.caught ? (d.weak ? "◐" : "✦") : "✕");
    const title = d.partial ? "子集不作结论"
      : (!d.caught ? "没抓到 —— 这一维看不见这类错误"
      : (d.weak ? "抓到了，但目标维度几乎没动" : "评测对这一类缺陷是敏感的"));
    let text;
    if (d.partial) {
      text = "这次只跑了 " + d.total + " 条（子集）。子集上的「抓没抓到」不作数 —— 把上限调成 0 跑全量才有结论。";
    } else if (!d.caught) {
      text = "注入缺陷后评测没掉分 —— 说明这一维看不见这类错误。这是评测盲区，不是被测系统没问题。";
    } else if (d.weak) {
      text = "评测对这种缺陷是敏感的，但目标维度只动了 " +
        Math.abs(d.deltas[d.targets[0]]).toFixed(3) + " —— 这一维的覆盖面很薄。";
    } else {
      text = "评测对这一类缺陷是敏感的 —— 分数掉了 " +
        Math.abs(d.deltas[d.targets[0]]).toFixed(3) + "，而且掉在声明的目标维度上。";
    }
    const rows = Object.entries(d.deltas).map(([k, v]) => {
      const cls = v < -1e-9 ? "down" : "up";
      const target = d.targets.includes(k);
      return "<tr><td>" + esc(k) + (target ? " <span class='pill warn'>目标</span>" : "") + "</td>" +
        "<td class='num'>" + d.baseline_means[k].toFixed(3) + "</td>" +
        "<td class='num'>" + d.mutant_means[k].toFixed(3) + "</td>" +
        "<td class='num delta " + cls + "'>" + (v > 0 ? "+" : "") + v.toFixed(3) + "</td></tr>";
    }).join("");
    $("#mtout").innerHTML =
      "<div class='verdict " + status + "'><span class='v-ico'>" + icon + "</span>" +
        "<div><b>" + esc(title) + "</b><p>" + esc(text) + "</p></div></div>" +
      "<div class='eval-stats'>" +
        "<div class='eval-stat'><span>缺陷</span><b class='small'>" + esc(d.id) + "</b></div>" +
        "<div class='eval-stat'><span>类型</span><b class='small'>" + esc(d.kind) + "</b></div>" +
        "<div class='eval-stat'><span>用例</span><b>" + d.total + "</b></div>" +
        "<div class='eval-stat'><span>通过率</span><b>" + (d.baseline_pass * 100).toFixed(0) + "% → " +
          (d.mutant_pass * 100).toFixed(0) + "%</b></div>" +
      "</div>" +
      "<div class='defect'>" + d.targets.map((t) => "<span class='pill warn'>" + esc(t) + "</span>").join("") +
        "<span class='muted'>" + esc(d.description) + "</span></div>" +
      "<div class='table-wrap'><table><thead><tr><th>维度</th><th class='num'>基线</th><th class='num'>注入后</th><th class='num'>Δ</th></tr></thead><tbody>" +
        rows + "</tbody></table></div>";
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
