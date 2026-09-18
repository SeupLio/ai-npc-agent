#!/usr/bin/env node
/**
 * Mineflayer 桥 —— 把我们的世界操作契约翻译成真实的 Minecraft 动作。
 *
 * ## 协议
 *
 * 一行一个 JSON 请求，一行一个 JSON 响应。**没有别的约定**。
 *
 *     → {"op": "move", "actor": "ayan", "target": "forest", "id": 1}
 *     ← {"ok": true, "reason": "", "data": {...}, "id": 1}
 *
 * `id` 会被原样回显：上层靠它确认响应对应的是哪个请求。
 * 串包意味着协议错位，上层会直接抛错而不是把锅甩给 NPC。
 *
 * ## 两种模式
 *
 *   --dry-run   不连游戏。协议照常工作，但改变世界的操作会返回
 *               "dry-run" 说明。用来在没有 Minecraft 服务端的机器上
 *               验证协议本身（tests/test_minecraft_env.py 就是这么用的）。
 *
 *   默认        连真实服务端。需要 npm i mineflayer mineflayer-pathfinder。
 *
 * ## 为什么失败也是数据
 *
 * 每个 op 都返回 `{ok, reason}`，即使出了问题也不崩。原因有两条：
 *   1. "挖不到矿"是**游戏内的事实**，要交给 NPC 的 Reflection 去学；
 *      桥崩了才是**工程故障**。两者混在一起，一次事故看起来会像一次模型失败。
 *   2. 一个 20 分钟的评测跑到第 30 条用例时桥挂掉，代价是全部重跑。
 */

'use strict';

const readline = require('readline');

// --------------------------------------------------------------------------- //
// 参数
// --------------------------------------------------------------------------- //
function parseArgs(argv) {
  const opts = { dryRun: false, host: 'localhost', port: 25565, username: 'npc_agent', version: null };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--dry-run') opts.dryRun = true;
    else if (arg === '--host') opts.host = argv[++i];
    else if (arg === '--port') opts.port = parseInt(argv[++i], 10);
    else if (arg === '--username') opts.username = argv[++i];
    else if (arg === '--version') opts.version = argv[++i];
  }
  return opts;
}

const opts = parseArgs(process.argv.slice(2));

// --------------------------------------------------------------------------- //
// 世界状态镜像
// --------------------------------------------------------------------------- //
// 桥需要知道"场景里有哪些人、哪些地点"才能把我们的 id 翻译成游戏内的
// 实体与坐标。这份信息由上层通过 configure 送进来。
const world = {
  actors: new Map(),   // id -> {id, name, kind, pos, inventory, poi}
  pois: new Map(),     // id -> {name, pos}
  resources: new Map(),// poi -> {block: count}
  placed: [],          // [{pos, block}]
  flags: new Set(),
  utterances: [],
  tick: 0,
};

let bot = null;
let botReady = false;
let botError = null;

const DAY_TICKS = 12;
const NIGHT_START = 8;

function timeOfDay() {
  return world.tick % DAY_TICKS >= NIGHT_START ? 'night' : 'day';
}

function poiOf(pos) {
  if (!pos) return null;
  for (const [id, entry] of world.pois) {
    const p = entry.pos || [];
    if (p[0] === pos[0] && p[1] === pos[1] && p[2] === pos[2]) return id;
  }
  return null;
}

function fail(reason) {
  return { ok: false, reason, data: {} };
}

function ok(data) {
  return { ok: true, reason: '', data: data || {} };
}

// --------------------------------------------------------------------------- //
// 连接真实服务端（只有非 dry-run 才会走）
// --------------------------------------------------------------------------- //
function connect() {
  let mineflayer;
  try {
    mineflayer = require('mineflayer');
  } catch (err) {
    botError = '没有装 mineflayer。在项目根目录跑 `npm i mineflayer mineflayer-pathfinder`，' +
               '或者用 --dry-run 只验证协议。';
    return;
  }
  try {
    bot = mineflayer.createBot({
      host: opts.host,
      port: opts.port,
      username: opts.username,
      version: opts.version || undefined,
    });
    bot.once('spawn', () => { botReady = true; });
    bot.on('error', (err) => { botError = String(err && err.message ? err.message : err); });
    bot.on('end', () => { botReady = false; });
  } catch (err) {
    botError = `建 bot 失败：${err.message}`;
  }
}

function requireBot() {
  if (opts.dryRun) return null;
  if (botError) return fail(botError);
  if (!botReady) return fail('bot 还没连上服务端（或者已经断线了）');
  return null;
}

// --------------------------------------------------------------------------- //
// 各个 op
// --------------------------------------------------------------------------- //
const ops = {
  // 声明场景：有哪些人、哪些地点、哪些资源点。
  // 真实服务端里世界早就存在，这一步只是把游戏内实体映射到我们的 id 上。
  configure(params) {
    const scenario = params.scenario || {};
    world.actors.clear();
    world.pois.clear();
    world.resources.clear();
    for (const spec of scenario.actors || []) {
      world.actors.set(spec.id, {
        id: spec.id,
        name: spec.name || spec.id,
        kind: spec.kind || 'npc',
        pos: null,
        poi: spec.start || null,
        inventory: Object.assign({}, spec.inventory || {}),
      });
    }
    for (const [id, entry] of Object.entries(scenario.pois || {})) {
      world.pois.set(id, { name: entry.name || id, pos: entry.pos || [0, 0, 0] });
    }
    for (const [poi, table] of Object.entries(scenario.resources || {})) {
      world.resources.set(poi, Object.assign({}, table));
    }
    return ok({ actors: world.actors.size, pois: world.pois.size });
  },

  reset() {
    world.flags.clear();
    world.placed = [];
    world.utterances = [];
    world.tick = 0;
    for (const actor of world.actors.values()) {
      actor.pos = null;
    }
    if (bot) {
      try { bot.chat('/clear'); } catch (err) { /* 单人测试服可能没权限，不算失败 */ }
    }
    return ok();
  },

  state() {
    const actors = {};
    for (const [id, actor] of world.actors) {
      actors[id] = {
        id: actor.id,
        name: actor.name,
        kind: actor.kind,
        pos: actor.pos || [],
        poi: actor.poi,
        inventory: Object.assign({}, actor.inventory),
        affinity: {},
      };
    }
    const pois = {};
    for (const [id, entry] of world.pois) pois[id] = { name: entry.name, pos: entry.pos };
    const resources = {};
    for (const [poi, table] of world.resources) resources[poi] = Object.assign({}, table);
    // ⚠️ `actors[*].pos` 是**镜像**（我们记的账），不是真实 bot 的位置。
    // 两者会不一致：`move` 里镜像在动作成功后才更新，但 dry-run、
    // 或者别的地方改了世界，都可能让账和现实对不上。
    // 所以这里额外报一份**真实遥测** `bot_pos`，让测试有东西可以断言在真状态上。
    let botPos = null;
    if (bot && bot.entity && bot.entity.position) {
      const p = bot.entity.position;
      botPos = [Math.floor(p.x), Math.floor(p.y), Math.floor(p.z)];
    }
    return ok({
      tick: world.tick,
      day: Math.floor(world.tick / DAY_TICKS),
      time_of_day: timeOfDay(),
      pois,
      actors,
      resources,
      placed: world.placed.map((e) => ({ pos: e.pos, block: e.block })),
      flags: Array.from(world.flags).sort(),
      utterances: world.utterances,
      dry_run: opts.dryRun,
      bot_connected: botReady,
      bot_pos: botPos,
    });
  },

  async move(params) {
    const blocked = requireBot();
    if (blocked) return blocked;
    const actor = world.actors.get(params.actor);
    if (!actor) return fail(`世界上没有 ${params.actor} 这个人`);
    const poi = world.pois.get(params.target);
    if (!poi) {
      const known = Array.from(world.pois.keys()).join('、') || '（没有地点）';
      return fail(`没有叫「${params.target}」的地方。能去的地方：${known}`);
    }
    if (bot) {
      // 真实路径：用 pathfinder 走过去。走不到就如实报告，不要假装成功 ——
      // NPC 以为自己到了、其实没到，后面每一步都会错。
      try {
        const { pathfinder, Movements, goals } = require('mineflayer-pathfinder');
        if (!bot.pathfinder) bot.loadPlugin(pathfinder);
        const movements = new Movements(bot);
        bot.pathfinder.setMovements(movements);
        const [x, y, z] = poi.pos;
        // 用 GoalNear 而不是 GoalBlock：GoalBlock 要求"恰好站在这个方块上"，
        // 于是**世界只要被改过一次就永远走不到** —— 比如 mine 挖掉了 POI 脚下
        // 那块方块，站立面矮了一格，GoalBlock(12,-60,6) 就成了一个悬空坐标，
        // pathfinder 只能超时（报 "Took to long to decide path to goal!"）。
        // 而"走到某个地方"本来就是"靠近它"，不是"像素级对齐"。
        await bot.pathfinder.goto(new goals.GoalNear(x, y, z, 2));
      } catch (err) {
        return fail(`走不到${poi.name}：${err.message}`);
      }
    }
    // ⚠️ 镜像**必须在动作成功之后**才更新。
    // 原来这两行写在 try 前面，于是走不到目的地时：
    //   - 返回值是 ok=false（对的）
    //   - 但镜像里 actor.poi / actor.pos 已经变成目的地了（错的）
    // 结果 state() 报告"它在林子"，实际它还在营地。
    // 这条注释原本就写着"不要假装成功"，而代码正好在假装 —— 只是假装在镜像里。
    actor.poi = params.target;
    actor.pos = poi.pos;
    return ok({ poi: params.target, pos: poi.pos });
  },

  async mine(params) {
    const blocked = requireBot();
    if (blocked) return blocked;
    const actor = world.actors.get(params.actor);
    if (!actor) return fail(`世界上没有 ${params.actor} 这个人`);
    const here = actor.poi;
    if (!here) return fail('这里没有可以采集的东西，先 move_to 到一个采集点');
    const table = world.resources.get(here) || {};
    if ((table[params.block] || 0) <= 0) {
      return fail(`${world.pois.get(here).name}这里已经被采空了`);
    }
    // 光照护栏与离线世界保持一致：夜里没有光源就采不了。
    // 两个世界必须给**同一句话**，否则模型在不同后端上学到的东西不一样。
    if (timeOfDay() === 'night' && !world.placed.some((e) => e.block === 'torch')) {
      return fail('天黑了，看不清矿脉。需要先做个火把（torch）放在附近，或者等天亮');
    }
    if (opts.dryRun) return fail('dry-run 模式不改变世界（mine 未执行）');

    try {
      const block = bot.blockAt(bot.entity.position.offset(0, -1, 0));
      if (!block) return fail('脚下没有可以挖的方块');
      await bot.dig(block);
    } catch (err) {
      return fail(`挖不动：${err.message}`);
    }
    table[params.block] -= 1;
    actor.inventory[params.block] = (actor.inventory[params.block] || 0) + 1;
    return ok({ block: params.block, count: actor.inventory[params.block], poi: here });
  },

  async craft(params) {
    const blocked = requireBot();
    if (blocked) return blocked;
    const actor = world.actors.get(params.actor);
    if (!actor) return fail(`世界上没有 ${params.actor} 这个人`);
    if (opts.dryRun) return fail('dry-run 模式不改变世界（craft 未执行）');
    // 真实路径：用 bot.craft。配方由游戏自己决定，我们不重复维护一张表 ——
    // 那正是"桥只是传输层"的意思。
    try {
      const item = bot.registry.itemsByName[params.item];
      if (!item) return fail(`游戏里没有「${params.item}」这个物品`);
      const recipe = bot.recipesFor(item.id, null, 1, null)[0];
      if (!recipe) return fail(`材料不够，做不出${params.item}`);
      await bot.craft(recipe, 1, null);
    } catch (err) {
      return fail(`合成失败：${err.message}`);
    }
    return ok({ item: params.item });
  },

  async place(params) {
    const blocked = requireBot();
    if (blocked) return blocked;
    const actor = world.actors.get(params.actor);
    if (!actor) return fail(`世界上没有 ${params.actor} 这个人`);
    if ((actor.inventory[params.block] || 0) <= 0) {
      return fail(`背包里没有${params.block}`);
    }
    const target = world.pois.get(params.target);
    if (!target) return fail(`没有叫「${params.target}」的地方`);
    if (actor.poi !== params.target) {
      return fail(`你不在${target.name}，放不了东西。需要先 move_to(${params.target})`);
    }
    if (world.placed.some((e) => e.pos.join(',') === target.pos.join(','))) {
      return fail(`${target.name}这里已经有一个方块了`);
    }
    if (opts.dryRun) return fail('dry-run 模式不改变世界（place 未执行）');

    try {
      const item = bot.inventory.items().find((it) => it.name === params.block);
      if (!item) return fail(`背包里找不到${params.block}`);
      await bot.equip(item, 'hand');
      const [x, y, z] = target.pos;
      const ref = bot.blockAt(new (require('vec3'))(x, y - 1, z));
      if (!ref) return fail('目标位置下方没有可以依附的方块');
      await bot.placeBlock(ref, new (require('vec3'))(0, 1, 0));
    } catch (err) {
      return fail(`放不下去：${err.message}`);
    }
    world.placed.push({ pos: target.pos, block: params.block });
    actor.inventory[params.block] -= 1;
    return ok({ block: params.block, at: params.target });
  },

  transfer(params) {
    const giver = world.actors.get(params.src);
    const taker = world.actors.get(params.dst);
    if (!giver || !taker) return fail('转交双方必须都在世界上');
    const count = params.count || 1;
    if ((giver.inventory[params.item] || 0) < count) {
      return fail(`${giver.name}手里没有足够的${params.item}`);
    }
    if (giver.poi !== taker.poi) {
      return fail(`${taker.name}不在附近，得先 move_to 过去才能把东西交给他`);
    }
    if (opts.dryRun) return fail('dry-run 模式不改变世界（transfer 未执行）');
    giver.inventory[params.item] -= count;
    taker.inventory[params.item] = (taker.inventory[params.item] || 0) + count;
    if (bot) {
      try { bot.chat(`/give ${taker.name} ${params.item} ${count}`); } catch (err) { /* 忽略 */ }
    }
    return ok({ item: params.item, to: params.dst, count });
  },

  consume(params) {
    const actor = world.actors.get(params.actor);
    if (!actor) return fail(`世界上没有 ${params.actor} 这个人`);
    const count = params.count || 1;
    if ((actor.inventory[params.item] || 0) < count) {
      return fail(`${params.item}不够，需要 ${count} 个`);
    }
    actor.inventory[params.item] -= count;
    if (actor.inventory[params.item] <= 0) delete actor.inventory[params.item];
    return ok({ item: params.item, count: actor.inventory[params.item] || 0 });
  },

  chat(params) {
    const actor = world.actors.get(params.actor);
    if (!actor) return fail(`世界上没有 ${params.actor} 这个人`);
    world.utterances.push({
      tick: world.tick,
      speaker_id: params.actor,
      speaker_name: actor.name,
      text: params.text,
      poi: actor.poi,
    });
    if (bot) {
      try { bot.chat(params.text); } catch (err) { /* 断线时不当成失败 */ }
    }
    return ok();
  },

  set_flag(params) {
    world.flags.add(params.flag);
    return ok({ flags: Array.from(world.flags).sort() });
  },

  advance_tick(params) {
    world.tick += Math.max(1, params.n || 1);
    return ok({ tick: world.tick, time_of_day: timeOfDay() });
  },
};

// --------------------------------------------------------------------------- //
// 主循环
// --------------------------------------------------------------------------- //
if (!opts.dryRun) connect();

const rl = readline.createInterface({ input: process.stdin, terminal: false });

// 操作必须**串行**执行。
//
// 上层是同步一问一答的，所以正常情况下不会有并发。但如果有人把多行请求
// 一次性灌进来（手测、或者上层加了流水线），readline 会把它们全交出来，
// 而 move / mine 都是 async 的 —— 两个操作同时作用在一个 bot 上，
// 结果是没有定义的。这里用一个 promise 队列把执行顺序固定下来，
// 代价是几行代码，换来的是"响应顺序 = 请求顺序"这个不变量。
let queue = Promise.resolve();

rl.on('line', (line) => {
  queue = queue.then(() => handleLine(line));
});

async function handleLine(line) {
  const trimmed = line.trim();
  if (!trimmed) return;

  let request;
  try {
    request = JSON.parse(trimmed);
  } catch (err) {
    // 坏行不能弄死桥：报告一次，继续听下一行。
    process.stdout.write(JSON.stringify({
      ok: false,
      reason: `不是合法的 JSON：${trimmed.slice(0, 120)}`,
      data: {},
      id: null,
    }) + '\n');
    return;
  }

  const handler = ops[request.op];
  let result;
  if (!handler) {
    result = fail(`未知操作：${request.op}。支持的操作：${Object.keys(ops).join('、')}`);
  } else {
    try {
      result = await handler(request);
    } catch (err) {
      // 单个 op 抛异常也不能带走整个桥
      result = fail(`执行 ${request.op} 时出错：${err.message}`);
    }
  }

  result.id = request.id;
  process.stdout.write(JSON.stringify(result) + '\n');
}

rl.on('close', () => {
  if (bot) {
    try { bot.quit(); } catch (err) { /* 已经在关了 */ }
  }
  process.exit(0);
});
