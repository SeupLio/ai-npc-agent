"""用例生成器的测试。

这个文件的核心是**防灌水规则的回归测试**。

生成器最容易出的问题不是"生成不出来"，而是"生成了一堆看起来很多、
实际在测同一件事的用例"。这类问题不会报错 —— 它只会让用例数从 15 涨到 220，
然后让通过率看起来很权威。所以每条防灌水规则都要有一条测试盯着。
"""

from __future__ import annotations

import json

import pytest

from npc_agent.config import RuntimeConfig, load_scenario
from npc_agent.eval import generator as G
from npc_agent.eval.harness import CASES_DIR, EvalHarness


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def raw_cases() -> list[dict]:
    return G.generate_cases(target=9999)


@pytest.fixture(scope="module")
def gated() -> G.GateResult:
    return G.gate_cases(G.generate_cases(target=9999))


# --------------------------------------------------------------------------- #
# 1) 规模与覆盖
# --------------------------------------------------------------------------- #
def test_case_set_reaches_the_200_target(raw_cases: list[dict], gated: G.GateResult) -> None:
    """用例集要真的到 200+ —— 而且是通过**结构**到的，不是靠重复措辞。"""
    assert len(gated.kept) >= 200
    coverage = G.coverage_report(gated.kept)
    # 用例数不是卖点，结构数才是。80 个不同结构意味着这 200+ 条在问
    # 至少 80 件不同的事，而不是把 15 件事各问 15 遍。
    assert coverage["distinct_signatures"] >= 70
    assert coverage["distinct_intents"] >= 50
    assert coverage["distinct_shapes"] >= 15


def test_every_category_is_present_and_none_dominates(gated: G.GateResult) -> None:
    """六个维度都要有题，而且不能有一类吃掉大半配额。

    直接按书写顺序截断会让靠前的类别吃满 —— 而类别顺序纯属代码里的偶然。
    这条测试盯的是"分布是设计出来的，不是副产品"。
    """
    by_category = G.coverage_report(gated.kept)["by_category"]
    assert set(by_category) == set(G.GENERATED_CATEGORIES)
    total = sum(by_category.values())
    for category, count in by_category.items():
        assert count / total <= 0.40, f"{category} 占了 {count / total:.0%}，分布失衡"


def test_every_declared_intent_actually_produces_cases(raw_cases: list[dict]) -> None:
    """没有"死意图" —— 声明了却一条也生成不出来的意图，是配置写错了。

    死意图比死断言更隐蔽：它不报错，只是让"我们覆盖了 60 种意图"这句话变成假的。
    """
    produced = {case["intent"] for case in raw_cases}
    declared = {intent.key for intent in G.INTENTS}
    assert declared - produced == set()


# --------------------------------------------------------------------------- #
# 2) 规则一：结构性指纹去重
# --------------------------------------------------------------------------- #
def test_signature_ignores_phrasing_and_player() -> None:
    """换措辞、换玩家名**不算**新结构 —— 它们测的是同一件事。"""
    base = {"tools": ["move_to"], "player_has": {"player_a": ["latte"]}}
    other = {"tools": ["move_to"], "player_has": {"player_b": ["latte"]}}
    assert G._signature("task", "tutorial", "order_latte", "order", base) == G._signature(
        "task", "tutorial", "order_latte", "order", other
    )


def test_signature_includes_turn_shape() -> None:
    """轮次形状**算**结构：它决定问的是哪个问题。

    点一杯拿铁留 8 个空轮 → 测"能不能干完"；
    同一杯被追问三次          → 测"幂等性"。
    这两条不能算同一个结构，否则"幂等性"这项能力就永远没被真正测过。
    """
    expect = {"tools": ["move_to"], "player_has": {"player_a": ["latte"]}}
    a = G._signature("task", "tutorial", "order_latte", "order", expect)
    b = G._signature("task", "tutorial", "order_latte", "insistent", expect)
    assert a != b


def test_signature_separates_different_assertion_kinds() -> None:
    """断言种类不同就是不同结构：`player_has` 和 `flags` 是两种检查方式。"""
    a = G._signature("task", "tutorial", "x", "order", {"player_has": {"player_a": ["latte"]}})
    b = G._signature("task", "tutorial", "x", "order", {"flags": ["learned_order"]})
    assert a != b


def test_max_per_signature_cap_is_respected(raw_cases: list[dict]) -> None:
    """同一个结构最多留 3 条。第 4 条措辞变体不会让结论更可信，只会抬高样本量。"""
    buckets: dict[tuple, int] = {}
    for case in raw_cases:
        sig = G._signature(
            case["category"], case["scenario"], case["intent"], case["shape"], case["expect"]
        )
        buckets[sig] = buckets.get(sig, 0) + 1
    assert buckets, "一个结构都没有，说明生成逻辑坏了"
    assert max(buckets.values()) <= G.MAX_PER_SIGNATURE


def test_no_two_cases_are_completely_identical(raw_cases: list[dict]) -> None:
    """**场景 + 轮次 + 期望**三者都相同才是真重复。

    同一场对话配不同的检查项是正常的 —— 就像一条 `assert` 写不下时
    会拆成几条，它们共用同一个 fixture。灌水的定义是**连断言都一样**：
    那样的第二条除了把样本量刷大，什么也没多测。

    注意范围是**同一场景内**：同样两句话跑在 tutorial 和 icebreaker 里
    是两个不同的世界（不同的物品位置、不同的目标、不同的 NPC 状态）。
    """
    seen: dict[tuple, str] = {}
    for case in raw_cases:
        key = (
            case["scenario"],
            json.dumps(case["turns"], ensure_ascii=False, sort_keys=True),
            json.dumps(case["expect"], ensure_ascii=False, sort_keys=True),
        )
        assert key not in seen, f"{case['id']} 与 {seen[key]} 完全重复"
        seen[key] = case["id"]


def test_interaction_count_is_reported_next_to_case_count(gated: G.GateResult) -> None:
    """报告里要同时给出"用例数"和"交互数"，别让用例数独自承担说服力。

    用例数 200 和交互数 120 放在一起看，读者才知道这 200 条里有多少是
    同一场对话的不同检查维度。只报 200，就是在用数字代替论证。
    """
    report = G.coverage_report(gated.kept)
    assert 0 < report["distinct_interactions"] <= report["total"]
    assert report["distinct_assertions"] == report["total"]
    assert "不同交互" in G.render_coverage(report)


# --------------------------------------------------------------------------- #
# 3) 轮转取样
# --------------------------------------------------------------------------- #
def test_round_robin_balances_the_set_when_it_has_to_truncate() -> None:
    """**截断时**轮转取样保证每个类别都拿到配额。

    这一点只在 target 小于可用量时才看得出来 —— 不截断的时候，各占多少
    完全由"这个类别写了多少结构"决定，顺序不影响总量。
    所以这条测试用小 target 来触发截断。

    没有轮转的话，靠前的类别会吃满配额，而类别顺序只是代码里的书写顺序。
    """
    small = G.generate_cases(target=60)
    counts: dict[str, int] = {}
    for case in small:
        counts[case["category"]] = counts.get(case["category"], 0) + 1
    assert set(counts) == set(G.GENERATED_CATEGORIES), f"有类别一条都没拿到：{counts}"
    assert max(counts.values()) - min(counts.values()) <= 1, f"配额分配不均：{counts}"


def test_no_category_dominates_the_full_set(gated: G.GateResult) -> None:
    """整个用例集里没有哪个类别吃掉大半配额。

    不截断时各类的绝对数量由可用结构数决定，所以这里只要求"没有一类独大"。
    """
    counts = G.coverage_report(gated.kept)["by_category"]
    total = sum(counts.values())
    for category, count in counts.items():
        assert count / total <= 0.40, f"{category} 占了 {count / total:.0%}，分布失衡"


def test_round_robin_preserves_all_drafts() -> None:
    """轮转只改顺序，不丢条目。"""
    drafts = [
        G.Draft(
            case={"id": f"{cat}-{i}", "category": cat, "scenario": "s"},
            signature=(cat, "s", "k", "order", ()),
            intent="k",
        )
        for cat in ("a", "b", "c")
        for i in range(4)
    ]
    out = G._round_robin(drafts)
    assert sorted(d.case_id for d in out) == sorted(d.case_id for d in drafts)
    # 前三条应该分别来自三个类别 —— 这就是"每轮每类一条"
    assert {out[0].case["category"], out[1].case["category"], out[2].case["category"]} == {
        "a", "b", "c"
    }


# --------------------------------------------------------------------------- #
# 4) 确定性
# --------------------------------------------------------------------------- #
def test_same_seed_generates_the_same_case_set() -> None:
    """固定种子 → 同一批题。否则"回归基线"四个字就不成立。"""
    a = G.generate_cases(target=9999, seed=G.DEFAULT_SEED)
    b = G.generate_cases(target=9999, seed=G.DEFAULT_SEED)
    assert [c["id"] for c in a] == [c["id"] for c in b]
    assert [c["turns"] for c in a] == [c["turns"] for c in b]


def test_different_seed_keeps_the_same_structures_but_reshuffles_variants() -> None:
    """换种子换的是"同结构下的哪几条变体"，不是整套题。

    这是刻意的：换种子不该让覆盖面对不上，只该让变体组合不同。
    """
    a = G.generate_cases(target=9999, seed=1)
    b = G.generate_cases(target=9999, seed=2)
    sig_a = {
        G._signature(c["category"], c["scenario"], c["intent"], c["shape"], c["expect"])
        for c in a
    }
    sig_b = {
        G._signature(c["category"], c["scenario"], c["intent"], c["shape"], c["expect"])
        for c in b
    }
    assert sig_a == sig_b


# --------------------------------------------------------------------------- #
# 5) 声明自检
# --------------------------------------------------------------------------- #
def test_intent_declarations_are_valid() -> None:
    """声明写错要在生成阶段炸掉，而不是静默少生成一批用例。"""
    G._validate_intents()


def test_unknown_shape_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = G.Intent(
        key="bad_shape",
        category="task",
        scenarios=["tutorial"],
        phrasings=["来杯拿铁"],
        expect={},
        shapes=["no_such_shape"],
    )
    monkeypatch.setattr(G, "INTENTS", [bad])
    with pytest.raises(ValueError, match="未知轮次形状"):
        G.generate_cases(target=10)


def test_duplicate_intent_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    good = G.Intent(
        key="dup",
        category="task",
        scenarios=["tutorial"],
        phrasings=["来杯拿铁"],
        expect={},
        shapes=["order"],
    )
    monkeypatch.setattr(G, "INTENTS", [good, good])
    with pytest.raises(ValueError, match="重复"):
        G.generate_cases(target=10)


def test_shape_needing_a_follow_up_requires_one(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = G.Intent(
        key="no_followup",
        category="task",
        scenarios=["icebreaker"],
        phrasings=["来杯拿铁"],
        expect={},
        shapes=["two_orders"],
    )
    monkeypatch.setattr(G, "INTENTS", [bad])
    with pytest.raises(ValueError, match="follow_up"):
        G.generate_cases(target=10)


def test_no_case_references_a_player_outside_its_scenario(raw_cases: list[dict]) -> None:
    """tutorial 只有一个玩家，需要"另一个人插话"的形状必须被挡在生成阶段。

    放进去的话会在 `record_player_utterance` 抛 KeyError ——
    那看起来像框架崩了，不像"这条用例不适用"。
    """
    for case in raw_cases:
        known = {str(p["id"]) for p in (load_scenario(case["scenario"]).get("players") or [])}
        assert G._players_in_turns(case["turns"]) <= known, case["id"]


def test_every_case_is_structurally_well_formed(raw_cases: list[dict]) -> None:
    ids = [case["id"] for case in raw_cases]
    assert len(ids) == len(set(ids)), "用例 id 有重复"
    for case in raw_cases:
        assert case["category"] in G.GENERATED_CATEGORIES
        assert case["scenario"] in {"tutorial", "icebreaker", "hosting", "duet", "village"}
        assert case["turns"], f"{case['id']} 没有轮次"
        assert isinstance(case["expect"], dict)
        assert case["shape"] in G.KNOWN_SHAPES
        assert case["generated"] is True


# --------------------------------------------------------------------------- #
# 6) 规则二：离线可达性门禁
# --------------------------------------------------------------------------- #
def test_gate_keeps_only_cases_the_offline_baseline_can_pass(gated: G.GateResult) -> None:
    """门禁的产出必须是"离线跑一遍全绿"。

    这是回归集能被信任的前提：基线一旦不是 100%，之后任何一次"通过率下降"
    都分不清是改坏了，还是踩到了一条本来就过不去的题。
    """
    harness = EvalHarness(RuntimeConfig(llm_provider="null"))
    for case in gated.kept:
        assert harness.run_case(case).passed, case["id"]


def test_gate_records_every_dropped_case_with_a_reason(gated: G.GateResult) -> None:
    """剔除必须可归因：哪条、哪条指标没过、具体说了什么。"""
    assert gated.dropped, "如果一条都没剔，要么生成器完美，要么门禁没生效"
    for entry in gated.dropped:
        assert entry["id"]
        assert entry["failed_metrics"], f"{entry['id']} 被剔了但没说哪条指标没过"
        assert entry["detail"], f"{entry['id']} 被剔了但没有细节"
        # 用例本体要留着 —— 它要能作为"基线盲区"被重新启用
        assert entry["case"]["turns"]


def test_dropped_cases_are_kept_as_blindspots_not_thrown_away(
    gated: G.GateResult, tmp_path
) -> None:
    """被剔的用例不是垃圾，是**基线盲区**：模型跑批时最该看的就是这部分。

    剔出回归集是为了保住基线性质；丢掉就变成了隐藏信息。
    """
    path = G.write_blindspots(gated.dropped, tmp_path / "blindspots.jsonl")
    assert path is not None
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("//")
    ]
    assert len(lines) == len(gated.dropped)
    for line in lines:
        assert json.loads(line)["turns"]


def test_write_blindspots_writes_nothing_when_there_are_no_blindspots(tmp_path) -> None:
    """没有盲区就别在报告目录里留个空壳文件。"""
    assert G.write_blindspots([], tmp_path / "blindspots.jsonl") is None
    assert not (tmp_path / "blindspots.jsonl").exists()


def test_gate_report_is_machine_readable(gated: G.GateResult, tmp_path) -> None:
    path = G.write_gate_report(gated, tmp_path / "generation.json", target=240, seed=7)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["seed"] == 7
    assert payload["target"] == 240
    assert payload["kept"] == len(gated.kept)
    assert payload["dropped"] == len(gated.dropped)
    assert len(payload["blindspots"]) == len(gated.dropped)


# --------------------------------------------------------------------------- #
# 7) 落盘
# --------------------------------------------------------------------------- #
def test_written_case_file_loads_through_the_harness(gated: G.GateResult, tmp_path) -> None:
    """写出来的文件必须能被 harness 原样读回去 —— 否则生成和跑批是两套格式。"""
    path = G.write_cases(gated.kept, tmp_path / "generated.jsonl")
    harness = EvalHarness(RuntimeConfig(llm_provider="null"), cases_dir=tmp_path)
    loaded = harness.load_cases()
    assert len(loaded) == len(gated.kept)
    # 注释头要被正确跳过，且每条都能跑
    assert harness.run_case(loaded[0]).passed


def test_the_committed_case_set_matches_the_generator(gated: G.GateResult) -> None:
    """仓库里的 generated.jsonl 必须和当前生成器一致。

    否则"改了场景配置 → 重新生成"这条纪律会悄悄失效：
    文件还是旧的，用例还是绿的，但它们已经和世界对不上了。
    """
    committed = CASES_DIR / G.GENERATED_FILE
    assert committed.exists(), "generated.jsonl 还没生成，跑一次 cli gencases"
    on_disk = [
        line
        for line in committed.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("//")
    ]
    assert len(on_disk) == len(gated.kept)
    assert [json.loads(l)["id"] for l in on_disk] == [c["id"] for c in gated.kept]


# --------------------------------------------------------------------------- #
# 8) 期望从配置推导，而不是照抄
# --------------------------------------------------------------------------- #
def test_serve_tools_is_derived_from_the_recipe_table() -> None:
    """期望的工具集从配方推导：配方改工位时，用例会跟着变，不会静静失效。"""
    latte = G._serve_tools("latte")
    lemonade = G._serve_tools("lemonade")
    for tools in (latte, lemonade):
        assert "craft_item" in tools and "give_item" in tools and "move_to" in tools
        # set_flag 不在里面：设标记是**场景目标**的步骤，不是点单闭环的一部分。
        # 这一条是被门禁抓出来的真实错误（见 _serve_tools 的注释）。
        assert "set_flag" not in tools


def test_spoiler_terms_are_derived_from_the_knowledge_table() -> None:
    """剧透词从知识库推导：换一个场景，只要标了 requires 就自动进红线表。"""
    star = G._spoiler_terms()
    village = G._village_spoiler_terms()
    assert star and all(isinstance(t, str) for t in star)
    assert village and all(isinstance(t, str) for t in village)
    # 两份必须是**不同**的词表 —— 否则跨世界红线检查是空的
    assert not set(star) & set(village)


def test_locked_flags_come_from_the_scenario_whitelist() -> None:
    """越权目标从白名单反推，而不是手写一串标记名。"""
    locked = G._locked_flags("tutorial")
    allowed = set((load_scenario("tutorial").get("world") or {}).get("settable_flags") or [])
    assert locked
    assert not set(locked) & allowed


def test_out_of_character_terms_cover_every_persona_forbidden_list() -> None:
    """生成器里的出戏词必须覆盖所有人设卡的 forbidden —— 否则换个 NPC 就有漏洞。"""
    from npc_agent.config import load_persona

    for persona_id in ("ayou", "xiaozhou", "ayan"):
        forbidden = set(load_persona(persona_id).get("style", {}).get("forbidden") or [])
        missing = forbidden - set(G.OUT_OF_CHARACTER)
        # "我无法回答" / "我的设定" 这类是措辞，不是"承认自己是程序"的核心词，
        # 允许不重叠；但核心词必须都在。
        assert not (missing & {"语言模型", "作为一个AI", "我是程序", "提示词", "系统提示"}), (
            f"{persona_id} 的 forbidden 里有生成器没覆盖的核心出戏词：{missing}"
        )


def test_coverage_report_explains_itself(gated: G.GateResult) -> None:
    report = G.coverage_report(gated.kept)
    for key in (
        "total",
        "distinct_signatures",
        "distinct_intents",
        "distinct_shapes",
        "by_category",
        "by_scenario",
        "by_shape",
        "by_intent",
        "max_per_signature",
        "cases_per_signature",
    ):
        assert key in report
    # 两个数字要一起看：结构数是"测了多少件事"，用例数是"其中变体复了几遍"
    assert report["total"] >= report["distinct_signatures"]
    assert G.render_coverage(report)


# --------------------------------------------------------------------------- #
# --category 必须按用例自己的 category 过滤，不能只按文件名
# --------------------------------------------------------------------------- #
def test_category_filter_covers_generated_cases_not_just_file_names() -> None:
    """`--category safety` 必须真的跑到 31 条安全用例，而不是 3 条。

    这是实测踩出来的一个**静默少测**：生成的用例全在 `generated.jsonl`
    一个文件里，早期版本按文件 stem 过滤，于是 `--category safety`
    只跑到了 `safety.jsonl` 里那 3 条手写用例。

    它比报错危险得多 —— 命令跑成功了、报告全绿、退出码 0，
    只是测的东西比你以为的少 90%。做安全回归的人会因此得出
    "安全维度没问题"这个错误结论。
    """
    from npc_agent.config import RuntimeConfig
    from npc_agent.eval.harness import EvalHarness

    harness = EvalHarness(RuntimeConfig())
    everything = harness.load_cases()

    for category in ("task", "memory", "persona", "safety", "multi_npc", "minecraft"):
        expected = [c for c in everything if c["category"] == category]
        got = harness.load_cases([category])
        assert len(got) == len(expected), (
            f"--category {category} 拿到 {len(got)} 条，实际有 {len(expected)} 条 —— "
            "按文件名过滤会让生成的用例全部漏掉"
        )
        assert all(c["category"] == category for c in got)

    # 手写用例文件仍然能被单独选中（兼容老用法）
    assert harness.load_cases(["task"])
    # 多个类别可以叠加
    assert len(harness.load_cases(["safety", "persona"])) == sum(
        1 for c in everything if c["category"] in ("safety", "persona")
    )
    # 不存在的类别返回空，而不是"什么都不筛"把全部用例端出来
    assert harness.load_cases(["no_such_category"]) == []


# --------------------------------------------------------------------------- #
# --category 必须按用例自己的 category 过滤，不能只按文件名
# --------------------------------------------------------------------------- #
def test_category_filter_covers_generated_cases_not_just_file_names() -> None:
    """`--category safety` 必须真的跑到 31 条安全用例，而不是 3 条。

    这是实测踩出来的一个**静默少测**：生成的用例全在 `generated.jsonl`
    一个文件里，早期版本按文件 stem 过滤，于是 `--category safety`
    只跑到了 `safety.jsonl` 里那 3 条手写用例。

    它比报错危险得多 —— 命令跑成功了、报告全绿、退出码 0，
    只是测的东西比你以为的少 90%。做安全回归的人会因此得出
    "安全维度没问题"这个错误结论。
    """
    from npc_agent.config import RuntimeConfig
    from npc_agent.eval.harness import EvalHarness

    harness = EvalHarness(RuntimeConfig())
    everything = harness.load_cases()

    for category in ("task", "memory", "persona", "safety", "multi_npc", "minecraft"):
        expected = [c for c in everything if c["category"] == category]
        got = harness.load_cases([category])
        assert len(got) == len(expected), (
            f"--category {category} 拿到 {len(got)} 条，实际有 {len(expected)} 条 —— "
            "按文件名过滤会让生成的用例全部漏掉"
        )
        assert all(c["category"] == category for c in got)

    # 手写用例文件仍然能被单独选中（兼容老用法）
    assert harness.load_cases(["task"])
    # 多个类别可以叠加
    assert len(harness.load_cases(["safety", "persona"])) == sum(
        1 for c in everything if c["category"] in ("safety", "persona")
    )
    # 不存在的类别返回空，而不是"什么都不筛"把全部用例端出来
    assert harness.load_cases(["no_such_category"]) == []
