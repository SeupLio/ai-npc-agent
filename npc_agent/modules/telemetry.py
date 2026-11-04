"""成本与延迟遥测 —— 回答上线前必问的两个问题。

## 为什么这个模块必须存在

米哈游公开方案里把「千万级并发与成本」列为工程三关之一。这套框架
此前只有 `TurnBudget` 的**字符估算**：它管「别把一轮聊爆」，但回答不了
产品侧真正的问题——

    「一万个玩家每人聊十轮，我要花多少钱、加多少台机器？」

这两个问题不解决，NPC 再像人也不能上线：成本是**上线决策**，
不是调参问题。所以这里做三件事，全部可离线复现：

1. **真实 token 计量**（`TokenUsage`）：LLM 响应里的 `usage` 字段
   被记录、被累计，不再只有"估算字符"。离线路径天然是 0 ——
   这本身就是一个有用的读数（启发式地板价）。
2. **延迟分布**（`LatencyStats`）：p50 / p95 / p99 与最大值。
   平均值会骗人——卡顿全发生在长尾上，玩家体感是 p95 决定的。
3. **成本画像**（`CostProfile`）：按可配单价把 token 折成钱，
   附上"每百轮会话成本"的推算，让接游戏的团队直接拿去算账。

## 设计约束

- **零依赖、离线可跑**：与仓库其它模块同一条纪律。
- **不改变任何决策路径**：遥测只读、不拦。拦截是 `TurnBudget` 的事，
  两个模块各管一段，混在一起会让"限流"和"计量"互相污染。
- **会话粒度**：一个 Cast 一个 `SessionTelemetry`。NPC 级归因
  （"阿柚比小舟费钱"）对成本决策没有增量——决策永远是
  "这套场景配置 + 这个模型要花多少"。
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _percentile(samples: list[float], ratio: float) -> float:
    """有序样本的分位数。空样本返回 0，而不是抛异常——
    遥测永远不该让主流程挂掉。"""
    if not samples:
        return 0.0
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, max(0, round(ratio * (len(ordered) - 1))))
    return ordered[idx]


@dataclass
class TokenUsage:
    """一段会话里真实的 token 消耗（来自 LLM 响应的 usage 字段）。

    `calls` 与 `tokens` 分开记：断言"到底调了几次模型"是
    排查成本异常的第一步——同一句话重试三次,token 会翻倍,
    但只有 calls 能暴露"重试"这个行为。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += max(0, int(prompt))
        self.completion_tokens += max(0, int(completion))
        self.calls += 1

    def merge(self, other: "TokenUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.calls += other.calls

    def render(self) -> str:
        if self.calls == 0:
            return "未调用模型（离线路径，0 token）"
        return (
            f"调用 {self.calls} 次：输入 {self.prompt_tokens}"
            f" + 输出 {self.completion_tokens} = {self.total_tokens} tokens"
        )


@dataclass
class LatencyStats:
    """每轮决策耗时的分布。单位毫秒。"""

    samples: list[float] = field(default_factory=list)

    def observe(self, ms: float) -> None:
        if ms >= 0:
            self.samples.append(float(ms))

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def p50(self) -> float:
        return _percentile(self.samples, 0.50)

    @property
    def p95(self) -> float:
        return _percentile(self.samples, 0.95)

    @property
    def p99(self) -> float:
        return _percentile(self.samples, 0.99)

    @property
    def peak(self) -> float:
        return max(self.samples) if self.samples else 0.0

    def render(self) -> str:
        if not self.samples:
            return "（暂无延迟样本）"
        return (
            f"轮数 {self.count}，均值 {self.mean:.0f}ms，"
            f"p50 {self.p50:.0f} / p95 {self.p95:.0f} / p99 {self.p99:.0f}ms，"
            f"峰值 {self.peak:.0f}ms"
        )


@dataclass
class CostProfile:
    """把 token 折成钱。单价由调用方给（不同厂商差一个数量级），
    默认值只是"留个能算的数"，不是任何一家的报价。"""

    #: 每百万输入 token 的价格（元）。默认值按主流 API 的量级取整。
    price_per_m_input: float = 2.0
    #: 每百万输出 token 的价格（元）。输出通常贵一个数量级。
    price_per_m_output: float = 8.0
    #: 换算显示用的货币单位。
    currency: str = "元"

    def cost_of(self, usage: TokenUsage) -> float:
        return (
            usage.prompt_tokens / 1_000_000 * self.price_per_m_input
            + usage.completion_tokens / 1_000_000 * self.price_per_m_output
        )

    @classmethod
    def from_env(cls) -> "CostProfile":
        """从环境变量读单价 —— 游戏服务/CLI 每次启动时调一次。

        为什么放环境变量而不是 YAML：单价是**部署参数**（同一个
        框架今天接 DeepSeek 明天接 GPT），跟着部署环境走、不跟着
        仓库代码走，改价不需要改一次配置文件、提一次 PR。
        """
        import os

        def _num(key: str, default: float) -> float:
            try:
                return float(os.environ.get(key, "") or default)
            except ValueError:
                return default

        return cls(
            price_per_m_input=_num("NPC_AGENT_PRICE_PER_M_INPUT", 2.0),
            price_per_m_output=_num("NPC_AGENT_PRICE_PER_M_OUTPUT", 8.0),
            currency=os.environ.get("NPC_AGENT_COST_CURRENCY", "元") or "元",
        )

    def render(self, usage: TokenUsage) -> str:
        cost = self.cost_of(usage)
        if usage.calls == 0:
            return "成本 0（离线路径不调模型）"
        return f"成本约 {cost:.4f} {self.currency}（按 {self.price_per_m_input}/{self.price_per_m_output} 每百万 token 计）"


class SessionTelemetry:
    """一个会话（一个 Cast）的全部遥测。

    用法：`cast.step()` 之后把 `AgentTurn.latency_ms` 喂进 `note_turn`，
    把一次玩家输入的墙钟耗时喂进 `note_round`；LLM 层每次响应的
    usage 喂进 `note_llm`。最后 `snapshot()` 出画像。

    两个延迟分开记，因为它们回答不同的问题：
    - NPC 轮延迟（`latency`）：单个 NPC 决策多快 —— 机器够不够；
    - 玩家输入轮延迟（`round_latency`）：玩家按下发送到看见回复 ——
      体验好不好。**后者才是产品口径**：一轮里 3 个 NPC 各 50ms，
      串行下来就是 150ms。
    """

    def __init__(self, profile: CostProfile | None = None) -> None:
        self.usage = TokenUsage()
        self.latency = LatencyStats()
        self.round_latency = LatencyStats()
        self.profile = profile or CostProfile()
        #: 发言轮数（有台词或动作的轮）。回合数不等于发言数——
        #: 大量"沉默轮"是冷场调度在正常工作，不该算进"NPC 很忙"。
        self.spoken_turns: int = 0
        self.total_turns: int = 0
        #: 玩家输入轮数（一次 /say 记一轮）。
        self.rounds: int = 0

    def note_turn(self, latency_ms: float, *, spoken: bool = False) -> None:
        self.latency.observe(latency_ms)
        self.total_turns += 1
        if spoken:
            self.spoken_turns += 1

    def note_round(self, latency_ms: float) -> None:
        self.round_latency.observe(latency_ms)
        self.rounds += 1

    def note_llm(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.usage.add(prompt_tokens, completion_tokens)

    def adopt_llm_usage(self, llm: object) -> None:
        """把 LLM 实例上累计的用量**直接采用**为本会话读数。

        为什么是替换而不是累加：LLM 实例上的计数器是单调递增的，
        逐轮读再叠加会重复计费。采用"累计值替换"语义，
        游戏服务端每轮之后调一次就永远是对的。

        离线路径（NullLLM / ScriptedLLM）没有 `usage` 属性 ——
        保持零读数，这正是"离线地板价"的含义。
        """
        usage = getattr(llm, "usage", None)
        if isinstance(usage, TokenUsage):
            self.usage = usage

    # ------------------------------------------------------------------ #
    def per_hundred_rounds(self) -> float:
        """每 100 次玩家输入的推算成本。空会话返回 0。"""
        if self.rounds == 0:
            return 0.0
        return self.profile.cost_of(self.usage) * 100.0 / self.rounds

    def snapshot(self) -> dict[str, float | int]:
        return {
            "rounds": self.rounds,
            "turns": self.total_turns,
            "spoken_turns": self.spoken_turns,
            "llm_calls": self.usage.calls,
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "total_tokens": self.usage.total_tokens,
            "round_latency_mean_ms": round(self.round_latency.mean, 1),
            "round_latency_p50_ms": round(self.round_latency.p50, 1),
            "round_latency_p95_ms": round(self.round_latency.p95, 1),
            "latency_mean_ms": round(self.latency.mean, 1),
            "latency_p50_ms": round(self.latency.p50, 1),
            "latency_p95_ms": round(self.latency.p95, 1),
            "latency_p99_ms": round(self.latency.p99, 1),
            "latency_peak_ms": round(self.latency.peak, 1),
            "estimated_cost": round(self.profile.cost_of(self.usage), 6),
            "cost_per_100_rounds": round(self.per_hundred_rounds(), 6),
        }

    def render(self) -> str:
        lines = [
            f"玩家输入 {self.rounds} 轮 / NPC 回合 {self.total_turns}（其中发言 {self.spoken_turns}）",
            f"{self.usage.render()}",
            f"NPC 轮延迟：{self.latency.render()}",
        ]
        if self.rounds:
            lines.append(f"玩家口径延迟：{self.round_latency.render()}")
            lines.append(f"每 100 轮约 {self.per_hundred_rounds():.4f} {self.profile.currency}")
        lines.append(self.profile.render(self.usage))
        return "；".join(lines)


__all__ = [
    "CostProfile",
    "LatencyStats",
    "SessionTelemetry",
    "TokenUsage",
]
