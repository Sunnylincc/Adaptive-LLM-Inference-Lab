from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class InferenceMode(StrEnum):
    BASELINE = "baseline"
    LOW_LATENCY = "low_latency"
    THROUGHPUT = "throughput"
    MEMORY_EFFICIENT = "memory_efficient"


class OptimizationTarget(StrEnum):
    BALANCED = "balanced"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"


@dataclass(frozen=True)
class PolicyDecision:
    mode: InferenceMode
    reason: str


@dataclass(frozen=True)
class RuleBasedPolicy:
    short_prompt_tokens: int = 64
    long_prompt_tokens: int = 256
    large_batch_size: int = 4
    tight_memory_budget_mb: float = 1024.0

    def select_mode(
        self,
        prompt_length: int,
        batch_size: int,
        optimization_target: str | OptimizationTarget = OptimizationTarget.BALANCED,
        memory_budget_mb: float | None = None,
    ) -> PolicyDecision:
        target = OptimizationTarget(optimization_target)
        if memory_budget_mb is not None and memory_budget_mb <= self.tight_memory_budget_mb:
            return PolicyDecision(InferenceMode.MEMORY_EFFICIENT, "memory budget is tight")
        if target == OptimizationTarget.MEMORY or prompt_length >= self.long_prompt_tokens:
            return PolicyDecision(InferenceMode.MEMORY_EFFICIENT, "long prompt or memory optimization target")
        if target == OptimizationTarget.THROUGHPUT or batch_size >= self.large_batch_size:
            return PolicyDecision(InferenceMode.THROUGHPUT, "large batch or throughput optimization target")
        if target == OptimizationTarget.LATENCY and prompt_length <= self.short_prompt_tokens and batch_size <= 2:
            return PolicyDecision(InferenceMode.LOW_LATENCY, "short prompt and small batch")
        if prompt_length <= self.short_prompt_tokens and batch_size == 1:
            return PolicyDecision(InferenceMode.LOW_LATENCY, "interactive short-prompt workload")
        return PolicyDecision(InferenceMode.BASELINE, "default balanced configuration")
