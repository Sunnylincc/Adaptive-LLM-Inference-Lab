from __future__ import annotations

from adaptive_inference.configs import BenchmarkConfig
from adaptive_inference.metrics import InferenceMetrics
from adaptive_inference.policies import InferenceMode, RuleBasedPolicy
from adaptive_inference.profiler import inference_profiler
from adaptive_inference.runners import HFGenerationRunner


def run_static_benchmark(config: BenchmarkConfig, mode: str | InferenceMode) -> list[InferenceMetrics]:
    runner = HFGenerationRunner(config.model_name, config.device)
    rows: list[InferenceMetrics] = []
    for batch_size in config.batch_sizes:
        prompts = (config.prompts * ((batch_size // len(config.prompts)) + 1))[:batch_size]
        with inference_profiler(runner.device) as finish:
            _, prompt_length, output_length = runner.generate(prompts, mode, config.generation)
            profile = finish()
        rows.append(InferenceMetrics.create(config.model_name, str(runner.device), str(mode), prompt_length, output_length, batch_size, profile.latency_seconds, profile.peak_memory_mb))
    return rows


def run_adaptive_benchmark(config: BenchmarkConfig, policy: RuleBasedPolicy | None = None) -> list[InferenceMetrics]:
    policy = policy or RuleBasedPolicy()
    runner = HFGenerationRunner(config.model_name, config.device)
    rows: list[InferenceMetrics] = []
    for target in config.optimization_targets:
        for batch_size in config.batch_sizes:
            prompts = (config.prompts * ((batch_size // len(config.prompts)) + 1))[:batch_size]
            encoded = runner.tokenize(prompts)
            prompt_length = int(encoded["attention_mask"].sum(dim=1).max().item())
            decision = policy.select_mode(prompt_length, batch_size, target, config.memory_budget_mb)
            with inference_profiler(runner.device) as finish:
                _, _, output_length = runner.generate(prompts, decision.mode, config.generation)
                profile = finish()
            rows.append(InferenceMetrics.create(config.model_name, str(runner.device), f"adaptive:{decision.mode}", prompt_length, output_length, batch_size, profile.latency_seconds, profile.peak_memory_mb))
    return rows
