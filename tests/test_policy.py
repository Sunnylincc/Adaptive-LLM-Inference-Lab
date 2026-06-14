from adaptive_inference.policies import InferenceMode, RuleBasedPolicy


def test_short_small_batch_prefers_low_latency():
    decision = RuleBasedPolicy().select_mode(prompt_length=16, batch_size=1, optimization_target="latency")
    assert decision.mode == InferenceMode.LOW_LATENCY


def test_large_batch_prefers_throughput():
    decision = RuleBasedPolicy().select_mode(prompt_length=32, batch_size=8, optimization_target="balanced")
    assert decision.mode == InferenceMode.THROUGHPUT


def test_long_prompt_prefers_memory_efficient():
    decision = RuleBasedPolicy().select_mode(prompt_length=512, batch_size=1, optimization_target="balanced")
    assert decision.mode == InferenceMode.MEMORY_EFFICIENT
