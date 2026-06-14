from adaptive_inference.metrics import InferenceMetrics, summarize_tradeoff


def test_tokens_per_second_uses_output_tokens_and_batch_size():
    metric = InferenceMetrics.create("m", "cpu", "baseline", 10, 5, 2, 2.0, None)
    assert metric.tokens_per_second == 5.0


def test_summary_handles_empty_input():
    assert summarize_tradeoff([])["count"] == 0.0
