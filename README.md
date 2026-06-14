# Adaptive Inference Lab

**A lightweight research lab for benchmarking and building adaptive inference policies for efficient generative AI deployment.**

Adaptive Inference Lab is a compact ML systems project for studying how language-model inference settings should change as workload characteristics change. It focuses on reproducible experiments, transparent profiling, and practical policy logic that can run on CPU or a modest GPU.

## Motivation

Modern generative AI workloads vary widely across prompt length, output length, batch size, latency targets, throughput goals, memory limits, and hardware availability. A single fixed inference configuration is often suboptimal: interactive requests may need low latency, batch jobs may need throughput, and long-context requests may need memory-aware execution. This repository provides a small but extensible framework for testing whether workload-aware inference policies improve latency-memory-throughput trade-offs.

## Research questions

- When does a fixed inference configuration become inefficient?
- Can workload-aware policies improve latency-memory-throughput trade-offs?
- Which workload features best predict the optimal inference configuration?
- How much efficiency can be gained before quality degradation becomes unacceptable?

## System overview

The project is organized around five components:

1. **Benchmark runner**: executes Hugging Face causal language models with configurable prompts, batch sizes, generation settings, and inference modes.
2. **Profiler**: records end-to-end latency, generated tokens per second, prompt length, output length, batch size, device, and peak memory when available.
3. **Inference modes**: exposes baseline, low-latency, throughput, and memory-efficient execution profiles.
4. **Adaptive policy**: uses simple workload rules to choose an inference mode from prompt length, batch size, memory budget, and optimization target.
5. **Result analysis**: writes CSV/JSON outputs and generates plots for latency, throughput, memory, and trade-off analysis.

## Repository layout

```text
adaptive-inference-lab/
├── configs/                  # YAML benchmark profiles
├── notebooks/                # Lightweight analysis notebook placeholders
├── results/                  # Sample CSV/JSON outputs and generated figures
├── scripts/                  # CLI entry points
├── src/adaptive_inference/   # Package source
└── tests/                    # Unit tests for policy and metrics logic
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

The default model is `sshleifer/tiny-gpt2`, which is intentionally small for reproducibility. CUDA is used automatically when available; otherwise the framework falls back to CPU.

## Example benchmark command

Run one static inference mode:

```bash
python scripts/run_benchmark.py \
  --config configs/default.yaml \
  --mode baseline \
  --output-prefix results/baseline_benchmark
```

This creates:

- `results/baseline_benchmark.csv`
- `results/baseline_benchmark.json`

## Example adaptive policy evaluation command

Compare static inference modes against adaptive policy selection:

```bash
python scripts/run_policy_eval.py \
  --config configs/default.yaml \
  --output-prefix results/policy_eval
```

## Plotting results

```bash
python scripts/plot_results.py \
  --input results/sample_results.csv \
  --output-dir results/figures
```

Generated figures include latency, throughput, memory, and latency-throughput trade-off plots.

## Inference modes

| Mode | Intended use | Example behavior |
| --- | --- | --- |
| `baseline` | General-purpose reference configuration | Standard greedy decoding with configured token limit |
| `low_latency` | Interactive, short-prompt requests | Greedy decoding, single beam, minimal overhead |
| `throughput` | Batched generation workloads | Cache-enabled generation for batch efficiency |
| `memory_efficient` | Long prompts or tight memory budgets | Cache disabled to reduce memory pressure |

## Adaptive policy

The first policy is intentionally rule-based and interpretable:

- Short prompt + small batch → `low_latency`
- Large batch → `throughput`
- Long prompt or tight memory budget → `memory_efficient`
- Otherwise → `baseline`

This creates a clear baseline for future learned policies, contextual bandits, Bayesian optimization, and quality-aware routing.

## Metrics schema

Each benchmark row contains:

| Field | Description |
| --- | --- |
| `model_name` | Hugging Face model identifier |
| `device` | Runtime device such as `cpu` or `cuda` |
| `inference_mode` | Static mode or adaptive selected mode |
| `prompt_length` | Maximum tokenized prompt length in the batch |
| `output_length` | Generated tokens per sequence |
| `batch_size` | Number of prompts in the batch |
| `latency_seconds` | End-to-end generation latency |
| `tokens_per_second` | Generated tokens per second across the batch |
| `peak_memory_mb` | Peak GPU memory or CPU RSS delta in MB |
| `timestamp` | UTC measurement timestamp |

## Example result table

| model_name | device | inference_mode | batch_size | latency_seconds | tokens_per_second | peak_memory_mb |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| sshleifer/tiny-gpt2 | cpu | baseline | 1 | 0.85 | 18.82 | 95.0 |
| sshleifer/tiny-gpt2 | cpu | low_latency | 1 | 0.38 | 21.05 | 82.0 |
| sshleifer/tiny-gpt2 | cpu | throughput | 4 | 1.25 | 51.20 | 140.0 |
| sshleifer/tiny-gpt2 | cpu | memory_efficient | 1 | 0.92 | 13.04 | 70.0 |

These sample rows are illustrative; actual values depend on hardware, installed libraries, and system load.

## Configuration

YAML files in `configs/` control model name, device, prompts, batch sizes, optimization targets, memory budget, and generation settings. Add new workload profiles by copying `configs/default.yaml` and changing the fields relevant to the experiment.

## Roadmap

- Quantization-aware benchmarking
- vLLM backend
- Speculative decoding experiments
- Learned policy selection
- VLM inference profiling
- Cloud deployment benchmarks

## License

MIT License. See [LICENSE](LICENSE).
