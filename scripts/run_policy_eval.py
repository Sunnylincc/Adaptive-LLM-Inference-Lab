#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_inference.benchmark import run_adaptive_benchmark, run_static_benchmark
from adaptive_inference.configs import load_config
from adaptive_inference.metrics import summarize_tradeoff, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare static modes against adaptive policy selection.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-prefix", default="results/policy_eval")
    args = parser.parse_args()
    config = load_config(args.config)
    rows = []
    for mode in ["baseline", "low_latency", "throughput", "memory_efficient"]:
        rows.extend(run_static_benchmark(config, mode))
    rows.extend(run_adaptive_benchmark(config))
    write_csv(rows, f"{args.output_prefix}.csv")
    write_json(rows, f"{args.output_prefix}.json")
    print(summarize_tradeoff(rows))


if __name__ == "__main__":
    main()
