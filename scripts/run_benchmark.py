#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_inference.benchmark import run_static_benchmark
from adaptive_inference.configs import load_config
from adaptive_inference.metrics import write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a static Hugging Face inference benchmark.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", default="baseline", choices=["baseline", "low_latency", "throughput", "memory_efficient"])
    parser.add_argument("--output-prefix", default="results/benchmark")
    args = parser.parse_args()
    rows = run_static_benchmark(load_config(args.config), args.mode)
    write_csv(rows, f"{args.output_prefix}.csv")
    write_json(rows, f"{args.output_prefix}.json")
    print(f"wrote {len(rows)} rows to {args.output_prefix}.csv/.json")


if __name__ == "__main__":
    main()
