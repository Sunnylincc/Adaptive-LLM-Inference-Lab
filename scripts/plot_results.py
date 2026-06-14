#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_inference.visualization import plot_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark CSV results.")
    parser.add_argument("--input", default="results/sample_results.csv")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()
    for path in plot_results(args.input, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
