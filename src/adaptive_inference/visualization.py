from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_results(input_csv: str | Path, output_dir: str | Path) -> list[Path]:
    df = pd.read_csv(input_csv)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for metric, ylabel in [("latency_seconds", "Latency (s)"), ("tokens_per_second", "Tokens/s"), ("peak_memory_mb", "Peak memory (MB)")]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        df.groupby("inference_mode")[metric].mean().sort_values().plot(kind="bar", ax=ax)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Mean {ylabel} by inference mode")
        fig.tight_layout()
        path = out / f"{metric}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(df["latency_seconds"], df["tokens_per_second"])
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Tokens/s")
    ax.set_title("Latency-throughput trade-off")
    fig.tight_layout()
    path = out / "tradeoff.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths
