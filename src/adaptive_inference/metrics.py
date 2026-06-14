from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class InferenceMetrics:
    model_name: str
    device: str
    inference_mode: str
    prompt_length: int
    output_length: int
    batch_size: int
    latency_seconds: float
    tokens_per_second: float
    peak_memory_mb: float | None
    timestamp: str

    @classmethod
    def create(
        cls,
        model_name: str,
        device: str,
        inference_mode: str,
        prompt_length: int,
        output_length: int,
        batch_size: int,
        latency_seconds: float,
        peak_memory_mb: float | None,
    ) -> "InferenceMetrics":
        total_tokens = max(output_length * batch_size, 0)
        tps = total_tokens / latency_seconds if latency_seconds > 0 else 0.0
        return cls(model_name, device, inference_mode, prompt_length, output_length, batch_size, latency_seconds, tps, peak_memory_mb, datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_tradeoff(rows: Iterable[InferenceMetrics]) -> dict[str, float]:
    data = list(rows)
    if not data:
        return {"count": 0.0, "mean_latency_seconds": 0.0, "mean_tokens_per_second": 0.0}
    return {
        "count": float(len(data)),
        "mean_latency_seconds": sum(r.latency_seconds for r in data) / len(data),
        "mean_tokens_per_second": sum(r.tokens_per_second for r in data) / len(data),
    }


def write_csv(rows: Iterable[InferenceMetrics], path: str | Path) -> None:
    records = [row.to_dict() for row in rows]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def write_json(rows: Iterable[InferenceMetrics], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump([row.to_dict() for row in rows], handle, indent=2)
