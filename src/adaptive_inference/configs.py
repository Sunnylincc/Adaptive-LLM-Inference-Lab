from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 24
    do_sample: bool = False
    temperature: float = 1.0
    num_beams: int = 1


@dataclass(frozen=True)
class BenchmarkConfig:
    model_name: str = "sshleifer/tiny-gpt2"
    device: str = "auto"
    batch_sizes: list[int] = field(default_factory=lambda: [1, 2, 4])
    prompts: list[str] = field(default_factory=lambda: ["Adaptive inference is"])
    optimization_targets: list[str] = field(default_factory=lambda: ["balanced", "latency", "throughput", "memory"])
    memory_budget_mb: float | None = None
    generation: GenerationSettings = field(default_factory=GenerationSettings)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(path: str | Path) -> BenchmarkConfig:
    raw = load_yaml(path)
    generation = GenerationSettings(**raw.get("generation", {}))
    allowed = {"model_name", "device", "batch_sizes", "prompts", "optimization_targets", "memory_budget_mb"}
    kwargs = {key: value for key, value in raw.items() if key in allowed}
    return BenchmarkConfig(**kwargs, generation=generation)
