from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import psutil
import torch


@dataclass(frozen=True)
class ProfileResult:
    latency_seconds: float
    peak_memory_mb: float | None


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


@contextmanager
def inference_profiler(device: torch.device):
    process = psutil.Process()
    start_cpu = process.memory_info().rss / (1024**2)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    yield lambda: _finish(start, start_cpu, process, device)


def _finish(start: float, start_cpu: float, process: psutil.Process, device: torch.device) -> ProfileResult:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak = max(process.memory_info().rss / (1024**2) - start_cpu, 0.0)
    return ProfileResult(time.perf_counter() - start, peak)
