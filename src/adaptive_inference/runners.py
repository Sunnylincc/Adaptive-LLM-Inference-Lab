from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from adaptive_inference.configs import GenerationSettings
from adaptive_inference.policies import InferenceMode
from adaptive_inference.profiler import resolve_device


@dataclass
class HFGenerationRunner:
    model_name: str = "sshleifer/tiny-gpt2"
    device_name: str = "auto"

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def tokenize(self, prompts: Sequence[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=False)
        return {key: value.to(self.device) for key, value in encoded.items()}

    def generate(self, prompts: Sequence[str], mode: str | InferenceMode, settings: GenerationSettings) -> tuple[torch.Tensor, int, int]:
        inputs = self.tokenize(prompts)
        prompt_length = int(inputs["attention_mask"].sum(dim=1).max().item())
        kwargs = self._mode_kwargs(InferenceMode(mode), settings)
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs, pad_token_id=self.tokenizer.eos_token_id)
        output_length = max(int(output.shape[-1]) - int(inputs["input_ids"].shape[-1]), 0)
        return output, prompt_length, output_length

    @staticmethod
    def _mode_kwargs(mode: InferenceMode, settings: GenerationSettings) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": settings.do_sample,
            "temperature": settings.temperature,
            "num_beams": settings.num_beams,
        }
        if mode == InferenceMode.LOW_LATENCY:
            kwargs.update({"num_beams": 1, "do_sample": False})
        elif mode == InferenceMode.THROUGHPUT:
            kwargs.update({"use_cache": True})
        elif mode == InferenceMode.MEMORY_EFFICIENT:
            kwargs.update({"use_cache": False, "num_beams": 1, "do_sample": False})
        return kwargs
