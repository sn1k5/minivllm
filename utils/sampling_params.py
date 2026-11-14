from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SamplingParams"]


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int = 128
    ignore_eos: bool = False
    priority: int = 0
    deadline_s: float | None = None

    def __post_init__(self) -> None:
        if self.max_tokens < 0:
            raise ValueError(f"max_tokens must be >= 0, got {self.max_tokens}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_p is not None and not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k is not None and self.top_k == 0:
            raise ValueError("top_k=0 is invalid; use None to disable")
        if self.min_p is not None and not (0.0 <= self.min_p <= 1.0):
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}")
        if self.deadline_s is not None and self.deadline_s <= 0:
            raise ValueError(f"deadline_s must be > 0, got {self.deadline_s}")
