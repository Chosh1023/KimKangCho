"""Shared training/checkpoint utilities."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Mapping

import torch


def move_to_device(batch: Mapping[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if torch.is_floating_point(value)
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for key, value in model.state_dict().items():
            if key in self.shadow:
                self.shadow[key].lerp_(value.detach(), 1.0 - self.decay)

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        self.decay = float(payload["decay"])
        self.shadow = dict(payload["shadow"])

    def copy_to(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        state.update(self.shadow)
        model.load_state_dict(state)

    @contextmanager
    def average_parameters(self, model: torch.nn.Module):
        """Temporarily evaluate with EMA weights and restore train weights."""

        backup = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if key in self.shadow
        }
        self.copy_to(model)
        try:
            yield model
        finally:
            state = model.state_dict()
            state.update(backup)
            model.load_state_dict(state)


def atomic_torch_save(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_json_atomic(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


class MetricAccumulator:
    def __init__(self):
        self.total: dict[str, float] = {}
        self.count = 0

    def update(self, metrics: Mapping[str, torch.Tensor | float], weight: int) -> None:
        self.count += int(weight)
        for key, value in metrics.items():
            number = float(value.detach().mean().cpu()) if isinstance(value, torch.Tensor) else float(value)
            self.total[key] = self.total.get(key, 0.0) + number * int(weight)

    def averages(self) -> dict[str, float]:
        denominator = max(1, self.count)
        return {key: value / denominator for key, value in self.total.items()}
