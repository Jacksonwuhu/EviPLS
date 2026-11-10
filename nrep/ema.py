"""Exponential moving average of trainable model parameters."""

from typing import Dict

import torch


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self) -> None:
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self.shadow[name].lerp_(parameter.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self) -> None:
        self.backup.clear()
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self.backup[name] = parameter.detach().clone()
                parameter.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self) -> None:
        if not self.backup:
            raise RuntimeError("EMA restore called before apply_shadow")
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(self.backup[name])
        self.backup.clear()
