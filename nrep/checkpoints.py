"""Checkpoint serialization shared by training rounds."""

from typing import Any, Dict, Iterable

import torch

from .runtime import capture_random_state


def build_training_state(
    models: Iterable[torch.nn.Module],
    optimizers: Iterable[torch.optim.Optimizer],
    schedulers: Iterable[Any],
    *,
    epoch: int,
    round_id: int,
) -> Dict[str, Any]:
    """Build one complete, resumable checkpoint for all model branches."""
    state = capture_random_state()
    state["epoch"] = epoch
    state["round_id"] = round_id
    state["models"] = [model.state_dict() for model in models]
    state["optimizers"] = [optimizer.state_dict() for optimizer in optimizers]
    state["schedulers"] = [scheduler.state_dict() for scheduler in schedulers]
    return state
