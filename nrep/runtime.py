"""Reproducibility helpers shared by experiment entry points."""

import random
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and every available PyTorch device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def capture_random_state() -> Dict[str, Any]:
    """Capture RNG state without requiring CUDA on CPU-only hosts."""
    state: Dict[str, Any] = {
        "torch_rng": torch.get_rng_state(),
        "random_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
    }
    state["cuda_rng"] = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    return state


def restore_random_state(state: Dict[str, Any]) -> None:
    """Restore a state produced by :func:`capture_random_state`."""
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["random_rng"])
    np.random.set_state(state["numpy_rng"])
    if torch.cuda.is_available() and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state(state["cuda_rng"])
