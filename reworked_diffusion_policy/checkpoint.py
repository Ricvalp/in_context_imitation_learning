"""Simple checkpoint management utilities for training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


class CheckpointManager:
    """Handles persistence of the latest training checkpoint."""

    def __init__(self, directory: str | Path, prefix: str = "diffusion_policy") -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix

    def latest_path(self) -> Path:
        return self.directory / f"{self.prefix}_latest.pt"

    def save(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        global_step: int,
        ema_model: Optional[torch.nn.Module] = None,
    ) -> Path:
        """Persist the latest checkpoint to disk."""

        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
        }
        if ema_model is not None:
            payload["ema_model"] = ema_model.state_dict()

        path = self.latest_path()
        torch.save(payload, path)
        return path

    def load(self, path: str | Path, map_location: Optional[str | torch.device] = None) -> dict:
        """Load a checkpoint payload from disk."""

        return torch.load(Path(path), map_location=map_location)


__all__ = ["CheckpointManager"]
