"""Simple checkpoint management utilities for training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch


class CheckpointManager:
    """Handles persistence of training checkpoints with optional Top-K tracking."""

    def __init__(
        self,
        directory: str | Path,
        prefix: str = "diffusion_policy",
        *,
        run_id: Optional[str] = None,
        top_k: int = 0,
        maximize_metric: bool = False,
    ) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.run_id = run_id
        self.top_k = max(0, int(top_k))
        self.maximize_metric = bool(maximize_metric)
        self._topk_entries: List[Dict[str, object]] = []

    def latest_path(self) -> Path:
        return self.directory / f"{self._base_prefix}_latest.pt"

    @property
    def _base_prefix(self) -> str:
        if self.run_id:
            return f"{self.prefix}_{self.run_id}"
        return self.prefix

    def set_run_id(self, run_id: str) -> None:
        self.run_id = run_id

    def save(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        global_step: int,
        ema_model: Optional[torch.nn.Module] = None,
        metric: Optional[float] = None,
        update_latest: bool = True,
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
        if update_latest:
            torch.save(payload, path)

        if self.top_k > 0 and metric is not None:
            self._update_topk(metric=metric, payload=payload, epoch=epoch, global_step=global_step)

        return path

    def load(self, path: str | Path, map_location: Optional[str | torch.device] = None) -> dict:
        """Load a checkpoint payload from disk."""

        return torch.load(Path(path), map_location=map_location)

    # ------------------------------------------------------------------
    def _update_topk(
        self,
        *,
        metric: float,
        payload: Dict[str, object],
        epoch: int,
        global_step: int,
    ) -> None:
        entries = self._topk_entries
        if len(entries) >= self.top_k:
            worst_metric = entries[-1]["metric"]  # type: ignore[index]
            is_better = metric > worst_metric if self.maximize_metric else metric < worst_metric
            if not is_better:
                return

        filename = (
            f"{self._base_prefix}_epoch{epoch:04d}_step{global_step:06d}_metric{metric:.6f}.pt"
        )
        path = self.directory / filename
        torch.save(payload, path)

        entries.append({"metric": metric, "path": path})
        entries.sort(key=lambda item: item["metric"], reverse=self.maximize_metric)

        while len(entries) > self.top_k:
            removed = entries.pop(-1)
            rem_path = removed["path"]
            if isinstance(rem_path, Path):
                _safe_unlink(rem_path)

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Remove cached Top-K entries from disk."""

        for entry in self._topk_entries:
            path = entry.get("path")
            if isinstance(path, Path):
                _safe_unlink(path)
        self._topk_entries.clear()


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)  # type: ignore[arg-type]
    except TypeError:
        if path.exists():
            path.unlink()


__all__ = ["CheckpointManager"]
