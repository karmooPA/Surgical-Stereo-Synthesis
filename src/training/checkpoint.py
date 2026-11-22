from __future__ import annotations

import math
import os
from pathlib import Path

import torch
from torch import nn


CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float | None,
    scaler=None,
) -> Path:

    if type(epoch) is not int or epoch < 1:
        raise ValueError("epoch must be a positive integer.")

    if best_metric is not None and (
        not isinstance(best_metric, (int, float))
        or not math.isfinite(best_metric)
    ):
        raise ValueError("best_metric must be finite or None.")

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
    }

    temporary_path = path.with_name(path.name + ".tmp")

    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
    strict: bool = True,
) -> dict:

    path = Path(path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint format.")

    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version.")

    required = {
        "epoch",
        "best_metric",
        "model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
    }

    if not required.issubset(checkpoint):
        raise ValueError("Checkpoint is missing required fields.")

    epoch = checkpoint["epoch"]

    if type(epoch) is not int or epoch < 1:
        raise ValueError("Invalid checkpoint epoch.")

    best_metric = checkpoint["best_metric"]

    if best_metric is not None and (
        not isinstance(best_metric, (int, float))
        or not math.isfinite(best_metric)
    ):
        raise ValueError("Invalid checkpoint metric.")

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=strict,
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if scaler is not None:
        scaler_state = checkpoint["scaler_state_dict"]

        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)

    return {
        "epoch": epoch,
        "start_epoch": epoch + 1,
        "best_metric": best_metric,
    }
