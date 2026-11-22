from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.data import DataLoader

from .device import DeviceSettings


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        objective: nn.Module,
        device_settings: DeviceSettings,
        gradient_clip_norm: float | None = None,
    ) -> None:

        self.device = device_settings.device
        self.mixed_precision = device_settings.mixed_precision
        self.amp_dtype = device_settings.amp_dtype

        if self.mixed_precision and (
            self.device.type != "cuda" or self.amp_dtype is None
        ):
            raise ValueError("Mixed precision requires a valid CUDA configuration.")

        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive.")

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.objective = objective.to(self.device)
        self.gradient_clip_norm = gradient_clip_norm

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.mixed_precision,
        )

    def _autocast(self):
        if self.mixed_precision:
            return torch.autocast(
                device_type="cuda",
                dtype=self.amp_dtype,
            )

        return nullcontext()

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
    ) -> dict[str, float | int]:

        if len(loader) == 0:
            raise ValueError("DataLoader contains no batches.")

        self.model.train(training)
        self.objective.train(training)

        total_loss = 0.0
        perceptual_loss = 0.0
        samples = 0
        batches = 0

        context = torch.enable_grad() if training else torch.no_grad()

        with context:
            for batch in loader:

                if not isinstance(batch, dict):
                    raise TypeError("Each batch must be a dictionary.")

                if "input" not in batch or "target" not in batch:
                    raise KeyError("Batch must contain input and target.")

                inputs = batch["input"].to(
                    self.device,
                    non_blocking=self.device.type == "cuda",
                )

                targets = batch["target"].to(
                    self.device,
                    non_blocking=self.device.type == "cuda",
                )

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                with self._autocast():
                    predictions = self.model(inputs)
                    losses = self.objective(predictions, targets)

                    if not isinstance(losses, dict):
                        raise TypeError("Objective must return a dictionary.")

                    if "total" not in losses or "perceptual" not in losses:
                        raise KeyError("Objective must return total and perceptual.")

                    loss = losses["total"]
                    perceptual = losses["perceptual"]

                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise FloatingPointError("Invalid total loss.")

                if perceptual.ndim != 0 or not torch.isfinite(perceptual):
                    raise FloatingPointError("Invalid perceptual loss.")

                if training:
                    if self.mixed_precision:
                        self.scaler.scale(loss).backward()

                        if self.gradient_clip_norm is not None:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.gradient_clip_norm,
                            )

                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                    else:
                        loss.backward()

                        if self.gradient_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.gradient_clip_norm,
                            )

                        self.optimizer.step()

                batch_size = inputs.shape[0]

                total_loss += loss.detach().item() * batch_size
                perceptual_loss += perceptual.detach().item() * batch_size
                samples += batch_size
                batches += 1

        if samples == 0:
            raise ValueError("No samples were processed.")

        return {
            "loss": total_loss / samples,
            "perceptual_loss": perceptual_loss / samples,
            "samples": samples,
            "batches": batches,
        }

    def train_epoch(
        self,
        loader: DataLoader,
    ) -> dict[str, float | int]:

        return self._run_epoch(loader, training=True)

    def validate(
        self,
        loader: DataLoader,
    ) -> dict[str, float | int]:

        return self._run_epoch(loader, training=False)
