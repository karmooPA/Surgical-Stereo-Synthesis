from __future__ import annotations

import math

import torch
from torch import nn


def build_optimizer(
    model: nn.Module,
    training_config: dict,
) -> torch.optim.Optimizer:

    optimizer_config = training_config.get("optimizer")

    if not isinstance(optimizer_config, dict):
        raise ValueError("Missing optimizer configuration.")

    if optimizer_config.get("name") != "adam":
        raise ValueError("Only Adam optimizer is supported.")

    learning_rate = optimizer_config.get("learning_rate")
    weight_decay = optimizer_config.get("weight_decay", 0.0)
    betas = optimizer_config.get("betas", [0.9, 0.999])

    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be positive and finite.")

    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, (int, float))
        or not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError("weight_decay must be non-negative and finite.")

    if (
        not isinstance(betas, (list, tuple))
        or len(betas) != 2
        or any(
            isinstance(beta, bool)
            or not isinstance(beta, (int, float))
            or not math.isfinite(beta)
            or not 0 <= beta < 1
            for beta in betas
        )
    ):
        raise ValueError("betas must contain two values in [0, 1).")

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not parameters:
        raise ValueError("Model has no trainable parameters.")

    return torch.optim.Adam(
        parameters,
        lr=float(learning_rate),
        betas=tuple(float(beta) for beta in betas),
        weight_decay=float(weight_decay),
    )
