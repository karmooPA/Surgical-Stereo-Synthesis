from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceSettings:
    device: torch.device
    mixed_precision: bool
    amp_dtype: torch.dtype | None


def resolve_device(
    requested: str = "auto",
    mixed_precision: bool = True,
) -> DeviceSettings:

    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {requested}")

    if type(mixed_precision) is not bool:
        raise ValueError("mixed_precision must be boolean.")

    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    elif requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        device = torch.device("cuda")

    elif requested == "mps":
        if (
            not hasattr(torch.backends, "mps")
            or not torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but is unavailable.")
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    amp_enabled = mixed_precision and device.type == "cuda"

    return DeviceSettings(
        device=device,
        mixed_precision=amp_enabled,
        amp_dtype=torch.float16 if amp_enabled else None,
    )
