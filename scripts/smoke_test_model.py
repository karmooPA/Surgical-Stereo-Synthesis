"""Run a lightweight forward-pass smoke test for the U-Net model."""

from __future__ import annotations

import argparse

import torch

from src.models import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--max-channels", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")

    device = torch.device(args.device)
    model = UNet(
        base_channels=args.base_channels,
        depth=args.depth,
        max_channels=args.max_channels,
    ).to(device)
    model.eval()

    inputs = torch.rand(1, 3, args.height, args.width, device=device)
    with torch.inference_mode():
        outputs = model(inputs)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("U-Net smoke test: PASSED")
    print(f"Device: {device}")
    print(f"Parameters: {parameter_count:,}")
    print(f"Input shape:  {tuple(inputs.shape)}")
    print(f"Output shape: {tuple(outputs.shape)}")
    print(f"Output range: [{outputs.min().item():.6f}, {outputs.max().item():.6f}]")
    print(f"Finite output: {bool(torch.isfinite(outputs).all())}")


if __name__ == "__main__":
    main()
