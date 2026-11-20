"""Configurable U-Net for single-frame stereo-view reconstruction.

This implementation is a clearly documented engineering interpretation of the
modified U-Net described by Brundyn et al. It uses symmetric encoder/decoder
stages, skip connections, no batch normalization, and transposed-convolution
upsampling. The paper does not provide a complete source-level layer/channel
specification, so channel widths and the exact block layout remain configurable
project choices rather than claims of an exact reproduction.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

OutputActivation = Literal["sigmoid", "identity", "tanh"]


class DoubleConv(nn.Module):
    """Two padded 3x3 convolutions, each followed by ReLU; no normalization."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


def _pad_to_match(x: Tensor, reference: Tensor) -> Tensor:
    """Symmetrically pad ``x`` to match ``reference`` without cropping.

    With floor-mode 2x2 max pooling followed by 2x2 stride-2 transposed
    convolution, the upsampled tensor is never larger than its corresponding
    encoder skip tensor. Therefore padding (not cropping) is sufficient and
    matches the decoder alignment convention described in the paper.
    """
    target_height, target_width = reference.shape[-2:]
    height, width = x.shape[-2:]

    pad_height = target_height - height
    pad_width = target_width - width
    if pad_height < 0 or pad_width < 0:
        raise RuntimeError(
            "Decoder feature map is larger than its skip connection; "
            "expected padding-only alignment, but got "
            f"decoder={(height, width)} and skip={(target_height, target_width)}."
        )

    if pad_height == 0 and pad_width == 0:
        return x

    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left
    return F.pad(x, (left, right, top, bottom))


class UNet(nn.Module):
    """Symmetric, configurable 2D U-Net for RGB-to-RGB reconstruction.

    ``depth`` is the number of encoder skip stages and also the number of
    downsampling and decoder upsampling stages. For example, ``depth=5`` has
    five encoder blocks, five 2x max-pooling operations, one bottleneck block,
    and five decoder blocks. The input height and width must each be at least
    ``2**depth`` pixels.

    Args:
        in_channels: Number of input channels (3 for RGB).
        out_channels: Number of output channels (3 for RGB reconstruction).
        base_channels: Number of channels in the first encoder stage.
        depth: Number of encoder/decoder skip stages and down/up-sampling steps.
        max_channels: Upper bound for encoder and bottleneck channel counts.
        output_activation: ``sigmoid`` for [0, 1], ``tanh`` for [-1, 1], or
            ``identity`` for unconstrained output.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        depth: int = 5,
        max_channels: int = 512,
        output_activation: OutputActivation = "sigmoid",
    ) -> None:
        super().__init__()

        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be positive")
        if base_channels < 1 or max_channels < base_channels:
            raise ValueError("max_channels must be at least base_channels")
        if depth < 2:
            raise ValueError("depth must be at least 2")
        if output_activation not in {"sigmoid", "identity", "tanh"}:
            raise ValueError("output_activation must be sigmoid, identity, or tanh")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = depth
        self.max_channels = max_channels
        self.output_activation = output_activation

        channels = [
            min(base_channels * (2**level), max_channels)
            for level in range(depth)
        ]
        bottleneck_channels = min(channels[-1] * 2, max_channels)

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        previous_channels = in_channels
        for stage_channels in channels:
            self.encoders.append(DoubleConv(previous_channels, stage_channels))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            previous_channels = stage_channels

        self.bottleneck = DoubleConv(channels[-1], bottleneck_channels)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        decoder_input_channels = bottleneck_channels
        for skip_channels in reversed(channels):
            self.upconvs.append(
                nn.ConvTranspose2d(
                    decoder_input_channels,
                    skip_channels,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.decoders.append(DoubleConv(skip_channels * 2, skip_channels))
            decoder_input_channels = skip_channels

        self.output_layer = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"expected input [B, C, H, W], received shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, received {x.shape[1]}"
            )

        height, width = x.shape[-2:]
        minimum_size = 2**self.depth
        if height < minimum_size or width < minimum_size:
            raise ValueError(
                f"input height and width must both be at least {minimum_size} "
                f"for depth={self.depth}; received H={height}, W={width}"
            )

        skips: list[Tensor] = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, decoder, skip in zip(
            self.upconvs, self.decoders, reversed(skips)
        ):
            x = upconv(x)
            x = _pad_to_match(x, skip)
            x = decoder(torch.cat((skip, x), dim=1))

        x = self.output_layer(x)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation == "tanh":
            return torch.tanh(x)
        return x
