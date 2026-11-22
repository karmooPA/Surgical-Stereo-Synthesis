
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import VGG16_Weights, vgg16


class VGG16PerceptualLoss(nn.Module):

    def __init__(
        self,
        pretrained: bool = True,
        feature_blocks: int = 3,
        feature_weights: Sequence[float] | None = None,
        distance: str = "mae",
        reduction: str = "mean",
        normalization: str = "imagenet",
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        if not 1 <= feature_blocks <= 3:
            raise ValueError("feature_blocks must be between 1 and 3.")

        if distance != "mae":
            raise ValueError("Only MAE distance is supported.")

        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'.")

        if normalization not in ("imagenet", "none"):
            raise ValueError("Unsupported normalization.")

        if feature_weights is None:
            feature_weights = [1.0] * feature_blocks

        if len(feature_weights) != feature_blocks:
            raise ValueError(
                "feature_weights length must match feature_blocks."
            )

        if any(weight < 0 for weight in feature_weights):
            raise ValueError("Feature weights must be non-negative.")

        if not any(weight > 0 for weight in feature_weights):
            raise ValueError("At least one feature weight must be positive.")

        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = vgg16(weights=weights)
        features = list(vgg.features.children())

        boundaries = [(0, 4), (4, 9), (9, 16)]

        self.blocks = nn.ModuleList([
            nn.Sequential(*features[start:end])
            for start, end in boundaries[:feature_blocks]
        ])

        del vgg

        self.feature_weights = tuple(float(w) for w in feature_weights)
        self.reduction = reduction
        self.freeze_backbone = freeze_backbone

        for parameter in self.blocks.parameters():
            parameter.requires_grad_(not freeze_backbone)

        if normalization == "imagenet":
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = [0.0, 0.0, 0.0]
            std = [1.0, 1.0, 1.0]

        self.register_buffer(
            "mean",
            torch.tensor(mean).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor(std).view(1, 3, 1, 1)
        )

        self.eval()

    def _extract_features(self, image: Tensor) -> list[Tensor]:

        image = (image - self.mean) / self.std
        features = []

        for block in self.blocks:
            image = block(image)
            features.append(image)

        return features

    def forward(
        self,
        prediction: Tensor,
        target: Tensor
    ) -> Tensor:

        if prediction.ndim != 4 or target.ndim != 4:
            raise ValueError("Inputs must have shape [B, 3, H, W].")

        if prediction.shape != target.shape:
            raise ValueError("Prediction and target shapes must match.")

        if prediction.shape[1] != 3:
            raise ValueError("Inputs must contain 3 RGB channels.")

        if not prediction.is_floating_point() or not target.is_floating_point():
            raise TypeError("Inputs must be floating point tensors.")

        if prediction.device != target.device:
            raise ValueError("Inputs must be on the same device.")

        if prediction.device != self.mean.device:
            raise ValueError("Move the loss module to the input device.")

        if min(prediction.shape[-2:]) < 4:
            raise ValueError("Image dimensions must be at least 4.")

        prediction = prediction.float()
        target = target.float()

        predicted_features = self._extract_features(prediction)

        with torch.no_grad():
            target_features = self._extract_features(target)

        loss = prediction.new_zeros(())

        for weight, pred_feature, target_feature in zip(
            self.feature_weights,
            predicted_features,
            target_features
        ):
            loss = loss + weight * F.l1_loss(
                pred_feature,
                target_feature,
                reduction=self.reduction
            )

        return loss
