
from pathlib import Path

import yaml

from .perceptual_loss import VGG16PerceptualLoss


def build_perceptual_loss(
    config_path: str | Path = "configs/loss.yaml",
    pretrained_override: bool | None = None,
) -> VGG16PerceptualLoss:

    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid loss configuration.")

    settings = config.get("perceptual_loss")

    if not isinstance(settings, dict):
        raise ValueError("Missing perceptual_loss configuration.")

    if settings.get("name") != "vgg16":
        raise ValueError("Only VGG16 perceptual loss is supported.")

    input_settings = settings.get("input", {})

    if input_settings.get("channels", 3) != 3:
        raise ValueError("VGG16 requires three input channels.")

    value_range = input_settings.get("value_range", [0.0, 1.0])

    if [float(value) for value in value_range] != [0.0, 1.0]:
        raise ValueError("Input images must use the [0, 1] range.")

    pretrained = settings.get("pretrained", True)

    if pretrained_override is not None:
        pretrained = pretrained_override

    return VGG16PerceptualLoss(
        pretrained=pretrained,
        feature_blocks=settings.get("feature_blocks", 3),
        feature_weights=settings.get("feature_weights"),
        distance=settings.get("distance", "mae"),
        reduction=settings.get("reduction", "mean"),
        normalization=input_settings.get(
            "normalization", "imagenet"
        ),
        freeze_backbone=settings.get("freeze_backbone", True),
    )
