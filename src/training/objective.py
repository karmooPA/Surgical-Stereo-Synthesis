from __future__ import annotations

from torch import Tensor, nn

from src.losses import build_perceptual_loss


class StereoTrainingObjective(nn.Module):

    def __init__(self, perceptual_loss: nn.Module) -> None:
        super().__init__()
        self.perceptual_loss = perceptual_loss
        self.perceptual_loss.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.perceptual_loss.eval()
        return self

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> dict[str, Tensor]:

        perceptual = self.perceptual_loss(prediction, target)

        return {
            "total": perceptual,
            "perceptual": perceptual,
        }


def build_training_objective(
    training_config: dict,
    pretrained_override: bool | None = None,
) -> StereoTrainingObjective:

    loss_config_path = training_config["paths"]["loss"]

    perceptual_loss = build_perceptual_loss(
        config_path=loss_config_path,
        pretrained_override=pretrained_override,
    )

    return StereoTrainingObjective(perceptual_loss)
