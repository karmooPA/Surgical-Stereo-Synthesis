
import torch

from src.losses import VGG16PerceptualLoss


def main():
    device = torch.device("cpu")

    loss_fn = VGG16PerceptualLoss(pretrained=False)
    loss_fn = loss_fn.to(device)

    prediction = torch.rand(
        1, 3, 64, 64,
        device=device,
        requires_grad=True
    )

    target = torch.rand(
        1, 3, 64, 64,
        device=device
    )

    loss = loss_fn(prediction, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0

    print("Perceptual Loss smoke test: PASSED")
    print(f"Device: {device}")
    print(f"Loss: {loss.item():.6f}")
    print(f"Gradient norm: {prediction.grad.norm().item():.6f}")
    print("Gradient propagation: PASSED")


if __name__ == "__main__":
    main()
