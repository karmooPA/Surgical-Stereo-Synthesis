
import pytest
import torch

from src.losses import VGG16PerceptualLoss


@pytest.fixture(scope="module")
def loss_fn():
    return VGG16PerceptualLoss(pretrained=False)


def test_loss_returns_finite_scalar(loss_fn):
    prediction = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)

    loss = loss_fn(prediction, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_identical_images_produce_zero_loss(loss_fn):
    image = torch.rand(1, 3, 64, 64)

    loss = loss_fn(image, image.clone())

    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_different_images_produce_positive_loss(loss_fn):
    prediction = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)

    loss = loss_fn(prediction, target)

    assert loss.item() > 0


def test_gradients_propagate_to_prediction(loss_fn):
    prediction = torch.rand(
        1, 3, 64, 64, requires_grad=True
    )
    target = torch.rand(
        1, 3, 64, 64, requires_grad=True
    )

    loss = loss_fn(prediction, target)
    loss.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0
    assert target.grad is None


def test_vgg_parameters_are_frozen(loss_fn):
    for parameter in loss_fn.parameters():
        assert parameter.requires_grad is False


def test_vgg_parameters_receive_no_gradients(loss_fn):
    prediction = torch.rand(
        1, 3, 64, 64, requires_grad=True
    )
    target = torch.rand(1, 3, 64, 64)

    loss = loss_fn(prediction, target)
    loss.backward()

    for parameter in loss_fn.parameters():
        assert parameter.grad is None


def test_batch_processing(loss_fn):
    prediction = torch.rand(2, 3, 64, 64)
    target = torch.rand(2, 3, 64, 64)

    loss = loss_fn(prediction, target)

    assert torch.isfinite(loss)


def test_different_spatial_dimensions(loss_fn):
    prediction = torch.rand(1, 3, 64, 80)
    target = torch.rand(1, 3, 64, 80)

    loss = loss_fn(prediction, target)

    assert torch.isfinite(loss)


def test_mismatched_shapes_raise_error(loss_fn):
    prediction = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 80)

    with pytest.raises(ValueError):
        loss_fn(prediction, target)


def test_invalid_channels_raise_error(loss_fn):
    prediction = torch.rand(1, 1, 64, 64)
    target = torch.rand(1, 1, 64, 64)

    with pytest.raises(ValueError):
        loss_fn(prediction, target)


def test_invalid_dimensions_raise_error(loss_fn):
    prediction = torch.rand(3, 64, 64)
    target = torch.rand(3, 64, 64)

    with pytest.raises(ValueError):
        loss_fn(prediction, target)
