from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from src.models import UNet
from src.models.unet import _pad_to_match


def test_model_initialization_and_standard_rgb_shape() -> None:
    model = UNet(base_channels=4, depth=3, max_channels=16)
    output = model(torch.rand(2, 3, 32, 40))
    assert output.shape == (2, 3, 32, 40)
    assert torch.isfinite(output).all()
    assert output.min() >= 0 and output.max() <= 1


def test_decoder_alignment_uses_padding_without_cropping() -> None:
    decoder_features = torch.ones(1, 2, 4, 5)
    skip_features = torch.zeros(1, 2, 5, 7)
    aligned = _pad_to_match(decoder_features, skip_features)
    assert aligned.shape[-2:] == skip_features.shape[-2:]
    with pytest.raises(RuntimeError, match="larger than its skip connection"):
        _pad_to_match(torch.ones(1, 2, 6, 5), skip_features)


def test_encoder_pooling_and_decoder_stages_are_symmetric() -> None:
    depth = 5
    model = UNet(base_channels=2, depth=depth, max_channels=32)
    assert len(model.encoders) == depth
    assert len(model.pools) == depth
    assert len(model.upconvs) == depth
    assert len(model.decoders) == depth


def test_paper_and_sample_resolutions_preserve_spatial_shape() -> None:
    # Narrow channels keep these full-resolution CPU smoke tests inexpensive.
    model = UNet(base_channels=2, depth=5, max_channels=32).eval()
    with torch.inference_mode():
        for height, width in ((192, 384), (288, 360)):
            output = model(torch.rand(1, 3, height, width))
            assert output.shape == (1, 3, height, width)
            assert torch.isfinite(output).all()


def test_batch_sizes_and_odd_spatial_dimensions() -> None:
    model = UNet(base_channels=4, depth=3, max_channels=16)
    for batch_size, height, width in ((1, 16, 24), (3, 31, 35), (4, 48, 29)):
        output = model(torch.rand(batch_size, 3, height, width))
        assert output.shape == (batch_size, 3, height, width)
        assert torch.isfinite(output).all()


def test_invalid_input_rank_and_channel_count_raise_clear_errors() -> None:
    model = UNet(base_channels=4, depth=3, max_channels=16)
    with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
        model(torch.rand(3, 32, 32))
    with pytest.raises(ValueError, match="input channels"):
        model(torch.rand(1, 1, 32, 32))


def test_input_smaller_than_required_downsampling_depth_is_rejected() -> None:
    model = UNet(base_channels=4, depth=5, max_channels=32)
    with pytest.raises(ValueError, match="at least 32"):
        model(torch.rand(1, 3, 31, 35))


def test_gradient_propagation_is_finite() -> None:
    model = UNet(base_channels=4, depth=3, max_channels=16)
    inputs = torch.rand(2, 3, 31, 35, requires_grad=True)
    model(inputs).mean().backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_parameter_configurability_and_output_activations() -> None:
    narrow = UNet(base_channels=2, depth=2, max_channels=4, output_activation="identity")
    wide = UNet(base_channels=4, depth=4, max_channels=32, output_activation="tanh")
    assert sum(p.numel() for p in wide.parameters()) > sum(
        p.numel() for p in narrow.parameters()
    )

    identity_output = narrow(torch.rand(1, 3, 17, 19))
    tanh_output = wide(torch.rand(1, 3, 33, 35))
    assert identity_output.shape == (1, 3, 17, 19)
    assert tanh_output.shape == (1, 3, 33, 35)
    assert tanh_output.min() >= -1 and tanh_output.max() <= 1


def test_model_contains_no_batch_normalization() -> None:
    model = UNet(base_channels=4, depth=3, max_channels=16)
    assert not any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules())


def test_state_dict_can_be_saved_and_reloaded(tmp_path: Path) -> None:
    model = UNet(base_channels=4, depth=3, max_channels=16).eval()
    inputs = torch.rand(1, 3, 31, 35)
    with torch.inference_mode():
        expected = model(inputs)

    checkpoint_path = tmp_path / "unet_state_dict.pt"
    torch.save(model.state_dict(), checkpoint_path)

    restored = UNet(base_channels=4, depth=3, max_channels=16).eval()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    restored.load_state_dict(state_dict)
    with torch.inference_mode():
        actual = restored(inputs)

    torch.testing.assert_close(actual, expected)


def test_compatibility_with_stereo_dataset(tmp_path: Path) -> None:
    # Import lazily so the model-only tests can run independently. In the full
    # repository this integration test exercises the existing Stage 2 loader.
    dataset_module = pytest.importorskip("src.datasets")
    StereoPairDataset = dataset_module.StereoPairDataset
    create_stereo_dataloader = dataset_module.create_stereo_dataloader

    for view, offset in (("image01", 0), ("image02", 10)):
        view_dir = tmp_path / "rectified01" / view
        view_dir.mkdir(parents=True)
        for frame in range(2):
            pixels = np.full((32, 35, 3), frame + offset, dtype=np.uint8)
            Image.fromarray(pixels).save(view_dir / f"frame_{frame:03d}.png")

    dataset = StereoPairDataset(tmp_path, input_view="image01", target_view="image02")
    loader = create_stereo_dataloader(dataset, batch_size=2, num_workers=0, shuffle=False)
    batch = next(iter(loader))
    model = UNet(base_channels=2, depth=2, max_channels=4)
    output = model(batch["input"])

    assert output.shape == batch["target"].shape == (2, 3, 32, 35)
    assert batch["input"].min() >= 0 and batch["input"].max() <= 1
