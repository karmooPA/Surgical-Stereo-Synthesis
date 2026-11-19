from pathlib import Path

import pytest
import torch
from PIL import Image

from src.datasets.stereo_dataset import StereoPairDataset, create_stereo_dataloader, inspect_sequence


def write_image(path: Path, color: tuple[int, int, int], size=(6, 4)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def make_sequence(root: Path, name: str, frame_ids=("000", "001")) -> None:
    for frame_id in frame_ids:
        write_image(root / name / "image01" / f"{frame_id}.jpg", (255, 0, 0))
        write_image(root / name / "image02" / f"{frame_id}.png", (0, 255, 0))


def test_multi_sequence_pairing_is_sequence_local(tmp_path: Path) -> None:
    make_sequence(tmp_path, "rectified01", ("000", "001"))
    make_sequence(tmp_path, "rectified02", ("000",))
    dataset = StereoPairDataset(tmp_path)

    assert len(dataset) == 3
    assert {(item.sequence_id, item.frame_id) for item in dataset.records} == {
        ("rectified01", "000"),
        ("rectified01", "001"),
        ("rectified02", "000"),
    }
    sample = dataset[0]
    assert sample["input"].shape == (3, 4, 6)
    assert sample["target"].shape == (3, 4, 6)
    assert torch.all((sample["input"] >= 0) & (sample["input"] <= 1))
    assert torch.isfinite(sample["input"]).all()


def test_dataloader_batches_without_device_transfer(tmp_path: Path) -> None:
    make_sequence(tmp_path, "rectified01", ("000", "001", "002"))
    loader = create_stereo_dataloader(StereoPairDataset(tmp_path), batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["input"].shape == (2, 3, 4, 6)
    assert batch["target"].shape == (2, 3, 4, 6)
    assert batch["sequence_id"] == ["rectified01", "rectified01"]
    assert batch["frame_id"] == ["000", "001"]


def test_resize_is_configurable(tmp_path: Path) -> None:
    make_sequence(tmp_path, "rectified01", ("000",))
    sample = StereoPairDataset(tmp_path, image_size=(8, 10))[0]
    assert sample["input"].shape == (3, 8, 10)
    assert sample["original_size"] == (4, 6)


def test_missing_pair_is_detected(tmp_path: Path) -> None:
    make_sequence(tmp_path, "rectified01", ("000", "001"))
    (tmp_path / "rectified01" / "image02" / "001.png").unlink()
    report = inspect_sequence(tmp_path / "rectified01", "image01", "image02", decode_limit=0)
    assert report.missing_target == ["001"]
    with pytest.raises(ValueError, match="missing target frames"):
        StereoPairDataset(tmp_path)


def test_duplicate_stem_is_detected(tmp_path: Path) -> None:
    make_sequence(tmp_path, "rectified01", ("000",))
    write_image(tmp_path / "rectified01" / "image01" / "000.png", (255, 0, 0))
    report = inspect_sequence(tmp_path / "rectified01", "image01", "image02", decode_limit=0)
    assert report.duplicate_input == ["000"]
    with pytest.raises(ValueError, match="duplicate input identifiers"):
        StereoPairDataset(tmp_path)
