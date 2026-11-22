from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from torch.utils.data import DataLoader

from src.datasets import StereoPairDataset, create_stereo_dataloader
from .split import split_dataset_by_sequence


@dataclass(frozen=True)
class StereoDataLoaders:
    train: DataLoader
    validation: DataLoader
    dataset: StereoPairDataset
    train_sequences: tuple[str, ...]
    validation_sequences: tuple[str, ...]


def build_dataloaders(
    training_config: dict[str, Any],
    dataset_root: str | Path | None = None,
) -> StereoDataLoaders:

    dataset_config_path = Path(
        training_config["paths"]["dataset"]
    ).expanduser().resolve()

    if not dataset_config_path.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {dataset_config_path}"
        )

    with dataset_config_path.open("r", encoding="utf-8") as file:
        dataset_config = yaml.safe_load(file)

    if not isinstance(dataset_config, dict):
        raise ValueError("Invalid dataset configuration.")

    data_settings = dataset_config.get("data")
    loader_settings = dataset_config.get("loader")

    if not isinstance(data_settings, dict):
        raise ValueError("Missing dataset data configuration.")

    if not isinstance(loader_settings, dict):
        raise ValueError("Missing dataset loader configuration.")

    configured_root = (
        dataset_root
        or os.environ.get("DATASET_ROOT")
        or data_settings.get("root")
    )

    if configured_root is None:
        raise ValueError(
            "Dataset root is not configured. Set DATASET_ROOT "
            "or pass dataset_root explicitly."
        )

    root = Path(configured_root).expanduser()

    if not root.is_absolute():
        root = dataset_config_path.parent.parent / root

    root = root.resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    image_size = training_config["data"]["image_size"]

    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(type(value) is not int or value < 1 for value in image_size)
    ):
        raise ValueError("Invalid training image_size.")

    batch_size = loader_settings.get("batch_size", 16)
    num_workers = loader_settings.get("num_workers", 0)
    shuffle = loader_settings.get("shuffle", True)
    pin_memory = loader_settings.get("pin_memory", False)
    drop_last = loader_settings.get("drop_last", False)

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer.")

    if type(shuffle) is not bool:
        raise ValueError("shuffle must be boolean.")

    if type(pin_memory) is not bool:
        raise ValueError("pin_memory must be boolean.")

    if type(drop_last) is not bool:
        raise ValueError("drop_last must be boolean.")

    dataset = StereoPairDataset(
        root=root,
        input_view=data_settings.get("input_view", "image01"),
        target_view=data_settings.get("target_view", "image02"),
        sequence_pattern=data_settings.get(
            "sequence_pattern", "rectified*"
        ),
        image_size=image_size,
        strict=True,
    )

    train_subset, validation_subset = split_dataset_by_sequence(
        dataset,
        validation_fraction=training_config["data"][
            "validation_fraction"
        ],
        seed=training_config["data"]["split_seed"],
    )

    if drop_last and len(train_subset) < batch_size:
        raise ValueError(
            "Training subset is smaller than batch_size while drop_last "
            "is enabled."
        )

    train_sequences = tuple(sorted({
        dataset.records[index].sequence_id
        for index in train_subset.indices
    }))

    validation_sequences = tuple(sorted({
        dataset.records[index].sequence_id
        for index in validation_subset.indices
    }))

    train_loader = create_stereo_dataloader(
        dataset=train_subset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    validation_loader = create_stereo_dataloader(
        dataset=validation_subset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return StereoDataLoaders(
        train=train_loader,
        validation=validation_loader,
        dataset=dataset,
        train_sequences=train_sequences,
        validation_sequences=validation_sequences,
    )
