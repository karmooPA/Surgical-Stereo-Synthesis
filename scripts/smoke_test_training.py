from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from PIL import Image

from src.training.config import load_training_config
from src.training.checkpoint import load_checkpoint
from src.training.optim import build_optimizer
from scripts.train import train, build_model


def write_yaml(path, data):
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


with TemporaryDirectory() as temporary_directory:
    root = Path(temporary_directory)
    dataset_root = root / "dataset"

    for sequence_number in range(1, 4):
        sequence = dataset_root / f"rectified{sequence_number:02d}"

        for view in ("image01", "image02"):
            directory = sequence / view
            directory.mkdir(parents=True)

            for frame_number in range(2):
                image = Image.new(
                    "RGB",
                    (32, 32),
                    (
                        sequence_number * 50,
                        frame_number * 70,
                        100,
                    ),
                )
                image.save(directory / f"{frame_number:06d}.jpg")

    config = load_training_config()

    model_path = root / "model.yaml"
    loss_path = root / "loss.yaml"
    train_path = root / "train.yaml"

    write_yaml(model_path, {
        "model": {
            "name": "unet",
            "in_channels": 3,
            "out_channels": 3,
            "base_channels": 4,
            "depth": 2,
            "max_channels": 16,
            "output_activation": "sigmoid",
        }
    })

    write_yaml(loss_path, {
        "perceptual_loss": {
            "name": "vgg16",
            "pretrained": False,
            "feature_blocks": 3,
            "distance": "mae",
            "reduction": "mean",
            "feature_weights": [1.0, 1.0, 1.0],
            "input": {
                "channels": 3,
                "value_range": [0.0, 1.0],
                "normalization": "imagenet",
            },
            "freeze_backbone": True,
        }
    })

    config["paths"]["model"] = str(model_path)
    config["paths"]["loss"] = str(loss_path)

    config["data"]["image_size"] = [64, 64]

    config["training"]["epochs"] = 1
    config["training"]["device"] = "cpu"
    config["training"]["mixed_precision"] = False
    config["training"]["resume_from"] = None
    config["training"]["output_dir"] = str(root / "runs")
    config["training"]["checkpoint_dir"] = str(root / "checkpoints")

    config["optimizer"]["learning_rate"] = 0.0001

    write_yaml(train_path, config)

    train(
        config_path=str(train_path),
        dataset_root=str(dataset_root),
    )

    last_checkpoint = root / "checkpoints" / "last.pt"
    best_checkpoint = root / "checkpoints" / "best.pt"

    assert last_checkpoint.is_file()
    assert best_checkpoint.is_file()

    model = build_model(model_path)
    optimizer = build_optimizer(model, config)

    state = load_checkpoint(
        last_checkpoint,
        model,
        optimizer,
    )

    assert state["epoch"] == 1

    print("First epoch checkpoint: PASSED")
    print("Best checkpoint creation: PASSED")

    config["training"]["epochs"] = 2
    config["training"]["resume_from"] = str(last_checkpoint)

    write_yaml(train_path, config)

    train(
        config_path=str(train_path),
        dataset_root=str(dataset_root),
    )

    resumed_model = build_model(model_path)
    resumed_optimizer = build_optimizer(resumed_model, config)

    resumed_state = load_checkpoint(
        last_checkpoint,
        resumed_model,
        resumed_optimizer,
    )

    assert resumed_state["epoch"] == 2

    print("Training resumption: PASSED")
    print("Second epoch checkpoint: PASSED")
    print("End-to-end training smoke test: PASSED")
