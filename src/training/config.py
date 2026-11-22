from pathlib import Path
from typing import Any

import yaml


def load_training_config(
    config_path: str | Path = "configs/train.yaml",
) -> dict[str, Any]:

    config_path = Path(config_path).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Training configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid training configuration.")

    required_sections = {
        "paths",
        "data",
        "training",
        "optimizer",
        "validation",
    }

    missing = required_sections - config.keys()

    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    root = config_path.parent.parent

    for name in ("dataset", "model", "loss"):
        value = config["paths"].get(name)

        if not isinstance(value, str) or not value:
            raise ValueError(f"Invalid paths.{name}.")

        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path

        path = path.resolve()

        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        config["paths"][name] = str(path)

    data = config["data"]
    image_size = data.get("image_size")

    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(type(value) is not int or value < 1 for value in image_size)
    ):
        raise ValueError("image_size must contain two positive integers.")

    fraction = data.get("validation_fraction")

    if not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    if data.get("split_by") != "sequence":
        raise ValueError("Only sequence-level splitting is supported.")

    training = config["training"]

    if type(training.get("epochs")) is not int or training["epochs"] < 1:
        raise ValueError("epochs must be a positive integer.")

    if training.get("device") not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("Unsupported training device.")

    if type(training.get("mixed_precision")) is not bool:
        raise ValueError("mixed_precision must be boolean.")

    if type(training.get("log_interval_steps")) is not int or training["log_interval_steps"] < 1:
        raise ValueError("log_interval_steps must be positive.")

    for name in ("output_dir", "checkpoint_dir"):
        value = training.get(name)

        if not isinstance(value, str) or not value:
            raise ValueError(f"Invalid training.{name}.")

        path = Path(value).expanduser()

        if not path.is_absolute():
            path = root / path

        training[name] = str(path.resolve())

    resume_from = training.get("resume_from")

    if resume_from is not None:
        path = Path(resume_from).expanduser()

        if not path.is_absolute():
            path = root / path

        training["resume_from"] = str(path.resolve())

    clip_norm = training.get("gradient_clip_norm")

    if clip_norm is not None and (
        not isinstance(clip_norm, (int, float)) or clip_norm <= 0
    ):
        raise ValueError("gradient_clip_norm must be positive or null.")

    optimizer = config["optimizer"]

    if optimizer.get("name") != "adam":
        raise ValueError("Only Adam optimizer is supported.")

    if not isinstance(optimizer.get("learning_rate"), (int, float)) or optimizer["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive.")

    betas = optimizer.get("betas")

    if (
        not isinstance(betas, list)
        or len(betas) != 2
        or any(not isinstance(beta, (int, float)) or not 0 <= beta < 1 for beta in betas)
    ):
        raise ValueError("Adam betas must contain two values in [0, 1).")

    if not isinstance(optimizer.get("weight_decay"), (int, float)) or optimizer["weight_decay"] < 0:
        raise ValueError("weight_decay must be non-negative.")

    validation = config["validation"]

    if type(validation.get("every_n_epochs")) is not int or validation["every_n_epochs"] < 1:
        raise ValueError("every_n_epochs must be positive.")

    if validation.get("monitor") != "perceptual_loss":
        raise ValueError("Only perceptual_loss monitoring is supported.")

    if validation.get("mode") not in {"min", "max"}:
        raise ValueError("Validation mode must be min or max.")

    return config
