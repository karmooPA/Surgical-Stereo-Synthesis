from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from src.models import UNet
from src.training.config import load_training_config
from src.training.data import build_dataloaders
from src.training.device import resolve_device
from src.training.optim import build_optimizer
from src.training.objective import build_training_objective
from src.training.trainer import Trainer
from src.training.checkpoint import save_checkpoint, load_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config_path: str | Path) -> UNet:
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid model configuration.")

    model_config = config.get("model")

    if not isinstance(model_config, dict):
        raise ValueError("Missing model configuration.")

    model_config = dict(model_config)

    model_name = model_config.pop("name", None)

    if model_name != "unet":
        raise ValueError(f"Unsupported model: {model_name}")

    return UNet(**model_config)


def train(config_path: str, dataset_root: str | None = None) -> None:

    config = load_training_config(config_path)

    training_settings = config["training"]
    validation_settings = config["validation"]

    set_seed(training_settings["seed"])

    device_settings = resolve_device(
        requested=training_settings["device"],
        mixed_precision=training_settings["mixed_precision"],
    )

    print("Selected device:", device_settings.device)
    print("Mixed precision:", device_settings.mixed_precision)

    dataloaders = build_dataloaders(
        training_config=config,
        dataset_root=dataset_root,
    )

    model = build_model(config["paths"]["model"])
    model = model.to(device_settings.device)

    objective = build_training_objective(config)

    optimizer = build_optimizer(
        model=model,
        training_config=config,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        objective=objective,
        device_settings=device_settings,
        gradient_clip_norm=training_settings["gradient_clip_norm"],
    )

    output_dir = Path(training_settings["output_dir"])
    checkpoint_dir = Path(training_settings["checkpoint_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_metric = None

    resume_from = training_settings["resume_from"]

    if resume_from is not None:
        resume_state = load_checkpoint(
            path=resume_from,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scaler=trainer.scaler,
        )

        start_epoch = resume_state["start_epoch"]
        best_metric = resume_state["best_metric"]

        print("Resuming from epoch:", resume_state["epoch"])

    epochs = training_settings["epochs"]
    validation_interval = validation_settings["every_n_epochs"]

    monitor = validation_settings["monitor"]
    mode = validation_settings["mode"]

    print("Training samples:", len(dataloaders.train.dataset))
    print("Validation samples:", len(dataloaders.validation.dataset))
    print("Training sequences:", dataloaders.train_sequences)
    print("Validation sequences:", dataloaders.validation_sequences)

    for epoch in range(start_epoch, epochs + 1):

        train_metrics = trainer.train_epoch(dataloaders.train)

        should_validate = (
            epoch % validation_interval == 0 or epoch == epochs
        )

        validation_metrics = None

        if should_validate:
            validation_metrics = trainer.validate(
                dataloaders.validation
            )

            current_metric = validation_metrics[monitor]

            improved = (
                best_metric is None
                or (
                    current_metric < best_metric
                    if mode == "min"
                    else current_metric > best_metric
                )
            )

            if improved:
                best_metric = current_metric

                if training_settings["save_best"]:
                    save_checkpoint(
                        path=checkpoint_dir / "best.pt",
                        model=trainer.model,
                        optimizer=trainer.optimizer,
                        epoch=epoch,
                        best_metric=best_metric,
                        scaler=trainer.scaler,
                    )

        if training_settings["save_last"]:
            save_checkpoint(
                path=checkpoint_dir / "last.pt",
                model=trainer.model,
                optimizer=trainer.optimizer,
                epoch=epoch,
                best_metric=best_metric,
                scaler=trainer.scaler,
            )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Train Perceptual: {train_metrics['perceptual_loss']:.6f}"
        )

        if validation_metrics is not None:
            print(
                f"Validation Loss: {validation_metrics['loss']:.6f} | "
                f"Validation Perceptual: "
                f"{validation_metrics['perceptual_loss']:.6f} | "
                f"Best: {best_metric:.6f}"
            )

    print("Training completed.")
    print("Best validation metric:", best_metric)


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    train(
        config_path=args.config,
        dataset_root=args.dataset_root,
    )


if __name__ == "__main__":
    main()
