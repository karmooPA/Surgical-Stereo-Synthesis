from __future__ import annotations

from math import ceil
from random import Random

from torch.utils.data import Dataset, Subset


def split_dataset_by_sequence(
    dataset: Dataset,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[Subset, Subset]:

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    if type(seed) is not int:
        raise ValueError("seed must be an integer.")

    records = getattr(dataset, "records", None)

    if records is None:
        raise TypeError("Dataset must expose a records attribute.")

    if not records:
        raise ValueError("Cannot split an empty dataset.")

    sequence_indices = {}

    for index, record in enumerate(records):
        sequence_id = getattr(record, "sequence_id", None)

        if not isinstance(sequence_id, str) or not sequence_id:
            raise ValueError("Every record must have a valid sequence_id.")

        sequence_indices.setdefault(sequence_id, []).append(index)

    sequence_ids = sorted(sequence_indices)

    if len(sequence_ids) < 2:
        raise ValueError(
            "Sequence-level validation requires at least two sequences. "
            "A frame-level fallback is intentionally not performed."
        )

    generator = Random(seed)
    generator.shuffle(sequence_ids)

    validation_count = max(
        1,
        min(
            len(sequence_ids) - 1,
            ceil(len(sequence_ids) * validation_fraction),
        ),
    )

    validation_sequences = set(sequence_ids[:validation_count])

    train_indices = []
    validation_indices = []

    for sequence_id, indices in sequence_indices.items():
        if sequence_id in validation_sequences:
            validation_indices.extend(indices)
        else:
            train_indices.extend(indices)

    if not train_indices or not validation_indices:
        raise RuntimeError("Invalid training or validation split.")

    return (
        Subset(dataset, sorted(train_indices)),
        Subset(dataset, sorted(validation_indices)),
    )
