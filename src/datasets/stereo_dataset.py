"""Configurable paired stereo-image dataset utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import fnmatch
import os

from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IGNORED_NAMES = {".ds_store"}


@dataclass(frozen=True)
class PairRecord:
    sequence_id: str
    frame_id: str
    input_path: Path
    target_path: Path


@dataclass
class SequenceInspection:
    sequence_id: str
    input_count: int
    target_count: int
    matched_count: int
    missing_input: List[str]
    missing_target: List[str]
    duplicate_input: List[str]
    duplicate_target: List[str]
    dimensions: List[Tuple[int, int]]
    formats: List[str]
    corrupted: List[str]

    @property
    def has_pairing_issues(self) -> bool:
        return bool(
            self.missing_input
            or self.missing_target
            or self.duplicate_input
            or self.duplicate_target
        )


def _is_usable_image(path: Path) -> bool:
    return (
        path.is_file()
        and not path.name.startswith(".")
        and path.name.lower() not in IGNORED_NAMES
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _index_view(directory: Path) -> Tuple[Dict[str, Path], List[str]]:
    """Index one view and return its unique files and duplicate stems."""
    by_stem: Dict[str, List[Path]] = {}
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            if _is_usable_image(path):
                by_stem.setdefault(path.stem, []).append(path)
    duplicates = sorted(stem for stem, paths in by_stem.items() if len(paths) > 1)
    unique = {stem: paths[0] for stem, paths in by_stem.items() if len(paths) == 1}
    return unique, duplicates


def discover_sequence_dirs(root: Path | str, sequence_pattern: str = "rectified*") -> List[Path]:
    """Discover sequence directories below *root*, without hardcoding a count."""
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {root}")

    candidates = [root] if fnmatch.fnmatch(root.name, sequence_pattern) else []
    candidates.extend(
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and not path.name.startswith(".") and fnmatch.fnmatch(path.name, sequence_pattern)
    )
    unique = {path.resolve(): path for path in candidates}
    return [unique[key] for key in sorted(unique)]


def inspect_sequence(
    sequence_dir: Path | str,
    input_view: str,
    target_view: str,
    decode_limit: Optional[int] = 5,
) -> SequenceInspection:
    """Inspect pairing and optionally decode up to *decode_limit* matched pairs."""
    sequence_dir = Path(sequence_dir)
    input_files, duplicate_input = _index_view(sequence_dir / input_view)
    target_files, duplicate_target = _index_view(sequence_dir / target_view)
    input_ids, target_ids = set(input_files), set(target_files)
    matched_ids = sorted(input_ids & target_ids)

    dimensions: List[Tuple[int, int]] = []
    formats: List[str] = []
    corrupted: List[str] = []
    ids_to_decode = matched_ids if decode_limit is None else matched_ids[:decode_limit]
    for frame_id in ids_to_decode:
        for path in (input_files[frame_id], target_files[frame_id]):
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    dimensions.append(image.size)
                    formats.append((image.format or path.suffix.lstrip(".")).upper())
            except (OSError, UnidentifiedImageError):
                corrupted.append(str(path))

    return SequenceInspection(
        sequence_id=sequence_dir.name,
        input_count=len(input_files) + len(duplicate_input),
        target_count=len(target_files) + len(duplicate_target),
        matched_count=len(matched_ids),
        missing_input=sorted(target_ids - input_ids),
        missing_target=sorted(input_ids - target_ids),
        duplicate_input=duplicate_input,
        duplicate_target=duplicate_target,
        dimensions=dimensions,
        formats=sorted(set(formats)),
        corrupted=corrupted,
    )


def build_pair_records(
    root: Path | str,
    input_view: str,
    target_view: str,
    sequence_pattern: str = "rectified*",
    strict: bool = True,
) -> Tuple[List[PairRecord], List[SequenceInspection]]:
    """Build sequence-local pair records and return inspection reports."""
    records: List[PairRecord] = []
    reports: List[SequenceInspection] = []
    sequence_dirs = discover_sequence_dirs(root, sequence_pattern)
    if not sequence_dirs:
        raise FileNotFoundError(f"No sequence directories matching {sequence_pattern!r} found under {root}")

    for sequence_dir in sequence_dirs:
        input_files, duplicate_input = _index_view(sequence_dir / input_view)
        target_files, duplicate_target = _index_view(sequence_dir / target_view)
        input_ids, target_ids = set(input_files), set(target_files)
        matched_ids = sorted(input_ids & target_ids)
        report = SequenceInspection(
            sequence_id=sequence_dir.name,
            input_count=len(input_files) + len(duplicate_input),
            target_count=len(target_files) + len(duplicate_target),
            matched_count=len(matched_ids),
            missing_input=sorted(target_ids - input_ids),
            missing_target=sorted(input_ids - target_ids),
            duplicate_input=duplicate_input,
            duplicate_target=duplicate_target,
            dimensions=[],
            formats=[],
            corrupted=[],
        )
        reports.append(report)
        if strict and report.has_pairing_issues:
            raise ValueError(_format_pairing_error(report))
        records.extend(
            PairRecord(sequence_dir.name, frame_id, input_files[frame_id], target_files[frame_id])
            for frame_id in matched_ids
        )
    return records, reports


def _format_pairing_error(report: SequenceInspection) -> str:
    issues = []
    if report.missing_input:
        issues.append(f"missing input frames: {report.missing_input}")
    if report.missing_target:
        issues.append(f"missing target frames: {report.missing_target}")
    if report.duplicate_input:
        issues.append(f"duplicate input identifiers: {report.duplicate_input}")
    if report.duplicate_target:
        issues.append(f"duplicate target identifiers: {report.duplicate_target}")
    return f"Pairing issues in sequence {report.sequence_id}: " + "; ".join(issues)


class StereoPairDataset(Dataset):
    """On-demand paired RGB stereo images from one or more sequences."""

    def __init__(
        self,
        root: Path | str,
        input_view: str = "image01",
        target_view: str = "image02",
        sequence_pattern: str = "rectified*",
        image_size: Optional[Sequence[int]] = None,
        strict: bool = True,
    ) -> None:
        self.root = Path(root).expanduser()
        self.input_view = input_view
        self.target_view = target_view
        self.sequence_pattern = sequence_pattern
        self.image_size = tuple(image_size) if image_size is not None else None
        if self.image_size is not None and len(self.image_size) != 2:
            raise ValueError("image_size must be [height, width] or null")
        self.records, self.inspections = build_pair_records(
            self.root, input_view, target_view, sequence_pattern, strict=strict
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        input_image = Image.open(record.input_path).convert("RGB")
        target_image = Image.open(record.target_path).convert("RGB")
        original_size = (input_image.height, input_image.width)
        if self.image_size is not None:
            height, width = self.image_size
            input_image = input_image.resize((width, height), Image.Resampling.BILINEAR)
            target_image = target_image.resize((width, height), Image.Resampling.BILINEAR)
        input_tensor = _to_tensor(input_image)
        target_tensor = _to_tensor(target_image)
        return {
            "input": input_tensor,
            "target": target_tensor,
            "sequence_id": record.sequence_id,
            "frame_id": record.frame_id,
            "original_size": original_size,
        }


def _to_tensor(image: Image.Image) -> torch.Tensor:
    # Avoid torchvision as a mandatory dependency for this data-only stage.
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    data = data.reshape(image.height, image.width, 3).permute(2, 0, 1)
    return data.to(dtype=torch.float32).div(255.0).contiguous()


def create_stereo_dataloader(
    dataset: StereoPairDataset,
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle: bool = True,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Create a DataLoader without moving data to CPU/GPU devices."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
