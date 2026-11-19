#!/usr/bin/env python3
"""Inspect one or more rectified stereo sequences without loading them into RAM."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.stereo_dataset import discover_sequence_dirs, inspect_sequence


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--sequence", type=str, default=None, help="Inspect only this sequence directory name")
    parser.add_argument("--input-view", type=str, default=None)
    parser.add_argument("--target-view", type=str, default=None)
    parser.add_argument("--sequence-pattern", type=str, default=None)
    parser.add_argument("--all", action="store_true", help="Inspect all discovered sequences")
    parser.add_argument("--limit", type=int, default=5, help="Decode at most this many pairs per sequence; use --full for all")
    parser.add_argument("--full", action="store_true", help="Decode every matched image pair")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config.get("data", {})
    root_value = args.root or os.environ.get("DATASET_ROOT") or data_config.get("root")
    if not root_value:
        parser.error("Set data.root in the config, pass --root, or set DATASET_ROOT")
    root = Path(root_value).expanduser()
    input_view = args.input_view or data_config.get("input_view", "image01")
    target_view = args.target_view or data_config.get("target_view", "image02")
    pattern = args.sequence_pattern or data_config.get("sequence_pattern", "rectified*")
    sequence_dirs = discover_sequence_dirs(root, pattern)
    if args.sequence:
        sequence_dirs = [path for path in sequence_dirs if path.name == args.sequence]
    elif not args.all and len(sequence_dirs) > 1:
        sequence_dirs = sequence_dirs[:1]

    decode_limit = None if args.full else max(args.limit, 0)
    reports = [inspect_sequence(path, input_view, target_view, decode_limit) for path in sequence_dirs]
    payload = {
        "root": str(root),
        "input_view": input_view,
        "target_view": target_view,
        "sequence_pattern": pattern,
        "sequences": [report.__dict__ for report in reports],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Root: {root}")
        print(f"Views: {input_view} -> {target_view}")
        for report in reports:
            print(f"\n{report.sequence_id}: {report.matched_count} matched pairs ({report.input_count} input, {report.target_count} target)")
            print(f"  missing input: {report.missing_input or 'none'}")
            print(f"  missing target: {report.missing_target or 'none'}")
            print(f"  duplicate input: {report.duplicate_input or 'none'}")
            print(f"  duplicate target: {report.duplicate_target or 'none'}")
            print(f"  decoded dimensions: {sorted(set(report.dimensions)) or 'not decoded'}")
            print(f"  formats: {report.formats or 'not decoded'}")
            print(f"  corrupted: {report.corrupted or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
