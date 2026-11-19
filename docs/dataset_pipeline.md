# Stage 2 — Production-Ready Stereo Dataset Pipeline

This stage provides a reusable paired-image pipeline for the lightweight local sample and the future complete Hamlyn collection. The same Python code discovers every configured `rectified*` sequence and uses only configuration values and paths to change environments.

## Dataset directory structure

The inspected sample archive, `rectified05 2.zip`, contains this structure after ignoring macOS metadata:

```text
rectified05/
├── image01/
│   ├── 0000000000.jpg
│   └── ...
├── image02/
│   ├── 0000000000.jpg
│   └── ...
├── depth01/
└── depth02/
```

The sample has 11 files in `image01` and 11 files in `image02`. All 11 filename stems match. The RGB images decode as JPEG with dimensions `360 × 288` pixels. Depth directories are intentionally ignored by the baseline dataset.

The paper reports a `384 × 192` dataset resolution. The inspected lightweight archive has a different native resolution, so resizing is left configurable and is not silently applied by default.

For the future collection, the loader expects a root containing any number of sequence directories matching the configurable pattern, for example:

```text
HamlynRectifiedDataset/
├── rectified01/
│   ├── image01/
│   └── image02/
├── rectified04/
│   ├── image01/
│   └── image02/
└── rectified05/
    ├── image01/
    └── image02/
```

The implementation does not assume which sequence IDs exist or how many frames each sequence contains. Secondary dataset documentation shows that Hamlyn mirrors can contain multiple `rectifiedNN` sequences with different frame counts and native resolutions. This is treated as a discovery result for pipeline design, not as proof of the exact original release used by the reproduction paper.

## Camera mapping

The default configuration maps `image01` to the input and `image02` to the target because those are the two RGB directories in the inspected sample. The code does not claim that `image01` is physically the left camera. Set `data.input_view` and `data.target_view` explicitly after the camera orientation is verified from the dataset documentation or an approved calibration check.

No depth directory is loaded by `StereoPairDataset`.

## Pair matching and validation

For every sequence independently:

1. Discover files only in the configured input and target directories.
2. Ignore hidden files, `.DS_Store`, `__MACOSX` content, and unsupported extensions.
3. Accept `.jpg`, `.jpeg`, and `.png` RGB files.
4. Use each filename stem as the frame identifier.
5. Match only equal stems inside the same sequence.
6. Report missing stems and duplicate stems.
7. In strict mode, fail initialization when a sequence has missing or duplicate identifiers.

This prevents a frame from one sequence ever being paired with a frame from another sequence. The dataset stores `sequence_id` and `frame_id` in each returned sample.

## Python API

```python
from src.datasets import StereoPairDataset, create_stereo_dataloader

dataset = StereoPairDataset(
    root="/path/to/HamlynRectifiedDataset",
    input_view="image01",
    target_view="image02",
    sequence_pattern="rectified*",
    image_size=None,  # preserve native (height, width)
)

loader = create_stereo_dataloader(
    dataset,
    batch_size=16,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
)
```

Each sample contains:

- `input`: RGB `torch.float32` tensor in `[0, 1]`, shape `(3, H, W)`.
- `target`: RGB `torch.float32` tensor in `[0, 1]`, shape `(3, H, W)`.
- `sequence_id`: sequence directory name.
- `frame_id`: filename stem.
- `original_size`: `(height, width)` before optional resizing.

Images are opened on demand in `__getitem__`. The complete dataset is not kept in RAM. No device transfer occurs in the Dataset or DataLoader.

`image_size` is either `null` or `[height, width]`. Resizing uses bilinear interpolation for both RGB views and preserves the original dimensions in metadata.

## Configuration

The baseline configuration is [configs/dataset.yaml](../configs/dataset.yaml):

```yaml
data:
  root: null
  input_view: image01
  target_view: image02
  sequence_pattern: "rectified*"
  image_size: null

loader:
  batch_size: 16
  num_workers: 0
  shuffle: true
  pin_memory: false
  drop_last: false
```

`root` is deliberately unset in Git. Use a machine-local path or the `DATASET_ROOT` environment variable. Do not put dataset archives or extracted images in the repository.

## Local Mac smoke test

The local smoke test uses only the 11-pair sample:

```bash
export DATASET_ROOT=/path/to/extracted/rectified05-parent
python scripts/inspect_dataset.py --root "$DATASET_ROOT" --all --full
PYTHONPATH=. python - <<'PY'
import torch
from src.datasets import StereoPairDataset, create_stereo_dataloader

dataset = StereoPairDataset("/path/to/extracted/rectified05-parent")
loader = create_stereo_dataloader(dataset, batch_size=2, num_workers=0, shuffle=False)
batch = next(iter(loader))
assert len(dataset) == 11
assert batch["input"].shape[0] == 2
assert batch["input"].min() >= 0 and batch["input"].max() <= 1
assert torch.isfinite(batch["input"]).all()
PY
```

Use `--limit 5` for a faster inspection. The default inspector decodes at most five matched pairs per sequence. `--full` decodes every matched pair and checks for corrupted images.

## Full dataset usage on the Linux GPU server

1. Clone this repository.
2. Install `requirements.txt` in the server environment.
3. Place or mount the complete dataset outside the Git checkout.
4. Set `DATASET_ROOT` or pass `--root` to the inspection command.
5. Run a limited inspection first.
6. Run full verification with `--all --full` before training.
7. Construct the same `StereoPairDataset` and `create_stereo_dataloader` objects with server-specific `batch_size`, `num_workers`, and `pin_memory` values.

Example:

```bash
export DATASET_ROOT=/mnt/datasets/HamlynRectifiedDataset
python scripts/inspect_dataset.py --root "$DATASET_ROOT" --all --limit 5
python scripts/inspect_dataset.py --root "$DATASET_ROOT" --all --full --json > /tmp/hamlyn_dataset_report.json
```

For local macOS testing use `num_workers=0`. On the Linux GPU server, increase `num_workers` only after the filesystem and CPU budget have been checked. Tensors remain on CPU until a future training loop explicitly transfers them to the selected device.

## Inspection tool outputs

`scripts/inspect_dataset.py` reports, per sequence:

- Input and target image counts.
- Matched pair count.
- Missing input and target frame identifiers.
- Duplicate identifiers.
- Decoded image dimensions and formats.
- Corrupted files.
- A machine-readable JSON mode.

The tool does not decode every image during normal Dataset construction. Full decoding is opt-in with `--full`.

## Automated tests

`tests/test_stereo_dataset.py` creates temporary synthetic sequence trees and checks:

- Multi-sequence discovery.
- Sequence-local matching.
- RGB conversion and `[0, 1]` tensor values.
- DataLoader batches.
- Configurable resizing.
- Missing-frame detection.
- Duplicate-stem detection.

Run:

```bash
PYTHONPATH=. pytest -q tests/test_stereo_dataset.py
```

## Known limitations and unresolved questions

- The physical camera orientation of `image01` and `image02` is not assumed by the code and requires confirmation before training.
- The paper's `384 × 192` resolution differs from the inspected sample's `360 × 288` native RGB dimensions. The decision to resize to the paper resolution belongs in a later approved configuration.
- The exact directory and filename convention of the complete original Hamlyn release has not been verified from a downloaded full archive.
- The reported paper sample counts cannot be independently confirmed until the exact release and manifests are available.
- This stage does not load depth, implement U-Net, implement perceptual loss, create a training loop, or run training.
