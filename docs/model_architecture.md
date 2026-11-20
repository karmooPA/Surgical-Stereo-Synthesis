# Stage 3 — U-Net Baseline Architecture

## Scope

This stage implements a single-frame, RGB-to-RGB U-Net for stereo-view
reconstruction. It does not implement VGG16, perceptual loss, an optimizer, a
training loop, or a training experiment.

Reference: Brundyn et al., *Stereo Video Reconstruction Without Explicit Depth
Maps for Endoscopic Surgery* ([arXiv:2109.08227](https://arxiv.org/abs/2109.08227)).

## Paper-reported design facts

The paper formulates the single-frame case as `K = 1`: the current input view is
used to predict the corresponding target view. The network is described as a
modified fully convolutional U-Net with encoder/decoder paths and skip
connections. The reported modifications include padding rather than cropping
when aligning decoder and skip features, and either transposed-convolution or
bilinear upsampling. The paper evaluates five- and six-layer U-Net variants;
its experimental description also refers to five U-Net blocks in the default
configuration.

The paper does not publish a complete layer-by-layer channel table or enough
source-level detail to establish an exact one-to-one mapping between its
"five-layer/five-block" terminology and this implementation's `DoubleConv`
modules. Consequently, this code is a reproducible U-Net engineering
interpretation, not a claim of exact source-code reproduction.

The paper reports rectified RGB images at 384 x 192 pixels. It also evaluates
both sigmoid and non-sigmoid output variants. The precise output activation for
every single-frame configuration is not fully specified in the information
used for this implementation.

## Implemented architecture

`src/models/unet.py` provides `UNet` and `DoubleConv`.

With the default `depth=5`, the implementation has:

- Five encoder `DoubleConv` stages with skip features at progressively lower
  resolutions.
- Five 2x2 max-pooling operations (one after each encoder stage).
- One bottleneck `DoubleConv` stage.
- Five decoder upsampling stages using 2x2, stride-2 transposed convolutions.
- Five decoder `DoubleConv` stages, each concatenating the matching encoder
  skip feature.
- A final 1x1 convolution mapping features to the requested output channels.

Each `DoubleConv` consists of two 3x3 convolutions with padding 1, each
followed by ReLU. No batch normalization is used.

The decoder uses symmetric padding to align upsampled features with skip
features. It does not crop feature maps. With floor-mode max pooling and
stride-2 transposed convolutions, an odd-sized skip feature can be one pixel
larger than its upsampled decoder counterpart; the missing pixel is padded.

## Configurable parameters

The default configuration in `configs/model.yaml` is:

| Parameter | Default | Meaning |
|---|---:|---|
| `in_channels` | 3 | RGB input channels |
| `out_channels` | 3 | RGB output channels |
| `base_channels` | 32 | First encoder stage width |
| `depth` | 5 | Number of encoder skip, pooling, and decoder stages |
| `max_channels` | 512 | Upper bound for stage and bottleneck channels |
| `output_activation` | `sigmoid` | Constrains output to [0, 1] |

The resulting encoder widths are 32, 64, 128, 256, and 512. Because the
configured maximum is 512, the bottleneck width is also capped at 512. With
these defaults, the model has **20,607,779 trainable parameters**. These widths
and the parameter count are implementation-specific; they are not claimed to
be the paper's unpublished channel configuration.

`depth` is the number of downsampling steps. Therefore, both input dimensions
must be at least `2**depth` pixels. For `depth=5`, the minimum height and width
are 32 pixels. The model supports dimensions that are not divisible by 32 and
returns exactly the input spatial dimensions.

## Tensor contract

```text
input:  [B, 3, H, W] floating-point tensor
        current dataset convention: values in [0, 1]
output: [B, 3, H, W] floating-point tensor
        default sigmoid convention: values in [0, 1]
```

The model does not select or move to a device. The caller moves the model and
input tensors to CPU or CUDA.

## Local verification

Run the model tests from the repository root:

```bash
python -m pytest -q tests/test_unet.py
```

The tests cover symmetric stage counts, RGB shapes, the paper and sample image
resolutions, odd dimensions, invalid input shapes, minimum spatial size,
gradients, activation choices, absence of batch normalization, state-dict
save/reload, and compatibility with the existing Stage 2 dataset loader.

The full project test suite should also be run before committing:

```bash
python -m pytest -q
```

## GPU workstation usage

The same implementation can be used on a Linux workstation with CUDA-enabled
PyTorch. The caller is responsible for choosing the device:

```python
import torch
from src.models import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
images = torch.rand(2, 3, 192, 384, device=device)
predictions = model(images)
```

No CUDA-specific operations are embedded in the model.

## Limitations and decisions

- Physical left/right mapping for `image01` and `image02` remains unverified;
  the dataset configuration must continue to control input and target views.
- The model architecture is single-frame (`K=1`); it does not use temporal
  context.
- Exact paper channel widths and source-level block definitions are not
  available, so this implementation documents its own explicit choices.
- Transposed convolution is the selected initial upsampling option. The paper
  also investigates bilinear upsampling; a comparison can be added as a
  separately scoped experiment if required.
- Sigmoid is selected to match the current dataset's [0, 1] tensor convention;
  it remains configurable because the paper evaluates sigmoid and non-sigmoid
  variants.
- No claim is made that an untrained model produces meaningful stereo
  reconstructions. Stage 3 verifies architecture execution, not reconstruction
  quality.
