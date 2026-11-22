cat > docs/perceptual_loss.md <<'EOF'
# VGG16 Perceptual Loss

## Overview

The perceptual loss compares the generated right-view image with the corresponding ground-truth right-view image using feature representations extracted from VGG16.

## Architecture

- Backbone: VGG16
- Pretraining: ImageNet
- Feature extraction: First three convolutional blocks
- Feature endpoints: conv1_2, conv2_2, conv3_3
- Distance metric: Mean Absolute Error
- Reduction: Mean
- Feature weights: Equal
- Backbone parameters: Frozen

## Mathematical Definition

The perceptual loss is defined as:

\[
L_{perc} =
\sum_{i=1}^{3}
\operatorname{MAE}
\left(
\phi_i(\hat{Y}),
\phi_i(Y)
\right)
\]

Where:

- Y is the ground-truth right-view image.
- Y_hat is the generated right-view image.
- phi_i represents the feature extraction function of VGG16 block i.

## Input Requirements

Both inputs must satisfy:

- Shape: [B, 3, H, W]
- Color format: RGB
- Data type: Floating point
- Value range: [0, 1]
- Matching dimensions
- Same device

## Preprocessing

ImageNet normalization is applied internally using:

- Mean: [0.485, 0.456, 0.406]
- Standard deviation: [0.229, 0.224, 0.225]

## Gradient Behavior

VGG16 parameters are frozen.

Gradients propagate through the predicted image and back to the generator.

Target feature extraction is performed without gradient tracking.

## Implementation Decisions

The exact feature endpoints, ImageNet normalization, and equal feature weighting are implementation choices where the paper does not provide complete implementation details.

## Testing

The implementation has passed:

- Scalar loss validation
- Finite loss validation
- Identical-image validation
- Positive loss validation
- Prediction gradient propagation
- Frozen VGG16 parameter validation
- Batch processing
- Variable spatial dimensions
- Invalid input validation

All current unit tests use pretrained=False. Verification with downloaded ImageNet-pretrained weights remains pending.

## Current Status

The loss implementation, configuration factory, and initial CPU tests are complete.

Configuration loading has been verified.

ImageNet-pretrained VGG16 (IMAGENET1K_V1) was successfully downloaded and tested.

Verification results:
- Three feature blocks successfully loaded.
- Input resolution: 192 x 384.
- Finite perceptual loss.
- Successful gradient propagation.
- VGG16 parameters remained frozen.

The implementation and local verification are complete.
EOF