# Stage 1 — Reproduction Protocol Audit

**Project:** Surgical Stereo Synthesis  
**Repository:** `karmooPA/Surgical-Stereo-Synthesis`  
**Audit date:** 2026-11-19  
**Scope:** Protocol and source audit only. This document does not implement a model, loss, training script, dataset downloader, or experiment.

## 1. Paper reference

> Annika Brundyn, Jesse Swanson, Kyunghyun Cho, Doug Kondziolka, and Eric Oermann. *Stereo Video Reconstruction Without Explicit Depth Maps for Endoscopic Surgery*. arXiv:2109.08227, 2021.

- Paper: <https://arxiv.org/abs/2109.08227>
- PDF: <https://arxiv.org/pdf/2109.08227>
- The paper describes stereo-view reconstruction for minimally invasive surgery and evaluates U-Net-based models that predict a missing right view from left-view video.

## 2. Repository audit

The existing repository was inspected locally and on GitHub. The verified repository is:

- Owner: `karmooPA`
- Name: `Surgical-Stereo-Synthesis`
- URL: <https://github.com/karmooPA/Surgical-Stereo-Synthesis>
- Default branch: `main`
- Visibility: Private

The current repository contains the initial project documentation and these top-level directories:

```text
configs/
data/raw/
data/processed/
data/splits/
src/models/
src/datasets/
src/training/
src/evaluation/
src/utils/
scripts/
experiments/
checkpoints/
results/figures/
results/metrics/
tests/
```

No Stage 1 implementation files were present before this audit.

## 3. Official implementation investigation

The paper states that the authors publicly released their code, but the arXiv paper, its accessible supplementary/appendix material, and the author-linked pages inspected during this audit did not expose a repository URL that can be verified as belonging to the authors or to this paper.

**Finding:** No official source-code repository for *Stereo Video Reconstruction Without Explicit Depth Maps for Endoscopic Surgery* was verified.

The repository [UZ-SLAMLab/Endo-Depth-and-Motion](https://github.com/UZ-SLAMLab/Endo-Depth-and-Motion) is author-linked to a different paper by Recasens et al. It provides a Hamlyn-related dataset mirror and code for depth, tracking, and volumetric reconstruction. It is not identified here as the implementation of the Brundyn et al. stereo-reconstruction paper and will not be substituted for it.

## 4. Dataset source audit

### 4.1 Dataset identified by the paper

The paper identifies the source as the **da Vinci endoscopic dataset from the Hamlyn Centre for Robotic Surgery**, citing Ye et al. (2017), *Self-Supervised Siamese Learning on Stereo Image Pairs for Depth Estimation in Robotic Surgery*.

- Hamlyn Centre dataset portal named by the paper: <http://hamlyn.doc.ic.ac.uk/vision/>
- Cited dataset paper: <https://arxiv.org/abs/1705.08260>
- Ye et al. Hamlyn Symposium paper: <https://www.hamlynsymposium.org/wp-content/uploads/2022/01/Proceedings_HSMR17-1.pdf>

The paper reports rectified stereo images at **384 × 192 pixels**, captured in vivo during a partial nephrectomy using a da Vinci surgical system.

### 4.2 Official download source

The official source named by the reproduction paper is the Hamlyn Centre portal. The portal was not downloadable during this audit because the site request timed out. No dataset files were downloaded.

A later, secondary Hamlyn rectified-data release is documented by the Endo-Depth-and-Motion project:

- Dataset documentation: <https://davidrecasens.github.io/EndoDepthAndMotion/>
- Code repository: <https://github.com/UZ-SLAMLab/Endo-Depth-and-Motion>
- Dataset mirror documented there: a SharePoint download containing rectified stereo images, calibration, depth ground truth, and splits.

This secondary release is recorded as a possible access route for later investigation. It is not asserted to be byte-identical to the release used by Brundyn et al.

### 4.3 Directory and filename structure

The Brundyn et al. paper does not publish a directory tree, filename convention, manifest format, or a camera-specific filename rule. It only states that the data are rectified stereo image sequences and shows left and right views in Figure 6.

Therefore, the following details remain unverified until the official archive or an accompanying manifest can be inspected:

- Root directory names.
- Whether left and right views are separate directories or paired within a common sequence directory.
- Exact filename pattern and extension.
- Whether frame identifiers are timestamps, sequential indices, or another key.
- Whether calibration and split files are included in the original release.

The secondary Endo-Depth-and-Motion documentation confirms that its rectified color images are RGB `.jpg` files and its depth maps are 16-bit `.png` files, but that documentation describes its own release and must not be assumed to describe the original Brundyn et al. archive.

### 4.4 Left/right matching and sample counts

The paper defines the left image as the observed input and the corresponding right image as the target. The pair is associated by the same video instant, and the images are rectified. The paper does not disclose a filename-level matching rule.

The reported dataset counts are **34,241 training image sequences** and **7,192 test image sequences**. They are sequence/sample counts as reported by the paper, not a verified count of files in a locally inspected archive. The reported samples cannot currently be identified by filename because the paper does not publish the relevant manifest or filenames.

## 5. Selected baseline

The selected Stage 2 baseline is:

> **Single-frame Modified U-Net + VGG16 perceptual loss**

### Paper-reported task definition

- Input: one RGB left-view image.
- Target: the corresponding RGB right-view image.
- Input sequence length: `K = 1`.
- Image resolution: `384 × 192` pixels, as reported by the paper.
- No explicit depth map is used as a model input.
- Camera intrinsic parameters are not used as model inputs.

### Architecture specifications

The paper reports a modified fully convolutional U-Net with these changes:

- Decoder blocks use padding rather than cropping when concatenating skip connections.
- Decoder upsampling can use strided transposed convolution or bilinear interpolation. The selected baseline uses transposed convolution.
- Batch-normalization layers are omitted.
- The paper compares five- and six-layer U-Nets.

The project brief selects **five double-convolution layers** for this baseline. The paper text confirms the five-layer variant but does not spell out the phrase “five double-convolution layers” or publish every channel width. That layer interpretation is therefore marked as a project protocol item pending source-code or supplementary confirmation.

### Perceptual loss

The paper defines the perceptual loss as the sum of mean absolute errors computed on feature maps from the first three VGG16 blocks. VGG16 is pretrained on ImageNet.

For an input image `x`, target right view `y`, prediction `ŷ`, and feature maps `φ_b` from the first three VGG16 blocks, the protocol is:

```text
L_perceptual(ŷ, y) = Σ_{b=1..3} MAE(φ_b(ŷ), φ_b(y))
```

The paper does not specify the exact layer endpoint represented by “block,” nor the reduction and weighting details beyond the sum of MAEs.

### Training hyperparameters

The following values are reported by the paper:

| Item | Paper-reported value |
| --- | --- |
| Optimizer | Adam |
| Learning rate | `0.0001`, constant |
| Batch size | `16` normally, with smaller batches when GPU memory requires |
| Precision | 16-bit precision wherever possible |
| U-Net initialization | Trained from scratch |
| Reported hardware context | One NVIDIA V100 GPU |
| Reported training duration | No more than 30 hours per model |

The hardware and duration are context from the paper, not requirements for this project.

## 6. Data splitting procedure

The paper's Appendix A reports the following procedure:

1. Split the full training image sequence into consecutive groups of 1,000 images.
2. Randomly allocate 80% of those groups to the training split.
3. Allocate the remaining 20% of groups to validation.
4. Apply a sliding window of length `K` within each group.
5. Shuffle individual samples again before training.

For the selected single-frame baseline, the protocol uses `K = 1`. The official paper does not report the random seed used for the 80/20 allocation.

The paper also states that the original video frame rate is undisclosed and that 20 frames per second produced realistic playback for evaluation. This does not define a training-time resampling rule.

## 7. Evaluation metrics

The paper reports these automatic image-quality metrics:

- PSNR
- SSIM
- FID
- LPIPS
- DISTS

The paper also evaluates generated frames in a frame-level two-alternative forced-choice reader study with ten experienced surgeons and evaluates selected reconstructed videos in a VR-based reader study. Those human studies are outside the Stage 2 implementation scope unless separately approved.

## 8. Details not fully specified in the paper

These items must not be silently invented in Stage 2:

- Image normalization and numeric input range.
- Data augmentation or transformation policy.
- Exact channel widths at every U-Net level.
- Exact interpretation of “five-layer” versus five double-convolution blocks.
- Output activation function and output range.
- VGG16 input preprocessing, including channel order, scaling, and ImageNet mean/std usage.
- Exact VGG16 feature extraction endpoints for the first three blocks.
- Whether VGG16 features are normalized before MAE computation.
- Reduction details for each feature-map MAE and any relative weighting.
- Random seed and random-number-generator implementation for the group split.
- Exact number of epochs or stopping criterion.
- Checkpoint selection strategy.
- Whether any training images are discarded at group boundaries for a given `K`.
- Exact directory and filename mapping for the official Hamlyn release.
- Exact manifest format needed to reproduce the reported 34,241 and 7,192 sample counts.

## 9. Engineering decisions requiring approval

No implementation decision is silently adopted by this document. Before Stage 2, approval is required for at least:

1. The exact official Hamlyn archive or a documented secondary release to use.
2. The verified left/right filename matching rule and a manifest format.
3. Image scaling and normalization.
4. Whether augmentation is disabled to match the paper or introduced as a separate experiment.
5. The exact five-block channel configuration.
6. The output activation and VGG16 preprocessing contract.
7. The VGG16 feature endpoints, MAE reductions, and weighting.
8. The random seed and checkpoint-selection rule.

## 10. Stage 1 restrictions and status

- No U-Net was implemented.
- No perceptual-loss code was implemented.
- No training script was created.
- No dataset was downloaded or uploaded.
- No experiment was run.
- No experimental result was generated.

**Stage 1 status:** protocol audit documented. Awaiting approval before Stage 2.
