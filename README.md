# Surgical Stereo Synthesis

Generating stereo endoscopic views using deep learning, from U-Net baselines to diffusion models.

Given a single left-camera frame from a robotic surgery recording, can a model predict what the right camera saw at the same instant? The recorded right frame provides a reference for evaluating both the appearance of tissue and instruments and their displacement between views.

## Baseline reproduction

The first milestone is to reproduce the single-frame U-Net with perceptual loss studied by Annika Brundyn, Jesse Swanson, Kyunghyun Cho, Doug Kondziolka, and Eric Oermann in [*Stereo Video Reconstruction Without Explicit Depth Maps for Endoscopic Surgery* (2021)](https://arxiv.org/abs/2109.08227). The model will take one left frame as input and predict its paired right frame. The perceptual loss compares features extracted from the prediction and the recorded target. This single-frame setting will serve as the reference for later generative models.

## Hamlyn surgical stereo dataset

The planned reproduction uses paired endoscopic views from the Hamlyn surgical stereo dataset. Original recordings belong in `data/raw/`, derived frame pairs in `data/processed/`, and partition manifests in `data/splits/`. Before training, the reproduction will document the selected recordings, camera ordering, frame pairing, and split protocol so that comparisons can use the same held-out pairs.

## Conditional diffusion research

A later phase will condition a diffusion model on the same left-camera input. A central question is how the model handles tissue revealed only in the right view, especially along instrument boundaries. Comparing its predictions with the U-Net on the same frame pairs will help distinguish visually plausible detail from agreement with the recorded anatomy.

## Project status

This initial revision establishes the project layout and research plan. Implementation, dataset preparation, and evaluation are planned milestones. Python dependencies will be recorded as implementation begins.
