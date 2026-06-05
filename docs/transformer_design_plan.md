# Transformer Route Design Plan

## 1. Objective

This document defines the first implementation plan for the raw-waveform Transformer route.

The goal is to build a lightweight end-to-end baseline for single-channel ship-noise classification that can be:

1. trained on `ShipsEar` and `DeepShip`
2. compared fairly against the `Mel + CNN` baseline
3. explained clearly in reports, resumes, and interviews


## 2. Design Positioning

This route is **not** intended to be the biggest or most complex model.

Instead, the first version should be:

- lightweight
- stable to train
- easy to explain
- small enough to run on a personal laptop
- expressive enough to learn from raw waveform directly

We use the Transformer route to answer a core project question:

> Can a lightweight end-to-end model using raw audio compete with or complement the `Mel + CNN` baseline on ship-radiated-noise classification?


## 3. Input Definition

Shared preprocessing for the Transformer route:

- mono audio
- sample rate: `4000 Hz`
- clip duration: `5 s`
- total waveform length: `20000` samples

Processing chain:

`wav -> mono -> resample(4k) -> crop/pad to 5 s -> amplitude normalization -> patch embedding -> Transformer Encoder -> classifier`


## 4. Dataset Strategy

### 4.1 ShipsEar

For `ShipsEar`, each sample is already clean and short enough to use as a fixed clip.

Recommended use:

- start Transformer development on `ShipsEar`
- verify convergence and debug the full training loop

### 4.2 DeepShip

For `DeepShip`, use the same recording-level split rule already established for the CNN route:

- split by recording first
- only then generate `5 s` training examples

This avoids leakage from the same raw recording into different data splits.


## 5. Recommended First Model

### 5.1 Summary

Recommended initial architecture:

- parameter scale: about `300k`
- encoder layers: `4`
- embedding dimension: `128`
- attention heads: `4`
- MLP ratio: `4`
- dropout: `0.1`

This is intentionally much smaller than a standard ViT and is suitable as a first raw-waveform baseline.

### 5.2 Tensor Flow

Input waveform:

- shape: `(B, 1, 20000)` or `(B, 20000)`

Patch partition:

- patch size candidate: `200`
- number of patches: `20000 / 200 = 100`

Patch embedding:

- each patch is projected from length `200` to embedding dimension `128`

Transformer tokens:

- token shape: `(B, 100, 128)`

After positional encoding and encoder stack:

- output shape: `(B, 100, 128)`

Pooling:

- first choice: mean pooling over tokens

Classifier:

- `LayerNorm`
- `Linear(128 -> num_classes)`


## 6. Architecture Details

### 6.1 Patch Embedding

Recommended first implementation:

- use non-overlapping 1D patches
- flatten each patch
- apply a learnable linear projection

Reason:

- simple
- easy to debug
- easy to explain

Alternative later:

- 1D convolutional patch embedding

This may improve local feature extraction, but it is not necessary for the first version.

### 6.2 Positional Encoding

Recommended first choice:

- learnable positional embedding

Reason:

- standard
- easy to implement
- works well for fixed-length token sequences

### 6.3 Transformer Encoder

Recommended first configuration:

- `num_layers = 4`
- `embed_dim = 128`
- `num_heads = 4`
- `mlp_ratio = 4`
- `dropout = 0.1`
- `attention_dropout = 0.1`

This gives a compact model with enough depth to model temporal relations across patches.

### 6.4 Pooling Strategy

Recommended first choice:

- mean pooling over all token outputs

Reason:

- simpler than introducing a `CLS` token
- often stable for classification baselines

Alternative later:

- prepend a learnable `CLS` token

### 6.5 Classification Head

Recommended first choice:

- `LayerNorm`
- `Linear(embed_dim -> num_classes)`

Optional later:

- `LayerNorm -> Linear(embed_dim -> hidden) -> GELU -> Dropout -> Linear(hidden -> num_classes)`

Not necessary for the first version.


## 7. Recommended Training Strategy

### 7.1 Optimizer

Recommended:

- `AdamW`

Suggested initial hyperparameters:

- learning rate: `5e-4`
- weight decay: `1e-4`

### 7.2 Scheduler

Recommended:

- `ReduceLROnPlateau`

Monitor:

- `val_macro_f1`

Suggested initial settings:

- factor: `0.5`
- patience: `2`
- threshold: `1e-3`
- min lr: `1e-6`

### 7.3 Loss

Recommended:

- `CrossEntropyLoss`

Suggested:

- label smoothing: `0.05`

For `DeepShip`, class-weighted cross entropy may also be useful, similar to the CNN route.

### 7.4 Early Stopping

Recommended:

- monitor `val_macro_f1`
- patience: `8` to `10`


## 8. Recommended Augmentation

For the raw-waveform Transformer route, augmentation should remain in the waveform domain for the first version.

### 8.1 Use in the first version

Recommended initial augmentation set:

- random time shift
- random crop position
- random gain perturbation
- small Gaussian noise

### 8.2 Do not start with

Not recommended in the first version:

- strong frequency-domain masking
- mixup before baseline is stable
- MAE-style pretraining before the supervised baseline is verified

These can be explored later as expansion experiments.


## 9. Preprocessing Requirements

### 9.1 Necessary

- resample to `4 kHz`
- fix duration to `5 s`
- zero-pad short clips
- normalize waveform amplitude

### 9.2 Normalization

Recommended first choice:

- per-sample zero-mean, unit-variance normalization

Alternative:

- clamp or scale to `[-1, 1]`

For the first version, standardization is preferred because it stabilizes Transformer training.


## 10. Why This Model Is Reasonable

This model is a good first version because:

- it is much smaller than typical ViT models
- it is lightweight enough for personal hardware
- it learns directly from raw waveform
- it keeps the project technically meaningful
- it supports fair comparison with the `Mel + CNN` baseline

The expected project value is not only raw accuracy, but also:

- demonstrating end-to-end audio modeling
- understanding patch embedding and sequence modeling
- comparing hand-crafted time-frequency representation versus raw signal modeling


## 11. Risks and Expectations

### 11.1 Likely Risks

- training may be less stable than `Mel + CNN`
- first-run accuracy may not exceed the CNN baseline
- model may need careful tuning of patch size and regularization

### 11.2 Realistic Expectation

Expected first result:

- a stable raw-waveform baseline
- meaningful comparison with the CNN baseline

Not guaranteed at first:

- better accuracy than `Mel + CNN`


## 12. Hyperparameters To Confirm

The most important hyperparameters that should be agreed before implementation are:

1. patch size
2. embedding dimension
3. number of encoder layers
4. number of attention heads
5. pooling type
6. waveform normalization method
7. augmentation strength

### Recommended Initial Defaults

- sample rate: `4000`
- clip duration: `5 s`
- patch size: `200`
- number of patches: `100`
- embed dim: `128`
- encoder layers: `4`
- attention heads: `4`
- mlp ratio: `4`
- dropout: `0.1`
- pooling: `mean`
- optimizer: `AdamW`
- learning rate: `5e-4`
- weight decay: `1e-4`
- label smoothing: `0.05`
- scheduler: `ReduceLROnPlateau`
- early stopping metric: `val_macro_f1`


## 13. Recommended Implementation Order

1. implement waveform dataset and preprocessing
2. implement patch embedding
3. implement lightweight Transformer encoder classifier
4. run first baseline on `ShipsEar`
5. verify outputs and visualizations
6. migrate the same pipeline to `DeepShip`
7. compare with `Mel + CNN`
