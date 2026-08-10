# Project Structure

This project focuses on single-channel ship audio classification using a CNN route
built on mel-spectrogram inputs.

The project exposes machine-local links for the datasets:

- `DeepShip/`
- `ShipsEar/`

Recommended structure:

```text
Deepship/
├── DeepShip/                     # existing raw dataset
├── ShipsEar/                     # existing raw dataset
├── configs/
│   ├── datasets/
│   └── mel_ml/
├── docs/
├── notebooks/
├── results/                      # small, reviewable metrics and figures
├── scripts/
│   ├── prepare/
│   ├── train/
│   └── eval/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   │   └── mel_ml/
│   ├── models/
│   ├── pipelines/
│   │   └── mel_ml/
│   └── utils/
└── tests/
```

Design choice:

- Share dataset indexing, label mapping, splitting, metrics, and visualization.
- Keep the mel-based data path, model definition, and experiment pipeline simple and reproducible.
- Keep source code, Git, documentation, and small result evidence in Dropbox.
- Keep raw data, checkpoints, caches, and complete runs under
  `/Volumes/T7/ProjectData/Deepship/`.
