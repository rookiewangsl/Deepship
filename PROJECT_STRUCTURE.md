# Project Structure

This project focuses on single-channel ship audio classification using a CNN route
built on mel-spectrogram inputs.

The current dataset folders remain in place:

- `DeepShip/`
- `ShipsEar/`

Recommended structure:

```text
Transformer/
├── DeepShip/                     # existing raw dataset
├── ShipsEar/                     # existing raw dataset
├── configs/
│   ├── datasets/
│   └── mel_ml/
├── docs/
├── notebooks/
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   └── reports/
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
- Keep outputs organized by experiment name under `outputs/`.
