# Project Structure

This project focuses on single-channel ship audio classification using two routes:

1. Mel-spectrogram or cepstral-style acoustic features with traditional machine learning
2. Raw waveform classification with a Transformer-based model

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
│   ├── mel_ml/
│   └── waveform_transformer/
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
│   │   └── waveform_transformer/
│   ├── pipelines/
│   │   ├── mel_ml/
│   │   └── waveform_transformer/
│   └── utils/
└── tests/
```

Design choice:

- Share dataset indexing, label mapping, splitting, metrics, and visualization.
- Separate the two routes at feature extraction, model definition, and experiment pipeline.
- Keep outputs separated by route so comparisons stay clear and reproducible.

Why both routes are worth keeping:

- `mel_ml` is a strong and interpretable baseline, easier to explain in interviews.
- `waveform_transformer` reflects a modern end-to-end deep learning route and gives the project stronger AI depth.
- Using both creates a clean "baseline vs advanced model" narrative for resumes and project storytelling.
