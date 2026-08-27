# Project Structure

This project focuses on single-channel ship audio classification using two controlled
routes: a CNN built on log-Mel inputs and a pretrained Wav2Vec2-Conformer built on
raw waveforms.

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
│   ├── experiments/              # experiment routes and frozen protocols
│   ├── guides/                   # environment, storage, and run instructions
│   ├── research/                 # literature benchmarks and domain studies
│   └── figures/                  # documentation figures
├── notebooks/
├── results/                      # small, reviewable metrics and figures
├── protocols/                    # frozen portable split manifests and audits
├── runs/                         # returned full runs, local and Git-ignored
├── scripts/
│   ├── prepare/
│   ├── train/
│   └── eval/
├── src/
│   ├── data/                     # manifests, waveform and Mel datasets
│   ├── evaluation/               # segment/recording/vessel metrics
│   ├── models/                   # MA-CNN-A and Wav2Vec2-Conformer
│   ├── pipelines/
│   │   ├── mel_ml/
│   │   └── waveform_conformer/
│   └── utils/
└── tests/
```

Design choice:

- Share dataset indexing, label mapping, frozen splits, metrics, and aggregation.
- Keep the Mel-CNN and raw-waveform Conformer pipelines separate and reproducible.
- Keep source code, Git, documentation, and small result evidence in Dropbox.
- Keep raw data, checkpoints, caches, and complete runs under
  `/Volumes/T7/ProjectData/Deepship/`.
- The current six Windows runs may also be present in the Git-ignored local
  `runs/` directory for analysis; they are not repository source artifacts.
- Keep literature protocol comparisons in
  `docs/research/deepship_literature_benchmarks.md` and the current research
  routes in `docs/experiments/`.
