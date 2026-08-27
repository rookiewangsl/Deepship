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
- The current six Windows runs may also be present in the Git-ignored local
  `runs/` directory for analysis; they are not repository source artifacts.
- Keep literature protocol comparisons in
  `docs/research/deepship_literature_benchmarks.md` and the current research
  routes in `docs/experiments/`.
