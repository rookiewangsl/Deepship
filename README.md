# DeepShip CNN Baseline

This repository keeps the DeepShip `MA-CNN-A` baseline:

- `16 kHz`
- `3 s` non-overlapping segments
- `64 x 94` log-Mel input
- `5000` samples per class with split `3500 / 1000 / 500`

## Project Handbook

The Chinese project handbook covers the dataset protocol, model architecture,
training pipeline, experiment history, result boundaries, run commands, and
interview questions:

- [`docs/handbook/README.md`](docs/handbook/README.md)

The archived `95.15%` result belongs to the earlier 486,838-parameter model.
The current 532,166-parameter architecture has passed end-to-end smoke checks
but still requires a full training run before a quality metric can be reported.

## Train MA-CNN-A

```bash
export DEEPSHIP_DATA_ROOT=/Volumes/T7/DeepShip/DeepShip

python scripts/train/train_deepship_macnna.py \
  --data-root "$DEEPSHIP_DATA_ROOT" \
  --output-root ./runs/deepship_macnna_paper
```

The CNN baseline now uses a `linear warmup + cosine annealing` learning-rate schedule by default.

The frozen three-protocol isolation comparison and Windows GPU workflow are documented in
[`docs/windows_training_guide.md`](docs/windows_training_guide.md).

Before a frozen three-protocol run, validate the mounted T7 against all committed manifests. On macOS,
the current dataset root is:

```bash
python scripts/prepare/validate_deepship_protocols.py \
  --data-root /Volumes/T7/DeepShip/DeepShip \
  --protocol all \
  --no-write-reports
```

On Windows, use `scripts\windows\check_environment.ps1` and pass the actual T7 drive letter. The manifest
files contain no drive letter or absolute data path.

Storage layout and migration details are documented in
[`docs/storage_layout.md`](docs/storage_layout.md). Small legacy metrics and
figures are retained under `results/`; checkpoints and full runs remain on T7.

## Main Files

- `src/data/deepship.py`: DeepShip scan, segment generation, and paper split protocol
- `src/models/ma_cnn_a.py`: MA-CNN-A implementation
- `src/pipelines/mel_ml/train_deepship_macnna.py`: training and evaluation
- `scripts/train/train_deepship_macnna.py`: command-line entrypoint
- `scripts/eval/summarize_isolation_runs.py`: validate and summarize the nine formal isolation runs
