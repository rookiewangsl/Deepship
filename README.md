# DeepShip CNN Baseline

This repository keeps the DeepShip `MA-CNN-A` baseline:

- `16 kHz`
- `3 s` non-overlapping segments
- `64 x 94` log-Mel input
- `5000` samples per class with split `3500 / 1000 / 500`

## Documentation

The documentation index covers experiment plans, execution guides, literature
benchmarks, and domain research:

- [`docs/README.md`](docs/README.md)
- [`docs/research/deepship_literature_benchmarks.md`](docs/research/deepship_literature_benchmarks.md)

The archived `95.15%` result belongs to the earlier 486,838-parameter model.
The current 532,166-parameter architecture has completed the three frozen
protocols for model seeds 42 and 43:

| Protocol | Primary aggregation | Accuracy, mean ± sample SD |
|---|---|---:|
| `segment_level` | segment | `97.30 ± 0.21%` |
| `recording_disjoint` | recording | `70.19 ± 4.08%` |
| `vessel_name_disjoint` | vessel-name group | `53.13 ± 4.42%` |

For the directly comparable segment metric, the same protocols score
`97.30 ± 0.21%`, `66.20 ± 0.49%`, and `50.98 ± 0.18%`. These are two-seed
results, not the originally planned three-seed final report. Vessel-name groups
are not verified MMSI/IMO physical identities.

## Train MA-CNN-A

```bash
export DEEPSHIP_DATA_ROOT=/Volumes/T7/ProjectData/Deepship/datasets/DeepShip

python scripts/train/train_deepship_macnna.py \
  --data-root "$DEEPSHIP_DATA_ROOT" \
  --output-root ./runs/deepship_macnna_paper
```

The CNN baseline now uses a `linear warmup + cosine annealing` learning-rate schedule by default.

The frozen three-protocol isolation comparison and Windows GPU workflow are documented in
[`docs/guides/windows_training_guide.md`](docs/guides/windows_training_guide.md).

Before a frozen three-protocol run, validate the mounted T7 against all committed manifests. On macOS,
the current dataset root is:

```bash
python scripts/prepare/validate_deepship_protocols.py \
  --data-root /Volumes/T7/ProjectData/Deepship/datasets/DeepShip \
  --protocol all \
  --no-write-reports
```

On the Windows machine used for the completed runs, the repository was
`C:\Users\shilongwang\Desktop\Deepship` and the T7 dataset root was
`D:\ProjectData\Deepship\datasets\DeepShip`. Use
`scripts\windows\check_environment.ps1` with the actual drive letter. The
manifest files contain no drive letter or absolute data path.

Storage layout and migration details are documented in
[`docs/guides/storage_layout.md`](docs/guides/storage_layout.md). Small legacy metrics and
figures are retained under `results/`; the current returned runs are available
under the Git-ignored local `runs/` directory, while raw data and durable large
artifacts remain on T7.

## Main Files

- `src/data/deepship.py`: DeepShip scan, segment generation, and paper split protocol
- `src/models/ma_cnn_a.py`: MA-CNN-A implementation
- `src/pipelines/mel_ml/train_deepship_macnna.py`: training and evaluation
- `scripts/train/train_deepship_macnna.py`: command-line entrypoint
- `scripts/eval/summarize_isolation_runs.py`: strict validator/summarizer for the configured three-seed run matrix
