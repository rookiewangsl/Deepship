# External storage layout

The source code and Git repository stay in Dropbox:

```text
/Users/shilongwang/Library/CloudStorage/Dropbox/Code/Deepship
```

Large datasets and generated artifacts stay on T7:

```text
/Volumes/T7/ProjectData/Deepship/
  datasets/DeepShip/
  datasets/ShipsEar/
  precomputed/
  checkpoints/
  runs/
  legacy/
```

The project-local `DeepShip` and `ShipsEar` entries are machine-local symbolic
links and are ignored by Git. The training code can also use an explicit path:

```bash
export DEEPSHIP_DATA_ROOT=/Volumes/T7/ProjectData/Deepship/datasets/DeepShip

python scripts/train/train_deepship_macnna.py \
  --data-root "$DEEPSHIP_DATA_ROOT" \
  --output-root /Volumes/T7/ProjectData/Deepship/runs/macnna_01
```

Run the read-only storage check before training:

```bash
python scripts/check_storage.py
```

If T7 is not mounted, stop instead of creating replacement dataset or output
directories on the internal disk. The `precomputed` directory contains legacy
Mel/STFT tensors; the current MA-CNN-A pipeline still reads source audio and
does not consume those tensors automatically. `--cache-root` is currently a
reserved experiment path, not an implemented feature-cache loader.

Small final metrics and figures may be copied into `results/`. Normally,
checkpoints, logs, and complete runs remain on T7 or another Git-ignored local
run directory rather than being committed.

## Current Windows and returned-run locations

The completed seed 42/43 isolation runs used:

```text
C:\Users\shilongwang\Desktop\Deepship
D:\ProjectData\Deepship\datasets\DeepShip
C:\Users\shilongwang\Desktop\Deepship\runs\isolation_comparison_v1_epoch_output
```

After training, the six complete run directories were copied into the local
repository root under `runs/` for analysis. `runs/` is Git-ignored because it
contains large checkpoints, predictions, logs, and figures. Documentation may
quote metrics from these runs, but Git should contain only the small legacy
evidence under `results/`, frozen protocol files under `protocols/`, and the
written summaries under `docs/`.
