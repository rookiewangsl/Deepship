# T7 storage migration record (2026-08-10)

## Result

The Git repository and source code remain at:

```text
/Users/shilongwang/Library/CloudStorage/Dropbox/Code/Deepship
```

Large datasets and generated artifacts were copied to:

```text
/Volumes/T7/ProjectData/Deepship
```

The old mixed code/data repository at `/Volumes/T7/DeepShip` has **not** been
deleted. It remains the rollback source until its removal is explicitly
approved.

## Migrated contents

| Content | New location | Validation |
| --- | --- | --- |
| DeepShip | `datasets/DeepShip` | 619 non-AppleDouble source files; 609 real WAV recordings; 29,483,305,387 logical bytes copied; full rsync checksum comparison passed |
| ShipsEar | `datasets/ShipsEar` | 2,223 files; 355,777,812 logical bytes; full rsync checksum comparison passed |
| Precomputed features | `precomputed` | 12 files; 2,155,456,556 logical bytes; full rsync checksum comparison passed |
| Legacy outputs excluding precomputed data | `runs/legacy` | 21 files; 48,121,876 logical bytes; full rsync checksum comparison passed |

macOS created `._*` AppleDouble metadata files on the exFAT volume. They were
removed from the new target, and both the dataset scanner and storage checker
now ignore them if macOS creates them again.

## Git recovery

The useful 15-commit history was recovered without the unreachable multi-GB
objects in the old `.git` directory. `git fsck` passed on the recovered
repository. Two rollback artifacts are retained on T7:

```text
/Volumes/T7/ProjectData/Deepship/legacy/deepship_reachable_history_20260810.bundle
/Volumes/T7/ProjectData/Deepship/legacy/dropbox_code_before_git_20260810.tar.gz
```

The active migration work is committed on branch
`codex/t7-storage-migration`. Dataset links, checkpoints, logs, complete run
directories, and model binaries are ignored by Git. Small final metrics and
figures may be retained under `results/`.

## Functional validation

The read-only storage check passed with these dataset counts:

| Class | Recordings | Available non-overlapping 3-second segments |
| --- | ---: | ---: |
| Cargo | 109 | 12,801 |
| Passenger | 191 | 15,410 |
| Tank | 240 | 14,762 |
| Tug | 69 | 13,495 |
| Total | 609 | 56,468 |

The complete paper-style split can be constructed with 5,000 segments per
class: 14,000 training segments, 4,000 validation segments, and 2,000 test
segments. Each class contributes 3,500/1,000/500 segments respectively.

The following runtime checks also passed using the existing Miniconda Python
environment:

- Read a representative 32 kHz mono WAV through `soundfile`.
- Convert a 3-second sample to a `[1, 64, 94]` Mel-spectrogram tensor.
- Run the current 532,166-parameter MA-CNN-A model and obtain finite `[1, 4]`
  logits.
- Complete a one-epoch, 4-class CPU smoke training run and write its
  checkpoint, metrics, plots, split report, and run configuration under
  `runs/storage_smoke_20260810` on T7.

The smoke-run accuracy is not a quality result because it intentionally uses
only four samples per class. It validates the data-to-output path. The
previously reported 95.15% result is preserved separately under
`results/legacy_paper_reproduction` with its architecture and split caveats.

## Remaining cleanup candidate

After the project has been used successfully in a normal session and rollback
is no longer needed, `/Volumes/T7/DeepShip` is a deletion candidate. It
currently occupies about 54 GB of allocated space, largely because its old
Git object database contains unreachable large blobs. Do not delete it until
the new layout and Git branch have been reviewed and deletion is explicitly
approved.
