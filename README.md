# DeepShip CNN Baseline

This repository keeps the DeepShip `MA-CNN-A` baseline:

- `16 kHz`
- `3 s` non-overlapping segments
- `64 x 94` log-Mel input
- `5000` samples per class with split `3500 / 1000 / 500`

## Train MA-CNN-A

```bash
python scripts/train/train_deepship_macnna.py \
  --data-root DeepShip \
  --output-root outputs/deepship_macnna_paper
```

## Main Files

- `src/data/deepship.py`: DeepShip scan, segment generation, and paper split protocol
- `src/models/ma_cnn_a.py`: MA-CNN-A implementation
- `src/pipelines/mel_ml/train_deepship_macnna.py`: training and evaluation
- `scripts/train/train_deepship_macnna.py`: command-line entrypoint
