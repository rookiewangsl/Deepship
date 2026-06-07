# DeepShip MA-CNN-A Paper Reproduction

This repository is reduced to the code path needed to reproduce the paper `jmse-12-00130.pdf`:

- `16 kHz` resampling
- `3 s` non-overlapping segments
- `20,000` total samples
- per-class split `3500 / 1000 / 500`
- `64 x 94` log-Mel input
- MA-CNN-A style multi-branch asymmetric CNN

## Train

```bash
python scripts/train/train_deepship_macnna.py \
  --data-root DeepShip \
  --output-root outputs/deepship_macnna_paper
```

## Main Files

- `src/data/deepship.py`: DeepShip scan, segment generation, paper split protocol
- `src/models/ma_cnn_a.py`: paper-oriented MA-CNN-A implementation
- `src/pipelines/mel_ml/train_deepship_macnna.py`: training and evaluation
- `docs/paper_reproduction_analysis.md`: paper vs previous implementation comparison

## Important Note

The paper does not publish source code. This implementation matches the explicit paper settings and clearly labels inferred parts in the saved run config.
