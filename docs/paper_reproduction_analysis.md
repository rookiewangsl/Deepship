# Paper vs Previous Implementation

Paper: `jmse-12-00130.pdf`

## Confirmed mismatches that explain the gap

1. The paper uses `16 kHz` audio, while the previous implementation trained at `4 kHz`.
2. The paper uses fixed `3 s` non-overlapping segments and exactly `20,000` total samples.
3. The paper uses per-class balanced counts: `3500 train + 1000 val + 500 test`.
4. The previous implementation split by recording first, then evaluated on `8445` test segments instead of the paper's `2000`.
5. The previous implementation used random crop training, waveform augmentation, time masking, frequency masking, weighted sampling, class weights, label smoothing, and cosine scheduling. The paper explicitly states no data augmentation and only specifies plain cross-entropy, batch size `16`, learning rate `1e-2`, and early stopping patience `10`.
6. The paper reports `1.00 M` parameters for MA-CNN-A. The previous implementation had only `181,895` parameters, so it was not the same network.
7. The current stable implementation is fixed to a three-branch MA-CNN-A variant with scales `8, 16, 32` and additive fusion. This is intentionally narrower than the four-branch paper description in exchange for a single reproducible code path.

## Likely dominant cause of the 98.2% vs ~70% gap

The largest source is the split protocol. The paper constructs a balanced `20,000`-segment subset and splits at the segment level. That means segments from the same original recording can appear in train, validation, and test. The previous implementation used recording-level splitting, which is a much stricter and more realistic evaluation.

## What this reproduction changes

1. Replaces the training pipeline with the paper protocol.
2. Replaces the model with a paper-oriented MA-CNN-A implementation.
3. Removes augmentation and optimizer tricks not described in the paper.
4. Saves the reproduction split manifest and explicitly marks the remaining inferred details in the run config.
