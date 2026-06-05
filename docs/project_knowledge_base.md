# Project Knowledge Base

## 1. Project Goal

This project is designed as an AI audio classification project for learning, implementation, result visualization, and resume presentation.

The final deliverable should support three goals:

1. Build a complete and runnable ship-radiated-noise classification pipeline
2. Understand the data processing, feature extraction, model training, and evaluation flow
3. Produce visual results and a project story that can be written into a resume and explained clearly in interviews


## 2. Core Positioning

The original Word documents describe two technical routes for optical fiber DAS data:

1. Mel-spectrum or cepstral-style features with traditional machine learning
2. Transformer-based end-to-end classification

For this project, we do **not** reproduce the original DAS pipeline.

We intentionally simplify the task to standard **single-channel ship audio classification** using public datasets:

- `ShipsEar`
- `DeepShip`

We do **not** consider:

- DAS multi-channel collaboration
- DAS array calibration
- DOA estimation
- complex channel-level cleaning
- optical fiber sensing signal reconstruction

This project focuses on migrating the **two classification ideas** from the DAS context into standard underwater ship-noise audio classification.


## 3. Dataset Strategy

### 3.1 Do not fuse ShipsEar and DeepShip

The two datasets should **not** be merged into one unified training dataset.

Reason:

- Their official label systems are different
- Their ship categories do not align perfectly
- Forced label merging would introduce ambiguity and weaken the project narrative

Recommended strategy:

1. Run experiments on `ShipsEar` independently
2. Run experiments on `DeepShip` independently
3. Compare methods within each dataset
4. Compare findings across datasets only at the conclusion stage


### 3.2 Official label understanding

#### ShipsEar

The common 5-class experimental setting is:

- `A`: fishing boats, trawlers, mussel boats, tugboats, dredgers
- `B`: motorboats, pilot boats, sailboats
- `C`: passenger ferries
- `D`: ocean liners, ro-ro vessels
- `E`: background noise

#### DeepShip

The official 4 ship classes are:

- `Cargo`
- `Passenger`
- `Tanker`
- `Tug`

Conclusion:

- `ShipsEar` and `DeepShip` labels are **not fully unified**
- They should be treated as two separate classification tasks


## 4. Project Execution Order

The recommended implementation order is:

1. Understand `ShipsEar`
2. Build and verify the Mel-spectrogram route on `ShipsEar`
3. Build and verify the raw-waveform Transformer route on `ShipsEar`
4. Repeat the same methodology on `DeepShip`
5. Produce final visual comparisons and resume-ready summary

Why start from `ShipsEar`:

- smaller and easier to understand
- better for fast iteration
- more suitable for building the first full pipeline


## 5. Common Data Pipeline

Both routes should share a common data-processing foundation.

### 5.1 Data understanding

For each dataset:

- inspect folder structure
- inspect class meanings
- inspect sampling rate
- inspect audio duration distribution
- inspect mono/stereo format
- count samples in each class

### 5.2 Data preprocessing

Recommended shared preprocessing steps:

1. resample all audio to a unified sampling rate
2. convert to mono if needed
3. fix audio length
4. slice long audio into fixed-length segments
5. zero-pad short audio
6. normalize waveform amplitude
7. define train/validation/test split

### 5.3 Required statistics output

For each dataset, output:

- class definitions
- sample count per class
- sampling rate statistics
- duration statistics
- split statistics for train/validation/test
- preprocessing parameter summary


## 6. Mel-Spectrogram Route

### 6.1 Route objective

Use `ShipsEar` first to build a standard and interpretable baseline.

This route is meant to be:

- easy to understand
- easy to explain in interviews
- suitable as a baseline for comparison with Transformer models

### 6.2 Task definition on ShipsEar

Use PyTorch to perform 5-class ship-noise classification on `ShipsEar`:

- `A`: small boat / fishing / tug-like vessels
- `B`: motorboat / sailboat-like vessels
- `C`: passenger ferry
- `D`: ocean liner / ro-ro vessels
- `E`: background noise

### 6.3 Workflow

The core processing chain is:

`wav -> resample -> slice -> mel spectrogram -> CNN -> CrossEntropyLoss -> predicted class`

### 6.4 Experiment stages

#### Stage 1: no augmentation

First run the full pipeline without augmentation:

- confirm data loading works
- confirm training loop works
- confirm evaluation works
- obtain first baseline results

#### Stage 2: augmentation

After the baseline is stable, add lightweight augmentation:

- random time shift
- frequency masking
- time masking

Then rerun training and compare with the baseline.

### 6.5 Required outputs

- accuracy
- precision
- recall
- F1-score
- confusion matrix
- classification report
- training and validation curves
- representative mel-spectrogram visualizations

### 6.6 Expansion step

After the route is stable on `ShipsEar`, apply the same idea to `DeepShip`.


## 7. Raw-Waveform Transformer Route

### 7.1 Route objective

This route is the end-to-end deep learning path.

Its role in the project is to:

- show a modern AI modeling approach
- directly model raw audio waveform
- create a stronger technical story for resume and interviews

### 7.2 First-stage principle

Do not begin with complex self-supervised pretraining.

First, build a clean supervised baseline on `ShipsEar`.

### 7.3 Workflow

The core processing chain is:

`wav -> resample -> fixed length -> slice or pad -> normalize -> patch embedding -> Transformer Encoder -> classification head -> CrossEntropyLoss -> predicted class`

### 7.4 Model design principles

The first version should remain simple and explainable:

- single-channel waveform input
- 1D patch or segment embedding
- positional encoding
- several Transformer Encoder layers
- mean pooling or CLS token
- linear classification head

The goal is not maximum novelty at first, but a clean, reproducible baseline.

### 7.5 Experiment stages

#### Stage 1: no augmentation

First run the supervised end-to-end route without augmentation:

- verify convergence
- verify validation performance
- verify output metrics and visualizations

#### Stage 2: waveform augmentation

After the baseline is stable, add simple waveform-domain augmentation:

- random time shift
- random gain perturbation
- small Gaussian noise
- random crop offset

Then rerun training and compare with the baseline.

#### Stage 3: optional MAE extension

If time and project maturity allow, MAE-style self-supervised pretraining can be added later as an extension experiment.

This is **optional**, not part of the first runnable version.

### 7.6 Required outputs

- accuracy
- precision
- recall
- F1-score
- confusion matrix
- classification report
- training and validation curves

Recommended extra outputs:

- waveform examples
- patching diagram or shape description
- attention or intermediate feature visualization
- comparison of error cases against the Mel route

### 7.7 Expansion step

After the route is stable on `ShipsEar`, apply the same methodology to `DeepShip` as a separate 4-class task.


## 8. Experiment Logic

The project should produce two layers of comparison:

### 8.1 Intra-dataset comparison

Within the same dataset:

- Mel-spectrogram + CNN baseline
- Raw waveform + Transformer Encoder

### 8.2 Cross-dataset comparison

Across datasets:

- ShipsEar experiment conclusions
- DeepShip experiment conclusions

Important:

- compare conclusions, not mixed-label joint training


## 9. Final Deliverables

The project should finally produce:

### 9.1 Technical outputs

- runnable training and evaluation pipelines
- saved models
- metrics tables
- confusion matrices
- classification reports
- learning curves
- representative feature visualizations

### 9.2 Project understanding outputs

- clear explanation of preprocessing
- clear explanation of label design
- clear explanation of the Mel route
- clear explanation of the Transformer route
- clear explanation of augmentation strategy
- clear explanation of evaluation metrics

### 9.3 Resume and interview outputs

The project should support:

- a concise resume bullet
- a longer project description
- a clear verbal explanation of the full pipeline
- explanation of why two routes were used
- explanation of why datasets were handled separately


## 10. Resume Narrative Direction

This project should eventually be expressible as:

- built a ship-radiated-noise classification project based on public underwater acoustic datasets
- implemented and compared a Mel-spectrogram CNN baseline and a raw-waveform Transformer model
- completed data preprocessing, label organization, training, evaluation, and visualization
- analyzed model performance through confusion matrices, classification reports, and feature visualizations


## 11. Current Agreed Development Priority

Current priority order:

1. understand `ShipsEar`
2. complete `ShipsEar` Mel-route baseline
3. add Mel-route augmentation experiment
4. complete `ShipsEar` Transformer-route baseline
5. add Transformer-route augmentation experiment
6. expand both routes to `DeepShip`
7. summarize final results for presentation and resume writing


## 12. Agreed Baseline Hyperparameters

The following values are currently agreed as the first runnable baseline.

### 12.1 Shared data settings

- sampling rate: `4000 Hz`
- channel mode: `mono`
- fixed audio length: `5 s`
- dataset split ratio: `70 / 15 / 15`
- split strategy: stratified split by class
- random seed: recommended default `42`

Important note for `ShipsEar`:

- the current local `ShipsEar` files are already organized as `16 kHz`, `5 s`, single-channel `.wav`
- for the project baseline, data will still be resampled to `4 kHz`

### 12.2 Mel-route baseline settings

Recommended first baseline values:

- `n_fft = 256`
- `hop_length = 64`
- `win_length = 256`
- `n_mels = 64`
- `f_min = 20`
- `f_max = 2000`

Reasoning:

- `4 kHz` sampling rate gives a Nyquist frequency of `2000 Hz`
- ship-radiated noise is mainly concentrated in low-frequency bands
- `n_fft = 256` provides a practical frequency resolution for the first baseline
- `hop_length = 64` provides sufficient temporal detail without excessive compute cost
- `64` Mel bins are enough for a compact and interpretable baseline

Recommended first baseline processing chain:

`wav -> resample to 4 kHz -> keep 5 s -> log-Mel spectrogram -> CNN classifier`

### 12.3 Training-policy reminder

For the first Mel-route baseline:

- do not use augmentation first
- run the clean pipeline first
- after the baseline is stable, add augmentation such as time shift, time masking, and frequency masking

