# Modelling Temporal Drift in Music Preferences using Emotion-Aware Sequential Representation Learning

MSc Research Thesis — Manohar Ganesh Kulkarni

This repository contains the full implementation code for the dissertation above: an emotion-aware
sequential learning framework that models temporal drift in music listening preferences using
LSTM-with-attention and Transformer architectures, validated through a novel Preference Drift Score
on two independent datasets.

## Repository Contents

| File | Description |
|---|---|
| `Music_Preference_Drift.ipynb` | Primary pipeline: Turkish Music Emotion Dataset — data cleaning, exploratory analysis, sequence construction, LSTM+Attention and Transformer training, classification evaluation, and Preference Drift Score validation. |
| `Memo2496_Preference_Drift_v2.ipynb` | Cross-dataset validation pipeline: reproduces the full framework on Memo2496 (IEEE DataPort), using valence-arousal-derived features, to test whether the Preference Drift Score generalises beyond the primary dataset. |
| `turkish_music_emotion_dataset.csv` | The Turkish Music Emotion Dataset in CSV form (400 tracks, 50 audio features, 4 emotion classes). |

## Datasets Used

- **Turkish Music Emotion Dataset** — Er, M. (2019). UCI Machine Learning Repository.
  https://archive.ics.uci.edu/dataset/862/turkish+music+emotion
- **Memo2496** — Expert-Annotated Dataset for Music Emotion Recognition. IEEE DataPort.
  https://ieee-dataport.org/documents/memo2496-expert-annotated-dataset-and-dual-view-adaptive-framework-music-emotion

  The Memo2496 annotation files are not included in this repository due to their size; they can be
  downloaded directly from the link above and placed in an `Annotations/` folder alongside the notebook.

## Key Results

| Dataset | LSTM Accuracy | Transformer Accuracy | Drift Score (label unchanged) | Drift Score (label changed) | Ratio |
|---|---|---|---|---|---|
| Turkish Music Emotion Dataset | 99.92% | 99.92% | 0.0084 | 0.4380 | ~52x |
| Memo2496 | 97.58% | 98.00% | 0.0160 | 0.4358 | ~27x |

The Preference Drift Score rises sharply and consistently at points where the ground-truth emotion
label genuinely changes, on two independently sourced datasets with different feature representations,
supporting the central claim that the score captures a real, generalisable phenomenon rather than a
dataset-specific artefact.

## Requirements

```
python >= 3.11
tensorflow
scikit-learn
pandas
numpy
matplotlib
seaborn
```


Manohar Ganesh Kulkarni (2026). *Modelling Temporal Drift in Music Preferences using Emotion-Aware Sequential
Representation Learning*. MSc Thesis.
