# Dry Bean Classification with Neural Networks

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Keras](https://img.shields.io/badge/Keras-TensorFlow-red) ![Task](https://img.shields.io/badge/Task-Multi--class%20classification-green)

A multi-class neural network (MLP) built with Keras that predicts the type of a dry bean from 16 shape measurements taken from images. The project covers the full workflow: Keras Sequential models, scikit-learn integration, cross validation, grid search, checkpoints, dropout and early stopping.

## Table of contents

1. [Problem statement](#problem-statement)
2. [Dataset](#dataset)
3. [Project workflow](#project-workflow)
4. [Model](#model)
5. [Experiments](#experiments)
6. [Results](#results)
7. [Error analysis](#error-analysis)
8. [Limitations and future work](#limitations-and-future-work)
9. [Repository structure](#repository-structure)
10. [How to run](#how-to-run)
11. [Tools](#tools)
12. [Reference](#reference)

## Problem statement

Sorting dry beans by type is usually done by hand, which is slow and inconsistent. The goal is to train a neural network that takes the geometric measurements of a bean (size, shape and roundness features) and predicts which of 7 registered bean types it belongs to.

## Dataset

- **Source:** [UCI Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) (Koklu and Ozkan, 2020)
- **Samples:** 13,611 rows, 13,543 unique beans after removing 68 duplicate rows
- **Features:** 16 numeric shape features, for example Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness and ShapeFactor1 to 4
- **Target:** 7 classes: Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira
- **Missing values:** none
- **Class balance:** imbalanced, so the classification report (precision, recall, F1) is used next to accuracy
- The dataset file `Dry_Bean_Dataset.xlsx` is included in the repository. If it is missing, the notebook downloads it automatically on the first run (internet connection needed).

## Project workflow

1. **Exploratory data analysis:** class distribution, correlation analysis (15 of 120 feature pairs have a correlation above 0.95, for example Area and ConvexArea) and class separation per feature
2. **Preprocessing:** duplicate rows are removed before the split, so the same bean cannot appear in both the training set and the test set. Then a stratified 70/15/15 train/validation/test split and standardization fitted on the training set only, to avoid data leakage
3. **Baseline model:** a small 16-8 network used as the reference
4. **Topology comparison:** five network sizes trained and compared
5. **Regularization:** dropout, early stopping and model checkpointing (both watch the validation loss, and the best weights are reloaded from the checkpoint file after training)
6. **Model selection:** 5-fold stratified cross validation and grid search with SciKeras and scikit-learn
7. **Final evaluation:** one evaluation on the held-out test set
8. **Diagnostics:** learning curves, learning-curve vs. training-size study and error analysis
9. **Saving:** the final model is saved in the Keras format
10. **Baseline comparison:** Logistic Regression and Random Forest trained on the same train/test split as the MLP

A fixed random seed (7) is used everywhere for reproducibility.

## Model

| Part | Choice |
|---|---|
| Input | 16 standardized features |
| Hidden layers | Dense layers with ReLU (best CV setting: 32-16) |
| Output | Dense(7) with softmax |
| Loss | Sparse categorical cross entropy |
| Optimizer | Adam (best from grid search) |
| Batch size | 64 |
| Regularization | Early stopping and checkpointing, both on the validation loss; dropout was tested |

## Experiments

**Topologies (validation accuracy):** all five topologies gave a similar validation accuracy (between 92.9% and 94.1% at the best epoch). Bigger networks fit the training set better but did not improve validation accuracy, so a small network is enough for this data.

**5-fold cross validation:**

| Network | CV accuracy |
|---|---|
| 16-8 | 92.32% |
| 32-16 | 92.76% |
| 64-32 with dropout | 92.80% |

**Grid search:** the best setting was Adam, no dropout, batch size 64 (about 92.8% CV accuracy).

**Comparison with classical models (same split, same seed):**

| Model | Test accuracy | Macro F1 |
|---|---|---|
| MLP 32-16 (final model) | 93.06% | 0.943 |
| Logistic Regression | 92.86% | 0.941 |
| Random Forest | 92.62% | 0.938 |

The MLP is at least as good as the classical models, but the gap is small (4 to 9 beans out of 2,032) and within the normal variation of a single split, so it is not claimed to be clearly better.

**Early stopping:** training stopped after 114 epochs and the best epoch was 99.

**Training size:** each size was trained with 3 different seeds and the mean and spread were measured. Validation accuracy was about 92.1% with only 10% of the training data and about 93.5% with all of it, with a spread of at most 0.5 points between seeds, so adding more data of the same kind helps very little.

## Results

| Metric | Value |
|---|---|
| Test accuracy | **93.06%** |
| Macro F1-score | **0.943** |
| Errors on the test set | 141 of 2,032 (6.9%) |

- Bombay is classified perfectly.
- Sira is the hardest class (F1 0.879).
- The gap between training and validation accuracy at the best epoch is only 0.08 points (93.39% vs 93.31%), so there is no real overfitting.

## Error analysis

- Most mistakes are between **Dermason** and **Sira**, two beans with very similar shapes (45 Dermason predicted as Sira, 17 Sira predicted as Dermason).
- The model is less confident when it is wrong: average confidence is about 0.71 for wrong predictions and 0.95 for correct ones. A confidence threshold could flag uncertain beans for manual checking.

## Limitations and future work

- Only shape features are used. Color and texture features could separate Dermason and Sira better.
- The dataset is imbalanced. Class weights or resampling could be tested.
- Only two classical baselines were tried with default-like settings. Tuned gradient boosting (for example XGBoost) was not compared.
- Cross validation and grid search used a small number of epochs to keep the run time reasonable.
- Results can differ slightly between machines even with a fixed seed.

## Repository structure

```
.
├── dry-bean-nn-project.ipynb   # full notebook with outputs
├── requirements.txt            # Python dependencies
├── README.md
└── .gitignore
```

## How to run

```bash
git clone https://github.com/shrouk01033/DryBean.git
cd DryBean
pip install -r requirements.txt
jupyter notebook dry-bean-nn-project.ipynb
```

Run all cells from top to bottom. The notebook also writes `best_beans.weights.h5` (best checkpoint) and the final saved model while it runs.

## Tools

Python, TensorFlow / Keras, scikit-learn, SciKeras, pandas, NumPy, Matplotlib.

## Reference

Koklu, M. and Ozkan, I. A. (2020). Multiclass classification of dry beans using computer vision and machine learning techniques. *Computers and Electronics in Agriculture*, 174, 105507.
