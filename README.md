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
- **Samples:** 13,611 beans (68 duplicate rows were kept)
- **Features:** 16 numeric shape features, for example Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness and ShapeFactor1 to 4
- **Target:** 7 classes: Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira
- **Missing values:** none
- **Class balance:** imbalanced, so the classification report (precision, recall, F1) is used next to accuracy
- The dataset file `Dry_Bean_Dataset.xlsx` is included in the repository. If it is missing, the notebook downloads it automatically on the first run (internet connection needed).

## Project workflow

1. **Exploratory data analysis:** class distribution, correlation analysis (15 of 120 feature pairs have a correlation above 0.95, for example Area and ConvexArea) and class separation per feature
2. **Preprocessing:** stratified 70/15/15 train/validation/test split and standardization fitted on the training set only, to avoid data leakage
3. **Baseline model:** a small 16-8 network used as the reference
4. **Topology comparison:** five network sizes trained and compared
5. **Regularization:** dropout, early stopping and model checkpointing
6. **Model selection:** 5-fold stratified cross validation and grid search with SciKeras and scikit-learn
7. **Final evaluation:** one evaluation on the held-out test set
8. **Diagnostics:** learning curves, learning-curve vs. training-size study and error analysis
9. **Saving:** the final model is saved in the Keras format

A fixed random seed (7) is used everywhere for reproducibility.

## Model

| Part | Choice |
|---|---|
| Input | 16 standardized features |
| Hidden layers | Dense layers with ReLU (best CV setting: 32-16) |
| Output | Dense(7) with softmax |
| Loss | Sparse categorical cross entropy |
| Optimizer | RMSprop (best from grid search) |
| Batch size | 64 |
| Regularization | Early stopping and checkpointing; dropout was tested |

## Experiments

**Topologies (validation accuracy):** all five topologies reached about 92% (between 91.8% and 92.5%). Bigger networks fit the training set better but did not improve validation accuracy, so a small network is enough for this data.

**5-fold cross validation:**

| Network | CV accuracy |
|---|---|
| 16-8 | 92.37% |
| 32-16 | 92.53% |
| 64-32 with dropout | 92.76% |

**Grid search:** the best setting was RMSprop, no dropout, batch size 64 (about 92.6% CV accuracy).

**Early stopping:** training stopped after 65 epochs and the best epoch was 50.

**Training size:** validation accuracy was about 91.3% with only 10% of the training data and stayed around 92% after that, so adding more data of the same kind helps very little.

## Results

| Metric | Value |
|---|---|
| Test accuracy | **93.49%** |
| Macro F1-score | **0.946** |
| Errors on the test set | 133 of 2,042 (6.5%) |

- Bombay is classified perfectly.
- Sira is the hardest class (F1 0.887).
- The gap between training and validation accuracy is small (about 1.3 points), so overfitting is mild.

## Error analysis

- Most mistakes are between **Dermason** and **Sira**, two beans with very similar shapes (39 Dermason predicted as Sira, 21 Sira predicted as Dermason).
- The model is less confident when it is wrong: average confidence is about 0.70 for wrong predictions and 0.95 for correct ones. A confidence threshold could flag uncertain beans for manual checking.

## Limitations and future work

- Only shape features are used. Color and texture features could separate Dermason and Sira better.
- The dataset is imbalanced. Class weights or resampling could be tested.
- Other model families (gradient boosting, for example) could be compared with the MLP.
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
git clone <your-repository-url>
cd <repository-folder>
pip install -r requirements.txt
jupyter notebook dry-bean-nn-project.ipynb
```

Run all cells from top to bottom. The notebook also writes `best_beans.weights.h5` (best checkpoint) and the final saved model while it runs.

## Tools

Python, TensorFlow / Keras, scikit-learn, SciKeras, pandas, NumPy, Matplotlib.

## Reference

Koklu, M. and Ozkan, I. A. (2020). Multiclass classification of dry beans using computer vision and machine learning techniques. *Computers and Electronics in Agriculture*, 174, 105507.
