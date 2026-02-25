# Glucose Biosensor ML

> An interactive R Shiny dashboard for glucose concentration prediction from biosensor deflection time-series data, featuring a full machine learning pipeline with nine regression models and four explainable AI (XAI) methods.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [App Structure](#app-structure)
- [Models](#models)
- [Explainable AI Methods](#explainable-ai-methods)
- [Configurable Parameters](#configurable-parameters)
- [Installation](#installation)
- [Usage](#usage)
- [Input Data Format](#input-data-format)
- [Output & Downloads](#output--downloads)
- [Dependencies](#dependencies)
- [Notes](#notes)

---

## Overview

This application implements a complete machine learning pipeline for predicting glucose concentration (mg/dL) from cantilever biosensor deflection time-series measurements. The pipeline covers:

1. Data upload and exploratory data analysis
2. Automated feature extraction (21 features) from three-phase curve characterisation (baseline, response, plateau)
3. Feature selection via Pearson correlation ranking
4. Training and cross-validation of nine regression models
5. Comprehensive model comparison
6. Four XAI methods for model interpretability
7. Interactive prediction with user-defined feature values

All plots are exportable at **600 DPI** and all tables are exportable as **CSV**.

---

## Features

- **No coding required** — fully interactive GUI built with `shinydashboard`
- **Three CV strategies** — LOOCV, k-Fold CV, Train/Test Split (all user-configurable)
- **29 tunable hyperparameters** — every model parameter exposed to the UI
- **Nine ML models** — whitebox and blackbox regressors trained and compared simultaneously
- **Four XAI methods** — SHAP, Permutation Importance, PDP, LIME
- **Prediction tab** — enter feature values manually and get predictions from any or all models
- **All outputs downloadable** — every plot (600 DPI PNG) and table (CSV)

---

## App Structure

| Tab | Description |
|-----|-------------|
| **Data Upload & EDA** | Upload CSV, view data summary, time-series plots (all groups + faceted) |
| **Feature Extraction** | 21-feature matrix, feature-glucose correlation bar chart, calibration curve |
| **Training Settings** | Configure all hyperparameters, CV strategy, seed, top-N features |
| **Model Training & Results** | Run all models, view LOOCV metrics table, model comparison plots, scatter plots, decision tree |
| **XAI - SHAP** | SHAP bar and beeswarm plots for Random Forest and XGBoost |
| **XAI - Permutation** | Permutation importance comparison across RF, SVM, and GBM |
| **XAI - PDP** | Partial dependence plots (1D and 2D interaction) for top SHAP features |
| **XAI - LIME** | Local interpretable model-agnostic explanations for all training samples |
| **RF Variable Importance** | Classic Random Forest % increase in MSE importance chart |
| **Prediction** | Manual feature input with median pre-fill, single-model result display, all-model comparison bar chart |

---

## Models

### Whitebox
| Model | Package |
|-------|---------|
| Linear Regression (LM) | base R |
| Decision Tree (DT) | `rpart` |

### Blackbox
| Model | Package |
|-------|---------|
| Random Forest (RF) | `randomForest` |
| Support Vector Machine (SVM, RBF kernel) | `e1071` |
| XGBoost | `xgboost` |
| K-Nearest Neighbours (KNN) | `FNN` |
| Neural Network (single hidden layer) | `nnet` |
| Gradient Boosting Machine (GBM) | `gbm` |
| Gaussian Process Regression (GPR, RBF kernel) | `kernlab` |

---

## Explainable AI Methods

| Method | Scope | Models |
|--------|-------|--------|
| **SHAP** (via `kernelshap`) | Global + local | Random Forest, XGBoost |
| **Permutation Importance** (via `iml`) | Global | RF, SVM, GBM |
| **Partial Dependence Plots** (via `pdp`) | Global | Random Forest (top 2 SHAP features, 1D + 2D) |
| **LIME** (via `lime`) | Local | Random Forest |

---

## Configurable Parameters

All parameters are set from the **Training Settings** tab before clicking *Run All Models*.

### General
| Parameter | Default | Description |
|-----------|---------|-------------|
| Random Seed | 42 | Reproducibility seed for all stochastic operations |
| Top N Features | 6 | Number of features selected by \|Pearson r\| ranking |
| CV Strategy | LOOCV | Leave-One-Out CV / k-Fold CV / Train-Test Split |
| k (folds) | 5 | Number of folds (k-Fold only) |
| Train ratio | 0.75 | Fraction of data for training (Train-Test only) |

### Model Hyperparameters
| Model | Parameters |
|-------|-----------|
| Random Forest | `ntree`, `mtry` (0 = auto p/3) |
| XGBoost | `nrounds`, `max_depth`, `eta`, `subsample`, `colsample_bytree` |
| SVM | `cost`, `epsilon` |
| KNN | `k` |
| Neural Network | hidden units, weight decay, max iterations |
| GBM | `n.trees`, `interaction.depth`, `shrinkage`, `n.minobsinnode` |
| GPR | noise variance |
| Decision Tree | `maxdepth`, `minsplit`, `cp` |

### XAI Parameters
| Parameter | Default |
|-----------|---------|
| LIME n_bins | 4 |
| LIME n_permutations | 2000 |
| Permutation repetitions | 50 |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/glucose-biosensor-ml.git
cd glucose-biosensor-ml
```

### 2. Install R dependencies

Open R or RStudio and run:

```r
install.packages(c(
  "shiny", "shinydashboard", "shinyjs", "DT",
  "tidyverse", "hms", "Metrics",
  "randomForest", "e1071", "xgboost",
  "rpart", "rpart.plot",
  "FNN", "nnet", "gbm", "kernlab",
  "ggplot2", "gridExtra", "pracma", "moments",
  "kernelshap", "shapviz",
  "iml", "pdp", "lime",
  "future"
))
```

### 3. Launch the app

```r
shiny::runApp("app.R")
```

Or from the terminal:

```bash
Rscript -e "shiny::runApp('app.R')"
```

---

## Usage

1. **Upload Data** — Go to *Data Upload & EDA* and upload your CSV file.
2. **Inspect Features** — Navigate to *Feature Extraction* to review the 21 extracted features and their correlation with glucose levels.
3. **Configure Settings** — Go to *Training Settings* and adjust CV strategy, hyperparameters, and XAI options as needed.
4. **Train Models** — Go to *Model Training & Results* and click **Run All Models**. Training progress is shown in a progress bar.
5. **Explore XAI** — Navigate through the four XAI tabs to interpret model behaviour.
6. **Predict** — Go to the *Prediction* tab, adjust feature values (pre-filled with training medians), select a model, and click **Predict**.

---

## Input Data Format

The app expects a **CSV file** with exactly three columns in this order:

| Column | Format | Example |
|--------|--------|---------|
| Time | `HH:MM:SS` | `00:01:30` |
| Deflection | numeric | `-0.00452` |
| Glucose | numeric (mg/dL) | `100` |

Column headers are not required — the app assigns them automatically. Each row is one time-point measurement. Multiple glucose concentrations should be stacked in the same file; the `Glucose` column acts as the group identifier.

**Example:**

```
00:00:00,-0.00102,50
00:00:01,-0.00098,50
...
00:00:00,-0.00215,100
00:00:01,-0.00211,100
```

---

## Output & Downloads

Every result in the app can be downloaded directly from its panel:

| Output | Format | Resolution |
|--------|--------|-----------|
| Time-series plots | PNG | 600 DPI |
| Feature correlation plot | PNG | 600 DPI |
| Calibration curve | PNG | 600 DPI |
| Model comparison plot | PNG | 600 DPI |
| LOOCV scatter plot | PNG | 600 DPI |
| Decision tree plot | PNG | 600 DPI |
| SHAP bar & beeswarm plots (RF + XGBoost) | PNG | 600 DPI |
| Permutation importance plot | PNG | 600 DPI |
| PDP plots (1D + 2D) | PNG | 600 DPI |
| LIME explanation plot | PNG | 600 DPI |
| RF variable importance plot | PNG | 600 DPI |
| Feature matrix | CSV | — |
| Regression metrics | CSV | — |
| Prediction results (all models) | CSV | — |

---

## Dependencies

| Package | Version (tested) | Role |
|---------|-----------------|------|
| shiny | >= 1.7 | Web framework |
| shinydashboard | >= 0.7 | Dashboard UI |
| shinyjs | >= 2.1 | JS helpers |
| DT | >= 0.28 | Interactive tables |
| tidyverse | >= 2.0 | Data wrangling |
| hms | >= 1.1 | Time parsing |
| Metrics | >= 0.1.4 | RMSE / MAE |
| randomForest | >= 4.7 | Random Forest |
| e1071 | >= 1.7 | SVM |
| xgboost | >= 1.7 | XGBoost |
| rpart / rpart.plot | >= 4.1 / 3.1 | Decision Tree |
| FNN | >= 1.1 | KNN |
| nnet | >= 7.3 | Neural Network |
| gbm | >= 2.1 | GBM |
| kernlab | >= 0.9 | GPR |
| pracma | >= 2.4 | Trapz integration |
| moments | >= 0.14 | Skewness / Kurtosis |
| kernelshap | >= 0.3 | Model-agnostic SHAP |
| shapviz | >= 0.9 | SHAP visualisation |
| iml | >= 0.11 | Permutation importance |
| pdp | >= 0.8 | Partial dependence |
| lime | >= 0.5 | LIME explanations |
| future | >= 1.33 | Sequential plan (memory management) |

---

## Notes

- **Small sample sizes:** The pipeline is designed for small-n biosensor datasets (e.g. n = 8). LOOCV is strongly recommended over Train/Test Split for datasets with fewer than 20 samples, as random splitting with small n produces highly variable and potentially misleading results.
- **Memory management:** SHAP computation via `kernelshap` and permutation importance via `iml` are forced to run sequentially (`future::plan("sequential")`) to prevent large model objects from exceeding R's parallel worker memory limits (default 500 MiB). This adds no overhead for small datasets.
- **Scaling:** Models that require standardised input (LM, KNN, Neural Network, GPR) are trained on z-scored features. The Prediction tab applies the same scaling automatically using training-set mean and standard deviation.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
