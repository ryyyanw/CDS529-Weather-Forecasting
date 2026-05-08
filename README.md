# CDS529-Weather-Forecasting
# WiDS Datathon 2023 — Temperature Prediction

This repository contains multiple approaches to solve the [WiDS Datathon 2023](https://www.kaggle.com/c/widsdatathon2023) challenge: **predicting average temperature 14 days out** (`contest-tmp2m-14d__tmp2m`). We explore linear models, tree-based ensembles, and gradient boosting with hyperparameter tuning. A comprehensive **exploratory data analysis (EDA)** is included to inform feature engineering and model design.

## Data
The dataset consists of historical weather variables, climate model forecasts, and geographic coordinates. The training set contains **375,734 samples** and **246 columns** (including the target). After memory optimization (downcasting data types), the data occupies about 352 MB.

- **Target variable:** `contest-tmp2m-14d__tmp2m` (average temperature in °C, 14‑day forecast).  
  Mean: 11.86, Std: 9.87, Range: [-20.36, 37.24], approximately normal (skewness -0.23).
- **Missing values:** Only 8 features have missing data; 6.16% of rows are affected. The maximum missing rate is 4.24% (e.g., NMME model forecasts `ccsm30`). Detailed missing‑value analysis and CSV exports are provided in the EDA notebook.
- **Features:** after EDA‑guided preprocessing and feature importance analysis, **13 features** were selected for modeling, including NMME forecasts, wind components, surface pressure, and latitude.

## Exploratory Data Analysis (EDA)
The notebook `EDA.ipynb` performs a thorough initial investigation of the training data:

- **Data summarization:** shape, memory usage, data types, and basic statistics.
- **Target distribution:** histograms, box plots, Q‑Q plots, skewness/kurtosis.
- **Temporal analysis:** time‑series plots by month/quarter, moving averages, annual trends (using `startdate`).
- **Geographical analysis:** scatter plots, 2D heatmaps, and density maps using `lat`/`lon` coordinates.
- **Missing value patterns:** bar charts of missing percentages, pattern matrix, and row‑wise missing counts.
- **Correlation analysis:** top features correlated with the target, scatter plots for the strongest predictors.
- **Data type distribution:** visual summary of column types.

The EDA generates several figures (e.g., `target_distribution.png`, `time_series_analysis.png`) and a comprehensive summary report (`data_summary_report.txt`). All findings are used to guide downstream feature engineering and modeling decisions.

## Methods

| Notebook | Models Used | Highlights |
|----------|-------------|------------|
| `EDA.ipynb` | – | Complete data overview, missing value analysis, visualizations |
| `LinearRegression.ipynb` | Linear Regression, **Ridge (CV)**, **Lasso (CV)** | Baseline; Lasso achieved best performance with α=0.117 |
| `RandomForest.ipynb` | Random Forest (300 trees, max_depth=40) | Feature selection via importance; top-13 features retained |
| `XGBoost.ipynb` | **XGBoost** + **LightGBM** ensemble | Optuna hyperparameter tuning (30 trials); SHAP analysis; weighted ensemble |

All models were evaluated using RMSE and R² on a time‑based validation split (first 80% of the dataset for training, last 20% for validation).

## Key Results
- The **XGBoost + LightGBM ensemble** gave the strongest validation performance.
- **Lasso** provided a simple, interpretable alternative with a validation R² around **0.901**.
- Random Forest achieved an R² of **0.953** on the validation set (with carefully selected hyperparameters).

## Repository Structure
├── EDA.ipynb # Exploratory data analysis

├── LinearRegression.ipynb # Linear regression, Ridge, Lasso

├── RandomForest.ipynb # Random Forest model & feature importance

├── XGBoost.ipynb # XGBoost + LightGBM ensemble, SHAP, Optuna tuning

├── README.md

## How to Run
1. Install required packages:
   ```bash
   pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost lightgbm optuna shap
2. Download the competition data and update the paths in the notebooks (e.g., ./train_data.csv).

3. Start with EDA.ipynb to understand the data and generate preliminary visualizations.

4. Execute the modeling notebooks to reproduce the results. All steps are fully documented.

Acknowledgements
Data source: WiDS Datathon 2023
Inspired by the community solutions and best practices in climate forecasting.
