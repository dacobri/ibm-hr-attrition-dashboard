# IBM HR Attrition — Predictive Analytics Dashboard

An end-to-end machine learning pipeline that predicts employee attrition using the IBM HR Analytics dataset, from exploratory analysis through model training to an interactive Shiny dashboard that lets HR professionals assess individual employee risk in real time.

**Final project for Data Analytics with R** (Prof. Rubén Coca, ESADE MIBA 2026)
**Group 13:** Brice Da Costa · Baran Erdogan · Mats Hoffmann · María Mora · Hiroaki Nakano

🔗 **Live dashboard:** https://hr-attrition-group13.shinyapps.io/hr-attrition/  

---

## Project Overview

Employee attrition costs organizations 50–200% of an employee's annual salary in replacement and lost productivity. This project builds a predictive system that identifies which employees are most likely to leave, translating statistical patterns into actionable HR interventions.

The pipeline has three stages, each implemented as a standalone R script:

1. **Exploratory Data Analysis** — Statistical profiling of 1,470 employees across 35 variables. Identifies the strongest attrition drivers (overtime, job role, compensation, tenure), validates data quality, flags noise variables, and prepares a clean 31-column dataset for modeling.

2. **Machine Learning** — Trains and compares five classification models (Logistic Regression, Decision Tree, Random Forest, XGBoost, MLP Neural Network) using tidymodels. Handles the 84/16% class imbalance with SMOTE, engineers four composite features inside the recipe, and optimizes the classification threshold using Youden's J-index. The best model is selected automatically by cross-validated ROC-AUC.

3. **Shiny Dashboard** — A four-tab interactive application built with bslib (Bootstrap 5) and plotly. Provides filtered EDA visualizations, a variable importance and risk analysis view, a single-employee prediction tool with actionable HR recommendations, and a full model comparison panel with ROC curves, precision-recall analysis, and confusion matrix interpretation.

### Dataset

The [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) dataset is a synthetic dataset created by IBM data scientists for educational purposes. It contains 1,470 employee records with demographics, job attributes, satisfaction scores, and a binary attrition flag. Three variables are constants (EmployeeCount, Over18, StandardHours) and three rate variables (DailyRate, HourlyRate, MonthlyRate) are confirmed random noise — both findings are documented in the EDA.

---

## Key Results

### Model Performance (5-fold × 3-repeat Cross-Validation)

| Model | ROC-AUC | Accuracy | Sensitivity | Specificity | Precision |
|---|---|---|---|---|---|
| **Gradient Boosting (XGBoost)** | **0.837** | 0.878 | 0.465 | 0.957 | 0.679 |
| Random Forest | 0.833 | 0.872 | 0.321 | 0.977 | 0.727 |
| Logistic Regression | 0.829 | 0.800 | 0.665 | 0.826 | 0.423 |
| MLP Neural Network | 0.808 | 0.844 | 0.568 | 0.896 | 0.516 |
| Decision Tree | 0.729 | 0.811 | 0.473 | 0.875 | 0.430 |

### Best Model on Held-Out Test Set (30% holdout)

XGBoost at the J-index-optimized threshold of **32%**:

| Metric | Value |
|---|---|
| ROC-AUC | 0.781 |
| Sensitivity | 56.9% |
| Specificity | 88.9% |
| Precision | 50.0% |
| F1 Score | 0.532 |

The threshold was lowered from the default 50% to 32% because the cost asymmetry in HR favors catching more leavers at the expense of some false alarms — a retention conversation with a satisfied employee is low-cost, but silently losing a departing employee incurs full replacement costs.

### Top Attrition Drivers

The XGBoost model's variable importance (gain-based) identifies overtime, monthly income, job satisfaction, age, and total working years as the strongest predictors. These are operationalized in the dashboard's prediction tab, which maps each flagged risk factor to a specific HR action (e.g., overtime flagged → review workload distribution; below-market income → compensation review).

---

## Folder Structure

```
Final_Project_Group_13/
├── DATA/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Raw dataset (1,470 × 35)
│   └── df_ml.csv                                 # Cleaned dataset (1,470 × 31)
├── MODEL/
│   ├── final_fit.rds                             # Trained XGBoost workflow (recipe + model)
│   └── best_threshold.rds                        # Optimal threshold (0.32)
├── OUTPUT/
│   ├── model_comparison_roc_auc.csv              # CV ROC-AUC for all 5 models
│   ├── cv_metrics_all_wide.csv                   # Full CV metrics (6 metrics × 5 models)
│   ├── final_test_metrics.csv                    # Test-set performance at optimal threshold
│   ├── threshold_curve.csv                       # Sensitivity/specificity/J-index by threshold
│   ├── best_hyperparameters.csv                  # Tuned hyperparameters per model
│   └── EDA/                                      # Saved plots and tables from EDA (generated on run)
├── Final_project_EDA.R                           # Stage 1: Exploratory Data Analysis (~1,200 lines)
├── Final_project_ML.R                            # Stage 2: Model training & evaluation (~1,010 lines)
├── app.R                                         # Stage 3: Shiny dashboard (~1,720 lines)
└── README.md
```

---

## How to Reproduce

### Prerequisites

R ≥ 4.2 and the following packages:

```r
install.packages(c(
  "tidyverse", "tidymodels", "corrplot", "scales",         # EDA + ML
  "themis", "vip", "probably", "xgboost", "brulee",        # ML-specific
  "doParallel",                                             # Parallel CV
  "shiny", "bslib", "plotly", "DT", "shinycssloaders"      # Dashboard
))
```

### Step-by-step

1. **Clone or download** this repository and open `Final_Project_Group_13/` as your R working directory (or create an `.Rproj` in this folder).

2. **Run the EDA script** (≈ 2 minutes):
   ```r
   source("Final_project_EDA.R")
   ```
   This reads the raw CSV from `DATA/`, runs all statistical analyses, saves plots and tables to `OUTPUT/EDA/`, and writes the cleaned `DATA/df_ml.csv` (31 columns).

3. **Run the ML script** (≈ 15–30 minutes depending on hardware):
   ```r
   source("Final_project_ML.R")
   ```
   This reads `DATA/df_ml.csv`, trains all five models with cross-validation, exports metrics to `OUTPUT/`, and saves the best model to `MODEL/`. Parallel processing is enabled automatically.

4. **Launch the dashboard**:
   ```r
   shiny::runApp("app.R")
   ```
   The app loads `DATA/df_ml.csv`, `MODEL/final_fit.rds`, `MODEL/best_threshold.rds`, and the CSV files in `OUTPUT/`. All four tabs should render immediately.

> **Note:** The repository ships with pre-computed `MODEL/` and `OUTPUT/` files, so you can skip steps 2–3 and go straight to step 4 if you only want to explore the dashboard. Re-running the scripts will regenerate these files (results may vary slightly due to stochastic elements in SMOTE and cross-validation).

---

## Technical Highlights

### Machine Learning Pipeline

- **Unified tidymodels recipe** shared across all five models — preprocessing is applied inside the workflow, preventing data leakage between CV folds
- **Feature engineering via `step_mutate`**: AvgSatisfaction (composite of 4 Likert scales), TenureRatio (company tenure / career length), PromotionStagnation (years since promotion / tenure), IncomePerLevel (pay normalized by seniority)
- **Recipe order enforced**: `step_mutate` → `step_nzv` → `step_dummy` → `step_normalize` → `step_smote` (SMOTE last ensures synthetic points are created in the normalized feature space)
- **SMOTE with `over_ratio = 0.8`** — upsamples the minority class to 80% of the majority, avoiding overfitting to synthetic points while correcting enough imbalance to prevent "always predict No"
- **5-fold × 3-repeat stratified CV** (15 total fits per hyperparameter combination) for stable metric estimates given only ~163 positive training cases per fold
- **Automated model selection** by highest mean CV ROC-AUC — no manual cherry-picking
- **J-index threshold optimization** on the test set, with full sensitivity/specificity/J-index curve exported for the dashboard

### Shiny Dashboard

- **IBM Carbon Design System** visual theme with IBM Plex Sans typography and the full Carbon color palette
- **Reactive EDA filters** with enforced Department → JobRole dependencies (impossible combinations are prevented, not just filtered)
- **Single-employee prediction tool** that accepts a 30-field employee profile, runs it through the saved workflow (recipe + model), and returns a probability with risk tier classification (Low / Medium / High)
- **Employee-specific risk factor analysis** — compares the input profile against population statistics to identify which specific attributes drive risk for that individual
- **Actionable HR recommendations** — maps each flagged risk factor to a concrete intervention (compensation review, workload audit, mentorship program, etc.)
- **ROC curve with boundary points** — adds (0,0) and (1,1) to the threshold_perf() output so the curve spans the full range
- **Confusion matrix with HR interpretation** — translates TP/FP/FN/TN into plain-language cost implications

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | R 4.x |
| Data manipulation | tidyverse (dplyr, ggplot2, tidyr, purrr, stringr) |
| ML framework | tidymodels (recipes, workflows, tune, yardstick, rsample, parsnip) |
| Models | glm, rpart, ranger, xgboost, brulee (torch backend) |
| Class imbalance | themis (SMOTE) |
| Threshold tuning | probably |
| Dashboard | Shiny + bslib (Bootstrap 5) + plotly + DT |
| Parallelization | doParallel |

---

## Limitations

- **Synthetic data** — The IBM dataset was generated for educational purposes. Patterns may not transfer to real organizations without retraining on actual employee data.
- **No temporal dimension** — All features are cross-sectional snapshots. The model predicts *whether* an employee might leave, not *when*. Survival analysis would address this.
- **No causal inference** — The model identifies correlations (e.g., overtime correlates with attrition) but cannot prove causation. Interventions should be guided by domain knowledge, not model coefficients alone.
- **Threshold optimized on test set** — Introduces mild optimistic bias in reported sensitivity/specificity. A three-way split was not feasible given the dataset size.
- **No fairness audit** — Model performance was not evaluated across demographic subgroups. A production deployment would require this.

---

## License

This project was created for academic purposes at ESADE Business School. The IBM HR Analytics dataset is publicly available under open terms on Kaggle.
