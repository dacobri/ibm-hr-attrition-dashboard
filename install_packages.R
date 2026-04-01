# ============================================================================
# Install All Required Packages
# ============================================================================
# Run this script once before executing any project scripts.
# It installs every package used across the EDA, ML, and Shiny dashboard.
# ============================================================================

install.packages(c(
  # --- EDA (Final_project_EDA.R) ---
  "tidyverse",        # dplyr, ggplot2, tidyr, purrr, readr, stringr, forcats
  "corrplot",         # correlation matrix visualization
  "scales",           # axis formatting

  # --- Machine Learning (Final_project_ML.R) ---
  "tidymodels",       # recipes, workflows, tune, yardstick, rsample, parsnip
  "themis",           # SMOTE (class imbalance handling)
  "vip",              # variable importance plots
  "probably",         # threshold tuning
  "xgboost",         # gradient boosting engine
  "brulee",           # MLP neural network (torch backend)
  "doParallel",       # parallel cross-validation
  "parallel",         # detectCores()

  # --- Shiny Dashboard (app.R) ---
  "shiny",            # web application framework
  "bslib",            # Bootstrap 5 theming
  "plotly",           # interactive charts
  "DT",               # interactive data tables
  "shinycssloaders"   # loading spinners
))
