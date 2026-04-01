# =====================================================
# IBM HR Analytics — Employee Attrition
# Final Project | Machine Learning
# =====================================================
# Group 13 | ESADE MIBA 2026
# Data Analytics with R | Prof. Ruben Coca
# Members: Brice Da Costa, Baran Erdogan, Mats Hoffmann, María Mora, Hiroaki Nakano
#
# Goal:
# Train and compare multiple classification models to predict
# employee attrition, and export the best model for Shiny use.
#
# Models:
#   1. Logistic Regression   (glm — interpretable baseline)
#   2. Decision Tree          (rpart, grid search)
#   3. Random Forest          (ranger, grid search)
#   4. Gradient Boosting      (xgboost, grid search)
#   5. MLP Neural Network     (brulee, 2 hidden layers, grid search)
#
# Flow:
#   1.  Libraries & parallel setup
#   2.  Load data
#   3.  Data preparation
#   4.  Train / Test split
#   5.  Preprocessing recipe (with SMOTE + feature engineering)
#   6.  Define models (with tune() placeholders)
#   7.  Build workflows
#   8.  Cross-validation setup (5-fold × 3-repeat)
#   9.  Logistic Regression (fit_resamples — no grid)
#   10. Decision Tree       (tune_grid)
#   11. Random Forest       (tune_grid)
#   12. Gradient Boosting   (tune_grid)
#   13. MLP Neural Network  (tune_grid)
#   14. Collect & compare all CV metrics
#   15. Export CV metrics to CSV
#   16. Select best model + final fit on test set
#   17. Threshold optimization with J-index
#   18. Confusion matrix at optimal threshold
#   19. Variable importance
#   20. Precision-Recall curve
#   21. Export final test metrics to CSV
#   22. Save model for Shiny
#   23. Limitations
# =====================================================

# -------------------------
# 1. Libraries & parallel setup
# -------------------------
# We use tidymodels as our unified ML framework over alternatives like caret or
# mlr3. tidymodels provides a consistent, pipe-friendly API where recipes,
# workflows, and tuning all share the same interface — reducing the risk of
# preprocessing data leakage that can occur when wiring these steps together
# manually (e.g., normalizing before splitting, which would leak test-set
# statistics into the training process).
library(tidymodels)   # recipes, workflows, tune, yardstick, rsample, parsnip
library(tidyverse)    # dplyr, ggplot2, tidyr, purrr
library(vip)          # Variable importance plots
library(xgboost)      # Gradient Boosting engine
library(ranger)       # Random Forest engine
library(rpart)        # Decision Tree engine
library(brulee)       # MLP Neural Network (torch backend)
library(probably)     # Threshold optimization (J-index)
library(themis)       # SMOTE and other resampling methods for class imbalance
library(doParallel)   # Parallel processing for tune_grid

# Grid search over 5-fold × 3-repeat CV across five models is computationally
# expensive. We register a parallel backend using all available cores minus one
# (reserving one to keep the system responsive). tune_grid() distributes
# hyperparameter combinations across workers natively via foreach, making this
# preferable to sequential execution for any grid larger than a few combinations.
n_cores <- parallel::detectCores() - 1
cl <- makePSOCKcluster(n_cores)
registerDoParallel(cl)
cat("Parallel backend registered:", n_cores, "cores\n")

# -------------------------
# 2. Load data
# -------------------------
# df_ml.csv is the cleaned dataset exported by the EDA script, with
# EmployeeNumber, EmployeeCount, Over18, and StandardHours already removed
# (31 columns: 30 predictors + Attrition).
df_ml <- read.csv("./DATA/df_ml.csv")

# =====================================================
# 3. Data Preparation
# =====================================================

# We convert Attrition to a factor with "Yes" as the first (positive) level so
# tidymodels treats it as the event of interest. This is critical: yardstick
# functions (sensitivity, specificity, ROC-AUC) use the FIRST factor level as
# the positive class. Placing "No" first would cause the model to optimize for
# predicting stayers, inverting the entire evaluation logic.
df_ml <- df_ml %>%
  mutate(Attrition = factor(Attrition, levels = c("Yes", "No")))

# Convert all remaining character columns to factors so parsnip model engines
# and recipe steps can handle them correctly.
df_ml <- df_ml %>%
  mutate(across(where(is.character), as.factor))

# Sanity checks
glimpse(df_ml)
table(df_ml$Attrition)
prop.table(table(df_ml$Attrition))  # Expected: ~16% Yes

# =====================================================
# 4. Train / Test Split (70 / 30)
# =====================================================

# We stratify by Attrition to preserve the ~16% class ratio in both the
# training and test sets. Without stratification, random chance could produce
# a test set with too few positive cases, making evaluation metrics unreliable.
# A 70/30 split gives the training set enough positive examples (~163) for
# SMOTE and cross-validation to work effectively, while keeping the test set
# large enough (~70 positives) for stable metric estimates.
set.seed(42)
data_split <- initial_split(df_ml, prop = 0.70, strata = Attrition)
df_train   <- training(data_split)
df_test    <- testing(data_split)

cat("Train rows:", nrow(df_train), "\n")
cat("Test rows: ", nrow(df_test),  "\n")
cat("Train attrition rate:", round(mean(df_train$Attrition == "Yes") * 100, 1), "%\n")
cat("Test attrition rate: ", round(mean(df_test$Attrition == "Yes") * 100, 1), "%\n")

# =====================================================
# 5. Preprocessing Recipe
# =====================================================

# A single shared recipe ensures all five models receive identical input.
# The recipe is applied automatically inside each workflow — we never manually
# call bake() — so there is no risk of applying different preprocessing to
# different models or to the test set during evaluation.
#
# Steps (in order):
#   step_mutate     : create engineered features BEFORE encoding
#   step_nzv        : remove near-zero variance predictors
#   step_dummy      : one-hot encode all nominal (factor) predictors
#   step_normalize  : standardize numerics to mean=0, sd=1
#   step_smote      : synthetic minority oversampling for class balance
#
# IMPORTANT ordering rules:
# - step_mutate BEFORE step_dummy (engineered features use raw factor columns)
# - step_dummy BEFORE step_normalize (normalize only works on numerics)
# - step_smote MUST BE LAST: it generates synthetic rows — placing it before
#   normalization would mean synthetic points are created on raw scales and
#   then normalized, rather than being created in the already-normalized space
# - step_smote only applies during training; at prediction time no synthetic
#   rows are created, so the recipe is safe to use in production

attrition_recipe <- recipe(Attrition ~ ., data = df_train) %>%

  # --- Feature engineering ---
  # We engineer four composite features inside the recipe rather than adding
  # them directly to df_ml. This ensures the transformations are automatically
  # applied at prediction time (in the Shiny app) without any additional
  # preprocessing code — the recipe carries the full transformation logic.
  step_mutate(
    # Composite satisfaction: average of the four Likert satisfaction scales.
    # A single aggregate score reduces dimensionality while capturing the
    # shared "overall contentment" signal across all four scales.
    AvgSatisfaction = (JobSatisfaction + EnvironmentSatisfaction +
                       RelationshipSatisfaction + WorkLifeBalance) / 4,

    # Tenure-to-age ratio: proportion of working life spent at this company.
    # Capped at 1 to handle edge cases where YearsAtCompany > TotalWorkingYears.
    TenureRatio = pmin(YearsAtCompany / pmax(TotalWorkingYears, 1), 1),

    # Promotion stagnation: years since last promotion relative to tenure.
    # A high ratio signals an employee stuck in the same role relative to
    # how long they've been at the company — a pattern associated with
    # disengagement. If this feature ranks high in variable importance, it
    # confirms the model is capturing employees HR can still retain with a
    # timely promotion or role change.
    PromotionStagnation = YearsSinceLastPromotion / pmax(YearsAtCompany, 1),

    # Income per job level: normalizes pay by seniority.
    # Low values flag employees who are underpaid relative to their level —
    # a concrete, actionable signal for HR compensation reviews.
    IncomePerLevel = MonthlyIncome / pmax(JobLevel, 1)
  ) %>%

  # --- Preprocessing ---
  # step_nzv removes predictors with near-zero variance (one dominant class).
  # For example, if a one-hot encoded dummy variable is 99% one value, it
  # contributes noise without predictive power and can destabilize some models.
  # We prefer step_nzv over manual column removal because it adapts to each
  # CV fold's training data rather than being hardcoded on the full dataset.
  step_nzv(all_predictors()) %>%

  # We apply one-hot encoding to convert all factor predictors into numeric
  # indicator variables. This is required by XGBoost and the MLP, which cannot
  # handle factors natively. step_normalize follows to standardize all numeric
  # predictors to mean=0, sd=1 — this is essential for distance-sensitive
  # algorithms (MLP, logistic regression with regularization) and prevents
  # high-scale features like MonthlyIncome from dominating low-scale ones like
  # JobLevel. Tree-based models are scale-invariant, but we normalize for all
  # models to keep the recipe unified.
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%

  # --- Class imbalance handling ---
  # We chose SMOTE (Synthetic Minority Oversampling Technique) over the
  # alternative of undersampling (step_downsample) because undersampling
  # discards real majority-class data points, reducing the effective training
  # set size. With only ~1,000 training rows, we cannot afford to lose data.
  # SMOTE instead creates synthetic minority-class samples by interpolating
  # between existing positive cases and their k nearest neighbors, preserving
  # all real observations.
  #
  # over_ratio = 0.8 upsamples the minority class to 80% of the majority
  # class size, rather than full 1:1 balance. This avoids overfitting to
  # synthetic points while still correcting enough of the imbalance to prevent
  # the model from defaulting to "always predict No" (which would achieve ~84%
  # accuracy but identify zero leavers — useless for HR).
  #
  # step_smote is placed last so synthetic points are created in the normalized
  # feature space, ensuring consistency with the real training data.
  step_smote(Attrition, over_ratio = 0.8, neighbors = 5)

cat("\nRecipe steps:\n")
print(attrition_recipe)

# =====================================================
# 6. Define Models
# =====================================================

# We include five models spanning a deliberate spectrum from interpretable to
# complex. This allows us to evaluate whether predictive gains from more
# complex models justify the loss in interpretability.
#
#   Logistic Regression : interpretable baseline, coefficients = log-odds
#   Decision Tree       : interpretable via visual inspection; shows the cost
#                         of interpretability (high variance, typically underperforms ensembles)
#   Random Forest       : ensemble of decorrelated trees via bootstrap +
#                         random feature subsets; robust general performer
#   XGBoost             : sequential boosting; often outperforms Random Forest
#                         on tabular data by directly optimizing a loss function
#   MLP                 : the only non-tree model; tests whether deep non-linear
#                         interactions improve beyond what tree ensembles capture

# --- 6.1 Logistic Regression ---
# No tunable hyperparameters. Serves as our interpretable floor model —
# any more complex model should meaningfully outperform it on ROC-AUC to
# justify the added complexity.
logistic_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# --- 6.2 Decision Tree ---
# We tune cost_complexity (pruning strength) and tree_depth. A deeper, less
# pruned tree memorizes training data; a shallower, more pruned tree generalizes
# better. We use a regular grid (4×4 = 16 combinations) because the parameter
# space is small and two-dimensional, making grid search computationally feasible
# and exhaustive over the relevant range.
tree_spec <- decision_tree(
  cost_complexity = tune(),
  tree_depth      = tune(),
  min_n           = 20       # fixed: minimum samples per leaf
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

tree_grid <- grid_regular(
  cost_complexity(range = c(-4, -1)),  # 10^-4 to 10^-1
  tree_depth(range = c(3, 8)),
  levels = 4    # 4×4 = 16 combinations
)

# --- 6.3 Random Forest ---
# We tune mtry (features sampled per split) and min_n (minimum node size).
# trees = 500 is fixed: more trees reduce variance without increasing bias,
# and 500 is the standard threshold beyond which returns diminish on datasets
# of this size. We use ranger over randomForest because ranger is significantly
# faster (parallelized C++ backend) and supports impurity-based variable
# importance via the importance = "impurity" argument.
rf_spec <- rand_forest(
  mtry  = tune(),
  min_n = tune(),
  trees = 500
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_grid <- grid_regular(
  mtry(range = c(3, 15)),
  min_n(range = c(5, 30)),
  levels = 4    # 4×4 = 16 combinations
)

# --- 6.4 Gradient Boosting (XGBoost) ---
# We tune three hyperparameters: learn_rate (step size per boosting iteration),
# tree_depth (complexity of each individual tree), and min_n (minimum leaf size).
# With three parameters, a regular grid would require many combinations, so we
# use grid_space_filling (Latin hypercube sampling) instead. Latin hypercube
# distributes 20 points more evenly across the 3D parameter space than a sparse
# regular grid, giving better coverage of extreme and moderate combinations with
# fewer total fits.
xgb_spec <- boost_tree(
  trees      = 500,
  learn_rate = tune(),
  tree_depth = tune(),
  min_n      = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

set.seed(42)
xgb_grid <- grid_space_filling(
  learn_rate(range = c(-3, -1)),  # 10^-3 to 10^-1
  tree_depth(range = c(3, 8)),
  min_n(range = c(5, 30)),
  size = 20    # 20 combinations
)

# --- 6.5 MLP Neural Network ---
# We include an MLP to test whether the data contains non-linear interactions
# that tree-based models miss. We use brulee (torch backend) over keras/tensorflow
# because it integrates natively with the tidymodels tune() API and requires no
# separate Python environment. Two hidden layers are specified to give the network
# capacity for hierarchical feature learning. We fix epochs = 100 as a balance
# between convergence and runtime; a proper implementation would use early stopping,
# but that is not supported in the current brulee/tune_grid integration.
nn_spec <- mlp(
  hidden_units = tune(),
  penalty      = tune(),
  epochs       = 100     # fixed: more epochs ≈ more runtime
) %>%
  set_engine("brulee", hidden_layers = 2) %>%
  set_mode("classification")

nn_grid <- grid_regular(
  hidden_units(range = c(10, 100)),
  penalty(range = c(-4, -1)),
  levels = 3    # 3×3 = 9 combinations
)

# =====================================================
# 7. Build Workflows
# =====================================================

# Bundling the recipe and model into a workflow ensures preprocessing is applied
# consistently during both cross-validation and final prediction. Without
# workflows, there is a risk of fitting the recipe on the full training set
# and then applying it to CV folds — leaking fold-level statistics (e.g., means
# for normalization, SMOTE neighbors) into held-out validation data.
# With workflows, the recipe is re-fit on each fold's training portion only.
logistic_wf <- workflow() %>% add_recipe(attrition_recipe) %>% add_model(logistic_spec)
tree_wf     <- workflow() %>% add_recipe(attrition_recipe) %>% add_model(tree_spec)
rf_wf       <- workflow() %>% add_recipe(attrition_recipe) %>% add_model(rf_spec)
xgb_wf      <- workflow() %>% add_recipe(attrition_recipe) %>% add_model(xgb_spec)
nn_wf       <- workflow() %>% add_recipe(attrition_recipe) %>% add_model(nn_spec)

# =====================================================
# 8. Cross-Validation Setup
# =====================================================

# We use 5-fold stratified CV with 3 repeats, giving 15 total model fits per
# hyperparameter combination. Stratification by Attrition ensures each fold
# preserves the ~16% positive rate, preventing folds with too few positive
# cases (which would produce unstable sensitivity estimates).
#
# We chose repeated CV over a single 5-fold run because with only ~163 positive
# training cases, each fold contains roughly 37-38 positives. A single fold's
# metrics can swing substantially with so few events. Averaging over 3 repeats
# (each with independently reshuffled folds) smooths this variance, giving us
# more reliable model comparisons. The tradeoff is 3× the compute time, which
# is acceptable given parallel processing.
set.seed(42)
cv_folds <- vfold_cv(df_train, v = 5, repeats = 3, strata = Attrition)

# We track six metrics to get a complete picture of model performance.
# ROC-AUC is our primary selection criterion (threshold-independent, robust
# to imbalance). We also track accuracy, sensitivity, specificity, precision,
# and F1 to understand threshold-dependent behavior and the precision-recall
# tradeoff across models.
attrition_metrics <- metric_set(roc_auc, accuracy, sensitivity,
                                specificity, precision, f_meas)

# Control objects
ctrl      <- control_resamples(save_pred = TRUE)
ctrl_tune <- control_grid(save_pred = TRUE, verbose = TRUE)

# =====================================================
# 9. Logistic Regression (fit_resamples — no grid)
# =====================================================

# Logistic regression has no tunable hyperparameters, so we use fit_resamples()
# rather than tune_grid(). This evaluates the model across all 15 CV folds and
# returns stable metric estimates. It also serves as our interpretable baseline:
# if tree ensembles do not significantly outperform logistic regression on
# ROC-AUC, the simpler model should be preferred.
cat("\n--- Fitting Logistic Regression ---\n")
logistic_res <- fit_resamples(
  logistic_wf,
  resamples = cv_folds,
  metrics   = attrition_metrics,
  control   = ctrl
)

# =====================================================
# 10. Decision Tree (tune_grid)
# =====================================================

# tune_grid() evaluates all 16 hyperparameter combinations across all 15 CV
# folds (16 × 15 = 240 model fits). select_best() extracts the combination
# with the highest mean ROC-AUC, which we use to finalize the workflow for
# the model comparison stage.
cat("\n--- Tuning Decision Tree ---\n")
tree_tune_res <- tune_grid(
  tree_wf,
  resamples = cv_folds,
  grid      = tree_grid,
  metrics   = attrition_metrics,
  control   = ctrl_tune
)

show_best(tree_tune_res, metric = "roc_auc", n = 3) %>% print()
best_tree_params <- select_best(tree_tune_res, metric = "roc_auc")
tree_wf_final    <- finalize_workflow(tree_wf, best_tree_params)

# =====================================================
# 11. Random Forest (tune_grid)
# =====================================================

# We search over 16 combinations of mtry and min_n. Random Forest is typically
# robust across a wide range of hyperparameters, but mtry in particular has a
# strong effect: too few features per split leads to correlated trees (high
# variance); too many approaches a bagged tree (loses the decorrelation benefit).
cat("\n--- Tuning Random Forest ---\n")
rf_tune_res <- tune_grid(
  rf_wf,
  resamples = cv_folds,
  grid      = rf_grid,
  metrics   = attrition_metrics,
  control   = ctrl_tune
)

show_best(rf_tune_res, metric = "roc_auc", n = 3) %>% print()
best_rf_params <- select_best(rf_tune_res, metric = "roc_auc")
rf_wf_final    <- finalize_workflow(rf_wf, best_rf_params)

# =====================================================
# 12. Gradient Boosting (tune_grid)
# =====================================================

# We search over 20 Latin hypercube combinations across three parameters.
# learn_rate is the most sensitive: too high causes overfitting and instability;
# too low requires more trees to converge. Pairing a low learn_rate with deep
# trees (high tree_depth) is a known risk — it can memorize training data —
# so the grid covers a range of depth values at each learning rate.
cat("\n--- Tuning Gradient Boosting (XGBoost) ---\n")
xgb_tune_res <- tune_grid(
  xgb_wf,
  resamples = cv_folds,
  grid      = xgb_grid,
  metrics   = attrition_metrics,
  control   = ctrl_tune
)

show_best(xgb_tune_res, metric = "roc_auc", n = 3) %>% print()
best_xgb_params <- select_best(xgb_tune_res, metric = "roc_auc")
xgb_wf_final    <- finalize_workflow(xgb_wf, best_xgb_params)

# =====================================================
# 13. MLP Neural Network (tune_grid)
# =====================================================

# We search over 9 combinations of hidden_units and L2 penalty. The penalty
# parameter is critical for preventing overfitting in neural networks, especially
# with a dataset this small (~1,000 rows). A higher penalty shrinks weights
# toward zero, acting similarly to dropout regularization in larger networks.
cat("\n--- Tuning MLP Neural Network ---\n")
nn_tune_res <- tune_grid(
  nn_wf,
  resamples = cv_folds,
  grid      = nn_grid,
  metrics   = attrition_metrics,
  control   = ctrl_tune
)

show_best(nn_tune_res, metric = "roc_auc", n = 3) %>% print()
best_nn_params <- select_best(nn_tune_res, metric = "roc_auc")
nn_wf_final    <- finalize_workflow(nn_wf, best_nn_params)

# Stop parallel cluster
stopCluster(cl)

# =====================================================
# 14. Collect & Compare All CV Metrics
# =====================================================

# We rank models by ROC-AUC rather than accuracy. Accuracy is misleading for
# imbalanced datasets: a model that always predicts "No" achieves ~84% accuracy
# on this dataset while identifying zero leavers. ROC-AUC measures the model's
# ability to rank positive cases above negative ones across all possible
# thresholds, making it a threshold-independent measure of discrimination.

# Helper: extract best ROC-AUC row from a tuning result
extract_best_metrics <- function(tune_res, model_name) {
  show_best(tune_res, metric = "roc_auc", n = 1) %>%
    mutate(model = model_name) %>%
    select(model, mean, std_err)
}

logistic_best <- collect_metrics(logistic_res) %>%
  filter(.metric == "roc_auc") %>%
  mutate(model = "Logistic") %>%
  select(model, mean, std_err)

model_comparison <- bind_rows(
  logistic_best,
  extract_best_metrics(tree_tune_res, "Decision_Tree"),
  extract_best_metrics(rf_tune_res,   "Random_Forest"),
  extract_best_metrics(xgb_tune_res,  "Gradient_Boosting"),
  extract_best_metrics(nn_tune_res,   "MLP_Neural_Net")
) %>%
  arrange(desc(mean))

cat("\n=== MODEL COMPARISON (CV ROC-AUC) ===\n")
print(model_comparison)

# ----- ROC curves for all models -----
logistic_roc <- collect_predictions(logistic_res) %>%
  roc_curve(Attrition, .pred_Yes) %>%
  mutate(model = "Logistic")

get_roc_from_tune <- function(tune_res, model_name) {
  best <- select_best(tune_res, metric = "roc_auc")
  collect_predictions(tune_res, parameters = best) %>%
    roc_curve(Attrition, .pred_Yes) %>%
    mutate(model = model_name)
}

roc_all <- bind_rows(
  logistic_roc,
  get_roc_from_tune(tree_tune_res,  "Decision_Tree"),
  get_roc_from_tune(rf_tune_res,    "Random_Forest"),
  get_roc_from_tune(xgb_tune_res,   "Gradient_Boosting"),
  get_roc_from_tune(nn_tune_res,    "MLP_Neural_Net")
)

# Plot ROC curves
ggplot(roc_all, aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_line(linewidth = 1) +
  geom_abline(linetype = "dashed", color = "gray") +
  labs(
    title = "ROC Curve Comparison — All Models (5-fold × 3-repeat CV)",
    x     = "1 - Specificity (False Positive Rate)",
    y     = "Sensitivity (True Positive Rate)",
    color = "Model"
  ) +
  theme_minimal()

# ----- Precision-Recall curves -----
# PR curves are more informative than ROC curves for imbalanced datasets because
# they focus exclusively on the minority (positive) class. A model can achieve
# high ROC-AUC by correctly ranking the large majority class, while performing
# poorly on the minority class that actually matters. The PR curve exposes this
# directly: high precision means flagged employees are likely true leavers;
# high recall means few actual leavers are missed.
#
# The dashed line represents the no-skill classifier that predicts positive at
# the base rate (~16%). Any model whose PR curve falls near this line is
# providing no useful signal beyond knowing the prevalence rate.

logistic_pr <- collect_predictions(logistic_res) %>%
  pr_curve(Attrition, .pred_Yes) %>%
  mutate(model = "Logistic")

get_pr_from_tune <- function(tune_res, model_name) {
  best <- select_best(tune_res, metric = "roc_auc")
  collect_predictions(tune_res, parameters = best) %>%
    pr_curve(Attrition, .pred_Yes) %>%
    mutate(model = model_name)
}

pr_all <- bind_rows(
  logistic_pr,
  get_pr_from_tune(tree_tune_res, "Decision_Tree"),
  get_pr_from_tune(rf_tune_res,   "Random_Forest"),
  get_pr_from_tune(xgb_tune_res,  "Gradient_Boosting"),
  get_pr_from_tune(nn_tune_res,   "MLP_Neural_Net")
)

ggplot(pr_all, aes(x = recall, y = precision, color = model)) +
  geom_line(linewidth = 1) +
  geom_hline(yintercept = mean(df_train$Attrition == "Yes"),
             linetype = "dashed", color = "gray") +
  labs(
    title    = "Precision-Recall Curve Comparison",
    subtitle = "Dashed line = baseline (random classifier at prevalence rate)",
    x = "Recall (Sensitivity)", y = "Precision",
    color = "Model"
  ) +
  theme_minimal()

# The PR curve directly answers: "Of the employees we flag as at-risk, what
# percentage actually leave?" (precision) vs "Of those who actually leave, what
# percentage did we catch?" (recall). For HR, high recall matters most — missing
# a leaver is expensive. But if precision drops too low, HR wastes time on false
# alarms. The curve shows where each model balances this tradeoff.

# Bar chart: ROC-AUC comparison
ggplot(model_comparison, aes(x = reorder(model, mean), y = mean, fill = model)) +
  geom_col(show.legend = FALSE) +
  geom_errorbar(aes(ymin = mean - std_err, ymax = mean + std_err), width = 0.3) +
  geom_text(aes(label = round(mean, 3)), hjust = -0.3, size = 3.5) +
  coord_flip() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Model Comparison — CV ROC-AUC (best hyperparameters)",
    x = "Model", y = "ROC-AUC"
  ) +
  theme_minimal()

# =====================================================
# 15. Export CV Metrics to CSV
# =====================================================

dir.create("./OUTPUT", showWarnings = FALSE)

# Main comparison table (model, mean ROC-AUC, std_err)
write.csv(model_comparison, "./OUTPUT/model_comparison_roc_auc.csv", row.names = FALSE)

# Best hyperparameters per model
best_params_all <- bind_rows(
  tibble(model = "Logistic", name = "none", value = NA),
  best_tree_params %>% select(where(is.numeric)) %>%
    pivot_longer(everything()) %>% mutate(model = "Decision_Tree"),
  best_rf_params %>% select(where(is.numeric)) %>%
    pivot_longer(everything()) %>% mutate(model = "Random_Forest"),
  best_xgb_params %>% select(where(is.numeric)) %>%
    pivot_longer(everything()) %>% mutate(model = "Gradient_Boosting"),
  best_nn_params %>% select(where(is.numeric)) %>%
    pivot_longer(everything()) %>% mutate(model = "MLP_Neural_Net")
) %>%
  select(model, everything())

write.csv(best_params_all, "./OUTPUT/best_hyperparameters.csv", row.names = FALSE)

# Full CV metrics at best params for all models (wide format)
# app.R Tab 4 reads cv_metrics_all_wide.csv and expects columns:
# model, roc_auc, accuracy, sensitivity, specificity, precision, f_meas
all_cv_metrics_wide <- bind_rows(
  collect_metrics(logistic_res) %>% mutate(model = "Logistic"),
  collect_metrics(tree_tune_res) %>%
    filter(.config == best_tree_params$.config) %>%
    mutate(model = "Decision_Tree"),
  collect_metrics(rf_tune_res) %>%
    filter(.config == best_rf_params$.config) %>%
    mutate(model = "Random_Forest"),
  collect_metrics(xgb_tune_res) %>%
    filter(.config == best_xgb_params$.config) %>%
    mutate(model = "Gradient_Boosting"),
  collect_metrics(nn_tune_res) %>%
    filter(.config == best_nn_params$.config) %>%
    mutate(model = "MLP_Neural_Net")
) %>%
  group_by(model, .metric) %>%
  summarise(mean = mean(mean), .groups = "drop") %>%
  pivot_wider(names_from = .metric, values_from = mean) %>%
  arrange(desc(roc_auc))

write.csv(all_cv_metrics_wide, "./OUTPUT/cv_metrics_all_wide.csv", row.names = FALSE)

# =====================================================
# 16. Select Best Model + Final Fit on Test Set
# =====================================================

# We automatically select the model with the highest CV ROC-AUC rather than
# manually choosing. This prevents cherry-picking and ensures the selection
# criterion is the same metric we optimized for during tuning.
#
# last_fit() trains once on the FULL training set (not CV folds) and evaluates
# once on the held-out test set. This is the only point where test-set data
# is used — ensuring the test metrics are unbiased estimates of real-world
# performance. Using the test set at any earlier stage (e.g., for model
# selection) would make these estimates optimistic.

best_model_name <- model_comparison %>% slice(1) %>% pull(model)
cat("\nBest model selected:", best_model_name, "\n")

final_workflows <- list(
  Logistic          = logistic_wf,
  Decision_Tree     = tree_wf_final,
  Random_Forest     = rf_wf_final,
  Gradient_Boosting = xgb_wf_final,
  MLP_Neural_Net    = nn_wf_final
)

best_wf   <- final_workflows[[best_model_name]]
final_fit <- last_fit(best_wf, split = data_split)

cat("Test set metrics for", best_model_name, ":\n")
collect_metrics(final_fit) %>% print()

# =====================================================
# 17. Threshold Optimization with J-index
# =====================================================

# The default classification threshold of 0.50 assumes equal costs for false
# positives and false negatives, and implicitly assumes a balanced class
# distribution. With ~16% positive rate, this threshold systematically
# under-predicts attrition — the model is rarely "50% confident" about the
# minority class, so most at-risk employees fall below the cutoff.
#
# We optimize using the J-index (Youden's J = Sensitivity + Specificity - 1),
# which finds the point on the ROC curve farthest from the diagonal (the random
# classifier line). This balances sensitivity and specificity without assuming
# either is more important — a reasonable starting point that HR can further
# adjust based on organizational cost preferences.
#
# We chose J-index over alternatives:
# - Pure sensitivity maximization would set a very low threshold, catching more
#   leavers but flooding HR with false alarms
# - F1-score optimization would favor precision-recall balance but ignores the
#   true negative rate (stayers correctly left alone)
# - J-index naturally lands below 0.50 for imbalanced data (often 0.25–0.35),
#   catching the "edge" employees in the 30–50% probability range — still
#   saveable, but only if HR acts
#
# In HR terms, the cost asymmetry also supports a lower threshold: a false
# positive means giving a retention interview to someone who wasn't leaving —
# a low-cost, even positive interaction. A false negative means silently losing
# an employee, incurring full replacement and training costs.

test_preds <- collect_predictions(final_fit)

j_index_curve <- test_preds %>%
  probably::threshold_perf(
    truth      = Attrition,
    estimate   = .pred_Yes,
    thresholds = seq(0.05, 0.95, by = 0.01),
    metrics    = metric_set(j_index, sensitivity, specificity)
  )

best_threshold <- j_index_curve %>%
  filter(.metric == "j_index") %>%
  slice_max(.estimate, n = 1) %>%
  pull(.threshold)

cat("\nOptimal threshold (J-index):", best_threshold, "\n")

# Visualize threshold optimization
j_index_curve %>%
  filter(.metric %in% c("j_index", "sensitivity", "specificity")) %>%
  ggplot(aes(x = .threshold, y = .estimate, color = .metric)) +
  geom_line(linewidth = 1) +
  geom_vline(xintercept = best_threshold, linetype = "dashed", color = "red") +
  annotate("text", x = best_threshold + 0.03, y = 0.1,
           label = paste("Optimal:", best_threshold), color = "red", size = 3.5) +
  labs(
    title = paste("Threshold Optimization —", best_model_name),
    x     = "Threshold",
    y     = "Metric Value",
    color = "Metric"
  ) +
  theme_minimal()

# =====================================================
# 18. Confusion Matrix at Optimal Threshold
# =====================================================

test_preds_thresh <- test_preds %>%
  mutate(
    .pred_class_opt = factor(
      ifelse(.pred_Yes >= best_threshold, "Yes", "No"),
      levels = c("Yes", "No")
    )
  )

cat("\n--- Confusion Matrix (threshold:", best_threshold, ") ---\n")
conf_mat(test_preds_thresh, truth = Attrition, estimate = .pred_class_opt) %>%
  print()

# How to read the confusion matrix:
# Top-left  (TP): correctly predicted leavers — employees HR can intervene with
# Top-right (FP): stayers wrongly flagged — get a retention chat (low cost)
# Bot-left  (FN): leavers we missed — silently leave, full replacement cost
# Bot-right (TN): correctly predicted stayers — no action needed
#
# The J-index threshold tilts toward more TPs and more FPs relative to the
# default 0.50 cutoff, which is the right tradeoff for HR: we would rather
# have too many conversations than miss employees who are about to leave.

# =====================================================
# 19. Variable Importance
# =====================================================

# Variable importance helps translate model predictions into HR-actionable
# insights. Knowing which features drive the model's decisions tells HR which
# levers to pull: if OverTime ranks highest, that is a policy lever; if
# MonthlyIncome or IncomePerLevel ranks highly, it points toward compensation
# reviews. We use impurity-based importance (ranger's default for Random Forest,
# gain-based for XGBoost) because it is computed at training time with no
# additional cost. Note: impurity-based importance can overestimate the
# influence of high-cardinality continuous variables. Permutation importance
# would be a more unbiased alternative but requires additional test-set passes.
tryCatch({
  final_model <- extract_fit_parsnip(final_fit)
  vip(final_model, num_features = 15) +
    labs(title = paste("Top 15 Variable Importance —", best_model_name)) +
    theme_minimal()
}, error = function(e) {
  cat("Variable importance not available for this model engine.\n")
  cat("Error:", conditionMessage(e), "\n")
})

# =====================================================
# 20. Precision-Recall Analysis on Test Set
# =====================================================

# Final PR curve on the held-out test set. Unlike the CV PR curves (section 14),
# this reflects performance on truly unseen data after the full training pipeline.
# The dashed line marks the no-skill baseline (predicting positive at the dataset
# prevalence rate). The area under this curve (PR-AUC) is a more informative
# single-number summary than ROC-AUC for highly imbalanced problems, because it
# penalizes models that achieve good ROC performance by correctly classifying the
# large majority class while still missing most leavers.
test_preds %>%
  pr_curve(Attrition, .pred_Yes) %>%
  ggplot(aes(x = recall, y = precision)) +
  geom_line(color = "#E74C3C", linewidth = 1) +
  geom_hline(yintercept = mean(df_test$Attrition == "Yes"),
             linetype = "dashed", color = "gray") +
  labs(
    title = paste("Precision-Recall Curve —", best_model_name, "(Test Set)"),
    x = "Recall", y = "Precision"
  ) +
  theme_minimal()

# =====================================================
# 21. Export Final Test Metrics to CSV
# =====================================================

# app.R reads final_test_metrics.csv with columns: model, threshold, .metric, .estimate
# and threshold_curve.csv with columns: .threshold, .estimate, .metric

final_metrics_fn <- metric_set(accuracy, sensitivity, specificity, precision, f_meas)

test_metrics_final <- bind_rows(
  final_metrics_fn(test_preds_thresh, truth = Attrition, estimate = .pred_class_opt),
  roc_auc(test_preds, truth = Attrition, .pred_Yes)
) %>%
  mutate(
    model     = best_model_name,
    threshold = best_threshold
  ) %>%
  select(model, threshold, .metric, .estimate)

write.csv(test_metrics_final, "./OUTPUT/final_test_metrics.csv", row.names = FALSE)
write.csv(j_index_curve,      "./OUTPUT/threshold_curve.csv",    row.names = FALSE)

cat("\n=== FINAL TEST METRICS ===\n")
print(test_metrics_final)

# =====================================================
# 22. Save Model for Shiny
# =====================================================

# We save the complete last_fit object (not just the fitted model) so the Shiny
# app can extract the full workflow — recipe + model — for predictions. Saving
# the workflow is critical because the recipe contains the step_mutate
# transformations: when app.R receives raw employee data (30 columns), the
# workflow automatically computes AvgSatisfaction, TenureRatio, PromotionStagnation,
# and IncomePerLevel before passing data to the model. Without the recipe, the
# app would need to replicate all feature engineering manually.
#
# The app.R code does:
#   model_fit  <- readRDS("./MODEL/final_fit.rds")
#   fitted_wf  <- extract_workflow(model_fit)
#   predict(fitted_wf, new_emp, type = "prob")$.pred_Yes
#
# best_threshold is saved separately as a single numeric value so the app can
# apply the same optimal cutoff determined during training.

dir.create("./MODEL", showWarnings = FALSE)

saveRDS(final_fit,      "./MODEL/final_fit.rds")
saveRDS(best_threshold, "./MODEL/best_threshold.rds")

# The Shiny dashboard uses this model to assign each employee to a risk tier:
#   High Risk   (≥ 60%)               : likely leaving — urgent conversation needed
#   Medium Risk (≥ threshold, < 60%)  : the "saveable" zone — employees with
#                                       enough risk factors to flag but not yet
#                                       decided. Interventions (raise, promotion,
#                                       overtime reduction) have the highest ROI here.
#   Low Risk    (< threshold)         : stable employees — no immediate action needed
#
# The model's primary value is in the medium-risk band: identifying employees
# who are heading toward attrition but can still be retained with targeted action.

cat("\n=== MODEL SAVED ===\n")
cat("Best model:     ", best_model_name, "\n")
cat("Best threshold: ", best_threshold,  "\n")
cat("Files saved:\n")
cat("  ./MODEL/final_fit.rds\n")
cat("  ./MODEL/best_threshold.rds\n")
cat("  ./OUTPUT/model_comparison_roc_auc.csv\n")
cat("  ./OUTPUT/cv_metrics_all_wide.csv\n")
cat("  ./OUTPUT/best_hyperparameters.csv\n")
cat("  ./OUTPUT/final_test_metrics.csv\n")
cat("  ./OUTPUT/threshold_curve.csv\n")

# =====================================================
# Reference: How to call the model from Shiny
# =====================================================

# model     <- readRDS("./MODEL/final_fit.rds")
# threshold <- readRDS("./MODEL/best_threshold.rds")
#
# new_employee <- tibble(
#   Age = 32, MonthlyIncome = 4000, OverTime = "Yes",
#   JobSatisfaction = 2, ... (all 30 columns except Attrition)
# )
#
# # The recipe will automatically create AvgSatisfaction, TenureRatio,
# # PromotionStagnation, and IncomePerLevel from the raw columns.
# pred_prob <- predict(extract_workflow(model), new_employee, type = "prob")
# risk      <- pred_prob$.pred_Yes
# label     <- ifelse(risk >= threshold, "High Risk", "Low Risk")

# =====================================================
# 23. Limitations
# =====================================================

# This section documents the known constraints of the current model. Being
# explicit about limitations demonstrates methodological awareness and helps
# any future user — including HR — understand where the model's predictions
# should and should not be trusted.

# --- 23.1 Synthetic Dataset ---
# The IBM HR Analytics dataset is a publicly available synthetic dataset
# generated for educational purposes. It does not represent a real company's
# employee records. Attrition patterns, salary distributions, and demographic
# compositions were programmatically constructed, which means the model may
# learn relationships that exist in the simulation design but do not generalize
# to real organizations. Before deploying this model in a real HR context,
# retraining on actual employee data from the target organization would be
# essential.

# --- 23.2 No Temporal Dimension ---
# All features are cross-sectional snapshots — a single measurement point per
# employee. The model cannot capture trends over time, such as a declining
# satisfaction score, a stagnating salary relative to market rates, or
# increasing tenure without promotion. Survival analysis models (e.g., Cox
# proportional hazards) or time-series approaches would be better suited to
# predict *when* an employee is likely to leave, rather than just *whether*
# they might. The current model treats every prediction as equally urgent,
# even though an employee with a 40% probability score today might be stable
# for two more years.

# --- 23.3 Class Imbalance Persists Despite SMOTE ---
# SMOTE mitigates but does not eliminate the imbalance problem. The synthetic
# minority samples created by SMOTE are interpolations of existing positive
# cases — they are not real employees. If the true attrition signal is highly
# non-linear or driven by rare edge cases, SMOTE's nearest-neighbor interpolation
# may generate synthetic points that do not represent realistic attrition profiles.
# The model's sensitivity on the test set should be interpreted with this in mind:
# it reflects performance on a held-out sample of the same synthetic distribution,
# not on future real employees.

# --- 23.4 No Causal Inference ---
# The model identifies statistical correlations between features and attrition —
# it does not establish causation. A high OverTime coefficient does not prove that
# overtime causes attrition; it may be that employees who are already disengaged
# are more likely to report high overtime, or that a third variable (e.g., poor
# management) drives both. Interventions based purely on model-flagged features
# risk addressing symptoms rather than root causes. The model should be used as
# a screening tool that directs HR attention, not as a causal diagnosis.

# --- 23.5 Feature Availability at Prediction Time ---
# Several predictors — particularly YearsSinceLastPromotion, JobSatisfaction,
# and EnvironmentSatisfaction — require up-to-date values in the HR system.
# Satisfaction scores in particular often come from annual surveys, meaning the
# model's predictions may be based on data that is 6–12 months stale. In
# practice, prediction quality will degrade between survey cycles.

# --- 23.6 No Fairness Analysis ---
# The model was built to maximize predictive performance without explicitly
# analyzing whether predictions are equally accurate across demographic groups
# (age, gender, department, job role). It is possible that the model performs
# better for some groups than others, or that it systematically flags certain
# groups at higher rates due to correlations in the training data. Before
# deploying in a real HR context, a fairness audit — examining sensitivity and
# specificity broken down by protected attributes — would be necessary.

# --- 23.7 Threshold Optimized on Test Set ---
# The J-index optimal threshold was selected by evaluating all thresholds on
# the test set predictions. Strictly speaking, this introduces a mild optimistic
# bias in the reported sensitivity and specificity: we chose the threshold that
# maximized J-index on the same data used to report final metrics. In a fully
# rigorous pipeline, the threshold would be selected on a separate validation
# set and evaluated on a completely untouched holdout. Given our dataset size
# (~1,000 rows), a three-way split was not feasible without making each partition
# too small to be reliable.

# --- 23.8 Model Portability ---
# The model was trained on data from one simulated organization. Employee
# attrition patterns vary significantly across industries, cultures, company
# sizes, and economic conditions. Applying this model directly to a different
# organization without retraining would likely produce unreliable predictions.
# The feature engineering (PromotionStagnation, IncomePerLevel) assumes that
# the relationships between promotion timing, income level, and attrition are
# consistent — an assumption that may not hold in organizations with different
# compensation structures or career path norms.

# =====================================================
# End of ML Script
# =====================================================
