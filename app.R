# =====================================================
# IBM HR Analytics — Employee Attrition
# Final Project | Shiny Dashboard
# =====================================================
# Group 13 | ESADE MIBA 2026
# Data Analytics with R | Prof. Ruben Coca
# Members: Brice Da Costa, Baran Erdogan, Mats Hoffmann, María Mora, Hiroaki Nakano
# Tabs:
#   1  EDA             — filtered charts + KPI cards
#   2  Risk Analysis   — variable importance + correlation heatmap
#   3  Prediction      — employee profile -> attrition probability gauge
#   4  Model Comparison— metrics tables + ROC curve + confusion matrix
#
# Dependencies (install if needed):
#   install.packages(c("shiny","bslib","tidyverse","tidymodels",
#                      "plotly","DT","shinycssloaders","vip","scales"))
# =====================================================


# -------------------------
# 1. Libraries
# -------------------------
library(xgboost)
library(shiny)
library(bslib)
library(tidyverse)
library(tidymodels)
library(plotly)
library(DT)
library(shinycssloaders)
library(vip)
library(scales)

# -------------------------
# 2. Load data & model (once at startup)
# -------------------------
df_ml <- read.csv("./DATA/df_ml.csv") %>%
  mutate(
    Attrition = factor(Attrition, levels = c("Yes", "No")),
    across(where(is.character), as.factor)
  )

model_fit  <- readRDS("./MODEL/final_fit.rds")
threshold  <- readRDS("./MODEL/best_threshold.rds")
fitted_wf  <- tryCatch(
  extract_workflow(model_fit),
  error = function(e) NULL
)

# Pre-computed outputs from ML script
cv_metrics     <- read.csv("./OUTPUT/model_comparison_roc_auc.csv")
cv_metrics_all <- read.csv("./OUTPUT/cv_metrics_all_wide.csv")
test_metrics   <- read.csv("./OUTPUT/final_test_metrics.csv")

cv_metrics_all <- cv_metrics_all %>%
  mutate(model = str_replace_all(model, "_", " "))
cv_metrics <- cv_metrics %>%
  mutate(model = str_replace_all(model, "_", " "))
test_metrics <- test_metrics %>%
  mutate(model = str_replace_all(model, "_", " "))

# Correlation matrix (numeric only)
cor_mat <- df_ml %>%
  select(where(is.numeric)) %>%
  cor()

# Variable importance (top 15)
vip_df <- tryCatch({
  vi(extract_fit_parsnip(model_fit)) %>%
    slice_max(Importance, n = 15) %>%
    arrange(Importance)
}, error = function(e) {
  tibble(Variable = "Not available", Importance = 0)
})

# Test-set predictions (for confusion matrix in Tab 4)
test_preds <- collect_predictions(model_fit) %>%
  mutate(
    .pred_class_opt = factor(
      ifelse(.pred_Yes >= threshold, "Yes", "No"),
      levels = c("Yes", "No")
    )
  )

overall_attrition_rate <- mean(df_ml$Attrition == "Yes")

HIGH_RISK_CUTOFF <- 0.6

## Used by observeEvent to constrain JobRole choices when Department changes
dept_role_map <- split(levels(df_ml$JobRole)[as.integer(df_ml$JobRole)], df_ml$Department)
dept_role_map <- lapply(dept_role_map, function(x) sort(unique(x)))

# -------------------------
# 3. Theme & shared palette — IBM Carbon Design System
# -------------------------

# IBM Carbon palette
ibm_blue_60  <- "#0f62fe"
ibm_blue_80  <- "#002d9c"
ibm_red_60   <- "#da1e28"
ibm_green_50 <- "#24a148"
ibm_yellow   <- "#f1c21b"
ibm_gray_10  <- "#f4f4f4"
ibm_gray_20  <- "#e0e0e0"
ibm_gray_70  <- "#525252"
ibm_gray_100 <- "#161616"

app_theme <- bs_theme(
  version      = 5,
  bg           = ibm_gray_10,
  fg           = ibm_gray_100,
  primary      = ibm_blue_60,
  secondary    = ibm_gray_70,
  success      = ibm_green_50,
  danger       = ibm_red_60,
  warning      = ibm_yellow,
  base_font    = font_google("IBM Plex Sans"),
  heading_font = font_google("IBM Plex Sans"),
  font_scale   = 0.95
)

pal_attr <- c("Yes" = ibm_red_60, "No" = ibm_blue_60)

# -------------------------
# 4. Helper: KPI value box
# -------------------------
kpi_box <- function(title, value, icon_name, color = ibm_blue_60) {
  div(
    class = "card border-0 shadow-sm h-100",
    style = paste0("border-left: 4px solid ", color, " !important;"),
    div(
      class = "card-body d-flex align-items-center gap-3 py-3",
      div(
        style = paste0("width:44px;height:44px;border-radius:10px;background:",
                       color, "18;display:flex;align-items:center;",
                       "justify-content:center;flex-shrink:0;"),
        icon(icon_name, style = paste0("color:", color, ";font-size:1.2rem;"))
      ),
      div(
        h4(value, class = "mb-0 fw-bold", style = paste0("color:", color)),
        p(title, class = "mb-0 text-muted small fw-semibold text-uppercase",
          style = "letter-spacing:.5px;font-size:.7rem;")
      )
    )
  )
}

# -------------------------
# 5. Shared plotly layout defaults
# -------------------------
clean_layout <- function(p, ...) {
  p %>% layout(
    plot_bgcolor  = "rgba(0,0,0,0)",
    paper_bgcolor = "rgba(0,0,0,0)",
    font          = list(family = "IBM Plex Sans, sans-serif", size = 12,
                         color = ibm_gray_100),
    margin        = list(t = 30, b = 50, l = 50, r = 30),
    ...
  )
}

## Compares an employee's inputs against population statistics to flag risk areas
get_employee_risk_factors <- function(inputs, data) {
  risk_factors <- list()

  # OverTime
  if (inputs$OverTime == "Yes") {
    ot_rate <- mean(data$Attrition[data$OverTime == "Yes"] == "Yes")
    no_ot_rate <- mean(data$Attrition[data$OverTime == "No"] == "Yes")
    risk_factors[["OverTime"]] <- list(
      label = "Works overtime",
      detail = paste0(round(ot_rate * 100), "% attrition vs ",
                      round(no_ot_rate * 100), "% without"),
      severity = "high"
    )
  }

  # MaritalStatus — Single employees have higher attrition
  if (inputs$MaritalStatus == "Single") {
    s_rate <- mean(data$Attrition[data$MaritalStatus == "Single"] == "Yes")
    risk_factors[["MaritalStatus"]] <- list(
      label = "Single (higher mobility)",
      detail = paste0(round(s_rate * 100), "% attrition rate for single employees"),
      severity = "medium"
    )
  }

  # Low JobSatisfaction (1-2 out of 4)
  if (inputs$JobSatisfaction <= 2) {
    low_rate <- mean(data$Attrition[data$JobSatisfaction <= 2] == "Yes")
    risk_factors[["JobSatisfaction"]] <- list(
      label = paste0("Low job satisfaction (", inputs$JobSatisfaction, "/4)"),
      detail = paste0(round(low_rate * 100), "% attrition when satisfaction <= 2"),
      severity = "high"
    )
  }

  # Low EnvironmentSatisfaction
  if (inputs$EnvironmentSatisfaction <= 2) {
    low_rate <- mean(data$Attrition[data$EnvironmentSatisfaction <= 2] == "Yes")
    risk_factors[["EnvSatisfaction"]] <- list(
      label = paste0("Low environment satisfaction (", inputs$EnvironmentSatisfaction, "/4)"),
      detail = paste0(round(low_rate * 100), "% attrition when satisfaction <= 2"),
      severity = "high"
    )
  }

  # Low WorkLifeBalance
  if (inputs$WorkLifeBalance <= 2) {
    low_rate <- mean(data$Attrition[data$WorkLifeBalance <= 2] == "Yes")
    risk_factors[["WorkLifeBalance"]] <- list(
      label = paste0("Poor work-life balance (", inputs$WorkLifeBalance, "/4)"),
      detail = paste0(round(low_rate * 100), "% attrition when WLB <= 2"),
      severity = "high"
    )
  }

  # Low MonthlyIncome relative to job level
  level_median <- median(data$MonthlyIncome[data$JobLevel == inputs$JobLevel])
  if (inputs$MonthlyIncome < level_median * 0.8) {
    risk_factors[["MonthlyIncome"]] <- list(
      label = paste0("Below-market income for Job Level ", inputs$JobLevel),
      detail = paste0("$", format(inputs$MonthlyIncome, big.mark = ","),
                      " vs median $", format(round(level_median), big.mark = ",")),
      severity = "high"
    )
  }

  # Years since last promotion (stagnation)
  if (inputs$YearsSinceLastPromotion >= 5 & inputs$YearsAtCompany >= 5) {
    risk_factors[["PromotionStagnation"]] <- list(
      label = paste0(inputs$YearsSinceLastPromotion, " years without promotion"),
      detail = "Long promotion gaps increase attrition risk",
      severity = "medium"
    )
  }

  # Young employee with low tenure (early career flight risk)
  if (inputs$Age <= 30 & inputs$YearsAtCompany <= 2) {
    risk_factors[["EarlyCareer"]] <- list(
      label = "Early-career employee with short tenure",
      detail = "Employees under 30 with < 2 years have elevated attrition",
      severity = "medium"
    )
  }

  # No stock options
  if (inputs$StockOptionLevel == 0) {
    rate_no_stock <- mean(data$Attrition[data$StockOptionLevel == 0] == "Yes")
    risk_factors[["StockOptions"]] <- list(
      label = "No stock options",
      detail = paste0(round(rate_no_stock * 100), "% attrition with no stock options"),
      severity = "medium"
    )
  }

  # Frequent travel
  if (inputs$BusinessTravel == "Travel_Frequently") {
    rate_freq <- mean(data$Attrition[data$BusinessTravel == "Travel_Frequently"] == "Yes")
    risk_factors[["Travel"]] <- list(
      label = "Travels frequently",
      detail = paste0(round(rate_freq * 100), "% attrition for frequent travellers"),
      severity = "medium"
    )
  }

  # Low JobInvolvement
  if (inputs$JobInvolvement <= 2) {
    low_rate <- mean(data$Attrition[data$JobInvolvement <= 2] == "Yes")
    risk_factors[["JobInvolvement"]] <- list(
      label = paste0("Low job involvement (", inputs$JobInvolvement, "/4)"),
      detail = paste0(round(low_rate * 100), "% attrition when involvement <= 2"),
      severity = "medium"
    )
  }

  risk_factors
}

get_hr_recommendations <- function(risk_factors) {
  recs <- list()

  if ("OverTime" %in% names(risk_factors)) {
    recs <- c(recs, list(list(
      icon = "clock",
      text = "Review workload and overtime patterns. Consider redistributing tasks or hiring support."
    )))
  }

  if ("JobSatisfaction" %in% names(risk_factors) ||
      "EnvSatisfaction" %in% names(risk_factors)) {
    recs <- c(recs, list(list(
      icon = "comments",
      text = "Schedule a stay interview to understand dissatisfaction drivers. Discuss growth opportunities."
    )))
  }

  if ("WorkLifeBalance" %in% names(risk_factors)) {
    recs <- c(recs, list(list(
      icon = "scale-balanced",
      text = "Explore flexible work arrangements: remote days, adjusted hours, or reduced travel."
    )))
  }

  if ("MonthlyIncome" %in% names(risk_factors)) {
    recs <- c(recs, list(list(
      icon = "dollar-sign",
      text = "Conduct a compensation review. This employee is paid below the median for their level."
    )))
  }

  if ("PromotionStagnation" %in% names(risk_factors)) {
    recs <- c(recs, list(list(
      icon = "arrow-up",
      text = "Discuss career development path. Consider a title change, lateral move, or stretch assignment."
    )))
  }

  if ("EarlyCareer" %in% names(risk_factors)) {
    recs <- c(recs, list(list(
      icon = "user-graduate",
      text = "Assign a mentor. Early-career employees benefit from structured onboarding and check-ins at 6/12/18 months."
    )))
  }

  if ("StockOptions" %in% names(risk_factors)) {
    recs <- c(recs, list(list(
      icon = "chart-line",
      text = "Consider equity participation. Stock options create long-term retention incentives."
    )))
  }

  if ("Travel" %in% names(risk_factors)) {
    recs <- c(recs, list(list(
      icon = "plane",
      text = "Review travel requirements. Explore video conferencing alternatives where possible."
    )))
  }

  if ("JobInvolvement" %in% names(risk_factors)) {
    recs <- c(recs, list(list(
      icon = "hands-helping",
      text = "Increase autonomy and ownership. Assign a meaningful project with visible impact."
    )))
  }

  # Fallback if no specific risks were flagged
  if (length(recs) == 0) {
    recs <- c(recs, list(list(
      icon = "check-circle",
      text = "No specific high-risk factors detected. Continue regular engagement and career conversations."
    )))
  }

  recs
}


# =====================================================
# 6. UI
# =====================================================
ui <- page_navbar(
  title = tags$span(
    icon("users", style = "margin-right:8px;"),
    tags$strong("IBM HR Attrition"), " \u2014 Predictive Analytics"
  ),
  theme    = app_theme,
  bg       = ibm_blue_80,
  fillable = FALSE,

  # Global CSS
  header = tags$head(tags$style(HTML(paste0("
    body { font-family: 'IBM Plex Sans', sans-serif; }
    .navbar-brand { font-size: 1rem; letter-spacing: .3px; }
    .card { border-radius: 12px; border: none;
            box-shadow: 0 1px 6px rgba(0,0,0,.08);
            background: #ffffff; }
    .card-header { background: #fff !important;
                   border-bottom: 1px solid ", ibm_gray_20, ";
                   font-weight: 600; font-size: .85rem;
                   border-radius: 12px 12px 0 0 !important;
                   color: ", ibm_gray_100, "; }
    .risk-badge { font-size:1.6rem; font-weight:700; margin:4px 0; }
    .risk-high   { color:", ibm_red_60, "; }
    .risk-medium { color:", ibm_yellow, "; }
    .risk-low    { color:", ibm_green_50, "; }
    .predict-btn { width:100%; font-weight:600; border-radius:8px;
                   padding:10px; font-size:.9rem;
                   background:", ibm_blue_60, "; border-color:", ibm_blue_60, "; }
    .predict-btn:hover { background:", ibm_blue_80, "; border-color:", ibm_blue_80, "; }
    .accordion-button { font-weight:600; font-size:.85rem; }
    .accordion-button:not(.collapsed) { background:", ibm_gray_10, ";
                                         color:", ibm_gray_100, "; }
    .filter-card { background:#fff; border-radius:12px; padding:16px 20px;
                   box-shadow:0 1px 6px rgba(0,0,0,.08); margin-bottom:16px; }
    .kpi-row { margin-bottom: 20px; }
    .summary-card { background: #ffffff; border-radius: 12px;
                    padding: 16px 24px; margin-bottom: 16px;
                    box-shadow: 0 1px 6px rgba(0,0,0,.08);
                    border-left: 4px solid ", ibm_blue_60, "; }
    .summary-card h5 { color: ", ibm_gray_100, "; font-weight: 700;
                        margin-bottom: 6px; font-size: .95rem; }
    .summary-card p  { color: ", ibm_gray_70, "; margin-bottom: 0;
                        font-size: .85rem; line-height: 1.5; }
    .risk-factor-item { padding: 8px 12px; margin-bottom: 6px;
                        border-radius: 8px; font-size: .85rem; }
    .risk-factor-high   { background: ", ibm_red_60, "12;
                          border-left: 3px solid ", ibm_red_60, "; }
    .risk-factor-medium { background: ", ibm_yellow, "20;
                          border-left: 3px solid ", ibm_yellow, "; }
    .rec-item { padding: 10px 14px; margin-bottom: 8px;
                background: ", ibm_blue_60, "08; border-radius: 8px;
                font-size: .85rem; border-left: 3px solid ", ibm_blue_60, "; }
    .rec-item i { color: ", ibm_blue_60, "; margin-right: 8px; }
    .help-text { color: ", ibm_gray_70, "; font-size: .8rem;
                 font-style: italic; margin-top: 4px; }
    .metric-explanation { background: #ffffff; border-radius: 8px;
                          padding: 14px 18px; font-size: .85rem;
                          color: ", ibm_gray_70, ";
                          border-left: 3px solid ", ibm_blue_60, "; }
  ")))),

  # ============================================================
  # TAB 1 — EDA
  # ============================================================
  nav_panel(
    title = tagList(icon("chart-bar"), " EDA"),

    div(class = "container-fluid py-3",

        uiOutput("exec_summary"),

        # KPI cards
        div(class = "row g-3 kpi-row",
            div(class = "col-6 col-md-3", uiOutput("kpi_total")),
            div(class = "col-6 col-md-3", uiOutput("kpi_rate")),
            div(class = "col-6 col-md-3", uiOutput("kpi_age")),
            div(class = "col-6 col-md-3", uiOutput("kpi_income"))
        ),

        # Filters
        div(class = "filter-card",
            div(class = "row g-3 align-items-end",
                div(class = "col-md-3",
                    selectInput("f_dept", "Department",
                                choices = c("All", levels(df_ml$Department)),
                                selected = "All")),
                div(class = "col-md-3",
                    selectInput("f_role", "Job Role",
                                choices = c("All", levels(df_ml$JobRole)),
                                selected = "All")),
                div(class = "col-md-2",
                    selectInput("f_ot", "OverTime",
                                choices = c("All", "Yes", "No"), selected = "All")),
                div(class = "col-md-3",
                    sliderInput("f_age", "Age range",
                                min = min(df_ml$Age), max = max(df_ml$Age),
                                value = c(min(df_ml$Age), max(df_ml$Age)), step = 1)),
                div(class = "col-md-1 d-flex align-items-end",
                    actionButton("btn_reset_filters", "Reset",
                                 icon = icon("rotate-left"),
                                 class = "btn btn-outline-secondary btn-sm",
                                 style = "width:100%; margin-bottom:15px;"))
            )
        ),

        # Row 1
        div(class = "row g-3 mb-3",
            div(class = "col-md-4",
                div(class = "card",
                    div(class = "card-header", "Attrition distribution"),
                    div(class = "card-body p-2",
                        withSpinner(plotlyOutput("p_dist", height = "240px"),
                                    type = 4, color = ibm_blue_60))
                )
            ),
            div(class = "col-md-4",
                div(class = "card",
                    div(class = "card-header", "Attrition rate by department"),
                    p(class = "help-text px-3 pt-1 mb-0",
                      "Dashed line = overall company rate. Bars above it indicate higher-risk groups."),
                    div(class = "card-body p-2",
                        withSpinner(plotlyOutput("p_dept", height = "240px"),
                                    type = 4, color = ibm_blue_60))
                )
            ),
            div(class = "col-md-4",
                div(class = "card",
                    div(class = "card-header", "Attrition rate by job role"),
                    p(class = "help-text px-3 pt-1 mb-0",
                      "Dashed line = overall company rate."),
                    div(class = "card-body p-2",
                        withSpinner(plotlyOutput("p_role", height = "240px"),
                                    type = 4, color = ibm_blue_60))
                )
            )
        ),

        # Row 2
        div(class = "row g-3 mb-3",
            div(class = "col-md-4",
                div(class = "card",
                    div(class = "card-header", "Monthly income by attrition"),
                    div(class = "card-body p-2",
                        withSpinner(plotlyOutput("p_income", height = "240px"),
                                    type = 4, color = ibm_blue_60))
                )
            ),
            div(class = "col-md-4",
                div(class = "card",
                    div(class = "card-header", "Age distribution by attrition"),
                    div(class = "card-body p-2",
                        withSpinner(plotlyOutput("p_age_hist", height = "240px"),
                                    type = 4, color = ibm_blue_60))
                )
            ),
            div(class = "col-md-4",
                div(class = "card",
                    div(class = "card-header", "Attrition rate by business travel"),
                    p(class = "help-text px-3 pt-1 mb-0",
                      "Dashed line = overall company rate."),
                    div(class = "card-body p-2",
                        withSpinner(plotlyOutput("p_travel", height = "240px"),
                                    type = 4, color = ibm_blue_60))
                )
            )
        ),

        div(class = "row g-3",
            div(class = "col-md-12",
                div(class = "card",
                    div(class = "card-header",
                        "Attrition rate by years at company"),
                    p(class = "help-text px-3 pt-1 mb-0",
                      "Shows when employees are most likely to leave. Key for onboarding and milestone check-in planning."),
                    div(class = "card-body p-2",
                        withSpinner(plotlyOutput("p_tenure", height = "260px"),
                                    type = 4, color = ibm_blue_60))
                )
            )
        )
    )
  ),

  # ============================================================
  # TAB 2 — RISK ANALYSIS
  # ============================================================
  nav_panel(
    title = tagList(icon("triangle-exclamation"), " Risk Analysis"),

    div(class = "container-fluid py-3",

        div(class = "row g-3 mb-3",
            div(class = "col-md-6",
                div(class = "card",
                    div(class = "card-header",
                        "Top 15 variable importance \u2014 Gradient Boosting (XGBoost)"),
                    p(class = "help-text px-3 pt-1 mb-0",
                      "Variables ranked by how much they influence the model's attrition predictions."),
                    div(class = "card-body p-2",
                        withSpinner(plotlyOutput("p_vip", height = "400px"),
                                    type = 4, color = ibm_blue_60))
                )
            ),
            div(class = "col-md-6",
                div(class = "card",
                    div(class = "card-header",
                        div(class = "d-flex justify-content-between align-items-center",
                            span("Attrition rate by category"),
                            selectInput("risk_var", NULL,
                                        choices = c("OverTime","BusinessTravel","Department",
                                                    "JobRole","MaritalStatus","Gender"),
                                        selected = "OverTime", width = "180px")
                        )
                    ),
                    p(class = "help-text px-3 pt-1 mb-0",
                      "Dashed line = overall company attrition rate. Compare groups above/below."),
                    div(class = "card-body p-2",
                        withSpinner(plotlyOutput("p_risk_cat", height = "400px"),
                                    type = 4, color = ibm_blue_60))
                )
            )
        ),

        div(class = "row g-3",
            div(class = "col-12",
                div(class = "card",
                    div(class = "card-header", "Correlation heatmap \u2014 numeric variables"),
                    div(class = "card-body p-2",
                        withSpinner(plotlyOutput("p_corr", height = "520px"),
                                    type = 4, color = ibm_blue_60))
                )
            )
        )
    )
  ),

  # ============================================================
  # TAB 3 — PREDICTION
  # ============================================================
  nav_panel(
    title = tagList(icon("user-check"), " Prediction"),

    div(class = "container-fluid py-3",
        div(class = "summary-card",
            h5("How to use this tool"),
            p("Adjust the employee profile on the left, then click 'Predict attrition risk'.",
              " The gauge shows the predicted probability of this employee leaving.",
              " Below it, you'll see their specific risk factors and suggested HR actions.",
              " A probability above the threshold (marked on the gauge) means the model",
              " flags this employee as likely to leave.")
        ),

        div(class = "row g-3",

            # LEFT: Input panel
            div(class = "col-md-5",
                div(class = "card",
                    div(class = "card-header", "Employee profile \u2014 adjust inputs"),
                    div(class = "card-body",

                        accordion(
                          open = FALSE,

                          accordion_panel(
                            title = "Personal",
                            div(class = "row g-2",
                                div(class = "col-6",
                                    numericInput("i_age", "Age", 35, 18, 60)),
                                div(class = "col-6",
                                    selectInput("i_gender", "Gender", c("Male","Female"))),
                                div(class = "col-6",
                                    selectInput("i_marital", "Marital status",
                                                c("Single","Married","Divorced"))),
                                div(class = "col-6",
                                    numericInput("i_dist", "Distance from home", 10, 1, 29)),
                                div(class = "col-6",
                                    sliderInput("i_edu", "Education (1-5)", 1, 5, 3)),
                                div(class = "col-6",
                                    selectInput("i_edu_field", "Education field",
                                                levels(df_ml$EducationField)))
                            )
                          ),

                          accordion_panel(
                            title = "Job",
                            div(class = "row g-2",
                                div(class = "col-6",
                                    selectInput("i_dept", "Department",
                                                levels(df_ml$Department))),
                                div(class = "col-6",
                                    selectInput("i_role", "Job role",
                                                levels(df_ml$JobRole))),
                                div(class = "col-6",
                                    sliderInput("i_joblevel", "Job level (1-5)", 1, 5, 2)),
                                div(class = "col-6",
                                    selectInput("i_travel", "Business travel",
                                                levels(df_ml$BusinessTravel))),
                                div(class = "col-6",
                                    selectInput("i_ot", "OverTime", c("Yes","No"))),
                                div(class = "col-6",
                                    sliderInput("i_training", "Training times last year",
                                                0, 6, 2))
                            )
                          ),

                          accordion_panel(
                            ## separate sub-section with note; kept for model compatibility
                            title = "Compensation",
                            div(class = "row g-2",
                                div(class = "col-6",
                                    numericInput("i_income", "Monthly income", 5000,
                                                 1009, 19999)),
                                div(class = "col-6",
                                    sliderInput("i_hike", "Salary hike (%)", 11, 25, 14)),
                                div(class = "col-6",
                                    sliderInput("i_stock", "Stock option level (0-3)",
                                                0, 3, 1))
                            ),
                            hr(),
                            p(class = "help-text",
                              "The fields below have low predictive power",
                              "(near-random in the data) but are included for completeness."),
                            div(class = "row g-2",
                                div(class = "col-4",
                                    numericInput("i_daily", "Daily rate", 800, 102, 1499)),
                                div(class = "col-4",
                                    numericInput("i_hourly", "Hourly rate", 65, 30, 100)),
                                div(class = "col-4",
                                    numericInput("i_monthly_rate", "Monthly rate",
                                                 14000, 2094, 26999))
                            )
                          ),

                          accordion_panel(
                            title = "Experience",
                            div(class = "row g-2",
                                div(class = "col-6",
                                    sliderInput("i_total_yrs", "Total working years",
                                                0, 40, 10)),
                                div(class = "col-6",
                                    sliderInput("i_num_co", "Num companies worked",
                                                0, 9, 2)),
                                div(class = "col-6",
                                    sliderInput("i_yrs_co", "Years at company",
                                                0, 40, 5)),
                                div(class = "col-6",
                                    sliderInput("i_yrs_role", "Years in current role",
                                                0, 18, 3)),
                                div(class = "col-6",
                                    sliderInput("i_yrs_promo",
                                                "Years since last promotion", 0, 15, 1)),
                                div(class = "col-6",
                                    sliderInput("i_yrs_mgr",
                                                "Years with current manager", 0, 17, 3))
                            )
                          ),

                          accordion_panel(
                            title = "Satisfaction",
                            div(class = "row g-2",
                                div(class = "col-6",
                                    sliderInput("i_job_sat", "Job satisfaction (1-4)",
                                                1, 4, 3)),
                                div(class = "col-6",
                                    sliderInput("i_env_sat",
                                                "Environment satisfaction (1-4)", 1, 4, 3)),
                                div(class = "col-6",
                                    sliderInput("i_rel_sat",
                                                "Relationship satisfaction (1-4)", 1, 4, 3)),
                                div(class = "col-6",
                                    sliderInput("i_wlb", "Work-life balance (1-4)",
                                                1, 4, 3)),
                                div(class = "col-6",
                                    sliderInput("i_job_inv", "Job involvement (1-4)",
                                                1, 4, 3)),
                                div(class = "col-6",
                                    sliderInput("i_perf", "Performance rating (3-4)",
                                                3, 4, 3))
                            )
                          )
                        ),

                        br(),
                        actionButton("btn_pred", "Predict attrition risk",
                                     icon  = icon("bolt"),
                                     class = "btn btn-danger predict-btn")
                    )
                )
            ),

            # RIGHT: Result panel
            div(class = "col-md-7",

                div(class = "card mb-3",
                    div(class = "card-header", "Predicted attrition probability"),
                    div(class = "card-body text-center",
                        withSpinner(plotlyOutput("p_gauge", height = "280px"),
                                    type = 4, color = ibm_red_60),
                        uiOutput("risk_label"),
                        hr(style = "margin: 8px 0;"),
                        p(paste0("Threshold: ", round(threshold * 100, 1),
                                 "% \u2014 employees above this line are flagged as at-risk."),
                          class = "text-muted small mb-0"),
                        p("This threshold was optimized to balance catching potential leavers",
                          " vs. minimizing false alarms (J-index).",
                          class = "help-text mb-0")
                    )
                ),

                div(class = "card mb-3",
                    div(class = "card-header",
                        "Risk factors for this employee"),
                    p(class = "help-text px-3 pt-1 mb-0",
                      "Based on how this employee's profile compares to historical attrition patterns."),
                    div(class = "card-body",
                        uiOutput("employee_risk_factors"))
                ),

                div(class = "card",
                    div(class = "card-header",
                        "Suggested HR actions"),
                    p(class = "help-text px-3 pt-1 mb-0",
                      "Practical steps based on the identified risk factors above."),
                    div(class = "card-body",
                        uiOutput("hr_recommendations"))
                )
            )
        )
    )
  ),

  # ============================================================
  # TAB 4 — MODEL COMPARISON
  # ============================================================
  nav_panel(
    title = tagList(icon("trophy"), " Model Comparison"),

    div(class = "container-fluid py-3",

        uiOutput("model_interpretation"),

        # CV Metrics table
        div(class = "row g-3 mb-3",
            div(class = "col-12",
                div(class = "card",
                    div(class = "card-header",
                        "Cross-validation performance \u2014 all models (5-fold \u00d7 3-repeat CV)"),
                    p(class = "help-text px-3 pt-1 mb-0",
                      "ROC-AUC = overall ranking ability (higher is better).",
                      " Sensitivity = % of actual leavers correctly caught.",
                      " Specificity = % of stayers correctly identified."),
                    div(class = "card-body",
                        DTOutput("tbl_metrics"))
                )
            )
        ),

        div(class = "row g-3 mb-3",
            div(class = "col-12",
                div(class = "card",
                    div(class = "card-header",
                        paste0("Test-set performance \u2014 best model (threshold: ",
                               round(threshold * 100, 1), "%)")),
                    p(class = "help-text px-3 pt-1 mb-0",
                      "Performance on unseen data (30% holdout).",
                      " This is the real-world estimate of how the model will perform."),
                    div(class = "card-body",
                        DTOutput("tbl_test_metrics"))
                )
            )
        ),

        div(class = "row g-3 mb-3",
            div(class = "col-md-6",
                div(class = "card",
                    div(class = "card-header",
                        "ROC Curve \u2014 Gradient Boosting"),
                    p(class = "help-text px-3 pt-1 mb-0",
                      "Plots true positive rate vs false positive rate.",
                      " The further from the diagonal, the better the model."),
                    div(class = "card-body p-2",
                        withSpinner(plotlyOutput("p_roc", height = "360px"),
                                    type = 4, color = ibm_blue_60))
                )
            ),
            div(class = "col-md-6",
                div(class = "card",
                    div(class = "card-header",
                        "Threshold Curve \u2014 Gradient Boosting"),
                    p(class = "help-text px-3 pt-1 mb-0",
                      "Shows how sensitivity and specificity change as the",
                      " classification threshold moves. Dashed line = chosen threshold."),
                    div(class = "card-body p-2",
                        withSpinner(plotlyOutput("p_auc", height = "360px"),
                                    type = 4, color = ibm_blue_60))
                )
            )
        ),

        # Confusion matrix
        div(class = "row g-3",
            div(class = "col-md-6",
                div(class = "card",
                    div(class = "card-header",
                        paste0("Confusion matrix \u2014 best model  |  threshold: ",
                               round(threshold * 100, 1), "%")),
                    p(class = "help-text px-3 pt-1 mb-0",
                      "True Positive = correctly flagged leaver.",
                      " False Positive = stayer incorrectly flagged.",
                      " False Negative = leaver we missed."),
                    div(class = "card-body p-2",
                        withSpinner(plotlyOutput("p_cm", height = "360px"),
                                    type = 4, color = ibm_blue_60))
                )
            ),
            div(class = "col-md-6",
                div(class = "card",
                    div(class = "card-header", "What the confusion matrix means for HR"),
                    div(class = "card-body",
                        uiOutput("cm_interpretation"))
                )
            )
        )
    )
  ),

  # ============================================================
  # ABOUT MODAL — Item #17
  # ============================================================
  nav_item(
    actionLink("show_about", label = tagList(icon("circle-info"), " About"))
  )
)

# =====================================================
# 7. Server
# =====================================================
server <- function(input, output, session) {

  # ── About modal (Item #17) ────────────────────────────
  observeEvent(input$show_about, {
    showModal(modalDialog(
      title = "About this dashboard",
      size = "l",
      easyClose = TRUE,
      div(
        h5("Data source", style = paste0("color:", ibm_blue_60)),
        p("IBM HR Analytics Employee Attrition dataset (fictional).",
          " 1,470 employees, 35 variables. Created by IBM data scientists",
          " for educational purposes. No real employee data is used."),
        hr(),
        h5("Methodology", style = paste0("color:", ibm_blue_60)),
        p("Five classification models were trained and compared: Logistic Regression,",
          " Decision Tree, Random Forest, XGBoost (Gradient Boosting), and MLP Neural Network.",
          " Training used 5-fold cross-validation repeated 3 times with SMOTE oversampling",
          " to address the 84/16% class imbalance. The best model (XGBoost) was evaluated",
          " on a 30% holdout test set. The classification threshold was optimized using",
          " Youden's J-index to balance sensitivity and specificity."),
        hr(),
        h5("Limitations", style = paste0("color:", ibm_blue_60)),
        p("This is a synthetic dataset \u2014 patterns may not reflect real-world dynamics.",
          " The model identifies statistical associations, not causal relationships.",
          " Predictions should inform conversations, not replace HR judgment."),
        hr(),
        p(class = "text-muted small",
          "ESADE MIBA 2026 \u2014 Data Analytics with R \u2014 Group 13")
      )
    ))
  })

  # ── EDA filter dependency: Department -> JobRole ────────
  # Reuses dept_role_map (defined at startup) so that the same
  # Department-to-JobRole mapping is shared with the Tab 3 prediction inputs.
  observeEvent(input$f_dept, {
    if (input$f_dept == "All") {
      valid_roles <- levels(df_ml$JobRole)
    } else {
      valid_roles <- dept_role_map[[input$f_dept]]
    }
    current <- input$f_role
    selected <- if (current %in% c("All", valid_roles)) current else "All"
    updateSelectInput(session, "f_role",
                      choices = c("All", valid_roles), selected = selected)
  })

  # ── Reset filters ─────────────────────────────────────
  observeEvent(input$btn_reset_filters, {
    updateSelectInput(session, "f_dept", selected = "All")
    updateSelectInput(session, "f_role",
                      choices = c("All", levels(df_ml$JobRole)),
                      selected = "All")
    updateSelectInput(session, "f_ot",   selected = "All")
    updateSliderInput(session, "f_age",
                      value = c(min(df_ml$Age), max(df_ml$Age)))
  })

  # ── Prediction input dependencies ──────────────────────

  ## Dependency 1: Department -> JobRole
  ## Only show roles that actually exist in the selected department.
  ## Mapping built from data at startup (dept_role_map).
  observeEvent(input$i_dept, {
    valid_roles <- dept_role_map[[input$i_dept]]
    if (!is.null(valid_roles)) {
      current <- input$i_role
      selected <- if (current %in% valid_roles) current else valid_roles[1]
      updateSelectInput(session, "i_role",
                        choices = valid_roles, selected = selected)
    }
  })

  ## Dependency 2: YearsAtCompany constrains YearsInCurrentRole
  ## Can't have been in current role longer than at the company.
  observeEvent(input$i_yrs_co, {
    yrs_co <- input$i_yrs_co
    if (input$i_yrs_role > yrs_co) {
      updateSliderInput(session, "i_yrs_role", value = yrs_co)
    }
    updateSliderInput(session, "i_yrs_role",
                      max = min(yrs_co, 18))
  })

  ## Dependency 3: YearsAtCompany constrains YearsWithCurrManager
  observeEvent(input$i_yrs_co, {
    yrs_co <- input$i_yrs_co
    if (input$i_yrs_mgr > yrs_co) {
      updateSliderInput(session, "i_yrs_mgr", value = yrs_co)
    }
    updateSliderInput(session, "i_yrs_mgr",
                      max = min(yrs_co, 17))
  }, priority = -1)

  ## Dependency 4: YearsAtCompany constrains YearsSinceLastPromotion
  observeEvent(input$i_yrs_co, {
    yrs_co <- input$i_yrs_co
    if (input$i_yrs_promo > yrs_co) {
      updateSliderInput(session, "i_yrs_promo", value = yrs_co)
    }
    updateSliderInput(session, "i_yrs_promo",
                      max = min(yrs_co, 15))
  }, priority = -2)

  ## Dependency 5: TotalWorkingYears >= YearsAtCompany
  ## Can't have been at this company longer than your total career.
  observeEvent(input$i_total_yrs, {
    total <- input$i_total_yrs
    if (input$i_yrs_co > total) {
      updateSliderInput(session, "i_yrs_co", value = total)
    }
    updateSliderInput(session, "i_yrs_co",
                      max = min(total, 40))
  })

  ## Dependency 6: Age constrains TotalWorkingYears
  ## Minimum working age ~16, so max work years = Age - 16.
  ## In the dataset, min(Age - TotalWorkingYears) = 18, confirming this.
  observeEvent(input$i_age, {
    age <- input$i_age
    max_work_yrs <- max(age - 16, 0)
    if (input$i_total_yrs > max_work_yrs) {
      updateSliderInput(session, "i_total_yrs", value = max_work_yrs)
    }
    updateSliderInput(session, "i_total_yrs",
                      max = min(max_work_yrs, 40))
  })

  ## Dependency 7: JobLevel constrains MonthlyIncome range
  ## Set sensible min/max based on observed data per level.
  observeEvent(input$i_joblevel, {
    lvl <- input$i_joblevel
    lvl_data <- df_ml$MonthlyIncome[df_ml$JobLevel == lvl]
    lvl_min <- max(floor(min(lvl_data) * 0.8), 1009)
    lvl_max <- min(ceiling(max(lvl_data) * 1.2), 19999)
    current <- input$i_income
    new_val <- max(min(current, lvl_max), lvl_min)
    updateNumericInput(session, "i_income",
                       min = lvl_min, max = lvl_max, value = new_val)
  })

  ## Dependency 8: NumCompaniesWorked constrained by TotalWorkingYears
  ## Can't have more company switches than years of experience (roughly)
  observeEvent(input$i_total_yrs, {
    total <- input$i_total_yrs
    max_companies <- min(total + 1, 9)  # +1 because current counts
    if (input$i_num_co > max_companies) {
      updateSliderInput(session, "i_num_co", value = max_companies)
    }
    updateSliderInput(session, "i_num_co", max = max_companies)
  }, priority = -1)

  # ── Filtered data (Tab 1) ───────────────────────────
  df_f <- reactive({
    d <- df_ml
    if (input$f_dept != "All") d <- filter(d, Department == input$f_dept)
    if (input$f_role != "All") d <- filter(d, JobRole    == input$f_role)
    if (input$f_ot   != "All") d <- filter(d, OverTime   == input$f_ot)
    filter(d, Age >= input$f_age[1], Age <= input$f_age[2])
  })

  # ── Executive summary (Item #6) ─────────────────────
  output$exec_summary <- renderUI({
    d <- df_f()
    n <- nrow(d)
    rate <- mean(d$Attrition == "Yes")
    n_at_risk <- sum(d$Attrition == "Yes")

    # Check if filters are active
    is_filtered <- (input$f_dept != "All" | input$f_role != "All" |
                      input$f_ot != "All" |
                      input$f_age[1] != min(df_ml$Age) |
                      input$f_age[2] != max(df_ml$Age))

    filter_note <- if (is_filtered) {
      paste0(" (filtered view: ", n, " of ", nrow(df_ml), " employees)")
    } else { "" }

    div(class = "summary-card",
        h5(paste0("Overview", filter_note)),
        p(paste0(
          format(n, big.mark = ","), " employees analyzed. ",
          "Attrition rate: ", round(rate * 100, 1), "% (",
          n_at_risk, " employees). ",
          "Key risk factors from our model: overtime, monthly income, ",
          "job satisfaction, and years at company."
        ))
    )
  })

  # ── KPI cards ───────────────────────────────────────
  output$kpi_total <- renderUI(
    kpi_box("Total employees",
            format(nrow(df_f()), big.mark = ","), "users", ibm_blue_60))

  output$kpi_rate <- renderUI(
    kpi_box("Attrition rate",
            scales::percent(mean(df_f()$Attrition == "Yes"), accuracy = 0.1),
            "right-from-bracket", ibm_red_60))

  output$kpi_age <- renderUI(
    kpi_box("Average age",
            round(mean(df_f()$Age), 1), "calendar", ibm_blue_60))

  output$kpi_income <- renderUI(
    kpi_box("Avg monthly income",
            paste0("$", format(round(mean(df_f()$MonthlyIncome)), big.mark = ",")),
            "dollar-sign", ibm_green_50))

  # ── EDA plots ────────────────────────────────────────
  output$p_dist <- renderPlotly({
    df_f() %>%
      count(Attrition) %>%
      mutate(pct = n / sum(n)) %>%
      plot_ly(x = ~Attrition, y = ~n, color = ~Attrition,
              colors = pal_attr, type = "bar",
              text  = ~paste0(n, " (", round(pct * 100, 1), "%)"),
              textposition = "outside") %>%
      clean_layout(showlegend = FALSE,
                   xaxis = list(title = ""),
                   yaxis = list(title = "Count"))
  })

  output$p_dept <- renderPlotly({
    df_f() %>%
      group_by(Department) %>%
      summarise(rate = mean(Attrition == "Yes"), .groups = "drop") %>%
      arrange(rate) %>%
      plot_ly(x = ~rate, y = ~reorder(Department, rate),
              type = "bar", orientation = "h",
              marker = list(color = ibm_red_60),
              text = ~paste0(round(rate * 100, 1), "%"),
              textposition = "outside") %>%
      clean_layout(
        shapes = list(list(
          type = "line",
          x0 = overall_attrition_rate, x1 = overall_attrition_rate,
          y0 = -0.5, y1 = 2.5,
          line = list(color = ibm_gray_70, width = 2, dash = "dash")
        )),
        xaxis = list(title = "Attrition rate", tickformat = ".0%"),
        yaxis = list(title = ""),
        margin = list(t = 30, b = 50, l = 160, r = 50))
  })

  output$p_role <- renderPlotly({
    role_data <- df_f() %>%
      group_by(JobRole) %>%
      summarise(rate = mean(Attrition == "Yes"), .groups = "drop") %>%
      arrange(rate)
    n_roles <- nrow(role_data)

    role_data %>%
      plot_ly(x = ~rate, y = ~reorder(JobRole, rate),
              type = "bar", orientation = "h",
              marker = list(color = ibm_red_60),
              text = ~paste0(round(rate * 100, 1), "%"),
              textposition = "outside") %>%
      clean_layout(
        shapes = list(list(
          type = "line",
          x0 = overall_attrition_rate, x1 = overall_attrition_rate,
          y0 = -0.5, y1 = n_roles - 0.5,
          line = list(color = ibm_gray_70, width = 2, dash = "dash")
        )),
        xaxis = list(title = "Attrition rate", tickformat = ".0%"),
        yaxis = list(title = ""),
        margin = list(t = 30, b = 50, l = 170, r = 50))
  })

  output$p_income <- renderPlotly({
    plot_ly(df_f(), x = ~Attrition, y = ~MonthlyIncome,
            color = ~Attrition, colors = pal_attr, type = "box") %>%
      clean_layout(showlegend = FALSE,
                   xaxis = list(title = ""),
                   yaxis = list(title = "Monthly income ($)"),
                   margin = list(t = 20, b = 40, l = 70, r = 10))
  })

  output$p_age_hist <- renderPlotly({
    plot_ly(df_f(), x = ~Age, color = ~Attrition,
            colors = pal_attr, type = "histogram",
            alpha = 0.7, nbinsx = 25) %>%
      clean_layout(barmode = "overlay",
                   xaxis = list(title = "Age"),
                   yaxis = list(title = "Count"),
                   margin = list(t = 20, b = 50, l = 55, r = 10))
  })

  output$p_travel <- renderPlotly({
    travel_data <- df_f() %>%
      group_by(BusinessTravel) %>%
      summarise(rate = mean(Attrition == "Yes"), .groups = "drop") %>%
      arrange(rate)
    n_travel <- nrow(travel_data)

    travel_data %>%
      plot_ly(x = ~rate, y = ~reorder(BusinessTravel, rate),
              type = "bar", orientation = "h",
              marker = list(color = ibm_yellow),
              text = ~paste0(round(rate * 100, 1), "%"),
              textposition = "outside") %>%
      clean_layout(
        shapes = list(list(
          type = "line",
          x0 = overall_attrition_rate, x1 = overall_attrition_rate,
          y0 = -0.5, y1 = n_travel - 0.5,
          line = list(color = ibm_gray_70, width = 2, dash = "dash")
        )),
        xaxis = list(title = "Attrition rate", tickformat = ".0%"),
        yaxis = list(title = ""),
        margin = list(t = 30, b = 50, l = 130, r = 50))
  })

  output$p_tenure <- renderPlotly({
    tenure_data <- df_f() %>%
      mutate(tenure_bin = cut(YearsAtCompany,
                              breaks = c(-1, 1, 2, 5, 10, 20, Inf),
                              labels = c("0-1", "1-2", "2-5",
                                         "5-10", "10-20", "20+"))) %>%
      group_by(tenure_bin) %>%
      summarise(rate = mean(Attrition == "Yes"),
                n = n(), .groups = "drop")

    tenure_data %>%
      plot_ly(x = ~tenure_bin, y = ~rate, type = "bar",
              marker = list(color = ibm_blue_60),
              text = ~paste0(round(rate * 100, 1), "% (n=", n, ")"),
              textposition = "outside") %>%
      clean_layout(
        shapes = list(list(
          type = "line",
          x0 = -0.5, x1 = nrow(tenure_data) - 0.5,
          y0 = overall_attrition_rate, y1 = overall_attrition_rate,
          line = list(color = ibm_gray_70, width = 2, dash = "dash")
        )),
        annotations = list(list(
          x = nrow(tenure_data) - 1, y = overall_attrition_rate + 0.02,
          text = paste0("Overall: ", round(overall_attrition_rate * 100, 1), "%"),
          showarrow = FALSE,
          font = list(size = 11, color = ibm_gray_70)
        )),
        xaxis = list(title = "Years at company"),
        yaxis = list(title = "Attrition rate", tickformat = ".0%"))
  })

  # ── Risk Analysis ─────────────────────────────────────
  output$p_vip <- renderPlotly({
    vip_df %>%
      plot_ly(x = ~Importance, y = ~reorder(Variable, Importance),
              type = "bar", orientation = "h",
              marker = list(
                color = ~Importance,
                colorscale = list(list(0, "#a6c8ff"), list(1, ibm_blue_80)),
                showscale = FALSE
              )) %>%
      clean_layout(xaxis = list(title = "Importance"),
                   yaxis = list(title = ""),
                   margin = list(t = 30, b = 50, l = 170, r = 30))
  })

  output$p_risk_cat <- renderPlotly({
    var <- input$risk_var
    cat_data <- df_ml %>%
      rename(cat = all_of(var)) %>%
      group_by(cat) %>%
      summarise(rate = mean(Attrition == "Yes"), n = n(), .groups = "drop") %>%
      arrange(rate)
    n_cats <- nrow(cat_data)

    cat_data %>%
      plot_ly(x = ~rate, y = ~reorder(cat, rate),
              type = "bar", orientation = "h",
              text  = ~paste0(round(rate * 100, 1), "% (n=", n, ")"),
              textposition = "outside",
              marker = list(color = ibm_red_60)) %>%
      clean_layout(
        shapes = list(list(
          type = "line",
          x0 = overall_attrition_rate, x1 = overall_attrition_rate,
          y0 = -0.5, y1 = n_cats - 0.5,
          line = list(color = ibm_gray_70, width = 2, dash = "dash")
        )),
        xaxis = list(title = "Attrition rate", tickformat = ".0%"),
        yaxis = list(title = ""),
        margin = list(t = 30, b = 50, l = 170, r = 60))
  })

  output$p_corr <- renderPlotly({
    cor_df <- as.data.frame(cor_mat) %>%
      rownames_to_column("var1") %>%
      pivot_longer(-var1, names_to = "var2", values_to = "corr")

    plot_ly(cor_df, x = ~var2, y = ~var1, z = ~corr,
            type = "heatmap",
            colorscale = list(
              list(0, ibm_blue_60), list(0.5, "#FFFFFF"), list(1, ibm_red_60)
            ),
            zmin = -1, zmax = 1,
            hovertemplate = "%{y} x %{x}<br>r = %{z:.2f}<extra></extra>") %>%
      clean_layout(
        xaxis = list(title = "", tickangle = -45, tickfont = list(size = 9)),
        yaxis = list(title = "", tickfont = list(size = 9)),
        margin = list(t = 10, b = 80, l = 80, r = 10)
      )
  })

  # ── Prediction ────────────────────────────────────────
  pred_res <- eventReactive(input$btn_pred, {
    new_emp <- tibble(
      Age                      = as.integer(input$i_age),
      BusinessTravel           = input$i_travel,
      DailyRate                = as.integer(input$i_daily),
      Department               = input$i_dept,
      DistanceFromHome         = as.integer(input$i_dist),
      Education                = as.integer(input$i_edu),
      EducationField           = input$i_edu_field,
      EnvironmentSatisfaction  = as.integer(input$i_env_sat),
      Gender                   = input$i_gender,
      HourlyRate               = as.integer(input$i_hourly),
      JobInvolvement           = as.integer(input$i_job_inv),
      JobLevel                 = as.integer(input$i_joblevel),
      JobRole                  = input$i_role,
      JobSatisfaction          = as.integer(input$i_job_sat),
      MaritalStatus            = input$i_marital,
      MonthlyIncome            = as.integer(input$i_income),
      MonthlyRate              = as.integer(input$i_monthly_rate),
      NumCompaniesWorked       = as.integer(input$i_num_co),
      OverTime                 = input$i_ot,
      PercentSalaryHike        = as.integer(input$i_hike),
      PerformanceRating        = as.integer(input$i_perf),
      RelationshipSatisfaction = as.integer(input$i_rel_sat),
      StockOptionLevel         = as.integer(input$i_stock),
      TotalWorkingYears        = as.integer(input$i_total_yrs),
      TrainingTimesLastYear    = as.integer(input$i_training),
      WorkLifeBalance          = as.integer(input$i_wlb),
      YearsAtCompany           = as.integer(input$i_yrs_co),
      YearsInCurrentRole       = as.integer(input$i_yrs_role),
      YearsSinceLastPromotion  = as.integer(input$i_yrs_promo),
      YearsWithCurrManager     = as.integer(input$i_yrs_mgr)
    ) %>%
      mutate(across(where(is.character), as.factor))

    prob  <- predict(fitted_wf, new_emp, type = "prob")$.pred_Yes
    label <- case_when(
      prob >= HIGH_RISK_CUTOFF ~ "High Risk",
      prob >= threshold        ~ "Medium Risk",
      TRUE                     ~ "Low Risk"
    )

    emp_inputs <- list(
      OverTime = input$i_ot,
      MaritalStatus = input$i_marital,
      JobSatisfaction = as.integer(input$i_job_sat),
      EnvironmentSatisfaction = as.integer(input$i_env_sat),
      WorkLifeBalance = as.integer(input$i_wlb),
      MonthlyIncome = as.integer(input$i_income),
      JobLevel = as.integer(input$i_joblevel),
      YearsSinceLastPromotion = as.integer(input$i_yrs_promo),
      YearsAtCompany = as.integer(input$i_yrs_co),
      Age = as.integer(input$i_age),
      StockOptionLevel = as.integer(input$i_stock),
      BusinessTravel = input$i_travel,
      JobInvolvement = as.integer(input$i_job_inv)
    )

    risk_factors <- get_employee_risk_factors(emp_inputs, df_ml)
    recommendations <- get_hr_recommendations(risk_factors)

    list(prob = prob, label = label,
         risk_factors = risk_factors, recommendations = recommendations)
  })

  output$p_gauge <- renderPlotly({
    req(pred_res())
    prob  <- pred_res()$prob
    color <- case_when(
      prob >= HIGH_RISK_CUTOFF ~ ibm_red_60,
      prob >= threshold        ~ ibm_yellow,
      TRUE                     ~ ibm_green_50
    )
    plot_ly(
      type  = "indicator",
      mode  = "gauge+number+delta",
      value = round(prob * 100, 1),
      delta = list(
        reference  = threshold * 100,
        increasing = list(color = ibm_red_60),
        decreasing = list(color = ibm_green_50)
      ),
      number = list(suffix = "%", font = list(size = 44, color = color)),
      gauge = list(
        axis  = list(range = list(0, 100), ticksuffix = "%",
                     tickfont = list(size = 11)),
        bar   = list(color = color, thickness = 0.25),
        bgcolor = "white",
        borderwidth = 0,
        steps = list(
          list(range = c(0, threshold * 100),      color = "#defbe6"),
          list(range = c(threshold * 100, 60),     color = "#fff8e1"),
          list(range = c(60, 100),                 color = "#fff1f1")
        ),
        threshold = list(
          line      = list(color = ibm_gray_100, width = 3),
          thickness = 0.75,
          value     = threshold * 100
        )
      )
    ) %>%
      layout(
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        margin        = list(t = 20, b = 0, l = 40, r = 40),
        font          = list(family = "IBM Plex Sans, sans-serif")
      )
  })

  output$risk_label <- renderUI({
    req(pred_res())
    lbl <- pred_res()$label
    cls <- switch(lbl,
                  "High Risk"   = "risk-high",
                  "Medium Risk" = "risk-medium",
                  "Low Risk"    = "risk-low"
    )
    div(class = "risk-badge mt-1", span(lbl, class = cls))
  })

  output$employee_risk_factors <- renderUI({
    req(pred_res())
    rfs <- pred_res()$risk_factors

    if (length(rfs) == 0) {
      return(p(class = "text-muted", "No specific risk factors detected for this profile."))
    }

    tags$div(
      lapply(rfs, function(rf) {
        css_class <- paste0("risk-factor-item risk-factor-", rf$severity)
        div(class = css_class,
            tags$strong(rf$label),
            tags$br(),
            tags$span(class = "text-muted", rf$detail)
        )
      })
    )
  })

  output$hr_recommendations <- renderUI({
    req(pred_res())
    recs <- pred_res()$recommendations

    tags$div(
      lapply(recs, function(rec) {
        div(class = "rec-item",
            icon(rec$icon), rec$text
        )
      })
    )
  })

  # ── Model Comparison ─────────────────────────────────
  output$model_interpretation <- renderUI({
    # Extract test metrics for interpretation
    test_sens <- test_metrics %>%
      filter(.metric == "sensitivity") %>%
      pull(.estimate)
    test_spec <- test_metrics %>%
      filter(.metric == "specificity") %>%
      pull(.estimate)
    test_prec <- test_metrics %>%
      filter(.metric == "precision") %>%
      pull(.estimate)
    test_auc  <- test_metrics %>%
      filter(.metric == "roc_auc") %>%
      pull(.estimate)

    div(class = "summary-card mb-3",
        h5("Model performance in plain language"),
        p(paste0(
          "Our best model (Gradient Boosting / XGBoost) achieves a ROC-AUC of ",
          round(test_auc, 3), " on unseen data. In practical terms: ",
          "it correctly identifies ", round(test_sens * 100), "% of employees who will leave, ",
          "while correctly classifying ", round(test_spec * 100), "% of those who stay. ",
          "When the model flags someone as high-risk, it is correct about ",
          round(test_prec * 100), "% of the time. ",
          "The remaining ", round((1 - test_prec) * 100), "% are false alarms ",
          "\u2014 employees flagged who would actually stay. This is an acceptable trade-off: ",
          "a retention conversation with a satisfied employee costs little, ",
          "but missing someone about to leave is expensive."
        ))
    )
  })

  output$tbl_metrics <- renderDT({
    cv_metrics_all %>%
      mutate(across(where(is.numeric), ~round(., 3))) %>%
      arrange(desc(roc_auc)) %>%
      rename(`ROC-AUC` = roc_auc, Accuracy = accuracy,
             Sensitivity = sensitivity, Specificity = specificity,
             Precision = precision, `F1` = f_meas,
             Model = model) %>%
      select(Model, `ROC-AUC`, Accuracy, Precision, Sensitivity, `F1`, Specificity) %>%
      datatable(
        rownames  = FALSE,
        selection = "none",
        options   = list(
          dom        = "t",
          pageLength = 10,
          columnDefs = list(list(className = "dt-center", targets = "_all"))
        )
      ) %>%
      formatStyle("Model", fontWeight = "bold") %>%
      formatStyle("ROC-AUC", fontWeight = "bold", color = ibm_blue_60)
  })

  output$tbl_test_metrics <- renderDT({
    test_metrics %>%
      select(.metric, .estimate) %>%
      mutate(
        .estimate = round(.estimate, 3),
        .metric = case_when(
          .metric == "roc_auc"     ~ "ROC-AUC",
          .metric == "accuracy"    ~ "Accuracy",
          .metric == "sensitivity" ~ "Sensitivity (recall)",
          .metric == "specificity" ~ "Specificity",
          .metric == "precision"   ~ "Precision",
          .metric == "f_meas"      ~ "F1 Score",
          TRUE ~ .metric
        )
      ) %>%
      rename(Metric = .metric, Value = .estimate) %>%
      datatable(
        rownames  = FALSE,
        selection = "none",
        options   = list(
          dom        = "t",
          pageLength = 10,
          columnDefs = list(list(className = "dt-center", targets = "_all"))
        )
      ) %>%
      formatStyle("Metric", fontWeight = "bold") %>%
      formatStyle("Value", fontWeight = "bold", color = ibm_blue_60)
  })

  output$p_roc <- renderPlotly({
    threshold_curve <- read.csv("./OUTPUT/threshold_curve.csv")

    # Build ROC data: sensitivity (TPR) vs 1-specificity (FPR)
    roc_df <- threshold_curve %>%
      filter(.metric %in% c("sensitivity", "specificity")) %>%
      select(.threshold, .metric, .estimate) %>%
      pivot_wider(names_from = .metric, values_from = .estimate) %>%
      mutate(fpr = 1 - specificity) %>%
      arrange(fpr)

    roc_fpr   <- c(0, as.numeric(roc_df$fpr), 1)
    roc_tpr   <- c(0, as.numeric(roc_df$sensitivity), 1)
    roc_thresh <- c("1.00", as.character(round(as.numeric(roc_df$.threshold), 2)), "0.00")

    # Pre-compute the threshold marker point as scalars
    idx <- which.min(abs(as.numeric(roc_df$.threshold) - threshold))
    marker_x <- roc_fpr[idx]
    marker_y <- roc_tpr[idx]

    # Get test AUC for annotation
    test_auc <- round(test_metrics$.estimate[test_metrics$.metric == "roc_auc"], 3)

    plot_ly() %>%
      # ROC curve with fill
      add_trace(x = roc_fpr, y = roc_tpr, type = "scatter", mode = "lines",
                line = list(color = ibm_blue_60, width = 2.5),
                fill = "tozeroy",
                fillcolor = "rgba(15, 98, 254, 0.08)",
                text = roc_thresh,
                hovertemplate = paste0(
                  "Threshold: %{text}<br>",
                  "TPR: %{y:.2f}<br>",
                  "FPR: %{x:.2f}<extra></extra>"),
                showlegend = FALSE) %>%
      # Diagonal reference line (random classifier)
      add_trace(x = c(0, 1), y = c(0, 1),
                type = "scatter", mode = "lines",
                line = list(color = ibm_gray_70, width = 1, dash = "dash"),
                showlegend = FALSE, hoverinfo = "skip") %>%
      # Mark the chosen threshold point as a single scalar coordinate
      add_trace(x = marker_x, y = marker_y,
                type = "scatter", mode = "markers",
                marker = list(color = ibm_red_60, size = 10, symbol = "circle"),
                showlegend = FALSE,
                hovertemplate = paste0(
                  "Chosen threshold (", round(threshold, 2), ")",
                  "<br>TPR: ", round(marker_y, 2),
                  "<br>FPR: ", round(marker_x, 2),
                  "<extra></extra>")) %>%
      clean_layout(
        showlegend = FALSE,
        annotations = list(list(
          x = 0.65, y = 0.25,
          text = paste0("AUC = ", test_auc),
          showarrow = FALSE,
          font = list(size = 14, color = ibm_blue_60, family = "IBM Plex Sans")
        )),
        xaxis = list(title = "False Positive Rate (1 - Specificity)",
                     range = c(0, 1)),
        yaxis = list(title = "True Positive Rate (Sensitivity)",
                     range = c(0, 1)),
        margin = list(t = 30, b = 60, l = 60, r = 30)
      )
  })

  output$p_auc <- renderPlotly({
    threshold_curve <- read.csv("./OUTPUT/threshold_curve.csv")

    threshold_curve %>%
      filter(.metric %in% c("j_index", "sensitivity", "specificity")) %>%
      plot_ly(x = ~.threshold, y = ~.estimate, color = ~.metric,
              type = "scatter", mode = "lines",
              colors = c("j_index"     = ibm_red_60,
                         "sensitivity" = ibm_green_50,
                         "specificity" = ibm_blue_60),
              line = list(width = 2)) %>%
      clean_layout(
        shapes = list(list(
          type = "line",
          x0 = threshold, x1 = threshold,
          y0 = 0, y1 = 1,
          line = list(color = ibm_gray_100, width = 2, dash = "dash")
        )),
        annotations = list(list(
          x = threshold + 0.03, y = 0.08,
          text = paste0("Optimal: ", round(threshold * 100, 1), "%"),
          showarrow = FALSE,
          font = list(color = ibm_gray_100, size = 11)
        )),
        xaxis = list(title = "Threshold", tickformat = ".2f"),
        yaxis = list(title = "Metric value", range = c(0, 1)),
        legend = list(orientation = "h", y = -0.2)
      )
  })

  output$p_cm <- renderPlotly({
    cm_df <- as.data.frame(
      conf_mat(test_preds,
               truth    = Attrition,
               estimate = .pred_class_opt)$table
    ) %>%
      mutate(
        label = case_when(
          Truth == "Yes" & Prediction == "Yes" ~ paste0(Freq, "\nTrue Positive"),
          Truth == "No"  & Prediction == "No"  ~ paste0(Freq, "\nTrue Negative"),
          Truth == "Yes" & Prediction == "No"  ~ paste0(Freq, "\nFalse Negative"),
          TRUE                                  ~ paste0(Freq, "\nFalse Positive")
        )
      )

    plot_ly(cm_df,
            x = ~Prediction, y = ~Truth, z = ~Freq,
            type = "heatmap",
            colorscale = list(list(0, "#edf5ff"), list(1, ibm_blue_80)),
            text = ~label,
            texttemplate = "<b>%{text}</b>",
            showscale    = FALSE,
            hovertemplate = "Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>") %>%
      clean_layout(
        xaxis  = list(title = "Predicted", side = "top"),
        yaxis  = list(title = "Actual", autorange = "reversed"),
        margin = list(t = 60, b = 20, l = 60, r = 20)
      )
  })

  output$cm_interpretation <- renderUI({
    cm <- conf_mat(test_preds,
                   truth    = Attrition,
                   estimate = .pred_class_opt)$table
    tp <- cm["Yes", "Yes"]
    fp <- cm["Yes", "No"]
    fn <- cm["No", "Yes"]
    tn <- cm["No", "No"]

    div(
      p(style = paste0("font-size:.9rem; color:", ibm_gray_100),
        tags$strong("Reading the matrix:")),
      tags$ul(style = "font-size:.85rem; list-style: none; padding-left: 0;",
        tags$li(style = paste0("margin-bottom:8px; padding:8px 12px; border-radius:6px;",
                               " background:", ibm_green_50, "12;"),
                icon("check-circle", style = paste0("color:", ibm_green_50,
                                                    "; margin-right:6px;")),
                tags$strong(paste0(tp, " True Positives:")),
                " Employees correctly identified as likely to leave.",
                " These are your intervention targets."),
        tags$li(style = paste0("margin-bottom:8px; padding:8px 12px; border-radius:6px;",
                               " background:", ibm_green_50, "12;"),
                icon("check-circle", style = paste0("color:", ibm_green_50,
                                                    "; margin-right:6px;")),
                tags$strong(paste0(tn, " True Negatives:")),
                " Employees correctly identified as likely to stay. No action needed."),
        tags$li(style = paste0("margin-bottom:8px; padding:8px 12px; border-radius:6px;",
                               " background:", ibm_yellow, "20;"),
                icon("exclamation-triangle", style = paste0("color:", ibm_yellow,
                                                            "; margin-right:6px;")),
                tags$strong(paste0(fp, " False Positives:")),
                " Stayers flagged as at-risk.",
                " Cost = unnecessary retention conversation (low)."),
        tags$li(style = paste0("margin-bottom:8px; padding:8px 12px; border-radius:6px;",
                               " background:", ibm_red_60, "15;"),
                icon("times-circle", style = paste0("color:", ibm_red_60,
                                                    "; margin-right:6px;")),
                tags$strong(paste0(fn, " False Negatives:")),
                " Leavers we missed. Cost = lost employee (high).",
                " This is the number we want to minimize.")
      ),
      p(class = "help-text",
        paste0("Our threshold of ", round(threshold * 100, 1),
               "% was set to catch more leavers at the cost of some extra false alarms ",
               "\u2014 because a retention chat is cheap, but losing talent is expensive."))
    )
  })
}

# =====================================================
# 8. Run
# =====================================================
shinyApp(ui = ui, server = server)
