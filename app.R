# -*- coding: utf-8 -*-
# =============================================================================
# SHINY APP: Glucose Concentration Prediction from Biosensor Deflection
# Feature Extraction + ML Pipeline with XAI
# All training parameters configurable by user
# =============================================================================

library(shiny)
library(shinydashboard)
library(shinyjs)
library(tidyverse)
library(hms)
library(Metrics)
library(randomForest)
library(e1071)
library(xgboost)
library(rpart)
library(rpart.plot)
library(FNN)
library(nnet)
library(gbm)
library(kernlab)
library(ggplot2)
library(gridExtra)
library(pracma)
library(moments)
library(shapviz)
library(kernelshap)
library(iml)
library(pdp)
library(lime)
library(DT)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

load_data <- function(filepath) {
  df <- read.csv(filepath)
  colnames(df) <- c("Time", "Deflection", "Glucose")
  df <- df %>%
    mutate(
      Time_sec = as.numeric(hms::as_hms(Time)),
      Glucose  = as.factor(Glucose)
    )
  return(df)
}

extract_features <- function(time_sec, deflection) {
  n <- length(deflection)
  feat_mean        <- mean(deflection)
  feat_sd          <- sd(deflection)
  feat_min         <- min(deflection)
  feat_max         <- max(deflection)
  feat_range       <- feat_max - feat_min
  feat_skewness    <- moments::skewness(deflection)
  feat_kurtosis    <- moments::kurtosis(deflection)
  feat_auc         <- pracma::trapz(time_sec, deflection)
  feat_auc_abs     <- pracma::trapz(time_sec, abs(deflection))
  feat_t_min       <- time_sec[which.min(deflection)]
  feat_t_max       <- time_sec[which.max(deflection)]
  baseline_idx     <- 1:max(1, round(n * 0.15))
  feat_baseline    <- mean(deflection[baseline_idx])
  feat_baseline_sd <- sd(deflection[baseline_idx])
  plateau_idx      <- round(n * 0.70):n
  feat_plateau     <- mean(deflection[plateau_idx])
  feat_plateau_sd  <- sd(deflection[plateau_idx])
  feat_response_amp  <- feat_plateau - feat_baseline
  feat_dip_depth     <- feat_min - feat_baseline
  feat_recovery      <- feat_plateau - feat_min
  rs <- round(n * 0.20); re <- round(n * 0.50)
  feat_slope <- if (re > rs + 1) coef(lm(deflection[rs:re] ~ time_sec[rs:re]))[2] else NA_real_
  feat_plateau_ratio <- ifelse(abs(feat_baseline) > 0.1, feat_plateau / feat_baseline, NA_real_)
  feat_plateau_cv    <- ifelse(abs(feat_plateau) > 0.1,
                               (feat_plateau_sd / abs(feat_plateau)) * 100, NA_real_)
  data.frame(
    mean = feat_mean, sd = feat_sd, min_val = feat_min, max_val = feat_max,
    range_val = feat_range, skewness = feat_skewness, kurtosis = feat_kurtosis,
    auc = feat_auc, auc_abs = feat_auc_abs, t_min = feat_t_min, t_max = feat_t_max,
    baseline_mean = feat_baseline, baseline_sd = feat_baseline_sd,
    plateau_mean = feat_plateau, plateau_sd = feat_plateau_sd,
    response_amp = feat_response_amp, dip_depth = feat_dip_depth,
    recovery = feat_recovery, slope_response = feat_slope,
    plateau_ratio = feat_plateau_ratio, plateau_cv = feat_plateau_cv
  )
}

# LOOCV
loocv_reg <- function(X, y, fit_fn, pred_fn = NULL) {
  preds <- numeric(length(y))
  for (i in seq_along(y)) {
    fit      <- fit_fn(X[-i, , drop = FALSE], y[-i])
    preds[i] <- if (is.null(pred_fn)) predict(fit, X[i, , drop = FALSE])
                else pred_fn(fit, X[i, , drop = FALSE])
  }
  data.frame(actual = y, predicted = preds)
}

# k-Fold CV
kfold_reg <- function(X, y, fit_fn, pred_fn = NULL, k = 5, seed = 42) {
  set.seed(seed)
  n     <- nrow(X)
  folds <- sample(rep(1:k, length.out = n))
  preds <- numeric(n)
  for (fold in 1:k) {
    itr <- which(folds != fold); ite <- which(folds == fold)
    fit       <- fit_fn(X[itr, , drop = FALSE], y[itr])
    preds[ite] <- if (is.null(pred_fn)) predict(fit, X[ite, , drop = FALSE])
                  else pred_fn(fit, X[ite, , drop = FALSE])
  }
  data.frame(actual = y, predicted = preds)
}

# Train / Test split
tt_reg <- function(X, y, fit_fn, pred_fn = NULL, train_ratio = 0.75, seed = 42) {
  set.seed(seed)
  n    <- nrow(X)
  itr  <- sample(seq_len(n), size = floor(n * train_ratio))
  ite  <- setdiff(seq_len(n), itr)
  if (length(ite) == 0) ite <- itr
  fit  <- fit_fn(X[itr, , drop = FALSE], y[itr])
  p    <- if (is.null(pred_fn)) predict(fit, X[ite, , drop = FALSE])
          else pred_fn(fit, X[ite, , drop = FALSE])
  data.frame(actual = y[ite], predicted = p)
}

reg_metrics <- function(res, label, type = "Blackbox") {
  rmse_val <- Metrics::rmse(res$actual, res$predicted)
  mae_val  <- Metrics::mae(res$actual, res$predicted)
  r2_val   <- tryCatch(cor(res$actual, res$predicted)^2, error = function(e) NA_real_)
  data.frame(Model = label, Type = type, RMSE = rmse_val, MAE = mae_val, Rsq = r2_val)
}

model_colors <- c(
  "LR"                = "#E41A1C", "DT"             = "#FF7F00",
  "RF"                = "#4DAF4A", "SVM"            = "#377EB8",
  "XGBoost"           = "#984EA3", "KNN"            = "#A65628",
  "NN"                = "#F781BF", "GBM"            = "#999999",
  "GPR"               = "#00CED1"
)

normalise_perm <- function(df, model_label) {
  df       <- as.data.frame(df)
  feat_col <- grep("^feature$", colnames(df), ignore.case = TRUE, value = TRUE)[1]
  imp_col  <- grep("^importance$", colnames(df), ignore.case = TRUE, value = TRUE)
  imp_col  <- if (length(imp_col) == 0) colnames(df)[sapply(df, is.numeric)][1] else imp_col[1]
  data.frame(Feature = df[[feat_col]], Importance = df[[imp_col]],
             Model = model_label, stringsAsFactors = FALSE)
}

# Universal CV dispatcher (for models without special DMatrix handling)
run_cv <- function(X, y, fit_fn, pred_fn, method, kk, ttr, seed) {
  if      (method == "loocv") loocv_reg(X, y, fit_fn, pred_fn)
  else if (method == "kfold") kfold_reg(X, y, fit_fn, pred_fn, k = kk,  seed = seed)
  else                        tt_reg   (X, y, fit_fn, pred_fn, train_ratio = ttr, seed = seed)
}

# =============================================================================
# UI
# =============================================================================

ui <- dashboardPage(
  skin = "blue",
  dashboardHeader(title = "Glucose Biosensor ML", titleWidth = 300),

  dashboardSidebar(
    width = 300,
    sidebarMenu(
      menuItem("Data Upload & EDA",        tabName = "eda",      icon = icon("upload")),
      menuItem("Feature Extraction",       tabName = "features", icon = icon("table")),
      menuItem("Training Settings",        tabName = "settings", icon = icon("sliders-h")),
      menuItem("Model Training & Results", tabName = "models",   icon = icon("cogs")),
      menuItem("XAI - SHAP",               tabName = "shap",     icon = icon("brain")),
      menuItem("XAI - Permutation",        tabName = "perm",     icon = icon("random")),
      menuItem("XAI - PDP",                tabName = "pdp_tab",  icon = icon("chart-line")),
      menuItem("XAI - LIME",               tabName = "lime_tab", icon = icon("lightbulb")),
      menuItem("RF Variable Importance",   tabName = "rf_imp",   icon = icon("star")),
      menuItem("Prediction",               tabName = "predict",  icon = icon("bullseye"))
    )
  ),

  dashboardBody(
    useShinyjs(),
    tags$head(tags$style(HTML("
      .content-wrapper { background-color: #f4f6f9; }
      .box { border-radius: 6px; }
      .download-btn { margin-top: 6px; }
      .section-lbl { font-weight: bold; color: #2c3e50; margin-top: 8px; font-size: 13px; }
    "))),

    tabItems(

      # -----------------------------------------------------------------------
      # 1. DATA UPLOAD & EDA
      # -----------------------------------------------------------------------
      tabItem(tabName = "eda",
        fluidRow(
          box(title = "Upload Data (CSV)", width = 4, status = "primary", solidHeader = TRUE,
            fileInput("datafile", "Select CSV file", accept = ".csv"),
            helpText("Expected columns: Time (HH:MM:SS), Deflection, Glucose"),
            hr(),
            verbatimTextOutput("data_summary")
          ),
          box(title = "Time-Series: All Groups", width = 8, status = "info", solidHeader = TRUE,
            plotOutput("plot_ts_all", height = "350px"),
            downloadButton("dl_ts_all", "Download PNG (600 DPI)", class = "download-btn")
          )
        ),
        fluidRow(
          box(title = "Time-Series: Faceted by Glucose Level", width = 12,
              status = "info", solidHeader = TRUE,
            plotOutput("plot_ts_facet", height = "450px"),
            downloadButton("dl_ts_facet", "Download PNG (600 DPI)", class = "download-btn")
          )
        )
      ),

      # -----------------------------------------------------------------------
      # 2. FEATURE EXTRACTION
      # -----------------------------------------------------------------------
      tabItem(tabName = "features",
        fluidRow(
          box(title = "Extracted Feature Matrix (21 features)", width = 12,
              status = "primary", solidHeader = TRUE,
            DTOutput("feature_table"),
            downloadButton("dl_feature_table", "Download CSV", class = "download-btn")
          )
        ),
        fluidRow(
          box(title = "Feature-Glucose Correlation", width = 6, status = "info", solidHeader = TRUE,
            plotOutput("plot_cor", height = "400px"),
            downloadButton("dl_cor", "Download PNG (600 DPI)", class = "download-btn")
          ),
          box(title = "Calibration Curve (Plateau Mean)", width = 6, status = "info", solidHeader = TRUE,
            plotOutput("plot_calib", height = "400px"),
            downloadButton("dl_calib", "Download PNG (600 DPI)", class = "download-btn")
          )
        )
      ),

      # -----------------------------------------------------------------------
      # 3. TRAINING SETTINGS  <-- NEW
      # -----------------------------------------------------------------------
      tabItem(tabName = "settings",
        fluidRow(
          # General
          box(title = "General", width = 4, status = "primary", solidHeader = TRUE,
            numericInput("seed_val", "Random Seed", value = 42, min = 0, step = 1),
            numericInput("top_n", "Top N Features (by |r|)", value = 6, min = 1, max = 21),
            hr(),
            p(class = "section-lbl", "Cross-Validation Strategy"),
            selectInput("cv_method", NULL,
                        choices = c("LOOCV (Leave-One-Out)" = "loocv",
                                    "k-Fold CV"             = "kfold",
                                    "Train / Test Split"    = "tt")),
            conditionalPanel("input.cv_method == 'kfold'",
              numericInput("kfold_k", "Number of Folds (k)", value = 5, min = 2, max = 20)
            ),
            conditionalPanel("input.cv_method == 'tt'",
              sliderInput("tt_ratio", "Training Set Ratio",
                          min = 0.5, max = 0.9, value = 0.75, step = 0.05),
              verbatimTextOutput("tt_info")
            )
          ),
          # Random Forest
          box(title = "RF", width = 4, status = "success", solidHeader = TRUE,
            numericInput("rf_ntree", "ntree  (# trees)",    value = 500, min = 50,  step = 50),
            numericInput("rf_mtry",  "mtry   (0 = auto p/3)", value = 0, min = 0,   step = 1)
          ),
          # XGBoost
          box(title = "XGBoost", width = 4, status = "warning", solidHeader = TRUE,
            numericInput("xgb_nrounds",   "nrounds",           value = 50,  min = 10,   step = 10),
            numericInput("xgb_max_depth", "max_depth",         value = 2,   min = 1,    step = 1),
            numericInput("xgb_eta",       "eta",               value = 0.1, min = 0.001, step = 0.01),
            numericInput("xgb_subsample", "subsample",         value = 0.8, min = 0.3, max = 1, step = 0.05),
            numericInput("xgb_colsample", "colsample_bytree",  value = 0.8, min = 0.3, max = 1, step = 0.05)
          )
        ),
        fluidRow(
          # SVM
          box(title = "SVM (RBF)", width = 3, status = "info", solidHeader = TRUE,
            numericInput("svm_cost",    "Cost",    value = 1,   min = 0.01, step = 0.5),
            numericInput("svm_epsilon", "Epsilon", value = 0.1, min = 0.001, step = 0.05)
          ),
          # KNN
          box(title = "KNN", width = 3, status = "info", solidHeader = TRUE,
            numericInput("knn_k", "k (neighbours)", value = 3, min = 1, step = 1)
          ),
          # nnet
          box(title = "NN", width = 3, status = "info", solidHeader = TRUE,
            numericInput("nnet_size",  "Hidden units", value = 4,    min = 1,   step = 1),
            numericInput("nnet_decay", "Weight decay", value = 0.01, min = 0,   step = 0.005),
            numericInput("nnet_maxit", "Max iter",     value = 500,  min = 100, step = 100)
          ),
          # GPR
          box(title = "GPR", width = 3, status = "info", solidHeader = TRUE,
            numericInput("gpr_var", "Noise variance", value = 0.1, min = 0.001, step = 0.05)
          )
        ),
        fluidRow(
          # GBM
          box(title = "GBM", width = 4, status = "info", solidHeader = TRUE,
            numericInput("gbm_ntrees",  "n.trees",           value = 100,  min = 10,  step = 10),
            numericInput("gbm_depth",   "interaction.depth", value = 2,    min = 1,   step = 1),
            numericInput("gbm_shrink",  "shrinkage",         value = 0.05, min = 0.001, step = 0.01),
            numericInput("gbm_minnobs", "n.minobsinnode",    value = 1,    min = 1,   step = 1)
          ),
          # Decision Tree
          box(title = "DT (rpart)", width = 4, status = "info", solidHeader = TRUE,
            numericInput("dt_maxdepth", "maxdepth", value = 3,    min = 1, step = 1),
            numericInput("dt_minsplit", "minsplit", value = 2,    min = 1, step = 1),
            numericInput("dt_cp",       "cp",       value = 0.01, min = 0, step = 0.005)
          ),
          # XAI
          box(title = "XAI Parameters", width = 4, status = "danger", solidHeader = TRUE,
            numericInput("lime_nbins", "LIME n_bins",             value = 4,    min = 2,  step = 1),
            numericInput("lime_nperm", "LIME n_permutations",     value = 2000, min = 100, step = 100),
            numericInput("perm_nrep",  "Permutation repetitions", value = 50,   min = 10, step = 10)
          )
        )
      ),

      # -----------------------------------------------------------------------
      # 4. MODEL TRAINING & RESULTS
      # -----------------------------------------------------------------------
      tabItem(tabName = "models",
        fluidRow(
          box(title = "Active Settings & Run", width = 4, status = "primary", solidHeader = TRUE,
            verbatimTextOutput("settings_summary"),
            hr(),
            actionButton("run_models", "Run All Models", icon = icon("play"),
                         class = "btn-success btn-lg", width = "100%"),
            hr(),
            verbatimTextOutput("selected_features")
          ),
          box(title = "Regression Metrics (CV)", width = 8, status = "success", solidHeader = TRUE,
            DTOutput("metrics_table"),
            downloadButton("dl_metrics_table", "Download CSV", class = "download-btn")
          )
        ),
        fluidRow(
          box(title = "Model Comparison (RMSE & R2)", width = 12, status = "info", solidHeader = TRUE,
            plotOutput("plot_model_comp", height = "400px"),
            downloadButton("dl_model_comp", "Download PNG (600 DPI)", class = "download-btn")
          )
        ),
        fluidRow(
          box(title = "Scatter: Actual vs Predicted", width = 12, status = "info", solidHeader = TRUE,
            plotOutput("plot_scatter", height = "550px"),
            downloadButton("dl_scatter", "Download PNG (600 DPI)", class = "download-btn")
          )
        ),
        fluidRow(
          box(title = "DT", width = 12, status = "warning", solidHeader = TRUE,
            plotOutput("plot_dt", height = "450px"),
            downloadButton("dl_dt", "Download PNG (600 DPI)", class = "download-btn")
          )
        )
      ),

      # -----------------------------------------------------------------------
      # 5. XAI - SHAP
      # -----------------------------------------------------------------------
      tabItem(tabName = "shap",
        fluidRow(
          box(title = "RF SHAP - Bar", width = 6, status = "primary", solidHeader = TRUE,
            plotOutput("plot_shap_rf_bar", height = "380px"),
            downloadButton("dl_shap_rf_bar", "Download PNG (600 DPI)", class = "download-btn")
          ),
          box(title = "RF SHAP - Beeswarm", width = 6, status = "primary", solidHeader = TRUE,
            plotOutput("plot_shap_rf_bee", height = "380px"),
            downloadButton("dl_shap_rf_bee", "Download PNG (600 DPI)", class = "download-btn")
          )
        ),
        fluidRow(
          box(title = "XGBoost SHAP - Bar", width = 6, status = "info", solidHeader = TRUE,
            plotOutput("plot_shap_xgb_bar", height = "380px"),
            downloadButton("dl_shap_xgb_bar", "Download PNG (600 DPI)", class = "download-btn")
          ),
          box(title = "XGBoost SHAP - Beeswarm", width = 6, status = "info", solidHeader = TRUE,
            plotOutput("plot_shap_xgb_bee", height = "380px"),
            downloadButton("dl_shap_xgb_bee", "Download PNG (600 DPI)", class = "download-btn")
          )
        )
      ),

      # -----------------------------------------------------------------------
      # 6. XAI - PERMUTATION
      # -----------------------------------------------------------------------
      tabItem(tabName = "perm",
        fluidRow(
          box(title = "Permutation Importance (RF, SVM, GBM)", width = 12,
              status = "primary", solidHeader = TRUE,
            plotOutput("plot_perm", height = "450px"),
            downloadButton("dl_perm", "Download PNG (600 DPI)", class = "download-btn")
          )
        )
      ),

      # -----------------------------------------------------------------------
      # 7. XAI - PDP
      # -----------------------------------------------------------------------
      tabItem(tabName = "pdp_tab",
        fluidRow(
          box(title = "Partial Dependence Plots (Top 2 SHAP Features)", width = 12,
              status = "primary", solidHeader = TRUE,
            plotOutput("plot_pdp", height = "380px"),
            downloadButton("dl_pdp", "Download PNG (600 DPI)", class = "download-btn")
          )
        ),
        fluidRow(
          box(title = "2D Interaction PDP", width = 12, status = "info", solidHeader = TRUE,
            plotOutput("plot_pdp2d", height = "400px"),
            downloadButton("dl_pdp2d", "Download PNG (600 DPI)", class = "download-btn")
          )
        )
      ),

      # -----------------------------------------------------------------------
      # 8. XAI - LIME
      # -----------------------------------------------------------------------
      tabItem(tabName = "lime_tab",
        fluidRow(
          box(title = "LIME Explanations (All Samples)", width = 12,
              status = "primary", solidHeader = TRUE,
            plotOutput("plot_lime", height = "550px"),
            downloadButton("dl_lime", "Download PNG (600 DPI)", class = "download-btn")
          )
        )
      ),

      # -----------------------------------------------------------------------
      # 9. RF VARIABLE IMPORTANCE
      # -----------------------------------------------------------------------
      tabItem(tabName = "rf_imp",
        fluidRow(
          box(title = "RF Variable Importance (% Inc MSE)", width = 12,
              status = "primary", solidHeader = TRUE,
            plotOutput("plot_rf_imp", height = "420px"),
            downloadButton("dl_rf_imp", "Download PNG (600 DPI)", class = "download-btn")
          )
        )
      ),

      # -----------------------------------------------------------------------
      # 10. PREDICTION
      # -----------------------------------------------------------------------
      tabItem(tabName = "predict",
        fluidRow(
          box(title = "Manual Feature Input", width = 5, status = "primary", solidHeader = TRUE,
            helpText("Enter feature values below. Fields are pre-filled with training-set medians once models are trained."),
            hr(),
            uiOutput("pred_inputs"),
            hr(),
            fluidRow(
              column(6, selectInput("pred_model", "Model",
                choices = c(
                  "RF"                = "rf",
                  "LR"                = "lm",
                  "DT"                = "dt",
                  "SVM"               = "svm",
                  "XGBoost"           = "xgb",
                  "KNN"               = "knn",
                  "NN"                = "nnet",
                  "GBM"               = "gbm",
                  "GPR"               = "gpr"
                )
              )),
              column(6, br(), actionButton("run_pred", "Predict", icon = icon("bullseye"),
                                           class = "btn-primary btn-lg", width = "100%"))
            )
          ),
          box(title = "Prediction Result", width = 7, status = "success", solidHeader = TRUE,
            tags$div(style = "text-align:center; padding: 20px;",
              tags$h2("Predicted Glucose Concentration"),
              tags$div(style = "font-size: 64px; font-weight: bold; color: #27ae60;",
                textOutput("pred_value")
              ),
              tags$h3("mg/dL"),
              hr(),
              tags$h4("All Models Comparison"),
              DTOutput("pred_all_table"),
              downloadButton("dl_pred_all", "Download CSV", class = "download-btn"),
              hr(),
              plotOutput("pred_bar_plot", height = "320px"),
              downloadButton("dl_pred_bar", "Download PNG (600 DPI)", class = "download-btn")
            )
          )
        )
      )
    )
  )
)

# =============================================================================
# SERVER
# =============================================================================

server <- function(input, output, session) {

  rv <- reactiveValues(
    df           = NULL, feature_list = NULL,
    X            = NULL, X_scaled     = NULL, y_reg = NULL,
    top_features = NULL, reg_comp     = NULL, loocv_all = NULL,
    lm_full      = NULL, dt_full      = NULL,
    rf_full      = NULL, svm_full     = NULL,
    xgb_full     = NULL, gbm_full     = NULL,
    nnet_full    = NULL, gpr_full     = NULL,
    shap_rf      = NULL, shap_xgb     = NULL,
    perm_all     = NULL, pdp_plots    = NULL,
    p_pdp2d      = NULL, lime_expl    = NULL,
    rf_imp_df    = NULL, models_ready = FALSE
  )

  # ---- Settings summary ----
  output$settings_summary <- renderText({
    cv_lbl <- switch(input$cv_method,
      loocv = "LOOCV",
      kfold = paste0(input$kfold_k, "-Fold CV"),
      tt    = paste0("Train/Test (", round(input$tt_ratio*100), "/",
                     round((1-input$tt_ratio)*100), "%)")
    )
    paste0(
      "Seed        : ", input$seed_val,    "\n",
      "Top N feats : ", input$top_n,       "\n",
      "CV method   : ", cv_lbl,            "\n",
      "RF ntree    : ", input$rf_ntree,    "\n",
      "RF mtry     : ", ifelse(input$rf_mtry==0,"auto",input$rf_mtry), "\n",
      "XGB nrounds : ", input$xgb_nrounds, "\n",
      "XGB depth   : ", input$xgb_max_depth,"\n",
      "XGB eta     : ", input$xgb_eta,     "\n",
      "SVM cost    : ", input$svm_cost,    "  eps: ", input$svm_epsilon, "\n",
      "KNN k       : ", input$knn_k,       "\n",
      "nnet size   : ", input$nnet_size,   "  decay: ", input$nnet_decay, "\n",
      "GBM ntrees  : ", input$gbm_ntrees,  "  depth: ", input$gbm_depth, "\n",
      "GPR var     : ", input$gpr_var,     "\n",
      "DT maxdepth : ", input$dt_maxdepth, "  cp: ", input$dt_cp, "\n",
      "LIME bins   : ", input$lime_nbins,  "  perm: ", input$lime_nperm, "\n",
      "Perm reps   : ", input$perm_nrep
    )
  })

  output$tt_info <- renderText({
    req(rv$feature_list)
    n  <- nrow(rv$feature_list)
    tr <- floor(n * input$tt_ratio)
    te <- n - tr
    paste0("n = ", n, "   Train = ", tr, "   Test = ", te)
  })

  # ---- Data loading ----
  observeEvent(input$datafile, {
    req(input$datafile)
    withProgress(message = "Loading data...", {
      tryCatch({
        df <- load_data(input$datafile$datapath)
        rv$df <- df
        feature_list <- df %>%
          dplyr::group_by(Glucose) %>%
          dplyr::group_modify(~ extract_features(.x$Time_sec, .x$Deflection)) %>%
          dplyr::ungroup()
        rv$feature_list <- feature_list
      }, error = function(e) {
        showNotification(paste("Error:", e$message), type = "error", duration = 10)
      })
    })
  })

  output$data_summary <- renderText({
    req(rv$df)
    df <- rv$df
    paste0(
      "Total observations : ", nrow(df), "\n",
      "Glucose levels     : ", paste(levels(df$Glucose), collapse = ", "), "\n\n",
      "Obs. per group:\n",
      paste(capture.output(print(table(df$Glucose))), collapse = "\n")
    )
  })

  # ---- EDA ----
  make_ts_all <- reactive({
    req(rv$df)
    ggplot(rv$df, aes(x = Time_sec/60, y = Deflection, color = Glucose)) +
      geom_line(linewidth = 0.8) + geom_point(size = 0.4, alpha = 0.5) +
      scale_color_viridis_d(name = "Glucose (mg/dL)") +
      labs(x = "Time (min)", y = "Deflection (a.u.)") +
      theme_minimal(base_size = 13) + theme(legend.position = "right")
  })
  make_ts_facet <- reactive({
    req(rv$df)
    ggplot(rv$df, aes(x = Time_sec/60, y = Deflection)) +
      geom_line(color = "steelblue", linewidth = 0.7) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
      facet_wrap(~ Glucose, scales = "free_y", labeller = label_both) +
      labs(x = "Time (min)", y = "Deflection (a.u.)") + theme_bw(base_size = 11)
  })
  output$plot_ts_all   <- renderPlot({ make_ts_all()   })
  output$plot_ts_facet <- renderPlot({ make_ts_facet() })
  output$dl_ts_all   <- downloadHandler("plot_timeseries_all.png",
    function(f) ggsave(f, make_ts_all(),   width=10, height=5,  dpi=600))
  output$dl_ts_facet <- downloadHandler("plot_timeseries_facet.png",
    function(f) ggsave(f, make_ts_facet(), width=12, height=8,  dpi=600))

  # ---- Features ----
  output$feature_table <- renderDT({
    req(rv$feature_list)
    datatable(rv$feature_list %>% mutate(across(where(is.numeric), ~ round(., 4))),
              options = list(scrollX = TRUE, pageLength = 10), rownames = FALSE)
  })
  output$dl_feature_table <- downloadHandler("feature_matrix.csv",
    function(f) write.csv(rv$feature_list, f, row.names = FALSE))

  # ---- Correlation ----
  cor_df_r <- reactive({
    req(rv$feature_list)
    fl  <- rv$feature_list
    gn  <- as.numeric(as.character(fl$Glucose))
    fcs <- setdiff(colnames(fl), "Glucose")
    cors <- sapply(fcs, function(col) {
      x <- fl[[col]]
      if (sum(!is.na(x)) > 2) cor(x, gn, use = "complete.obs") else NA_real_
    })
    data.frame(Feature = names(cors), Correlation = cors) %>%
      dplyr::filter(!is.na(Correlation)) %>%
      dplyr::arrange(desc(abs(Correlation)))
  })
  make_cor_plot <- reactive({
    cd <- cor_df_r()
    ggplot(cd, aes(x = reorder(Feature, abs(Correlation)), y = Correlation, fill = Correlation > 0)) +
      geom_col() + coord_flip() +
      scale_fill_manual(values = c("TRUE"="#2196F3","FALSE"="#F44336"),
                        labels = c("Negative","Positive"), name="Direction") +
      labs(x="Feature", y="Correlation (r)") + theme_minimal(base_size=12)
  })
  make_calib_plot <- reactive({
    req(rv$feature_list)
    fl <- rv$feature_list; gn <- as.numeric(as.character(fl$Glucose))
    ggplot(fl, aes(x=gn, y=plateau_mean)) +
      geom_point(size=4, color="steelblue") +
      geom_smooth(method="lm", se=TRUE, color="tomato", linetype="dashed") +
      geom_text(aes(label=Glucose), vjust=-1, size=3.5) +
      labs(x="Water Glucose Level (mg/dL)", y="Plateau Deflection (a.u.)") +
      theme_minimal(base_size=13)
  })
  output$plot_cor   <- renderPlot({ make_cor_plot()   })
  output$plot_calib <- renderPlot({ make_calib_plot() })
  output$dl_cor   <- downloadHandler("plot_feature_correlation.png",
    function(f) ggsave(f, make_cor_plot(),   width=9, height=7, dpi=600))
  output$dl_calib <- downloadHandler("plot_calibration_curve.png",
    function(f) ggsave(f, make_calib_plot(), width=7, height=5, dpi=600))

  output$selected_features <- renderText({
    req(rv$feature_list)
    top <- cor_df_r() %>% dplyr::slice_head(n=input$top_n) %>% dplyr::pull(Feature)
    paste0("Top ", input$top_n, " features:\n", paste(top, collapse="\n"))
  })

  # ====================================================================
  # RUN MODELS
  # ====================================================================
  observeEvent(input$run_models, {
    req(rv$feature_list)

    # Snapshot all inputs
    SEED       <- input$seed_val
    TOP_N      <- input$top_n
    CV_METHOD  <- input$cv_method
    KFOLD_K    <- input$kfold_k
    TT_RATIO   <- input$tt_ratio
    RF_NTREE   <- input$rf_ntree
    RF_MTRY    <- input$rf_mtry
    XGB_NR     <- input$xgb_nrounds
    XGB_DEPTH  <- input$xgb_max_depth
    XGB_ETA    <- input$xgb_eta
    XGB_SUB    <- input$xgb_subsample
    XGB_COL    <- input$xgb_colsample
    SVM_COST   <- input$svm_cost
    SVM_EPS    <- input$svm_epsilon
    KNN_K      <- input$knn_k
    NNET_SIZE  <- input$nnet_size
    NNET_DECAY <- input$nnet_decay
    NNET_MAXIT <- input$nnet_maxit
    GBM_NTREES <- input$gbm_ntrees
    GBM_DEPTH  <- input$gbm_depth
    GBM_SHRINK <- input$gbm_shrink
    GBM_MINN   <- input$gbm_minnobs
    GPR_VAR    <- input$gpr_var
    DT_DEPTH   <- input$dt_maxdepth
    DT_MINSPL  <- input$dt_minsplit
    DT_CP      <- input$dt_cp
    LIME_BINS  <- input$lime_nbins
    LIME_PERM  <- input$lime_nperm
    PERM_NREP  <- input$perm_nrep

    withProgress(message = "Training models... please wait.", value = 0, {

      fl  <- rv$feature_list
      gn  <- as.numeric(as.character(fl$Glucose))
      top_features <- cor_df_r() %>% dplyr::slice_head(n=TOP_N) %>% dplyr::pull(Feature)
      rv$top_features <- top_features

      feat_mat <- fl %>%
        dplyr::select(dplyr::all_of(top_features)) %>%
        dplyr::mutate(dplyr::across(dplyr::everything(),
                                    ~ ifelse(is.na(.), median(., na.rm=TRUE), .)))

      X         <- as.data.frame(feat_mat)
      X_scaled  <- as.data.frame(scale(X))
      y_reg     <- gn
      n_samples <- nrow(X)
      rv$X <- X; rv$X_scaled <- X_scaled; rv$y_reg <- y_reg

      auto_mtry  <- max(1L, floor(ncol(X) / 3L))
      rf_mtry_use <- if (RF_MTRY == 0) auto_mtry else min(as.integer(RF_MTRY), ncol(X))

      # ---- Linear Regression ----
      setProgress(0.05, detail = "LR...")
      lm_fn      <- function(X_tr, y_tr) lm(y ~ ., data = cbind(X_tr, y=y_tr))
      lm_pred_fn <- function(fit, X_te) predict(fit, newdata=X_te)
      set.seed(SEED)
      lm_res     <- run_cv(X_scaled, y_reg, lm_fn, lm_pred_fn, CV_METHOD, KFOLD_K, TT_RATIO, SEED)
      lm_metrics <- reg_metrics(lm_res, "LR", "Whitebox")
      rv$lm_full <- lm(y ~ ., data = cbind(X_scaled, y=y_reg))

      # ---- Decision Tree ----
      setProgress(0.12, detail = "DT...")
      dt_fn <- function(X_tr, y_tr)
        rpart(y ~ ., data=cbind(X_tr, y=y_tr), method="anova",
              control=rpart.control(maxdepth=DT_DEPTH, minsplit=DT_MINSPL, cp=DT_CP))
      dt_pred_fn <- function(fit, X_te) predict(fit, newdata=X_te)
      set.seed(SEED)
      dt_res     <- run_cv(X, y_reg, dt_fn, dt_pred_fn, CV_METHOD, KFOLD_K, TT_RATIO, SEED)
      dt_metrics <- reg_metrics(dt_res, "DT", "Whitebox")
      rv$dt_full <- rpart(y ~ ., data=cbind(X, y=y_reg), method="anova",
                          control=rpart.control(maxdepth=DT_DEPTH, minsplit=DT_MINSPL, cp=DT_CP))

      # ---- Random Forest ----
      setProgress(0.22, detail = "RF...")
      rf_fn <- function(X_tr, y_tr)
        randomForest(x=X_tr, y=y_tr, ntree=RF_NTREE, mtry=min(rf_mtry_use, ncol(X_tr)))
      set.seed(SEED)
      rf_res     <- run_cv(X, y_reg, rf_fn, NULL, CV_METHOD, KFOLD_K, TT_RATIO, SEED)
      rf_metrics <- reg_metrics(rf_res, "RF")

      # ---- SVM ----
      setProgress(0.32, detail = "SVM...")
      svm_fn <- function(X_tr, y_tr)
        e1071::svm(x=X_tr, y=y_tr, kernel="radial", cost=SVM_COST, epsilon=SVM_EPS, scale=TRUE)
      set.seed(SEED)
      svm_res     <- run_cv(X, y_reg, svm_fn, NULL, CV_METHOD, KFOLD_K, TT_RATIO, SEED)
      svm_metrics <- reg_metrics(svm_res, "SVM")

      # ---- XGBoost (special: DMatrix) ----
      setProgress(0.40, detail = "XGBoost...")
      xgb_params <- list(objective="reg:squarederror", max_depth=XGB_DEPTH,
                         eta=XGB_ETA, subsample=XGB_SUB,
                         colsample_bytree=XGB_COL, verbosity=0)
      xgb_train_fn <- function(X_tr, y_tr)
        xgb.train(params=xgb_params,
                  data=xgb.DMatrix(data=as.matrix(X_tr), label=y_tr),
                  nrounds=XGB_NR, verbose=0)
      xgb_pred_fn <- function(fit, X_te) predict(fit, xgb.DMatrix(data=as.matrix(X_te)))

      if (CV_METHOD == "loocv") {
        xgb_preds <- numeric(n_samples)
        set.seed(SEED)
        for (i in seq_len(n_samples))
          xgb_preds[i] <- xgb_pred_fn(xgb_train_fn(X[-i,,drop=F], y_reg[-i]), X[i,,drop=F])
        xgb_res <- data.frame(actual=y_reg, predicted=xgb_preds)
      } else if (CV_METHOD == "kfold") {
        set.seed(SEED)
        folds <- sample(rep(1:KFOLD_K, length.out=n_samples))
        xgb_preds <- numeric(n_samples)
        for (fold in 1:KFOLD_K) {
          itr <- which(folds!=fold); ite <- which(folds==fold)
          m   <- xgb_train_fn(X[itr,,drop=F], y_reg[itr])
          xgb_preds[ite] <- xgb_pred_fn(m, X[ite,,drop=F])
        }
        xgb_res <- data.frame(actual=y_reg, predicted=xgb_preds)
      } else {
        set.seed(SEED)
        itr <- sample(seq_len(n_samples), size=floor(n_samples*TT_RATIO))
        ite <- setdiff(seq_len(n_samples), itr); if (length(ite)==0) ite <- itr
        m   <- xgb_train_fn(X[itr,,drop=F], y_reg[itr])
        xgb_res <- data.frame(actual=y_reg[ite], predicted=xgb_pred_fn(m, X[ite,,drop=F]))
      }
      xgb_metrics <- reg_metrics(xgb_res, "XGBoost")

      # ---- KNN (special: no model object) ----
      setProgress(0.50, detail = "KNN...")
      if (CV_METHOD == "loocv") {
        knn_preds <- numeric(n_samples)
        for (i in seq_len(n_samples))
          knn_preds[i] <- FNN::knn.reg(X_scaled[-i,,drop=F], X_scaled[i,,drop=F],
                                        y_reg[-i], k=KNN_K)$pred
        knn_res <- data.frame(actual=y_reg, predicted=knn_preds)
      } else if (CV_METHOD == "kfold") {
        set.seed(SEED)
        folds <- sample(rep(1:KFOLD_K, length.out=n_samples))
        knn_preds <- numeric(n_samples)
        for (fold in 1:KFOLD_K) {
          itr <- which(folds!=fold); ite <- which(folds==fold)
          knn_preds[ite] <- FNN::knn.reg(X_scaled[itr,,drop=F], X_scaled[ite,,drop=F],
                                          y_reg[itr], k=KNN_K)$pred
        }
        knn_res <- data.frame(actual=y_reg, predicted=knn_preds)
      } else {
        set.seed(SEED)
        itr <- sample(seq_len(n_samples), size=floor(n_samples*TT_RATIO))
        ite <- setdiff(seq_len(n_samples), itr); if (length(ite)==0) ite <- itr
        knn_res <- data.frame(actual=y_reg[ite],
                              predicted=FNN::knn.reg(X_scaled[itr,,drop=F],
                                                     X_scaled[ite,,drop=F],
                                                     y_reg[itr], k=KNN_K)$pred)
      }
      knn_metrics <- reg_metrics(knn_res, "KNN")

      # ---- Neural Network ----
      setProgress(0.58, detail = "NN...")
      nnet_fn <- function(X_tr, y_tr)
        nnet::nnet(y ~ ., data=cbind(as.data.frame(X_tr), y=y_tr),
                   size=NNET_SIZE, linout=TRUE, decay=NNET_DECAY, maxit=NNET_MAXIT, trace=FALSE)
      nnet_pred_fn <- function(fit, X_te) as.numeric(predict(fit, newdata=as.data.frame(X_te)))
      set.seed(SEED)
      nnet_res     <- run_cv(X_scaled, y_reg, nnet_fn, nnet_pred_fn, CV_METHOD, KFOLD_K, TT_RATIO, SEED)
      nnet_metrics <- reg_metrics(nnet_res, "NN")

      # ---- GBM ----
      setProgress(0.66, detail = "GBM...")
      gbm_fn <- function(X_tr, y_tr)
        gbm::gbm(y ~ ., data=cbind(as.data.frame(X_tr), y=y_tr),
                 distribution="gaussian", n.trees=GBM_NTREES,
                 interaction.depth=GBM_DEPTH, shrinkage=GBM_SHRINK,
                 bag.fraction=1.0, n.minobsinnode=GBM_MINN, verbose=FALSE)
      gbm_pred_fn <- function(fit, X_te)
        predict(fit, newdata=as.data.frame(X_te), n.trees=GBM_NTREES)
      set.seed(SEED)
      gbm_res     <- run_cv(X, y_reg, gbm_fn, gbm_pred_fn, CV_METHOD, KFOLD_K, TT_RATIO, SEED)
      gbm_metrics <- reg_metrics(gbm_res, "GBM")

      # ---- GPR ----
      setProgress(0.74, detail = "GPR...")
      gpr_fn <- function(X_tr, y_tr)
        kernlab::gausspr(x=as.matrix(X_tr), y=y_tr, kernel="rbfdot", var=GPR_VAR)
      gpr_pred_fn <- function(fit, X_te) as.numeric(predict(fit, newdata=as.matrix(X_te)))
      set.seed(SEED)
      gpr_res     <- run_cv(X_scaled, y_reg, gpr_fn, gpr_pred_fn, CV_METHOD, KFOLD_K, TT_RATIO, SEED)
      gpr_metrics <- reg_metrics(gpr_res, "GPR")

      # ---- Compile ----
      rv$reg_comp <- bind_rows(lm_metrics, dt_metrics, rf_metrics, svm_metrics,
                               xgb_metrics, knn_metrics, nnet_metrics, gbm_metrics, gpr_metrics) %>%
        dplyr::arrange(RMSE)

      rv$loocv_all <- bind_rows(
        lm_res   %>% mutate(Model="LR",                Type="Whitebox"),
        dt_res   %>% mutate(Model="DT",                Type="Whitebox"),
        rf_res   %>% mutate(Model="RF",                Type="Blackbox"),
        svm_res  %>% mutate(Model="SVM",               Type="Blackbox"),
        xgb_res  %>% mutate(Model="XGBoost",           Type="Blackbox"),
        knn_res  %>% mutate(Model="KNN",               Type="Blackbox"),
        nnet_res %>% mutate(Model="NN",                Type="Blackbox"),
        gbm_res  %>% mutate(Model="GBM",               Type="Blackbox"),
        gpr_res  %>% mutate(Model="GPR",               Type="Blackbox")
      )

      # ---- Full-dataset models for XAI ----
      setProgress(0.78, detail = "Full-dataset models for XAI...")
      set.seed(SEED)
      rf_full  <- randomForest(x=X, y=y_reg, ntree=RF_NTREE,
                               mtry=min(rf_mtry_use, ncol(X)), importance=TRUE)
      rv$rf_full <- rf_full
      rv$svm_full <- e1071::svm(x=X, y=y_reg, kernel="radial",
                                cost=SVM_COST, epsilon=SVM_EPS, scale=TRUE)
      rv$xgb_full <- xgb.train(params=xgb_params,
                                data=xgb.DMatrix(data=as.matrix(X), label=y_reg),
                                nrounds=XGB_NR, verbose=0)
      set.seed(SEED)
      rv$gbm_full <- gbm::gbm(y ~ ., data=cbind(X, y=y_reg),
                               distribution="gaussian", n.trees=GBM_NTREES,
                               interaction.depth=GBM_DEPTH, shrinkage=GBM_SHRINK,
                               bag.fraction=1.0, n.minobsinnode=GBM_MINN, verbose=FALSE)

      # nnet and GPR full-dataset models (needed for prediction tab)
      set.seed(SEED)
      rv$nnet_full <- nnet::nnet(y ~ ., data=cbind(as.data.frame(X_scaled), y=y_reg),
                                 size=NNET_SIZE, linout=TRUE, decay=NNET_DECAY,
                                 maxit=NNET_MAXIT, trace=FALSE)
      rv$gpr_full  <- kernlab::gausspr(x=as.matrix(X_scaled), y=y_reg,
                                       kernel="rbfdot", var=GPR_VAR)

      rv$rf_imp_df <- importance(rf_full, type=1) %>% as.data.frame() %>%
        tibble::rownames_to_column("Feature") %>%
        dplyr::rename(Importance=`%IncMSE`) %>% dplyr::arrange(desc(Importance))

      # ---- SHAP ----
      setProgress(0.82, detail = "SHAP values...")
      rf_pred_wr <- function(model, newdata) predict(model, newdata=as.data.frame(newdata))
      set.seed(SEED)
      ks_rf    <- kernelshap(rf_full, X=as.matrix(X), pred_fun=rf_pred_wr, bg_X=as.matrix(X))
      rv$shap_rf  <- shapviz(ks_rf)
      rv$shap_xgb <- shapviz(rv$xgb_full, X_pred=as.matrix(X))

      # ---- Permutation ----
      # iml predict.function signature: function(model, newdata) -> numeric vector
      setProgress(0.87, detail = "Permutation Importance...")
      rf_iml_fn  <- function(model, newdata)
        as.numeric(predict(model, newdata = as.data.frame(newdata)))
      svm_iml_fn <- function(model, newdata)
        as.numeric(predict(model, newdata = as.data.frame(newdata)))
      gbm_iml_fn <- function(model, newdata)
        as.numeric(predict(model, newdata = as.data.frame(newdata), n.trees = GBM_NTREES))

      mk_pred <- function(m, fn)
        iml::Predictor$new(m, data = X, y = y_reg, predict.function = fn)

      perm_rf  <- iml::FeatureImp$new(mk_pred(rv$rf_full,  rf_iml_fn),
                                      loss = "rmse", n.repetitions = PERM_NREP)
      perm_svm <- iml::FeatureImp$new(mk_pred(rv$svm_full, svm_iml_fn),
                                      loss = "rmse", n.repetitions = PERM_NREP)
      perm_gbm <- iml::FeatureImp$new(mk_pred(rv$gbm_full, gbm_iml_fn),
                                      loss = "rmse", n.repetitions = PERM_NREP)
      rv$perm_all <- bind_rows(normalise_perm(perm_rf$results,  "RF"),
                               normalise_perm(perm_svm$results, "SVM"),
                               normalise_perm(perm_gbm$results, "GBM"))

      # ---- PDP ----
      setProgress(0.91, detail = "PDPs...")
      top2 <- colMeans(abs(rv$shap_rf$S)) %>% sort(decreasing=TRUE) %>% head(2) %>% names()
      rv$pdp_plots <- lapply(top2, function(feat) {
        pd <- pdp::partial(rf_full, pred.var=feat, train=X, plot=FALSE, grid.resolution=20)
        ggplot(pd, aes_string(x=feat, y="yhat")) +
          geom_line(color="steelblue", linewidth=1.2) +
          geom_rug(data=X, aes_string(x=feat), sides="b", alpha=0.5, inherit.aes=FALSE) +
          labs(x=feat, y="Predicted Glucose (mg/dL)") + theme_minimal(base_size=12)
      })
      if (length(top2) == 2) {
        pd2d <- pdp::partial(rf_full, pred.var=top2, train=X, plot=FALSE, grid.resolution=15)
        rv$p_pdp2d <- ggplot(pd2d, aes_string(x=top2[1], y=top2[2], fill="yhat")) +
          geom_tile() + scale_fill_viridis_c(name="Predicted\nGlucose\n(mg/dL)") +
          labs(x=top2[1], y=top2[2]) + theme_minimal(base_size=12)
      }

      # ---- LIME ----
      setProgress(0.95, detail = "LIME...")
      model_type.randomForest    <- function(x, ...) "regression"
      predict_model.randomForest <- function(x, newdata, type, ...)
        data.frame(Response=predict(x, newdata=newdata))
      registerS3method("model_type",    "randomForest", model_type.randomForest,
                       envir=asNamespace("lime"))
      registerS3method("predict_model", "randomForest", predict_model.randomForest,
                       envir=asNamespace("lime"))
      le <- lime::lime(x=X, model=rv$rf_full, bin_continuous=TRUE, n_bins=LIME_BINS)
      rv$lime_expl <- lime::explain(x=X, explainer=le,
                                    n_features=ncol(X), n_permutations=LIME_PERM,
                                    feature_select="auto")

      rv$models_ready <- TRUE
      setProgress(1, detail="Done!")
      showNotification("All models trained successfully!", type="message", duration=5)
    })
  })

  # ---- Metrics table ----
  output$metrics_table <- renderDT({
    req(rv$reg_comp)
    datatable(rv$reg_comp %>% mutate(across(where(is.numeric), ~ round(.,4))),
              options=list(pageLength=10), rownames=FALSE)
  })
  output$dl_metrics_table <- downloadHandler("regression_metrics.csv",
    function(f) write.csv(rv$reg_comp, f, row.names=FALSE))

  # ---- SHAP helpers ----
  make_shap_bar <- function(shap_obj) {
    ma <- sort(colMeans(abs(shap_obj$S)), decreasing=FALSE)
    ggplot(data.frame(Feature=names(ma), V=as.numeric(ma)),
           aes(x=reorder(Feature,V), y=V)) +
      geom_col(fill="steelblue", alpha=0.85) + coord_flip() +
      labs(x="Feature", y="Mean |SHAP|") + theme_minimal(base_size=12)
  }
  make_shap_bee <- function(shap_obj, X_df) {
    sm  <- shap_obj$S
    ord <- names(sort(colMeans(abs(sm)), decreasing=FALSE))
    dl  <- as.data.frame(sm) %>% mutate(obs=seq_len(nrow(sm))) %>%
      pivot_longer(-obs, names_to="Feature", values_to="SHAP") %>%
      left_join(as.data.frame(X_df) %>% mutate(obs=seq_len(nrow(X_df))) %>%
                  pivot_longer(-obs, names_to="Feature", values_to="FV"),
                by=c("obs","Feature")) %>%
      mutate(Feature=factor(Feature, levels=ord))
    ggplot(dl, aes(x=SHAP, y=Feature, color=FV)) +
      geom_jitter(height=0.15, size=2.5, alpha=0.85) +
      scale_color_viridis_c(name="Feature\nValue") +
      geom_vline(xintercept=0, linetype="dashed", color="gray40") +
      labs(x="SHAP Value", y="Feature") + theme_minimal(base_size=12)
  }
  output$plot_shap_rf_bar  <- renderPlot({ req(rv$shap_rf);        make_shap_bar(rv$shap_rf) })
  output$plot_shap_rf_bee  <- renderPlot({ req(rv$shap_rf,rv$X);   make_shap_bee(rv$shap_rf,  rv$X) })
  output$plot_shap_xgb_bar <- renderPlot({ req(rv$shap_xgb);       make_shap_bar(rv$shap_xgb) })
  output$plot_shap_xgb_bee <- renderPlot({ req(rv$shap_xgb,rv$X);  make_shap_bee(rv$shap_xgb, rv$X) })
  output$dl_shap_rf_bar  <- downloadHandler("plot_shap_rf_bar.png",
    function(f){ req(rv$shap_rf);       ggsave(f,make_shap_bar(rv$shap_rf),       width=7,height=5,dpi=600)})
  output$dl_shap_rf_bee  <- downloadHandler("plot_shap_rf_beeswarm.png",
    function(f){ req(rv$shap_rf,rv$X);  ggsave(f,make_shap_bee(rv$shap_rf,rv$X),  width=8,height=5,dpi=600)})
  output$dl_shap_xgb_bar <- downloadHandler("plot_shap_xgb_bar.png",
    function(f){ req(rv$shap_xgb);      ggsave(f,make_shap_bar(rv$shap_xgb),      width=7,height=5,dpi=600)})
  output$dl_shap_xgb_bee <- downloadHandler("plot_shap_xgb_beeswarm.png",
    function(f){ req(rv$shap_xgb,rv$X); ggsave(f,make_shap_bee(rv$shap_xgb,rv$X),width=8,height=5,dpi=600)})

  # ---- Model comparison ----
  make_model_comp <- reactive({
    req(rv$reg_comp); rc <- rv$reg_comp
    p1 <- ggplot(rc, aes(x=reorder(Model,RMSE),y=RMSE,fill=Model,alpha=Type)) +
      geom_col(width=0.7) + geom_text(aes(label=round(RMSE,2)),hjust=-0.15,size=3.5) +
      coord_flip() + scale_fill_manual(values=model_colors,guide="none") +
      scale_alpha_manual(values=c(Whitebox=0.6,Blackbox=1.0),name="Type") +
      labs(x="Model",y="RMSE (mg/dL)") + theme_minimal(base_size=12) +
      theme(legend.position="bottom")
    p2 <- ggplot(rc, aes(x=reorder(Model,Rsq),y=Rsq,fill=Model,alpha=Type)) +
      geom_col(width=0.7) + geom_text(aes(label=round(Rsq,3)),hjust=-0.15,size=3.5) +
      coord_flip() + scale_fill_manual(values=model_colors,guide="none") +
      scale_alpha_manual(values=c(Whitebox=0.6,Blackbox=1.0),name="Type") +
      scale_y_continuous(limits=c(0,1.1)) +
      labs(x="Model",y="R-squared") + theme_minimal(base_size=12) +
      theme(legend.position="bottom")
    gridExtra::arrangeGrob(p1,p2,ncol=2)
  })
  make_scatter <- reactive({
    req(rv$loocv_all)
    ggplot(rv$loocv_all, aes(x=actual,y=predicted,color=Model)) +
      geom_abline(slope=1,intercept=0,linetype="dashed",color="gray40") +
      geom_point(size=2.5,alpha=0.9) +
      scale_color_manual(values=model_colors) +
      facet_wrap(~Model,nrow=3) +
      labs(x="Actual Glucose (mg/dL)",y="Predicted Glucose (mg/dL)") +
      theme_bw(base_size=11) + theme(legend.position="none")
  })
  output$plot_model_comp <- renderPlot({ grid.arrange(make_model_comp()) })
  output$plot_scatter    <- renderPlot({ make_scatter() })
  output$plot_dt <- renderPlot({
    req(rv$dt_full)
    rpart.plot(rv$dt_full, type=4, extra=101, fallen.leaves=TRUE, tweak=1.2, roundint=FALSE)
  })
  output$dl_model_comp <- downloadHandler("plot_model_comparison.png",
    function(f) ggsave(f, make_model_comp(), width=14, height=6, dpi=600))
  output$dl_scatter <- downloadHandler("plot_loocv_scatter.png",
    function(f) ggsave(f, make_scatter(), width=14, height=10, dpi=600))
  output$dl_dt <- downloadHandler("plot_decision_tree.png", function(f) {
    req(rv$dt_full)
    png(f, width=3600, height=2400, res=600)
    rpart.plot(rv$dt_full, type=4, extra=101, fallen.leaves=TRUE, tweak=1.2, roundint=FALSE)
    dev.off()
  })

  # ---- Permutation ----
  make_perm <- reactive({
    req(rv$perm_all)
    ggplot(rv$perm_all, aes(x=reorder(Feature,Importance),y=Importance,fill=Model)) +
      geom_col(position="dodge",alpha=0.85) + coord_flip() +
      scale_fill_brewer(palette="Set2") +
      labs(x="Feature",y="Permutation Importance (RMSE ratio)") +
      theme_minimal(base_size=12) + theme(legend.position="bottom")
  })
  output$plot_perm <- renderPlot({ make_perm() })
  output$dl_perm   <- downloadHandler("plot_permutation_importance.png",
    function(f) ggsave(f, make_perm(), width=9, height=6, dpi=600))

  # ---- PDP ----
  output$plot_pdp <- renderPlot({ req(rv$pdp_plots); grid.arrange(grobs=rv$pdp_plots, ncol=2) })
  output$dl_pdp   <- downloadHandler("plot_pdp.png", function(f) {
    req(rv$pdp_plots)
    ggsave(f, gridExtra::arrangeGrob(grobs=rv$pdp_plots,ncol=2), width=10, height=5, dpi=600)
  })
  output$plot_pdp2d <- renderPlot({ req(rv$p_pdp2d); rv$p_pdp2d })
  output$dl_pdp2d   <- downloadHandler("plot_pdp_2d_interaction.png",
    function(f){ req(rv$p_pdp2d); ggsave(f, rv$p_pdp2d, width=7, height=5, dpi=600) })

  # ---- LIME ----
  make_lime <- reactive({
    req(rv$lime_expl)
    lime::plot_features(rv$lime_expl, ncol=4) + theme_minimal(base_size=10)
  })
  output$plot_lime <- renderPlot({ make_lime() })
  output$dl_lime   <- downloadHandler("plot_lime_explanations.png",
    function(f) ggsave(f, make_lime(), width=16, height=10, dpi=600))

  # ---- RF Variable Importance ----
  make_rf_imp <- reactive({
    req(rv$rf_imp_df)
    ggplot(rv$rf_imp_df, aes(x=reorder(Feature,Importance),y=Importance)) +
      geom_col(fill="steelblue",alpha=0.85) + coord_flip() +
      labs(x="Feature",y="% Increase in MSE") + theme_minimal(base_size=12)
  })
  output$plot_rf_imp <- renderPlot({ make_rf_imp() })
  output$dl_rf_imp   <- downloadHandler("plot_feature_importance_rf.png",
    function(f) ggsave(f, make_rf_imp(), width=8, height=5, dpi=600))

  # ==========================================================================
  # PREDICTION TAB
  # ==========================================================================

  # Dynamically render one numericInput per top feature, pre-filled with median
  output$pred_inputs <- renderUI({
    req(rv$X, rv$top_features)
    X <- rv$X
    lapply(rv$top_features, function(feat) {
      med_val <- round(median(X[[feat]], na.rm = TRUE), 5)
      numericInput(
        inputId = paste0("pinp_", feat),
        label   = feat,
        value   = med_val,
        step    = round(sd(X[[feat]], na.rm = TRUE) / 10, 6)
      )
    })
  })

  # Helper: collect user input row as a data.frame
  get_input_row <- reactive({
    req(rv$top_features, rv$X)
    vals <- sapply(rv$top_features, function(feat) {
      v <- input[[paste0("pinp_", feat)]]
      if (is.null(v)) median(rv$X[[feat]], na.rm = TRUE) else as.numeric(v)
    })
    as.data.frame(t(vals))
  })

  # Helper: predict with a single model
  single_pred <- function(model_id, new_row, new_row_scaled) {
    tryCatch({
      switch(model_id,
        rf   = as.numeric(predict(rv$rf_full,  newdata = new_row)),
        lm   = as.numeric(predict(rv$lm_full,  newdata = new_row_scaled)),
        dt   = as.numeric(predict(rv$dt_full,  newdata = new_row)),
        svm  = as.numeric(predict(rv$svm_full, newdata = new_row)),
        xgb  = as.numeric(predict(rv$xgb_full,
                 xgb.DMatrix(data = as.matrix(new_row)))),
        knn  = {
          req(rv$X_scaled, rv$y_reg)
          as.numeric(FNN::knn.reg(
            train = rv$X_scaled, test = new_row_scaled,
            y = rv$y_reg, k = isolate(input$knn_k))$pred)
        },
        nnet = {
          req(rv$nnet_full)
          as.numeric(predict(rv$nnet_full, newdata = new_row_scaled))
        },
        gbm  = {
          ntrees <- isolate(input$gbm_ntrees)
          as.numeric(predict(rv$gbm_full, newdata = new_row, n.trees = ntrees))
        },
        gpr  = as.numeric(predict(rv$gpr_full, newdata = as.matrix(new_row_scaled))),
        NA_real_
      )
    }, error = function(e) NA_real_)
  }

  # Reactive: predictions from all models
  all_preds <- eventReactive(input$run_pred, {
    req(rv$models_ready, rv$X, rv$X_scaled, rv$y_reg)
    new_row        <- get_input_row()
    # Scale using training-set parameters
    train_means    <- colMeans(rv$X)
    train_sds      <- apply(rv$X, 2, sd)
    train_sds[train_sds == 0] <- 1
    new_row_scaled <- as.data.frame(t((unlist(new_row) - train_means) / train_sds))
    colnames(new_row_scaled) <- colnames(new_row)

    models <- c("rf","lm","dt","svm","xgb","knn","nnet","gbm","gpr")
    labels <- c("RF","LR","DT",
                "SVM","XGBoost","KNN","NN","GBM","GPR")
    types  <- c("Blackbox","Whitebox","Whitebox",
                "Blackbox","Blackbox","Blackbox","Blackbox","Blackbox","Blackbox")

    preds <- sapply(models, function(m) single_pred(m, new_row, new_row_scaled))

    data.frame(
      Model      = labels,
      Type       = types,
      Prediction = round(preds, 2),
      stringsAsFactors = FALSE
    )
  })

  # Selected model prediction display
  output$pred_value <- renderText({
    df <- all_preds()
    sel_label <- switch(input$pred_model,
      rf="RF", lm="LR", dt="DT",
      svm="SVM", xgb="XGBoost", knn="KNN", nnet="NN",
      gbm="GBM", gpr="GPR")
    val <- df$Prediction[df$Model == sel_label]
    if (length(val) == 0 || is.na(val)) "N/A" else format(round(val, 2), nsmall = 2)
  })

  # All models table
  output$pred_all_table <- renderDT({
    req(all_preds())
    datatable(all_preds(),
              options = list(pageLength = 9, dom = "t"),
              rownames = FALSE) %>%
      formatStyle("Prediction",
        background = styleColorBar(range(all_preds()$Prediction, na.rm=TRUE), "#a8d8a8"),
        backgroundSize = "90% 70%", backgroundRepeat = "no-repeat",
        backgroundPosition = "center")
  })

  output$dl_pred_all <- downloadHandler("predictions.csv",
    function(f) write.csv(all_preds(), f, row.names = FALSE))

  # Bar plot of all model predictions
  make_pred_bar <- reactive({
    req(all_preds())
    df  <- all_preds()
    sel_label <- switch(input$pred_model,
      rf="RF", lm="LR", dt="DT",
      svm="SVM", xgb="XGBoost", knn="KNN", nnet="NN",
      gbm="GBM", gpr="GPR")
    df$Selected <- df$Model == sel_label
    ggplot(df, aes(x = reorder(Model, Prediction), y = Prediction,
                   fill = Model, alpha = Selected)) +
      geom_col(width = 0.7) +
      geom_text(aes(label = round(Prediction, 1)), hjust = -0.2, size = 4) +
      coord_flip() +
      scale_fill_manual(values = model_colors, guide = "none") +
      scale_alpha_manual(values = c("TRUE" = 1.0, "FALSE" = 0.45), guide = "none") +
      scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
      labs(x = "Model", y = "Predicted Glucose (mg/dL)",
           title = "All Models Prediction Comparison") +
      theme_minimal(base_size = 12) +
      theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  })

  output$pred_bar_plot <- renderPlot({ make_pred_bar() })
  output$dl_pred_bar   <- downloadHandler("plot_prediction_comparison.png",
    function(f) ggsave(f, make_pred_bar(), width = 9, height = 5, dpi = 600))
}

# =============================================================================
shinyApp(ui = ui, server = server)
