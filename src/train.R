#!/usr/bin/env Rscript
# ------------------------------------------------------------------------------
# src/train.R
#
# Isolated, reproducible training of the Agibank credit risk model.
#
# Pipeline:
#   1. Resolve CLI configuration / environment variables
#   2. Load RDS feature store (data/feature_store.rds)
#   3. Extract target and predictor features (excluding SK_ID_CURR & is_train)
#   4. Impute missing values (median for numeric, mode for categorical)
#   5. Train Random Forest (ranger) predicting TARGET
#   6. Evaluate performance on hold-out validation split
#   7. Save model artifact (models/model_v1.rds)
#   8. Export feature store metadata schema (data/features_metadata.json)
# ------------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(jsonlite)
  library(optparse)
  library(parallel)
  library(ranger)
})

# --- 1. Configuration ---------------------------------------------------------

option_list <- list(
  make_option(c("-d", "--data-path"),
    type = "character",
    default = Sys.getenv("DATA_PATH", "data/feature_store.rds"),
    dest = "data_path",
    help = "Path to input RDS feature store [default: %default]"
  ),
  make_option(c("-o", "--model-output-path"),
    type = "character",
    default = Sys.getenv("MODEL_OUTPUT_PATH", "models/model_v1.rds"),
    dest = "model_output_path",
    help = "Path where the model artifact is saved [default: %default]"
  ),
  make_option(c("-m", "--metadata-output-path"),
    type = "character",
    default = Sys.getenv("METADATA_OUTPUT_PATH", "data/features_metadata.json"),
    dest = "metadata_output_path",
    help = "Path where feature metadata JSON is saved [default: %default]"
  ),
  make_option(c("-l", "--log-level"),
    type = "character",
    default = Sys.getenv("LOG_LEVEL", "INFO"),
    dest = "log_level",
    help = "Log level: TRACE|DEBUG|INFO|WARN|ERROR|FATAL [default: %default]"
  ),
  make_option("--n-folds",
    type = "integer",
    default = 5L,
    dest = "n_folds",
    help = "Number of cross-validation folds [default: %default]"
  ),
  make_option("--test-set-path",
    type = "character",
    default = Sys.getenv("TEST_SET_PATH", "data/test_set.rds"),
    dest = "test_set_path",
    help = "Path to test set RDS for final evaluation [default: %default]"
  ),
  make_option("--performance-output-path",
    type = "character",
    default = Sys.getenv("PERFORMANCE_OUTPUT_PATH", "models/performance_result.json"),
    dest = "performance_output_path",
    help = "Path where performance result JSON is saved [default: %default]"
  ),
  make_option("--num-trees",
    type = "integer",
    default = as.integer(Sys.getenv("NUM_TREES", "300")),
    dest = "num_trees",
    help = "Number of trees in the Random Forest [default: %default]"
  ),
  make_option("--seed",
    type = "integer",
    default = 42,
    dest = "seed",
    help = "Random seed for reproducibility [default: %default]"
  )
)

opt <- parse_args(OptionParser(option_list = option_list))
set.seed(opt$seed)

# --- Structured JSON logging --------------------------------------------------

log_json <- function(level, message, ...) {
  entry <- list(
    timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
    level = level,
    logger = "train",
    message = message,
    ...
  )
  cat(toJSON(entry, auto_unbox = TRUE), "\n")
}

known_levels <- c("TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL")
level_rank <- c(TRACE = 0, DEBUG = 1, INFO = 2, WARN = 3, ERROR = 4, FATAL = 5)

requested_level <- toupper(opt$log_level)
if (!requested_level %in% known_levels) {
  log_json("WARN", paste("Unknown LOG_LEVEL:", opt$log_level, "- falling back to INFO"))
  requested_level <- "INFO"
}
min_rank <- level_rank[[requested_level]]

log_at <- function(level, ...) {
  if (level_rank[[level]] >= min_rank) log_json(level, ...)
}

# --- Evaluation Metrics & Helper Utilities ------------------------------------

get_mode <- function(x) {
  ux <- unique(x[!is.na(x)])
  if (length(ux) == 0) NA else ux[which.max(tabulate(match(x, ux)))]
}

auc_roc <- function(probs, labels) {
  n1 <- sum(labels == 1)
  n0 <- sum(labels == 0)
  if (n1 == 0 || n0 == 0) return(NA_real_)
  r <- rank(probs)
  (sum(r[labels == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

log_loss <- function(probs, labels) {
  eps <- 1e-15
  -mean(ifelse(labels == 1, log(pmax(probs, eps)), log(pmax(1 - probs, eps))))
}

# --- 2. Load Feature Store ----------------------------------------------------

if (!file.exists(opt$data_path)) {
  log_at("FATAL", paste("Feature store RDS file not found at:", opt$data_path))
  quit(save = "no", status = 1)
}

log_at("INFO", paste("Loading feature store from:", opt$data_path))
data <- readRDS(opt$data_path)

if (!is.data.frame(data)) {
  data <- as.data.frame(data)
}

# Filter training partition if 'is_train' flag exists
if ("is_train" %in% colnames(data)) {
  data <- data[data$is_train == 1, ]
  log_at("INFO", "Filtered dataset for training partition (is_train == 1)")
}

if (!"TARGET" %in% colnames(data)) {
  log_at("FATAL", "Data is missing required target column 'TARGET'")
  quit(save = "no", status = 1)
}

# Exclude identifiers and split indicators from predictor features
exclude_cols <- c("SK_ID_CURR", "TARGET", "is_train")
feature_names <- setdiff(colnames(data), exclude_cols)

if (length(feature_names) == 0) {
  log_at("FATAL", "No predictor features found in dataset")
  quit(save = "no", status = 1)
}

log_at("INFO", "Dataset loaded successfully",
  total_rows = nrow(data),
  total_features = length(feature_names)
)

# --- 3. Imputation and Preprocessing -----------------------------------------

imputation_map <- list()

for (col in feature_names) {
  if (any(is.na(data[[col]]))) {
    if (is.numeric(data[[col]])) {
      imp_val <- median(data[[col]], na.rm = TRUE)
      data[[col]][is.na(data[[col]])] <- imp_val
      imputation_map[[col]] <- imp_val
    } else {
      imp_val <- get_mode(data[[col]])
      data[[col]][is.na(data[[col]])] <- imp_val
      imputation_map[[col]] <- imp_val
    }
  }
}

# --- 4. 5-Fold Cross-Validation & Training ------------------------------------

train_df <- data[, c(feature_names, "TARGET")]
train_df$TARGET <- factor(train_df$TARGET, levels = c("0", "1"))
train_df <- train_df[!is.na(train_df$TARGET), ]

if (nrow(train_df) < 10) {
  log_at("FATAL", paste("Insufficient valid target rows for training:", nrow(train_df)))
  quit(save = "no", status = 1)
}

n_folds <- opt$n_folds
fold_ids <- sample(rep(seq_len(n_folds), length.out = nrow(train_df)))

# Detect available cores; mclapply uses forking (Unix only — falls back to 1 on Windows)
n_cores <- if (.Platform$OS.type == "windows") 1L else min(n_folds, detectCores(logical = FALSE))

log_at("INFO", "Starting parallel cross-validation",
  algorithm = "ranger (Random Forest)",
  num_trees = opt$num_trees,
  num_features = length(feature_names),
  n_folds = n_folds,
  n_cores = n_cores,
  total_rows = nrow(train_df)
)

oof_labels <- as.integer(as.character(train_df$TARGET))

# Run all folds in parallel — each worker returns preds + metrics for its fold
fold_results <- mclapply(seq_len(n_folds), function(k) {
  fold_train <- train_df[fold_ids != k, ]
  fold_val   <- train_df[fold_ids == k, ]

  fold_model <- ranger::ranger(
    formula = TARGET ~ .,
    data = fold_train,
    probability = TRUE,
    num.trees = opt$num_trees,
    mtry = max(1, floor(sqrt(length(feature_names)))),
    min.node.size = 5,
    verbose = FALSE
  )

  fold_preds  <- predict(fold_model, data = fold_val)$predictions[, "1"]
  fold_labels <- as.integer(as.character(fold_val$TARGET))

  list(
    fold       = k,
    mask       = fold_ids == k,
    preds      = fold_preds,
    auc_roc    = round(auc_roc(fold_preds, fold_labels), 6),
    accuracy   = round(mean((fold_preds >= 0.5) == (fold_labels == 1)), 6),
    log_loss   = round(log_loss(fold_preds, fold_labels), 6),
    val_rows   = nrow(fold_val)
  )
}, mc.cores = n_cores, mc.set.seed = TRUE)

# Reconstruct out-of-fold probability vector and per-fold metrics
oof_probs   <- rep(NA_real_, nrow(train_df))
fold_metrics <- vector("list", n_folds)

for (res in fold_results) {
  oof_probs[res$mask] <- res$preds
  fold_met <- list(
    fold     = res$fold,
    auc_roc  = res$auc_roc,
    accuracy = res$accuracy,
    log_loss = res$log_loss,
    val_rows = res$val_rows
  )
  fold_metrics[[res$fold]] <- fold_met
  log_at("INFO", paste("Fold", res$fold, "complete"), metrics = fold_met)
}

# --- 5. Aggregate CV Metrics --------------------------------------------------

cv_metrics <- list(
  algorithm = "ranger",
  num_trees = opt$num_trees,
  n_folds   = n_folds,
  auc_roc   = round(auc_roc(oof_probs, oof_labels), 6),
  accuracy  = round(mean((oof_probs >= 0.5) == (oof_labels == 1)), 6),
  log_loss  = round(log_loss(oof_probs, oof_labels), 6),
  total_rows = length(oof_labels)
)

log_at("INFO", "Cross-validation evaluation complete", cv_metrics = cv_metrics)

# --- 5b. Train Final Model on All Training Data ------------------------------

log_at("INFO", "Re-training final model on full training data",
  train_rows = nrow(train_df)
)

model <- ranger::ranger(
  formula = TARGET ~ .,
  data = train_df,
  probability = TRUE,
  num.trees = opt$num_trees,
  mtry = max(1, floor(sqrt(length(feature_names)))),
  min.node.size = 5,
  verbose = FALSE
)

metrics <- cv_metrics
metrics$mtry <- model$mtry

log_at("INFO", "Final model training complete")

# --- 6. Write Model Artifact (models/model_v1.rds) ---------------------------

dir.create(dirname(opt$model_output_path), showWarnings = FALSE, recursive = TRUE)

model_artifact <- list(
  model = model,
  features = feature_names,
  imputation_map = imputation_map,
  target_levels = c("0", "1"),
  default_label = "1",
  trained_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  metrics = metrics
)
class(model_artifact) <- "credit_model"

# Save RDS (overwrites models/model_v1.rds if present)
saveRDS(model_artifact, opt$model_output_path)

log_at("INFO", "Model artifact saved successfully",
  path = opt$model_output_path,
  file_size_bytes = file.info(opt$model_output_path)$size
)

# --- 7. Write Features Metadata (data/features_metadata.json) ----------------

dir.create(dirname(opt$metadata_output_path), showWarnings = FALSE, recursive = TRUE)

features_list <- lapply(feature_names, function(fname) {
  col_data <- data[[fname]]
  list(
    name = fname,
    type = if (is.numeric(col_data)) "numeric" else "categorical",
    required = TRUE,
    imputed_default = if (!is.null(imputation_map[[fname]])) imputation_map[[fname]] else NULL
  )
})

metadata_payload <- list(
  model_version = "v1",
  schema_version = "1.0",
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  total_features = length(feature_names),
  features = features_list
)

write_json(metadata_payload, opt$metadata_output_path, auto_unbox = TRUE, pretty = TRUE)

log_at("INFO", "Features metadata saved successfully",
  path = opt$metadata_output_path
)

# --- 8. Test Set Evaluation ---------------------------------------------------

test_metrics <- NULL

if (file.exists(opt$test_set_path)) {
  log_at("INFO", paste("Loading test set from:", opt$test_set_path))
  test_data <- readRDS(opt$test_set_path)
  if (!is.data.frame(test_data)) test_data <- as.data.frame(test_data)

  # Only evaluate if TARGET column exists and has non-NA values
  if ("TARGET" %in% colnames(test_data) && any(!is.na(test_data$TARGET))) {
    # Apply same imputation from training
    for (col in feature_names) {
      if (col %in% colnames(test_data) && any(is.na(test_data[[col]]))) {
        if (!is.null(imputation_map[[col]])) {
          test_data[[col]][is.na(test_data[[col]])] <- imputation_map[[col]]
        }
      }
    }

    test_eval_df <- test_data[, intersect(c(feature_names, "TARGET"), colnames(test_data))]
    test_eval_df$TARGET <- factor(test_eval_df$TARGET, levels = c("0", "1"))
    test_eval_df <- test_eval_df[!is.na(test_eval_df$TARGET), ]

    if (nrow(test_eval_df) > 0) {
      probs_final_test <- predict(model, data = test_eval_df)$predictions[, "1"]
      labels_final_test <- as.integer(as.character(test_eval_df$TARGET))

      test_metrics <- list(
        auc_roc  = round(auc_roc(probs_final_test, labels_final_test), 6),
        accuracy = round(mean((probs_final_test >= 0.5) == (labels_final_test == 1)), 6),
        log_loss = round(log_loss(probs_final_test, labels_final_test), 6),
        test_rows = length(labels_final_test)
      )
      log_at("INFO", "Test set evaluation complete", test_metrics = test_metrics)
    } else {
      log_at("WARN", "Test set has no valid TARGET rows after filtering. Skipping test evaluation.")
    }
  } else {
    log_at("WARN", "Test set has no TARGET column or all TARGET values are NA. Skipping test evaluation.")
  }
} else {
  log_at("WARN", paste("Test set not found at:", opt$test_set_path, "- skipping test evaluation."))
}

# --- 9. Save Performance Result (models/performance_result.json) --------------

dir.create(dirname(opt$performance_output_path), showWarnings = FALSE, recursive = TRUE)

performance_result <- list(
  model_version = "v1",
  trained_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  cv_metrics = list(
    auc_roc  = cv_metrics$auc_roc,
    accuracy = cv_metrics$accuracy,
    log_loss = cv_metrics$log_loss,
    n_folds  = cv_metrics$n_folds
  ),
  test_metrics = test_metrics,
  training_config = list(
    num_trees = opt$num_trees,
    mtry      = model$mtry,
    seed      = opt$seed,
    n_folds   = opt$n_folds
  )
)

write_json(performance_result, opt$performance_output_path, auto_unbox = TRUE, pretty = TRUE, null = "null")

log_at("INFO", "Performance result saved successfully",
  path = opt$performance_output_path
)