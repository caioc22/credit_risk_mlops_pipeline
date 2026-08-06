#!/usr/bin/env Rscript
# ------------------------------------------------------------------------------
# src/data_modelling.R
#
# Processes the Home Credit Default Risk relational dataset:
#   1. Unzips dataset archives if compressed files exist in data/
#   2. Aggregates 1:N relational tables up to the SK_ID_CURR grain
#   3. Merges aggregated features with application_train and application_test
#   4. Generates a compressed RDS Feature Store (data/feature_store.rds)
# ------------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
  library(optparse)
})

# --- 1. CLI Arguments & Configuration -----------------------------------------

option_list <- list(
  make_option(c("-d", "--data-dir"),
    type = "character",
    default = Sys.getenv("DATA_DIR", "data"),
    dest = "data_dir",
    help = "Directory containing zip files or raw CSVs [default: %default]"
  ),
  make_option(c("-o", "--output-path"),
    type = "character",
    default = Sys.getenv("FEATURE_STORE_PATH", "data/feature_store.rds"),
    dest = "output_path",
    help = "Path for output RDS feature store [default: %default]"
  )
)

opt <- parse_args(OptionParser(option_list = option_list))

# optparse normalizes option names to hyphens; map back to underscore accessors.
if (is.null(opt$output_path)) {
  opt$output_path <- opt$`output-path`
}

log_json <- function(level, message, ...) {
  entry <- list(
    timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
    level = level,
    logger = "data_modelling",
    message = message,
    ...
  )
  cat(toJSON(entry, auto_unbox = TRUE), "\n")
}

# --- 2. Unzip Dataset Handler -------------------------------------------------

unzip_dataset <- function(data_dir) {
  zip_files <- list.files(data_dir, pattern = "\\.zip$", full.names = TRUE)
  if (length(zip_files) > 0) {
    log_json("INFO", paste("Found", length(zip_files), "zip file(s). Extracting..."))
    for (zf in zip_files) {
      log_json("INFO", paste("Unzipping:", basename(zf)))
      unzip(zf, exdir = data_dir)
    }
  } else {
    log_json("INFO", "No .zip files found. Reading existing CSV files.")
  }
}


# Utility to safely load CSV if present
safe_fread <- function(file_path) {
  if (file.exists(file_path)) {
    log_json("INFO", paste("Reading:", basename(file_path)))
    return(fread(file_path))
  } else {
    log_json("WARN", paste("File missing:", basename(file_path), "- skipping aggregation for this table."))
    return(NULL)
  }
}


# --- Utility to clean up unzipped CSV files -----------------------------------

cleanup_csv_files <- function(data_dir) {
  csv_files <- list.files(data_dir, pattern = "\\.csv$", full.names = TRUE)
  
  if (length(csv_files) > 0) {
    log_json("INFO", paste("Cleaning up", length(csv_files), "CSV file(s) from:", data_dir))
    removed_status <- file.remove(csv_files)
    
    if (all(removed_status)) {
      log_json("INFO", "All CSV files removed successfully. Original .zip archive(s) preserved.")
    } else {
      log_json("WARN", "Some CSV files could not be deleted.")
    }
  } else {
    log_json("INFO", "No CSV files found to remove.")
  }
}

unzip_dataset(opt$data_dir)

# --- 3. Feature Engineering & Relational Aggregations ------------------------

# 3.1 Bureau & Bureau Balance
process_bureau <- function(data_dir) {
  bureau <- safe_fread(file.path(data_dir, "bureau.csv"))
  bureau_bal <- safe_fread(file.path(data_dir, "bureau_balance.csv"))
  
  if (is.null(bureau)) return(NULL)
  
  if (!is.null(bureau_bal)) {
    # Aggregate monthly status to bureau level
    bal_agg <- bureau_bal[, .(
      bb_months_count = .N,
      bb_status_active_ratio = mean(STATUS %in% c("0", "C"), na.rm = TRUE),
      bb_dpd_max = max(suppressWarnings(as.numeric(STATUS)), na.rm = TRUE)
    ), by = SK_ID_BUREAU]
    
    bureau <- merge(bureau, bal_agg, by = "SK_ID_BUREAU", all.x = TRUE)
  }
  
  # Aggregate up to SK_ID_CURR
  bureau_agg <- bureau[, .(
    bureau_count = .N,
    bureau_credit_active_mean = mean(CREDIT_ACTIVE == "Closed", na.rm = TRUE),
    bureau_sum_credit = sum(AMT_CREDIT_SUM, na.rm = TRUE),
    bureau_sum_debt = sum(AMT_CREDIT_SUM_DEBT, na.rm = TRUE),
    bureau_max_overdue = max(AMT_CREDIT_MAX_OVERDUE, na.rm = TRUE),
    bureau_avg_days_credit = mean(DAYS_CREDIT, na.rm = TRUE)
  ), by = SK_ID_CURR]
  
  return(bureau_agg)
}

# 3.2 Previous Applications
process_previous_app <- function(data_dir) {
  prev <- safe_fread(file.path(data_dir, "previous_application.csv"))
  if (is.null(prev)) return(NULL)
  
  prev_agg <- prev[, .(
    prev_app_count = .N,
    prev_approval_rate = mean(NAME_CONTRACT_STATUS == "Approved", na.rm = TRUE),
    prev_refusal_rate = mean(NAME_CONTRACT_STATUS == "Refused", na.rm = TRUE),
    prev_mean_credit = mean(AMT_CREDIT, na.rm = TRUE),
    prev_mean_annuity = mean(AMT_ANNUITY, na.rm = TRUE),
    prev_downpayment_ratio = mean(AMT_DOWN_PAYMENT / (AMT_GOODS_PRICE + 1e-5), na.rm = TRUE)
  ), by = SK_ID_CURR]
  
  return(prev_agg)
}

# 3.3 POS CASH Balance
process_pos_cash <- function(data_dir) {
  pos <- safe_fread(file.path(data_dir, "POS_CASH_balance.csv"))
  if (is.null(pos)) return(NULL)
  
  pos_agg <- pos[, .(
    pos_months_count = .N,
    pos_max_dpd = max(SK_DPD, na.rm = TRUE),
    pos_completed_ratio = mean(NAME_CONTRACT_STATUS == "Completed", na.rm = TRUE)
  ), by = SK_ID_CURR]
  
  return(pos_agg)
}

# 3.4 Installment Payments (Behavioral Payment Performance)
process_installments <- function(data_dir) {
  inst <- safe_fread(file.path(data_dir, "installments_payments.csv"))
  if (is.null(inst)) return(NULL)
  
  inst[, payment_delay := DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT]
  inst[, payment_diff := AMT_INSTALMENT - AMT_PAYMENT]
  
  inst_agg <- inst[, .(
    inst_count = .N,
    inst_max_delay = max(payment_delay, na.rm = TRUE),
    inst_mean_delay = mean(payment_delay, na.rm = TRUE),
    inst_mean_underpayment = mean(payment_diff, na.rm = TRUE)
  ), by = SK_ID_CURR]
  
  return(inst_agg)
}

# 3.5 Credit Card Balance
process_credit_card <- function(data_dir) {
  cc <- safe_fread(file.path(data_dir, "credit_card_balance.csv"))
  if (is.null(cc)) return(NULL)
  
  cc_agg <- cc[, .(
    cc_months_count = .N,
    cc_mean_balance = mean(AMT_BALANCE, na.rm = TRUE),
    cc_mean_drawing = mean(AMT_DRAWINGS_CURRENT, na.rm = TRUE),
    cc_max_dpd = max(SK_DPD, na.rm = TRUE)
  ), by = SK_ID_CURR]
  
  return(cc_agg)
}

# --- 4. Consolidate and Build Feature Store ----------------------------------

app_train <- safe_fread(file.path(opt$data_dir, "application_train.csv"))
app_test  <- safe_fread(file.path(opt$data_dir, "application_test.csv"))

if (is.null(app_train)) {
  log_json("FATAL", "application_train.csv not found. Aborting feature store creation.")
  quit(save = "no", status = 1)
}

# Add split identifier
app_train[, is_train := 1]
if (!is.null(app_test)) {
  app_test[, TARGET := NA_integer_]
  app_test[, is_train := 0]
  full_app <- rbind(app_train, app_test, use.names = TRUE, fill = TRUE)
} else {
  full_app <- app_train
}

# Select core features from main application table
core_cols <- c(
  "SK_ID_CURR", "TARGET", "is_train", "AMT_INCOME_TOTAL", "AMT_CREDIT",
  "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_2", "EXT_SOURCE_3"
)
existing_core_cols <- intersect(core_cols, colnames(full_app))
feature_dt <- full_app[, ..existing_core_cols]

# Run feature extractions and merge
bureau_agg <- process_bureau(opt$data_dir)
if (!is.null(bureau_agg)) feature_dt <- merge(feature_dt, bureau_agg, by = "SK_ID_CURR", all.x = TRUE)

prev_agg <- process_previous_app(opt$data_dir)
if (!is.null(prev_agg)) feature_dt <- merge(feature_dt, prev_agg, by = "SK_ID_CURR", all.x = TRUE)

pos_agg <- process_pos_cash(opt$data_dir)
if (!is.null(pos_agg)) feature_dt <- merge(feature_dt, pos_agg, by = "SK_ID_CURR", all.x = TRUE)

inst_agg <- process_installments(opt$data_dir)
if (!is.null(inst_agg)) feature_dt <- merge(feature_dt, inst_agg, by = "SK_ID_CURR", all.x = TRUE)

cc_agg <- process_credit_card(opt$data_dir)
if (!is.null(cc_agg)) feature_dt <- merge(feature_dt, cc_agg, by = "SK_ID_CURR", all.x = TRUE)

# --- 5a. Export to RDS Feature Store -----------------------------------------

dir.create(dirname(opt$output_path), showWarnings = FALSE, recursive = TRUE)
saveRDS(feature_dt, file = opt$output_path, compress = "gzip")

# --- 5b. Export Test Set for Final Performance Tracking ----------------------

test_partition <- feature_dt[is_train == 0]
if (nrow(test_partition) > 0) {
  test_set_path <- file.path(dirname(opt$output_path), "test_set.rds")
  saveRDS(test_partition, file = test_set_path, compress = "gzip")
  log_json("INFO", "Test set saved for final performance tracking",
    output_path = test_set_path,
    total_rows = nrow(test_partition),
    total_columns = ncol(test_partition)
  )
} else {
  log_json("WARN", "No test partition found (is_train == 0). Skipping test_set.rds generation.")
}

cleanup_csv_files(opt$data_dir)

log_json("INFO", "Feature Store created successfully",
  output_path = opt$output_path,
  total_rows = nrow(feature_dt),
  total_columns = ncol(feature_dt),
  file_size_mb = round(file.info(opt$output_path)$size / (1024^2), 2)
)