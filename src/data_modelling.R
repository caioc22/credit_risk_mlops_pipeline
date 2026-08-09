#!/usr/bin/env Rscript
# ------------------------------------------------------------------------------
# src/data_modelling.R
#
# Builds the feature store for the credit risk pipeline from the Home Credit
# Default Risk raw zip archive.
#
# Sequential flow:
#   1. Extract raw CSV files from data/home-credit-default-risk.zip if needed
#   2. Load core application tables and aggregate the relational tables
#   3. Merge all features onto application_train and application_test
#   4. Persist data/feature_store.rds and data/test_set.rds
#   5. Remove the extracted CSVs and the raw .zip archive (unless
#      --keep-raw-zip is set) so the container/data dir stays lightweight
# ------------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(data.table)
  library(logger)
  library(optparse)
})

# --- CLI configuration --------------------------------------------------------

option_list <- list(
  make_option(c("-d", "--data-dir"),
    type = "character",
    default = Sys.getenv("DATA_DIR", "data"),
    dest = "data_dir",
    help = "Directory containing the raw zip archive and extracted CSV files [default: %default]"
  ),
  make_option(c("-z", "--zip-path"),
    type = "character",
    default = Sys.getenv("RAW_DATA_ZIP", "data/home-credit-default-risk.zip"),
    dest = "zip_path",
    help = "Path to the Home Credit raw zip archive [default: %default]"
  ),
  make_option(c("-o", "--feature-store-path"),
    type = "character",
    default = Sys.getenv("FEATURE_STORE_PATH", "data/feature_store.rds"),
    dest = "feature_store_path",
    help = "Path where the training feature store is saved [default: %default]"
  ),
  make_option(c("-t", "--test-set-path"),
    type = "character",
    default = Sys.getenv("TEST_SET_PATH", "data/test_set.rds"),
    dest = "test_set_path",
    help = "Path where the testing feature set is saved [default: %default]"
  ),
  make_option("--keep-raw-zip",
    action = "store_true",
    default = tolower(Sys.getenv("KEEP_RAW_ZIP", "false")) == "true",
    dest = "keep_raw_zip",
    help = "Keep the raw .zip archive after building the feature store (default: it is removed to keep the pipeline lightweight) [default: %default]"
  )
)

opt <- parse_args(OptionParser(option_list = option_list))

# --- Logging helpers ----------------------------------------------------------

safe_info <- function(message) logger::log_info(message)
safe_warn <- function(message) logger::log_warn(message)
safe_error <- function(message) logger::log_error(message)

safe_sum <- function(x) {
  if (length(x) == 0 || all(is.na(x))) return(NA_real_)
  sum(x, na.rm = TRUE)
}

safe_max <- function(x) {
  if (length(x) == 0 || all(is.na(x))) return(NA_real_)
  max(x, na.rm = TRUE)
}

safe_mean <- function(x) {
  if (length(x) == 0 || all(is.na(x))) return(NA_real_)
  mean(x, na.rm = TRUE)
}

# --- Constants ----------------------------------------------------------------

required_raw_csvs <- c(
  "application_train.csv",
  "application_test.csv",
  "bureau.csv",
  "bureau_balance.csv",
  "previous_application.csv",
  "installments_payments.csv",
  "POS_CASH_balance.csv",
  "credit_card_balance.csv"
)

files_to_keep_after_cleanup <- c(
  "HomeCredit_columns_description.csv",
  "sample_credit_data.csv"
)

application_feature_cols <- c(
  "AMT_CREDIT",
  "AMT_ANNUITY",
  "AMT_INCOME_TOTAL",
  "DAYS_BIRTH",
  "DAYS_EMPLOYED",
  "EXT_SOURCE_1",
  "EXT_SOURCE_2",
  "EXT_SOURCE_3"
)

third_party_score_map <- c(
  THIRD_PARTY_CREDIT_SCORE_2 = "EXT_SOURCE_2",
  THIRD_PARTY_CREDIT_SCORE_3 = "EXT_SOURCE_3"
)

# --- IO helpers ---------------------------------------------------------------

ensure_data_directory <- function(data_dir) {
  dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
  invisible(data_dir)
}

extract_raw_archive <- function(data_dir, zip_path) {
  raw_paths <- file.path(data_dir, required_raw_csvs)
  if (all(file.exists(raw_paths))) {
    safe_info(sprintf("Raw CSV files already exist in %s. Skipping extraction.", data_dir))
    return(invisible(FALSE))
  }

  if (!file.exists(zip_path)) {
    stop(sprintf("Raw zip archive not found: %s", zip_path))
  }

  safe_info(sprintf("Missing raw CSV files. Extracting %s into %s.", basename(zip_path), data_dir))
  tryCatch(
    {
      utils::unzip(zip_path, exdir = data_dir, overwrite = TRUE)
      safe_info("Extraction completed successfully.")
      invisible(TRUE)
    },
    error = function(e) {
      safe_error(sprintf("Failed to extract archive: %s", conditionMessage(e)))
      stop(e)
    }
  )
}

read_csv_selected <- function(file_path, required_cols, optional_cols = character(), strict = TRUE) {
  if (!file.exists(file_path)) {
    if (strict) stop(sprintf("Required CSV file not found: %s", file_path))
    safe_warn(sprintf("CSV file not found, skipping: %s", basename(file_path)))
    return(NULL)
  }

  header <- tryCatch(
    names(data.table::fread(file_path, nrows = 0, showProgress = FALSE)),
    error = function(e) {
      stop(sprintf("Failed to read header from %s: %s", file_path, conditionMessage(e)))
    }
  )

  missing_required <- setdiff(required_cols, header)
  if (length(missing_required) > 0) {
    msg <- sprintf(
      "%s is missing required column(s): %s",
      basename(file_path),
      paste(missing_required, collapse = ", ")
    )
    if (strict) stop(msg)
    safe_warn(msg)
    return(NULL)
  }

  cols_to_read <- intersect(unique(c(required_cols, optional_cols)), header)
  safe_info(sprintf(
    "Loading %s with %d selected columns.",
    basename(file_path),
    length(cols_to_read)
  ))

  tryCatch(
    data.table::fread(file_path, select = cols_to_read, showProgress = FALSE),
    error = function(e) {
      stop(sprintf("Failed to read %s: %s", file_path, conditionMessage(e)))
    }
  )
}

rename_third_party_scores <- function(dt) {
  for (new_name in names(third_party_score_map)) {
    old_name <- third_party_score_map[[new_name]]
    if (old_name %in% names(dt) && !(new_name %in% names(dt))) {
      data.table::setnames(dt, old_name, new_name)
    }
  }
  dt
}

merge_feature_tables <- function(base_dt, tables) {
  result <- data.table::copy(base_dt)
  for (tbl in tables) {
    if (!is.null(tbl)) {
      result <- merge(result, tbl, by = "SK_ID_CURR", all.x = TRUE, sort = FALSE)
    }
  }
  result
}

cleanup_extracted_csvs <- function(data_dir) {
  csv_files <- list.files(data_dir, pattern = "\\.csv$", full.names = TRUE)
  removable <- csv_files[!basename(csv_files) %in% files_to_keep_after_cleanup]

  if (length(removable) == 0) {
    safe_info("No extracted CSV files found to remove.")
    return(invisible(TRUE))
  }

  safe_info(sprintf("Removing %d extracted CSV file(s) from %s.", length(removable), data_dir))
  removed <- file.remove(removable)

  if (all(removed)) {
    safe_info("CSV cleanup completed successfully.")
  } else {
    failed <- basename(removable)[!removed]
    safe_warn(sprintf("Some CSV files could not be deleted: %s", paste(failed, collapse = ", ")))
  }

  invisible(all(removed))
}

# --- Aggregations -------------------------------------------------------------

build_bureau_aggregates <- function(data_dir) {
  bureau <- read_csv_selected(
    file.path(data_dir, "bureau.csv"),
    required_cols = c("SK_ID_CURR", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_OVERDUE"),
    strict = FALSE
  )
  if (is.null(bureau)) return(NULL)

  bureau[, .(
    bureau_sum_debt = safe_sum(AMT_CREDIT_SUM),
    bureau_max_overdue = safe_max(AMT_CREDIT_SUM_OVERDUE)
  ), by = SK_ID_CURR]
}

build_previous_application_aggregates <- function(data_dir) {
  prev <- read_csv_selected(
    file.path(data_dir, "previous_application.csv"),
    required_cols = c("SK_ID_CURR", "NAME_CONTRACT_STATUS"),
    strict = FALSE
  )
  if (is.null(prev)) return(NULL)

  prev[, .(
    prev_approval_rate = safe_mean(NAME_CONTRACT_STATUS == "Approved"),
    prev_refusal_rate = safe_mean(NAME_CONTRACT_STATUS == "Refused")
  ), by = SK_ID_CURR]
}

build_installment_aggregates <- function(data_dir) {
  inst <- read_csv_selected(
    file.path(data_dir, "installments_payments.csv"),
    required_cols = c("SK_ID_CURR", "DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT", "AMT_INSTALMENT", "AMT_PAYMENT"),
    strict = FALSE
  )
  if (is.null(inst)) return(NULL)

  inst[, payment_delay := DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT]
  inst[, payment_underpayment := AMT_INSTALMENT - AMT_PAYMENT]

  inst[, .(
    inst_max_delay = safe_max(payment_delay),
    inst_mean_underpayment = safe_mean(payment_underpayment)
  ), by = SK_ID_CURR]
}

build_pos_cash_aggregates <- function(data_dir) {
  pos <- read_csv_selected(
    file.path(data_dir, "POS_CASH_balance.csv"),
    required_cols = c("SK_ID_CURR", "SK_DPD"),
    strict = FALSE
  )
  if (is.null(pos)) return(NULL)

  pos[, .(
    pos_max_dpd = safe_max(SK_DPD)
  ), by = SK_ID_CURR]
}

build_credit_card_aggregates <- function(data_dir) {
  cc <- read_csv_selected(
    file.path(data_dir, "credit_card_balance.csv"),
    required_cols = c("SK_ID_CURR", "SK_DPD"),
    strict = FALSE
  )
  if (is.null(cc)) return(NULL)

  cc[, .(
    cc_max_dpd = safe_max(SK_DPD)
  ), by = SK_ID_CURR]
}

# --- Application loading ------------------------------------------------------

build_application_base <- function(file_path, is_train) {
  required_cols <- if (is_train) {
    c("SK_ID_CURR", "TARGET", application_feature_cols)
  } else {
    c("SK_ID_CURR", application_feature_cols)
  }

  app <- read_csv_selected(file_path, required_cols = required_cols, strict = TRUE)
  app <- rename_third_party_scores(app)
  app[, is_train := as.integer(is_train)]
  app
}

build_feature_store <- function(base_dt, aggregated_tables) {
  merged <- merge_feature_tables(base_dt, aggregated_tables)
  data.table::setcolorder(merged, c("SK_ID_CURR", setdiff(names(merged), "SK_ID_CURR")))
  merged
}

# --- Main pipeline ------------------------------------------------------------

main <- function() {
  tryCatch(
    {
      ensure_data_directory(opt$data_dir)

      safe_info(sprintf("Starting feature store build in %s.", opt$data_dir))

      # If this is a re-run after a successful build (the raw archive is
      # removed on purpose to keep the pipeline lightweight), reuse the
      # existing feature store instead of failing.
      raw_csv_paths <- file.path(opt$data_dir, required_raw_csvs)
      outputs_exist <- file.exists(opt$feature_store_path) && file.exists(opt$test_set_path)
      if (!file.exists(opt$zip_path) && !all(file.exists(raw_csv_paths))) {
        if (outputs_exist) {
          safe_warn(sprintf(
            "Raw archive and extracted CSVs not found in %s; reusing existing artifacts %s and %s.",
            opt$data_dir, opt$feature_store_path, opt$test_set_path
          ))
          return(invisible(list(train = NULL, test = NULL)))
        }
        stop(sprintf(
          "No raw data available: missing zip archive '%s' and missing CSVs in '%s'.",
          opt$zip_path, opt$data_dir
        ))
      }

      extract_raw_archive(opt$data_dir, opt$zip_path)

      app_train <- build_application_base(file.path(opt$data_dir, "application_train.csv"), is_train = TRUE)
      app_test <- build_application_base(file.path(opt$data_dir, "application_test.csv"), is_train = FALSE)

      safe_info(sprintf(
        "Application tables loaded successfully: train rows = %d, test rows = %d.",
        nrow(app_train),
        nrow(app_test)
      ))

      bureau_agg <- build_bureau_aggregates(opt$data_dir)
      prev_agg <- build_previous_application_aggregates(opt$data_dir)
      inst_agg <- build_installment_aggregates(opt$data_dir)
      pos_agg <- build_pos_cash_aggregates(opt$data_dir)
      cc_agg <- build_credit_card_aggregates(opt$data_dir)

      safe_info("Merging aggregated relational features onto application_train.")
      train_store <- build_feature_store(
        app_train,
        list(bureau_agg, prev_agg, inst_agg, pos_agg, cc_agg)
      )

      safe_info("Merging aggregated relational features onto application_test.")
      test_store <- build_feature_store(
        app_test,
        list(bureau_agg, prev_agg, inst_agg, pos_agg, cc_agg)
      )

      dir.create(dirname(opt$feature_store_path), recursive = TRUE, showWarnings = FALSE)
      dir.create(dirname(opt$test_set_path), recursive = TRUE, showWarnings = FALSE)

      safe_info(sprintf("Saving training feature store to %s.", opt$feature_store_path))
      saveRDS(train_store, file = opt$feature_store_path, compress = "gzip")

      safe_info(sprintf("Saving testing feature set to %s.", opt$test_set_path))
      saveRDS(test_store, file = opt$test_set_path, compress = "gzip")

      cleanup_extracted_csvs(opt$data_dir)

      if (!opt$keep_raw_zip && file.exists(opt$zip_path)) {
        if (file.remove(opt$zip_path)) {
          safe_info(sprintf("Raw archive removed to keep the pipeline lightweight: %s.", opt$zip_path))
        } else {
          safe_warn(sprintf("Could not remove raw archive: %s.", opt$zip_path))
        }
      }

      safe_info(sprintf(
        "Feature store build completed successfully. train rows = %d, test rows = %d, train cols = %d, test cols = %d.",
        nrow(train_store), nrow(test_store), ncol(train_store), ncol(test_store)
      ))

      invisible(list(train = train_store, test = test_store))
    },
    error = function(e) {
      safe_error(sprintf("Feature store build failed: %s", conditionMessage(e)))
      quit(save = "no", status = 1)
    }
  )
}

main()
