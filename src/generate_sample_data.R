#!/usr/bin/env Rscript
# ------------------------------------------------------------------------------
# src/generate_sample_data.R
#
# Generates synthetic Home Credit-style data for CI and local smoke tests.
# Training does NOT read these outputs directly: src/data_modelling.R unpacks
# the mock .zip CSV tables into data/feature_store.rds, and src/train.R
# consumes that RDS artifact.
#
# Configuration (CLI args override environment variables, which override
# defaults):
#   SAMPLE_DATA_PATH  (env, default: data/sample_credit_data.csv)
#   RAW_DATA_ZIP      (env, default: data/home-credit-default-risk.zip)
#   N_ROWS            (env, default: 3000)
#   --seed            (default: 42)
#
# Outputs:
#   - data/sample_credit_data.csv              (standalone demo CSV)
#   - data/home-credit-default-risk.zip        (mock raw tables for data_modelling.R)
# ------------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(jsonlite)
  library(optparse)
})

log_json <- function(level, message, ...) {
  cat(toJSON(list(
    timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
    level = level,
    event = "generate_sample_data",
    message = message,
    ...
  ), auto_unbox = TRUE), "\n")
}

option_list <- list(
  make_option(c("-o", "--output-path"),
    type = "character",
    default = Sys.getenv("SAMPLE_DATA_PATH", "data/sample_credit_data.csv"),
    help = "Output demo CSV path [default: %default]"
  ),
  make_option(c("-z", "--zip-path"),
    type = "character",
    default = Sys.getenv("RAW_DATA_ZIP", "data/home-credit-default-risk.zip"),
    help = "Output mock Home Credit zip archive [default: %default]"
  ),
  make_option(c("-n", "--n-rows"),
    type = "integer",
    default = as.integer(Sys.getenv("N_ROWS", "3000")),
    help = "Number of training rows to generate [default: %default]"
  ),
  make_option("--seed",
    type = "integer",
    default = 42,
    help = "Random seed for reproducibility [default: %default]"
  ),
  make_option("--force",
    action = "store_true",
    default = FALSE,
    help = "Overwrite output files even if they already exist"
  )
)

opt <- parse_args(OptionParser(option_list = option_list))

opt$output_path <- opt$`output-path`
opt$zip_path <- opt$`zip-path`
opt$n_rows <- opt$`n-rows`

set.seed(opt$seed)

csv_exists <- file.exists(opt$output_path)
zip_exists <- file.exists(opt$zip_path)

if (csv_exists && zip_exists && !opt$force) {
  log_json("INFO", "Output files already exist; skipping generation",
    csv_path = opt$output_path,
    zip_path = opt$zip_path
  )
  quit(save = "no", status = 0)
}

dir.create(dirname(opt$output_path), showWarnings = FALSE, recursive = TRUE)
dir.create(dirname(opt$zip_path), showWarnings = FALSE, recursive = TRUE)

n_train <- opt$n_rows
n_test <- max(100L, as.integer(n_train / 5))

# --- Synthetic feature generation (seeded, so fully reproducible) ------------
ext_source_1 <- pmax(0, pmin(1, rnorm(n_train, mean = 0.5, sd = 0.2)))
ext_source_2 <- pmax(0, pmin(1, rnorm(n_train, mean = 0.55, sd = 0.2)))
ext_source_3 <- pmax(0, pmin(1, rnorm(n_train, mean = 0.45, sd = 0.2)))

days_birth <- as.integer(-runif(n_train, min = 5000, max = 30000))
days_employed <- as.integer(-runif(n_train, min = 0, max = 15000))

amt_income_total <- round(rlnorm(n_train, meanlog = 10.7, sdlog = 0.55), 2)
credit_ratio <- runif(n_train, min = 1.5, max = 4.0)
amt_credit <- round(amt_income_total * credit_ratio, 2)
amt_annuity <- round(amt_credit / runif(n_train, min = 6, max = 20), 2)

z <- -0.4 +
  -2.5 * (ext_source_2 - 0.5) -
  3.0 * (ext_source_3 - 0.45) +
  0.6 * ((days_birth / 365 / 50) + 0.4) +
  0.6 * (log(credit_ratio) - 0.9) +
  rnorm(n_train, 0, 0.2)
p_default <- plogis(z)
target <- as.integer(rbinom(n_train, 1, p_default))

sk_ids <- 100001:(100000 + n_train)

application_train <- data.frame(
  SK_ID_CURR = sk_ids,
  TARGET = target,
  AMT_INCOME_TOTAL = amt_income_total,
  AMT_CREDIT = amt_credit,
  AMT_ANNUITY = amt_annuity,
  DAYS_BIRTH = days_birth,
  DAYS_EMPLOYED = days_employed,
  EXT_SOURCE_1 = round(ext_source_1, 4),
  EXT_SOURCE_2 = round(ext_source_2, 4),
  EXT_SOURCE_3 = round(ext_source_3, 4)
)

# Standalone demo CSV (not consumed by src/train.R).
write.csv(application_train, opt$output_path, row.names = FALSE)

set.seed(opt$seed + 1L)
test_sk_ids <- (200001:(200000 + n_test))

application_test <- data.frame(
  SK_ID_CURR = test_sk_ids,
  AMT_INCOME_TOTAL = round(rlnorm(n_test, meanlog = 10.7, sdlog = 0.55), 2),
  AMT_CREDIT = round(runif(n_test, min = 50000, max = 600000), 2),
  AMT_ANNUITY = round(runif(n_test, min = 5000, max = 40000), 2),
  DAYS_BIRTH = as.integer(-runif(n_test, min = 5000, max = 30000)),
  DAYS_EMPLOYED = as.integer(-runif(n_test, min = 0, max = 15000)),
  EXT_SOURCE_1 = round(pmax(0, pmin(1, rnorm(n_test, 0.5, 0.2))), 4),
  EXT_SOURCE_2 = round(pmax(0, pmin(1, rnorm(n_test, 0.55, 0.2))), 4),
  EXT_SOURCE_3 = round(pmax(0, pmin(1, rnorm(n_test, 0.45, 0.2))), 4)
)

bureau <- data.frame(
  SK_ID_CURR = rep(sk_ids, each = 2L),
  AMT_CREDIT_SUM = round(runif(n_train * 2, min = 1000, max = 200000), 2),
  AMT_CREDIT_SUM_OVERDUE = round(runif(n_train * 2, min = 0, max = 5000), 2)
)

bureau_balance <- data.frame(
  SK_ID_BUREAU = seq_len(n_train * 2),
  MONTHS_BALANCE = sample(-12:0, n_train * 2, replace = TRUE),
  STATUS = sample(c("0", "1", "C"), n_train * 2, replace = TRUE)
)

previous_application <- data.frame(
  SK_ID_CURR = sample(sk_ids, n_train * 3, replace = TRUE),
  NAME_CONTRACT_STATUS = sample(c("Approved", "Refused", "Canceled"), n_train * 3, replace = TRUE)
)

installments_payments <- data.frame(
  SK_ID_CURR = sample(sk_ids, n_train * 4, replace = TRUE),
  DAYS_ENTRY_PAYMENT = as.integer(runif(n_train * 4, min = -30, max = 30)),
  DAYS_INSTALMENT = as.integer(runif(n_train * 4, min = -60, max = 0)),
  AMT_INSTALMENT = round(runif(n_train * 4, min = 100, max = 5000), 2),
  AMT_PAYMENT = round(runif(n_train * 4, min = 50, max = 5000), 2)
)

pos_cash_balance <- data.frame(
  SK_ID_CURR = sample(sk_ids, n_train * 2, replace = TRUE),
  SK_DPD = as.integer(runif(n_train * 2, min = 0, max = 30))
)

credit_card_balance <- data.frame(
  SK_ID_CURR = sample(sk_ids, n_train * 2, replace = TRUE),
  SK_DPD = as.integer(runif(n_train * 2, min = 0, max = 30))
)

zip_staging <- tempfile("mock_home_credit_")
dir.create(zip_staging, recursive = TRUE)

csv_files <- list(
  application_train.csv = application_train,
  application_test.csv = application_test,
  bureau.csv = bureau,
  bureau_balance.csv = bureau_balance,
  previous_application.csv = previous_application,
  installments_payments.csv = installments_payments,
  POS_CASH_balance.csv = pos_cash_balance,
  credit_card_balance.csv = credit_card_balance
)

for (name in names(csv_files)) {
  write.csv(csv_files[[name]], file.path(zip_staging, name), row.names = FALSE)
}

if (zip_exists) {
  unlink(opt$zip_path)
}
csv_paths <- file.path(zip_staging, names(csv_files))
utils::zip(opt$zip_path, files = csv_paths, flags = "-j")
unlink(zip_staging, recursive = TRUE)

log_json("INFO", "Sample data generated successfully",
  train_rows = nrow(application_train),
  test_rows = nrow(application_test),
  default_rate = round(mean(application_train$TARGET), 4),
  csv_path = opt$output_path,
  zip_path = opt$zip_path
)
