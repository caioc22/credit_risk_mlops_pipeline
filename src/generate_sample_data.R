#!/usr/bin/env Rscript
# ------------------------------------------------------------------------------
# src/generate_sample_data.R
#
# Generates a synthetic dataset that mimics the Home Credit Default Risk schema
# used by Agibank's Data Science team. The data is only created if the target
# file does not exist yet (unless --force is passed), so the pipeline stays
# deterministic and reproducible across runs.
#
# Configuration (CLI args override environment variables, which override
# defaults):
#   SAMPLE_DATA_PATH    (env, default: data/sample_credit_data.csv)
#   FEATURE_STORE_PATH  (env, default: data/feature_store.rds)
#   N_ROWS              (env, default: 3000)
#   --seed              (default: 42)
#
# Note: DATA_PATH is reserved for src/train.R (feature store input) and must
# not be reused here, because the Dockerfile sets it to data/feature_store.rds.
#
# Outputs:
#   - data/sample_credit_data.csv  (mock raw dataset for CI / local runs)
#   - data/feature_store.rds       (RDS store consumed by src/train.R)
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
    help = "Output CSV path [default: %default]"
  ),
  make_option(c("-f", "--feature-store-path"),
    type = "character",
    default = Sys.getenv("FEATURE_STORE_PATH", "data/feature_store.rds"),
    help = "Output RDS feature store path [default: %default]"
  ),
  make_option(c("-n", "--n-rows"),
    type = "integer",
    default = as.integer(Sys.getenv("N_ROWS", "3000")),
    help = "Number of rows to generate [default: %default]"
  ),
  make_option("--seed",
    type = "integer",
    default = 42,
    help = "Random seed for reproducibility [default: %default]"
  ),
  make_option("--force",
    action = "store_true",
    default = FALSE,
    help = "Overwrite the output file even if it already exists"
  )
)

opt <- parse_args(OptionParser(option_list = option_list))

# optparse normalizes option names to hyphens; map back to underscore accessors.
opt$output_path <- opt$`output-path`
opt$feature_store_path <- opt$`feature-store-path`
opt$n_rows <- opt$`n-rows`

set.seed(opt$seed)

csv_exists <- file.exists(opt$output_path)
rds_exists <- file.exists(opt$feature_store_path)

if (csv_exists && rds_exists && !opt$force) {
  log_json("INFO", "Output files already exist; skipping generation",
    csv_path = opt$output_path,
    feature_store_path = opt$feature_store_path
  )
  quit(save = "no", status = 0)
}

dir.create(dirname(opt$output_path), showWarnings = FALSE, recursive = TRUE)
dir.create(dirname(opt$feature_store_path), showWarnings = FALSE, recursive = TRUE)

n <- opt$n_rows

# --- Synthetic feature generation (seeded, so fully reproducible) ------------
# EXT_SOURCE scores: normalized external credit scores in [0, 1].
ext_source_2 <- pmax(0, pmin(1, rnorm(n, mean = 0.55, sd = 0.2)))
ext_source_3 <- pmax(0, pmin(1, rnorm(n, mean = 0.45, sd = 0.2)))

# Age / employment history in days (negative by Home Credit convention).
days_birth <- as.integer(-runif(n, min = 5000, max = 30000))
days_employed <- as.integer(-runif(n, min = 0, max = 15000))

# Income, credit amount and annuity with realistic magnitudes.
amt_income_total <- round(rlnorm(n, meanlog = 10.7, sdlog = 0.55), 2)
credit_ratio <- runif(n, min = 1.5, max = 4.0)
amt_credit <- round(amt_income_total * credit_ratio, 2)
amt_annuity <- round(amt_credit / runif(n, min = 6, max = 20), 2)

# --- Target: TARGET = 1 with a probability that depends on the features -------
# Higher EXT_SOURCE scores / older clients / lower credit-to-income ratios
# reduce the default probability, injecting real predictive signal.
z <- -0.4 +
  -2.5 * (ext_source_2 - 0.5) -
  3.0 * (ext_source_3 - 0.45) +
  0.6 * ((days_birth / 365 / 50) + 0.4) +
  0.6 * (log(credit_ratio) - 0.9) +
  rnorm(n, 0, 0.2)
p_default <- plogis(z)
target <- as.integer(rbinom(n, 1, p_default))

df <- data.frame(
  SK_ID_CURR = 100001:(100000 + n),
  TARGET = target,
  AMT_INCOME_TOTAL = amt_income_total,
  AMT_CREDIT = amt_credit,
  AMT_ANNUITY = amt_annuity,
  DAYS_BIRTH = days_birth,
  DAYS_EMPLOYED = days_employed,
  EXT_SOURCE_2 = round(ext_source_2, 4),
  EXT_SOURCE_3 = round(ext_source_3, 4)
)

write.csv(df, opt$output_path, row.names = FALSE)

# Train.R expects an RDS feature store with an is_train partition flag.
feature_store <- df
feature_store$is_train <- 1L
saveRDS(feature_store, opt$feature_store_path, compress = "gzip")

log_json("INFO", "Sample data generated successfully",
  rows = nrow(df),
  columns = ncol(df),
  default_rate = round(mean(df$TARGET), 4),
  csv_path = opt$output_path,
  feature_store_path = opt$feature_store_path
)
