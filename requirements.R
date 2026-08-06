#!/usr/bin/env Rscript
# ------------------------------------------------------------------------------
# requirements.R
#
# Installs every R package required by the project in a reproducible way.
# Used by the Docker build step; safe to run repeatedly (idempotent).
#
# Packages:
#   plumber    - REST API framework
#   jsonlite   - JSON parsing/serialization
#   logger     - structured logging
#   ranger     - Random Forest training
#   optparse   - CLI argument parsing
#   testthat   - integration test framework
#   httr       - HTTP client used by the integration tests
#   data.table - fast CSV reading & 1:N relational aggregations (data_modelling.R)
# ------------------------------------------------------------------------------

options(repos = c(CRAN = Sys.getenv("R_REPOS", "https://cloud.r-project.org")))
options(Ncpus = max(1, parallel::detectCores() - 1))

required_packages <- c(
  "plumber",
  "jsonlite",
  "logger",
  "ranger",
  "optparse",
  "testthat",
  "httr",
  "data.table"
)

install_if_missing <- function(pkg) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    message("[requirements] Already installed: ", pkg)
    return(invisible(TRUE))
  }
  message("[requirements] Installing: ", pkg)
  tryCatch(
    install.packages(pkg, quiet = TRUE),
    error = function(e) message("[requirements] Install warning for ", pkg, ": ", conditionMessage(e))
  )
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("[requirements] Failed to install required package: ", pkg)
  }
}

invisible(lapply(required_packages, install_if_missing))

cat("[requirements] All required R packages are installed.\n")
