# ------------------------------------------------------------------------------
# api/plumber.R
#
# Agibank Credit Risk Model API - Plumber definition.
#
# Endpoints:
#   GET  /health      Health check (200 / 503)
#   GET  /model-info  Model metadata, schema and runtime information
#   POST /predict     Credit default probability (single or batch)
#   GET  /__docs__/   Auto-generated OpenAPI/Swagger documentation
#
# Logging: structured JSON middleware via a request filter.
# ------------------------------------------------------------------------------

#* @apiTitle Agibank Credit Risk Model API
#* @apiDescription Production REST API serving the Agibank credit risk model. \
#*   Calculates the default probability of a customer using a Random Forest \
#*   (ranger) trained on Home Credit Default Risk-style data.
#* @apiVersion v1.0.0
#* @apiTag risk "Credit risk prediction endpoints"
#* @apiTag system "Operational endpoints"

suppressPackageStartupMessages({
  library(jsonlite)
  library(plumber)
  library(ranger)
})

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a


# --- Configuration (environment variables with sensible defaults) -------------

# plumber changes the process working directory to the router file's directory
# before sourcing it, so relative paths must be resolved against the project
# root (one level above the api/ directory).
PROJECT_ROOT <- tryCatch(
  dirname(normalizePath(getwd())),
  error = function(e) getwd()
)

resolve_path <- function(p) {
  if (is.null(p) || !nzchar(p) || grepl("^/", p)) p else file.path(PROJECT_ROOT, p)
}

MODEL_OUTPUT_PATH <- Sys.getenv("MODEL_OUTPUT_PATH", "models/model_v1.rds")
METADATA_PATH     <- Sys.getenv("METADATA_PATH", "data/features_metadata.json")
LOG_LEVEL <- toupper(Sys.getenv("LOG_LEVEL", "INFO"))

# --- Structured JSON logging --------------------------------------------------
# Self-contained JSON logger: bypassing the `logger` package's glue formatter
# avoids brace-mangling of JSON messages entirely, while keeping LOG_LEVEL gating.

now_iso <- function() format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")

LOG_LEVEL_RANK <- c(TRACE = 0, DEBUG = 1, INFO = 2, WARN = 3, ERROR = 4, FATAL = 5)

CURRENT_LOG_LEVEL <- LOG_LEVEL
if (!(CURRENT_LOG_LEVEL %in% names(LOG_LEVEL_RANK))) {
  cat(toJSON(list(
    timestamp = now_iso(),
    level = "WARN",
    logger = "api",
    message = paste0("Unknown LOG_LEVEL '", LOG_LEVEL, "', falling back to INFO")
  ), auto_unbox = TRUE), "\n")
  CURRENT_LOG_LEVEL <- "INFO"
}

log_at <- function(level, message) {
  if (LOG_LEVEL_RANK[[level]] >= LOG_LEVEL_RANK[[CURRENT_LOG_LEVEL]]) {
    cat(toJSON(list(
      timestamp = now_iso(),
      level = level,
      logger = "api",
      message = message
    ), auto_unbox = TRUE), "\n")
  }
}

# --- Model and metadata loading -----------------------------------------------

MODEL <- NULL
MODEL_LOADED <- FALSE

tryCatch(
  {
    if (!file.exists(MODEL_OUTPUT_PATH)) {
      stop(sprintf("Model artifact not found: %s (run training first)", MODEL_OUTPUT_PATH))
    }
    MODEL <- readRDS(MODEL_OUTPUT_PATH)
    if (is.null(MODEL$model)) stop("Invalid model artifact: missing 'model' object")
    if (is.null(MODEL$features)) stop("Invalid model artifact: missing 'features' list")
    MODEL_LOADED <- TRUE
    log_at("INFO", paste0(
      "Model loaded successfully (version ",
      MODEL$version %||% "unknown",
      ", features: ", length(MODEL$features), ") from ", MODEL_OUTPUT_PATH
    ))
  },
  error = function(e) {
    log_at("ERROR", paste("Failed to load model:", conditionMessage(e)))
  }
)

METADATA <- NULL
if (file.exists(METADATA_PATH)) {
  METADATA <- tryCatch(
    fromJSON(METADATA_PATH, simplifyVector = FALSE),
    error = function(e) {
      log_at("WARN", paste("Failed to parse metadata file:", conditionMessage(e)))
      NULL
    }
  )
} else {
  log_at("WARN", paste0("Metadata file not found at '", METADATA_PATH, "'"))
}

# Features the trained model expects (authoritative for validation).
FEATURE_NAMES <- MODEL$features %||% {
  vapply(METADATA$features %||% list(), function(f) f$name, character(1))
}
FEATURE_NAMES <- as.character(FEATURE_NAMES)

# Feature specs from the schema file, keyed by name, used for type checks.
FEATURE_SPECS <- list()
if (!is.null(METADATA$features)) {
  for (spec in METADATA$features) FEATURE_SPECS[[spec$name]] <- spec$type %||% "numeric"
}

RISK_CUTOFF   <- METADATA$thresholds$high_risk_cutoff %||% 0.5
MODEL_VERSION <- MODEL$version %||% METADATA$model_version %||% "unknown"

# Build a static example body from feature metadata (used in 400 responses).
EXAMPLE_BODY <- local({
  example_defaults <- list(
    AMT_INCOME_TOTAL = 150000.0, AMT_CREDIT = 450000.0,
    AMT_ANNUITY = 25000.0, DAYS_BIRTH = -12000,
    DAYS_EMPLOYED = -2000, EXT_SOURCE_2 = 0.55,
    EXT_SOURCE_3 = 0.42
  )
  out <- list()
  for (fname in FEATURE_NAMES) {
    out[[fname]] <- example_defaults[[fname]] %||%
      if ((FEATURE_SPECS[[fname]] %||% "numeric") == "numeric") 0.0 else "value"
  }
  out
})

# --- Middleware: structured request logging -----------------------------------

#* @filter log_request
function(req, res) {
  start <- Sys.time()
  forward()
  log_at("INFO", toJSON(list(
    event = "http_request",
    method = req$REQUEST_METHOD,
    path = req$PATH_INFO,
    query = req$QUERY_STRING %||% "",
    user_agent = req$HTTP_USER_AGENT %||% "",
    remote_addr = req$REMOTE_ADDR %||% "",
    status = res$status %||% 500,
    duration_ms = round(as.numeric(difftime(Sys.time(), start, units = "secs")) * 1000, 2)
  ), auto_unbox = TRUE))
}

# --- Helpers ------------------------------------------------------------------

error_payload <- function(error_code, message, include_example = FALSE) {
  payload <- list(error = error_code, message = message, timestamp = now_iso())
  if (include_example) payload$example_body <- EXAMPLE_BODY
  payload
}

new_prediction_id <- function() {
  paste0("req-", paste(sample(c(as.character(0:9), letters[1:6]), 8, replace = TRUE), collapse = ""))
}

risk_label <- function(probability) {
  if (probability >= RISK_CUTOFF) "HIGH_RISK" else "LOW_RISK"
}

# Validates a single record against the required features and their types.
# Returns a character vector of problems (empty vector when the record is valid).
validate_record <- function(record) {
  problems <- character(0)

  if (!is.list(record) || length(record) == 0) {
    return("Request body must be a JSON object or an array of objects")
  }

  for (fname in FEATURE_NAMES) {
    if (!fname %in% names(record)) {
      problems <- c(problems, paste0("Missing required field: ", fname))
      next
    }

    value <- record[[fname]]
    expected_type <- FEATURE_SPECS[[fname]] %||% "numeric"

    if (is.null(value)) {
      problems <- c(problems, paste0("Field '", fname, "' must not be null"))
      next
    }
    if (length(value) != 1 || is.list(value)) {
      problems <- c(problems, paste0("Field '", fname, "' must be a single value"))
      next
    }

    if (expected_type == "numeric" && (!is.numeric(value) || is.na(value) || is.infinite(value))) {
      problems <- c(problems, paste0(
        "Field '", fname, "' must be a finite numeric value, got: ",
        paste(deparse(value), collapse = "")
      ))
    } else if (expected_type %in% c("categorical", "string", "text") && !is.character(value)) {
      problems <- c(problems, paste0(
        "Field '", fname, "' must be a character value, got: ",
        paste(deparse(value), collapse = "")
      ))
    }
  }

  problems
}

# Scores a validated record, returning the default probability.
predict_record <- function(record) {
  df <- as.data.frame(matrix(unlist(record[FEATURE_NAMES]), nrow = 1))
  names(df) <- FEATURE_NAMES
  for (col in FEATURE_NAMES) df[[col]] <- as.numeric(df[[col]])

  pred <- predict(MODEL$model, data = df)
  prob_col <- match(MODEL$default_label %||% "1", colnames(pred$predictions))
  if (is.na(prob_col)) prob_col <- ncol(pred$predictions)
  as.numeric(pred$predictions[, prob_col])
}

# --- Endpoints ----------------------------------------------------------------

#* Root endpoint with API overview
#* @get /
#* @serializer json list(auto_unbox = TRUE)
#* @tag system
function() {
  list(
    name = "agibank-credit-risk-api",
    version = MODEL_VERSION,
    endpoints = list(
      health = "/health",
      model_info = "/model-info",
      predict = "/predict",
      docs = "/__docs__/"
    ),
    schema_version = METADATA$schema_version %||% NA_character_
  )
}

#* Health check for container monitoring and orchestration
#* @get /health
#* @serializer json list(auto_unbox = TRUE)
#* @tag system
#* @response 200 API is healthy and the model is loaded
#* @response 503 Model failed to load
function(res) {
  if (!MODEL_LOADED) {
    res$status <- 503
    return(list(
      status = "unhealthy",
      timestamp = now_iso(),
      model_loaded = FALSE,
      model_version = MODEL_VERSION
    ))
  }
  list(
    status = "healthy",
    timestamp = now_iso(),
    model_loaded = TRUE,
    model_version = MODEL_VERSION
  )
}

#* Model metadata, feature expectations and runtime information
#* @get /model-info
#* @serializer json list(auto_unbox = TRUE)
#* @tag system
#* @response 200 Model information returned
function() {
  list(
    model_version = MODEL_VERSION,
    schema_version = METADATA$schema_version %||% NA_character_,
    target = METADATA$target %||% list(name = "TARGET", type = "binary"),
    features = METADATA$features %||% lapply(FEATURE_NAMES, function(f) list(name = f, type = "numeric", required = TRUE)),
    thresholds = METADATA$thresholds %||% list(high_risk_cutoff = RISK_CUTOFF),
    runtime = list(
      r_version = R.version.string,
      model_file = basename(MODEL_OUTPUT_PATH),
      model_size_bytes = if (file.exists(MODEL_OUTPUT_PATH)) file.size(MODEL_OUTPUT_PATH) else NA_integer_,
      model_hash_md5 = if (file.exists(MODEL_OUTPUT_PATH)) unname(tools::md5sum(MODEL_OUTPUT_PATH)) else NA_character_,
      trained_at = MODEL$trained_at %||% NA_character_
    )
  )
}

#* Calculates the credit default probability for one or more customer records
#* @post /predict
#* @serializer json list(auto_unbox = TRUE)
#* @tag risk
#* @param body:object Body containing the required credit risk features \
#*   (AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, DAYS_BIRTH, DAYS_EMPLOYED, \
#*   EXT_SOURCE_2, EXT_SOURCE_3). Accepts a single object or an array of objects.
#* @response 200 Prediction returned
#* @response 400 Validation failed
#* @response 503 Model not loaded
function(req, res) {
  start <- Sys.time()

  if (!MODEL_LOADED) {
    res$status <- 503
    return(error_payload("ModelNotLoaded", "Model artifact is not loaded; run training first"))
  }

  # --- Pre-validate raw JSON to catch syntax errors (trailing commas, etc.) ---
  raw_body <- req$postBody
  if (!is.null(raw_body) && nzchar(raw_body)) {
    parsed <- tryCatch(
      fromJSON(raw_body, simplifyVector = FALSE),
      error = function(e) e
    )
    if (inherits(parsed, "error")) {
      res$status <- 400
      return(list(
        error     = "InvalidJSON",
        message   = paste0(
          "The request body contains invalid JSON: ",
          conditionMessage(parsed),
          ". Common causes: trailing commas, unquoted keys, or missing fields."
        ),
        hint            = "Remove any trailing commas and ensure all keys are quoted strings.",
        required_fields = FEATURE_NAMES,
        example_body    = EXAMPLE_BODY,
        timestamp       = now_iso()
      ))
    }
  }

  body <- req$body

  # Parse raw JSON when plumber did not auto-parse it (e.g. no content-type).
  if (is.character(body)) {
    body <- tryCatch(
      fromJSON(body, simplifyVector = FALSE),
      error = function(e) {
        res$status <- 400
        return(list(
          error           = "InvalidJSON",
          message         = paste0("Invalid JSON payload: ", conditionMessage(e)),
          required_fields = FEATURE_NAMES,
          example_body    = EXAMPLE_BODY,
          timestamp       = now_iso()
        ))
      }
    )
    if (res$status == 400) return(body)
  }

  if (is.null(body)) {
    res$status <- 400
    return(error_payload("ValidationFailed", "Request body is required", include_example = TRUE))
  }

  if (length(body) == 0) {
    res$status <- 400
    return(error_payload("ValidationFailed", "Request body must not be empty", include_example = TRUE))
  }

  # Defensive: plumber may parse an array of objects as a data.frame.
  if (is.data.frame(body)) {
    body <- lapply(seq_len(nrow(body)), function(i) as.list(body[i, , drop = FALSE]))
  }

  # Single record vs batch of records.
  is_single <- !is.null(names(body)) && any(names(body) %in% FEATURE_NAMES)
  records <- if (is_single) list(body) else body
  if (!is.list(records)) records <- list(records)

  # Validate every record first; fail fast with explicit details on any error.
  all_problems <- list()
  for (i in seq_along(records)) {
    problems <- validate_record(records[[i]])
    if (length(problems) > 0) all_problems[[as.character(i)]] <- problems
  }

  if (length(all_problems) > 0) {
    res$status <- 400
    details <- vapply(names(all_problems), function(i) {
      paste0("[record ", i, "] ", paste(all_problems[[i]], collapse = "; "))
    }, character(1))
    return(list(
      error           = "ValidationFailed",
      message         = paste(details, collapse = " | "),
      required_fields = FEATURE_NAMES,
      example_body    = EXAMPLE_BODY,
      timestamp       = now_iso()
    ))
  }

  results <- lapply(records, function(record) {
    probability <- tryCatch(
      predict_record(record),
      error = function(e) {
        log_at("ERROR", paste("Prediction failed:", conditionMessage(e)))
        NULL
      }
    )

    if (is.null(probability)) {
      list(
        error = "PredictionError",
        message = "Model failed to score the record",
        timestamp = now_iso()
      )
    } else {
      list(
        prediction_id = new_prediction_id(),
        timestamp = now_iso(),
        default_probability = round(probability, 6),
        risk_label = risk_label(probability),
        model_version = MODEL_VERSION,
        processing_time_ms = round(as.numeric(difftime(Sys.time(), start, units = "secs")) * 1000, 2)
      )
    }
  })

  if (any(vapply(results, function(r) !is.null(r$error), logical(1)))) {
    res$status <- 500
  }

  if (is_single) results[[1]] else results
}
