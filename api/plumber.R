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

MODEL_OUTPUT_PATH <- resolve_path(Sys.getenv("MODEL_OUTPUT_PATH", "models/model_v1.rds"))
METADATA_PATH     <- resolve_path(Sys.getenv("METADATA_PATH", "data/features_metadata.json"))
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

# External credit source fields (EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3) are
# NOT required by the API. When the client supplies one of them, its value is
# fed to the model together with the required features. When the client omits
# it, the API falls back to the average value of that field computed over the
# training feature store when possible (and to 0.0 otherwise).
OPTIONAL_REQUEST_FIELDS <- c("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")

# Each optional request field maps to the model features it can consume. The
# legacy EXT_SOURCE_2/EXT_SOURCE_3 columns were renamed to
# THIRD_PARTY_CREDIT_SCORE_2/THIRD_PARTY_CREDIT_SCORE_3 in the trained model
# contract, so both names are candidates depending on the model artifact.
EXT_SOURCE_MODEL_ALIASES <- list(
  EXT_SOURCE_1 = c("EXT_SOURCE_1"),
  EXT_SOURCE_2 = c("THIRD_PARTY_CREDIT_SCORE_2", "EXT_SOURCE_2"),
  EXT_SOURCE_3 = c("THIRD_PARTY_CREDIT_SCORE_3", "EXT_SOURCE_3")
)

# Request field -> model feature(s) it actually feeds (only those the trained
# model really contains). Model features -> request fields is derived below.
REQUEST_TO_MODEL_FEATURES <- list()
OPTIONAL_MODEL_FEATURES <- character(0)
for (req_field in OPTIONAL_REQUEST_FIELDS) {
  present <- EXT_SOURCE_MODEL_ALIASES[[req_field]]
  present <- present[present %in% FEATURE_NAMES]
  REQUEST_TO_MODEL_FEATURES[[req_field]] <- present
  OPTIONAL_MODEL_FEATURES <- unique(c(OPTIONAL_MODEL_FEATURES, present))
}
OPTIONAL_MODEL_FEATURES <- as.character(OPTIONAL_MODEL_FEATURES)

MODEL_FEATURE_REQUEST_FIELDS <- setNames(vector("list", length(FEATURE_NAMES)), FEATURE_NAMES)
for (req_field in names(REQUEST_TO_MODEL_FEATURES)) {
  for (mf in REQUEST_TO_MODEL_FEATURES[[req_field]]) {
    MODEL_FEATURE_REQUEST_FIELDS[[mf]] <- c(MODEL_FEATURE_REQUEST_FIELDS[[mf]], req_field)
  }
}

# Only these model fields are mandatory in the payload; the external credit
# source fields are excluded from the required set.
REQUIRED_MODEL_FEATURES <- setdiff(FEATURE_NAMES, OPTIONAL_MODEL_FEATURES)
RECOGNIZED_INPUT_FIELDS <- unique(c(FEATURE_NAMES, OPTIONAL_REQUEST_FIELDS))

# Feature specs from the schema file, keyed by name, used for type checks.
FEATURE_SPECS <- list()
if (!is.null(METADATA$features)) {
  for (spec in METADATA$features) FEATURE_SPECS[[spec$name]] <- spec$type %||% "numeric"
}

FEATURE_STORE_PATH <- resolve_path(Sys.getenv("FEATURE_STORE_PATH", "data/feature_store.rds"))
FEATURE_DEFAULTS <- NULL

load_feature_defaults <- function() {
  if (!is.null(FEATURE_DEFAULTS)) return(FEATURE_DEFAULTS)

  defaults <- list()

  if (!is.null(METADATA$features)) {
    for (spec in METADATA$features) {
      if (!is.null(spec$imputed_default)) {
        defaults[[spec$name]] <- spec$imputed_default
      }
    }
  }

  if (!is.null(MODEL$imputation_map)) {
    for (fname in names(MODEL$imputation_map)) {
      defaults[[fname]] <- MODEL$imputation_map[[fname]]
    }
  }

  if (file.exists(FEATURE_STORE_PATH)) {
    feature_store <- tryCatch(
      readRDS(FEATURE_STORE_PATH),
      error = function(e) {
        log_at("WARN", paste("Failed to load feature store defaults:", conditionMessage(e)))
        NULL
      }
    )

    if (!is.null(feature_store) && is.data.frame(feature_store)) {
      store_defaults <- c(
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        FEATURE_NAMES
      )
      for (fname in unique(store_defaults)) {
        if (fname %in% names(feature_store) && is.numeric(feature_store[[fname]])) {
          defaults[[fname]] <- mean(feature_store[[fname]], na.rm = TRUE)
        }
      }
    }
  }

  if ("EXT_SOURCE_2" %in% names(defaults) && "THIRD_PARTY_CREDIT_SCORE_2" %in% FEATURE_NAMES) {
    defaults[["THIRD_PARTY_CREDIT_SCORE_2"]] <- defaults[["EXT_SOURCE_2"]]
  }
  if ("EXT_SOURCE_3" %in% names(defaults) && "THIRD_PARTY_CREDIT_SCORE_3" %in% FEATURE_NAMES) {
    defaults[["THIRD_PARTY_CREDIT_SCORE_3"]] <- defaults[["EXT_SOURCE_3"]]
  }

  FEATURE_DEFAULTS <<- defaults
  FEATURE_DEFAULTS
}

resolve_feature_default <- function(feature_name) {
  defaults <- load_feature_defaults()
  default_value <- defaults[[feature_name]] %||% NULL
  if (!is.null(default_value) && length(default_value) == 1 && !is.na(default_value)) {
    return(default_value)
  }
  if (feature_name %in% c("THIRD_PARTY_CREDIT_SCORE_2", "EXT_SOURCE_2")) {
    return(0.0)
  }
  if (feature_name %in% c("THIRD_PARTY_CREDIT_SCORE_3", "EXT_SOURCE_3")) {
    return(0.0)
  }
  if (feature_name == "EXT_SOURCE_1") {
    return(0.0)
  }
  0.0
}

RISK_CUTOFF   <- METADATA$thresholds$high_risk_cutoff %||% 0.5
MODEL_VERSION <- MODEL$version %||% METADATA$model_version %||% "unknown"

# Build a static example body from feature metadata (used in 400 responses).
EXAMPLE_BODY <- local({
  example_defaults <- list(
    AMT_INCOME_TOTAL = 150000.0, AMT_CREDIT = 450000.0,
    AMT_ANNUITY = 25000.0, DAYS_BIRTH = -12000,
    DAYS_EMPLOYED = -2000
  )
  out <- list()
  for (fname in FEATURE_NAMES) {
    out[[fname]] <- example_defaults[[fname]] %||%
      resolve_feature_default(fname)
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

normalize_record <- function(record) {
  canonical <- list()

  for (fname in FEATURE_NAMES) {
    candidates <- c(fname, MODEL_FEATURE_REQUEST_FIELDS[[fname]])
    present <- candidates[candidates %in% names(record)]

    if (length(present) > 0 && !is.null(record[[present[1]]])) {
      # Use the value the client actually provided.
      canonical[[fname]] <- record[[present[1]]]
    } else if (fname %in% OPTIONAL_MODEL_FEATURES) {
      # Optional external credit source omitted: fall back to the average
      # value from training when possible.
      canonical[[fname]] <- resolve_feature_default(fname)
    } else {
      canonical[[fname]] <- NULL
    }
  }

  canonical
}

new_prediction_id <- function() {
  paste0("req-", paste(sample(c(as.character(0:9), letters[1:6]), 8, replace = TRUE), collapse = ""))
}

risk_label <- function(probability) {
  if (probability >= RISK_CUTOFF) "HIGH_RISK" else "LOW_RISK"
}

# Validates a single feature value against the expected type.
# Returns a character vector of problems (empty vector when the value is valid).
validate_feature_value <- function(fname, value) {
  expected_type <- FEATURE_SPECS[[fname]] %||% "numeric"

  if (length(value) != 1 || is.list(value)) {
    return(paste0("Field '", fname, "' must be a single value"))
  }

  if (expected_type %in% c("categorical", "string", "text")) {
    if (!is.character(value)) {
      return(paste0(
        "Field '", fname, "' must be a character value, got: ",
        paste(deparse(value), collapse = "")
      ))
    }
    return(character(0))
  }

  if (!is.numeric(value) || is.na(value) || is.infinite(value)) {
    return(paste0(
      "Field '", fname, "' must be a finite numeric value, got: ",
      paste(deparse(value), collapse = "")
    ))
  }

  character(0)
}

# Validates a single record against the required features and their types.
# Returns a character vector of problems (empty vector when the record is valid).
validate_record <- function(record) {
  problems <- character(0)

  if (!is.list(record) || length(record) == 0) {
    return("Request body must be a JSON object or an array of objects")
  }

  normalized <- normalize_record(record)

  for (fname in REQUIRED_MODEL_FEATURES) {
    if (is.null(normalized[[fname]])) {
      problems <- c(problems, paste0("Missing required field: ", fname))
      next
    }
    problems <- c(problems, validate_feature_value(fname, normalized[[fname]]))
  }

  # Optional external credit source fields are never mandatory, but a value
  # the client supplies must still be valid so it can feed the model.
  for (req_field in OPTIONAL_REQUEST_FIELDS) {
    if (req_field %in% names(record) && !is.null(record[[req_field]])) {
      problems <- c(problems, validate_feature_value(req_field, record[[req_field]]))
    }
  }

  problems
}

# Scores a validated record, returning the default probability.
predict_record <- function(record) {
  normalized <- normalize_record(record)
  df <- as.data.frame(matrix(unlist(normalized[FEATURE_NAMES]), nrow = 1))
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
  base_features <- METADATA$features %||% lapply(FEATURE_NAMES, function(f) list(name = f, type = "numeric", required = TRUE))
  features <- lapply(base_features, function(f) {
    if (!is.null(f$name) && f$name %in% OPTIONAL_MODEL_FEATURES) {
      f$required <- FALSE
    }
    f
  })

  optional_features <- lapply(OPTIONAL_REQUEST_FIELDS, function(field_name) {
    model_fields <- REQUEST_TO_MODEL_FEATURES[[field_name]]
    description <- if (length(model_fields) > 0) {
      paste0(
        "Optional external credit source. When provided, its value feeds the model feature(s): ",
        paste(model_fields, collapse = ", "),
        ". When omitted, the API falls back to the average training value when available."
      )
    } else {
      paste0(
        "Optional external credit source accepted for compatibility; not part of the current ",
        "trained model contract. When omitted, the API falls back to the average training value ",
        "when available."
      )
    }
    list(
      name = field_name,
      required = FALSE,
      type = "numeric",
      description = description
    )
  })

  list(
    model_version = MODEL_VERSION,
    schema_version = METADATA$schema_version %||% NA_character_,
    target = METADATA$target %||% list(name = "TARGET", type = "binary"),
    features = features,
    optional_features = optional_features,
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
#*   (AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, DAYS_BIRTH, DAYS_EMPLOYED). \
#*   External credit source fields EXT_SOURCE_1, EXT_SOURCE_2 and EXT_SOURCE_3 \
#*   are NOT required: when provided they are fed to the model (EXT_SOURCE_2 and \
#*   EXT_SOURCE_3 map to the trained features THIRD_PARTY_CREDIT_SCORE_2 and \
#*   THIRD_PARTY_CREDIT_SCORE_3), and when omitted they are filled with the \
#*   average training value for that field when available.
#*   Accepts a single object or an array of objects.
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
        required_fields = REQUIRED_MODEL_FEATURES,
        optional_fields = OPTIONAL_REQUEST_FIELDS,
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
          required_fields = REQUIRED_MODEL_FEATURES,
          optional_fields = OPTIONAL_REQUEST_FIELDS,
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
  is_single <- !is.null(names(body)) && any(names(body) %in% RECOGNIZED_INPUT_FIELDS)
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
      required_fields = REQUIRED_MODEL_FEATURES,
      optional_fields = OPTIONAL_REQUEST_FIELDS,
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
