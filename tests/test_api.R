# ------------------------------------------------------------------------------
# tests/test_api.R - integration test suite for the Plumber API.
#
# Requires a running API instance. Target URL is configurable via API_URL
# (default: http://localhost:8080).
#
# Run with:
#   Rscript -e "testthat::test_file('tests/test_api.R', reporter = 'summary')"
#
# testthat >= 3.0 makes test_file() fail with a non-zero exit code whenever a
# test fails, which is what CI relies on.
# ------------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(httr)
  library(testthat)
  library(jsonlite)
})

api_url <- sub("/$", "", Sys.getenv("API_URL", "http://localhost:8080"))

# Wait until the API is reachable (bounded retry, useful for local runs).
wait_for_api <- function(timeout_sec = 60) {
  deadline <- Sys.time() + timeout_sec
  repeat {
    ok <- tryCatch(status_code(GET(paste0(api_url, "/health"), timeout(2))) == 200, error = function(e) FALSE)
    if (ok) return(invisible(TRUE))
    if (Sys.time() > deadline) {
      stop("API not reachable at ", api_url, " within ", timeout_sec, "s")
    }
    Sys.sleep(1)
  }
}

parse_body <- function(resp) {
  fromJSON(content(resp, "text", encoding = "UTF-8"), simplifyVector = FALSE)
}

wait_for_api()

valid_payload <- list(
  AMT_INCOME_TOTAL = 150000.0,
  AMT_CREDIT = 450000.0,
  AMT_ANNUITY = 25000.0,
  DAYS_BIRTH = -12000,
  DAYS_EMPLOYED = -2000,
  THIRD_PARTY_CREDIT_SCORE_2 = 0.55,
  THIRD_PARTY_CREDIT_SCORE_3 = 0.42
)

# External credit sources (EXT_SOURCE_*) are NOT required; only the five core
# features are mandatory.
minimal_payload <- valid_payload[!names(valid_payload) %in% c("THIRD_PARTY_CREDIT_SCORE_2", "THIRD_PARTY_CREDIT_SCORE_3")]

test_that("GET /health returns a healthy status", {
  resp <- GET(paste0(api_url, "/health"))
  expect_equal(status_code(resp), 200)
  body <- parse_body(resp)
  expect_equal(body$status, "healthy")
  expect_true(isTRUE(body$model_loaded))
  expect_equal(body$model_version, "v1.0.0")
  expect_true(nchar(body$timestamp) > 0)
})

test_that("GET /model-info exposes schema metadata and runtime info", {
  resp <- GET(paste0(api_url, "/model-info"))
  expect_equal(status_code(resp), 200)
  body <- parse_body(resp)
  expect_equal(body$schema_version, "1.0.0")
  expect_equal(body$model_version, "v1.0.0")
  expect_gte(length(body$features), 7)
  expect_true(any(vapply(body$features, function(f) identical(f$name, "AMT_INCOME_TOTAL"), logical(1))))
  optional_names <- vapply(body$optional_features, function(f) f$name, character(1))
  expect_true(all(c("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") %in% optional_names))
  expect_equal(body$thresholds$high_risk_cutoff, 0.5)
  expect_true(nchar(body$runtime$r_version) > 0)
  expect_true(is.character(body$runtime$model_hash_md5) && nchar(body$runtime$model_hash_md5) == 32)
  expect_true(body$runtime$model_size_bytes > 0)
})

test_that("POST /predict returns a valid prediction for a single record", {
  resp <- POST(paste0(api_url, "/predict"), body = valid_payload, encode = "json")
  expect_equal(status_code(resp), 200)
  body <- parse_body(resp)
  expect_match(body$prediction_id, "^req-")
  expect_true(body$default_probability >= 0 && body$default_probability <= 1)
  expect_true(body$risk_label %in% c("LOW_RISK", "HIGH_RISK"))
  expect_true(is.numeric(body$processing_time_ms) && body$processing_time_ms >= 0)
  expect_equal(body$model_version, "v1.0.0")
})

test_that("POST /predict rejects missing required fields with a 400", {
  bad <- valid_payload
  bad$AMT_CREDIT <- NULL
  resp <- POST(paste0(api_url, "/predict"), body = bad, encode = "json")
  expect_equal(status_code(resp), 400)
  body <- parse_body(resp)
  expect_equal(body$error, "ValidationFailed")
  expect_match(body$message, "AMT_CREDIT")
})

test_that("POST /predict accepts a payload without external credit source fields", {
  resp <- POST(paste0(api_url, "/predict"), body = minimal_payload, encode = "json")
  expect_equal(status_code(resp), 200)
  body <- parse_body(resp)
  expect_true(body$default_probability >= 0 && body$default_probability <= 1)
  expect_true(body$risk_label %in% c("LOW_RISK", "HIGH_RISK"))
})

test_that("POST /predict accepts legacy EXT_SOURCE_2/EXT_SOURCE_3 aliases", {
  payload <- minimal_payload
  payload$EXT_SOURCE_2 <- 0.55
  payload$EXT_SOURCE_3 <- 0.42
  resp <- POST(paste0(api_url, "/predict"), body = payload, encode = "json")
  expect_equal(status_code(resp), 200)
})

test_that("POST /predict accepts the optional EXT_SOURCE_1 field", {
  payload <- valid_payload
  payload$EXT_SOURCE_1 <- 0.3
  resp <- POST(paste0(api_url, "/predict"), body = payload, encode = "json")
  expect_equal(status_code(resp), 200)
})

test_that("POST /predict rejects non-numeric values with a 400", {
  bad <- valid_payload
  bad$AMT_INCOME_TOTAL <- "one hundred thousand"
  resp <- POST(paste0(api_url, "/predict"), body = bad, encode = "json")
  expect_equal(status_code(resp), 400)
  body <- parse_body(resp)
  expect_equal(body$error, "ValidationFailed")
  expect_match(body$message, "AMT_INCOME_TOTAL")
})

test_that("POST /predict rejects explicit null values for required fields with a 400", {
  resp <- POST(
    paste0(api_url, "/predict"),
    body = paste0(
      '{"AMT_INCOME_TOTAL":150000,"AMT_CREDIT":450000,"AMT_ANNUITY":25000,',
      '"DAYS_BIRTH":null,"DAYS_EMPLOYED":-2000,"THIRD_PARTY_CREDIT_SCORE_2":0.55,"THIRD_PARTY_CREDIT_SCORE_3":0.42}'
    ),
    content_type_json()
  )
  expect_equal(status_code(resp), 400)
  body <- parse_body(resp)
  expect_equal(body$error, "ValidationFailed")
})

test_that("POST /predict treats null optional external fields as absent", {
  resp <- POST(
    paste0(api_url, "/predict"),
    body = paste0(
      '{"AMT_INCOME_TOTAL":150000,"AMT_CREDIT":450000,"AMT_ANNUITY":25000,',
      '"DAYS_BIRTH":-12000,"DAYS_EMPLOYED":-2000,"THIRD_PARTY_CREDIT_SCORE_2":0.55,"THIRD_PARTY_CREDIT_SCORE_3":null}'
    ),
    content_type_json()
  )
  expect_equal(status_code(resp), 200)
})

test_that("POST /predict rejects an empty body with a 400", {
  resp <- POST(paste0(api_url, "/predict"), body = "", content_type_json())
  expect_equal(status_code(resp), 400)
})

test_that("POST /predict supports batch (array) requests", {
  resp <- POST(
    paste0(api_url, "/predict"),
    body = list(valid_payload, valid_payload),
    encode = "json"
  )
  expect_equal(status_code(resp), 200)
  body <- parse_body(resp)
  expect_length(body, 2)
  expect_match(body[[1]]$prediction_id, "^req-")
})

test_that("POST /predict rejects a batch containing an invalid record", {
  bad <- valid_payload
  bad$DAYS_BIRTH <- "not-a-number"
  resp <- POST(
    paste0(api_url, "/predict"),
    body = list(valid_payload, bad),
    encode = "json"
  )
  expect_equal(status_code(resp), 400)
  body <- parse_body(resp)
  expect_equal(body$error, "ValidationFailed")
})

test_that("GET /__docs__/ serves the Swagger/OpenAPI documentation", {
  resp <- GET(paste0(api_url, "/__docs__/"))
  expect_equal(status_code(resp), 200)
})

test_that("Unknown endpoints return 404", {
  resp <- GET(paste0(api_url, "/does-not-exist"))
  expect_equal(status_code(resp), 404)
})
