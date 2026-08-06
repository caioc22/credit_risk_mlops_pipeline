#!/usr/bin/env Rscript
# ------------------------------------------------------------------------------
# api/start_api.R
#
# Entrypoint for serving the Plumber API. Binds to 0.0.0.0 so the API is
# reachable from outside the container and exposes the Swagger/OpenAPI docs
# at /__docs__/.
#
# Configuration:
#   PORT (env, default: 8080)
#   HOST (env, default: 0.0.0.0)
# ------------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(jsonlite)
  library(plumber)
})

host <- Sys.getenv("HOST", "0.0.0.0")
port <- as.integer(Sys.getenv("PORT", "8080"))

pr <- plumber::pr("api/plumber.R")

cat(jsonlite::toJSON(list(
  timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  level = "INFO",
  logger = "api",
  event = "startup",
  message = "Starting Plumber API",
  host = host,
  port = port,
  docs = "/__docs__/"
), auto_unbox = TRUE), "\n")

plumber::pr_run(pr, host = host, port = port, docs = TRUE, debug = FALSE)
