#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# entrypoint.sh - single execution entrypoint for the Agibank MLOps pipeline.
#
# Usage:
#   bash entrypoint.sh train     - train the model and save model_v1.rds
#   bash entrypoint.sh serve     - start the Plumber API (default)
#   bash entrypoint.sh generate  - (re)generate mock raw zip + demo CSV (not used by train.R directly)
#   bash entrypoint.sh modelling - build the Parquet feature store from raw CSVs
#   bash entrypoint.sh test      - run integration tests against a live API
# ------------------------------------------------------------------------------
set -euo pipefail

MODE="${1:-serve}"
if [ $# -gt 0 ]; then shift; fi

case "${MODE}" in
  train)
    echo "[entrypoint] Mode=train -> Step 1/2: src/data_modelling.R (unzip -> feature store)"
    Rscript src/data_modelling.R
    echo "[entrypoint] Mode=train -> Step 2/2: src/train.R"
    exec Rscript src/train.R "$@"
    ;;
  serve)
    echo "[entrypoint] Mode=serve -> Rscript api/start_api.R (PORT=${PORT:-8080})"
    exec Rscript api/start_api.R
    ;;
  generate)
    echo "[entrypoint] Mode=generate -> Rscript src/generate_sample_data.R"
    exec Rscript src/generate_sample_data.R "$@"
    ;;
  modelling)
    echo "[entrypoint] Mode=modelling -> Rscript src/data_modelling.R"
    exec Rscript src/data_modelling.R "$@"
    ;;
  test)
    echo "[entrypoint] Mode=test -> running integration tests"
    API_URL="${API_URL:-http://localhost:8080}"
    API_PID=""
    if ! Rscript -e "suppressMessages(library(httr)); ok <- tryCatch(status_code(GET('${API_URL}/health', timeout(2))) == 200, error = function(e) FALSE); quit(status = if (ok) 0 else 1)"; then
      echo "[entrypoint] No API reachable at ${API_URL}; starting a local instance for testing..."
      Rscript api/start_api.R > /tmp/api-test.log 2>&1 &
      API_PID=$!
      Rscript -e "suppressMessages(library(httr)); deadline <- Sys.time() + 60; repeat { ok <- tryCatch(status_code(GET('${API_URL}/health', timeout(2))) == 200, error = function(e) FALSE); if (ok) break; if (Sys.time() > deadline) { cat('API did not become ready\n'); quit(status = 1) }; Sys.sleep(1) }" || { tail -50 /tmp/api-test.log >&2; exit 1; }
    fi
    Rscript -e "testthat::test_file('tests/test_api.R', reporter = 'summary')"
    STATUS=$?
    if [ -n "${API_PID}" ]; then kill "${API_PID}" 2>/dev/null || true; fi
    exit "${STATUS}"
    ;;
  *)
    echo "[entrypoint] Unknown mode: ${MODE}" >&2
    echo "Usage: entrypoint.sh [train|serve|generate|modelling|test]" >&2
    exit 1
    ;;
esac
