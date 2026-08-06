# Product Requirements Document (PRD): Agibank MLOps Engineer Challenge

**Project Name:** Agibank Credit Risk MLOps Pipeline

**Repository Name:** `agibank-mlops-challenge`

**Target Role:** MLOps Engineer Pleno - Agibank

**Stack:** R (`data.table`, `ranger`, `plumber`, `optparse`), Docker, GitHub Actions, JSON Schema

---

## 1. Executive Summary & Objective

Agibank's Data Science team relies heavily on R for credit risk modeling. As an MLOps Engineer, the goal is to operationalize an experimental R credit risk model (based on the *Home Credit Default Risk* dataset) into a robust, reproducible, unified, and automated production-ready pipeline.

This PRD serves as the authoritative specification for building and maintaining the pipeline architecture.

### Key Architectural Updates & Decisions

1. **Unified Container & Workflow:** Single Docker image and entrypoint that handles both model training/feature building and model serving without dual-image overhead.
2. **Native RDS Feature Store:** Replaced Parquet with native compressed R Data Serialization (`.rds`) format for the Feature Store (`data/feature_store.rds`). This eliminates heavy C++ compilation dependencies (`arrow`/`cmake`), drastically reduces Docker build times, and natively preserves R data types.
3. **Automated Feature Engineering & Cleanup:** `src/data_modelling.R` handles unzipping relational archives, aggregating 1:N tables to customer level (`SK_ID_CURR`), saving the feature store, and cleaning up unzipped CSV files to conserve disk space while preserving original `.zip` archives.
4. **Dynamic Metadata & Model Synchronization:** `src/train.R` reads from `data/feature_store.rds`, trains a Random Forest model (`ranger`), persists `models/model_v1.rds`, and dynamically exports the feature schema definition to `data/features_metadata.json`.

---

## 2. Directory Structure Specification

```text
agibank-mlops-challenge/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD Pipeline
├── src/
│   ├── data_modelling.R           # Unzips, aggregates 1:N tables, generates feature_store.rds & cleans CSVs
│   ├── train.R                    # Consumes feature_store.rds, trains model, writes model_v1.rds & features_metadata.json
│   └── generate_sample_data.R     # Generates fallback sample dataset if zip/CSVs are absent
├── api/
│   ├── plumber.R                  # Plumber REST API endpoints & input validation
│   └── start_api.R                # Entrypoint runner script for Plumber API
├── models/
│   └── model_v1.rds               # Trained model artifact (S3 class object with embedded metadata)
├── data/
│   ├── feature_store.rds          # Consolidated feature store (RDS format)
│   └── features_metadata.json     # Dynamically generated feature schema and imputation rules
├── tests/
│   └── test_api.R                 # API integration test suite
├── Dockerfile                     # Multi-purpose Docker image build specification
├── docker-compose.yml             # Service orchestration for pipeline and API
├── entrypoint.sh                  # Operational modes switcher (train vs serve)
├── .gitignore                     # Git tracking exclusion rules
├── requirements.R                 # R package dependencies installer script
└── README.md                      # Comprehensive project documentation

```

---

## 3. Pillar-by-Pillar Functional Requirements

### Pillar 1: Feature Engineering & Containerized Training

#### System Dependencies & Packages (`requirements.R` / `Dockerfile`)

* Base Image: `rocker/r-ver:4.3.2` or equivalent Ubuntu-based R image.
* OS Libraries: `libcurl4-openssl-dev`, `libssl-dev`, `libxml2-dev`, `libsodium-dev`, `libuv1-dev`, `pkg-config`.
* Core R Packages: `data.table`, `plumber`, `jsonlite`, `logger`, `ranger`, `optparse`, `testthat`, `httr`.

#### Data Modelling & Feature Store Creation (`src/data_modelling.R`)

* **Inputs:** Raw `.zip` archives or CSV files in `data/` (`application_train.csv`, `bureau.csv`, `previous_application.csv`, `POS_CASH_balance.csv`, `installments_payments.csv`, `credit_card_balance.csv`).
* **Processing:**
* Automatically detects and unzips raw archives in `data/`.
* Aggregates relational 1:N tables up to the customer grain (`SK_ID_CURR`).
* Consolidates train and test partitions into a unified structure tagged with `is_train`.


* **Output:** Saves compressed `.rds` feature store to `data/feature_store.rds` (`compress = "gzip"`).
* **Cleanup:** Executes `cleanup_csv_files()` post-processing to delete intermediate extracted CSVs while preserving the source `.zip` files.

#### Model Training (`src/train.R`)

* **Inputs:** `data/feature_store.rds`.
* **Preprocessing:**
* Filters records where `is_train == 1`.
* Dynamically detects predictor features, excluding non-feature columns (`SK_ID_CURR`, `TARGET`, `is_train`).
* Imputes missing values (median for numeric features, mode for categorical features) and tracks imputation mappings.


* **Outputs:**
* Trains a `ranger` probability Random Forest predicting `TARGET`.
* Evaluates out-of-fold metrics (AUC-ROC, Accuracy, LogLoss) and prints structured JSON logs to stdout.
* Overwrites the model artifact at `models/model_v1.rds`.
* Exports schema definition to `data/features_metadata.json`.



---

### Pillar 2: Inference REST API (`plumber`)

#### Path Resolution & Configuration

All paths in `api/plumber.R` resolve relative to working directory or environment variables to prevent path mismatch errors across local and containerized environments:

* `MODEL_OUTPUT_PATH` (default: `models/model_v1.rds`)
* `METADATA_PATH` (default: `data/features_metadata.json`)

#### Endpoints

1. **`GET /health`**
* **Purpose:** Container health probes.
* **Response Body (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2026-08-06T14:30:00Z",
  "model_loaded": true,
  "model_version": "v1"
}

```


* **Response Body (503 Service Unavailable):** Returned if `models/model_v1.rds` fails to load at startup.


2. **`GET /model-info`**
* **Purpose:** Exposes active model information, runtime statistics, MD5 hashes, and feature expectations from `data/features_metadata.json`.


3. **`POST /predict`**
* **Purpose:** Computes credit default probability for single or batch payload records.
* **Input Payload:**
```json
{
  "AMT_INCOME_TOTAL": 150000.0,
  "AMT_CREDIT": 450000.0,
  "AMT_ANNUITY": 25000.0,
  "DAYS_BIRTH": -12000,
  "DAYS_EMPLOYED": -2000,
  "EXT_SOURCE_2": 0.55,
  "EXT_SOURCE_3": 0.42
}

```


* **Response Body (200 OK):**
```json
{
  "prediction_id": "req-9a8b7c6d",
  "timestamp": "2026-08-06T14:30:05Z",
  "default_probability": 0.1425,
  "risk_label": "LOW_RISK",
  "model_version": "v1",
  "processing_time_ms": 12.4
}

```


* **Validation:** Returns `400 Bad Request` with structured error details if required features are missing or types mismatch.


4. **OpenAPI / Swagger UI:** Exposes interactive docs at `/__docs__/`.

---

### Pillar 3: CI/CD Pipeline (`GitHub Actions`)

#### Workflow Steps (`.github/workflows/ci.yml`)

1. **Trigger:** On `push` or `pull_request` to `main`.
2. **Lint & Syntax Check:** Validates R code syntax using `parse()`.
3. **Docker Build & Pipeline Verification:**
* Builds single multi-stage image: `docker build -t agibank-mlops:test .`.
* Executes feature store generation and training inside container:
```bash
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models agibank-mlops:test train

```


* Asserts existence of `data/feature_store.rds`, `models/model_v1.rds`, and `data/features_metadata.json`.


4. **API Integration Testing:**
* Launches API container in background mode on port 8080.
* Runs test suite (`tests/test_api.R`) validating `/health`, `/model-info`, and `/predict`.



---

### Pillar 4: Metadata Management (`data/features_metadata.json`)

#### Schema Format

Generated automatically during model training:

```json
{
  "model_version": "v1",
  "schema_version": "1.0",
  "generated_at": "2026-08-06T14:30:00Z",
  "total_features": 7,
  "features": [
    {
      "name": "AMT_INCOME_TOTAL",
      "type": "numeric",
      "required": true,
      "imputed_default": 147150.0
    },
    {
      "name": "EXT_SOURCE_2",
      "type": "numeric",
      "required": true,
      "imputed_default": 0.565
    }
  ]
}

```

---

## 4. Technical Tradeoffs & Architectural Rationale

* **Why `.rds` over `.parquet`:** The Apache Arrow C++ engine requires significant compilation overhead, heavy OS dependencies (`cmake`, `libssl-dev`, `libxml2-dev`), and custom environment flag overrides (`NOT_CRAN=true`) during Linux container builds. `.rds` is native to R, supports GZIP compression out of the box, introduces zero external binary dependencies, and loads instantly into memory.
* **Why Unified Image Architecture:** Merging training and serving into a single Docker container standardizes library versions, simplifies dependency management, and streamlines CI/CD pipelines.

---

## 5. Definition of Done (Validation Checklist)

* [x] Single container architecture established for training and inference.
* [x] `src/data_modelling.R` unzips raw data, aggregates relational tables, outputs `data/feature_store.rds`, and removes intermediate CSVs.
* [x] Heavy `arrow` dependency replaced with native `.rds` feature store.
* [x] `src/train.R` consumes `data/feature_store.rds`, trains Random Forest (`ranger`), saves `models/model_v1.rds`, and outputs `data/features_metadata.json`.
* [x] Options parsing fixed using `dest` bindings in `optparse` across all scripts.
* [x] `api/plumber.R` relative path issues resolved (`getwd()` alignment for model and metadata loading).
* [x] Rest endpoints (`/health`, `/model-info`, `/predict`) functioning with valid schema validation and structured JSON logs.
* [x] CI/CD workflow configured in `.github/workflows/ci.yml`.
* [x] Documentation (`README.md`) updated with setup steps, pipeline execution, and sample `curl` calls.

---

## 6. Tasks

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 1 | Add 5-fold cross-validation to model training to enhance model performance | `src/train.R` | ✅ Completed |
| 2 | Perform final performance evaluation using the test set (`data/test_set.rds`) | `src/train.R` | ✅ Completed |
| 3 | Apply the same feature-engineering modelling pipeline to `application_test.csv` and generate `data/test_set.rds` for final performance tracking | `src/data_modelling.R` | ✅ Completed |
| 4 | Save final performance tracking results as `models/performance_result.json` | `src/train.R` | ✅ Completed |
| 5 | Run cross-validation folds in parallel (using `parallel::mclapply`) to reduce training time | `src/train.R` | ✅ Completed |