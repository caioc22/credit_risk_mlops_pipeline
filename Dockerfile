# ------------------------------------------------------------------------------
# Agibank Credit Risk MLOps Pipeline - Docker image
#
# Multi-purpose image able to run every pipeline mode via entrypoint.sh:
#   docker run agibank-mlops train      # train models/model_v1.rds
#   docker run agibank-mlops serve      # start the Plumber API on :8080
#   docker run agibank-mlops generate   # (re)generate sample data
#   docker run agibank-mlops test       # run integration tests
# ------------------------------------------------------------------------------

FROM rocker/r-ver:4.3.2

ENV DEBIAN_FRONTEND=noninteractive \
    R_REPOS=https://cloud.r-project.org \
    DATA_PATH=data/feature_store.rds \
    FEATURE_STORE_PATH=data/feature_store.rds \
    TEST_SET_PATH=data/test_set.rds \
    RAW_DATA_ZIP=data/home-credit-default-risk.zip \
    MODEL_OUTPUT_PATH=models/model_v1.rds \
    METADATA_PATH=src/features/metadata.json \
    PORT=8080 \
    LOG_LEVEL=INFO

# OS-level dependencies required to compile the R packages below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libsodium-dev \
    libuv1-dev \
    r-base-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install R packages first so dependency layers stay cached across code changes.
COPY requirements.R /app/requirements.R
RUN Rscript /app/requirements.R

WORKDIR /app
COPY . /app

RUN chmod +x /app/entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
CMD ["serve"]
