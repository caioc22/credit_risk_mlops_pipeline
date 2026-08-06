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
    DATA_PATH=data/sample_credit_data.csv \
    MODEL_OUTPUT_PATH=models/model_v1.rds \
    METADATA_PATH=src/features/metadata.json \
    PORT=8080 \
    LOG_LEVEL=INFO

# OS-level dependencies required to compile the R packages below.
# Note: cmake and zlib1g-dev were removed — they were only needed by the
# Apache Arrow C++ engine (arrow/parquet), which has been replaced by native .rds.
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
