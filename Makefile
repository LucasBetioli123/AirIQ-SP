# Usage: `make help`. Every target uses the venv in .venv (created by `make setup`).
ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python
BOOT_PY ?= python
# prebuilt llama-cpp-python CPU wheel (compiling on Windows needs Visual Studio + CMake)
LLAMA_WHEEL := --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
LLAMA_ENV :=
else
PY ?= .venv/bin/python
BOOT_PY ?= python3
SHELL := /bin/bash
LLAMA_ENV := CMAKE_ARGS="-DGGML_NATIVE=OFF"
endif
export PYTHONPATH := $(CURDIR)
export PYTHONIOENCODING := utf-8

.PHONY: help setup models data sample dataset bench train evaluate maps predict api \
        rag-download rag-ingest rag-bench rag-eval ask chat notebooks test docker all-sample

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## create .venv and install pinned dependencies
	$(BOOT_PY) -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(LLAMA_ENV) $(PY) -m pip install -r requirements-llm.txt $(LLAMA_WHEEL)
	$(PY) -m pip install -r requirements-nn.txt

models: ## download the GGUF models (LLM, embeddings, reranker) from Docker Hub
	$(PY) -m src.rag.models_dl

# ---------------- Module 1: PM2.5 (satellite + ML) ----------------
data: ## download ALL raw data for the configured period (hours; see README)
	$(PY) -m src.aq.data.openaq
	$(PY) -m src.aq.data.era5
	$(PY) -m src.aq.data.goes_aod
	$(PY) -m src.aq.data.goes_fire
	$(PY) -m src.aq.data.s5p_no2
	$(PY) -m src.aq.data.geo
	$(PY) -m src.aq.data.static
	$(PY) -m src.aq.build_dataset

sample: ## copy the training tables from the versioned sample (no download)
	$(PY) -m src.aq.build_dataset --from-sample

dataset: ## build the training tables from already-downloaded raw data
	$(PY) -m src.aq.build_dataset

bench: ## model benchmark (baselines vs. ML) under spatio-temporal validation
	$(PY) -m src.aq.benchmark

train: ## train the final model + conformal intervals + D+1 forecast
	$(PY) -m src.aq.train

evaluate: ## metrics, SHAP, extreme episodes
	$(PY) -m src.aq.evaluate

maps: ## daily PM2.5 and uncertainty map (DATE=yyyy-mm-dd)
	$(PY) -m src.aq.maps $(if $(DATE),--date $(DATE),)

predict: ## point estimate: make predict LAT=-23.50 LON=-47.45 DATE=2021-08-23 [H=1]
	$(PY) -m src.aq.cli --lat $(LAT) --lon $(LON) --date $(DATE) --horizon $(or $(H),0)

api: ## start the FastAPI API on :8000 (GET /predict)
	$(PY) -m uvicorn src.aq.api:app --host 0.0.0.0 --port 8000

# ---------------- Module 2: RAG agent ----------------
rag-download: ## download the regulations from official sources
	$(PY) -m src.rag.download

rag-ingest: ## parse + chunking + indexes (BM25 and dense)
	$(PY) -m src.rag.ingest

rag-bench: ## retrieval benchmark (BM25, dense, hybrid, rerank)
	$(PY) -m eval.bench_retrieval

rag-eval: ## end-to-end agent evaluation (answers, citations, refusals)
	$(PY) -m eval.run_rag_eval

ask: ## ask the agent (in Portuguese): make ask Q="..."
	$(PY) -m src.rag.cli "$(Q)"

chat: ## chat UI (FastAPI + HTML) on :7860
	$(PY) -m uvicorn src.rag.app:app --host 0.0.0.0 --port 7860

# ---------------- Engineering ----------------
notebooks: ## execute the notebooks with visible outputs
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

test: ## automated tests
	$(PY) -m pytest -q

all-sample: ## full pipeline on the sample (also available without make)
	$(PY) -m src.aq.pipeline

docker: ## reproducible Docker image
	docker build -t airiq-sp .
