# Uso: `make help`. Todos os alvos usam o venv em .venv (criado por `make setup`).
ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python
BOOT_PY ?= python
# wheel pronta do llama-cpp-python para CPU (no Windows compilar exige Visual Studio + CMake)
LLAMA_WHEEL := --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
else
PY ?= .venv/bin/python
BOOT_PY ?= python3
SHELL := /bin/bash
endif
export PYTHONPATH := $(CURDIR)
export PYTHONIOENCODING := utf-8

.PHONY: help setup models data sample dataset bench train evaluate maps predict api \
        rag-download rag-ingest rag-bench rag-eval ask chat notebooks test docker all-sample

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## cria .venv e instala dependências com versões fixas
	$(BOOT_PY) -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	CMAKE_ARGS="-DGGML_NATIVE=OFF" $(PY) -m pip install -r requirements-llm.txt $(LLAMA_WHEEL)
	$(PY) -m pip install -r requirements-nn.txt

models: ## baixa os modelos GGUF (LLM, embeddings, reranker) do Docker Hub
	$(PY) -m src.rag.models_dl

# ---------------- Parte 1 ----------------
data: ## baixa TODOS os dados brutos do período do config (horas; ver README)
	$(PY) -m src.aq.data.openaq
	$(PY) -m src.aq.data.era5
	$(PY) -m src.aq.data.goes_aod
	$(PY) -m src.aq.data.goes_fire
	$(PY) -m src.aq.data.s5p_no2
	$(PY) -m src.aq.data.geo
	$(PY) -m src.aq.data.static
	$(PY) -m src.aq.build_dataset

sample: ## copia as bases de treino da amostra versionada (sem download)
	$(PY) -m src.aq.build_dataset --from-sample

dataset: ## monta a base de treino a partir dos brutos já baixados
	$(PY) -m src.aq.build_dataset

bench: ## benchmark de modelos (baselines x ML) na validação espaço-temporal
	$(PY) -m src.aq.benchmark

train: ## treina o modelo final + intervalos conformais + previsão D+1
	$(PY) -m src.aq.train

evaluate: ## métricas, SHAP, episódios extremos
	$(PY) -m src.aq.evaluate

maps: ## mapa diário de PM2.5 e de incerteza (DATE=aaaa-mm-dd)
	$(PY) -m src.aq.maps $(if $(DATE),--date $(DATE),)

predict: ## estimativa pontual: make predict LAT=-22.90 LON=-47.06 DATE=2023-03-10 [H=1]
	$(PY) -m src.aq.cli --lat $(LAT) --lon $(LON) --date $(DATE) --horizon $(or $(H),0)

api: ## sobe a API FastAPI em :8000 (GET /predict)
	$(PY) -m uvicorn src.aq.api:app --host 0.0.0.0 --port 8000

# ---------------- Parte 2 ----------------
rag-download: ## baixa as normas das fontes oficiais
	$(PY) -m src.rag.download

rag-ingest: ## parse + chunking + índices (BM25 e vetorial)
	$(PY) -m src.rag.ingest

rag-bench: ## benchmark de recuperação (BM25, denso, híbrido, rerank)
	$(PY) -m eval.bench_retrieval

rag-eval: ## avaliação ponta a ponta do agente (respostas, citações, recusa)
	$(PY) -m eval.run_rag_eval

ask: ## pergunta ao agente: make ask Q="..."
	$(PY) -m src.rag.cli "$(Q)"

chat: ## interface de chat (FastAPI + HTML) em :7860
	$(PY) -m uvicorn src.rag.app:app --host 0.0.0.0 --port 7860

# ---------------- Engenharia ----------------
notebooks: ## executa os notebooks com saída visível
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

test: ## testes automatizados
	$(PY) -m pytest -q

all-sample: sample bench train evaluate ## pipeline de modelagem da Parte 1 sobre a amostra (+ mapas)
	$(PY) -m src.aq.maps --date 2022-10-11
	$(PY) -m src.aq.maps --date 2021-08-24

docker: ## imagem Docker reprodutível
	docker build -t teste-ia .
