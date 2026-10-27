# Qualidade do ar em SP (PM2.5) com satélite + meteorologia, e agente RAG de normas ambientais

Solução do desafio técnico em duas partes:

1. **PM2.5 diário em qualquer ponto do estado de SP**: estimativa do dia e previsão D+1, com
   intervalo de confiança de 90%. Usa satélite (AOD GOES-16, NO2 Sentinel-5P, focos de
   queimada GOES-16), meteorologia ERA5, uso do solo e a rede de estações CETESB.
2. **Agente com RAG** sobre normas ambientais (CONAMA 491/2018, 506/2024, 03/1990,
   05/1989, Lei 6.938/1981, Decreto 99.274/1990, Lei 9.605/1998). O agente cita o
   dispositivo exato, recusa quando a base não cobre a pergunta e chama o modelo da
   Parte 1 como ferramenta.

Decisões, resultados com números, limitações e próximos passos estão no
**[RELATORIO.md](RELATORIO.md)**. O uso de IA está declarado em **[USO_DE_IA.md](USO_DE_IA.md)**.

## Requisitos de máquina

- Linux/macOS (ou Docker), Python 3.13, `make`, compilador C/C++ e `cmake` (para o llama.cpp)
- CPU x86-64 com AVX2. **Não precisa de GPU.** Desenvolvido e medido em 4 vCPU / 16 GB RAM
- Disco: ~1 GB para a amostra. Para os dados completos, ~6 GB de brutos e ~12 GB com
  todos os modelos GGUF do benchmark (~4 GB só com os padrões)
- Internet só para baixar dados e modelos. A inferência é 100% local e gratuita

## Instalação

```bash
make setup      # cria .venv, instala requirements.txt (versões fixas) e compila llama-cpp-python (~10 min)
make models     # baixa os GGUF padrão: LLM Qwen3-4B, embeddings Qwen3-0.6B, reranker Qwen3-0.6B (~4 GB)
cp .env.example .env   # opcional: trocar o provedor de LLM (ver "LLM configurável")
```

Ou, com Docker:

```bash
docker build -t teste-ia .
docker run --rm -it -v $PWD/models:/app/models -v $PWD/data:/app/data teste-ia make all-sample
```

## Como rodar (um comando por parte)

| O quê | Comando | Tempo esperado (4 vCPU) |
|---|---|---|
| **Parte 1 com a amostra versionada** (base → treino → métricas → SHAP → mapas) | `make all-sample` | ~10 min |
| Parte 1 completa: baixar todos os dados brutos (2021-01 a 2023-04) | `make data` | ~1 h (rede) |
| Parte 1 completa: benchmark + treino + avaliação + mapas | `make bench train evaluate maps` | ~25 min |
| Estimativa para uma coordenada/data | `make predict LAT=-22.90 LON=-47.06 DATE=2022-09-10 H=1` | ~5 s |
| API HTTP (`GET /predict`) | `make api` | — |
| **Parte 2**: baixar normas → índice | `make rag-download rag-ingest` | ~1 min |
| Pergunta ao agente | `make ask Q="qual o PM2.5 previsto para Campinas amanhã e ele ultrapassa o padrão da CONAMA 491?"` | ~30-90 s em CPU |
| Benchmark de recuperação | `make rag-bench` | ~10 min |
| Avaliação ponta a ponta do agente (≥ 25 perguntas) | `make rag-eval` | ~30-60 min em CPU |
| Testes | `make test` | ~10 s |
| Notebooks de evidência (com saída) | `make notebooks` | ~2 min |

`make help` lista todos os alvos.

## LLM configurável (sempre com opção gratuita)

| `LLM_PROVIDER` | Exemplo | Custo |
|---|---|---|
| `llamacpp` (padrão) | `LLM_MODEL=models/llm/qwen3-4b-instruct-2507-q4_k_m.gguf` | grátis, local (CPU) |
| `ollama` | `LLM_MODEL=qwen3:4b` (`LLM_BASE_URL=http://localhost:11434/v1`) | grátis, local |
| `openai_compat` | Groq (`https://api.groq.com/openai/v1`, `llama-3.1-8b-instant`) ou Gemini (`https://generativelanguage.googleapis.com/v1beta/openai/`) | camadas gratuitas |

Outras variáveis: `RAG_MODE` (`bm25`, `dense`, `hybrid`, `hybrid_rerank`), `EMB_MODEL`,
`RERANK_MODEL` e `AGENT_TODAY` (a data de referência para "hoje/amanhã"; ver limitações).

## Estrutura

```
configs/            config.yaml (recorte, período, split, faixas IQAr), normas.yaml (base de conhecimento)
src/aq/data/        coletores: openaq (CETESB), goes_aod, goes_fire, s5p_no2, era5, static, geo
src/aq/             features, benchmark, train (CQR + D+1), evaluate (SHAP, episódios), maps, predict, cli, api
src/rag/            download, parse, chunk, ingest, embed, retrieve, llm, tools, agent, runlog, cli
eval/               rag_questions.yaml (conjunto de avaliação), bench_retrieval.py, run_rag_eval.py
notebooks/          01_dados, 02_modelo, 03_rag (executados, com saída)
reports/            figures/ e metrics/ (evidências geradas pelo código)
logs/               agent_runs.jsonl (registro de cada execução do agente)
data/sample/        amostra pequena versionada para rodar tudo em minutos
tests/              pytest
```
