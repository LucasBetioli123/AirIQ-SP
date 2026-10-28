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

- Linux/macOS (ou Docker), Python 3.12 ou 3.13, `make`, compilador C/C++ e `cmake` (para o llama.cpp)
- **Windows também funciona** (testado no Windows 10, Python 3.12): o `make setup` usa a wheel
  pronta de CPU do llama-cpp-python, sem compilar. Sem `make`, use os comandos `python -m ...`
  da seção "Sem make" abaixo
- CPU x86-64 com AVX2. **Não precisa de GPU.** Medido em duas máquinas: 4 vCPU / 16 GB RAM
  (nuvem, Linux) e notebook i5-9300H (4 núcleos) / 12 GB RAM (Windows). Os tempos abaixo são do notebook
- Disco: ~1 GB para a amostra. Para os dados completos, ~6 GB de brutos e ~12 GB com
  todos os modelos GGUF do benchmark (~4 GB só com os padrões)
- Internet só para baixar dados e modelos. A inferência é 100% local e gratuita

## Instalação

```bash
make setup      # cria .venv, instala requirements.txt (versões fixas) e compila llama-cpp-python (~10 min)
make models     # baixa os GGUF padrão: LLM Qwen3-4B, embeddings Granite-278M, reranker Qwen3-0.6B (~4,3 GB)
cp .env.example .env   # opcional: trocar o provedor de LLM (ver "LLM configurável")
```

Ou, com Docker:

```bash
docker build -t teste-ia .
docker run --rm -it -v $PWD/models:/app/models -v $PWD/data:/app/data teste-ia make all-sample
```

## Como rodar (um comando por parte)

| O quê | Comando | Tempo esperado (i5-9300H, só CPU) |
|---|---|---|
| **Parte 1 com a amostra versionada** (base → treino → métricas → SHAP → mapas) | `make all-sample` | ~10 min (4 vCPU); só `make sample train`: 7,5 min |
| Parte 1 completa: baixar todos os dados brutos (2021-01 a 2023-04) | `make data` | ~1 h (rede) |
| Parte 1 completa: benchmark + treino + avaliação + mapas | `make bench train evaluate maps` | ~25 min |
| Estimativa para uma coordenada/data | `make predict LAT=-22.90 LON=-47.06 DATE=2022-09-10 H=1` | ~5 s |
| API HTTP (`GET /predict`) | `make api` | — |
| **Parte 2**: baixar normas → índice | `make rag-download rag-ingest` | ~1 min (ingestão: 3 s; 1ª consulta indexa os vetores: ~5 min) |
| Pergunta ao agente | `make ask Q="qual o PM2.5 previsto para Campinas amanhã e ele ultrapassa o padrão da CONAMA 491?"` | 4-8 min em CPU (o prompt do LLM sai a ~16 tokens/s) |
| Chat no navegador | `make chat` → http://localhost:7860 | idem por pergunta |
| Benchmark de recuperação (3 embeddings + reranker) | `make rag-bench` | ~1 h 20 (35 min de indexação + 38 min de reranker) |
| Avaliação ponta a ponta do agente (37 perguntas + juiz) | `make rag-eval` | ~3 h em CPU com Qwen3-4B (com GPU ou API, minutos) |
| Testes | `make test` | ~10 s |
| Notebooks de evidência (com saída) | `make notebooks` | ~2 min |

`make help` lista todos os alvos.

### Normas que o script não consegue baixar

`make rag-download` tenta as URLs de `configs/normas.yaml` e valida o conteúdo. Dois sites
recusam clientes automatizados: o **DOU (in.gov.br)** responde 403 e o **Planalto** derruba a
conexão (em algumas redes). Por isso:

- a 491/2018 e a 506/2024 vêm de cópias públicas da página do DOU (FAOLEX/FAO e Prefeitura do
  Rio), com a URL registrada em `data/raw/normas/<id>.url`;
- se uma URL falhar, o script avisa `FALHOU (coloque o arquivo manualmente em data/raw/normas/)`.
  Basta abrir a URL de `configs/normas.yaml` no navegador e salvar como
  `data/raw/normas/<id>.html` (ou `.pdf`); o `make rag-ingest` usa o arquivo local;
- **não é preciso baixar nada para rodar o agente**: os chunks das 7 normas estão versionados
  em `data/sample/rag_chunks.jsonl` e são usados quando `data/processed/rag/` não existe.

### Sem make (Windows)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -r requirements-llm.txt --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
$env:PYTHONPATH="."; $env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python -m src.rag.models_dl                  # make models
.venv\Scripts\python -m src.aq.build_dataset --from-sample # make sample
.venv\Scripts\python -m src.aq.train                       # make train (recria models/estimativa.joblib)
.venv\Scripts\python -m src.rag.cli "qual o PM2.5 previsto para Campinas amanhã e ele ultrapassa o padrão da CONAMA 491?"
.venv\Scripts\python -m uvicorn src.rag.app:app --port 7860 # make chat -> http://localhost:7860
```

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
