# PM2.5 em São Paulo e agente de normas ambientais

Desafio técnico em duas partes: estimativa diária e previsão D+1 de PM2.5 com intervalo de 90%, e agente RAG que consulta normas e chama o modelo de qualidade do ar.

**Dados históricos de jan/2021 a abr/2023.** Não há feed em tempo real. A previsão é emitida ao fim do dia D, com satélite, meteorologia e medições até D e calendário de D+1. Nenhuma observação de amanhã é usada. No agente, “hoje/amanhã” referem-se à data histórica de demonstração informada na resposta.

- [Relatório técnico, resultados e limitações](RELATORIO.md)
- [Declaração de uso de IA](USO_DE_IA.md)
- [Roteiro da demonstração de até 10 minutos](ROTEIRO_VIDEO.md)
- **Vídeo não listado: gravação pendente; o candidato adicionará o link aqui antes do envio.**

## Instalação e máquina

Python 3.12 ou 3.13, CPU x86-64 com AVX2, sem GPU. Referência: i5-9300H, 12 GB RAM; 16 GB recomendados. A amostra versionada ocupa aproximadamente 12 MB. Os três GGUF padrão ocupam cerca de 4,3 GB. Reserve também espaço para o ambiente Python e os dados completos opcionais (~6 GB de brutos). A primeira instalação requer internet; as consultas locais são gratuitas.

Linux/macOS com compilador C/C++, CMake e GNU Make:

```bash
make setup
make models
cp .env.example .env
```

O `.env` é carregado automaticamente. Variáveis já exportadas no sistema têm precedência. Reserve 10–20 minutos para instalação, dependendo da rede e compilação. Provedores externos são opcionais e exigem configuração pelo usuário.

Windows PowerShell, sem `make`:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -r requirements-llm.txt --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
.venv\Scripts\python -m pip install -r requirements-nn.txt
Copy-Item .env.example .env
.venv\Scripts\python -m src.rag.models_dl
```

## Um comando por parte

Para uma demonstração rápida da Parte 1, com treino e calibração de produção,
previsão de Campinas e um mapa, use `python -m src.aq.pipeline --quick` com o
Python do ambiente (no Windows: `.venv\Scripts\python -m src.aq.pipeline --quick`).
Foi validada em cerca de 2–3 minutos em container limpo nesta revisão e conserva as métricas de teste versionadas; o comando
abaixo refaz também todas as validações e comparações.

A Parte 1 inteira (amostra → benchmark → treino e calibração → avaliação, SHAP e mapas):

```bash
make all-sample
```

No Windows, ou sem `make`:

```powershell
.venv\Scripts\python -m src.aq.pipeline
```

Em Linux/macOS sem make, use `.venv/bin/python -m src.aq.pipeline`. Reserve 30–60 minutos no notebook de referência para o benchmark completo (incluindo redes neurais) e o treino com validação dos intervalos D+1; os logs da revisão estão em `logs/benchmark_causal.txt` e `logs/train_causal.txt`. Para refazer também download e engenharia de variáveis: `python -m src.aq.pipeline --full`, usando o interpretador do ambiente. Alternativa: `make data` e depois `make bench train evaluate maps`. Downloads completos podem levar horas.

A Parte 2 funciona diretamente com os chunks versionados das sete normas:

```bash
make ask Q="Qual o PM2.5 previsto para Campinas amanhã e ele ultrapassa o padrão da CONAMA 491?"
```

Windows:

```powershell
.venv\Scripts\python -m src.rag.cli "Qual o PM2.5 previsto para Campinas amanhã e ele ultrapassa o padrão da CONAMA 491?"
```

Treine a Parte 1 antes de pedir estimativas ao agente. Perguntas exclusivamente normativas só precisam dos GGUF e da base normativa. Reserve 4–12 minutos por pergunta no notebook de referência; prompts maiores ou pressão de memória podem ampliar esse tempo. A primeira consulta pode levar alguns minutos adicionais para indexar os embeddings. `make chat` abre a interface em http://localhost:7860; sem make: `python -m uvicorn src.rag.app:app --port 7860`.

## Outros comandos

| Operação | Comando | Tempo aproximado |
|---|---|---|
| Estimativa pontual | `make predict LAT=-22.90 LON=-47.06 DATE=2022-10-11 H=1` | segundos |
| API HTTP | `make api` → `/predict?municipio=Campinas&date=2022-10-11&horizon=1` | segundos por consulta |
| Download e ingestão das normas | `make rag-download rag-ingest` | depende dos sites públicos |
| Benchmark de recuperação | `make rag-bench` | cerca de 1 h 20 em CPU |
| Avaliação completa do agente | `make rag-eval` | cerca de 3 h em CPU |
| Avaliação com juiz adicional | `python -m eval.run_rag_eval --judge models/llm/gemma3-4b-it-q4_k_m.gguf --tag avaliacao-com-juiz` | superior à avaliação sem juiz |
| Testes | `make test` ou `python -m pytest -q` | cerca de 20 s |
| Notebooks com saída | `make notebooks` | alguns minutos |

Comandos `python` pressupõem o ambiente ativado ou o caminho explícito para seu interpretador. `horizon=1` retorna a previsão para `date + 1`; só a data de referência precisa de observações. As janelas disponíveis da amostra estão em `configs/config.yaml > sample.context_windows`, com dias adicionais de contexto.

Para repetir as comparações de embeddings do relatório, baixe também
`python -m src.rag.models_dl ai/embeddinggemma:300M-Q8_0 ai/qwen3-embedding:0.6b-q8_0`.
O benchmark pula candidatos ausentes. O juiz opcional exige
`python -m src.rag.models_dl ai/gemma3:4B-Q4_K_M` antes do comando com `--judge`.
`make rag-eval` usa o gerador padrão e as checagens automáticas, sem carregar um juiz.

## Docker

```bash
docker build -t teste-ia .
docker run --rm teste-ia python -m pytest -q
docker run --rm -v "$PWD/models:/app/models" teste-ia python -m src.aq.pipeline --quick
docker run --rm -v "$PWD/models:/app/models" -v "$PWD/reports:/app/reports" teste-ia make all-sample
docker run --rm -v "$PWD/models:/app/models" teste-ia make models
docker run --rm -p 7860:7860 -v "$PWD/models:/app/models" teste-ia make chat
```

No PowerShell, use `${PWD}` no lugar de `$PWD` ao compor caminhos com `:`. A imagem inclui a amostra e as dependências; `.dockerignore` exclui ambiente local, dados brutos e GGUF. Para conservar dados processados entre containers, monte também `data/processed`; evite montar uma pasta `data` vazia sobre a amostra incluída. O diretório `models` compartilhado conserva os GGUF e os modelos treinados.

## Base normativa e downloads incompletos

CONAMA 491/2018, 506/2024, 03/1990 e 05/1989; Lei 6.938/1981; Decreto 99.274/1990; Lei 9.605/1998. `configs/normas.yaml` registra URLs, vigência e revogações. As cópias públicas da 491 e 506 reproduzem o DOU e têm sua origem registrada.

Alguns sites podem bloquear downloads. Nesse caso, a ingestão preserva a base existente e usa a amostra completa. Ela nunca sobrescreve `data/sample/rag_chunks.jsonl`. Também é possível salvar o PDF/HTML em `data/raw/normas/<id>.pdf` ou `.html` e repetir a ingestão. Uma base processada incompleta não substitui a amostra na recuperação.

## LLM configurável

| `LLM_PROVIDER` | Configuração |
|---|---|
| `llamacpp` (padrão, gratuito e local) | `LLM_MODEL=models/llm/qwen3-4b-instruct-2507-q4_k_m.gguf` |
| `ollama` (local) | `LLM_MODEL=<modelo instalado>` e `LLM_BASE_URL=http://localhost:11434/v1` |
| `openai_compat` | `LLM_MODEL`, `LLM_BASE_URL` e `LLM_API_KEY` do provedor escolhido |

`RAG_MODE`: `bm25`, `dense`, `hybrid` ou `hybrid_rerank` (padrão). `EMB_MODEL` e `RERANK_MODEL` selecionam os modelos de recuperação. `AGENT_TODAY=2023-04-04` fixa a referência histórica da demonstração. Sem essa variável, usa-se o último dia com observações e contexto.

## Estrutura e evidências

`src/aq/data/`: coletores públicos; `src/aq/`: variáveis, validação, modelos, mapas, CLI e API; `src/rag/`: ingestão, recuperação, LLM, ferramentas e logs; `eval/`: 37 perguntas e avaliação; `reports/metrics/` e `reports/figures/`: resultados; `notebooks/`: evidências executadas; `data/sample/`: amostra; `tests/`: regressões.

Modelos binários e dados grandes não são versionados. O esquema de variáveis é `causal-spatial-aod-v2`; modelos antigos são recusados com instrução de retreino. Resultados anteriores à revisão são identificados como históricos no relatório e nos logs.

## Antes do envio

- Gravar o vídeo não listado de até dez minutos e inserir o link acima.
- O repositório é privado: conceder acesso aos avaliadores quando seus usuários forem informados.
- Enviar o link do repositório por e-mail dentro das 72 horas contadas do recebimento do desafio.
