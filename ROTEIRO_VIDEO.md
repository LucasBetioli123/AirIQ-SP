# Roteiro do vídeo (≤ 10 min)

> Gravação de tela (OBS, ou Loom com link não listado). Terminal à esquerda e o
> repositório/figuras à direita. Os números citados estão em `reports/metrics/`.

> **Revisão causal:** use os resultados atuais do RELATORIO, sem repetir os números
> da versão anterior. AOD usa só vizinhanças espaciais; D+1 usa observações até D e
> calendário conhecido, com cobertura própria em `intervalos_d1.csv`. Os dados são
> históricos: informe a referência da demonstração. A gravação ainda está pendente.

Antes de gravar, conclua a instalação e o treino. Uma resposta do LLM pode levar
mais de oito minutos nesta CPU: grave a chamada e a saída, acelere ou corte a espera
e mostre o tempo real registrado no log. Identifique o corte/aceleração no vídeo.
Deixe os mapas, CSVs e uma resposta já concluída disponíveis para a explicação caber
em dez minutos. Não precisa repetir o treinamento inteiro durante a gravação.

| Tempo | Cena | O que mostrar / falar |
|---|---|---|
| 0:00-0:40 | Contexto | O problema do órgão ambiental (poucas estações). Duas peças: estimativa/previsão de PM2.5 com incerteza e o agente de normas. Prioridade: Parte 1 bem validada |
| 0:40-2:00 | Dados | `configs/config.yaml` e a tabela de fontes do RELATORIO. Por que cada troca: QUALAR com login → espelho OpenAQ; MAIAC com login → GOES-16 (8 varreduras/dia, menos falha por nuvem); CDS → ERA5 do NCAR. Rodar `make sample` |
| 2:00-3:30 | Variáveis e validação | `src/aq/features.py`: AOD/BLH, correção higroscópica, ventilação, fogo a favor do vento, preenchimento hierárquico de AOD. Esquema da validação: estação nunca vista + período posterior; lag espacial sem vazamento |
| 3:30-5:00 | Resultados Parte 1 | `benchmark_mae.png` (baselines x modelos); ganho de 31% sobre o IDW a 25-50 km das estações, que é o caso de uso; cobertura do CQR (`intervalos.png`); D+1 x persistência; `faixas_iqar.csv`: 90% dos dias são "Boa", o modelo suaviza episódios e o alerta deve usar o limite superior do IC (detecta ~91% dos dias acima de 25); SHAP (o que o modelo aprendeu faz sentido físico?) |
| 5:00-6:00 | Mapas e interface | `mapa_pm25_*.png` (PM2.5 + incerteza: a incerteza cresce longe das estações). `make predict ...` e `curl localhost:8000/predict?...` |
| 6:00-6:40 | Episódios extremos | `episodios_extremos.csv`: onde o modelo erra (fumaça acima da camada limite, AOD ausente por nuvem/fumaça densa) |
| 6:40-8:40 | Parte 2 | Um chunk real do Anexo I reescrito com o nome da coluna (`notebooks/03_rag.ipynb`) e a revogação parcial da 491 pela 506 (`configs/normas.yaml`). Benchmark de recuperação (`rag_retrieval.csv`: híbrido Granite 0,983 x BM25 0,917; o reranker fica pela recusa). `make chat` ou `make ask` com a pergunta de Campinas: chamada de `estimar_pm25`, citação do Anexo I e comparação com o padrão vigente na data. Uma pergunta sem resposta (recusa pelo limiar). O registro em `logs/agent_runs.jsonl` |
| 8:40-9:40 | Avaliação RAG | Mostrar a linha `qwen3-4b_causal` em `rag_eval_resumo.csv`, as recusas e as checagens de data, horizonte, valor, intervalo e faixa correta. Explicar que 34 respostas normativas foram reaproveitadas e rechecadas, e que as três perguntas ligadas ao modelo foram executadas novamente. Fidelidade numérica não prova toda a semântica; custo em CPU e limitações do conjunto pequeno |
| 9:40-10:00 | Limitações / próximos passos | Período 2021-2023 e sem feed em tempo real; a previsão já usa apenas observações até D, mas a latência da reanálise ERA5 não é simulada. Próximos: meteorologia disponível no horário de emissão, mais estações e avaliação independente do RAG |
