# Roteiro do vídeo (≤ 10 min)

> Gravação de tela (OBS, ou Loom com link não listado). Terminal à esquerda e o
> repositório/figuras à direita. Os números citados estão em `reports/metrics/`.

| Tempo | Cena | O que mostrar / falar |
|---|---|---|
| 0:00-0:40 | Contexto | O problema do órgão ambiental (poucas estações). Duas peças: estimativa/previsão de PM2.5 com incerteza e o agente de normas. Prioridade: Parte 1 bem validada |
| 0:40-2:00 | Dados | `configs/config.yaml` e a tabela de fontes do RELATORIO. Por que cada troca: QUALAR com login → espelho OpenAQ; MAIAC com login → GOES-16 (8 varreduras/dia, menos falha por nuvem); CDS → ERA5 do NCAR. Rodar `make sample` |
| 2:00-3:30 | Variáveis e validação | `src/aq/features.py`: AOD/BLH, correção higroscópica, ventilação, fogo a favor do vento, preenchimento hierárquico de AOD. Esquema da validação: estação nunca vista + período posterior; lag espacial sem vazamento |
| 3:30-5:00 | Resultados Parte 1 | `benchmark_mae.png` (baselines x modelos), cobertura do CQR (`intervalos.png`), D+1 x persistência, SHAP (o que o modelo aprendeu faz sentido físico?) |
| 5:00-6:00 | Mapas e interface | `mapa_pm25_*.png` (PM2.5 + incerteza: a incerteza cresce longe das estações). `make predict ...` e `curl localhost:8000/predict?...` |
| 6:00-6:40 | Episódios extremos | `episodios_extremos.csv`: onde o modelo erra (fumaça acima da camada limite, AOD ausente por nuvem/fumaça densa) |
| 6:40-8:40 | Parte 2 | Chunking por artigo/inciso (mostrar um chunk), benchmark de recuperação, `make ask` com a pergunta de Campinas (chamada de ferramenta + citação do Anexo I), uma pergunta sem resposta (recusa), o log JSONL |
| 8:40-9:40 | Avaliação RAG e LLMs | `rag_eval_resumo.csv`: acerto, fidelidade, recusa, LLMs comparados (qualidade x latência em CPU) |
| 9:40-10:00 | Limitações / próximos passos | Período 2021-2023 (fonte), *perfect prognosis* no D+1, sem feed em tempo real; próximos: GFS para D+1, MAIAC, mais estações (MonitorAr) |
