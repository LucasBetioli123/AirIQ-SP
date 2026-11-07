# Amostra versionada

Gerada por `python -m src.aq.build_dataset --make-sample` a partir dos dados completos.

| Arquivo | Conteúdo |
|---|---|
| `station_days.parquet` | base de ESTIMATIVA: 1 linha por estação x dia (jan/2021-abr/2023), PM2.5 medido + todas as variáveis |
| `station_days_fc.parquet` | base de PREVISÃO D+1 (observações até D + calendário conhecido de D+1, alvo PM2.5 de D+1) |
| `stations.parquet`, `pm25_daily.parquet` | estações CETESB (via OpenAQ) e médias diárias |
| `aod_daily.nc`, `no2_daily.nc`, `era5_daily.nc`, `fires_daily.parquet` | cubos diários (grade de SP) **só nas janelas** de `configs/config.yaml > sample.context_windows` |
| `static_0p01.nc` | uso do solo (WorldCover) e relevo (Copernicus DEM), grade de 0.01° |
| `municipios_sp.csv`, `sp_boundary.wkt` | sedes municipais e contorno do estado (IBGE) |
| `rag_chunks.jsonl` | base normativa já dividida em trechos (para o agente rodar sem baixar as normas) |

Com a amostra, `make all-sample` reproduz benchmark, intervalos, D+1, SHAP e episódios
(os mesmos números de docs/DETALHES_TECNICOS.md) em poucos minutos. Mapas, CLI e API funcionam nas
datas das janelas. Para o período completo: `make data`.

Esquema `causal-spatial-aod-v2`: somente preenchimento espacial do AOD (célula,
3x3, 9x9, ou NaN); não utiliza AOD nem meteorologia observados de amanhã.
A migração das bases legadas é reproduzida por `python -m scripts.migrate_causal_sample`.
`make all-sample` também está disponível como `python -m src.aq.pipeline`, sem GNU Make.
