# Relatório técnico

> Todos os números deste relatório vêm de arquivos em `reports/metrics/`, gerados pelo
> código do repositório (`make bench train evaluate rag-bench rag-eval`).

## 0. Resumo

<!--RESUMO-->

## 1. Premissas e restrições

- **Recorte**: estado de São Paulo, grade de 0.05° (~5,5 km), dados diários.
- **Período: jan/2021 a abr/2023**, e não "2019 em diante". O PM2.5 medido (a variável
  alvo) veio do único espelho público e anônimo da rede CETESB que encontrei, o arquivo
  da OpenAQ no AWS. Para as estações de SP ele cobre 2017 e de jan/2021 a abr/2023,
  com ~50% dos dias (lacunas de ingestão da própria OpenAQ). O QUALAR exige login e
  não estava acessível no ambiente de desenvolvimento.
- **Ambiente**: o container de desenvolvimento só tinha acesso a PyPI, GitHub e buckets
  públicos (AWS/GCS/Docker Hub). Sites gov.br, Open-Meteo, CDS, INPE, Planetary
  Computer e HuggingFace estavam bloqueados. Isso pesou em todas as escolhas de fonte
  abaixo. Para não depender de cadastro, toda fonte escolhida é anônima.
- **Custo zero**: modelos abertos rodando em CPU (llama.cpp). Nenhuma chave paga.
- Sem resposta a tempo para dúvidas, segui com as premissas acima.

## 2. Dados (Parte 1)

| Dado | Fonte usada | Sugerida | Por que a troca |
|---|---|---|---|
| PM2.5 medido | **CETESB via arquivo OpenAQ (S3)**, horário → média de 24 h com ≥ 18 h válidas | QUALAR / MonitorAr | QUALAR exige login e limita a 1 estação x 1 parâmetro x 1 ano por consulta; MonitorAr inacessível |
| AOD | **GOES-16 ABI L2 AODF** (2 km, 10 min) | MAIAC MCD19A2 | MAIAC exige Earthdata/GEE. O GOES é geoestacionário: com 8 varreduras por dia (09-16 h local), uma única hora sem nuvem já dá AOD no dia |
| NO2 troposférico | **Sentinel-5P TROPOMI** (COG do MEEO no S3), qa ≥ 0.75 | igual | mesma fonte, em formato que permite leitura parcial |
| Queimadas | **GOES-16 ABI L2 FDCF** (máscara de fogo + FRP) | INPE BDQueimadas | INPE inacessível. O GOES-16 é um dos satélites do próprio BDQueimadas |
| Meteorologia | **ERA5 (espelho NCAR no S3)**: BLH, T, Td→UR, vento, pressão, nuvens, chuva | ERA5 (CDS) | mesmo dado. O CDS exige chave |
| Uso do solo / relevo | **ESA WorldCover 2021** (10 m) + **Copernicus DEM** (30 m) | MapBiomas / IBGE | o MapBiomas é um GeoTIFF nacional de ~1 GB por ano; para frações de uso a 1 km o WorldCover é equivalente e vem em blocos |
| Limites / municípios | malha IBGE (via geodata-br) + sedes municipais IBGE | IBGE | gov.br inacessível; mesmos dados IBGE espelhados no GitHub |

**Engenharia de leitura**: nenhum arquivo é baixado inteiro. NetCDF/HDF5 (GOES, ERA5) e
COG (S5P, WorldCover, DEM) são lidos por *HTTP range requests*, só a janela de SP. O
GOES lê ~2 MB de 38 MB por varredura; o ERA5, ~2 MB de ~1,4 GB por variável-mês. Os
coletores usam processos, não threads, porque o HDF5 não é *thread-safe* (tive
*segfault* com threads).

### 2.1 Casamento estação ↔ pixel e falhas de AOD

- **Tempo**: tudo agregado no dia **local** (UTC-3). O registro horário da OpenAQ marca o
  fim da hora, então a hora 00:00 pertence ao dia anterior.
- **Espaço**: AOD/NO2 na célula de 0.05° que contém o ponto, mais médias 3x3 e 9x9;
  ERA5 por interpolação bilinear; estáticos em 1 km e em janela de 5 km.
- **Falhas de AOD por nuvem**, em camadas: (0) composição das 8 varreduras do dia →
  (1) janela 3x3 → (2) janela 9x9 → (3) média de D-1/D+1 (só D-1 na previsão, para não
  olhar o futuro) → (4) sem AOD: o NaN fica para o LightGBM, que aprende um ramo próprio,
  e a flag `aod_fill_level` vira variável.

### 2.2 Variáveis com justificativa física

| Variável | Física |
|---|---|
| `aod_blh = AOD / BLH` | o AOD integra a coluna. Dividir pela altura de mistura dá um proxy da extinção (≈ concentração) junto à superfície |
| `aod_dry = AOD·(1-UR)^0.6` | crescimento higroscópico: com UR alta a água infla as partículas e o AOD sobe sem aumento da massa seca que o monitor mede |
| `vent = BLH·vento` (log) | índice de ventilação, a capacidade de diluição da atmosfera |
| `fire_up_{150,400}km` | focos a favor do vento: soma de `max(0, cos(rumo_ponto→foco − direção_de_onde_vem_o_vento))·log(1+FRP)/dist` (+ D-1, pelo tempo de transporte) |
| `prev_tp`, `tp` | remoção úmida (chuva de hoje e de ontem) |
| `blh_min` | estabilidade noturna (inversões) |
| `no2` | traçador de combustão (tráfego/indústria) e precursor de nitrato |
| `built_5km`, `tree_5km`, `elev` | contexto urbano/vegetação/altitude |
| `idw` (lag espacial) | PM2.5 das **outras** estações no mesmo dia (sinal regional). Na validação, as estações de teste nunca entram como referência |

<!--RESULTADOS_P1-->

<!--P2-->

<!--FIM-->
