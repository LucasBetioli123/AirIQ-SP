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

### 2.3 Base de treino

- 30 estações CETESB com PM2.5 e 11.543 estação-dias (2021-01-13 a 2023-04-04), média de 24 h com ≥ 18 h válidas.
- AOD observado na célula do pixel em 50,8% das estação-dias. Com o preenchimento hierárquico chega a 89,3%: 11% pela janela 3x3, 12% pela 9x9, 15% pela média temporal, e 10,7% ficam sem AOD.
- Correlação de Spearman com o PM2.5: `fire_up_400km` +0,48, `no2_filled` +0,46, chuva de ontem −0,52, UR −0,41.
  O **AOD sozinho quase não correlaciona (+0,04)**.
  - Investiguei se era defeito do dado. Refiz um subconjunto só com pixels de alta qualidade (DQF = 0): ρ = 0,14.
  - Conferi a geolocalização no mapa. As lacunas dos rios e represas batem com o contorno de SP, e o NO2 concentra na RMSP.
  - O dado está certo. O sinal do AOD do GOES para o PM2.5 de superfície em SP é fraco, porque a coluna é dominada por umidade e aerossol em altitude. Por isso o AOD entra **combinado** com camada limite e UR.

## 3. Validação (o que simula o uso real)

- **Espaço**: GroupKFold por estação (5 folds). A estação de teste nunca entra no treino nem como referência do IDW.
- **Tempo**: origem móvel com dois cortes, testando sempre **depois** do treino:
  - **T1**: treino até mai/2021, calibração em jun/2021, **teste de jul a dez/2021**. É a estação seca, com os episódios de fumaça de ago–set/2021.
  - **T2**: treino até mar/2022, calibração de abr a jun/2022, **teste de jul/2022 a abr/2023**.
- **Por que dois cortes**: no começo usei só o T2. Mas o arquivo OpenAQ tem só 243 estação-dias em jul–set/2022, então o teste ficou quase todo na estação chuvosa (média 10,7 e máximo 41 µg/m³). Os modelos pareciam bons, mas não eram testados no cenário difícil. O T1 corrige isso.
- **A validação aleatória fica só como referência**. Ela daria R² 0,84 contra 0,72 na validação correta, porque o modelo "vê" a mesma estação e o mesmo dia pelas vizinhas.
- Todas as métricas são calculadas nas **mesmas linhas** (7.761 estação-dias de teste) para todos os modelos.

## 4. Resultados da Parte 1

### 4.1 Estimativa do dia em local sem estação (`reports/metrics/benchmark_estimativa.csv`)

| Modelo | MAE | RMSE | R² | Ganho MAE vs IDW | Acerto faixa IQAr |
|---|---|---|---|---|---|
| **LightGBM regularizado, alvo relativo ao IDW (escolhido)** | **3,00** | **4,77** | **0,722** | **+3,4%** | 94,4% |
| Ensemble RF + LightGBM reg. (offset) | 3,01 | 4,77 | 0,721 | +3,3% | 94,2% |
| Krigagem ordinária (baseline) | 3,08 | 4,91 | 0,705 | +1,1% | 94,0% |
| XGBoost reg. forte (offset) | 3,09 | 4,86 | 0,711 | +0,9% | 94,1% |
| **IDW entre estações vizinhas (baseline)** | 3,11 | 4,87 | 0,710 | 0 | 94,3% |
| Random Forest (offset) | 3,13 | 4,90 | 0,706 | −0,6% | 94,0% |
| LightGBM (offset) **sem satélite** | 3,48 | 5,46 | 0,636 | −11,8% | 93,0% |
| MLP PyTorch 64-32 com saída tanh (offset) | 3,71 | 5,88 | 0,576 | −19,1% | — |
| LightGBM com alvo log, sem offset | 3,72 | 6,12 | 0,541 | −19,7% | 92,0% |
| LightGBM **sem lag espacial** (só satélite + meteorologia) | 4,40 | 7,01 | 0,399 | −41,4% | 91,1% |
| Climatologia mensal | 6,26 | 9,34 | −0,07 | −101% | 90,2% |
| *Validação aleatória (otimista, só referência)* | *2,15* | *3,39* | *0,836* | | |

**Recortes** (`benchmark_estimativa_recortes.csv`), MAE em µg/m³:

| | IDW | Krigagem | **LightGBM reg. offset** | XGBoost forte | MLP tanh |
|---|---|---|---|---|---|
| T1 (estação seca 2021) | 3,62 | 3,65 | 3,65 | 3,77 | 4,90 |
| T2 (jul/22–abr/23) | 2,54 | 2,43 | **2,29** | 2,32 | 2,36 |
| Estação de referência < 10 km (17 est.) | 2,89 | 3,00 | 2,96 | 3,05 | 3,76 |
| 25–50 km (6 est.) | 3,51 | 2,99 | **2,45** | 2,48 | 3,49 |
| > 50 km (1 est.) | 4,07 | 4,61 | 4,26 | 4,27 | 4,54 |

**Leitura honesta dos números:**
- **O baseline é forte.** Na RMSP a rede é densa: 17 das 30 estações têm outra a menos de 10 km. Ali o IDW do mesmo dia já explica 71% da variância.
  - O ganho do modelo aparece quando a estação mais próxima está a 25–50 km: MAE 2,45 contra 3,51, **−30%**.
  - Também aparece no período T2: 2,29 contra 2,54, **−10%**.
- **No T1, nenhum modelo de ML bate o IDW.** O treino tem só 5 meses (jan–mai), sem estação seca. Os modelos com alvo absoluto tiveram viés de −2,4 µg/m³ e MAE 4,8.
- **A decisão que mais rendeu foi modelar o PM2.5 relativo ao IDW** (`log1p(PM) − log1p(IDW)`). O nível regional do dia vem das medições, e o modelo aprende só a correção local (urbanização, relevo, NO2, fogo, meteorologia). Com isso o MAE caiu de 3,72 para 3,00. Árvores não extrapolam nível, e o offset resolve isso.
- **O satélite agrega**: sem AOD/NO2, o MAE sobe de 3,24 para 3,48 na mesma configuração com offset.
- **O lag espacial é essencial**: só com satélite e meteorologia o MAE vai a 4,40. As estações carregam o sinal do dia.

**Por que LightGBM e não os outros** (cada família teve 2 a 3 configurações escolhidas *a priori*, sem otimização no teste):
- **XGBoost**: só bate o IDW na versão rasa e muito regularizada (profundidade 3, `min_child_weight` 60), com MAE 3,09.
- **LightGBM regularizado**: `num_leaves` 15, `min_child_samples` 60, `extra_trees` (cortes aleatórios) e `colsample` 0,5. É a regularização mais forte e a que melhor generaliza para estação nova.
- **Rede neural (PyTorch)**: MLP 64-32 com tanh, **saída tanh e alvo normalizado em [−0,7; +0,7]**, para trabalhar na parte quase linear da tanh. A mesma saída limita a previsão à faixa do treino.
  - Com saída linear e StandardScaler, a MLP divergia (MAE 9,7).
  - Com QuantileTransformer: 6,1. Com saída tanh: 5,4–6,2. Com dropout 0,1 + weight decay 1e-2: **3,71**.
  - No T2 ela bate o IDW (2,36 contra 2,54) e quase empata com o LightGBM (2,29). No T1, que tem mudança de regime, falha (4,90).
  - Com ~6 mil linhas e 30 locais, os GBMs são mais robustos. A rede seria candidata com mais anos de dados.
- **Ridge**: mesmo com QuantileTransformer, extrapola nos episódios (MAE 5,2).

### 4.2 Intervalo de confiança de 90% (`intervalos.csv`)

| Método | Cobertura real (nominal 90%) | Largura média | Cobertura quando PM > 25 |
|---|---|---|---|
| Quantis LightGBM 5–95% sem calibrar | 79,5% | 9,6 | 75,0% |
| **CQR (quantis conformalizados)** | **90,5%** (T1 90,2%, T2 90,9%) | 13,5 | **84,7%** |
| Conformal do resíduo absoluto | 92,0% | 14,7 | 56,5% |

- **Por que CQR**:
  - Os quantis puros são estreitos demais em estação nunca vista: cobrem 79,5% em vez de 90%.
  - O CQR corrige a cobertura e mantém o intervalo **adaptativo**, que é mais largo onde o PM é alto.
  - O conformal do resíduo absoluto tem a mesma largura em todo lugar. Ele cobre bem os dias limpos e mal os poluídos (56,5% acima de 25 µg/m³).
- A calibração usa previsões **fora da estação**: cross-fitting por grupos de estação no período de calibração. Assim, os escores refletem o erro num local sem monitor.
- Cobertura condicional por faixa de PM (`intervalos_por_faixa.csv`): 91,8% em 0–15, 90,3% em 15–25, 85% em 25–50 e 80,8% acima de 50. Nos extremos o intervalo ainda fica curto (ver 4.5).

### 4.3 Previsão do dia seguinte (`previsao_d1.csv`)

| Cenário | Persistência | **Modelo D+1** | Ganho |
|---|---|---|---|
| A) local **com** estação (sabe o PM de hoje) | MAE 4,30 / R² 0,49 | **3,40 / 0,67** | **−21%** |
| B) local **sem** estação ("persistência" = IDW de hoje) | MAE 4,92 / R² 0,37 | **3,98 / 0,55** | **−19%** |

O ganho vale nos dois cortes (A: −19% no T1 e −25% no T2). Para D+1, a meteorologia vem do ERA5 do próprio dia D+1 (*perfect prognosis*). Na operação ela viria de uma previsão numérica (GFS/WRF), com algum erro a mais; isso está nas limitações.

### 4.4 Explicação (SHAP, `shap_resumo.png`, `shap_dependencia.png`)

O modelo explica a **correção** em relação ao IDW. Principais efeitos, todos com sinal físico coerente:
- **IDW alto puxa a correção para baixo**: é encolhimento para a média. O nível do dia já vem do IDW.
- **Área construída e NO2 altos aumentam**: fontes locais (tráfego, indústria).
- **Árvores, campo e altitude diminuem.**
- **Chuva de ontem diminui** (remoção úmida). **UR alta diminui.**
- **`blh_min` alto diminui**: noite menos estável dispersa mais.
- **Fogo a favor do vento em D−1 aumenta**: transporte de fumaça com defasagem.
- **Exemplo local** (`shap_exemplo_local.csv`): Osasco, 24/08/2021, 85 µg/m³ medido. Os fatores são área construída alta, NO2 alto, fogo a favor do vento em D−1 e nenhuma chuva.

### 4.5 Episódios extremos (`episodios_extremos.csv`, `piores_dias.csv`)

| Regime | n | PM médio | MAE IDW | MAE modelo | Viés modelo |
|---|---|---|---|---|---|
| Normal | 7.286 | 11,9 | 2,74 | **2,52** | −0,5 |
| Fumaça (fogo a favor do vento ≥ p95) | 393 | 34,3 | **8,03** | 8,85 | −3,7 |
| PM alto sem fogo (≥ p95) | 179 | 38,9 | **7,33** | 9,28 | −7,8 |

**Onde o modelo falha e por quê.** Nos episódios ele **subestima** e perde para o IDW:
1. As árvores encolhem para a média do treino. No T1, o treino não tinha nenhum dia de estação seca.
2. Na fumaça densa, o AOD muitas vezes falta (é marcado como nuvem) ou satura.
3. A fumaça transportada viaja **acima** da camada limite, então AOD alto não implica PM alto na superfície. Quando ela desce, o modelo não tem como saber.
4. Os piores dias (setembro/2021) têm subestimação média de 4–7 µg/m³.

Mitigações possíveis (próximos passos): perda quantílica/assimétrica para os altos, peso maior nos episódios, variável de altura da pluma (CAMS) e mais anos de dados secos no treino.

### 4.6 Mapas e interface

- `mapa_pm25_<data>_h0.png`: estimativa e incerteza (largura do IC 90%) na grade de 0,05° de SP, com as estações medidas sobrepostas.
  - Dia típico de teste: 2022-11-16, 100% do estado na faixa "Boa", IC médio de 8,8.
  - Pior dia do teste: 2022-10-11.
  - Episódio de fumaça: 2021-08-24, 93% "Moderada", 2,4% "Ruim", IC médio de 41,9.
- A **incerteza cresce longe das estações e nos dias de episódio**, como esperado.
- `mapa_pm25_2021-08-24_h1.png`: previsão D+1 do mesmo dia.
- CLI: `make predict LAT=-22.90 LON=-47.06 DATE=2022-10-11 H=1`. API: `GET /predict?municipio=Campinas&date=...&horizon=1`.
  - Devolvem PM2.5, IC 90%, faixa do IQAr, padrão legal vigente e se ele é ultrapassado, além de um diagnóstico (AOD disponível? quantas estações no IDW? distância da mais próxima).
- **Modelos finais** (`models/meta.json`): treinados com **todos os dados** (11.543 estação-dias, 30 estações). Os folds serviram só para as métricas.

<!--RESULTADOS_P1-->

## 5. Parte 2: agente com RAG sobre normas ambientais

### 5.1 Base de conhecimento (`configs/normas.yaml`)

| Documento | Por que está na base | Fonte do texto | Chunks |
|---|---|---|---|
| Res. CONAMA 491/2018 | padrões de qualidade do ar (pedida no desafio); Anexos III e IV ainda vigentes | impressão da página do DOU guardada na FAOLEX (FAO) | 35 |
| Res. CONAMA 506/2024 | padrões **vigentes hoje**, que revogaram parte da 491 | impressão da página do DOU publicada pela Prefeitura do Rio | 31 |
| Res. CONAMA 03/1990 | padrões até 2018 (perguntas "o que valia em 2015") | conama.mma.gov.br | 14 |
| Res. CONAMA 05/1989 | PRONAR | conama.mma.gov.br | 18 |
| Lei 6.938/1981 | Política Nacional do Meio Ambiente (pedida no desafio); competência do CONAMA | Planalto (texto compilado) | 183 |
| Decreto 99.274/1990 | regulamenta a 6.938 (CONAMA, licenças LP/LI/LO) | Planalto | 71 |
| Lei 9.605/1998 | crimes ambientais (art. 54: poluição) | Planalto | 100 |

- **Premissas sobre as fontes.**
  - O DOU (in.gov.br) responde 403 a qualquer acesso automatizado. Para a 491 e a 506, usei cópias públicas da própria página do DOU (o PDF traz o cabeçalho "Imprensa Nacional" e a URL original). A URL de cada arquivo fica em `data/raw/normas/<id>.url`.
  - O Planalto derruba a conexão de clientes que não se identificam como navegador, pelo menos na rede onde rodei. O script avisa, e o arquivo pode ser salvo à mão pelo navegador (README). Os chunks das 7 normas estão versionados em `data/sample/rag_chunks.jsonl`, então o agente roda sem baixar nada.
- **A 05/1989 foi revogada pela Res. CONAMA 513/2026** (DOU de 09/04/2026), conforme a nota de correlação no próprio PDF do CONAMA. O texto da 513 só está no DOU, então ficou fora da base. A 05/1989 entra como norma revogada (útil para perguntas históricas), e a pergunta sobre a 513 faz parte do conjunto de recusa.

### 5.2 Divisão em trechos (`src/rag/parse.py`, `src/rag/chunk.py`)

O texto normativo tem hierarquia explícita (LC 95/1998): artigo > parágrafo > inciso > alínea, além dos anexos. A divisão segue essa estrutura e não um tamanho fixo:

- **Unidade = artigo inteiro** (caput + §§ + incisos). Um inciso solto ("II - 60 µg/m³") não diz nada sem o caput. Um artigo com mais de 320 palavras é dividido em blocos de §§/incisos, e **cada bloco repete o caput**.
- **Cada chunk carrega o caminho citável** (documento › capítulo › seção › artigo) e a lista de dispositivos que contém ("art. 4º, § 2º", "art. 2º, inciso III"). É o que permite citar o dispositivo exato.
- **Tabelas dos anexos: uma linha por chunk, com o nome da coluna ao lado de cada valor.** O PDF perde a grade: a linha sai como "Material Particulado - MP2,5 24 horas 60 50 37 25 15 -", e um LLM de 4B erra a coluna com facilidade. Reescrevo como "MP2,5 | 24 horas | PI-1: 60 µg/m³ | PI-2: 50 µg/m³ | … | PF: 15 µg/m³", repetindo o título e as notas. O rótulo da linha vira a citação ("Anexo I (Material Particulado - MP2,5, 24 horas)"). As células vazias somem no PDF e são alinhadas à direita (o "9" do CO cai na coluna PF em ppm).
- **Resoluções sem artigos** (a 05/1989 usa itens "2.2.1 -"): o item numerado é a unidade citável.
- **Dispositivo riscado no Planalto** (redação revogada) fica fora do índice.

Problemas que só apareceram com os textos reais, e o que fiz:

| Problema | Correção |
|---|---|
| cabeçalho/rodapé de impressão do DOU ("4/20/23, 9:46 AM …", URL, "3/6") no meio do texto | filtro de linhas de ruído |
| ligaduras (ﬁ), "μ g/m ᶟ", subscritos (MP₂,₅) | normalização pontual (NFKC trocaria "º" por "o") |
| HTML do Planalto malformado (`<p></font><font>texto`): o inciso II do art. 6º da 6.938 (o que define o CONAMA) sumia | o texto fora de qualquer bloco também vira parágrafo, na ordem do documento |
| "Art. 1º - Esta Lei" virava "art. 1º-E"; "Art. 9 o -A." não era reconhecido | a letra só é sufixo se vier seguida de ponto ou travessão |

**Descartado:** chunk de tamanho fixo (corta inciso do caput), um chunk por inciso (perde o caput), tabela como texto corrido (troca de coluna) e a tabela inteira num chunk só (passa do contexto do modelo de embedding).

### 5.3 Vigência das normas (diferencial)

- Cada chunk tem `vigencia_inicio`/`vigencia_fim`. **A revogação é por dispositivo**: a 506/2024 (art. 11, I) revogou os arts. 1º a 8º, 12 a 14 e o Anexo I da 491/2018, mas **os arts. 9º a 11 e os Anexos II a IV continuam valendo** (os níveis de atenção/alerta/emergência do Anexo III ainda são da 491). O mesmo vale para os itens 2.2.1 e 2.3 da 05/1989, revogados pela 491.
- A busca filtra pelo que estava vigente na data da pergunta: hoje, ou a data que o plano extrai ("em 2015" → 2015). Uma norma citada pelo número ("CONAMA 491", "03/1990") entra mesmo se revogada, porque a pergunta é sobre ela.
- Trecho revogado vai para o LLM com a nota "REVOGADO: vigente de … até …; revogado por …". É isso que responde "a 03/1990 ainda está em vigor?" sem inventar.
- A ferramenta de PM2.5 compara com o padrão vigente **na data-alvo** (`src/aq/aqi.py`): PI-1 de 60 µg/m³ da 491 até 08/07/2024, PI-1 da 506 até 31/12/2024 e PI-2 de 50 µg/m³ a partir de 01/01/2025.

### 5.4 Recuperação (`src/rag/retrieve.py`, `reports/metrics/rag_retrieval.csv`)

Método: as 31 perguntas do conjunto de avaliação que têm trecho-ouro. O **recall@k** é a fração dos trechos-ouro que aparecem no top-k, em média por pergunta; o **MRR** é a média de 1/posição do primeiro trecho-ouro. O filtro de vigência fica ligado em todos os casos, como no uso real. A latência é por consulta, em CPU (i5-9300H).

| Configuração | recall@1 | recall@3 | **recall@5** | hit@5 | MRR | latência |
|---|---|---|---|---|---|---|
| BM25 (stem PT + sinônimos de poluentes) | 0,617 | 0,850 | 0,917 | 0,933 | 0,797 | 8 ms |
| Vetorial EmbeddingGemma-300M | 0,683 | 0,850 | 0,883 | 0,933 | 0,838 | 162 ms |
| Vetorial Granite-multilingual-278M | 0,583 | 0,750 | 0,900 | 0,933 | 0,764 | 116 ms |
| Vetorial Qwen3-Embedding-0.6B | 0,500 | 0,733 | 0,867 | 0,900 | 0,662 | 1.361 ms |
| Híbrido RRF BM25 + EmbeddingGemma | 0,650 | 0,850 | 0,950 | 0,967 | 0,832 | 169 ms |
| **Híbrido RRF BM25 + Granite (escolhido)** | **0,717** | 0,833 | **0,983** | **1,000** | **0,864** | 139 ms |
| Híbrido RRF BM25 + Qwen3-Embedding | 0,600 | 0,833 | 0,917 | 0,933 | 0,782 | 1.410 ms |
| *Ablação: o escolhido sem limite de 2 trechos por dispositivo* | 0,717 | 0,817 | 0,933 | 1,000 | 0,864 | 133 ms |
| Híbrido Granite + reranker Qwen3-Reranker-0.6B (top 8) | 0,700 | 0,867 | 0,950 | 1,000 | 0,864 | 61,5 s |

- **Por que híbrido.** O texto normativo mistura termos exatos ("MP2,5", "PI-1", "art. 14", números de resolução), em que o BM25 é forte, com paráfrases ("quem fiscaliza", "poluição que tira os moradores"), em que o vetorial é forte. O RRF combina os dois rankings sem calibrar escalas. Cada um sozinho chega a 0,90–0,92; juntos, 0,983.
- **Por que Granite.** É o melhor no híbrido, o mais rápido e o menor. Indexar os 452 trechos levou 5,3 min, contra 6,6 min do EmbeddingGemma e ~25 min do Qwen3-Embedding, que era o padrão inicial. O Qwen3-Embedding ficou pior **e** 10x mais lento.
- **Duas correções que vieram da análise dos erros**, e não de ajuste de parâmetro:
  1. "segundo a CONAMA 491" não era reconhecido como menção à norma (o padrão exigia "491/2018"), e o filtro de vigência descartava os artigos revogados da 491. Agora a menção sem ano conta.
  2. As 15 linhas do Anexo I repetem título e cabeçalho e ocupavam o top-5 inteiro, e o segundo trecho das perguntas compostas ficava de fora. Agora entram no máximo 2 trechos do mesmo dispositivo no topo. A ablação mostra o efeito: 0,933 → 0,983.
  - Com as duas, o BM25 subiu de 0,850 para 0,917 e o híbrido de 0,867 para 0,983.
- **O reranker não melhora a recuperação, mas fica pela recusa.** O recall@5 cai de 0,983 para 0,950, e cada reordenação custa ~60 s em CPU (o GGUF é F16, 7,7 s por trecho). Em compensação, o escore dele separa o que a base não cobre (`rag_limiar_recusa.csv`): as 31 perguntas respondíveis têm escore ≥ 0,93, e 5 das 6 sem resposta ficam abaixo de 0,65. Com limiar 0,8, o agente recusa essas 5 **sem chamar o LLM** e sem nenhuma recusa indevida. O cosseno do Granite separa pior: o limiar de 0,675 só pega 2 das 6. A exceção é a pergunta sobre a validade da licença de operação (0,99), porque o art. 19 do Decreto 99.274 define a LO, só não o prazo. Essa recusa fica a cargo do LLM.
  - **Ressalva honesta:** o limiar foi calibrado no próprio conjunto de avaliação; com mais perguntas, ele deveria ser escolhido num conjunto separado.
- O agente roda o reranker **uma vez**, sobre a união dos candidatos das buscas do plano e contra a pergunta original. Reordenar a cada busca multiplicaria o custo pelo número de buscas.

### 5.5 Agente (`src/rag/agent.py`)

Poucas chamadas ao LLM, porque em CPU cada uma custa minutos:

1. **Planejar** (LLM, saída JSON restrita por gramática): até 3 buscas curtas na base, as consultas de PM2.5 (local, data, horizonte) e, se a pergunta for sobre uma data passada, a data da norma.
2. **Executar as ferramentas.**
   - `buscar_normas`: as buscas do plano mais a pergunta original. Os candidatos são unidos e reordenados uma vez pelo reranker. Trechos com escore < 0,05 ficam fora do contexto, que tem no máximo 6 trechos.
   - `estimar_pm25`: o modelo da Parte 1. Devolve PM2.5, IC 90%, faixa do IQAr e o padrão vigente na data-alvo.
3. **Recusa sem gerar**: se não há ferramenta e o melhor escore do reranker fica abaixo de 0,8, o agente responde que não encontrou a informação na base.
4. **Responder** (LLM, JSON `{encontrado, resposta, citacoes}`): cita [n] para os trechos e [Fn] para as ferramentas. O prompt traz a data de referência ("hoje"/"amanhã") e a regra de comparar com o padrão vigente **na data da estimativa**.
5. **Validar sem LLM**:
   - as citações existem?
   - cada número da resposta aparece nos trechos citados, na nota de vigência, na saída da ferramenta ou na pergunta?
   - Se não, há uma regeneração que aponta o problema.
6. **Registrar** em `logs/agent_runs.jsonl`: pergunta, plano, trechos (com escores), chamadas de ferramenta com argumentos e resultados, saída bruta de cada chamada ao LLM, tokens, tempo e custo estimado.

**"Hoje" é a data de referência do sistema**, o último dia com dados completos (04/04/2023), porque não há feed em tempo real (seção 1). Dá para trocar com `AGENT_TODAY`. A vigência das normas, por outro lado, usa a data real, a menos que a pergunta fale de outra época.

**Erros que a primeira execução real mostrou** (log da pergunta de Campinas), e que corrigi:
- o LLM recusou porque não sabia que "amanhã" era 05/04/2023 e achou que não podia comparar com a 491, revogada depois;
- o plano inventou uma "data da norma" (2018-03-28) e repetiu a chamada da ferramenta.

Depois das correções, a resposta ficou assim (log `538dc57c86`):

> O PM2.5 previsto para Campinas amanhã (2023-04-05) é de 20,7 µg/m³, com intervalo de 90% entre 11,7 e 29,1 µg/m³. O padrão vigente na data de referência é o da Resolução CONAMA 491/2018, art. 4º, § 1º e Anexo I, PI-1 de 60 µg/m³ (24 h) [1][6]. O valor previsto é menor que 60 µg/m³, portanto não ultrapassa o padrão legal. A 491/2018 foi revogada pela 506/2024 (art. 11, I), mas na data da estimativa ainda era vigente [1].

**LLM configurável** (`LLM_PROVIDER`): `llamacpp` (padrão, GGUF local, grátis), `ollama` (local, grátis) ou `openai_compat` (Groq, Gemini, OpenRouter, com camadas gratuitas). Os provedores remotos recebem `response_format=json_object`; no llama.cpp a saída é restrita por gramática JSON, o que torna modelos de 4B confiáveis no formato.

<!--P2_AVALIACAO-->

## 6. O que foi descartado e por quê

| Descartado | Por quê |
|---|---|
| Validação aleatória como resultado principal | vaza a mesma estação e o mesmo dia pelas vizinhas (R² 0,84 contra 0,72 na validação correta); fica só como referência |
| Alvo absoluto de PM2.5 (sem offset do IDW) | as árvores não extrapolam o nível do dia; MAE 3,72 contra 3,00 |
| MLP, Ridge, XGBoost profundo | pior que o LightGBM regularizado em estação nunca vista (seção 4.1) |
| Conformal do resíduo absoluto | largura fixa: cobre mal os dias poluídos (56,5% acima de 25 µg/m³) |
| Chunk de tamanho fixo / um chunk por inciso | corta o inciso do caput, ou perde o caput |
| Tabela dos anexos como texto corrido | o LLM troca a coluna (PI-1 × PF) |
| Qwen3-Embedding-0.6B (o padrão inicial) | pior no benchmark (recall@5 de 0,917 no híbrido, contra 0,983) e 10x mais lento em CPU |
| Reranker a cada busca do plano | ~60 s por reordenação em CPU; passou a rodar uma vez sobre a união dos candidatos |
| GPU (GTX 1050) para o LLM | a wheel CUDA do llama-cpp-python usa instruções que o i5-9300H não tem (*illegal instruction*); compilar exigiria o CUDA Toolkit |
| Res. CONAMA 513/2026 na base | o texto só está no DOU, que bloqueia acesso automatizado |

## 7. Limitações

**Parte 1**
- **Período de 2021 a abr/2023 e ~50% dos dias**, por causa da única fonte pública e anônima de PM2.5 da CETESB que encontrei (seção 1). Não há feed em tempo real: "hoje" é o último dia com dados.
- **D+1 em *perfect prognosis***: a meteorologia de amanhã vem do ERA5 do próprio dia. Na operação viria de uma previsão numérica, com erro maior.
- **Episódios de fumaça são subestimados** (viés de −3,7 µg/m³; seção 4.5), e o IC 90% cobre só 80,8% acima de 50 µg/m³.
- O AOD do GOES-16 tem sinal fraco para o PM2.5 de superfície em SP; o ganho do satélite é real, mas pequeno perto do lag espacial.

**Parte 2**
- **Tempo em CPU**: 4 a 8 min por pergunta num notebook i5 sem GPU, porque o prompt de ~2.500 tokens sai a ~16 tokens/s. Com Ollama em GPU ou uma API gratuita (`LLM_PROVIDER=openai_compat`), cai para segundos sem mudar o código.
- **Conjunto de avaliação pequeno (37 perguntas)**, escrito por mim. Os limiares de recusa foram calibrados nele mesmo, então os números de recusa são otimistas.
- **O LLM-juiz é outro modelo de 4B** e pode errar. Por isso também reporto a checagem automática (números e citações com suporte), que não depende de LLM.
- A base tem 7 normas. Perguntas sobre licenciamento detalhado (CONAMA 237), água (CONAMA 357) ou normas estaduais de SP são recusadas, por design.
- A fonte de 2 das 7 normas é uma cópia pública do DOU, não o próprio DOU (o texto é a impressão da página oficial, com a URL de origem).

## 8. Próximos passos

1. **D+1 operacional**: trocar o ERA5 do dia seguinte pela previsão do GFS/ECMWF aberto e medir a perda em relação ao *perfect prognosis*.
2. **Mais dados de PM2.5**: QUALAR com login (2019 em diante) e MonitorAr, para treinar com mais estações secas e melhorar os episódios.
3. **Episódios de fumaça**: perda quantílica ou peso maior nos dias altos, e altura da pluma do CAMS como variável.
4. **Agente**: rodar em GPU ou com API gratuita; ampliar a base (CONAMA 237, 357, 382, a 513/2026 e normas da CETESB); conjunto de avaliação maior, com um conjunto separado para calibrar os limiares.

<!--FIM-->
