# Relatório técnico

## 1. Resultado e escopo

Versão final de variáveis: **causal-spatial-aod-v2**. São 11.543 estação-dias de 30 estações CETESB, entre 2021-01-13 e 2023-04-04, no estado de São Paulo. Grade de 0,05° (~5,5 km).

Na validação de estações não vistas e período posterior, o LightGBM regularizado com offset do IDW obteve **MAE 2.98 µg/m³, RMSE 4.72 µg/m³ e R² 0.728**. O ganho de MAE sobre o IDW foi **4.34%**. O intervalo nominal de 90% para a estimativa diária teve cobertura empírica de **90.61%**. A cobertura de D+1 é medida separadamente na seção 5. O ganho em MAE vem com uma ressalva, detalhada na seção 5.1: o modelo suaviza os episódios e detecta menos dias acima de 25 µg/m³ que o IDW e a persistência. Para alerta, o limite superior do intervalo é o sinal adequado.

Não há feed atual: o produto é uma demonstração histórica. Uma emissão ao fim de D usa as agregações completas de D e produz a estimativa de D e a previsão de D+1. “Hoje/amanhã” no agente usam uma referência histórica explícita. A CLI recebe a data de referência; horizonte 1 significa data-alvo = referência + um dia.

O desafio recomenda 2019 em diante, mas permite recorte e fontes alternativos justificados. O arquivo público anônimo de PM2.5 utilizado limita o período e tem lacunas em aproximadamente metade dos dias. Priorizei as duas partes funcionais, validação espaço-temporal, incerteza medida e rastreabilidade das respostas.

## 2. Dados, fontes e reprodução

| Variável | Fonte utilizada | Justificativa |
|---|---|---|
| PM2.5 horário | CETESB via arquivo OpenAQ público no S3 | Mesmas estações da rede CETESB; o QUALAR exige cadastro e não tem API anônima. Média diária só com pelo menos 18 h válidas |
| AOD | GOES-16 ABI L2 AODF | O MCD19A2 (MAIAC) não está no catálogo do Planetary Computer (conferido em 30/10/2026; há só outros produtos MODIS) e GEE/LP DAAC exigem conta. O GOES-16 é geoestacionário: oito varreduras diurnas por dia contra duas passagens MODIS, menos dias perdidos por nuvem; controle de qualidade DQF |
| NO2 troposférico | Sentinel-5P TROPOMI, COG público MEEO/S3 | Sinal de tráfego/indústria e precursor de aerossol secundário; qa ≥ 0,75 |
| Fogo e FRP | GOES-16 ABI L2 FDCF | O BDQueimadas do INPE não respondia no ambiente inicial e agrega, entre outros, as detecções do próprio GOES-16. O FDCF fornece também a potência radiativa (FRP), usada no peso do fogo. O INPE voltou a responder na revisão; não troquei a fonte para não refazer as variáveis no fim do prazo |
| Meteorologia | ERA5, espelho público NCAR/S3 | Vento, umidade, chuva, temperatura, nuvens e camada limite, sem chave CDS |
| Uso do solo e relevo | ESA WorldCover 2021 e Copernicus DEM | Blocos públicos com leitura parcial |
| Limite estadual e municípios | IBGE espelhado no GitHub | Mesmo recorte territorial, sem dependência do acesso inicial a gov.br |

Os coletores estão em `src/aq/data/`; usam leituras parciais HTTP de NetCDF/HDF5 e COG. `python -m src.aq.pipeline --full` baixa, monta as bases, compara, treina e gera evidências. `python -m src.aq.pipeline` refaz a modelagem completa sobre a amostra de aproximadamente 12 MB, sem baixar brutos. Para demonstrar em poucos minutos, `python -m src.aq.pipeline --quick` treina e calibra a produção, faz previsão e um mapa, conservando os CSVs de teste já fornecidos. A amostra contém as bases de treino completas com variáveis e cubos de contexto em algumas janelas, suficientes para mapas, CLI e agente.

As datas são agregadas no dia local UTC−3 do período escolhido. AOD/NO2 usam célula de 0,05° e vizinhanças 3×3/9×9; ERA5 usa interpolação bilinear; estáticos usam 1 km e média de 5 km. O instante horário do alvo marca o fim da hora: 00:00 é atribuído ao dia anterior.

### Falhas de AOD e causalidade

Hierarquia exclusivamente espacial: célula observada (nível 0), média 3×3 (1), média 9×9 (2), ausente (4). Após essas etapas, NaN permanece como ausência aprendida pelo LightGBM, acompanhado pela flag. Não há preenchimento temporal e nenhuma observação de D+1 entra nas variáveis de D.

AOD observado na célula em **50.8%** das linhas; após preenchimento espacial, **73.9%**. O AOD isolado é um sinal fraco na superfície: mede a coluna, influenciada por umidade, altura e transporte dos aerossóis. Por isso entram combinações físicas, NO2, fogo e estações vizinhas.

A revisão retirou o antigo preenchimento D−1/D+1 e a meteorologia realizada de amanhã. `scripts/migrate_causal_sample.py` faz uma migração idempotente das variáveis espaciais armazenadas, recalcula as combinações físicas e remove campos meteorológicos futuros. O download completo já gera o novo esquema diretamente. Bases e modelos antigos são recusados por verificação do identificador de esquema.

## 3. Variáveis e validação

| Grupo | Motivação física |
|---|---|
| AOD / altura da camada limite | Aproxima a relação entre coluna e concentração perto da superfície |
| AOD × (1−UR)^0,6 | Correção aproximada do crescimento higroscópico |
| BLH × velocidade do vento | Índice de ventilação e dispersão |
| Chuva de D e D−1 | Remoção úmida |
| Fogo a 150/400 km ponderado por direção do vento, FRP e distância | Transporte de fumaça, incluindo defasagem de um dia |
| NO2, urbanização, vegetação e relevo | Fontes locais e contexto de dispersão |
| IDW das outras estações em D | Nível regional observado; a estação-alvo é excluída |
| Calendário de D e D+1 | Sazonalidade e dia da semana conhecido na emissão |

O alvo do modelo pontual é `log1p(PM2.5) − log1p(IDW)`: o modelo aprende a correção local sobre um baseline regional forte. A escolha do LightGBM regularizado foi mantida da etapa inicial; o benchmark foi refeito com as variáveis causais.

As estações são divididas em cinco grupos determinísticos. Cada grupo fica totalmente fora do treino e das referências do IDW durante seu teste. O split temporal cruza esses grupos com:

- T1: treino até maio/2021, calibração em junho, teste julho–dezembro/2021 (seca/fumaça).
- T2: treino até março/2022, calibração abril–junho, teste julho/2022–abril/2023.

Na previsão, a **data-alvo D+1** determina os limites de treino, calibração e teste, impedindo que o último rótulo do treino atravesse a fronteira. A validação aleatória é apenas referência otimista. As métricas agregam os folds; não há um teste externo intocado para seleção de toda a família de modelos, portanto o benchmark deve ser interpretado como comparação exploratória, não uma estimativa sem viés de seleção.

## 4. Estimativa diária, intervalos e explicação

Todos os modelos da tabela principal são comparados nas mesmas linhas com previsões válidas, inclusive os baselines. Métricas em µg/m³; ganho positivo representa redução do MAE.

| modelo | n | MAE | RMSE | R2 | ganho_MAE_vs_baseline_% | acerto_faixa_IQAr |
| --- | --- | --- | --- | --- | --- | --- |
| LightGBM regularizado (offset IDW) | 7761 | 2.9771 | 4.7190 | 0.7275 | 4.3361 | 0.9437 |
| Ensemble RF + LightGBM reg. (offset IDW) | 7761 | 3.0029 | 4.7514 | 0.7237 | 3.5078 | 0.9423 |
| XGBoost reg. forte (offset IDW) | 7761 | 3.0594 | 4.8313 | 0.7143 | 1.6902 | 0.9405 |
| Krigagem ordinária | 7761 | 3.0785 | 4.9122 | 0.7047 | 1.0777 | 0.9401 |
| IDW (baseline) | 7761 | 3.1120 | 4.8673 | 0.7101 | 0.0000 | 0.9434 |
| Random Forest (offset IDW) | 7761 | 3.1368 | 4.9130 | 0.7046 | -0.7944 | 0.9401 |
| XGBoost regularizado (offset IDW) | 7761 | 3.1580 | 4.9528 | 0.6998 | -1.4780 | 0.9397 |
| LightGBM (offset IDW) | 7761 | 3.2539 | 5.0440 | 0.6886 | -4.5590 | 0.9378 |
| XGBoost (offset IDW) | 7761 | 3.2981 | 5.1354 | 0.6773 | -5.9790 | 0.9347 |
| LightGBM (offset) sem satélite | 7761 | 3.4781 | 5.4551 | 0.6358 | -11.7619 | 0.9295 |
| Ensemble RF + LightGBM reg. (log) | 7761 | 3.5520 | 6.0304 | 0.5550 | -14.1355 | 0.9192 |
| Random Forest (log) | 7761 | 3.5798 | 6.0149 | 0.5572 | -15.0312 | 0.9210 |
| LightGBM regularizado (log) | 7761 | 3.6269 | 6.1373 | 0.5390 | -16.5430 | 0.9132 |
| LightGBM (alvo bruto) | 7761 | 3.6341 | 5.6513 | 0.6091 | -16.7756 | 0.9241 |
| HistGB (log) | 7761 | 3.7211 | 6.0726 | 0.5487 | -19.5711 | 0.9199 |
| LightGBM (log) | 7761 | 3.7268 | 6.1012 | 0.5444 | -19.7535 | 0.9192 |
| MLP PyTorch 64-32, saída tanh [-0,7; 0,7] (offset IDW) | 7761 | 3.7347 | 5.8299 | 0.5840 | -20.0083 | 0.9281 |
| XGBoost (log) | 7761 | 3.8954 | 6.4924 | 0.4841 | -25.1725 | 0.9152 |
| LightGBM sem satélite | 7761 | 3.9507 | 6.5747 | 0.4710 | -26.9483 | 0.9094 |
| MLP PyTorch 64-32, saída tanh [-0,7; 0,7] (log) | 7761 | 4.3091 | 6.8496 | 0.4258 | -38.4652 | 0.9129 |
| LightGBM sem lag espacial | 7761 | 4.4031 | 6.9993 | 0.4004 | -41.4847 | 0.9111 |
| Ridge (offset IDW) | 7761 | 5.5639 | 10.1769 | -0.2675 | -78.7871 | 0.8742 |
| Ridge (log) | 7761 | 5.6717 | 10.1238 | -0.2543 | -82.2496 | 0.8670 |
| Climatologia mensal | 7761 | 6.2549 | 9.3348 | -0.0664 | -100.9908 | 0.9017 |
| MLP 64-32 (offset IDW) | 7761 | 6.3432 | 11.2786 | -0.5568 | -103.8279 | 0.8815 |
| LightGBM (offset IDW) - validação ALEATÓRIA [otimista] | 11543 | 2.1488 | 3.3919 | 0.8355 | — | 0.9627 |

Recortes por tempo e distância à estação de referência mais próxima:

| recorte | grupo | modelo | n | MAE | estacoes |
| --- | --- | --- | --- | --- | --- |
| tfold | T1 | IDW (baseline) | 4102 | 3.6195 | 29 |
| tfold | T1 | Krigagem ordinária | 4102 | 3.6547 | 29 |
| tfold | T1 | LightGBM regularizado (offset IDW) | 4102 | 3.6253 | 29 |
| tfold | T2 | IDW (baseline) | 3659 | 2.5432 | 30 |
| tfold | T2 | Krigagem ordinária | 3659 | 2.4325 | 30 |
| tfold | T2 | LightGBM regularizado (offset IDW) | 3659 | 2.2505 | 30 |
| faixa_dist | <10 km | IDW (baseline) | 4624 | 2.8924 | 17 |
| faixa_dist | <10 km | Krigagem ordinária | 4624 | 2.9984 | 17 |
| faixa_dist | <10 km | LightGBM regularizado (offset IDW) | 4624 | 2.9390 | 17 |
| faixa_dist | 10-25 km | IDW (baseline) | 1475 | 3.2819 | 6 |
| faixa_dist | 10-25 km | Krigagem ordinária | 1475 | 3.2237 | 6 |
| faixa_dist | 10-25 km | LightGBM regularizado (offset IDW) | 1475 | 3.4852 | 6 |
| faixa_dist | 25-50 km | IDW (baseline) | 1477 | 3.5106 | 6 |
| faixa_dist | 25-50 km | Krigagem ordinária | 1477 | 2.9925 | 6 |
| faixa_dist | 25-50 km | LightGBM regularizado (offset IDW) | 1477 | 2.4170 | 6 |
| faixa_dist | >50 km | IDW (baseline) | 185 | 4.0659 | 1 |
| faixa_dist | >50 km | Krigagem ordinária | 185 | 4.6109 | 1 |
| faixa_dist | >50 km | LightGBM regularizado (offset IDW) | 185 | 4.3492 | 1 |

O ganho se concentra onde o problema pede: a 25–50 km da referência mais próxima, o MAE cai de 3.51 (IDW) para 2.42 µg/m³ (31%). Perto das estações (< 25 km), o IDW já é difícil de bater e o modelo empata ou perde. Acima de 50 km há uma só estação, então o recorte não sustenta conclusão.

Intervalos: quantis LightGBM em log, correção CQR por escores de calibração fora da estação (cross-fitting), comparados ao conformal do resíduo absoluto. A calibração usa somente o período anterior ao teste. Dependência espaço-temporal e cross-fitting impedem prometer a garantia exata do split conformal clássico; reporto a cobertura empírica.

| recorte | metodo | n | cobertura_nominal | cobertura_real | largura_media | cobertura_pm25_>25 |
| --- | --- | --- | --- | --- | --- | --- |
| todos | Quantis LightGBM 5-95% (sem calibrar) | 7859 | 0.9000 | 0.7972 | 9.5762 | 0.7300 |
| todos | CQR (quantis conformalizados) | 7859 | 0.9000 | 0.9061 | 13.4851 | 0.8362 |
| todos | Conformal do resíduo absoluto | 7859 | 0.9000 | 0.9202 | 14.6120 | 0.5727 |
| T1 | Quantis LightGBM 5-95% (sem calibrar) | 4102 | 0.9000 | 0.8137 | 12.2344 | 0.7528 |
| T1 | CQR (quantis conformalizados) | 4102 | 0.9000 | 0.9018 | 16.3427 | 0.8478 |
| T1 | Conformal do resíduo absoluto | 4102 | 0.9000 | 0.8854 | 15.7418 | 0.5810 |
| T2 | Quantis LightGBM 5-95% (sem calibrar) | 3757 | 0.9000 | 0.7791 | 6.6738 | 0.3830 |
| T2 | CQR (quantis conformalizados) | 3757 | 0.9000 | 0.9108 | 10.3651 | 0.6596 |
| T2 | Conformal do resíduo absoluto | 3757 | 0.9000 | 0.9582 | 13.3786 | 0.4468 |

Cobertura condicional por faixa:

| faixa | n | cobertura_cqr | largura_cqr |
| --- | --- | --- | --- |
| 0-15 | 5471.0000 | 0.9205 | 9.9933 |
| 15-25 | 1598.0000 | 0.9055 | 17.2020 |
| 25-35 | 478.0000 | 0.8368 | 26.9316 |
| 35-50 | 233.0000 | 0.8455 | 35.2784 |
| >50 | 52.0000 | 0.7885 | 44.9669 |

SHAP explica a correção em log sobre o IDW, não diretamente µg/m³. `shap_importancia.csv`, `shap_exemplo_local.csv`, `shap_resumo.png` e `shap_dependencia.png` mostram importância global e exemplo local. O ajuste SHAP é do modelo final; ele é uma explicação descritiva, não uma avaliação adicional fora da amostra.

Episódios de fumaça são definidos pelo percentil 95 do índice de fogo a favor do vento. O recorte é descritivo, definido para análise do teste; não é um classificador operacional de fumaça.

| regime | modelo | n | PM2.5_medio | MAE | vies |
| --- | --- | --- | --- | --- | --- |
| PM2.5 alto sem fogo (≥ p95) | IDW (baseline) | 179 | 38.9340 | 7.3290 | -5.8080 |
| PM2.5 alto sem fogo (≥ p95) | LightGBM regularizado (offset IDW) | 179 | 38.9340 | 9.1610 | -7.6050 |
| fumaça (fogo a favor do vento ≥ p95) | IDW (baseline) | 393 | 34.3240 | 8.0280 | 1.3100 |
| fumaça (fogo a favor do vento ≥ p95) | LightGBM regularizado (offset IDW) | 393 | 34.3240 | 8.7780 | -3.6750 |
| normal | IDW (baseline) | 7286 | 11.8510 | 2.7370 | 0.8370 |
| normal | LightGBM regularizado (offset IDW) | 7286 | 11.8510 | 2.4990 | -0.4140 |

As árvores tendem a encolher os extremos para a média; o T1 tem pouco histórico seco no treino. A fumaça pode estar acima da camada limite e o AOD pode faltar nos episódios densos. Esses fatores motivam as análises de `piores_dias.csv` e `episodios_serie.png`; são hipóteses físicas sobre os erros, não atribuições causais demonstradas.

## 5. Previsão real de D+1 em horizonte, sem meteorologia futura

Usa as variáveis observadas de D e D−1 e o calendário conhecido de D+1. Não é mais o experimento de perfect prognosis da versão inicial. A reanálise ERA5 continua sendo uma reconstrução histórica do dia D: sua latência operacional e possíveis revisões não são simuladas. O próximo passo de operação é substituir a fonte por observações/previsões disponíveis no horário de emissão e monitorar sua latência.

A: local com estação, persistência = PM2.5 de D. B: local sem estação, estações-alvo totalmente excluídas, persistência = IDW de D. São cenários distintos, reportados separadamente.

| cenario | recorte | metodo | n | MAE | RMSE | R2 | acerto_faixa_IQAr | ganho_MAE_vs_baseline_% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A) local com estação | todos | Persistência (amanhã = hoje) | 7373 | 4.2942 | 6.4293 | 0.4941 | 0.9226 | — |
| A) local com estação | todos | LightGBM reg. (offset IDW) D+1 | 7373 | 4.0507 | 6.0732 | 0.5486 | 0.9265 | 5.6688 |
| A) local com estação | T1 | Persistência (amanhã = hoje) | 3898 | 5.2008 | 7.6330 | 0.5046 | 0.8728 | — |
| A) local com estação | T1 | LightGBM reg. (offset IDW) D+1 | 3898 | 5.0827 | 7.3743 | 0.5376 | 0.8725 | 2.2718 |
| A) local com estação | T2 | Persistência (amanhã = hoje) | 3475 | 3.2772 | 4.7275 | 0.0828 | 0.9784 | — |
| A) local com estação | T2 | LightGBM reg. (offset IDW) D+1 | 3475 | 2.8932 | 4.1543 | 0.2917 | 0.9871 | 11.7162 |
| B) local sem estação | todos | Persistência do IDW de hoje | 7372 | 4.9109 | 7.1697 | 0.3709 | 0.9109 | — |
| B) local sem estação | todos | LightGBM reg. (offset IDW) D+1 | 7372 | 4.4724 | 6.7807 | 0.4373 | 0.9200 | 8.9299 |
| B) local sem estação | T1 | Persistência do IDW de hoje | 3898 | 5.9129 | 8.6064 | 0.3701 | 0.8486 | — |
| B) local sem estação | T1 | LightGBM reg. (offset IDW) D+1 | 3898 | 5.6358 | 8.3464 | 0.4076 | 0.8610 | 4.6860 |
| B) local sem estação | T2 | Persistência do IDW de hoje | 3474 | 3.7867 | 5.0964 | -0.0657 | 0.9807 | — |
| B) local sem estação | T2 | LightGBM reg. (offset IDW) D+1 | 3474 | 3.1670 | 4.4048 | 0.2039 | 0.9862 | 16.3655 |

**Cobertura do intervalo de D+1 medida no teste**, sem reutilizar a cobertura da estimativa de D:

Na revisão, a cobertura agregada foi **88.17% em A** e **88.14% em B**, abaixo do nominal de 90%. Os intervalos são identificados pelo nível nominal; não prometem cobertura de 90% em todo período ou faixa. A queda na seca e nos valores altos exige cautela e mais dados de calibração representativos.

| cenario | recorte | n | cobertura_nominal | cobertura_real | largura_media | cobertura_pm25_>25 |
| --- | --- | --- | --- | --- | --- | --- |
| A) local com estação | todos | 7373 | 0.9000 | 0.8817 | 16.2825 | 0.7792 |
| A) local com estação | T1 | 3898 | 0.9000 | 0.8533 | 19.2866 | 0.8146 |
| A) local com estação | T2 | 3475 | 0.9000 | 0.9137 | 12.9128 | 0.2143 |
| B) local sem estação | todos | 7372 | 0.9000 | 0.8814 | 18.7554 | 0.7440 |
| B) local sem estação | T1 | 3898 | 0.9000 | 0.8438 | 21.5158 | 0.7743 |
| B) local sem estação | T2 | 3474 | 0.9000 | 0.9237 | 15.6580 | 0.2619 |

Em A, quantis e CQR são calibrados nas estações monitoradas no período de calibração. Em B, os escores vêm de cross-fitting interno entre estações de referência. O ponto e o intervalo usados na inferência pertencem ao mesmo cenário. `oof_previsao_d1.parquet`, gerado pelo treino, contém rótulos, previsões, limites e datas para auditoria das métricas.

### 5.1 Acerto na faixa do IQAr: o número bruto engana

90.2% dos estação-dias de teste estão na faixa Boa (≤ 25 µg/m³). Prever sempre “Boa” já acerta essa fração, então o acerto bruto de ~94% das tabelas acima diz pouco. A tabela abaixo (`faixas_iqar.csv`, mesmas linhas das tabelas anteriores) acrescenta acurácia balanceada, F1 macro e sensibilidade/precisão para dias acima de 25 µg/m³:

| tarefa | metodo | n | acerto_faixa | acerto_sempre_Boa | acuracia_balanceada | f1_macro | n_obs_>25 | sensibilidade_>25 | precisao_>25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| estimativa D | IDW (baseline) | 7761 | 0.9434 | 0.9017 | 0.4893 | 0.4866 | 763 | 0.7851 | 0.7314 |
| estimativa D | Krigagem ordinária | 7761 | 0.9401 | 0.9017 | 0.4434 | 0.4598 | 763 | 0.6802 | 0.7555 |
| estimativa D | LightGBM regularizado (offset IDW) | 7761 | 0.9437 | 0.9017 | 0.4439 | 0.4758 | 763 | 0.6225 | 0.8176 |
| estimativa D | Alerta: limite superior do IC 90% > 25 | 7859 | — | — | — | — | 763 | 0.9423 | 0.3796 |
| D+1 A) local com estação | Persistência | 7373 | 0.9226 | 0.9036 | 0.5540 | 0.5560 | 711 | 0.6442 | 0.6370 |
| D+1 A) local com estação | LightGBM reg. (offset IDW) D+1 | 7373 | 0.9265 | 0.9036 | 0.3436 | 0.3750 | 711 | 0.3741 | 0.8261 |
| D+1 A) local com estação | Alerta: limite superior do IC 90% > 25 | 7373 | — | — | — | — | 711 | 0.9128 | 0.3342 |
| D+1 B) local sem estação | Persistência | 7372 | 0.9109 | 0.9036 | 0.4207 | 0.4164 | 711 | 0.6273 | 0.5733 |
| D+1 B) local sem estação | LightGBM reg. (offset IDW) D+1 | 7372 | 0.9200 | 0.9036 | 0.3339 | 0.3517 | 711 | 0.3783 | 0.7060 |
| D+1 B) local sem estação | Alerta: limite superior do IC 90% > 25 | 7372 | — | — | — | — | 711 | 0.8889 | 0.2545 |

Matriz de confusão da estimativa do modelo escolhido (linhas = faixa medida, colunas = faixa estimada; `faixas_iqar_confusao.csv`):

| medida | Boa | Moderada | Ruim | Muito Ruim | Péssima |
| --- | --- | --- | --- | --- | --- |
| Boa | 6892 | 105 | 1 | 0 | 0 |
| Moderada | 281 | 423 | 7 | 0 | 0 |
| Ruim | 6 | 31 | 9 | 0 | 0 |
| Muito Ruim | 1 | 0 | 5 | 0 | 0 |
| Péssima | 0 | 0 | 0 | 0 | 0 |

Leitura honesta: o modelo reduz o MAE, mas **suaviza os episódios**. Na estimativa, detecta 62% dos dias acima de 25 µg/m³, contra 79% do IDW. Em D+1 a diferença cresce: 37% contra 64% da persistência no cenário A. Ele erra menos para cima (precisão maior), mas deixa passar dias poluídos. Para alerta, o ponto central não deve ser usado sozinho. A regra fixa “limite superior do IC 90% acima de 25”, sem nenhum ajuste no teste, detecta 91% dos dias poluídos em D+1 (cenário A), com precisão de 33%. Ou seja, o intervalo carrega a informação de risco que o ponto central perde. Um limiar ou classificador de alerta calibrado fora do teste é o próximo passo natural.

As faixas Moderada a Péssima seguem o Guia Técnico do MMA. A Res. CONAMA 491/2018 (Anexo IV) publica só a linha N1 – Boa (MP2,5 0–25 µg/m³). A Res. 506/2024 (Anexo II) reduziu essa faixa para 0–15 µg/m³ a partir de 9/7/2024. Os dados vão de 2021 a 2023, então as métricas usam corretamente a regra da 491. Uma consulta com data posterior a 2024 precisaria da faixa da 506.

## 6. Mapas e interfaces

`reports/figures/mapa_pm25_*` contém mapas diários de estimativa/previsão e da largura do intervalo de 90%. O comando portável gera dias típico, poluído e de fumaça; o mapa D+1 de 24/08/2021 é emitido em 23/08/2021. CLI, mapa, API e ferramenta do agente compartilham a mesma função de inferência.

A saída inclui concentração, limites, faixa IQAr, data de referência e alvo, padrão normativo na data-alvo, ultrapassagem e diagnóstico do AOD/IDW. A distância à estação mais próxima é calculada com pares reais de latitude/longitude. A produção é treinada com todos os dados para demonstração: os mapas históricos podem ser dentro da amostra e não substituem as métricas dos folds.

## 7. Agente com RAG

Sete documentos públicos: CONAMA 491/2018, 506/2024, 03/1990, 05/1989; Lei 6.938/1981; Decreto 99.274/1990; Lei 9.605/1998. As fontes e revogações parciais estão em `configs/normas.yaml`. Não há pretensão de cobrir toda a legislação ambiental vigente.

O parser preserva capítulo, artigo, parágrafo, inciso, alínea e anexo. A unidade preferida é o artigo; artigos longos são subdivididos com metadados de hierarquia. Tabelas são transformadas em linhas com o nome da coluna junto a cada valor, evitando confundir PI-1, PI-2 e PF. A recuperação combina BM25 e embeddings Granite-278M por RRF; reranker Qwen3-0.6B reordena candidatos e fornece um escore para a recusa. Resultados do benchmark existente, cuja base textual não mudou:

| config | recall@1 | recall@3 | recall@5 | recall@10 | hit@1 | hit@3 | hit@5 | hit@10 | MRR | latencia_ms | n | indexacao_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.6167 | 0.8500 | 0.9167 | 1.0000 | 0.7000 | 0.9000 | 0.9333 | 1.0000 | 0.7972 | 7.5979 | 30 | — |
| Vetorial (embeddinggemma-300m) | 0.6833 | 0.8500 | 0.8833 | 0.9667 | 0.7667 | 0.9000 | 0.9333 | 1.0000 | 0.8378 | 162.3455 | 30 | 1.6672 |
| Híbrido RRF (BM25 + embeddinggemma-300m) | 0.6500 | 0.8500 | 0.9500 | 0.9833 | 0.7333 | 0.9000 | 0.9667 | 1.0000 | 0.8317 | 168.9394 | 30 | — |
| Vetorial (granite-multilingual-278m) | 0.5833 | 0.7500 | 0.9000 | 0.9833 | 0.6667 | 0.8000 | 0.9333 | 1.0000 | 0.7639 | 116.1801 | 30 | 3.8798 |
| Híbrido RRF (BM25 + granite-multilingual-278m) | 0.7167 | 0.8333 | 0.9833 | 1.0000 | 0.8000 | 0.9000 | 1.0000 | 1.0000 | 0.8639 | 138.9072 | 30 | — |
| Vetorial (qwen3-emb-0.6b) | 0.5000 | 0.7333 | 0.8667 | 0.8667 | 0.5333 | 0.8000 | 0.9000 | 0.9000 | 0.6622 | 1360.6512 | 30 | 1.6245 |
| Híbrido RRF (BM25 + qwen3-emb-0.6b) | 0.6000 | 0.8333 | 0.9167 | 0.9833 | 0.6667 | 0.8667 | 0.9333 | 1.0000 | 0.7817 | 1410.2095 | 30 | — |
| Híbrido RRF (BM25 + granite-multilingual-278m) sem diversificação | 0.7167 | 0.8167 | 0.9333 | 0.9833 | 0.8000 | 0.9000 | 1.0000 | 1.0000 | 0.8639 | 132.6790 | 30 | — |
| Híbrido + reranker Qwen3-0.6B (top 8, granite-multilingual-278m) | 0.7000 | 0.8667 | 0.9500 | 1.0000 | 0.7667 | 0.9667 | 1.0000 | 1.0000 | 0.8639 | 61485.5350 | 30 | — |

O LLM Qwen3-4B local planeja consultas, recebe trechos e saídas de ferramentas, responde com referências e passa por uma validação numérica. A ferramenta de PM2.5 retorna valor, intervalo e comparações calculadas; a ferramenta de padrão vigente fornece cronograma e evidência normativa. Perguntas fora da base podem ser recusadas pelo limiar ou pelo modelo. Toda execução registra pergunta, recuperação, chamadas, resposta, tokens, tempo e custo; o custo local é zero.

Downloads incompletos não substituem a base; a ingestão grava atomicamente somente o conjunto completo e preserva a amostra versionada. A recuperação também detecta base processada incompleta e usa a amostra completa. `.env` é carregado sem executar código e sem substituir variáveis já exportadas.

### Avaliação, gabaritos e limites das métricas

37 perguntas: fatos, composição de trechos, vigência, três ferramentas e seis sem resposta. A precisão das citações agora verifica o **dispositivo**, não apenas o documento. Nas ferramentas são exigidos local, data-alvo, horizonte, valor e dois limites do IC; quando a pergunta pede faixa, exige a faixa retornada, em vez de aceitar qualquer nome. A citação do padrão é verificada separadamente; a semântica da comparação depende da análise das fontes/juiz, não só de presença de números.

Resultados da revisão:

| metrica | valor |
| --- | --- |
| modelo | qwen3-4b_causal |
| n | 37 |
| acerto | 0.9355 |
| acerto_parcial | 0.9516 |
| cit_precisao | 0.7902 |
| cit_recall | 0.9483 |
| fidelidade_auto | 1.0000 |
| fidelidade_juiz | 1.0000 |
| recusa_correta (sem resposta) | 1.0000 |
| recusa_indevida | 0.0323 |
| ferramenta_ok | 1.0000 |
| tempo_medio_s | 299.8430 |
| tokens_medio | 2488.8919 |
| acerto_fato | 1.0000 |
| acerto_ferramenta | 1.0000 |
| acerto_multi | 0.8333 |
| acerto_temporal | 0.8000 |
| metodo_avaliacao | dispositivo-exato-e-ferramenta-v2 |
| n_respondidas | 30.0000 |
| n_fidelidade_juiz | 30.0000 |

Proveniência: a tag `qwen3-4b_causal` reaproveita as 34 respostas normativas da avaliação anterior, com rechecagem dos gabaritos e citações exatas, e executa novamente as três perguntas que chamam o modelo alterado. O log preserva os timestamps originais e as novas execuções. Não representa uma rodada nova de 37 gerações nem um conjunto independente. As tags anteriores são históricas; suas métricas e previsões não representam o modelo causal atual.

“Fidelidade automática” mede rastreabilidade de números e referências, não equivalência semântica completa. O juiz é outro modelo local de 4B e também pode errar. Limiares de recusa foram ajustados no próprio conjunto; seis recusas corretas não demonstram generalização para qualquer pergunta fora da base. São limitações assumidas, a serem resolvidas com conjunto separado de calibração e avaliação maior.

## 8. Priorização, descartes e próximos passos

- MAIAC/QUALAR/CDS/INPE foram substituídos por fontes públicas acessíveis no ambiente inicial, mantendo a natureza física dos sinais (justificativa por fonte na seção 2).
- O modelo foi escolhido pelo MAE. Pelo critério de detecção de dias poluídos, o IDW e a persistência são melhores (seção 5.1). Não troquei o critério depois de ver o teste; registro o conflito e uso o intervalo para alerta.
- Preenchimento com AOD de D+1 e meteorologia realizada de D+1 foram retirados para respeitar o horizonte de emissão.
- Tendência do episódio no D+1 (PM2.5 e IDW de D−1, conhecidos ao fim de D) foi testada e descartada: piorou o MAE (A: 4,05 → 4,14; B: 4,47 → 4,54) e a detecção acima de 25 µg/m³. O experimento reproduz exatamente as linhas de base publicadas (`scripts/exp_tendencia_d1.py`, `logs/exp_tendencia_d1.txt`).
- Alvos absolutos, redes neurais e variantes mais complexas permanecem no benchmark; o escolhido aprende a correção sobre o IDW. Ganhos pequenos ou negativos em alguns recortes são reportados.
- Não foi implementado feed diário atual nem simulação da latência ERA5. Não apresento a demonstração histórica como monitoramento atual.
- Não há avaliação em estados externos, garantia de cobertura nos extremos ou norma estadual paulista completa na base.
- As faixas do IQAr seguem a 491/2018 para todas as datas. Correto para 2021–2023, mas desatualizado após a 506/2024 (Boa ≤ 15 µg/m³).
- Próximos passos: mais estações e anos, fonte meteorológica disponível na emissão, previsão numérica opcional, peso nos extremos/altura de pluma, limiar de alerta calibrado fora do teste, faixas do IQAr por data de vigência, teste externo de RAG e calibração independente dos limiares.

## 9. Engenharia, evidências e uso de IA

Dependências fixadas, Dockerfile e Makefile; pipeline portável com um comando, testes de causalidade (retirar todas as observações futuras não altera as variáveis), preservação da base normativa, `.env` e avaliação das ferramentas. CSVs, figuras, notebooks com saída e logs são versionados; modelos grandes e brutos são reconstruídos.

A imagem foi reconstruída sem cache a partir de um clone limpo e validada em Linux/Python 3.13: **29 testes aprovados** e demonstração rápida completa em **2,3 minutos**, incluindo a inicialização do container, sem usar os modelos já treinados da máquina. Evidência: `logs/docker_build.txt` e `logs/docker_validacao.txt`. A instalação do README para Windows (Python 3.12, venv novo) também foi refeita em clone limpo: 29 testes aprovados e `--quick` em 3,6 minutos. A saída é UTF-8 também fora do `make`, para que respostas com “≤” possam ser redirecionadas a arquivo no Windows.

Os logs `benchmark_causal.txt`, `train_causal.txt`, `docker_build.txt`, `docker_validacao.txt` e `validacao_entrega.txt` registram a revisão. [USO_DE_IA.md](USO_DE_IA.md) declara a implementação inicial com Claude Code, a revisão causal com Codex e a auditoria final com Claude Code (análise de faixas do IQAr e justificativa das fontes). O candidato deve conseguir explicar e modificar o código na entrevista.

**Pendências externas à implementação:** gravar o vídeo não listado de até dez minutos e inserir o link no README; conceder acesso aos avaliadores quando seus usuários do GitHub forem informados (o repositório é privado). O prazo de 72 horas conta do recebimento do desafio; o repositório não comprova essa data nem a entrega por e-mail.
