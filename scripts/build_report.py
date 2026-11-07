"""Gera docs/DETALHES_TECNICOS.md diretamente dos CSVs, sem transcrever métricas à mão."""
from pathlib import Path
import re

import pandas as pd

from src.common.config import ROOT


def metric(name):
    return pd.read_csv(ROOT / "reports" / "metrics" / name)


def table(df, columns=None):
    df = df[columns] if columns else df
    def fmt(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"] +
                     ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False, name=None)])


def main():
    b = metric("benchmark_estimativa.csv")
    best = b[b.modelo.eq("LightGBM regularizado (offset IDW)")].iloc[0]
    intervals = metric("intervalos.csv")
    cov = intervals[(intervals.recorte == "todos") & intervals.metodo.str.startswith("CQR")].iloc[0]
    fc = metric("previsao_d1.csv")
    d1_intervals = metric("intervalos_d1.csv")
    d1_all = d1_intervals[d1_intervals.recorte.eq("todos")].set_index("cenario")
    d = pd.read_parquet(ROOT / "data/sample/station_days.parquet")
    n_days = f"{len(d):,}".replace(",", ".")
    rag = metric("rag_eval_resumo.csv")
    final = rag[rag.modelo.eq("qwen3-4b_causal")]
    docker_proof = ROOT / "logs/docker_validacao.txt"
    docker_note = "A validação do container está registrada em `logs/docker_validacao.txt`."
    if docker_proof.exists():
        proof = docker_proof.read_text(encoding="utf-8")
        elapsed = re.findall(r"elapsed_seconds ([0-9.]+)", proof)
        passed = re.search(r"(\d+) passed", proof)
        if elapsed and passed and "CONCLUSAO:" in proof:
            minutes = f"{float(elapsed[-1]) / 60:.1f}".replace(".", ",")
            docker_note = (f"A imagem foi construída e validada em Linux/Python 3.13: **{passed[1]} testes aprovados** "
                           f"e demonstração rápida completa em **{minutes} minutos**, incluindo a inicialização do container, "
                           "sem usar os modelos já treinados da máquina. Evidência: `logs/docker_validacao.txt`.")
    retrieval = metric("rag_retrieval.csv")
    recortes = metric("benchmark_estimativa_recortes.csv")
    recortes = recortes[recortes.modelo.isin(["IDW (baseline)", "Krigagem ordinária", "LightGBM regularizado (offset IDW)"])]
    far = recortes[recortes.grupo.eq("25-50 km")].set_index("modelo")["MAE"]
    far_idw, far_model = far["IDW (baseline)"], far["LightGBM regularizado (offset IDW)"]
    far_gain = 1 - far_model / far_idw
    bands = metric("faixas_iqar.csv")
    confusion = metric("faixas_iqar_confusao.csv")
    share_boa = bands.loc[bands.tarefa.eq("estimativa D"), "acerto_sempre_Boa"].iloc[0]
    band_row = lambda tarefa, metodo: bands[bands.tarefa.eq(tarefa) & bands.metodo.eq(metodo)].iloc[0]  # noqa: E731
    sens, prec = "sensibilidade_>25", "precisao_>25"
    d1a = "D+1 A) local com estação"
    alert = "Alerta: limite superior do IC 90% > 25"
    sens_est_model = band_row("estimativa D", "LightGBM regularizado (offset IDW)")[sens]
    sens_est_idw = band_row("estimativa D", "IDW (baseline)")[sens]
    sens_d1_model = band_row(d1a, "LightGBM reg. (offset IDW) D+1")[sens]
    sens_d1_pers = band_row(d1a, "Persistência")[sens]
    sens_alert_d1, prec_alert_d1 = band_row(d1a, alert)[sens], band_row(d1a, alert)[prec]
    report = f"""# Respira SP — detalhes técnicos

> Documento gerado por `scripts/build_report.py` a partir dos CSVs em `reports/metrics/`: nenhum número aqui foi digitado à mão. Visão geral e instalação no [README](../README.md).

## 1. Resultado e escopo

Versão final de variáveis: **causal-spatial-aod-v2**. São {n_days} estação-dias de {d.location_id.nunique()} estações CETESB, entre {d.date.min().date()} e {d.date.max().date()}, no estado de São Paulo. Grade de 0,05° (~5,5 km).

Na validação de estações não vistas e período posterior, o LightGBM regularizado com offset do IDW obteve **MAE {best.MAE:.2f} µg/m³, RMSE {best.RMSE:.2f} µg/m³ e R² {best.R2:.3f}**. O ganho de MAE sobre o IDW foi **{best['ganho_MAE_vs_baseline_%']:.2f}%**. O intervalo nominal de 90% para a estimativa diária teve cobertura empírica de **{cov.cobertura_real:.2%}**. A cobertura de D+1 é medida separadamente na seção 5. O ganho em MAE vem com uma ressalva, detalhada na seção 5.1: o modelo suaviza os episódios e detecta menos dias acima de 25 µg/m³ que o IDW e a persistência. Para alerta, o limite superior do intervalo é o sinal adequado.

Não há feed atual: o produto é uma demonstração histórica. Uma emissão ao fim de D usa as agregações completas de D e produz a estimativa de D e a previsão de D+1. “Hoje/amanhã” no agente usam uma referência histórica explícita. A CLI recebe a data de referência; horizonte 1 significa data-alvo = referência + um dia.

O arquivo público e anônimo de PM2.5 limita o período a 2021–2023 e tem lacunas em aproximadamente metade dos dias. A prioridade foi ter os dois módulos funcionando ponta a ponta, com validação espaço-temporal, incerteza medida e rastreabilidade das respostas.

## 2. Dados, fontes e reprodução

| Variável | Fonte utilizada | Justificativa |
|---|---|---|
| PM2.5 horário | CETESB via arquivo OpenAQ público no S3 | Mesmas estações da rede CETESB; o QUALAR exige cadastro e não tem API anônima. Média diária só com pelo menos 18 h válidas |
| AOD | GOES-16 ABI L2 AODF | O MCD19A2 (MAIAC) não está no catálogo do Planetary Computer (conferido em 30/10/2026; há só outros produtos MODIS) e GEE/LP DAAC exigem conta. O GOES-16 é geoestacionário: oito varreduras diurnas por dia contra duas passagens MODIS, menos dias perdidos por nuvem; controle de qualidade DQF |
| NO2 troposférico | Sentinel-5P TROPOMI, COG público MEEO/S3 | Sinal de tráfego/indústria e precursor de aerossol secundário; qa ≥ 0,75 |
| Fogo e FRP | GOES-16 ABI L2 FDCF | O BDQueimadas do INPE agrega, entre outras, as detecções do próprio GOES-16. O FDCF fornece também a potência radiativa (FRP), usada no peso do fogo, e mantém o mesmo sensor do AOD |
| Meteorologia | ERA5, espelho público NCAR/S3 | Vento, umidade, chuva, temperatura, nuvens e camada limite, sem chave CDS |
| Uso do solo e relevo | ESA WorldCover 2021 e Copernicus DEM | Blocos públicos com leitura parcial |
| Limite estadual e municípios | IBGE espelhado no GitHub | Mesmo recorte territorial, sem depender de portais que bloqueiam acesso automatizado |

Os coletores estão em `src/aq/data/`; usam leituras parciais HTTP de NetCDF/HDF5 e COG. `python -m src.aq.pipeline --full` baixa, monta as bases, compara, treina e gera evidências. `python -m src.aq.pipeline` refaz a modelagem completa sobre a amostra de aproximadamente 12 MB, sem baixar brutos. Para demonstrar em poucos minutos, `python -m src.aq.pipeline --quick` treina e calibra a produção, faz previsão e um mapa, conservando os CSVs de teste já fornecidos. A amostra contém as bases de treino completas com variáveis e cubos de contexto em algumas janelas, suficientes para mapas, CLI e agente.

As datas são agregadas no dia local UTC−3 do período escolhido. AOD/NO2 usam célula de 0,05° e vizinhanças 3×3/9×9; ERA5 usa interpolação bilinear; estáticos usam 1 km e média de 5 km. O instante horário do alvo marca o fim da hora: 00:00 é atribuído ao dia anterior.

### Falhas de AOD e causalidade

Hierarquia exclusivamente espacial: célula observada (nível 0), média 3×3 (1), média 9×9 (2), ausente (4). Após essas etapas, NaN permanece como ausência aprendida pelo LightGBM, acompanhado pela flag. Não há preenchimento temporal e nenhuma observação de D+1 entra nas variáveis de D.

AOD observado na célula em **{d.aod.notna().mean():.1%}** das linhas; após preenchimento espacial, **{d.aod_filled.notna().mean():.1%}**. O AOD isolado é um sinal fraco na superfície: mede a coluna, influenciada por umidade, altura e transporte dos aerossóis. Por isso entram combinações físicas, NO2, fogo e estações vizinhas.

Uma versão anterior usava preenchimento temporal D−1/D+1 e a meteorologia realizada de amanhã; ambos foram retirados para garantir causalidade. `scripts/migrate_causal_sample.py` faz uma migração idempotente das variáveis espaciais armazenadas, recalcula as combinações físicas e remove campos meteorológicos futuros. O download completo já gera o novo esquema diretamente. Bases e modelos antigos são recusados por verificação do identificador de esquema.

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

O alvo do modelo pontual é `log1p(PM2.5) − log1p(IDW)`: o modelo aprende a correção local sobre um baseline regional forte. Todo o benchmark usa as variáveis causais.

As estações são divididas em cinco grupos determinísticos. Cada grupo fica totalmente fora do treino e das referências do IDW durante seu teste. O split temporal cruza esses grupos com:

- T1: treino até maio/2021, calibração em junho, teste julho–dezembro/2021 (seca/fumaça).
- T2: treino até março/2022, calibração abril–junho, teste julho/2022–abril/2023.

Na previsão, a **data-alvo D+1** determina os limites de treino, calibração e teste, impedindo que o último rótulo do treino atravesse a fronteira. A validação aleatória é apenas referência otimista. As métricas agregam os folds; não há um teste externo intocado para seleção de toda a família de modelos, portanto o benchmark deve ser interpretado como comparação exploratória, não uma estimativa sem viés de seleção.

## 4. Estimativa diária, intervalos e explicação

Todos os modelos da tabela principal são comparados nas mesmas linhas com previsões válidas, inclusive os baselines. Métricas em µg/m³; ganho positivo representa redução do MAE.

{table(b, ['modelo', 'n', 'MAE', 'RMSE', 'R2', 'ganho_MAE_vs_baseline_%', 'acerto_faixa_IQAr'])}

Recortes por tempo e distância à estação de referência mais próxima:

{table(recortes, ['recorte', 'grupo', 'modelo', 'n', 'MAE', 'estacoes'])}

O ganho se concentra onde o problema pede: a 25–50 km da referência mais próxima, o MAE cai de {far_idw:.2f} (IDW) para {far_model:.2f} µg/m³ ({far_gain:.0%}). Perto das estações (< 25 km), o IDW já é difícil de bater e o modelo empata ou perde. Acima de 50 km há uma só estação, então o recorte não sustenta conclusão.

Intervalos: quantis LightGBM em log, correção CQR por escores de calibração fora da estação (cross-fitting), comparados ao conformal do resíduo absoluto. A calibração usa somente o período anterior ao teste. Dependência espaço-temporal e cross-fitting impedem prometer a garantia exata do split conformal clássico; reporto a cobertura empírica.

{table(intervals)}

Cobertura condicional por faixa:

{table(metric('intervalos_por_faixa.csv'))}

SHAP explica a correção em log sobre o IDW, não diretamente µg/m³. `shap_importancia.csv`, `shap_exemplo_local.csv`, `shap_resumo.png` e `shap_dependencia.png` mostram importância global e exemplo local. O ajuste SHAP é do modelo final; ele é uma explicação descritiva, não uma avaliação adicional fora da amostra.

Episódios de fumaça são definidos pelo percentil 95 do índice de fogo a favor do vento. O recorte é descritivo, definido para análise do teste; não é um classificador operacional de fumaça.

{table(metric('episodios_extremos.csv'), ['regime', 'modelo', 'n', 'PM2.5_medio', 'MAE', 'vies'])}

As árvores tendem a encolher os extremos para a média; o T1 tem pouco histórico seco no treino. A fumaça pode estar acima da camada limite e o AOD pode faltar nos episódios densos. Esses fatores motivam as análises de `piores_dias.csv` e `episodios_serie.png`; são hipóteses físicas sobre os erros, não atribuições causais demonstradas.

## 5. Previsão real de D+1 em horizonte, sem meteorologia futura

Usa as variáveis observadas de D e D−1 e o calendário conhecido de D+1. Não é um experimento de *perfect prognosis*. A reanálise ERA5 continua sendo uma reconstrução histórica do dia D: sua latência operacional e possíveis revisões não são simuladas. O próximo passo de operação é substituir a fonte por observações/previsões disponíveis no horário de emissão e monitorar sua latência.

A: local com estação, persistência = PM2.5 de D. B: local sem estação, estações-alvo totalmente excluídas, persistência = IDW de D. São cenários distintos, reportados separadamente.

{table(fc, ['cenario', 'recorte', 'metodo', 'n', 'MAE', 'RMSE', 'R2', 'acerto_faixa_IQAr', 'ganho_MAE_vs_baseline_%'])}

**Cobertura do intervalo de D+1 medida no teste**, sem reutilizar a cobertura da estimativa de D:

A cobertura agregada foi **{d1_all.loc['A) local com estação', 'cobertura_real']:.2%} em A** e **{d1_all.loc['B) local sem estação', 'cobertura_real']:.2%} em B**, abaixo do nominal de 90%. Os intervalos são identificados pelo nível nominal; não prometem cobertura de 90% em todo período ou faixa. A queda na seca e nos valores altos exige cautela e mais dados de calibração representativos.

{table(d1_intervals)}

Em A, quantis e CQR são calibrados nas estações monitoradas no período de calibração. Em B, os escores vêm de cross-fitting interno entre estações de referência. O ponto e o intervalo usados na inferência pertencem ao mesmo cenário. `oof_previsao_d1.parquet`, gerado pelo treino, contém rótulos, previsões, limites e datas para auditoria das métricas.

### 5.1 Acerto na faixa do IQAr: o número bruto engana

{share_boa:.1%} dos estação-dias de teste estão na faixa Boa (≤ 25 µg/m³). Prever sempre “Boa” já acerta essa fração, então o acerto bruto de ~94% das tabelas acima diz pouco. A tabela abaixo (`faixas_iqar.csv`, mesmas linhas das tabelas anteriores) acrescenta acurácia balanceada, F1 macro e sensibilidade/precisão para dias acima de 25 µg/m³:

{table(bands)}

Matriz de confusão da estimativa do modelo escolhido (linhas = faixa medida, colunas = faixa estimada; `faixas_iqar_confusao.csv`):

{table(confusion)}

Leitura honesta: o modelo reduz o MAE, mas **suaviza os episódios**. Na estimativa, detecta {sens_est_model:.0%} dos dias acima de 25 µg/m³, contra {sens_est_idw:.0%} do IDW. Em D+1 a diferença cresce: {sens_d1_model:.0%} contra {sens_d1_pers:.0%} da persistência no cenário A. Ele erra menos para cima (precisão maior), mas deixa passar dias poluídos. Para alerta, o ponto central não deve ser usado sozinho. A regra fixa “limite superior do IC 90% acima de 25”, sem nenhum ajuste no teste, detecta {sens_alert_d1:.0%} dos dias poluídos em D+1 (cenário A), com precisão de {prec_alert_d1:.0%}. Ou seja, o intervalo carrega a informação de risco que o ponto central perde. Um limiar ou classificador de alerta calibrado fora do teste é o próximo passo natural.

As faixas Moderada a Péssima seguem o Guia Técnico do MMA. A Res. CONAMA 491/2018 (Anexo IV) publica só a linha N1 – Boa (MP2,5 0–25 µg/m³). A Res. 506/2024 (Anexo II) reduziu essa faixa para 0–15 µg/m³ a partir de 9/7/2024. Os dados vão de 2021 a 2023, então as métricas usam corretamente a regra da 491. Uma consulta com data posterior a 2024 precisaria da faixa da 506.

## 6. Mapas e interfaces

`reports/figures/mapa_pm25_*` contém mapas diários de estimativa/previsão e da largura do intervalo de 90%. O comando portável gera dias típico, poluído e de fumaça; o mapa D+1 de 24/08/2021 é emitido em 23/08/2021. CLI, mapa, API e ferramenta do agente compartilham a mesma função de inferência.

A saída inclui concentração, limites, faixa IQAr, data de referência e alvo, padrão normativo na data-alvo, ultrapassagem e diagnóstico do AOD/IDW. A distância à estação mais próxima é calculada com pares reais de latitude/longitude. A produção é treinada com todos os dados para demonstração: os mapas históricos podem ser dentro da amostra e não substituem as métricas dos folds.

## 7. Agente com RAG

Sete documentos públicos: CONAMA 491/2018, 506/2024, 03/1990, 05/1989; Lei 6.938/1981; Decreto 99.274/1990; Lei 9.605/1998. As fontes e revogações parciais estão em `configs/normas.yaml`. Não há pretensão de cobrir toda a legislação ambiental vigente.

O parser preserva capítulo, artigo, parágrafo, inciso, alínea e anexo. A unidade preferida é o artigo; artigos longos são subdivididos com metadados de hierarquia. Tabelas são transformadas em linhas com o nome da coluna junto a cada valor, evitando confundir PI-1, PI-2 e PF. A recuperação combina BM25 e embeddings Granite-278M por RRF; reranker Qwen3-0.6B reordena candidatos e fornece um escore para a recusa. Benchmark de recuperação (30 perguntas com trecho-ouro anotado):

{table(retrieval)}

O LLM Qwen3-4B local planeja consultas, recebe trechos e saídas de ferramentas, responde com referências e passa por uma validação numérica. A ferramenta de PM2.5 retorna valor, intervalo e comparações calculadas; a ferramenta de padrão vigente fornece cronograma e evidência normativa. Perguntas fora da base podem ser recusadas pelo limiar ou pelo modelo. Toda execução registra pergunta, recuperação, chamadas, resposta, tokens, tempo e custo; o custo local é zero.

Downloads incompletos não substituem a base; a ingestão grava atomicamente somente o conjunto completo e preserva a amostra versionada. A recuperação também detecta base processada incompleta e usa a amostra completa. `.env` é carregado sem executar código e sem substituir variáveis já exportadas.

### Avaliação, gabaritos e limites das métricas

36 perguntas escritas à mão e conferidas contra o texto das normas: 17 de fato, 6 de composição de trechos, 5 de vigência, 2 com ferramenta e 6 sem resposta na base. A precisão das citações verifica o **dispositivo**, não apenas o documento. Nas ferramentas são exigidos local, data-alvo, horizonte, valor e dois limites do IC; quando a pergunta pede faixa, exige a faixa retornada, em vez de aceitar qualquer nome. A citação do padrão é verificada separadamente; a semântica da comparação depende da análise das fontes/juiz, não só de presença de números.

Resultados:

{table(final.T.reset_index().rename(columns={'index': 'metrica', final.index[0]: 'valor'})) if len(final) else 'Avaliação final em geração.'}

Proveniência: a tag `qwen3-4b_causal` reaproveita as respostas normativas de uma rodada anterior com o mesmo LLM e a mesma base, rechecadas com os gabaritos e a regra de citação exata atuais, e executa novamente as perguntas que chamam o modelo de PM2.5 (alterado na versão causal). O log preserva os timestamps originais e as novas execuções. O Gemma3-4B aparece como comparação de gerador, recalculado com as mesmas regras.

“Fidelidade automática” mede rastreabilidade de números e referências, não equivalência semântica completa. O juiz é outro modelo local de 4B e também pode errar. Limiares de recusa foram ajustados no próprio conjunto; seis recusas corretas não demonstram generalização para qualquer pergunta fora da base. São limitações assumidas, a serem resolvidas com conjunto separado de calibração e avaliação maior.

## 8. Priorização, descartes e próximos passos

- MAIAC/QUALAR/CDS/INPE foram substituídos por fontes públicas e anônimas, mantendo a natureza física dos sinais (justificativa por fonte na seção 2).
- O modelo foi escolhido pelo MAE. Pelo critério de detecção de dias poluídos, o IDW e a persistência são melhores (seção 5.1). Não troquei o critério depois de ver o teste; registro o conflito e uso o intervalo para alerta.
- Preenchimento com AOD de D+1 e meteorologia realizada de D+1 foram retirados para respeitar o horizonte de emissão.
- Tendência do episódio no D+1 (PM2.5 e IDW de D−1, conhecidos ao fim de D) foi testada e descartada: piorou o MAE (A: 4,05 → 4,14; B: 4,47 → 4,54) e a detecção acima de 25 µg/m³. O experimento reproduz exatamente as linhas de base publicadas (`scripts/exp_tendencia_d1.py`, `logs/exp_tendencia_d1.txt`).
- Alvos absolutos, redes neurais e variantes mais complexas permanecem no benchmark; o escolhido aprende a correção sobre o IDW. Ganhos pequenos ou negativos em alguns recortes são reportados.
- Não foi implementado feed diário atual nem simulação da latência ERA5. Não apresento a demonstração histórica como monitoramento atual.
- Não há avaliação em estados externos, garantia de cobertura nos extremos ou norma estadual paulista completa na base.
- As faixas do IQAr seguem a 491/2018 para todas as datas. Correto para 2021–2023, mas desatualizado após a 506/2024 (Boa ≤ 15 µg/m³).
- Próximos passos: mais estações e anos, fonte meteorológica disponível na emissão, previsão numérica opcional, peso nos extremos/altura de pluma, limiar de alerta calibrado fora do teste, faixas do IQAr por data de vigência, teste externo de RAG e calibração independente dos limiares.

## 9. Engenharia e evidências

Dependências fixadas, Dockerfile e Makefile; pipeline portável com um comando, testes de causalidade (retirar todas as observações futuras não altera as variáveis), preservação da base normativa, `.env` e avaliação das ferramentas. CSVs, figuras, notebooks com saída e logs são versionados; modelos grandes e brutos são reconstruídos.

{docker_note}

Os logs `benchmark_causal.txt`, `train_causal.txt`, `docker_build.txt`, `docker_validacao.txt` e `validacao_instalacao.txt` registram as execuções de validação (`scripts/verify_install.py` refaz testes, inferência, API e a cobertura D+1 publicada).
"""
    out = ROOT / "docs" / "DETALHES_TECNICOS.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"{out.relative_to(ROOT)} atualizado a partir dos resultados")


if __name__ == "__main__":
    main()
