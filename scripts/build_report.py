"""Gera o relatório final diretamente dos CSVs, sem transcrever métricas à mão."""
from pathlib import Path

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
    rag = metric("rag_eval_resumo.csv")
    final = rag[rag.modelo.eq("qwen3-4b_causal")]
    retrieval = metric("rag_retrieval.csv")
    recortes = metric("benchmark_estimativa_recortes.csv")
    recortes = recortes[recortes.modelo.isin(["IDW (baseline)", "Krigagem ordinária", "LightGBM regularizado (offset IDW)"])]
    report = f"""# Relatório técnico

## 1. Resultado e escopo

Versão final de variáveis: **causal-spatial-aod-v2**. São {len(d):,} estação-dias de {d.location_id.nunique()} estações CETESB, entre {d.date.min().date()} e {d.date.max().date()}, no estado de São Paulo. Grade de 0,05° (~5,5 km).

Na validação de estações não vistas e período posterior, o LightGBM regularizado com offset do IDW obteve **MAE {best.MAE:.2f} µg/m³, RMSE {best.RMSE:.2f} µg/m³ e R² {best.R2:.3f}**. O ganho de MAE sobre o IDW foi **{best['ganho_MAE_vs_baseline_%']:.2f}%**. O intervalo nominal de 90% para a estimativa diária teve cobertura empírica de **{cov.cobertura_real:.2%}**. A cobertura de D+1 é medida separadamente na seção 5.

Não há feed atual: o produto é uma demonstração histórica. Uma emissão ao fim de D usa as agregações completas de D e produz a estimativa de D e a previsão de D+1. “Hoje/amanhã” no agente usam uma referência histórica explícita. A CLI recebe a data de referência; horizonte 1 significa data-alvo = referência + um dia.

O desafio recomenda 2019 em diante, mas permite recorte e fontes alternativos justificados. O arquivo público anônimo de PM2.5 utilizado limita o período e tem lacunas em aproximadamente metade dos dias. Priorizei as duas partes funcionais, validação espaço-temporal, incerteza medida e rastreabilidade das respostas.

## 2. Dados, fontes e reprodução

| Variável | Fonte utilizada | Justificativa |
|---|---|---|
| PM2.5 horário | CETESB via arquivo OpenAQ público no S3 | Alternativa anônima ao QUALAR, que exige login. Média diária só com pelo menos 18 h válidas |
| AOD | GOES-16 ABI L2 AODF | Alternativa pública ao MAIAC; oito varreduras diurnas por dia e controle de qualidade DQF |
| NO2 troposférico | Sentinel-5P TROPOMI, COG público MEEO/S3 | Sinal de tráfego/indústria e precursor de aerossol secundário; qa ≥ 0,75 |
| Fogo e FRP | GOES-16 ABI L2 FDCF | Alternativa ao INPE, indisponível no ambiente inicial |
| Meteorologia | ERA5, espelho público NCAR/S3 | Vento, umidade, chuva, temperatura, nuvens e camada limite, sem chave CDS |
| Uso do solo e relevo | ESA WorldCover 2021 e Copernicus DEM | Blocos públicos com leitura parcial |
| Limite estadual e municípios | IBGE espelhado no GitHub | Mesmo recorte territorial, sem dependência do acesso inicial a gov.br |

Os coletores estão em `src/aq/data/`; usam leituras parciais HTTP de NetCDF/HDF5 e COG. `python -m src.aq.pipeline --full` baixa, monta as bases, compara, treina e gera evidências. A alternativa rápida `python -m src.aq.pipeline` usa a amostra de aproximadamente 12 MB, sem baixar brutos. A amostra contém as bases de treino completas com variáveis e cubos de contexto em algumas janelas, suficientes para mapas, CLI e agente.

As datas são agregadas no dia local UTC−3 do período escolhido. AOD/NO2 usam célula de 0,05° e vizinhanças 3×3/9×9; ERA5 usa interpolação bilinear; estáticos usam 1 km e média de 5 km. O instante horário do alvo marca o fim da hora: 00:00 é atribuído ao dia anterior.

### Falhas de AOD e causalidade

Hierarquia exclusivamente espacial: célula observada (nível 0), média 3×3 (1), média 9×9 (2), ausente (4). Após essas etapas, NaN permanece como ausência aprendida pelo LightGBM, acompanhado pela flag. Não há preenchimento temporal e nenhuma observação de D+1 entra nas variáveis de D.

AOD observado na célula em **{d.aod.notna().mean():.1%}** das linhas; após preenchimento espacial, **{d.aod_filled.notna().mean():.1%}**. O AOD isolado é um sinal fraco na superfície: mede a coluna, influenciada por umidade, altura e transporte dos aerossóis. Por isso entram combinações físicas, NO2, fogo e estações vizinhas.

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

{table(b, ['modelo', 'n', 'MAE', 'RMSE', 'R2', 'ganho_MAE_vs_baseline_%', 'acerto_faixa_IQAr'])}

Recortes por tempo e distância à estação de referência mais próxima:

{table(recortes, ['recorte', 'grupo', 'modelo', 'n', 'MAE', 'estacoes'])}

Intervalos: quantis LightGBM em log, correção CQR por escores de calibração fora da estação (cross-fitting), comparados ao conformal do resíduo absoluto. A calibração usa somente o período anterior ao teste. Dependência espaço-temporal e cross-fitting impedem prometer a garantia exata do split conformal clássico; reporto a cobertura empírica.

{table(intervals)}

Cobertura condicional por faixa:

{table(metric('intervalos_por_faixa.csv'))}

SHAP explica a correção em log sobre o IDW, não diretamente µg/m³. `shap_importancia.csv`, `shap_exemplo_local.csv`, `shap_resumo.png` e `shap_dependencia.png` mostram importância global e exemplo local. O ajuste SHAP é do modelo final; ele é uma explicação descritiva, não uma avaliação adicional fora da amostra.

Episódios de fumaça são definidos pelo percentil 95 do índice de fogo a favor do vento. O recorte é descritivo, definido para análise do teste; não é um classificador operacional de fumaça.

{table(metric('episodios_extremos.csv'), ['regime', 'modelo', 'n', 'PM2.5_medio', 'MAE', 'vies'])}

As árvores tendem a encolher os extremos para a média; o T1 tem pouco histórico seco no treino. A fumaça pode estar acima da camada limite e o AOD pode faltar nos episódios densos. Esses fatores motivam as análises de `piores_dias.csv` e `episodios_serie.png`; são hipóteses físicas sobre os erros, não atribuições causais demonstradas.

## 5. Previsão real de D+1 em horizonte, sem meteorologia futura

Usa as variáveis observadas de D e D−1 e o calendário conhecido de D+1. Não é mais o experimento de perfect prognosis da versão inicial. A reanálise ERA5 continua sendo uma reconstrução histórica do dia D: sua latência operacional e possíveis revisões não são simuladas. O próximo passo de operação é substituir a fonte por observações/previsões disponíveis no horário de emissão e monitorar sua latência.

A: local com estação, persistência = PM2.5 de D. B: local sem estação, estações-alvo totalmente excluídas, persistência = IDW de D. São cenários distintos, reportados separadamente.

{table(fc, ['cenario', 'recorte', 'metodo', 'n', 'MAE', 'RMSE', 'R2', 'acerto_faixa_IQAr', 'ganho_MAE_vs_baseline_%'])}

**Cobertura do intervalo de D+1 medida no teste**, sem reutilizar a cobertura da estimativa de D:

Na revisão, a cobertura agregada foi **{d1_all.loc['A) local com estação', 'cobertura_real']:.2%} em A** e **{d1_all.loc['B) local sem estação', 'cobertura_real']:.2%} em B**, abaixo do nominal de 90%. Os intervalos são identificados pelo nível nominal; não prometem cobertura de 90% em todo período ou faixa. A queda na seca e nos valores altos exige cautela e mais dados de calibração representativos.

{table(d1_intervals)}

Em A, quantis e CQR são calibrados nas estações monitoradas no período de calibração. Em B, os escores vêm de cross-fitting interno entre estações de referência. O ponto e o intervalo usados na inferência pertencem ao mesmo cenário. `oof_previsao_d1.parquet`, gerado pelo treino, contém rótulos, previsões, limites e datas para auditoria das métricas.

## 6. Mapas e interfaces

`reports/figures/mapa_pm25_*` contém mapas diários de estimativa/previsão e da largura do intervalo de 90%. O comando portável gera dias típico, poluído e de fumaça; o mapa D+1 de 24/08/2021 é emitido em 23/08/2021. CLI, mapa, API e ferramenta do agente compartilham a mesma função de inferência.

A saída inclui concentração, limites, faixa IQAr, data de referência e alvo, padrão normativo na data-alvo, ultrapassagem e diagnóstico do AOD/IDW. A distância à estação mais próxima é calculada com pares reais de latitude/longitude. A produção é treinada com todos os dados para demonstração: os mapas históricos podem ser dentro da amostra e não substituem as métricas dos folds.

## 7. Agente com RAG

Sete documentos públicos: CONAMA 491/2018, 506/2024, 03/1990, 05/1989; Lei 6.938/1981; Decreto 99.274/1990; Lei 9.605/1998. As fontes e revogações parciais estão em `configs/normas.yaml`. Não há pretensão de cobrir toda a legislação ambiental vigente.

O parser preserva capítulo, artigo, parágrafo, inciso, alínea e anexo. A unidade preferida é o artigo; artigos longos são subdivididos com metadados de hierarquia. Tabelas são transformadas em linhas com o nome da coluna junto a cada valor, evitando confundir PI-1, PI-2 e PF. A recuperação combina BM25 e embeddings Granite-278M por RRF; reranker Qwen3-0.6B reordena candidatos e fornece um escore para a recusa. Resultados do benchmark existente, cuja base textual não mudou:

{table(retrieval)}

O LLM Qwen3-4B local planeja consultas, recebe trechos e saídas de ferramentas, responde com referências e passa por uma validação numérica. A ferramenta de PM2.5 retorna valor, intervalo e comparações calculadas; a ferramenta de padrão vigente fornece cronograma e evidência normativa. Perguntas fora da base podem ser recusadas pelo limiar ou pelo modelo. Toda execução registra pergunta, recuperação, chamadas, resposta, tokens, tempo e custo; o custo local é zero.

Downloads incompletos não substituem a base; a ingestão grava atomicamente somente o conjunto completo e preserva a amostra versionada. A recuperação também detecta base processada incompleta e usa a amostra completa. `.env` é carregado sem executar código e sem substituir variáveis já exportadas.

### Avaliação, gabaritos e limites das métricas

37 perguntas: fatos, composição de trechos, vigência, três ferramentas e seis sem resposta. A precisão das citações agora verifica o **dispositivo**, não apenas o documento. Nas ferramentas são exigidos local, data-alvo, horizonte, valor e dois limites do IC; quando a pergunta pede faixa, exige a faixa retornada, em vez de aceitar qualquer nome. A citação do padrão é verificada separadamente; a semântica da comparação depende da análise das fontes/juiz, não só de presença de números.

Resultados da revisão:

{table(final.T.reset_index().rename(columns={'index': 'metrica', final.index[0]: 'valor'})) if len(final) else 'Avaliação final em geração.'}

Proveniência: a tag `qwen3-4b_causal` reaproveita as 34 respostas normativas da avaliação anterior, com rechecagem dos gabaritos e citações exatas, e executa novamente as três perguntas que chamam o modelo alterado. O log preserva os timestamps originais e as novas execuções. Não representa uma rodada nova de 37 gerações nem um conjunto independente. As tags anteriores são históricas; suas métricas e previsões não representam o modelo causal atual.

“Fidelidade automática” mede rastreabilidade de números e referências, não equivalência semântica completa. O juiz é outro modelo local de 4B e também pode errar. Limiares de recusa foram ajustados no próprio conjunto; seis recusas corretas não demonstram generalização para qualquer pergunta fora da base. São limitações assumidas, a serem resolvidas com conjunto separado de calibração e avaliação maior.

## 8. Priorização, descartes e próximos passos

- MAIAC/QUALAR/CDS/INPE foram substituídos por fontes públicas acessíveis no ambiente inicial, mantendo a natureza física dos sinais.
- Preenchimento com AOD de D+1 e meteorologia realizada de D+1 foram retirados para respeitar o horizonte de emissão.
- Alvos absolutos, redes neurais e variantes mais complexas permanecem no benchmark; o escolhido aprende a correção sobre o IDW. Ganhos pequenos ou negativos em alguns recortes são reportados.
- Não foi implementado feed diário atual nem simulação da latência ERA5. Não apresento a demonstração histórica como monitoramento atual.
- Não há avaliação em estados externos, garantia de cobertura nos extremos ou norma estadual paulista completa na base.
- Próximos passos: mais estações e anos, fonte meteorológica disponível na emissão, previsão numérica opcional, peso nos extremos/altura de pluma, teste externo de RAG e calibração independente dos limiares.

## 9. Engenharia, evidências e uso de IA

Dependências fixadas, Dockerfile e Makefile; pipeline portável com um comando, testes de causalidade (retirar todas as observações futuras não altera as variáveis), preservação da base normativa, `.env` e avaliação das ferramentas. CSVs, figuras, notebooks com saída e logs são versionados; modelos grandes e brutos são reconstruídos.

Os logs `benchmark_causal.txt`, `train_causal.txt`, `docker_build.txt` e `validacao_entrega.txt` registram a revisão. [USO_DE_IA.md](USO_DE_IA.md) declara a implementação inicial com Claude Code e esta revisão com Codex. O candidato deve conseguir explicar e modificar o código na entrevista.

**Pendência externa à implementação:** gravar o vídeo não listado de até dez minutos e inserir o link no README antes do envio. O prazo de 72 horas conta do recebimento do desafio; o repositório não comprova essa data nem a entrega por e-mail.
"""
    (ROOT / "RELATORIO.md").write_text(report, encoding="utf-8")
    print("RELATORIO.md atualizado a partir dos resultados")


if __name__ == "__main__":
    main()
