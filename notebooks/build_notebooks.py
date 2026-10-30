"""Gera os notebooks de evidência (executados depois com `make notebooks`).

Os notebooks não reimplementam nada: leem as saídas dos módulos (métricas, figuras,
logs) e mostram com contexto, para a evidência ficar consistente com o código.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
SETUP = """import os, sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import pandas as pd, numpy as np
from IPython.display import Image, display, Markdown
pd.set_option('display.max_columns', 40); pd.set_option('display.width', 200)
fig = lambda n: display(Image(filename=str(ROOT / 'reports/figures' / n)))
met = lambda n: pd.read_csv(ROOT / 'reports/metrics' / n)"""


def nb(cells):
    n = nbf.v4.new_notebook()
    n["cells"] = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
    n["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return n


NB1 = [
    "md:# 01 - Dados: da coleta à base de treino\n\nTodas as fontes são públicas e anônimas (AWS/GCS Open Data). "
    "Ver `RELATORIO.md` para as trocas de fonte e suas justificativas.",
    SETUP,
    "from src.common.config import load_config, data_path\ncfg = load_config()\n"
    "st = pd.read_parquet('data/interim/stations.parquet') if Path('data/interim/stations.parquet').exists() else pd.read_parquet('data/sample/stations.parquet')\n"
    "print(len(st), 'estações CETESB com PM2.5'); st.head(40)",
    "df = pd.read_parquet('data/processed/station_days.parquet')\n"
    "print(df.shape); print(df['date'].min().date(), '->', df['date'].max().date())\n"
    "df.groupby('name').agg(dias=('pm25','size'), pm25_medio=('pm25','mean'), pm25_p95=('pm25', lambda s: s.quantile(.95))).sort_values('dias', ascending=False).round(1)",
    "md:## Falhas de AOD por nuvem e preenchimento espacial causal\nNível 0 = célula com retrieval no dia; 1 = janela 3x3; 2 = janela 9x9; 4 = sem AOD (NaN tratado pelo modelo). Nenhuma observação futura é usada.",
    "lvl = df['aod_fill_level'].value_counts(normalize=True).sort_index().rename('fração'); display(lvl.to_frame())\n"
    "df.assign(mes=df.date.dt.month).groupby('mes')['aod'].apply(lambda s: s.notna().mean()).rename('AOD observado na célula (fração)').to_frame().T.round(2)",
    "md:## Coerência física: correlações com o PM2.5 medido",
    "cols = ['aod','aod_filled','aod_dry_blh','no2_filled','blh_pm','log_vent','rh','tp','fire_up_400km','built_5km']\n"
    "df[cols + ['pm25']].corr(method='spearman')['pm25'].drop('pm25').sort_values().round(3).to_frame('Spearman com PM2.5')",
]

NB2 = [
    "md:# 02 - Modelo: benchmark, validação espaço-temporal, intervalos, D+1, SHAP e mapas\n\n"
    "**Validação principal**: estações inteiras fora do treino (GroupKFold, 5 folds) **e** período posterior "
    "(cortes T1: jul–dez/2021 e T2: jul/2022–abr/2023). A validação aleatória aparece só como referência otimista.",
    SETUP,
    "b = met('benchmark_estimativa.csv'); b[['modelo','n','MAE','RMSE','R2','ganho_MAE_vs_baseline_%','acerto_faixa_IQAr','MAE_std_folds']].round(3)",
    "fig('benchmark_mae.png'); fig('dispersao_idw_vs_lgbm.png')",
    "md:## Intervalo de confiança (90%) - cobertura real no teste",
    "display(met('intervalos.csv').round(3)); display(met('intervalos_por_faixa.csv').round(3)); fig('intervalos.png')",
    "md:## Previsão do dia seguinte x persistência",
    "display(met('previsao_d1.csv')[['cenario','metodo','n','MAE','RMSE','R2','acerto_faixa_IQAr','ganho_MAE_vs_baseline_%']].round(3)); fig('previsao_d1.png')",
    "md:Previsão emitida ao fim de D com observações até D e calendário de D+1. Cobertura do intervalo de amanhã medida separadamente, fora da amostra:",
    "met('intervalos_d1.csv').round(3)",
    "md:## Faixa do IQAr além do acerto bruto\n\n~90% dos dias são \"Boa\": prever sempre \"Boa\" já acerta ~90%. "
    "Acurácia balanceada, F1 macro e sensibilidade acima de 25 µg/m³ mostram se os dias poluídos são detectados. "
    "O alerta pelo limite superior do IC 90% é uma regra fixa, sem ajuste no teste.",
    "display(met('faixas_iqar.csv').round(3)); met('faixas_iqar_confusao.csv')",
    "md:## Explicação (SHAP)",
    "display(met('shap_importancia.csv').head(15).round(4)); fig('shap_resumo.png'); fig('shap_dependencia.png')",
    "display(Markdown('Exemplo local (maior PM2.5 previsto na amostra de teste):')); met('shap_exemplo_local.csv')",
    "md:## Episódios extremos (fumaça)",
    "display(met('episodios_extremos.csv').round(2)); display(met('piores_dias.csv').head(10)); fig('episodios_serie.png')",
    "md:## Mapas diários (PM2.5 e incerteza)",
    "for p in sorted((ROOT/'reports/figures').glob('mapa_pm25_*.png')): display(Markdown(f'`{p.name}`')); fig(p.name)",
    "md:## Interface: função/CLI/API para coordenada e data",
    "from src.aq.predict import predict_one\nprint(json.dumps(predict_one(-22.9053, -47.0659, '2022-10-11', 0), ensure_ascii=False, indent=1))\n"
    "print(json.dumps(predict_one(-22.9053, -47.0659, '2022-10-11', 1), ensure_ascii=False, indent=1))",
]

NB3 = [
    "md:# 03 - Agente RAG: base normativa, recuperação, avaliação e logs",
    SETUP,
    "import collections\nfrom src.rag.retrieve import load_chunks\nch = load_chunks(); print(len(ch), 'chunks')\n"
    "pd.Series(collections.Counter(c['sigla'] for c in ch)).rename('chunks').to_frame()",
    "md:Exemplo de chunk (unidade = artigo, com dispositivos citáveis):",
    "c = next(c for c in ch if c['doc_id']=='conama_491_2018' and c['label'].startswith('art.')); print(json.dumps({k: c[k] for k in ['id','sigla','path','label','dispositivos','vigencia_inicio','vigencia_fim']}, ensure_ascii=False, indent=1)); print(c['text'][:1200])",
    "md:## Benchmark de recuperação (recall@k, MRR)",
    "met('rag_retrieval.csv').round(3)",
    "md:Limiar de recusa: maior escore do reranker por pergunta (respondíveis x sem resposta na base)",
    "lim = met('rag_limiar_recusa.csv'); lim = lim[lim['modo']=='hybrid_rerank']\n"
    "lim.assign(sem_resposta=lim['tipo'].eq('sem_resposta')).groupby('sem_resposta')['top_rerank'].describe().round(3)",
    "md:## Avaliação ponta a ponta (acerto, citação, fidelidade, recusa, ferramenta)",
    "met('rag_eval_resumo.csv').round(3).T",
    "d = met('rag_eval_detalhe_qwen3-4b_causal.csv'); display(d.drop(columns=['resposta']).round(2))",
    "md:## Exemplo de execução registrada (pergunta, trechos, ferramentas, resposta, tempo, custo)",
    "runs = [json.loads(l) for l in open('logs/eval_runs_qwen3-4b_causal.jsonl', encoding='utf-8')]\nr = next(r for r in reversed(runs) if any(t['ferramenta']=='estimar_pm25' for t in r['ferramentas']))\n"
    "print(r['pergunta']); print(); print(r['resposta']); print()\n"
    "print(json.dumps({k: r[k] for k in ['tempo_total_s','tokens_entrada','tokens_saida','custo_estimado_usd','trechos','validacao']}, ensure_ascii=False, indent=1))",
]

if __name__ == "__main__":
    for name, cells in [("01_dados.ipynb", NB1), ("02_modelo.ipynb", NB2), ("03_rag.ipynb", NB3)]:
        nbf.write(nb(cells), HERE / name)
        print("escrito", name)
