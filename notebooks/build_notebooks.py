"""Generate the evidence notebooks (executed afterwards with `make notebooks`).

The notebooks reimplement nothing: they read the modules' outputs (metrics, figures, logs)
and present them with context, so the evidence always matches the code. CSV column names
and labels are in Portuguese (data keys); the narrative is in English.
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
    "md:# 01 - Data: from collection to the training table\n\nEvery source is public and anonymous (AWS/GCS Open Data). "
    "See `docs/TECHNICAL_DETAILS.md` for the source choices and their rationale.",
    SETUP,
    "from src.common.config import load_config, data_path\ncfg = load_config()\n"
    "st = pd.read_parquet('data/interim/stations.parquet') if Path('data/interim/stations.parquet').exists() else pd.read_parquet('data/sample/stations.parquet')\n"
    "print(len(st), 'CETESB stations with PM2.5'); st.head(40)",
    "df = pd.read_parquet('data/processed/station_days.parquet')\n"
    "print(df.shape); print(df['date'].min().date(), '->', df['date'].max().date())\n"
    "df.groupby('name').agg(days=('pm25','size'), pm25_mean=('pm25','mean'), pm25_p95=('pm25', lambda s: s.quantile(.95))).sort_values('days', ascending=False).round(1)",
    "md:## Cloud gaps in AOD and causal spatial gap-filling\nLevel 0 = cell with a retrieval on the day; 1 = 3x3 window; 2 = 9x9 window; 4 = no AOD (NaN handled by the model). No future observation is used.",
    "lvl = df['aod_fill_level'].value_counts(normalize=True).sort_index().rename('share'); display(lvl.to_frame())\n"
    "df.assign(month=df.date.dt.month).groupby('month')['aod'].apply(lambda s: s.notna().mean()).rename('AOD observed in the cell (share)').to_frame().T.round(2)",
    "md:## Physical consistency: correlations with measured PM2.5",
    "cols = ['aod','aod_filled','aod_dry_blh','no2_filled','blh_pm','log_vent','rh','tp','fire_up_400km','built_5km']\n"
    "df[cols + ['pm25']].corr(method='spearman')['pm25'].drop('pm25').sort_values().round(3).to_frame('Spearman with PM2.5')",
]

NB2 = [
    "md:# 02 - Model: benchmark, spatio-temporal validation, intervals, D+1, SHAP and maps\n\n"
    "**Main validation**: whole stations held out of training (GroupKFold, 5 folds) **and** a later period "
    "(splits T1: Jul–Dec 2021 and T2: Jul 2022–Apr 2023). Random validation appears only as an optimistic reference.",
    SETUP,
    "b = met('benchmark_estimativa.csv'); b[['modelo','n','MAE','RMSE','R2','ganho_MAE_vs_baseline_%','acerto_faixa_IQAr','MAE_std_folds']].round(3)",
    "fig('benchmark_mae.png'); fig('dispersao_idw_vs_lgbm.png')",
    "md:## Confidence interval (90%): real coverage on the test set",
    "display(met('intervalos.csv').round(3)); display(met('intervalos_por_faixa.csv').round(3)); fig('intervalos.png')",
    "md:## Next-day forecast vs. persistence",
    "display(met('previsao_d1.csv')[['cenario','metodo','n','MAE','RMSE','R2','acerto_faixa_IQAr','ganho_MAE_vs_baseline_%']].round(3)); fig('previsao_d1.png')",
    "md:The forecast is issued at the end of D with observations up to D and the calendar of D+1. Coverage of tomorrow's interval is measured separately, out of sample:",
    "met('intervalos_d1.csv').round(3)",
    "md:## IQAr band metrics beyond raw accuracy\n\n~90% of days are \"Boa\" (good): always predicting \"Boa\" already scores ~90%. "
    "Balanced accuracy, macro F1 and recall above 25 µg/m³ show whether polluted days are caught. "
    "The alert on the upper bound of the 90% interval is a fixed rule, not tuned on the test set.",
    "display(met('faixas_iqar.csv').round(3)); met('faixas_iqar_confusao.csv')",
    "md:## Explanation (SHAP)",
    "display(met('shap_importancia.csv').head(15).round(4)); fig('shap_resumo.png'); fig('shap_dependencia.png')",
    "display(Markdown('Local example (highest predicted PM2.5 in the test sample):')); met('shap_exemplo_local.csv')",
    "md:## Extreme episodes (smoke)",
    "display(met('episodios_extremos.csv').round(2)); display(met('piores_dias.csv').head(10)); fig('episodios_serie.png')",
    "md:## Daily maps (PM2.5 and uncertainty)",
    "for p in sorted((ROOT/'reports/figures').glob('mapa_pm25_*.png')): display(Markdown(f'`{p.name}`')); fig(p.name)",
    "md:## Interface: function/CLI/API for a coordinate and date (output keys in Portuguese)",
    "from src.aq.predict import predict_one\nprint(json.dumps(predict_one(-23.4969, -47.4451, '2021-08-24', 0), ensure_ascii=False, indent=1))\n"
    "print(json.dumps(predict_one(-23.4969, -47.4451, '2021-08-23', 1), ensure_ascii=False, indent=1))",
]

NB3 = [
    "md:# 03 - RAG agent: regulation base, retrieval, evaluation and logs\n\n"
    "The regulations are Brazilian federal law, so the knowledge base, the questions and the answers are in Portuguese.",
    SETUP,
    "import collections\nfrom src.rag.retrieve import load_chunks\nch = load_chunks(); print(len(ch), 'chunks')\n"
    "pd.Series(collections.Counter(c['sigla'] for c in ch)).rename('chunks').to_frame()",
    "md:Example chunk (unit = article, with citable provisions):",
    "c = next(c for c in ch if c['doc_id']=='conama_491_2018' and c['label'].startswith('art.')); print(json.dumps({k: c[k] for k in ['id','sigla','path','label','dispositivos','vigencia_inicio','vigencia_fim']}, ensure_ascii=False, indent=1)); print(c['text'][:1200])",
    "md:## Retrieval benchmark (recall@k, MRR)",
    "met('rag_retrieval.csv').round(3)",
    "md:Refusal threshold: best reranker score per question (answerable vs. not in the base)",
    "lim = met('rag_limiar_recusa.csv'); lim = lim[lim['modo']=='hybrid_rerank']\n"
    "lim.assign(unanswerable=lim['tipo'].eq('sem_resposta')).groupby('unanswerable')['top_rerank'].describe().round(3)",
    "md:## End-to-end evaluation (accuracy, citations, faithfulness, refusals, tool use)",
    "met('rag_eval_resumo.csv').round(3).T",
    "d = met('rag_eval_detalhe_qwen3-4b_causal.csv'); display(d.drop(columns=['resposta']).round(2))",
    "md:## A logged run (question, passages, tools, answer, latency, cost)",
    "runs = [json.loads(l) for l in open('logs/eval_runs_qwen3-4b_causal.jsonl', encoding='utf-8')]\nr = next(r for r in reversed(runs) if any(t['ferramenta']=='estimar_pm25' for t in r['ferramentas']))\n"
    "print(r['pergunta']); print(); print(r['resposta']); print()\n"
    "print(json.dumps({k: r[k] for k in ['tempo_total_s','tokens_entrada','tokens_saida','custo_estimado_usd','trechos','validacao']}, ensure_ascii=False, indent=1))",
]

if __name__ == "__main__":
    for name, cells in [("01_data.ipynb", NB1), ("02_model.ipynb", NB2), ("03_rag.ipynb", NB3)]:
        nbf.write(nb(cells), HERE / name)
        print("written", name)
