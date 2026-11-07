"""English display names for the labels stored in the metric CSVs.

The CSVs keep Portuguese column names and labels: they are data keys shared by the code, the
logs, the reports and the dashboard, so renaming them would break comparability with past
runs. Figures and the technical report translate them only for display.
"""
from __future__ import annotations

import pandas as pd

COLUMNS = {
    "modelo": "model", "ganho_MAE_vs_baseline_%": "MAE_gain_vs_baseline_%", "acerto_faixa_IQAr": "IQAr_band_accuracy",
    "recorte": "slice", "grupo": "group", "estacoes": "stations", "metodo": "method",
    "cobertura_nominal": "nominal_coverage", "cobertura_real": "real_coverage", "largura_media": "mean_width",
    "cobertura_pm25_>25": "coverage_pm25_>25", "faixa": "band", "cobertura_cqr": "cqr_coverage",
    "largura_cqr": "cqr_width", "PM2.5_medio": "mean_PM2.5", "vies": "bias", "cenario": "scenario",
    "tarefa": "task", "acerto_faixa": "band_accuracy", "acerto_sempre_Boa": "always_Good_accuracy",
    "acuracia_balanceada": "balanced_accuracy", "sensibilidade_>25": "recall_>25", "precisao_>25": "precision_>25",
    "medida": "measured", "latencia_ms": "latency_ms", "indexacao_s": "indexing_s",
    "metrica": "metric", "valor": "value",
}

# exact cell values
EXACT = {
    "todos": "all", "tfold": "time split", "faixa_dist": "distance band", "estimativa D": "estimate D",
    "Boa": "Good", "Moderada": "Moderate", "Ruim": "Poor", "Muito Ruim": "Very poor", "Péssima": "Hazardous",
    # RAG evaluation metrics
    "acerto": "accuracy", "acerto_parcial": "partial_accuracy", "cit_precisao": "citation_precision",
    "cit_recall": "citation_recall", "fidelidade_auto": "faithfulness_auto", "fidelidade_juiz": "faithfulness_judge",
    "recusa_correta (sem resposta)": "correct_refusal (unanswerable)", "recusa_indevida": "wrongful_refusal",
    "ferramenta_ok": "tool_ok", "tempo_medio_s": "mean_time_s", "tokens_medio": "mean_tokens",
    "acerto_fato": "accuracy_fact", "acerto_ferramenta": "accuracy_tool", "acerto_multi": "accuracy_multi",
    "acerto_temporal": "accuracy_temporal", "metodo_avaliacao": "eval_method", "n_respondidas": "n_answered",
    "n_fidelidade_juiz": "n_judged",
}

# substrings, applied in order (longest first)
SUBSTR = [
    ("Alerta: limite superior do IC 90% > 25", "Alert: upper bound of the 90% interval > 25"),
    ("A) local com estação", "A) site with station"), ("B) local sem estação", "B) site without station"),
    ("Persistência do IDW de hoje", "Persistence of today's IDW"),
    ("Persistência (amanhã = hoje)", "Persistence (tomorrow = today)"), ("Persistência", "Persistence"),
    ("Quantis LightGBM 5-95% (sem calibrar)", "LightGBM 5-95% quantiles (uncalibrated)"),
    ("CQR (quantis conformalizados)", "CQR (conformalized quantiles)"),
    ("Conformal do resíduo absoluto", "Absolute-residual conformal"),
    ("validação ALEATÓRIA [otimista]", "RANDOM validation [optimistic]"),
    ("fumaça (fogo a favor do vento ≥ p95)", "smoke (upwind fire ≥ p95)"),
    ("PM2.5 alto sem fogo (≥ p95)", "high PM2.5 without fire (≥ p95)"),
    ("Krigagem ordinária", "Ordinary kriging"), ("Climatologia mensal", "Monthly climatology"),
    ("regularizado", "regularized"), ("reg. forte", "strong reg."), ("sem satélite", "no satellite"),
    ("sem lag espacial", "no spatial lag"), ("alvo bruto", "raw target"),
    ("saída tanh [-0,7; 0,7]", "tanh output [-0.7, 0.7]"),
    ("sem diversificação", "without diversification"), ("Híbrido RRF", "Hybrid RRF"),
    ("Híbrido + reranker", "Hybrid + reranker"), ("Vetorial (", "Dense ("),
]


def en(text):
    """Translate one label (non-strings pass through)."""
    if not isinstance(text, str):
        return text
    if text in EXACT:
        return EXACT[text]
    for pt, eng in SUBSTR:
        text = text.replace(pt, eng)
    return text


def df_en(df):
    """Copy of a DataFrame with English column names and labels."""
    out = df.rename(columns=lambda c: COLUMNS.get(c, EXACT.get(c, c))).copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(en)
    out.index = out.index.map(en)
    out.index.name = COLUMNS.get(df.index.name, df.index.name)
    return out
