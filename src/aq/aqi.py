"""Air Quality Index (IQAr) bands and legal 24-hour PM2.5 standards in Brazil.

Concentration bands used by CETESB and by the Ministry of Environment's (MMA) technical
monitoring guide for 24 h PM2.5 (band names stay in Portuguese, as in the regulations):
Boa (good) 0-25 | Moderada >25-50 | Ruim >50-75 | Muito Ruim >75-125 | Péssima >125 µg/m³.

Legal 24 h PM2.5 standards (µg/m³):
  - CONAMA Res. 03/1990: did not regulate PM2.5
  - CONAMA Res. 491/2018: PI-1 = 60, PI-2 = 50, PI-3 = 37, PF = 25 (PI = interim target,
    PF = final standard). PI-1 applies from publication; later stages depend on control plans
  - CONAMA Res. 506/2024 (Official Gazette of 2024-07-09): repeals art. 4 and Annex I of 491
    and sets the schedule: PI-1 (60) until 2024-12-31 (art. 4, § 1), PI-2 (50) from
    2025-01-01 (§ 2), PI-3 (37) in 2033, PI-4 (25) in 2044. Checked against the text of the
    regulation (data/raw/normas/conama_506_2024.pdf); the agent cites the passage through
    the RAG module (src/rag).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import load_config

BANDS = load_config()["aqi"]["pm25_bands"]
LABELS = [b[2] for b in BANDS]


def band(pm25: float | np.ndarray):
    edges = [b[1] for b in BANDS[:-1]]
    idx = np.digitize(np.asarray(pm25, dtype=float), edges, right=True)
    if np.ndim(idx) == 0:
        return LABELS[int(idx)]
    return np.array(LABELS, dtype=object)[idx]


def legal_standard_24h(date) -> dict:
    """24 h PM2.5 standard in force on the given date, according to the CONAMA resolutions."""
    d = pd.Timestamp(date)
    if d < pd.Timestamp("2018-11-21"):
        return {"norma": "Resolução CONAMA 03/1990", "etapa": None, "valor": None,
                "obs": "MP2,5 não tinha padrão nacional antes da Res. CONAMA 491/2018"}
    if d < pd.Timestamp("2024-07-09"):
        return {"norma": "Resolução CONAMA 491/2018, art. 4º, § 1º e Anexo I", "etapa": "PI-1", "valor": 60.0}
    if d < pd.Timestamp("2025-01-01"):
        return {"norma": "Resolução CONAMA 506/2024, art. 4º, § 1º e Anexo I", "etapa": "PI-1", "valor": 60.0}
    if d < pd.Timestamp("2033-01-01"):
        return {"norma": "Resolução CONAMA 506/2024, art. 4º, § 2º e Anexo I", "etapa": "PI-2", "valor": 50.0}
    if d < pd.Timestamp("2044-01-01"):
        return {"norma": "Resolução CONAMA 506/2024, art. 4º, § 3º e Anexo I", "etapa": "PI-3", "valor": 37.0}
    return {"norma": "Resolução CONAMA 506/2024, art. 4º, § 4º e Anexo I", "etapa": "PI-4", "valor": 25.0}


# 24 h PM2.5, Annex I of each resolution (µg/m³), checked against the text (data/raw/normas)
STAGES_24H = {"Resolução CONAMA 491/2018": {"PI-1": 60, "PI-2": 50, "PI-3": 37, "PF": 25},
              "Resolução CONAMA 506/2024": {"PI-1": 60, "PI-2": 50, "PI-3": 37, "PI-4": 25, "PF": 15}}


def compare_stages(pm25: float, date) -> list[str]:
    """Precomputed comparison of the estimate with every stage of the standard in force on the date.
    4B LLMs get numeric comparisons wrong ("51.4 exceeds 60"); with the sentence ready-made, the
    answer only has to copy it. The text stays in Portuguese because the agent answers in Portuguese."""
    std = legal_standard_24h(date)
    if std["valor"] is None:
        return []
    norma = next(n for n in STAGES_24H if std["norma"].startswith(n))
    fmt = lambda x: f"{x:.1f}".replace(".", ",")
    return [f"{fmt(pm25)} µg/m³ {'ULTRAPASSA' if pm25 > v else 'NÃO ultrapassa'} o {etapa} de {v} µg/m³ "
            f"({norma}, Anexo I){' - etapa vigente na data' if etapa == std['etapa'] else ''}"
            for etapa, v in STAGES_24H[norma].items()]
