"""Faixas do Índice de Qualidade do Ar (IQAr) e padrões legais para MP2,5 (média de 24 h).

Faixas de concentração usadas pela CETESB e pelo Guia Técnico de Monitoramento do MMA
para MP2,5 em 24 h: Boa 0-25 | Moderada >25-50 | Ruim >50-75 | Muito Ruim >75-125 |
Péssima >125 µg/m³.

Padrões legais de MP2,5 em 24 h (µg/m³):
  - Res. CONAMA 03/1990: não regulava MP2,5
  - Res. CONAMA 491/2018: PI-1 = 60, PI-2 = 50, PI-3 = 37, PF = 25. A primeira etapa
    (PI-1) vigora desde a publicação, e as seguintes dependem de planos de controle
  - Res. CONAMA 506/2024: substitui o cronograma, com PI-1 até 31/12/2024 e PI-2
    (50 µg/m³) a partir de 1/1/2025. Os valores são conferidos no texto da norma
    pelo agente (Parte 2); aqui ficam como referência para a ferramenta.
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
    """Padrão de MP2,5 (24 h) vigente na data, segundo as resoluções CONAMA."""
    d = pd.Timestamp(date)
    if d < pd.Timestamp("2018-11-21"):
        return {"norma": "Resolução CONAMA 03/1990", "etapa": None, "valor": None,
                "obs": "MP2,5 não tinha padrão nacional antes da Res. CONAMA 491/2018"}
    if d < pd.Timestamp("2025-01-01"):
        return {"norma": "Resolução CONAMA 491/2018 (e 506/2024)", "etapa": "PI-1", "valor": 60.0}
    return {"norma": "Resolução CONAMA 506/2024", "etapa": "PI-2", "valor": 50.0}
