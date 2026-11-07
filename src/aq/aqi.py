"""Faixas do Índice de Qualidade do Ar (IQAr) e padrões legais para MP2,5 (média de 24 h).

Faixas de concentração usadas pela CETESB e pelo Guia Técnico de Monitoramento do MMA
para MP2,5 em 24 h: Boa 0-25 | Moderada >25-50 | Ruim >50-75 | Muito Ruim >75-125 |
Péssima >125 µg/m³.

Padrões legais de MP2,5 em 24 h (µg/m³):
  - Res. CONAMA 03/1990: não regulava MP2,5
  - Res. CONAMA 491/2018: PI-1 = 60, PI-2 = 50, PI-3 = 37, PF = 25. A primeira etapa
    (PI-1) vigora desde a publicação, e as seguintes dependem de planos de controle
  - Res. CONAMA 506/2024 (DOU de 9/7/2024): revoga o art. 4º e o Anexo I da 491 e fixa o
    cronograma. PI-1 (60) até 31/12/2024 (art. 4º, § 1º), PI-2 (50) a partir de 1/1/2025
    (§ 2º), PI-3 (37) em 2033, PI-4 (25) em 2044. Conferido no texto da norma
    (data/raw/normas/conama_506_2024.pdf); o agente cita o trecho pelo módulo RAG (src/rag).
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
    if d < pd.Timestamp("2024-07-09"):
        return {"norma": "Resolução CONAMA 491/2018, art. 4º, § 1º e Anexo I", "etapa": "PI-1", "valor": 60.0}
    if d < pd.Timestamp("2025-01-01"):
        return {"norma": "Resolução CONAMA 506/2024, art. 4º, § 1º e Anexo I", "etapa": "PI-1", "valor": 60.0}
    if d < pd.Timestamp("2033-01-01"):
        return {"norma": "Resolução CONAMA 506/2024, art. 4º, § 2º e Anexo I", "etapa": "PI-2", "valor": 50.0}
    if d < pd.Timestamp("2044-01-01"):
        return {"norma": "Resolução CONAMA 506/2024, art. 4º, § 3º e Anexo I", "etapa": "PI-3", "valor": 37.0}
    return {"norma": "Resolução CONAMA 506/2024, art. 4º, § 4º e Anexo I", "etapa": "PI-4", "valor": 25.0}


# MP2,5 em 24 h, Anexo I de cada resolução (µg/m³), conferido no texto (data/raw/normas)
STAGES_24H = {"Resolução CONAMA 491/2018": {"PI-1": 60, "PI-2": 50, "PI-3": 37, "PF": 25},
              "Resolução CONAMA 506/2024": {"PI-1": 60, "PI-2": 50, "PI-3": 37, "PI-4": 25, "PF": 15}}


def compare_stages(pm25: float, date) -> list[str]:
    """Comparação já calculada da estimativa com todas as etapas da norma vigente na data. LLMs de 4B
    erram comparação numérica ("51,4 ultrapassa 60"); com o texto pronto, a resposta só copia."""
    std = legal_standard_24h(date)
    if std["valor"] is None:
        return []
    norma = next(n for n in STAGES_24H if std["norma"].startswith(n))
    fmt = lambda x: f"{x:.1f}".replace(".", ",")
    return [f"{fmt(pm25)} µg/m³ {'ULTRAPASSA' if pm25 > v else 'NÃO ultrapassa'} o {etapa} de {v} µg/m³ "
            f"({norma}, Anexo I){' - etapa vigente na data' if etapa == std['etapa'] else ''}"
            for etapa, v in STAGES_24H[norma].items()]
