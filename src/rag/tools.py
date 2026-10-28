"""Ferramentas do agente.

  buscar_normas(consulta, data)        -> trechos normativos com citação (RAG)
  estimar_pm25(local, data, horizonte) -> modelo da Parte 1 (estimativa/previsão + IC 90%)

Datas relativas ("hoje", "amanhã") são resolvidas contra a DATA DE REFERÊNCIA do
sistema. Como não há feed em tempo real (os dados públicos usados vão até abr/2023),
o padrão é o último dia com dados completos. Dá para mudar com AGENT_TODAY=aaaa-mm-dd.
"""
from __future__ import annotations

import functools
import os
import re
import unicodedata

import pandas as pd

from src.rag.retrieve import Retriever


@functools.lru_cache(maxsize=1)
def reference_today() -> pd.Timestamp:
    if os.environ.get("AGENT_TODAY"):
        return pd.Timestamp(os.environ["AGENT_TODAY"])
    try:
        from src.aq.predict import available_index, station_obs
        idx = available_index()
        obs_days = pd.DatetimeIndex(station_obs()["date"].unique())
        # último dia com medições nas estações (para o IDW) e com contexto para D+1
        ok = [d for d in obs_days if d in idx and d + pd.Timedelta(days=1) in idx]
        return max(ok)
    except Exception:
        return pd.Timestamp.today().normalize()


def _strip(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


def resolve_date(text: str | None, horizon: int = 0) -> tuple[pd.Timestamp, int]:
    """Devolve (data de referência, horizonte). 'amanhã' = previsão D+1 feita hoje."""
    t = _strip(text or "hoje")
    today = reference_today()
    if t in ("hoje", "agora", "atual", ""):
        return today, horizon
    if t in ("amanha", "amanha (d+1)", "d+1"):
        return today, 1
    if t == "ontem":
        return today - pd.Timedelta(days=1), 0
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", t) or re.search(r"(\d{2})/(\d{2})/(\d{4})", t)
    if m:
        g = m.groups()
        d = pd.Timestamp(f"{g[0]}-{g[1]}-{g[2]}") if len(g[0]) == 4 else pd.Timestamp(f"{g[2]}-{g[1]}-{g[0]}")
        return (d - pd.Timedelta(days=1), 1) if horizon == 1 else (d, 0)
    return today, horizon


def estimar_pm25(local: str, data: str | None = None, horizonte: int = 0) -> dict:
    from src.aq.data.geo import geocode
    from src.aq.predict import DataUnavailable, predict_one

    m = re.match(r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", local or "")
    if m:
        lat, lon, nome = float(m.group(1)), float(m.group(2)), local
    else:
        g = geocode(local or "")
        if g is None:
            return {"erro": f"município '{local}' não encontrado no estado de SP"}
        lat, lon, nome = g["lat"], g["lon"], g["municipio"]
    ref, h = resolve_date(data, horizonte)
    try:
        r = predict_one(lat, lon, str(ref.date()), h)
    except DataUnavailable as e:
        return {"erro": str(e)}
    r["local"] = nome
    r["data_referencia_sistema"] = str(reference_today().date())
    return r


@functools.lru_cache(maxsize=4)
def get_retriever(mode: str | None = None) -> Retriever:
    return Retriever(mode or os.environ.get("RAG_MODE", "hybrid_rerank"))


def nota_vigencia(c: dict) -> str | None:
    """Texto curto sobre a vigência do trecho, para o LLM não apresentar regra revogada como atual."""
    fim, por = c.get("vigencia_fim"), (c.get("meta") or {}).get("revogado_por")
    if not fim:
        return None
    desde = (pd.Timestamp(fim) + pd.Timedelta(days=1)).date()
    return (f"REVOGADO a partir de {desde}" + (f" por {por}" if por else "") +
            f"; valia de {c.get('vigencia_inicio')} a {fim}")


def _as_dict(h) -> dict:
    return {"id": h.chunk["id"], "citacao": f"{h.chunk['sigla']}, {h.chunk['label']}", "doc_id": h.chunk["doc_id"],
            "dispositivos": h.chunk["dispositivos"], "texto": h.chunk["text"], "url": h.chunk.get("url"),
            "vigencia": nota_vigencia(h.chunk), "score": round(h.score, 4), "rerank": h.rerank, "cos": h.cos,
            "rank_bm25": h.rank_bm25, "rank_dense": h.rank_dense}


def buscar_normas(consulta: str, data: str | None = None, k: int = 5, mode: str | None = None) -> list[dict]:
    mode = mode or os.environ.get("RAG_MODE", "hybrid_rerank")
    return [_as_dict(h) for h in get_retriever(mode).search(consulta, k=k, date=data, mode=mode)]


def buscar_normas_varias(consultas: list[str], pergunta: str, data: str | None = None, k: int = 4,
                         mode: str | None = None) -> tuple[list[dict], dict[str, list[str]]]:
    """Várias consultas do plano -> união dos candidatos (intercalando os rankings) -> reranker UMA vez,
    contra a pergunta original. Devolve os trechos e, para o log, os ids que cada consulta trouxe."""
    from src.rag.retrieve import rerank

    mode = mode or os.environ.get("RAG_MODE", "hybrid_rerank")
    base = mode.removesuffix("_rerank")
    ret = get_retriever(mode)
    per_query = {q: ret.search(q, k=k, date=data, mode=base) for q in consultas}
    union, seen = [], set()
    for rank in range(k):  # intercala: 1º de cada consulta, depois 2º de cada...
        for hs in per_query.values():
            if rank < len(hs) and hs[rank].chunk["id"] not in seen:
                seen.add(hs[rank].chunk["id"])
                union.append(hs[rank])
    if mode.endswith("_rerank"):
        union = rerank(pergunta, union, ret.rerank_top)
    return [_as_dict(h) for h in union], {q: [h.chunk["id"] for h in hs] for q, hs in per_query.items()}
