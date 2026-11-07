"""Agent tools (names and outputs in Portuguese: the agent works on Brazilian law).

  buscar_normas(consulta, data)        -> regulation passages with citations (RAG)
  estimar_pm25(local, data, horizonte) -> PM2.5 model (estimate/forecast + 90% interval)
  padrao_mp25_vigente(data)            -> PM2.5 standard stage in force on the date + schedule (art. 4)

Relative dates ("hoje" = today, "amanhã" = tomorrow) are resolved against the system's
REFERENCE DATE. As there is no real-time feed (the public data used end in Apr 2023), the
default is the last day with complete data. Override with AGENT_TODAY=yyyy-mm-dd.
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
        # Does not require tomorrow's observations: only the context up to issue time.
        ok = [d for d in obs_days if d in idx]
        return max(ok)
    except Exception:
        return pd.Timestamp.today().normalize()


def _strip(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


def resolve_date(text: str | None, horizon: int = 0) -> tuple[pd.Timestamp, int]:
    """Return (reference date, horizon). 'amanhã' (tomorrow) = D+1 forecast issued today."""
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


_AQ_WORDS = re.compile(r"\b(?:pm\s?2[.,]?5|mp\s?2[.,]?5|qualidade do ar|polui)", re.I)


def pm25_call_from_question(question: str) -> dict | None:
    """Deterministic backstop for the plan: if the question is about PM2.5/air quality in a São Paulo
    municipality on a date (or today/tomorrow) and the LLM did not request the tool, build the call.
    In evaluation, Qwen3-4B skipped the tool on 1 of 3 such questions."""
    if not _AQ_WORDS.search(question):
        return None
    from src.aq.data.geo import municipios_sp

    q = _strip(question)
    q = re.sub(r"estado de sao paulo", " ", q)
    names = sorted(municipios_sp()["nome"], key=len, reverse=True)
    local = next((n for n in names if re.search(rf"\b{re.escape(_strip(n))}\b", q)), None)
    if local is None:
        return None
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", question) or re.search(r"\b(\d{2}/\d{2}/\d{4})\b", question)
    if m:
        return {"local": local, "data": m.group(1), "horizonte": 0}
    if "amanha" in q:
        return {"local": local, "data": "amanhã", "horizonte": 1}
    if "hoje" in q or "agora" in q:
        return {"local": local, "data": "hoje", "horizonte": 0}
    return None


_STD_WORDS = re.compile(r"\b(?:mp|pm)\s?2[.,]?5\b|\bPI-[1-4]\b", re.I)
_WHEN_WORDS = re.compile(r"\b(?:em vigor|vigente|vigora|vale|valia|aplicad[oa]|desde quando|quando entra|hoje|atual|"
                         r"(?:em|no ano de) (?:19|20)\d{2})\b", re.I)


def padrao_call_from_question(question: str) -> bool:
    """Is the question about WHICH PM2.5 standard stage applies on a date (or when one takes effect)? Then
    the agent looks up the schedule instead of letting the LLM infer the stage in force: in evaluation,
    Qwen3-4B answered PF (25) for 2020, when PI-1 (60) applied. Questions about the value of a stage in
    a given regulation do not trigger it (the 506 schedule would only distract)."""
    return bool(_STD_WORDS.search(question) and _WHEN_WORDS.search(question))


# start of each PM2.5 stage, checked against art. 4 of each resolution
_STAGE_SINCE = {("Resolução CONAMA 491/2018", "PI-1"): "2018-11-21", ("Resolução CONAMA 506/2024", "PI-1"): "2024-07-09",
                ("Resolução CONAMA 506/2024", "PI-2"): "2025-01-01", ("Resolução CONAMA 506/2024", "PI-3"): "2033-01-01",
                ("Resolução CONAMA 506/2024", "PI-4"): "2044-01-01"}
CRONOGRAMA_506 = {"PI-1": "60 µg/m³, vigora até 2024-12-31 (art. 4º, § 1º)",
                  "PI-2": "50 µg/m³, em vigor a partir de 2025-01-01 (art. 4º, § 2º)",
                  "PI-3": "37 µg/m³, em vigor a partir de 2033-01-01 (art. 4º, § 3º)",
                  "PI-4": "25 µg/m³, em vigor a partir de 2044-01-01, data que pode ser antecipada ou prorrogada "
                          "uma única vez por até quatro anos (art. 4º, § 4º)",
                  "PF": "15 µg/m³, em data a ser definida em resolução do Conama (art. 4º, § 5º)"}


def padrao_mp25_vigente(data: str | None = None) -> dict:
    """Stage of the national 24 h PM2.5 standard in force on the date (default: today, the real date) and
    the CONAMA 506/2024 schedule. Deterministic: the rule comes from art. 4 of the resolutions."""
    d = pd.Timestamp(data) if data else pd.Timestamp.today().normalize()
    from src.aq.aqi import legal_standard_24h
    std = legal_standard_24h(d)
    norma = std["norma"].split(",")[0]
    out = {"data": str(d.date()), "padrao_24h_vigente_na_data": std,
           "vigente_desde": _STAGE_SINCE.get((norma, std["etapa"])),
           "cronograma_mp25_24h_conama_506_2024": CRONOGRAMA_506}
    if norma == "Resolução CONAMA 491/2018":
        out["obs"] = ("na 491/2018, PI-2, PI-3 e PF seriam adotados de forma subsequente, conforme os planos "
                      "estaduais (art. 4º, § 3º); nacionalmente valia o PI-1")
    return out


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
    from src.aq.aqi import compare_stages
    r["local"] = nome
    r["comparacao_com_padroes"] = compare_stages(r["pm25_ug_m3"], r["data_alvo"])
    r["data_referencia_sistema"] = str(reference_today().date())
    return r


@functools.lru_cache(maxsize=4)
def get_retriever(mode: str | None = None) -> Retriever:
    return Retriever(mode or os.environ.get("RAG_MODE", "hybrid_rerank"))


def nota_vigencia(c: dict) -> str | None:
    """Short note on the passage's validity, so the LLM never presents a repealed rule as current."""
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


def trecho_por_id(chunk_id: str, mode: str | None = None) -> dict | None:
    from src.rag.retrieve import Hit
    c = next((c for c in get_retriever(mode or os.environ.get("RAG_MODE", "hybrid_rerank")).chunks
              if c["id"] == chunk_id), None)
    return _as_dict(Hit(c, 0.0)) if c else None


def base_legal_padrao(r: dict) -> list[str]:
    """Passages that padrao_mp25_vigente relies on (art. 4 of the resolutions), so the answer can cite them."""
    ids = ["conama_506_2024:art4"]
    if r["padrao_24h_vigente_na_data"]["norma"].startswith("Resolução CONAMA 491/2018"):
        ids.insert(0, "conama_491_2018:art4")
    return ids


def buscar_normas(consulta: str, data: str | None = None, k: int = 5, mode: str | None = None) -> list[dict]:
    mode = mode or os.environ.get("RAG_MODE", "hybrid_rerank")
    return [_as_dict(h) for h in get_retriever(mode).search(consulta, k=k, date=data, mode=mode)]


def buscar_normas_varias(consultas: list[str], pergunta: str, data: str | None = None, k: int = 4,
                         mode: str | None = None) -> tuple[list[dict], dict[str, list[str]]]:
    """Several queries from the plan -> union of candidates (interleaving the rankings) -> reranker ONCE,
    against the original question. Returns the passages and, for the log, the ids each query found."""
    from src.rag.retrieve import rerank

    mode = mode or os.environ.get("RAG_MODE", "hybrid_rerank")
    base = mode.removesuffix("_rerank")
    ret = get_retriever(mode)
    per_query = {q: ret.search(q, k=k, date=data, mode=base) for q in consultas}
    union, seen = [], set()
    for rank in range(k):  # interleave: 1st of each query, then 2nd of each...
        for hs in per_query.values():
            if rank < len(hs) and hs[rank].chunk["id"] not in seen:
                seen.add(hs[rank].chunk["id"])
                union.append(hs[rank])
    if mode.endswith("_rerank"):
        union = rerank(pergunta, union, ret.rerank_top)
    return [_as_dict(h) for h in union], {q: [h.chunk["id"] for h in hs] for q, hs in per_query.items()}
