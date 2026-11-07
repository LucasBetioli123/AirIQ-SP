"""Ferramentas do agente.

  buscar_normas(consulta, data)        -> trechos normativos com citação (RAG)
  estimar_pm25(local, data, horizonte) -> modelo de PM2.5 (estimativa/previsão + IC 90%)
  padrao_mp25_vigente(data)            -> etapa do padrão de MP2,5 vigente na data + cronograma (art. 4º)

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
        # Não exige dados observados de amanhã: somente o contexto até a emissão.
        ok = [d for d in obs_days if d in idx]
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


_AQ_WORDS = re.compile(r"\b(?:pm\s?2[.,]?5|mp\s?2[.,]?5|qualidade do ar|polui)", re.I)


def pm25_call_from_question(question: str) -> dict | None:
    """Reforço determinístico ao plano: se a pergunta fala de PM2.5/qualidade do ar num município de SP
    e numa data (ou hoje/amanhã) e o LLM não pediu a ferramenta, monta a chamada. Na avaliação, o
    Qwen3-4B deixou de chamar a ferramenta em 1 de 3 perguntas desse tipo."""
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
    """A pergunta é sobre QUAL etapa do padrão de MP2,5 vale numa data (ou quando entra em vigor)? Então o
    agente consulta o cronograma, em vez de deixar o LLM deduzir a etapa vigente: na avaliação, o Qwen3-4B
    respondeu o PF (25) para 2020, quando valia o PI-1 (60). Perguntas sobre o valor de uma etapa numa
    norma não disparam (o cronograma da 506 só distrairia)."""
    return bool(_STD_WORDS.search(question) and _WHEN_WORDS.search(question))


# início de cada etapa do MP2,5, conferido no art. 4º de cada resolução
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
    """Etapa do padrão nacional de MP2,5 (24 h) vigente na data (padrão: hoje, data real) e o cronograma da
    Resolução CONAMA 506/2024. Determinístico: a regra vem do art. 4º das resoluções."""
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


def trecho_por_id(chunk_id: str, mode: str | None = None) -> dict | None:
    from src.rag.retrieve import Hit
    c = next((c for c in get_retriever(mode or os.environ.get("RAG_MODE", "hybrid_rerank")).chunks
              if c["id"] == chunk_id), None)
    return _as_dict(Hit(c, 0.0)) if c else None


def base_legal_padrao(r: dict) -> list[str]:
    """Trechos em que padrao_mp25_vigente se apoia (art. 4º das resoluções), para a resposta poder citá-los."""
    ids = ["conama_506_2024:art4"]
    if r["padrao_24h_vigente_na_data"]["norma"].startswith("Resolução CONAMA 491/2018"):
        ids.insert(0, "conama_491_2018:art4")
    return ids


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
