"""Recuperação de trechos normativos: BM25, vetorial, híbrida (RRF) e híbrida + reranker.

Por que híbrida: o texto normativo mistura vocabulário exato ("MP2,5", "PI-1",
"art. 6º", números de resolução), em que o BM25 é forte, com paráfrases ("poluição
por fumaça", "quem fiscaliza"), em que o vetorial é forte. O RRF (Reciprocal Rank
Fusion) combina os rankings sem precisar calibrar escalas de escore. O reranker
(cross-encoder) reordena só os ~20 primeiros, o que custa pouco em CPU. A escolha
final sai do benchmark (eval/bench_retrieval.py).

Filtro temporal: por padrão, recupera só o que está vigente na data de referência
(hoje ou a data da pergunta). Uma norma revogada só entra se a pergunta a citar
explicitamente (ex.: "Resolução CONAMA 03/1990") ou se a data pedida cair na sua vigência.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from rank_bm25 import BM25Okapi

from src.common.config import data_path

_STEM = SnowballStemmer("portuguese")
STOP = set("""a o e é de do da dos das em no na nos nas um uma uns umas por para pelo pela pelos pelas com
como que se ao aos à às ou os as seu sua seus suas este esta estes estas esse essa isso isto ser são foi
qual quais quando onde quem sobre entre até mais menos não sim já também lhe lhes nesta neste desta deste
dessa desse""".split())
SYN = [(r"\b(?:pm|mp)\s?2[,.]?5\b", " mp25 "), (r"\b(?:pm|mp)\s?10\b", " mp10 "),
       (r"material particulado fino", " mp25 "), (r"\bno2\b|di[oó]xido de nitrog[eê]nio", " no2 "),
       (r"\bo3\b|oz[oô]nio", " o3 "), (r"\bso2\b|di[oó]xido de enxofre", " so2 "),
       (r"µg/m³|ug/m3|μg/m³", " ugm3 "), (r"\bpi\s?-\s?(\d)", r" pi\1 "), (r"\bpf\b", " pf ")]


def _strip(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def tokenize(text: str) -> list[str]:
    t = text.lower()
    for pat, rep in SYN:
        t = re.sub(pat, rep, t)
    t = _strip(t)
    toks = re.findall(r"[a-z0-9]+", t)
    return [_STEM.stem(w) if not w.isdigit() else w for w in toks if w not in STOP and len(w) > 1]


def load_chunks() -> list[dict]:
    p = data_path("processed", "rag", "chunks.jsonl")
    if not p.exists():
        p = data_path("sample", "rag_chunks.jsonl")
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def index_text(c: dict) -> str:
    head = f"{c['sigla']} - {c['path'] + ' - ' if c['path'] else ''}{c['label']}"
    return f"{head}\n{c['text']}"


# menções explícitas a uma norma: "491/2018", "nº 491", "03/1990", "6.938", e também sem o ano
# ("CONAMA 491", "Resolução 506", "Lei 9605"), que é como a pergunta costuma vir
def mentioned_docs(query: str, chunks: list[dict]) -> set[str]:
    q = _strip(query.lower())  # sem acento; "nº" vira "no"
    out = set()
    for did in {c["doc_id"] for c in chunks}:
        m = re.match(r"([a-z]+)_(\d+)_(\d{4})", did)
        if not m:
            continue
        tipo, num, year = m.group(1), str(int(m.group(2))), m.group(3)
        dotted = rf"{num[:-3]}\.?{num[-3:]}" if len(num) > 3 else num
        prefix = r"(?:conama|resolucao)\s*(?:conama\s*)?" if tipo == "conama" else rf"{tipo}\s*"
        pats = [rf"\b0*{num}\s*/\s*{year}\b", rf"\b0*{num}\s*/\s*{year[2:]}\b", rf"n[oº°]?\s*0*{num}\b.*{year}",
                rf"\b{prefix}(?:n[o.]?\s*)?0*{dotted}\b(?![.,/]?\d)"]
        if len(num) > 3:
            pats.append(rf"\b{num[:-3]}\.{num[-3:]}\b")
        if any(re.search(p, q) for p in pats):
            out.add(did)
    return out


def diversify(hits: list, max_per_group: int = 2) -> list:
    """No máximo `max_per_group` trechos do mesmo dispositivo (mesma tabela ou partes do mesmo artigo)
    no topo; o excedente desce para o fim. Sem isso, as 15 linhas do Anexo I (que repetem título e
    cabeçalho) tomam o top-k inteiro e o segundo trecho de uma pergunta composta fica de fora."""
    head, tail, count = [], [], {}
    for h in hits:
        g = (h.chunk["doc_id"], h.chunk["label"].split(" (")[0])
        count[g] = count.get(g, 0) + 1
        (head if count[g] <= max_per_group else tail).append(h)
    return head + tail


def vigente(c: dict, date: pd.Timestamp) -> bool:
    ini = pd.Timestamp(c["vigencia_inicio"]) if c.get("vigencia_inicio") else pd.Timestamp.min
    fim = pd.Timestamp(c["vigencia_fim"]) if c.get("vigencia_fim") else pd.Timestamp.max
    return ini <= date <= fim and not c.get("revogado")


@dataclass
class Hit:
    chunk: dict
    score: float
    rank_bm25: int | None = None
    rank_dense: int | None = None
    rerank: float | None = None
    cos: float | None = None       # similaridade de cosseno (vetorial), usada no limiar de recusa


class Retriever:
    def __init__(self, mode: str = "hybrid", emb_path: str | None = None, rrf_k: int = 60, rerank_top: int | None = None):
        self.mode = mode
        self.chunks = load_chunks()
        self.texts = [index_text(c) for c in self.chunks]
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts], k1=1.2, b=0.6)
        self.rrf_k = rrf_k
        self.rerank_top = rerank_top or int(os.environ.get("RAG_RERANK_TOP", "8"))
        self.max_per_group = int(os.environ.get("RAG_MAX_PER_GROUP", "2"))  # 0 desliga a diversificação
        self.emb = None
        if mode in ("dense", "hybrid", "hybrid_rerank", "dense_rerank"):
            from src.rag.embed import get_embedder
            self.emb = get_embedder(emb_path)
            self.doc_vecs = self.emb.documents(self.texts, cache_key="chunks")

    def _allowed(self, query: str, date) -> np.ndarray:
        d = pd.Timestamp(date) if date is not None else pd.Timestamp.today().normalize()
        explicit = mentioned_docs(query, self.chunks)
        # vigente na data, OU norma citada explicitamente (mesmo que já revogada como um todo)
        return np.array([vigente(c, d) or (c["doc_id"] in explicit and not c.get("revogado"))
                         for c in self.chunks])

    def search(self, query: str, k: int = 5, date=None, mode: str | None = None) -> list[Hit]:
        mode = mode or self.mode
        allowed = self._allowed(query, date)
        n = len(self.chunks)
        bm = self.bm25.get_scores(tokenize(query))
        bm = np.where(allowed, bm, -np.inf)
        r_bm = np.empty(n, int)
        r_bm[np.argsort(-bm)] = np.arange(n)
        cos = None
        if self.emb is not None and mode != "bm25":
            qv = self.emb.queries([query])[0]
            cos = self.doc_vecs @ qv
            dn = np.where(allowed, cos, -np.inf)
            r_dn = np.empty(n, int)
            r_dn[np.argsort(-dn)] = np.arange(n)
        if mode == "bm25":
            score = bm
        elif mode in ("dense", "dense_rerank"):
            score = dn
        else:  # RRF
            score = np.where(allowed, 1 / (self.rrf_k + r_bm + 1) + 1 / (self.rrf_k + r_dn + 1), -np.inf)
        order = [i for i in np.argsort(-score) if np.isfinite(score[i])]
        hits = [Hit(self.chunks[i], float(score[i]), int(r_bm[i]) + 1,
                    int(r_dn[i]) + 1 if cos is not None else None,
                    cos=float(cos[i]) if cos is not None else None) for i in order]
        if self.max_per_group:
            hits = diversify(hits, self.max_per_group)
        if mode.endswith("_rerank"):
            hits = rerank(query, hits, self.rerank_top)
        return hits[:k]


def rerank(query: str, hits: list[Hit], top: int) -> list[Hit]:
    """Reordena os `top` primeiros com o cross-encoder (o resto mantém a ordem original). Em CPU o
    reranker custa segundos por trecho, então o agente chama isto UMA vez, sobre a união dos
    candidatos de todas as buscas, e não uma vez por busca."""
    from src.rag.embed import get_reranker

    head = hits[:top]
    for h, v in zip(head, get_reranker().score(query, [index_text(h.chunk) for h in head])):
        h.rerank = float(v)
        h.score = float(v)
    return sorted(head, key=lambda h: -h.score) + hits[top:]
