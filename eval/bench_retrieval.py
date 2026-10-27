"""Benchmark de recuperação sobre o conjunto de avaliação (eval/rag_questions.yaml).

Para cada configuração (BM25, vetorial com cada modelo de embedding, híbrido RRF,
híbrido + reranker), calcula sobre as perguntas COM resposta na base:
  recall@k  fração dos chunks-ouro presentes no top-k (média por pergunta)
  hit@k     pelo menos um chunk-ouro no top-k
  MRR       1/posição do primeiro chunk-ouro
e a latência média por consulta. Saída: reports/metrics/rag_retrieval.csv
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import yaml

from src.common.config import ROOT, data_path
from src.rag.embed import EMB_MODELS
from src.rag.retrieve import Retriever


def load_questions():
    with open(ROOT / "eval" / "rag_questions.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["perguntas"]


def gold_match(chunk_id: str, gold: list[str]) -> bool:
    """Um chunk-ouro 'conselho:art4' também casa com as partes 'conselho:art4:p2'."""
    base = chunk_id.split("#")[0]
    return any(base == g or base.startswith(g + ":") for g in gold)


def evaluate(ret: Retriever, qs, mode: str, ks=(1, 3, 5, 10)) -> dict:
    rec = {k: [] for k in ks}
    hit = {k: [] for k in ks}
    rr, lat = [], []
    for q in qs:
        t0 = time.time()
        hits = ret.search(q["pergunta"], k=max(ks), date=q.get("data"), mode=mode)
        lat.append(time.time() - t0)
        ids = [h.chunk["id"] for h in hits]
        gold = q["gold"]
        first = next((i for i, c in enumerate(ids) if gold_match(c, gold)), None)
        rr.append(0 if first is None else 1 / (first + 1))
        for k in ks:
            found = {g for g in gold if any(gold_match(c, [g]) for c in ids[:k])}
            rec[k].append(len(found) / len(gold))
            hit[k].append(float(bool(found)))
    out = {f"recall@{k}": float(np.mean(v)) for k, v in rec.items()}
    out.update({f"hit@{k}": float(np.mean(v)) for k, v in hit.items()})
    out.update({"MRR": float(np.mean(rr)), "latencia_ms": 1000 * float(np.mean(lat)), "n": len(qs)})
    return out


def main(emb_models=None, rerank=True):
    qs = [q for q in load_questions() if q.get("gold")]
    rows = []
    bm = Retriever("bm25")
    rows.append({"config": "BM25", **evaluate(bm, qs, "bm25")})
    for name in emb_models or EMB_MODELS:
        path = EMB_MODELS[name]
        if not (ROOT / path).exists():
            print(f"[pulando] {name}: {path} não baixado")
            continue
        r = Retriever("hybrid", emb_path=path)
        rows.append({"config": f"Vetorial ({name})", **evaluate(r, qs, "dense")})
        rows.append({"config": f"Híbrido RRF (BM25 + {name})", **evaluate(r, qs, "hybrid")})
        if rerank and name == "qwen3-emb-0.6b" and (ROOT / "models/rerank/qwen3-reranker-0.6b.gguf").exists():
            rows.append({"config": f"Híbrido + reranker Qwen3-0.6B ({name})", **evaluate(r, qs, "hybrid_rerank")})
        print(pd.DataFrame(rows).tail(3).to_string(index=False))
    df = pd.DataFrame(rows)
    df.to_csv(data_path("metrics", "rag_retrieval.csv"), index=False, float_format="%.4f")
    print(df[["config", "recall@1", "recall@3", "recall@5", "hit@5", "MRR", "latencia_ms"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="*")
    ap.add_argument("--no-rerank", action="store_true")
    a = ap.parse_args()
    main(a.emb, not a.no_rerank)
