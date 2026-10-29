"""Embeddings e reranker locais (GGUF via llama.cpp), sem custo e sem rede na inferência.

Cada família de modelo tem um formato de instrução próprio para consulta x documento.
Usar o formato errado derruba bastante o recall, por isso o formato vai codificado aqui.
"""
from __future__ import annotations

import functools
import hashlib
import math
import os
from pathlib import Path

import numpy as np

from src.common.config import ROOT, data_path

TASK = "Dada uma pergunta sobre legislação ambiental brasileira, recupere o trecho normativo que a responde"
FORMATS = {
    "qwen3-embedding": (lambda q: f"Instruct: {TASK}\nQuery: {q}", lambda d: d),
    "embeddinggemma": (lambda q: f"task: search result | query: {q}", lambda d: f"title: none | text: {d}"),
    "nomic-embed": (lambda q: f"search_query: {q}", lambda d: f"search_document: {d}"),
    "granite-embedding": (lambda q: q, lambda d: d),
}

# padrão escolhido no benchmark (reports/metrics/rag_retrieval.csv): melhor híbrido e o mais rápido em CPU
DEFAULT_EMB = "granite-multilingual-278m"
EMB_MODELS = {
    "qwen3-emb-0.6b": "models/emb/qwen3-embedding-0.6b-q8_0.gguf",
    "embeddinggemma-300m": "models/emb/embeddinggemma-300m-q8_0.gguf",
    "granite-multilingual-278m": "models/emb/granite-embedding-multilingual-278m-f16.gguf",
    "nomic-v1.5": "models/emb/nomic-embed-text-v1.5-f16.gguf",
}


def _family(path: str) -> str:
    return next(k for k in FORMATS if k in Path(path).name)


class Embedder:
    def __init__(self, path: str | None = None, n_ctx: int = 1024):
        from llama_cpp import Llama

        self.path = str(ROOT / (path or os.environ.get("EMB_MODEL", EMB_MODELS[DEFAULT_EMB])))
        self.fmt_q, self.fmt_d = FORMATS[_family(self.path)]
        self.llm = Llama(model_path=self.path, embedding=True, n_ctx=n_ctx, n_batch=n_ctx, n_ubatch=n_ctx,
                         n_threads=os.cpu_count(), verbose=False)
        train_ctx = self.llm._model.n_ctx_train()
        if train_ctx < n_ctx:  # ex.: Granite (512): além disso a posição sai do treino e o vetor degrada
            n_ctx = train_ctx
            self.llm = Llama(model_path=self.path, embedding=True, n_ctx=n_ctx, n_batch=n_ctx, n_ubatch=n_ctx,
                             n_threads=os.cpu_count(), verbose=False)
        self.n_ctx = n_ctx

    def _embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            toks = self.llm.tokenize(t.encode(), add_bos=True)
            if len(toks) > self.n_ctx - 8:  # trunca com folga (chunks já são curtos)
                t = self.llm.detokenize(toks[: self.n_ctx - 8]).decode(errors="ignore")
            v = np.asarray(self.llm.embed(t), dtype="float32")
            if v.ndim == 2:  # modelo sem pooling -> média dos tokens
                v = v.mean(0)
            out.append(v / (np.linalg.norm(v) + 1e-9))
        return np.vstack(out)

    def queries(self, qs: list[str]) -> np.ndarray:
        return self._embed([self.fmt_q(q) for q in qs])

    def documents(self, ds: list[str], cache_key: str | None = None) -> np.ndarray:
        if cache_key:
            h = hashlib.sha1(("\n\x00".join(ds) + self.path).encode()).hexdigest()[:12]
            p = data_path("processed", "rag", f"emb_{cache_key}_{h}.npy")
            if p.exists():
                return np.load(p)
            v = self._embed([self.fmt_d(d) for d in ds])
            np.save(p, v)
            return v
        return self._embed([self.fmt_d(d) for d in ds])


@functools.lru_cache(maxsize=4)
def get_embedder(path: str | None = None) -> Embedder:
    return Embedder(path)


class Reranker:
    """Qwen3-Reranker (cross-encoder gerativo): score = P("yes") para a pergunta
    "o documento atende à consulta?". Usa o template oficial do modelo."""

    PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
              "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, path: str | None = None, n_ctx: int = 1536):
        from llama_cpp import Llama

        self.path = str(ROOT / (path or os.environ.get("RERANK_MODEL", "models/rerank/qwen3-reranker-0.6b.gguf")))
        self.llm = Llama(model_path=self.path, n_ctx=n_ctx, n_batch=512, logits_all=False,
                         n_threads=os.cpu_count(), verbose=False)
        self.yes = self.llm.tokenize(b"yes", add_bos=False, special=False)[-1]
        self.no = self.llm.tokenize(b"no", add_bos=False, special=False)[-1]
        self.n_ctx = n_ctx

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        out = []
        for d in docs:
            body = f"<Instruct>: {TASK}\n<Query>: {query}\n<Document>: {d}"
            toks = self.llm.tokenize((self.PREFIX + body + self.SUFFIX).encode(), add_bos=False, special=True)
            if len(toks) > self.n_ctx - 4:
                keep = self.n_ctx - 4 - len(self.llm.tokenize(self.SUFFIX.encode(), add_bos=False, special=True))
                toks = toks[:keep] + self.llm.tokenize(self.SUFFIX.encode(), add_bos=False, special=True)
            self.llm.reset()
            self.llm.eval(toks)
            # sem logits_all, o llama-cpp-python >= 0.3 não copia mais os logits para `scores`:
            # lê os do último token direto do contexto
            ptr = self.llm._ctx.get_logits_ith(-1)
            logits = np.ctypeslib.as_array(ptr, shape=(self.llm.n_vocab(),))
            a, b = float(logits[self.yes]), float(logits[self.no])
            out.append(1.0 / (1.0 + math.exp(b - a)))
        return np.asarray(out)


@functools.lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()
