"""Download open GGUF models (LLM, embeddings, reranker) to run locally at zero cost.

Source: the OCI artifacts in Docker Hub's `ai/` namespace (Docker Model Runner), which
package the same GGUF files published on HuggingFace, with anonymous access and no
account. With Ollama installed you can use `LLM_PROVIDER=ollama` and `ollama pull qwen3:4b`
instead.

  python -m src.rag.models_dl                  # default models (see DEFAULT)
  python -m src.rag.models_dl --all            # every benchmark candidate
  python -m src.rag.models_dl ai/qwen3:4b-instruct-2507-q4_K_M
"""
from __future__ import annotations

import argparse
import time

from src.common.config import ROOT
import requests

_S = requests.Session()  # no automatic retry on 429: Docker Hub's limit is per hour


def session():
    return _S

REGISTRY = "https://registry-1.docker.io/v2"
GGUF_TYPES = ("application/vnd.docker.ai.gguf.v3", "application/vnd.docker.ai.gguf")

# (Docker Hub reference, destination folder, file name)
DEFAULT = [
    ("ai/qwen3:4b-instruct-2507-q4_K_M", "llm", "qwen3-4b-instruct-2507-q4_k_m.gguf"),
    ("ai/granite-embedding-multilingual:278M-F16", "emb", "granite-embedding-multilingual-278m-f16.gguf"),
    ("ai/qwen3-reranker:0.6B", "rerank", "qwen3-reranker-0.6b.gguf"),
]
BENCH = [
    ("ai/qwen2.5:7B-Q4_K_M", "llm", "qwen2.5-7b-instruct-q4_k_m.gguf"),
    ("ai/llama3.2:3B-Q4_K_M", "llm", "llama3.2-3b-instruct-q4_k_m.gguf"),
    ("ai/gemma3:4B-Q4_K_M", "llm", "gemma3-4b-it-q4_k_m.gguf"),
    ("ai/qwen3:0.6B-Q4_K_M", "llm", "qwen3-0.6b-q4_k_m.gguf"),
    ("ai/embeddinggemma:300M-Q8_0", "emb", "embeddinggemma-300m-q8_0.gguf"),
    ("ai/qwen3-embedding:0.6b-q8_0", "emb", "qwen3-embedding-0.6b-q8_0.gguf"),
    ("ai/nomic-embed-text-v1.5:137M-F16", "emb", "nomic-embed-text-v1.5-f16.gguf"),
]


def _token(repo: str) -> str:
    r = session().get("https://auth.docker.io/token",
                      params={"service": "registry.docker.io", "scope": f"repository:{repo}:pull"}, timeout=30)
    r.raise_for_status()
    return r.json()["token"]


def _manifest(repo: str, tag: str, tok: str) -> dict:
    accept = ",".join(["application/vnd.oci.image.manifest.v1+json", "application/vnd.oci.image.index.v1+json",
                       "application/vnd.docker.distribution.manifest.v2+json"])
    for attempt in range(6):
        r = session().get(f"{REGISTRY}/{repo}/manifests/{tag}",
                          headers={"Authorization": f"Bearer {tok}", "Accept": accept}, timeout=30)
        if r.status_code == 429:  # Docker Hub anonymous limit (100 manifests/h per IP)
            wait = 60 * (attempt + 1)
            print(f"  Docker Hub rate limit reached; waiting {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        m = r.json()
        if "manifests" in m:  # multi-platform index -> take the first entry
            return _manifest(repo, m["manifests"][0]["digest"], tok)
        return m
    raise RuntimeError("Docker Hub: rate limit reached; try again later")


def pull(ref: str, sub: str, fname: str) -> str:
    dest = ROOT / "models" / sub / fname
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest)
    repo, tag = ref.split(":")
    tok = _token(repo)
    m = _manifest(repo, tag, tok)
    # newer artifacts use the CNCF format (vnd.cncf.model.weight.v1.raw) with the file name in an annotation
    layer = next(l for l in m["layers"] if l["mediaType"] in GGUF_TYPES or
                 l.get("annotations", {}).get("org.cncf.model.filepath", "").endswith(".gguf"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"downloading {ref} ({layer['size'] / 1e9:.2f} GB) -> {dest.relative_to(ROOT)}")
    with session().get(f"{REGISTRY}/{repo}/blobs/{layer['digest']}", headers={"Authorization": f"Bearer {tok}"},
                       stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
    tmp.rename(dest)
    return str(dest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("refs", nargs="*")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    known = {it[0]: it for it in DEFAULT + BENCH}  # known refs go to the folder/name the code expects
    items = [known.get(r, (r, "llm", r.split("/")[-1].replace(":", "-") + ".gguf")) for r in a.refs] or \
        (DEFAULT + BENCH if a.all else DEFAULT)
    for it in items:
        try:
            print(pull(*it))
        except Exception as e:  # keep going with the rest
            print(f"failed {it[0]}: {e}")
