"""Ingestão: arquivos das normas -> linhas estruturais -> chunks -> data/processed/rag/chunks.jsonl

  python -m src.rag.ingest
"""
from __future__ import annotations

import json
from collections import Counter

from src.rag.chunk import chunk_document
from src.rag.download import existing, registry
from src.rag.parse import read_lines
from src.common.config import data_path


def chunks_path():
    return data_path("processed", "rag", "chunks.jsonl")


def main():
    all_chunks = []
    for doc in registry():
        p = existing(doc["id"])
        if p is None:
            print(f"[aviso] {doc['id']}: arquivo ausente (rode `make rag-download`)")
            continue
        urlf = p.with_suffix(".url")
        url = urlf.read_text().strip() if urlf.exists() else (doc.get("urls") or [None])[0]
        lines = read_lines(p)
        ch = chunk_document(doc, lines, url=url)
        kinds = Counter(l.kind for l in lines)
        print(f"{doc['id']}: {len(lines)} linhas ({dict(kinds)}) -> {len(ch)} chunks "
              f"({sum(c.revogado for c in ch)} revogados)")
        all_chunks += ch
    with open(chunks_path(), "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    # cópia versionada para rodar o agente sem baixar nada (amostra)
    sample = data_path("sample", "rag_chunks.jsonl")
    sample.write_text(chunks_path().read_text(encoding="utf-8"), encoding="utf-8")
    print(f"{len(all_chunks)} chunks -> {chunks_path()}")


if __name__ == "__main__":
    main()
