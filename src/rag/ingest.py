"""Ingestion: regulation files -> structural lines -> chunks -> data/processed/rag/chunks.jsonl

  python -m src.rag.ingest
"""
from __future__ import annotations

import json
import os
import tempfile
from collections import Counter

from src.rag.chunk import chunk_document
from src.rag.download import existing, registry
from src.rag.parse import read_lines
from src.common.config import data_path


def chunks_path():
    return data_path("processed", "rag", "chunks.jsonl")


def _write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main():
    all_chunks = []
    missing = []
    for doc in registry():
        p = existing(doc["id"])
        if p is None:
            print(f"[warning] {doc['id']}: file missing (run `make rag-download`)")
            missing.append(doc["id"])
            continue
        urlf = p.with_suffix(".url")
        url = urlf.read_text().strip() if urlf.exists() else (doc.get("urls") or [None])[0]
        lines = read_lines(p)
        ch = chunk_document(doc, lines, url=url)
        if not ch:
            missing.append(doc["id"])
        kinds = Counter(l.kind for l in lines)
        print(f"{doc['id']}: {len(lines)} lines ({dict(kinds)}) -> {len(ch)} chunks "
              f"({sum(c.revogado for c in ch)} repealed)")
        all_chunks += ch
    if missing:
        print("[warning] incomplete ingestion; existing base and sample preserved: " + ", ".join(missing))
        if not chunks_path().exists():
            sample = data_path("sample", "rag_chunks.jsonl")
            rows = [json.loads(s) for s in sample.read_text(encoding="utf-8").splitlines() if s.strip()]
            if not {d["id"] for d in registry()} <= {c["doc_id"] for c in rows}:
                raise RuntimeError("incomplete regulation sample; obtain the missing documents")
            _write_atomic(chunks_path(), sample.read_text(encoding="utf-8"))
            print(f"using the complete sample: {len(rows)} chunks")
        return
    _write_atomic(chunks_path(), "".join(json.dumps(c.to_dict(), ensure_ascii=False) + "\n" for c in all_chunks))
    # Ingestion never changes the versioned sample; updating it requires an explicit review.
    print(f"{len(all_chunks)} chunks -> {chunks_path()}")


if __name__ == "__main__":
    main()
