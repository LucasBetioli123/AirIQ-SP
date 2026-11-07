"""Download the regulations from official sources (Planalto, DOU/in.gov.br, CONAMA).

For each document in configs/normas.yaml: try the URLs in order. If none works and
`busca_dou` is set, look the act up in the Official Gazette (DOU) search. The content is
validated (`must_contain`) so an error page is never indexed. Files already present in
data/raw/normas/ (e.g. placed by hand) are kept.
"""
from __future__ import annotations

import re
import urllib.parse

import yaml
from bs4 import BeautifulSoup

from src.common.config import ROOT, data_path
from src.common.http import session


def registry() -> list[dict]:
    with open(ROOT / "configs" / "normas.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["documentos"]


def existing(doc_id: str):
    for ext in ("html", "pdf", "txt"):
        p = data_path("raw", "normas", f"{doc_id}.{ext}")
        if p.exists() and p.stat().st_size > 500:
            return p
    return None


def _text_of(content: bytes, ctype: str) -> str:
    if "pdf" in ctype or content[:4] == b"%PDF":
        import io

        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(content)).pages)
    return BeautifulSoup(content, "lxml").get_text(" ")


def _valid(text: str, must: list[str]) -> bool:
    t = re.sub(r"\s+", " ", text).lower()
    return all(m.lower() in t for m in must)


def _dou_search(query: str) -> list[str]:
    """Search the Official Gazette (in.gov.br) and return article links."""
    u = "https://www.in.gov.br/consulta/-/buscar/dou?" + urllib.parse.urlencode(
        {"q": f'"{query}"', "s": "todos", "exactDate": "all", "sortType": "0"})
    r = session().get(u, timeout=60)
    r.raise_for_status()
    # the page carries the results in an embedded JSON (urlTitle)
    slugs = re.findall(r'"urlTitle"\s*:\s*"([^"]+)"', r.text)
    return [f"https://www.in.gov.br/en/web/dou/-/{s}" for s in slugs[:3]]


def fetch(doc: dict) -> str:
    if p := existing(doc["id"]):
        return f"already present: {p.name}"
    urls = list(doc.get("urls", []))
    if doc.get("busca_dou"):
        try:
            urls += _dou_search(doc["busca_dou"])
        except Exception as e:  # the search is only a fallback
            print(f"  DOU search failed: {e}")
    for u in urls:
        try:
            r = session().get(u, timeout=60)
            r.raise_for_status()
            ctype = r.headers.get("content-type", "")
            if not _valid(_text_of(r.content, ctype), doc["must_contain"]):
                continue
            ext = "pdf" if ("pdf" in ctype or r.content[:4] == b"%PDF") else "html"
            p = data_path("raw", "normas", f"{doc['id']}.{ext}")
            p.write_bytes(r.content)
            data_path("raw", "normas", f"{doc['id']}.url").write_text(u)
            return f"ok: {u}"
        except Exception as e:
            print(f"  {u}: {e}")
    return "FAILED (place the file manually in data/raw/normas/)"


def main():
    for d in registry():
        print(f"{d['id']}: {fetch(d)}")


if __name__ == "__main__":
    main()
