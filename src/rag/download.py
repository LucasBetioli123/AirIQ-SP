"""Baixa as normas das fontes oficiais (Planalto, DOU/in.gov.br, CONAMA).

Para cada documento de configs/normas.yaml: tenta as URLs em ordem. Se nenhuma
funcionar e houver `busca_dou`, procura o ato na busca do DOU. Valida o conteúdo
(`must_contain`) para não indexar uma página de erro. Arquivos já presentes em
data/raw/normas/ (por exemplo, colocados à mão) são respeitados.
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
    """Busca no DOU (in.gov.br) e devolve links de matérias."""
    u = "https://www.in.gov.br/consulta/-/buscar/dou?" + urllib.parse.urlencode(
        {"q": f'"{query}"', "s": "todos", "exactDate": "all", "sortType": "0"})
    r = session().get(u, timeout=60)
    r.raise_for_status()
    # a página traz os resultados num JSON embutido (urlTitle)
    slugs = re.findall(r'"urlTitle"\s*:\s*"([^"]+)"', r.text)
    return [f"https://www.in.gov.br/en/web/dou/-/{s}" for s in slugs[:3]]


def fetch(doc: dict) -> str:
    if p := existing(doc["id"]):
        return f"já existe: {p.name}"
    urls = list(doc.get("urls", []))
    if doc.get("busca_dou"):
        try:
            urls += _dou_search(doc["busca_dou"])
        except Exception as e:  # busca é só um reforço
            print(f"  busca DOU falhou: {e}")
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
    return "FALHOU (coloque o arquivo manualmente em data/raw/normas/)"


def main():
    for d in registry():
        print(f"{d['id']}: {fetch(d)}")


if __name__ == "__main__":
    main()
