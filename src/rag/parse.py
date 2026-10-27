"""Parse de texto normativo (HTML do Planalto/DOU ou PDF) em linhas estruturais.

O texto normativo brasileiro tem hierarquia explícita (LC 95/1998, art. 10):
  Título > Capítulo > Seção > Subseção > Artigo > Parágrafo (§) > Inciso (I, II...) >
  Alínea (a, b...) > Item (1, 2...), além de Anexos (muitas vezes tabelas).
Cada linha é classificada por expressões regulares. No HTML do Planalto, o texto
riscado (<strike>/<s>/<del>) marca dispositivos revogados ou com redação antiga. A
linha fica marcada como `revogado` e, por padrão, não é indexada como vigente.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup

ORD = r"(?:º|°|o|ª)?"
RE = {
    "titulo": re.compile(r"^T[ÍI]TULO\s+([IVXLC]+)\b", re.I),
    "capitulo": re.compile(r"^CAP[ÍI]TULO\s+([IVXLC]+(?:-[A-Z])?)\b", re.I),
    "secao": re.compile(r"^SE[ÇC][ÃA]O\s+([IVXLC]+(?:-[A-Z])?)\b", re.I),
    "subsecao": re.compile(r"^SUBSE[ÇC][ÃA]O\s+([IVXLC]+)\b", re.I),
    "anexo": re.compile(r"^ANEXO\s*([IVXLC]+|[0-9]+|[ÚU]NICO)?\b"),
    "art": re.compile(rf"^Art(?:igo)?\.?\s*(\d+){ORD}(?:\s*-\s*([A-Z]))?\s*[\.\-–—:]?\s*"),
    "par": re.compile(rf"^(?:§\s*(\d+){ORD}(?:\s*-\s*([A-Z]))?|Par[áa]grafo\s+[úu]nico)\s*[\.\-–—:]?\s*", re.I),
    "inciso": re.compile(r"^([IVXLC]+)(?:\s*-\s*([A-Z]))?\s*[-–—]\s+"),
    "alinea": re.compile(r"^([a-z])\)\s+"),
    "item": re.compile(r"^(\d{1,2})[\.\)]\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇa-z])"),
}


@dataclass
class Line:
    kind: str            # titulo|capitulo|secao|subsecao|anexo|art|par|inciso|alinea|item|texto
    label: str           # "art. 4º", "§ 2º", "inciso II", "alínea a"...
    text: str
    revogado: bool = False
    extra: dict = field(default_factory=dict)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).replace("\xa0", " ")
    return re.sub(r"[ \t]+", " ", s).strip()


def _ord(n: str) -> str:
    """'1' -> '1º' (até o 9º se usa ordinal; do 10 em diante, cardinal, conforme LC 95/98)."""
    return f"{n}º" if n.isdigit() and int(n) < 10 else n


def classify(text: str) -> tuple[str, str]:
    t = text.strip()
    for kind in ("titulo", "capitulo", "secao", "subsecao"):
        if (m := RE[kind].match(t)) and len(t) < 120:
            return kind, f"{kind.replace('secao', 'seção').replace('capitulo', 'capítulo').replace('titulo', 'título')} {m.group(1)}"
    if (m := RE["anexo"].match(t)) and len(t) < 160:
        return "anexo", f"Anexo {m.group(1) or ''}".strip()
    if m := RE["art"].match(t):
        suf = f"-{m.group(2)}" if m.group(2) else ""
        return "art", f"art. {_ord(m.group(1))}{suf}"
    if m := RE["par"].match(t):
        if m.group(1):
            suf = f"-{m.group(2)}" if m.group(2) else ""
            return "par", f"§ {_ord(m.group(1))}{suf}"
        return "par", "parágrafo único"
    if m := RE["inciso"].match(t):
        suf = f"-{m.group(2)}" if m.group(2) else ""
        return "inciso", f"inciso {m.group(1)}{suf}"
    if m := RE["alinea"].match(t):
        return "alinea", f"alínea {m.group(1)}"
    if m := RE["item"].match(t):
        return "item", f"item {m.group(1)}"
    return "texto", ""


# ----------------------------------------------------------------------------- leitores
def _html_paragraphs(path: Path) -> list[tuple[str, bool]]:
    raw = path.read_bytes()
    for enc in ("utf-8", "windows-1252", "latin-1"):
        try:
            html = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "head"]):
        t.decompose()
    # DOU (in.gov.br): o texto do ato fica em div.texto-dou
    root = soup.select_one("div.texto-dou") or soup.body or soup
    out = []
    blocks = root.find_all(["p", "tr", "h1", "h2", "h3", "h4", "li"]) or [root]
    for b in blocks:
        if b.name == "p" and b.find_parent("tr"):
            continue  # a linha da tabela inteira já é capturada pelo <tr>
        if b.name == "tr":
            cells = [_norm(c.get_text(" ")) for c in b.find_all(["td", "th"])]
            text = " | ".join(c for c in cells if c)
            struck = False
        else:
            text = _norm(b.get_text(" "))
            if not text:
                continue
            struck_txt = " ".join(_norm(s.get_text(" ")) for s in b.find_all(["strike", "s", "del"]))
            struck = len(struck_txt) >= 0.8 * len(text)
        if text:
            out.append((text, struck))
    return out


def _pdf_paragraphs(path: Path) -> list[tuple[str, bool]]:
    from pypdf import PdfReader

    lines = []
    for page in PdfReader(str(path)).pages:
        lines += [_norm(l) for l in (page.extract_text() or "").splitlines()]
    # junta linhas quebradas pelo layout: uma linha que não começa um dispositivo
    # continua a anterior (a menos que a anterior termine com ponto final e esta comece com maiúscula)
    paras: list[str] = []
    for l in lines:
        if not l or re.fullmatch(r"\d+", l):  # números de página
            continue
        kind, _ = classify(l)
        if paras and kind == "texto" and not re.search(r"[.:;]$", paras[-1]):
            paras[-1] += " " + l
        else:
            paras.append(l)
    return [(p, False) for p in paras]


def read_lines(path: Path) -> list[Line]:
    path = Path(path)
    if path.suffix == ".pdf":
        paras = _pdf_paragraphs(path)
    elif path.suffix == ".txt":
        paras = [(_norm(l), False) for l in path.read_text(encoding="utf-8").splitlines() if _norm(l)]
    else:
        paras = _html_paragraphs(path)
    out = []
    for text, struck in paras:
        kind, label = classify(text)
        rev = struck or bool(re.search(r"\((?:Revogad[oa]|Vetad[oa])\b", text, re.I)) and len(text) < 200
        out.append(Line(kind, label, text, revogado=rev))
    return out
