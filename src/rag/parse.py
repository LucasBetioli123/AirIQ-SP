"""Parse legal text (Planalto/DOU HTML or PDF) into structural lines.

Brazilian legal text has an explicit hierarchy (Complementary Law 95/1998, art. 10):
  Título > Capítulo > Seção > Subseção > Artigo > Parágrafo (§) > Inciso (I, II...) >
  Alínea (a, b...) > Item (1, 2...), plus Anexos (annexes, often tables).
Each line is classified with regular expressions (which match the Portuguese text). In
Planalto's HTML, struck-through text (<strike>/<s>/<del>) marks repealed provisions or old
wording; the line is flagged `revogado` (repealed) and, by default, not indexed as in force.
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
    # "Art. 9º-A.", "Art. 9 o -A." (Planalto HTML with <sup>o</sup>). The letter is only a suffix when
    # followed by a period/dash; otherwise "Art. 1º - Esta Lei" would become "art. 1º-E"
    "art": re.compile(rf"^Art(?:igo)?\.?\s*(\d+)\s?{ORD}(?:\s?-\s?([A-Z])(?=\s*[.\-–—]))?\s*[\.\-–—:]?\s*"),
    "par": re.compile(rf"^(?:§\s*(\d+)\s?{ORD}(?:\s?-\s?([A-Z])(?=\s*[.\-–—]))?|Par[áa]grafo\s+[úu]nico)"
                      rf"\s*[\.\-–—:]?\s*", re.I),
    "inciso": re.compile(r"^([IVXLC]+)(?:\s*-\s*([A-Z]))?\s*[-–—]\s+"),
    "alinea": re.compile(r"^([a-z])\)\s+"),
    "item": re.compile(r"^(\d{1,2})[\.\)]\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇa-z])"),
    # numbered items "2 - ESTRATÉGIAS", "2.2.1 - Ficam..." (old resolutions without articles, e.g. CONAMA 05/1989)
    "num": re.compile(r"^(\d{1,2}(?:\.\d{1,2})*)\s*-\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇa-z])"),
}


@dataclass
class Line:
    kind: str            # titulo|capitulo|secao|subsecao|anexo|art|par|inciso|alinea|item|texto
    label: str           # "art. 4º", "§ 2º", "inciso II", "alínea a" (provision labels used in citations)...
    text: str
    revogado: bool = False
    extra: dict = field(default_factory=dict)


# DOU pages "printed" to PDF carry ligatures (ﬁ), a spaced Greek mu ("μ g/m ᶟ") and subscripts
# (MP₂,₅). NFKC would fix them but also turns "º" into "o" and "³" into "3", so targeted replacements are used.
_CHARS = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "ᶟ": "³", "μ": "µ",
                        **{chr(0x2080 + i): str(i) for i in range(10)}})

# header/footer the browser adds when printing a DOU page to PDF
_PDF_NOISE = [re.compile(p) for p in (
    r"^\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}",   # "4/20/23, 9:46 AM RESOLUÇÃO Nº 491..."
    r"^https?://\S+(\s+\d+/\d+)?$",                  # page URL + "3/6"
    r"^Este conteúdo não substitui o publicado na versão certificada",
    r"^RESOLUÇÕES DO CONAMA$",                          # header of the CONAMA compilation
    r"^.*RESOLUÇÃO CONAMA nº \d+ de \d{4}$",            # "CONTROLE DA POLUIÇÃO DO AR – PRONAR RESOLUÇÃO CONAMA nº 5 de 1989"
)]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).replace("\xa0", " ").translate(_CHARS)
    s = re.sub(r"µ\s*g\s*/\s*m\s*³", "µg/m³", s)
    return re.sub(r"\s+", " ", s).strip()  # Planalto HTML breaks lines inside paragraphs


def _ord(n: str) -> str:
    """'1' -> '1º' (ordinals up to 9º; cardinals from 10 on, per Complementary Law 95/98)."""
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
    if m := RE["num"].match(t):
        return "num", f"item {m.group(1)}"
    return "texto", ""


# ----------------------------------------------------------------------------- leitores
_BLOCK = {"p", "tr", "h1", "h2", "h3", "h4", "li"}


def _blocks_in_order(root):
    """Text blocks (p, tr, h*, li) in document order, without nesting (a <p> inside a <tr> comes with the
    row). Loose text between blocks is returned as a string so no provision is lost."""
    from bs4 import NavigableString, Tag

    orphan: list[str] = []
    for node in root.descendants:
        if isinstance(node, Tag) and node.name in _BLOCK:
            if node.find_parent(list(_BLOCK)) is not None:
                continue
            if orphan:
                yield " ".join(orphan)
                orphan = []
            yield node
        elif isinstance(node, NavigableString) and node.find_parent(list(_BLOCK)) is None:
            if type(node) is NavigableString and node.strip():  # skip comments/doctype
                orphan.append(str(node))
    if orphan:
        yield " ".join(orphan)


def _html_paragraphs(path: Path) -> list[tuple[str, bool]]:
    raw = path.read_bytes()
    for enc in ("utf-8", "windows-1252", "latin-1"):
        try:
            html = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    # malformed Planalto HTML: "<p ...></font><font>text</font></p>". The stray </font> closes the <p>
    # and the provision's text (e.g. art. 6º, II of Law 6.938) would fall outside every block.
    html = re.sub(r"(<p[^>]*>)\s*</font>", r"\1", html, flags=re.I)
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "head"]):
        t.decompose()
    # DOU (in.gov.br): the act's text lives in div.texto-dou
    root = soup.select_one("div.texto-dou") or soup.body or soup
    out = []
    for b in _blocks_in_order(root):
        if isinstance(b, str):  # text outside any block (broken HTML)
            if text := _norm(b):
                out.append((text, False))
            continue
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
    # join lines broken by the layout: a line that does not start a provision continues the previous
    # one (unless the previous one ends with a period and this one starts with a capital letter).
    # In annexes (tables) each PDF line is a table row, so nothing is joined there.
    paras: list[str] = []
    in_annex = False
    for l in lines:
        if not l or re.fullmatch(r"\d+", l) or any(p.match(l) for p in _PDF_NOISE):  # page number, header
            continue
        kind, _ = classify(l)
        in_annex = in_annex or kind == "anexo"
        if paras and kind == "texto" and not in_annex and not re.search(r"[.:;]$", paras[-1]):
            paras[-1] += " " + l
        else:
            paras.append(l)
    return [(p, False) for p in _rewrite_tables(paras)]


# ----------------------------------------------------------------------------- annex tables
# PDFs lose the table grid: "Material Particulado - MP2,5 24 horas 60 50 37 25 15 -". A small LLM easily
# picks the wrong column, so every row is rewritten with the column name next to the value
# ("PI-1: 60 µg/m³ | PI-2: 50 µg/m³ ..."). Rows become "a | b | c" and the chunker treats them as a table.
_PERIOD = re.compile(r"^(?P<nome>.*?)\s*(?P<per>24 horas|Anual|1 hora|8 horas)\s*(?:[¹²³⁴⁵]|[1-5](?=\s))?\s+"
                     r"(?P<vals>(?:[\d.,]+|-)(?:\s+(?:[\d.,]+|-))*)\s*$")
_UNITS_ONLY = re.compile(r"^(?:\s*(?:µg/m³|µ/m³|ppm))+\s*$")
_LEVEL_COLS = ["SO2 (µg/m³, média de 24h)", "MP10 (µg/m³, média de 24h)", "MP2,5 (µg/m³, média de 24h)",
               "CO (ppm, média móvel de 8h)", "O3 (µg/m³, média móvel de 8h)", "NO2 (µg/m³, média de 1h)"]
_IQAR_UNIT = {"CO": "ppm"}


def _standards_table(rows: list[str]) -> list[str] | None:
    """Standards table (Annex I of Res. 491/2018 and 506/2024): columns PI-1..PF (+ ppm)."""
    head = next((i for i, r in enumerate(rows) if re.search(r"\bPI-1\b", r)), None)
    if head is None:
        return None
    cols = re.findall(r"PI-\d|PF", rows[head]) + ["PF (ppm)"]
    out, name, pending = [f"Poluente | Período de referência | {' | '.join(cols[:-1])} (µg/m³; CO em ppm)"], "", ""
    for r in rows[head + 1:]:
        if _UNITS_ONLY.match(r):
            continue
        m = _PERIOD.match(r)
        if not m:
            if re.match(r"^\d\s*-\s", r):           # footnotes "1 - média aritmética anual"
                out.append(f"Nota {r}")
            else:                                   # pollutant name split over two lines
                pending = (pending + " " + r).strip()
            continue
        if m["nome"] or pending:
            name = re.sub(r"(?<=Pb)[5⁵]$", "", (pending + " " + m["nome"]).strip())
            pending = ""
        vals = m["vals"].split()
        vals = ["-"] * (len(cols) - len(vals)) + vals  # empty cells vanish in the PDF; align to the right
        cells = [f"{c}: {v}{'' if c.endswith('(ppm)') else ' µg/m³'}" if v != "-" else f"{c}: -"
                 for c, v in zip(cols, vals)]
        cells = [c for c in cells if not c.startswith("PF (ppm): -")]
        out.append(f"{name} | {m['per']} | " + " | ".join(cells))
    return out


def _levels_table(rows: list[str]) -> list[str] | None:
    """Attention/alert/emergency levels (Annex III of Res. 491/2018). The header comes out scrambled from
    the PDF; the column order is that of the published table."""
    data = [r for r in rows if re.match(r"^(Atenção|Alerta|Emergência)\s", r)]
    if not data:
        return None
    out = ["Nível | " + " | ".join(_LEVEL_COLS)]
    for r in data:
        lvl, *vals = r.split()
        if len(vals) != len(_LEVEL_COLS):
            return None
        out.append(f"{lvl} | " + " | ".join(f"{c}: {v}" for c, v in zip(_LEVEL_COLS, vals)))
    notes = [r for r in rows if r.startswith(("SO2=", "MP2,5=", "O3=", "SO2 =", "MP2,5 ="))]
    return out + notes


def _iqar_table(rows: list[str]) -> list[str] | None:
    """IQAr bands (Annex IV of 491/2018, Annex II of 506/2024): 'N1 - Boa 0 - 40 0 - 50 ...'."""
    head = next((r for r in rows if r.startswith("Qualidade Índice")), None)
    if head is None:
        return None
    pols = head.split()[2:]
    per = next((r.split() for r in rows if re.fullmatch(r"(?:\d+h\s*)+", r)), ["?"] * len(pols))
    out = ["Qualidade (faixa do IQAr) | Índice | " + " | ".join(f"{p} ({h})" for p, h in zip(pols, per))]
    for r in rows:
        m = re.match(r"^(N\d\s*-\s*[A-Za-zÀ-ú ]+?)\s+(\d.*)$", r)
        if not m:
            if not (r == head or _UNITS_ONLY.match(r.replace("(", "").replace(")", "")) or re.fullmatch(r"(?:\d+h\s*)+", r)):
                out.append(r)
            continue
        rng = re.findall(r"([\d.,]+)\s*-\s*([\d.,]+)", m[2])
        if len(rng) != len(pols) + 1:
            out.append(r)
            continue
        cells = [f"índice {rng[0][0]}-{rng[0][1]}"] + [
            f"{p} ({h}): {a}-{b} {_IQAR_UNIT.get(p, 'µg/m³')}" for p, h, (a, b) in zip(pols, per, rng[1:])]
        out.append(f"{m[1]} | " + " | ".join(cells))
    return out


def _rewrite_tables(paras: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(paras):
        out.append(paras[i])
        if classify(paras[i])[0] != "anexo":
            i += 1
            continue
        j = i + 1
        while j < len(paras) and classify(paras[j])[0] not in ("anexo", "art"):
            j += 1
        body = paras[i + 1:j]
        for fn in (_standards_table, _levels_table, _iqar_table):
            if (new := fn(body)) is not None:
                # annex title (lines before the table) + rewritten table
                title = [b for b in body[:2] if b.isupper() and "|" not in b]
                body = title + new
                break
        out += body
        i = j
    return out


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
