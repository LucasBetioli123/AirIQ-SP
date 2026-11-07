"""Split legal text into citable passages (chunks).

Strategy (rationale in docs/TECHNICAL_DETAILS.md):
  - Basic unit = the whole ARTICLE (caput + paragraphs + items + sub-items). In legal
    practice the article is the unit of meaning: a lone item ("II - 60 µg/m³") says
    nothing without the caput that introduces it
  - Long article (> MAX_WORDS): split into blocks of items/paragraphs, each starting with
    the CAPUT repeated, so it stays self-contained
  - Every chunk carries a citable breadcrumb: document › chapter › section › article,
    plus the list of provisions it contains (art. 4º, § 1º, inciso II...). This is what
    lets the agent cite "art. 4º, § 2º" precisely
  - The preamble (summary + recitals) becomes its own chunk
  - Annexes/tables: one chunk per table or small annex. A large table becomes one chunk
    per row with the header repeated (so "MP2,5 24 h PI-1 = 60" is retrievable on its own)
  - Repealed provisions (struck through on Planalto) stay out of the in-force index, and
    a fully repealed regulation (vigencia_fim) is flagged for the temporal filter
Labels such as "preâmbulo" stay in Portuguese: they are part of the citations.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from src.rag.parse import RE, Line

MAX_WORDS = 320
RE_NUM = RE["num"]


@dataclass
class Chunk:
    id: str
    doc_id: str
    sigla: str
    titulo: str
    path: str                   # "Capítulo II > Seção I"
    label: str                  # "art. 4º" | "art. 4º (incisos I a V)" | "Anexo I" | "preâmbulo"
    dispositivos: list[str]     # ["art. 4º", "art. 4º, § 1º", "art. 4º, inciso II", ...]
    text: str
    vigencia_inicio: str | None = None
    vigencia_fim: str | None = None
    url: str | None = None
    revogado: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"{self.sigla}, {self.label}"

    def for_index(self) -> str:
        """Text used for indexing (context + content)."""
        head = f"{self.sigla} - {self.path + ' - ' if self.path else ''}{self.label}"
        return f"{head}\n{self.text}"

    def to_dict(self):
        return asdict(self)


def _words(s: str) -> int:
    return len(s.split())


def _dispositivos(art_label: str, lines: list[Line]) -> list[str]:
    out = [art_label]
    cur_par = None
    cur_inc = None
    for l in lines:
        if l.kind == "par":
            cur_par, cur_inc = l.label, None
            out.append(f"{art_label}, {l.label}")
        elif l.kind == "inciso":
            cur_inc = l.label
            out.append(", ".join(x for x in (art_label, cur_par, l.label) if x))
        elif l.kind == "alinea":
            out.append(", ".join(x for x in (art_label, cur_par, cur_inc, l.label) if x))
    return out


def _table_chunks(base: dict, label: str, rows: list[Line], start_idx: int) -> list[Chunk]:
    """Table (rows 'a | b | c'): if small, 1 chunk; if large, 1 chunk per row + header."""
    texts = [r.text for r in rows]
    if _words(" ".join(texts)) <= MAX_WORDS:
        return [Chunk(id=f"{base['doc_id']}:{_slug(label)}", label=label, dispositivos=[label],
                      text="\n".join(texts), **_b(base))]
    # title/notes + header row repeated on every row; the label names the row by its first cells
    rows = [t for t in texts if "|" in t]
    first = texts.index(rows[0])
    title = [t for t in texts if "|" not in t and _words(t) <= 60][:8]  # title and short notes
    head, data = rows[:1], rows[1:]
    out = []
    for i, t in enumerate(data, start=1):
        name = ", ".join(c.strip() for c in t.split("|")[:2])
        out.append(Chunk(id=f"{base['doc_id']}:{_slug(label)}:l{i}", label=f"{label} ({name})",
                         dispositivos=[label], text="\n".join(title + head + [t]), **_b(base)))
    return out


def _slug(s: str) -> str:
    s = s.lower().replace("º", "").replace("§", "par").replace("ú", "u").replace("ç", "c").replace("ã", "a")
    return re.sub(r"[^a-z0-9]+", "", s)


def _b(base: dict) -> dict:
    return {k: v for k, v in base.items()}


def _numbered_as_units(lines: list[Line]) -> list[Line]:
    """Regulation without articles (CONAMA 05/1989 is organized in items "I -", "2 -", "2.2.1 -"): each
    numbered item becomes the citable unit, playing the role of the article. The "I -" item before the
    first numbered item is item I of the operative part."""
    out, started = [], False
    for l in lines:
        if l.kind == "num" or (l.kind == "inciso" and not started):
            started = True
            num = RE_NUM.match(l.text)
            lab = f"item {num.group(1) if num else l.label.replace('inciso ', '')}"
            out.append(Line("art", lab, l.text, l.revogado, l.extra))
        else:
            out.append(l)
    return out


def _apply_partial_revocations(doc: dict, chunks: list[Chunk]) -> None:
    """Repeal of specific provisions (e.g. 506/2024 repealed arts. 1-8, 12-14 and Annex I of 491/2018,
    but kept arts. 9-11 and Annexes II-IV). The chunk stays in the index with its own vigencia_fim
    (end of validity), to answer "what applied on date X"."""
    for rv in doc.get("revogacoes_parciais", []):
        alvo = set(rv["dispositivos"])
        for c in chunks:
            if c.label.split(" (")[0].split(",")[0] in alvo:
                c.vigencia_fim = rv["vigencia_fim"]
                c.meta["revogado_por"] = rv["revogado_por"]
    if doc.get("revogada_por"):
        for c in chunks:
            c.meta.setdefault("revogado_por", doc["revogada_por"])


def chunk_document(doc: dict, lines: list[Line], url: str | None = None) -> list[Chunk]:
    if not any(l.kind == "art" for l in lines):
        lines = _numbered_as_units(lines)
    chunks = _chunk_lines(doc, lines, url)
    _apply_partial_revocations(doc, chunks)
    return chunks


def _chunk_lines(doc: dict, lines: list[Line], url: str | None = None) -> list[Chunk]:
    base = dict(doc_id=doc["id"], sigla=doc["sigla"], titulo=doc["titulo"],
                vigencia_inicio=doc.get("vigencia_inicio"), vigencia_fim=doc.get("vigencia_fim"),
                url=url, path="")
    chunks: list[Chunk] = []
    hier: dict[str, str] = {}
    pre: list[str] = []
    i = 0
    n = len(lines)
    # preamble: everything before the first article
    while i < n and lines[i].kind not in ("art", "anexo"):
        if lines[i].kind in ("titulo", "capitulo", "secao", "subsecao"):
            hier[lines[i].kind] = lines[i].text
        else:
            pre.append(lines[i].text)
        i += 1
    if pre:
        chunks.append(Chunk(id=f"{doc['id']}:preambulo", label="preâmbulo (ementa e considerandos)",
                            dispositivos=["preâmbulo"], text="\n".join(pre), **base))
    while i < n:
        l = lines[i]
        if l.kind in ("titulo", "capitulo", "secao", "subsecao"):
            order = ["titulo", "capitulo", "secao", "subsecao"]
            for k in order[order.index(l.kind):]:
                hier.pop(k, None)
            # the chapter name usually comes on the next line ("DOS PADRÕES ...")
            name = l.text
            if i + 1 < n and lines[i + 1].kind == "texto" and lines[i + 1].text.isupper():
                name += " - " + lines[i + 1].text
                i += 1
            hier[l.kind] = name
            i += 1
            continue
        path = " > ".join(hier[k] for k in ["titulo", "capitulo", "secao", "subsecao"] if k in hier)
        if l.kind == "art":
            j = i + 1
            while j < n and lines[j].kind not in ("art", "anexo", "titulo", "capitulo", "secao", "subsecao"):
                j += 1
            body = lines[i:j]
            vig = [x for x in body if not x.revogado]
            revog = len(vig) == 0
            use = vig or body
            b = {**base, "path": path}
            if _words(" ".join(x.text for x in use)) <= MAX_WORDS:
                chunks.append(Chunk(id=f"{doc['id']}:{_slug(l.label)}", label=l.label,
                                    dispositivos=_dispositivos(l.label, use), text="\n".join(x.text for x in use),
                                    revogado=revog, **b))
            else:
                caput, rest = use[0], use[1:]
                block, part = [], 1
                for x in rest + [None]:
                    if x is None or (_words(" ".join(y.text for y in block + [x])) > MAX_WORDS - _words(caput.text)
                                     and x.kind in ("par", "inciso") and block):
                        if block:
                            first, last = block[0].label or "", block[-1].label or ""
                            lab = f"{l.label} ({first}{' a ' + last if last and last != first else ''})"
                            chunks.append(Chunk(id=f"{doc['id']}:{_slug(l.label)}:p{part}", label=lab,
                                                dispositivos=_dispositivos(l.label, [caput] + block),
                                                text="\n".join([caput.text] + [y.text for y in block]),
                                                revogado=revog, **b))
                            part += 1
                        block = [x] if x is not None else []
                    else:
                        block.append(x)
            i = j
            continue
        if l.kind == "anexo":
            j = i + 1
            while j < n and lines[j].kind not in ("anexo", "art"):
                j += 1
            body = lines[i:j]
            label = l.label
            b = {**base, "path": ""}
            rows = [x for x in body[1:] if "|" in x.text]
            if rows and len(rows) >= 0.5 * (len(body) - 1):
                k = next(n for n, x in enumerate(body) if "|" in x.text)
                title = [x.text for x in body[:k]] + [x.text for x in body[k:] if "|" not in x.text and
                                                     re.match(r"^Nota \d", x.text)]
                tbl = [Line("texto", "", t) for t in title] + rows
                chunks.extend(_table_chunks(b, label, tbl, 0))
            else:
                # annex in running text: blocks of ~MAX_WORDS
                buf, part = [], 1
                for x in body:
                    buf.append(x.text)
                    if _words(" ".join(buf)) > MAX_WORDS:
                        chunks.append(Chunk(id=f"{doc['id']}:{_slug(label)}:b{part}", label=f"{label} (parte {part})",
                                            dispositivos=[label], text="\n".join(buf), **b))
                        buf, part = [body[0].text], part + 1
                if len(buf) > 1 or part == 1:
                    lab = label if part == 1 else f"{label} (parte {part})"
                    chunks.append(Chunk(id=f"{doc['id']}:{_slug(label)}" + (f":b{part}" if part > 1 else ""),
                                        label=lab, dispositivos=[label], text="\n".join(buf), **b))
            i = j
            continue
        i += 1  # loose text between provisions (rare): ignored
    # unique ids (articles repeated in consolidated texts, e.g. old + new wording)
    seen: dict[str, int] = {}
    for c in chunks:
        if c.id in seen:
            seen[c.id] += 1
            c.id = f"{c.id}#{seen[c.id]}"
        else:
            seen[c.id] = 0
    return chunks
