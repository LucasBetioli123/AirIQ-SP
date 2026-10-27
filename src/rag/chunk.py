"""Divisão do texto normativo em trechos citáveis (chunks).

Estratégia (justificativa no RELATORIO.md):
  - Unidade básica = ARTIGO inteiro (caput + §§ + incisos + alíneas). Na prática
    jurídica, o artigo é a unidade de sentido: um inciso solto ("II - 60 µg/m³") não diz
    nada sem o caput que o introduz
  - Artigo longo (> MAX_WORDS): divide em blocos de incisos/§§, e cada bloco leva o
    CAPUT repetido no início, para continuar autocontido
  - Cada chunk carrega um "breadcrumb" citável: documento › capítulo › seção › art.,
    mais a lista dos dispositivos que contém (art. 4º, § 1º, inciso II...). É isso que
    permite ao agente citar "art. 4º, § 2º" com precisão
  - Preâmbulo (ementa + considerandos) vira um chunk próprio
  - Anexos/tabelas: um chunk por tabela ou anexo pequeno. Tabela grande vira um
    chunk por linha, com o cabeçalho repetido (assim "MP2,5 24 h PI-1 = 60" fica
    recuperável sozinho)
  - Dispositivos revogados (riscados no Planalto) ficam fora do índice vigente, e a
    norma revogada inteira (vigencia_fim) fica marcada para o filtro temporal
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from src.rag.parse import Line

MAX_WORDS = 320


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
        """Texto usado na indexação (contexto + conteúdo)."""
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
    """Tabela (linhas 'a | b | c'): se pequena, 1 chunk; se grande, 1 chunk por linha + cabeçalho."""
    texts = [r.text for r in rows]
    if _words(" ".join(texts)) <= MAX_WORDS:
        return [Chunk(id=f"{base['doc_id']}:{_slug(label)}", label=label, dispositivos=[label],
                      text="\n".join(texts), **_b(base))]
    header = [t for t in texts[:3] if "|" in t][:2]
    out = []
    for i, t in enumerate(texts[len(header):], start=1):
        out.append(Chunk(id=f"{base['doc_id']}:{_slug(label)}:l{i}", label=f"{label}, linha {i}",
                         dispositivos=[label], text="\n".join(header + [t]), **_b(base)))
    return out


def _slug(s: str) -> str:
    s = s.lower().replace("º", "").replace("§", "par").replace("ú", "u").replace("ç", "c").replace("ã", "a")
    return re.sub(r"[^a-z0-9]+", "", s)


def _b(base: dict) -> dict:
    return {k: v for k, v in base.items()}


def chunk_document(doc: dict, lines: list[Line], url: str | None = None) -> list[Chunk]:
    base = dict(doc_id=doc["id"], sigla=doc["sigla"], titulo=doc["titulo"],
                vigencia_inicio=doc.get("vigencia_inicio"), vigencia_fim=doc.get("vigencia_fim"),
                url=url, path="")
    chunks: list[Chunk] = []
    hier: dict[str, str] = {}
    pre: list[str] = []
    i = 0
    n = len(lines)
    # preâmbulo: tudo antes do primeiro artigo
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
            # o nome do capítulo costuma vir na linha seguinte ("DOS PADRÕES ...")
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
                title = [x.text for x in body if "|" not in x.text]
                tbl = [Line("texto", "", "\n".join(title))] + rows if title else rows
                chunks.extend(_table_chunks(b, label, tbl, 0))
            else:
                # anexo em texto corrido: blocos de ~MAX_WORDS
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
        i += 1  # texto solto entre dispositivos (raro): ignorado
    # ids únicos (artigos repetidos em textos compilados, p.ex. redação antiga + nova)
    seen: dict[str, int] = {}
    for c in chunks:
        if c.id in seen:
            seen[c.id] += 1
            c.id = f"{c.id}#{seen[c.id]}"
        else:
            seen[c.id] = 0
    return chunks
