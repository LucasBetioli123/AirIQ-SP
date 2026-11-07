from pathlib import Path

from src.rag.chunk import chunk_document
from src.rag.parse import classify, read_lines

TXT = """RESOLUÇÃO Nº 999, DE 1 DE JANEIRO DE 2020
Dispõe sobre padrões de teste.
O CONSELHO resolve:
CAPÍTULO I
DAS DISPOSIÇÕES GERAIS
Art. 1º Esta Resolução estabelece padrões.
Art. 2º Para efeito desta Resolução, considera-se:
I - poluente atmosférico: qualquer forma de matéria;
II - padrão de qualidade do ar: valor de concentração;
a) padrões intermediários;
b) padrão final;
Parágrafo único. Os padrões aplicam-se em todo o território.
Art. 3º Ficam estabelecidos os padrões do Anexo I.
§ 1º Os padrões intermediários são PI-1, PI-2 e PI-3.
§ 2º O padrão final é PF.
ANEXO I
Poluente | Período | PI-1 | PI-2
MP2,5 | 24 horas | 60 | 50
"""


def test_classify():
    assert classify("Art. 12. Texto")[1] == "art. 12"
    assert classify("Art. 4º A qualidade")[1] == "art. 4º"
    assert classify("§ 2º O padrão")[1] == "§ 2º"
    assert classify("II - padrão de qualidade")[1] == "inciso II"
    assert classify("Parágrafo único. Os")[1] == "parágrafo único"
    assert classify("a) padrões")[1] == "alínea a"
    assert classify("Isto é texto")[0] == "texto"


def test_chunks_preserve_structure(tmp_path: Path):
    p = tmp_path / "doc.txt"
    p.write_text(TXT, encoding="utf-8")
    doc = {"id": "teste", "sigla": "Resolução Teste 999/2020", "titulo": "Resolução 999"}
    chunks = chunk_document(doc, read_lines(p))
    ids = [c.id for c in chunks]
    assert ids[0] == "teste:preambulo"
    assert "teste:art2" in ids and "teste:anexoi" in ids
    art2 = next(c for c in chunks if c.id == "teste:art2")
    assert "art. 2º, inciso II" in art2.dispositivos
    assert "art. 2º, inciso II, alínea b" in art2.dispositivos
    assert "art. 2º, parágrafo único" in art2.dispositivos
    assert art2.path.startswith("CAPÍTULO I")
    anexo = next(c for c in chunks if c.id == "teste:anexoi")
    assert "MP2,5 | 24 horas | 60 | 50" in anexo.text


def test_classify_suffix_and_planalto_ordinal():
    # "Art. 1º - Esta Lei" is not art. 1º-E; "Art. 9 o -A." (Planalto HTML) is art. 9º-A
    assert classify("Art. 1º - Esta Lei estabelece")[1] == "art. 1º"
    assert classify("Art. 5º - A política")[1] == "art. 5º"
    assert classify("Art. 9 o -A. O proprietário")[1] == "art. 9º-A"
    assert classify("Art. 17-A. São estabelecidos")[1] == "art. 17-A"
    assert classify("2.2.1 - Ficam estabelecidos")[1] == "item 2.2.1"


def test_rewrite_standards_table():
    from src.rag.parse import _rewrite_tables

    paras = ["ANEXO I", "PADRÕES DE QUALIDADE DO AR",
             "Poluente Atmosférico Período deReferência PI-1 PI-2 PI-3 PI-4 PF",
             "µg/m³ µg/m³ µg/m³ µg/m³ µg/m³ ppm",
             "Material Particulado - MP2,5 24 horas 60 50 37 25 15 -",
             "Anual¹ 20 17 15 10 5 -",
             "Monóxido de Carbono - CO 8 horas ³ - - - - 9",
             "1 - média aritmética anual"]
    out = _rewrite_tables(paras)
    assert "Material Particulado - MP2,5 | 24 horas | PI-1: 60 µg/m³ | PI-2: 50 µg/m³ | PI-3: 37 µg/m³ | " \
           "PI-4: 25 µg/m³ | PF: 15 µg/m³" in out
    assert any(r.startswith("Material Particulado - MP2,5 | Anual |") and "PF: 5 µg/m³" in r for r in out)
    # an empty cell vanishes in the PDF: the CO row is right-aligned (9 ppm lands in the PF ppm column)
    assert any(r.startswith("Monóxido de Carbono - CO | 8 horas") and r.endswith("PF (ppm): 9") for r in out)
    assert "Nota 1 - média aritmética anual" in out


def test_partial_revocation_and_numbered_items(tmp_path: Path):
    p = tmp_path / "doc.txt"
    p.write_text("RESOLUÇÃO Nº 5\nO CONSELHO resolve:\nI - Instituir o programa.\n2 - ESTRATÉGIAS\n"
                 "A estratégia básica.\n2.1 - LIMITES\nTexto do item.\n", encoding="utf-8")
    doc = {"id": "r5", "sigla": "Res. 5", "titulo": "Res. 5", "vigencia_inicio": "1989-08-25",
           "revogacoes_parciais": [{"dispositivos": ["item 2.1"], "vigencia_fim": "2018-11-20",
                                    "revogado_por": "Res. 491, art. 14"}]}
    chunks = {c.id: c for c in chunk_document(doc, read_lines(p))}
    assert {"r5:itemi", "r5:item2", "r5:item21"} <= set(chunks)
    assert chunks["r5:item21"].vigencia_fim == "2018-11-20"
    assert chunks["r5:item21"].meta["revogado_por"] == "Res. 491, art. 14"
    assert chunks["r5:item2"].vigencia_fim is None


def test_mentioned_docs_without_year_and_diversify():
    from src.rag.retrieve import Hit, diversify, mentioned_docs

    ch = [{"doc_id": d} for d in ("conama_491_2018", "conama_005_1989", "lei_9605_1998", "lei_6938_1981")]
    assert mentioned_docs("ultrapassa o padrão da Resolução CONAMA 491?", ch) == {"conama_491_2018"}
    assert mentioned_docs("o que diz a Lei nº 6.938?", ch) == {"lei_6938_1981"}
    assert mentioned_docs("pena da Lei 9605", ch) == {"lei_9605_1998"}
    assert mentioned_docs("PM2,5 de 24 horas acima de 491 µg/m³", ch) == set()
    rows = [Hit({"doc_id": "d", "label": f"Anexo I (linha {i})"}, 1.0) for i in range(4)]
    art = Hit({"doc_id": "d", "label": "art. 4º"}, 0.5)
    out = diversify(rows + [art], max_per_group=2)
    assert out[2] is art and len(out) == 5
