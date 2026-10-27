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
