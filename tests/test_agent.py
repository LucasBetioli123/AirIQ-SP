"""Test the agent flow (plan -> tools -> cited answer -> validation -> log) with a fake LLM and a
FICTITIOUS regulation base (in Portuguese, like the real one). Needs no models and no network."""
import json

import pytest

from src.rag import agent as A
from src.rag.llm import LLMResult

FAKE_CHUNKS = [
    {"id": "res_999_2020:art3", "doc_id": "res_999_2020", "sigla": "Resolução Teste 999/2020", "titulo": "Teste",
     "path": "", "label": "art. 3º", "dispositivos": ["art. 3º"], "text": "Art. 3º O padrão de 24 horas é 60 µg/m³.",
     "vigencia_inicio": "2020-01-01", "vigencia_fim": None, "url": "http://exemplo", "revogado": False, "meta": {}},
]


class FakeLLM:
    name = "fake"

    def __init__(self, answers):
        self.answers = list(answers)

    def chat_json(self, system, user, schema, max_tokens=700, temperature=0.0):
        data = self.answers.pop(0)
        return LLMResult(json.dumps(data), data, 10, 5, 0.01, "fake", 0.0)


@pytest.fixture
def patched(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_LOG", str(tmp_path / "runs.jsonl"))
    monkeypatch.setattr(A, "reference_today", lambda: __import__("pandas").Timestamp("2023-04-03"))

    def fake_search(consultas, pergunta, data=None, k=4, mode=None):
        c = FAKE_CHUNKS[0]
        hit = {"id": c["id"], "citacao": f"{c['sigla']}, {c['label']}", "doc_id": c["doc_id"], "dispositivos": [],
               "texto": c["text"], "url": c["url"], "score": 0.9, "rerank": 0.9, "cos": 0.8, "rank_bm25": 1,
               "rank_dense": 1}
        return [hit], {q: [c["id"]] for q in consultas}

    calls = []

    def fake_pm25(local, data=None, horizonte=0):
        calls.append((local, data, horizonte))
        return {"local": local, "pm25_ug_m3": 18.2, "intervalo_90": [11.0, 29.5], "data_alvo": "2023-04-04"}

    monkeypatch.setattr(A, "buscar_normas_varias", fake_search)
    monkeypatch.setattr(A, "estimar_pm25", fake_pm25)
    return tmp_path, calls


def test_agent_calls_tool_and_cites(patched):
    tmp, calls = patched
    llm = FakeLLM([
        {"buscas": ["padrão 24 horas"], "pm25": [{"local": "Sorocaba", "data": "amanhã", "horizonte": 1}]},
        {"encontrado": True, "citacoes": [1],
         "resposta": "Previsão de 18.2 µg/m³ (IC 90%: 11.0-29.5) [F1]; não ultrapassa o padrão de 60 µg/m³ [1]."},
    ])
    ag = A.Agent.__new__(A.Agent)
    ag.llm, ag.rag_mode, ag.k, ag.max_sources = llm, "hybrid_rerank", 4, 8
    r = ag.ask("PM2.5 amanhã em Sorocaba ultrapassa o padrão?")
    assert calls == [("Sorocaba", "amanhã", 1)]
    assert r["encontrado"] and r["citacoes"] == [1] and r["validacao"]["ok"]
    assert "Resolução Teste 999/2020, art. 3º" in r["resposta"]
    log = [json.loads(l) for l in open(tmp / "runs.jsonl", encoding="utf-8")]
    assert log[0]["ferramentas"][-1]["ferramenta"] == "estimar_pm25" and "tempo_total_s" in log[0]


def test_agent_flags_unsupported_number(patched):
    llm = FakeLLM([
        {"buscas": ["padrão"], "pm25": []},
        {"encontrado": True, "citacoes": [1], "resposta": "O padrão é 75 µg/m³ [1]."},
        {"encontrado": True, "citacoes": [1], "resposta": "O padrão é 75 µg/m³ [1]."},
    ])
    ag = A.Agent.__new__(A.Agent)
    ag.llm, ag.rag_mode, ag.k, ag.max_sources = llm, "hybrid_rerank", 4, 8
    r = ag.ask("Qual o padrão?")
    assert "75" in r["validacao"]["numeros_sem_suporte"]
    # the number was still unsourced after the correction: refuse instead of delivering it
    assert not r["encontrado"] and r["resposta"].startswith("Não encontrei") and r["bloqueado_pela_validacao"] == ["75"]


def test_agent_refuses_when_not_found(patched):
    llm = FakeLLM([{"buscas": ["ruído"], "pm25": []}, {"encontrado": False, "citacoes": [], "resposta": ""}])
    ag = A.Agent.__new__(A.Agent)
    ag.llm, ag.rag_mode, ag.k, ag.max_sources = llm, "hybrid_rerank", 4, 8
    r = ag.ask("Qual o limite de ruído noturno?")
    assert not r["encontrado"] and r["resposta"].startswith("Não encontrei")


def test_validate_numbers():
    src = [{"texto": "Art. 4º O prazo é de 3 (três) anos e o padrão 60 µg/m³.", "citacao": "Res X, art. 4º"}]
    assert A.validate("O prazo é de 3 anos e o padrão 60 µg/m³ [1].", [1], src, [], "q")["ok"]
    assert not A.validate("O padrão é 61 µg/m³ [1].", [1], src, [], "q")["ok"]
    assert A.validate("Ver [7].", [7], src, [], "q")["citacoes_invalidas"] == [7]
    # the text cites [1]; the `citacoes` field also listed a missing passage: the text wins
    assert A.validate("O padrão é 60 µg/m³ [1] [F1].", [1, 7], src, [], "q")["ok"]
    assert A.used_citations("Sem número no texto.", [1]) == {1}
    assert A.validate("O padrão de MP2,5 e de PM10 é 60 µg/m³ [1].", [1], src, [], "q")["ok"]  # a name is not a value


def test_norm_date_and_query_cleanup():
    assert A.norm_date("2015-06-01", "Em 2015, qual era o padrão?") == "2015-06-01"
    assert A.norm_date("2015", "Em 2015, qual era o padrão?") == "2015-07-01"
    # a date invented by the plan, with no year in the question, is discarded
    assert A.norm_date("2018-03-28", "Sorocaba amanhã ultrapassa o padrão da CONAMA 491?") is None
    assert A.norm_date(None, "Em 2015, qual resolução definia os padrões?") == "2015-07-01"
    # the year inside a regulation number is not a validity date
    assert A.norm_date(None, "O que diz a Resolução CONAMA 491/2018?") is None
    assert A.norm_date("", "Em 2021-08-24 o ar em Sorocaba atendia ao padrão?") is None
    assert A._clean_query('buscar_normas("padrão de MP2,5")') == "padrão de MP2,5"
    assert A._clean_query("padrão de MP2,5 (24 h)") == "padrão de MP2,5 (24 h)"


def test_agent_calls_tool_when_plan_forgets(patched):
    # the plan does not request the tool, but the question names PM2.5, a municipality and a date: the rule calls it
    tmp, calls = patched
    llm = FakeLLM([
        {"buscas": [], "pm25": []},
        {"encontrado": True, "citacoes": [], "resposta": "Estimativa de 18.2 µg/m³ [F1], faixa Boa."},
    ])
    ag = A.Agent.__new__(A.Agent)
    ag.llm, ag.rag_mode, ag.k, ag.max_sources = llm, "hybrid_rerank", 4, 8
    r = ag.ask("Qual a estimativa de PM2.5 em Ribeirão Preto em 2022-09-10?")
    assert calls == [("Ribeirão Preto", "2022-09-10", 0)]
    assert r["ferramenta_por_regra"] and r["encontrado"]


def test_padrao_vigente_tool():
    from src.rag.tools import padrao_call_from_question, padrao_mp25_vigente
    assert padrao_call_from_question("Em 2020, qual era o padrão nacional de MP2,5 para 24 horas aplicado?")
    assert padrao_call_from_question("Quando entram em vigor os padrões PI-3 e PI-4 da 506?")
    assert not padrao_call_from_question("Qual o PF de MP2,5 em 24 horas na CONAMA 491/2018?")  # a value, not validity
    r = padrao_mp25_vigente("2020-07-01")
    assert r["padrao_24h_vigente_na_data"]["etapa"] == "PI-1" and r["vigente_desde"] == "2018-11-21"
    r = padrao_mp25_vigente("2026-10-28")
    assert r["padrao_24h_vigente_na_data"]["valor"] == 50 and r["vigente_desde"] == "2025-01-01"
