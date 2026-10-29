from copy import deepcopy
import json

import pandas as pd

from eval.run_rag_eval import tool_answer_checks


def test_tool_eval_rejects_wrong_band_date_and_missing_interval():
    q = {"local": "Campinas", "data_ferramenta": "2022-09-10", "horizonte_esperado": 0,
         "pergunta": "Qual a faixa?"}
    run = {"resposta": "18 µg/m³, IC 90%: 11 a 29,5. Faixa Boa.",
           "ferramentas": [{"ferramenta": "estimar_pm25", "args": {"local": "Campinas"},
                            "resultado": {"pm25_ug_m3": 18.0, "intervalo_90": [11.0, 29.5],
                                          "faixa_iqar": "Boa", "horizonte_dias": 0, "data_alvo": "2022-09-10"}}]}
    assert all(tool_answer_checks(q, run).values())
    bad = deepcopy(run)
    bad["resposta"] = "18 µg/m³, faixa Ruim."
    checks = tool_answer_checks(q, bad)
    assert not checks["faixa"] and not checks["intervalo"]
    bad = deepcopy(run)
    bad["ferramentas"][0]["resultado"]["data_alvo"] = "2022-09-11"
    assert not tool_answer_checks(q, bad)["data_horizonte"]


def test_rescore_accepts_legacy_numeric_booleans(tmp_path, monkeypatch):
    from eval import run_rag_eval as evaluate
    q = {"id": "q1", "tipo": "ferramenta", "pergunta": "Qual a faixa?", "local": "Campinas",
         "data_ferramenta": "2022-09-10", "horizonte_esperado": 0, "fatos_chave": [],
         "deve_chamar_ferramenta": True}
    run = {"pergunta": q["pergunta"], "encontrado": True, "trechos_citados": [],
           "resposta": "18,0 µg/m³, IC: 11 a 29,5, faixa Boa.", "validacao": {"ok": True},
           "ferramentas": [{"ferramenta": "estimar_pm25", "args": {"local": "Campinas"},
                            "resultado": {"pm25_ug_m3": 18, "intervalo_90": [11, 29.5], "faixa_iqar": "Boa",
                                          "data_alvo": "2022-09-10", "horizonte_dias": 0}}]}
    (tmp_path / "eval_runs_teste.jsonl").write_text(json.dumps(run) + "\n", encoding="utf-8")
    pd.DataFrame([{"id": "q1", "tipo": "ferramenta", "encontrado": True, "tempo_s": 1, "tokens": 10,
                   "acerto": True, "acerto_parcial": 1.0, "recusa_indevida": False,
                   "ferramenta_ok": 1.0, "fidelidade_auto": 1.0}]).to_csv(
        tmp_path / "rag_eval_detalhe_teste.csv", index=False)
    monkeypatch.setattr(evaluate, "load_questions", lambda: [q])
    monkeypatch.setattr(evaluate, "data_path", lambda _, name: tmp_path / name)
    result = evaluate.rescore("teste")
    assert result["acerto"] == 1.0 and result["ferramenta_ok"] == 1.0
