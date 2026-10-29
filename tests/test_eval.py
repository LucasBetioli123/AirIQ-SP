from copy import deepcopy

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
