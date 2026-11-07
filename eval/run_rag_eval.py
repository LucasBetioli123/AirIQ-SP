"""End-to-end agent evaluation over eval/rag_questions.yaml (questions in Portuguese).

Metrics (method in docs/TECHNICAL_DETAILS.md; column names stay in Portuguese):
  acerto         accuracy: every `fatos_chave` (key fact) of the expected answer appears in the
                 answer (normalized numbers/terms); `acerto_parcial` = share of facts present
  cit_precisao   citation precision: share of cited passages matching the gold provision
  cit_recall     citation recall: share of gold chunks actually cited
  fidelidade     faithfulness: (a) automatic check that the answer's numbers and citations are
                 supported by the cited passages/tools; (b) LLM judge (a DIFFERENT model from the
                 generator) scoring 0-1 for "is everything stated in the cited sources?"
  recusa         refusals: correct-refusal rate on unanswerable questions; wrongful-refusal rate
                 on answerable ones
  ferramenta     tool use: place, target date, horizon, value, interval bounds and band when
                 asked; the numeric check uses the tool output recorded in the run

  python -m eval.run_rag_eval [--llm path.gguf] [--judge path.gguf] [--tag name]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd

from eval.bench_retrieval import gold_match, load_questions
from src.common.config import data_path
from src.rag.agent import Agent, blocked_by_validation


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = s.replace("µg/m³", "ug/m3").replace("μg/m³", "ug/m3")
    s = re.sub(r"(\d)\.(\d{3})\b", r"\1\2", s)          # 6.938 -> 6938
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)                 # 2,5 -> 2.5
    s = s.replace("º", "").replace("°", "")
    return re.sub(r"\s+", " ", s)


def fact_present(fact, answer: str) -> bool:
    """A fact can be a string or a list of alternatives (any one counts)."""
    alts = fact if isinstance(fact, list) else [fact]
    a = _norm(answer)
    for f in alts:
        f = _norm(f)
        if re.fullmatch(r"[\d.]+", f):
            if re.search(rf"(?<![\d.]){re.escape(f)}(?![\d])", a):
                return True
        elif f in a:
            return True
    return False


def tool_answer_checks(q: dict, run: dict) -> dict:
    """Check the tool's arguments, output and answer, including the correct band and the interval.

    The tool result recorded in the run is the numeric answer key for that run, so retrained
    models never require editing expected values by hand.
    """
    checks = {"local": False, "data_horizonte": False, "valor": False, "intervalo": False,
              "faixa": True, "padrao_citado": True}
    def has_value(value, text):
        return any(abs(float(n.replace(",", ".")) - float(value)) < 0.051
                   for n in re.findall(r"\d+(?:[.,]\d+)?", text))
    for call in run.get("ferramentas", []):
        if call["ferramenta"] != "estimar_pm25":
            continue
        args, result = call.get("args", {}), call.get("resultado", {})
        if _norm(q.get("local", "")) not in _norm(args.get("local", "")) or "erro" in result:
            continue
        checks["local"] = True
        horizon = q.get("horizonte_esperado", 0)
        day = q.get("data_ferramenta")
        if day == "amanhã":
            expected = pd.Timestamp(run["data_referencia_sistema"]) + pd.Timedelta(days=1)
        else:
            expected = pd.Timestamp(day)
        checks["data_horizonte"] = result.get("data_alvo") == str(expected.date()) and result.get("horizonte_dias") == horizon
        answer = run.get("resposta", "").split("\n\nFontes:")[0]
        checks["valor"] = result.get("pm25_ug_m3") is not None and has_value(result["pm25_ug_m3"], answer)
        interval = result.get("intervalo_90", [])
        checks["intervalo"] = len(interval) == 2 and all(has_value(v, answer) for v in interval)
        if "faixa" in q["pergunta"].lower():
            checks["faixa"] = fact_present(result.get("faixa_iqar", "__ausente__"), answer)
        if q.get("etapa_comparacao"):
            from src.aq.aqi import STAGES_24H
            stage = q["etapa_comparacao"]
            norm = result.get("padrao_24h_vigente", {}).get("norma", "").split(",")[0]
            limit = STAGES_24H.get(norm, {}).get(stage)
            checks["padrao_citado"] = limit is not None and has_value(limit, answer) and stage.lower() in answer.lower()
    return checks


JUDGE_SYS = ("Você avalia fidelidade de respostas. Dadas FONTES e uma RESPOSTA, verifique se TODAS as afirmações "
             "factuais da resposta (números, artigos, obrigações) estão sustentadas pelas fontes. Responda JSON com "
             "nota de 0 a 1 (1 = tudo sustentado) e a lista de afirmações sem suporte.")
JUDGE_SCHEMA = {"type": "object", "properties": {"nota": {"type": "number"},
                                                  "sem_suporte": {"type": "array", "items": {"type": "string"}}},
                "required": ["nota", "sem_suporte"]}


def judge(llm, run: dict, sources_text: str) -> dict:
    if not run["encontrado"]:
        return {"nota": None, "sem_suporte": []}
    user = f"FONTES:\n{sources_text}\n\nRESPOSTA:\n{run['resposta']}"
    r = llm.chat_json(JUDGE_SYS, user, JUDGE_SCHEMA, max_tokens=300)
    return r.data or {"nota": None, "sem_suporte": []}


def _merge(rows: list[dict], tag: str, only) -> pd.DataFrame:
    """With --only, the re-run questions replace their rows in the saved evaluation with the same tag."""
    df = pd.DataFrame(rows)
    p = data_path("metrics", f"rag_eval_detalhe_{tag}.csv")
    if only and p.exists():
        prev = pd.read_csv(p)
        order = [q["id"] for q in load_questions()]
        df = pd.concat([prev[~prev["id"].isin(df["id"])], df])
        df = df.sort_values("id", key=lambda s: s.map(order.index)).reset_index(drop=True)
    return df


def main(llm_model=None, judge_model=None, tag=None, only=None):
    qs = load_questions()
    if only:
        qs = [q for q in qs if q["id"] in only]
    agent = Agent(model=llm_model)
    tag = tag or agent.llm.name.replace(".gguf", "")
    os.environ["AGENT_LOG"] = str(data_path("logs", f"eval_runs_{tag}.jsonl"))
    from src.rag.tools import get_retriever
    chunks = {c["id"]: c for c in get_retriever(agent.rag_mode).chunks}
    rows, to_judge = [], []
    for q in qs:
        run = agent.ask(q["pergunta"])
        cited = run["trechos_citados"]
        gold = q.get("gold") or []
        r = {"id": q["id"], "tipo": q["tipo"], "encontrado": run["encontrado"], "tempo_s": run["tempo_total_s"],
             "tokens": run["tokens_entrada"] + run["tokens_saida"]}
        if q.get("deve_recusar"):
            r["recusa_correta"] = not run["encontrado"]
        else:
            facts = q.get("fatos_chave", [])
            pres = [fact_present(f, run["resposta"]) for f in facts]
            r["acerto"] = bool(run["encontrado"] and all(pres))
            r["acerto_parcial"] = float(np.mean(pres)) if pres else float(run["encontrado"])
            r["recusa_indevida"] = not run["encontrado"]
            if run["encontrado"] and cited:
                r["cit_precisao"] = float(np.mean([gold_match(c, gold) for c in cited])) if gold else None
                r["cit_recall"] = float(np.mean([any(gold_match(c, [g]) for c in cited) for g in gold])) if gold else None
            v = run.get("validacao") or {}
            r["fidelidade_auto"] = float(v.get("ok", False)) if run["encontrado"] else None
            if judge_model and run["encontrado"]:
                # CITED passages only: faithfulness = what was stated is in the sources the answer cites
                src = "\n\n".join(f"[{t['n']}] {chunks[t['id']]['sigla']}, {chunks[t['id']]['label']}\n{chunks[t['id']]['text']}"
                                  for t in run["trechos"] if t["id"] in chunks and t["n"] in run["citacoes"])
                tools = [t["resultado"] for t in run["ferramentas"] if t["ferramenta"] != "buscar_normas"]
                if tools:
                    src += "\n\nFERRAMENTA:\n" + json.dumps(tools, ensure_ascii=False)
                to_judge.append((r, run, src))
        if q.get("deve_chamar_ferramenta"):
            checks = tool_answer_checks(q, run)
            r["ferramenta_ok"] = all(checks.values())
            r["checagens_ferramenta"] = json.dumps(checks, ensure_ascii=False)
            # without the estimate there is no answer: citing only the legal standard does not count
            r["acerto"] = bool(r.get("acerto")) and r["ferramenta_ok"]
        r["resposta"] = run["resposta"][:600]
        rows.append(r)
        _merge(rows, tag, only).to_csv(data_path("metrics", f"rag_eval_detalhe_{tag}.csv"), index=False)  # parcial
        print(f"{q['id']} [{q['tipo']}] encontrado={run['encontrado']} acerto={r.get('acerto', r.get('recusa_correta'))} "
              f"t={run['tempo_total_s']}s")
    if judge_model:
        # the judge runs afterwards, with the generator unloaded: two 4B LLMs together strain 12 GB of RAM
        import gc

        from src.rag.llm import LLM, get_llm
        del agent
        get_llm.cache_clear()
        gc.collect()
        jl = LLM(model=judge_model)
        for r, run, src in to_judge:
            j = judge(jl, run, src)
            r["fidelidade_juiz"] = j.get("nota")
            r["juiz_sem_suporte"] = "; ".join(j.get("sem_suporte", []))[:300]
            print(f"juiz {r['id']}: {r['fidelidade_juiz']}")
    df = _merge(rows, tag, only)
    df.to_csv(data_path("metrics", f"rag_eval_detalhe_{tag}.csv"), index=False)
    return summarize(df, tag)


def rescore(tag: str):
    """Recompute accuracy/refusals with the current answer keys from the full answers in the evaluation log
    (without running the LLM again). Judge scores and latencies are kept."""
    qs = {q["pergunta"]: q for q in load_questions()}
    runs = {}
    with open(data_path("logs", f"eval_runs_{tag}.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            runs[r["pergunta"]] = r  # the latest run of each question
    p = data_path("metrics", f"rag_eval_detalhe_{tag}.csv")
    df = pd.read_csv(p)
    if "checagens_ferramenta" not in df:
        df["checagens_ferramenta"] = pd.Series(pd.NA, index=df.index, dtype="string")
    # older CSVs may encode True as 1.0. Pandas 3 does not allow assigning a bool
    # into a float column; also keep unevaluated cells as NA.
    for col in ("encontrado", "acerto", "recusa_correta", "recusa_indevida", "ferramenta_ok"):
        if col in df:
            df[col] = df[col].map(lambda v: pd.NA if pd.isna(v) else str(v).lower() in ("true", "1", "1.0")).astype("boolean")
    for i, row in df.iterrows():
        q = next(q for q in qs.values() if q["id"] == row["id"])
        run = runs[q["pergunta"]]
        if q.get("deve_chamar_ferramenta"):
            checks = tool_answer_checks(q, run)
            df.at[i, "ferramenta_ok"] = all(checks.values())
            df.at[i, "checagens_ferramenta"] = json.dumps(checks, ensure_ascii=False)
        found = bool(run["encontrado"])
        if blocked_by_validation(run, run.get("validacao")):
            # run recorded before the rule "unsourced number after the correction -> refuse"
            # (src/rag/agent.py): the rule is deterministic, so it is applied to the log here
            found = False
            for c in ("cit_precisao", "cit_recall", "fidelidade_auto", "fidelidade_juiz"):
                if c in df:
                    df.at[i, c] = np.nan
        df.at[i, "encontrado"] = found
        gold = q.get("gold") or []
        cited = run.get("trechos_citados", [])
        if found and gold and cited:
            df.at[i, "cit_precisao"] = float(np.mean([gold_match(c, gold) for c in cited]))
            df.at[i, "cit_recall"] = float(np.mean([any(gold_match(c, [g]) for c in cited) for g in gold]))
        if q.get("deve_recusar"):
            df.at[i, "recusa_correta"] = not found
        else:
            pres = [fact_present(f, run["resposta"]) for f in q.get("fatos_chave", [])] if found else \
                [False] * len(q.get("fatos_chave", []))
            df.at[i, "acerto"] = bool(found and all(pres)) and \
                (not q.get("deve_chamar_ferramenta") or bool(df.at[i, "ferramenta_ok"]))
            df.at[i, "acerto_parcial"] = float(np.mean(pres)) if pres else float(found)
            df.at[i, "recusa_indevida"] = not found
    df.to_csv(p, index=False)
    return summarize(df, tag)


def summarize(df: pd.DataFrame, tag: str) -> dict:
    ans = df[df["tipo"] != "sem_resposta"]
    summ = {"modelo": tag, "n": len(df), "metodo_avaliacao": "dispositivo-exato-e-ferramenta-v2",
            "n_respondidas": int(ans["encontrado"].sum()),
            "n_fidelidade_juiz": int(ans["fidelidade_juiz"].notna().sum()) if "fidelidade_juiz" in ans else 0,
            "acerto": ans["acerto"].mean(), "acerto_parcial": ans["acerto_parcial"].mean(),
            "cit_precisao": ans.get("cit_precisao", pd.Series(dtype=float)).mean(),
            "cit_recall": ans.get("cit_recall", pd.Series(dtype=float)).mean(),
            "fidelidade_auto": ans["fidelidade_auto"].mean(),
            "fidelidade_juiz": ans["fidelidade_juiz"].mean() if "fidelidade_juiz" in ans else None,
            "recusa_correta (sem resposta)": (df.loc[df["tipo"] == "sem_resposta", "recusa_correta"].mean()
                                              if "recusa_correta" in df else None),
            "recusa_indevida": ans["recusa_indevida"].mean(),
            "ferramenta_ok": df["ferramenta_ok"].mean() if "ferramenta_ok" in df else None,
            "tempo_medio_s": df["tempo_s"].mean(), "tokens_medio": df["tokens"].mean()}
    for t, g in ans.groupby("tipo"):
        summ[f"acerto_{t}"] = g["acerto"].mean()
    out = data_path("metrics", "rag_eval_resumo.csv")
    prev = pd.read_csv(out) if out.exists() else pd.DataFrame()
    prev = prev[prev.get("modelo", pd.Series(dtype=str)) != tag] if len(prev) else prev
    pd.concat([prev, pd.DataFrame([summ])]).to_csv(out, index=False, float_format="%.4f")
    print(json.dumps(summ, indent=1, ensure_ascii=False, default=float))
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm")
    ap.add_argument("--judge")
    ap.add_argument("--tag")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--rescore", metavar="TAG", help="recompute the scores of an evaluation already run")
    a = ap.parse_args()
    if a.rescore:
        rescore(a.rescore)
    else:
        main(a.llm, a.judge, a.tag, a.only)
