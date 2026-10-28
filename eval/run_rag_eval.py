"""Avaliação ponta a ponta do agente sobre eval/rag_questions.yaml.

Métricas (método no RELATORIO.md):
  acerto         todos os `fatos_chave` da resposta esperada aparecem na resposta
                 (números/termos normalizados); `acerto_parcial` = fração dos fatos
  cit_precisao   fração dos trechos citados que pertencem aos documentos/dispositivos-ouro
  cit_recall     fração dos chunks-ouro efetivamente citados
  fidelidade     (a) checagem automática: números e citações da resposta têm suporte nos
                 trechos citados/ferramentas; (b) LLM-juiz (modelo DIFERENTE do gerador)
                 nota de 0 a 1 para "tudo o que foi afirmado está nas fontes citadas?"
  recusa         nas perguntas sem resposta na base: taxa de recusa correta. Nas com
                 resposta: taxa de recusa indevida
  ferramenta     nas perguntas de qualidade do ar: chamou estimar_pm25 com o local certo

  python -m eval.run_rag_eval [--llm caminho.gguf] [--judge caminho.gguf] [--tag nome]
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
from src.rag.agent import Agent


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = s.replace("µg/m³", "ug/m3").replace("μg/m³", "ug/m3")
    s = re.sub(r"(\d)\.(\d{3})\b", r"\1\2", s)          # 6.938 -> 6938
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)                 # 2,5 -> 2.5
    s = s.replace("º", "").replace("°", "")
    return re.sub(r"\s+", " ", s)


def fact_present(fact, answer: str) -> bool:
    """Um fato pode ser string ou lista de alternativas (qualquer uma vale)."""
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
        gold_docs = {g.split(":")[0] for g in gold}
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
                r["cit_precisao"] = float(np.mean([c.split(":")[0] in gold_docs for c in cited])) if gold else None
                r["cit_recall"] = float(np.mean([any(gold_match(c, [g]) for c in cited) for g in gold])) if gold else None
            v = run.get("validacao") or {}
            r["fidelidade_auto"] = float(v.get("ok", False)) if run["encontrado"] else None
            if judge_model and run["encontrado"]:
                # só os trechos CITADOS: fidelidade = o que foi afirmado está nas fontes que a resposta cita
                src = "\n\n".join(f"[{t['n']}] {chunks[t['id']]['sigla']}, {chunks[t['id']]['label']}\n{chunks[t['id']]['text']}"
                                  for t in run["trechos"] if t["id"] in chunks and t["n"] in run["citacoes"])
                tools = [t["resultado"] for t in run["ferramentas"] if t["ferramenta"] == "estimar_pm25"]
                if tools:
                    src += "\n\nFERRAMENTA:\n" + json.dumps(tools, ensure_ascii=False)
                to_judge.append((r, run, src))
        if q.get("deve_chamar_ferramenta"):
            calls = [t for t in run["ferramentas"] if t["ferramenta"] == "estimar_pm25"]
            loc = _norm(q.get("local", ""))
            r["ferramenta_ok"] = bool(calls) and any(loc in _norm(json.dumps(c["args"], ensure_ascii=False)) for c in calls)
        r["resposta"] = run["resposta"][:600]
        rows.append(r)
        pd.DataFrame(rows).to_csv(data_path("metrics", f"rag_eval_detalhe_{tag}.csv"), index=False)  # parcial
        print(f"{q['id']} [{q['tipo']}] encontrado={run['encontrado']} acerto={r.get('acerto', r.get('recusa_correta'))} "
              f"t={run['tempo_total_s']}s")
    if judge_model:
        # o juiz roda depois, com o gerador descarregado: dois LLMs de 4B juntos apertam 12 GB de RAM
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
    df = pd.DataFrame(rows)
    df.to_csv(data_path("metrics", f"rag_eval_detalhe_{tag}.csv"), index=False)
    return summarize(df, tag)


def rescore(tag: str):
    """Recalcula acerto/recusa com o gabarito atual a partir das respostas completas do log da avaliação
    (sem rodar o LLM de novo). As notas do juiz e os tempos são mantidos."""
    qs = {q["pergunta"]: q for q in load_questions()}
    runs = {}
    with open(data_path("logs", f"eval_runs_{tag}.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            runs[r["pergunta"]] = r  # a última execução de cada pergunta
    p = data_path("metrics", f"rag_eval_detalhe_{tag}.csv")
    df = pd.read_csv(p)
    for i, row in df.iterrows():
        q = next(q for q in qs.values() if q["id"] == row["id"])
        run = runs[q["pergunta"]]
        if not q.get("deve_recusar"):
            pres = [fact_present(f, run["resposta"]) for f in q.get("fatos_chave", [])]
            df.at[i, "acerto"] = bool(run["encontrado"] and all(pres))
            df.at[i, "acerto_parcial"] = float(np.mean(pres)) if pres else float(run["encontrado"])
    df.to_csv(p, index=False)
    return summarize(df, tag)


def summarize(df: pd.DataFrame, tag: str) -> dict:
    ans = df[df["tipo"] != "sem_resposta"]
    summ = {"modelo": tag, "n": len(df),
            "acerto": ans["acerto"].mean(), "acerto_parcial": ans["acerto_parcial"].mean(),
            "cit_precisao": ans.get("cit_precisao", pd.Series(dtype=float)).mean(),
            "cit_recall": ans.get("cit_recall", pd.Series(dtype=float)).mean(),
            "fidelidade_auto": ans["fidelidade_auto"].mean(),
            "fidelidade_juiz": ans["fidelidade_juiz"].mean() if "fidelidade_juiz" in ans else None,
            "recusa_correta (sem resposta)": df.loc[df["tipo"] == "sem_resposta", "recusa_correta"].mean(),
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
    ap.add_argument("--rescore", metavar="TAG", help="recalcula as notas de uma avaliação já feita")
    a = ap.parse_args()
    if a.rescore:
        rescore(a.rescore)
    else:
        main(a.llm, a.judge, a.tag, a.only)
