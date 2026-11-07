"""Environmental-law agent with RAG and an air-quality tool.

Flow (few LLM calls, because each one costs seconds on CPU):
  1. PLAN (LLM, JSON): which searches over the regulations and which PM2.5 lookups
     (place, date, horizon) the question needs
  2. RUN the tools (buscar_normas, estimar_pm25 and, for questions about which PM2.5
     standard stage applies on a date, padrao_mp25_vigente)
  3. ANSWER (LLM, JSON): answer in Portuguese, citing passages [n] and tool results
     [F1..]. `encontrado=false` (not found) when the base does not cover the question
  4. VALIDATE (no LLM): do the citations exist? Does every number in the answer appear in
     the cited passages, the tool outputs or the question? If not, one regeneration is
     attempted; if an unsourced number remains, the agent refuses
  5. LOG everything to logs/agent_runs.jsonl
Honest refusal: if the best passage scores below the relevance threshold and no tool was
requested, the agent refuses without generating, so it never makes things up.

Prompts, refusal text and answers are in Portuguese on purpose: the knowledge base is
Brazilian law and the users ask in Portuguese.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid

import pandas as pd

from src.rag.llm import get_llm
from src.rag.runlog import log_run
from src.rag.tools import (base_legal_padrao, buscar_normas_varias, estimar_pm25, padrao_call_from_question,
                           padrao_mp25_vigente, pm25_call_from_question, reference_today, trecho_por_id)

REFUSAL = ("Não encontrei essa informação na base de normas consultada "
           "(Resoluções CONAMA 491/2018, 506/2024, 03/1990, 05/1989; Lei 6.938/1981; Decreto 99.274/1990; "
           "Lei 9.605/1998). Prefiro não responder sem uma fonte.")
# refusal thresholds before generation (calibrated in reports/metrics/rag_limiar_recusa.csv): reranker
# score (*_rerank modes) or the top cosine of dense retrieval (dense/hybrid modes)
RERANK_MIN = float(os.environ.get("RAG_RERANK_MIN", "0.8"))
COS_MIN = float(os.environ.get("RAG_COS_MIN", "0.675"))
KEEP_MIN = 0.05  # passages (beyond the first 2) with a rerank score below this are not sent to the LLM


def gate_score(sources: list[dict], mode: str) -> float | None:
    """Relevance of the best passage, for refusing without generating. None = mode without a calibratable score (BM25)."""
    key = "rerank" if mode.endswith("_rerank") else "cos"
    vals = [s[key] for s in sources if s.get(key) is not None]
    return max(vals) if vals else None

PLAN_SYS = """Você planeja como responder perguntas sobre normas ambientais brasileiras e qualidade do ar no estado de São Paulo.
Ferramentas:
- buscar_normas(consulta): busca trechos em resoluções CONAMA e leis ambientais. Use 1 a 3 consultas curtas e específicas (termos técnicos da norma; separe assuntos diferentes em consultas diferentes).
- estimar_pm25(local, data, horizonte): modelo de PM2.5 (µg/m³) com intervalo de confiança para um município de SP ou "lat,lon". data = "hoje", "amanhã" ou "AAAA-MM-DD". horizonte = 1 se for previsão para o dia seguinte (amanhã), senão 0.
Só peça estimar_pm25 se a pergunta pedir concentração/qualidade do ar de um lugar e data. Se a pergunta comparar com um padrão legal, peça também uma busca sobre esse padrão.
data_da_norma: preencha com AAAA-MM-DD só se a pergunta for sobre a regra que valia numa data passada (ex.: "em 2015"); senão deixe vazio.
Responda apenas o JSON."""

PLAN_SCHEMA = {"type": "object", "properties": {
    "buscas": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    "pm25": {"type": "array", "maxItems": 3, "items": {"type": "object", "properties": {
        "local": {"type": "string"}, "data": {"type": "string"}, "horizonte": {"type": "integer", "enum": [0, 1]}},
        "required": ["local", "data", "horizonte"]}},
    "data_da_norma": {"type": "string"}},
    "required": ["buscas", "pm25"]}

ANSWER_SYS = """Você é um assistente técnico de normas ambientais. Responda em português, de forma objetiva, usando SOMENTE as FONTES fornecidas.
Regras:
1. Toda afirmação sobre norma deve citar a fonte entre colchetes, ex.: [1] ou [2][3]. Cite o dispositivo exato (ex.: "art. 4º, inciso II", "Anexo I") e o documento.
2. Valores de PM2.5 estimados/previstos vêm das ferramentas [F1], [F2]...; informe o valor e o intervalo de 90%, e a data.
3. Ao comparar uma estimativa com um padrão legal, use o padrão que valia NA DATA DA ESTIMATIVA (campo padrao_24h_vigente da ferramenta), confirme o valor no trecho da norma nas FONTES e diga claramente se ultrapassa ou não. Use o campo comparacao_com_padroes da ferramenta, que já traz a comparação calculada com cada etapa (PI-1, PI-2, ..., PF): não refaça a conta. Uma norma revogada depois dessa data ainda vale para ela.
4. Se as fontes não contêm a resposta, retorne encontrado=false. Não invente números, artigos ou normas.
5. Se precisar juntar dois trechos, cite ambos.
6. Fonte marcada como REVOGADO só vale para a época indicada; diga que foi revogada (e por qual norma) se isso for relevante.
7. Seja direto: no máximo 5 frases, sem repetir a pergunta.
Responda apenas o JSON."""

ANSWER_SCHEMA = {"type": "object", "properties": {
    "encontrado": {"type": "boolean"}, "resposta": {"type": "string"},
    "citacoes": {"type": "array", "items": {"type": "integer"}}},
    "required": ["encontrado", "resposta", "citacoes"]}


def _clean_query(q: str) -> str:
    """Small models sometimes return the whole call: 'buscar_normas("padrão MP2,5")'."""
    m = re.match(r'^\s*\w+\(\s*["\']?(.*?)["\']?\s*\)\s*$', q, flags=re.S)
    return (m.group(1) if m else q).strip()


def norm_date(plan_date, question: str) -> str | None:
    """Date on which the regulation must be in force. Comes from the plan ('YYYY-MM-DD' or just 'YYYY') or,
    if the plan has none, from a year cited in the question ('Em 2015, ...'). No date = in force today."""
    d = str(plan_date or "").strip()
    # the plan sometimes invents a date (e.g. "2018-03-28" for "padrão da CONAMA 491"); it only counts
    # if the question itself contains a year or a date
    if not re.search(r"\b(?:19|20)\d{2}\b(?!\s*\?)", re.sub(r"\b\d+\s*/\s*(?:19|20)?\d{2}\b", "", question)):
        d = ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
        return d
    if re.fullmatch(r"(?:19|20)\d{2}", d):
        return f"{d}-07-01"
    m = re.search(r"\b(\d{2})/(\d{2})/((?:19|20)\d{2})\b", question)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    # a standalone year ("em 2015", "no ano de 2020"), but not the year inside a regulation number ("491/2018")
    m = re.search(r"\b(?:em|no ano de|durante|até)\s+((?:19|20)\d{2})\b(?!\s*[-/])", question, flags=re.I)
    return f"{m.group(1)}-07-01" if m else None


def _numbers(s: str) -> set[str]:
    out = set()
    for n in re.findall(r"\d+(?:[.,]\d+)?", s):
        n = n.replace(",", ".")
        if n.count(".") == 1 and len(n.split(".")[1]) == 3 and n.split(".")[0].isdigit():  # 6.938 -> 6938
            n = n.replace(".", "")
        out.add(n.rstrip("0").rstrip(".") if "." in n else n)
    return out


def used_citations(answer: str, cited: list[int]) -> set[int]:
    """Citations [n] in the text win; the JSON `citacoes` field is used only when the text cites nothing.
    In evaluation, Qwen3-4B cited [1] in the text and listed [1..7] in the field (including a missing passage)."""
    inline = {int(x) for x in re.findall(r"\[(\d+)\]", answer)}
    return inline or {c for c in cited if isinstance(c, int)}


def validate(answer: str, cited: list[int], sources: list[dict], tools: list[dict], question: str) -> dict:
    valid_ids = set(range(1, len(sources) + 1))
    used = used_citations(answer, cited)
    bad = sorted(c for c in used if c not in valid_ids)
    support = " ".join(sources[i - 1]["texto"] + " " + sources[i - 1]["citacao"] + " " + (sources[i - 1].get("vigencia") or "")
                       for i in used & valid_ids)
    support += " " + json.dumps(tools, ensure_ascii=False) + " " + question
    # pollutant names (MP2,5, PM10) are not values
    plain = re.sub(r"\b(?:MP|PM)\s?(?:2[.,]5|10)\b", " ", re.sub(r"\[(?:F?\d+)\]", "", answer), flags=re.I)
    nums = {n for n in _numbers(plain) if len(n) > 0}
    sup_nums = _numbers(support)
    unsupported = sorted(n for n in nums if n not in sup_nums and not (n.isdigit() and int(n) < 10))
    return {"citacoes_invalidas": bad, "numeros_sem_suporte": unsupported, "cita_algo": bool(used & valid_ids),
            "ok": not bad and not unsupported}


def blocked_by_validation(ans: dict, v: dict | None) -> bool:
    return bool(ans.get("encontrado") and v and v.get("numeros_sem_suporte"))


class Agent:
    def __init__(self, provider: str | None = None, model: str | None = None, rag_mode: str | None = None,
                 k_per_search: int = 4, max_sources: int | None = None):
        self.llm = get_llm(provider, model)
        self.rag_mode = rag_mode or os.environ.get("RAG_MODE", "hybrid_rerank")
        self.k = k_per_search
        # each passage costs ~400-600 prompt tokens; on CPU the prompt dominates response time
        self.max_sources = max_sources or int(os.environ.get("AGENT_MAX_SOURCES", "6"))

    def ask(self, question: str, log: bool = True) -> dict:
        t0 = time.time()
        run = {"id": uuid.uuid4().hex[:10], "pergunta": question, "llm": self.llm.name, "rag_mode": self.rag_mode,
               "data_referencia_sistema": str(reference_today().date()), "chamadas_llm": [], "ferramentas": []}
        # 1) plano
        plan_user = f"Data de referência do sistema (hoje): {reference_today().date()}.\nPergunta: {question}"
        p = self.llm.chat_json(PLAN_SYS, plan_user, PLAN_SCHEMA, max_tokens=300)
        run["chamadas_llm"].append(_usage("plano", p))
        plan = p.data or {"buscas": [question], "pm25": []}
        buscas = [_clean_query(b) for b in plan.get("buscas", []) if isinstance(b, str) and b.strip()] or [question]
        if question not in buscas:
            buscas.append(question)  # the original question is always one of the queries
        date = norm_date(plan.get("data_da_norma"), question)
        # 2) ferramentas
        t1 = time.time()
        # union of candidates from every query; the reranker (if any) runs only once
        sources, per_query = buscar_normas_varias(buscas[:4], question, data=date, k=self.k, mode=self.rag_mode)
        for q, ids in per_query.items():
            run["ferramentas"].append({"ferramenta": "buscar_normas", "args": {"consulta": q, "data": date},
                                       "resultado": ids})
        sources = sources[: self.max_sources]
        best = gate_score(sources, self.rag_mode)
        # a passage with a very low reranker score only wastes prompt (the prompt dominates CPU time)
        if best is not None and self.rag_mode.endswith("_rerank"):
            sources = [s for i, s in enumerate(sources) if i < 2 or (s["rerank"] or 0) >= KEEP_MIN]
        tools_out, seen_calls = [], set()
        calls = [c for c in plan.get("pm25", []) if isinstance(c, dict)][:3]
        if not calls and (c := pm25_call_from_question(question)):
            calls = [c]
            run["ferramenta_por_regra"] = True  # the plan did not ask for it; the question rule did
        for c in calls:
            key = json.dumps(c, sort_keys=True, ensure_ascii=False)
            if key in seen_calls:  # o plano às vezes repete a mesma chamada
                continue
            seen_calls.add(key)
            r = estimar_pm25(c.get("local", ""), c.get("data"), int(c.get("horizonte", 0) or 0))
            tools_out.append(r)
            run["ferramentas"].append({"ferramenta": "estimar_pm25", "args": c, "resultado": r})
        pm25_used = bool(tools_out)
        if not pm25_used and padrao_call_from_question(question):
            # (with estimar_pm25 the stage in force already comes in the result, in comparacao_com_padroes)
            r = padrao_mp25_vigente(date)
            tools_out.append(r)
            run["ferramentas"].append({"ferramenta": "padrao_mp25_vigente", "args": {"data": date}, "resultado": r})
            # art. 4 (the schedule) goes into the context: without it, the model refused even with the tool (q23)
            have = {s["id"] for s in sources}
            extra = [t for i in base_legal_padrao(r) if i not in have and (t := trecho_por_id(i, self.rag_mode))]
            if extra:
                sources = sources[: max(2, self.max_sources - len(extra))] + extra
        run["tempo_ferramentas_s"] = round(time.time() - t1, 2)
        run["trechos"] = [{"n": i + 1, "id": s["id"], "citacao": s["citacao"], "score": s["score"], "rerank": s["rerank"]}
                          for i, s in enumerate(sources)]
        run["escore_relevancia"] = best
        # refuse without generating if the base clearly does not cover it and no tool is involved
        if not pm25_used and best is not None and best < (RERANK_MIN if self.rag_mode.endswith("rerank") else COS_MIN):
            return self._finish(run, t0, {"encontrado": False, "resposta": REFUSAL, "citacoes": []}, sources, tools_out,
                                question, gate=True, log=log)
        # 3) resposta
        ctx = "\n\n".join(f"[{i + 1}] {s['citacao']}" + (f" ({s['vigencia']})" if s.get("vigencia") else "") +
                          f"\n{s['texto']}" for i, s in enumerate(sources))
        tctx = "\n".join(f"[F{i + 1}] {json.dumps(t, ensure_ascii=False)}" for i, t in enumerate(tools_out))
        # the reference date only matters when there is a PM2.5 estimate ("hoje"/"amanhã"); in
        # regulation-only questions it distracts the model
        today = reference_today()
        ref = (f"Data de referência do sistema: hoje = {today.date()}, amanhã = {(today + pd.Timedelta(days=1)).date()}.\n"
               if pm25_used else "")
        user = (f"{ref}PERGUNTA: {question}\n\nFONTES:\n{ctx}\n\n" + (f"FERRAMENTAS:\n{tctx}\n" if tools_out else ""))
        a = self.llm.chat_json(ANSWER_SYS, user, ANSWER_SCHEMA, max_tokens=450)
        run["chamadas_llm"].append(_usage("resposta", a))
        ans = a.data or {"encontrado": False, "resposta": REFUSAL, "citacoes": []}
        v = validate(ans.get("resposta", ""), ans.get("citacoes", []), sources, tools_out, ref + question)
        if ans.get("encontrado") and not v["ok"]:
            # 4) one regeneration that points out the problem
            fix = (f"{user}\nATENÇÃO: sua resposta anterior tinha problemas: {json.dumps(v, ensure_ascii=False)}. "
                   "Corrija usando apenas números e dispositivos presentes nas FONTES/FERRAMENTAS.")
            a2 = self.llm.chat_json(ANSWER_SYS, fix, ANSWER_SCHEMA, max_tokens=450)
            run["chamadas_llm"].append(_usage("correcao", a2))
            if a2.data:
                v2 = validate(a2.data.get("resposta", ""), a2.data.get("citacoes", []), sources, tools_out, ref + question)
                if v2["ok"] or len(v2["numeros_sem_suporte"]) < len(v["numeros_sem_suporte"]):
                    ans, v = a2.data, v2
        if blocked_by_validation(ans, v):
            # a number is still unsourced after the correction: refusing beats delivering an invented value
            # (in evaluation, the model invented "desde 12 de abril de 2024" for the PM2.5 standard)
            run["bloqueado_pela_validacao"] = v["numeros_sem_suporte"]
            run["resposta_bloqueada"] = ans.get("resposta", "")
            ans = {"encontrado": False, "resposta": REFUSAL, "citacoes": []}
        return self._finish(run, t0, ans, sources, tools_out, question, validation=v, log=log)

    def _finish(self, run, t0, ans, sources, tools_out, question, validation=None, gate=False, log=True):
        found = bool(ans.get("encontrado"))
        text = ans.get("resposta", "") if found else REFUSAL
        if found and not ans.get("resposta"):
            text, found = REFUSAL, False
        cited = sorted(c for c in used_citations(text, ans.get("citacoes", [])) if 1 <= c <= len(sources))
        refs = [f"[{c}] {sources[c - 1]['citacao']} — {sources[c - 1].get('url') or ''}".rstrip(" —") for c in cited]
        final = text + ("\n\nFontes:\n" + "\n".join(refs) if found and refs else "")
        if found and any(f["ferramenta"] == "estimar_pm25" for f in run["ferramentas"]):
            final += ("\n\nObs.: estimativa do modelo de PM2.5 (satélite GOES-16 + ERA5 + rede CETESB); "
                      f"data de referência do sistema = {run['data_referencia_sistema']}.")
        run.update({"encontrado": found, "recusa_por_limiar": gate, "resposta": final, "citacoes": cited,
                    "trechos_citados": [sources[c - 1]["id"] for c in cited], "validacao": validation,
                    "tempo_total_s": round(time.time() - t0, 2),
                    "tokens_entrada": sum(c["tokens_entrada"] for c in run["chamadas_llm"]),
                    "tokens_saida": sum(c["tokens_saida"] for c in run["chamadas_llm"]),
                    "custo_estimado_usd": round(sum(c["custo_usd"] for c in run["chamadas_llm"]), 6)})
        if log:
            log_run(run)
        return run


def _usage(step, r):
    return {"etapa": step, "modelo": r.model, "tokens_entrada": r.prompt_tokens, "tokens_saida": r.completion_tokens,
            "segundos": round(r.seconds, 2), "custo_usd": r.cost_usd, "saida_bruta": r.text[:2000]}
