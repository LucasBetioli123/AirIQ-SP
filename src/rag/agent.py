"""Agente de normas ambientais com RAG e ferramenta de qualidade do ar.

Fluxo (poucas chamadas ao LLM, porque em CPU cada chamada custa segundos):
  1. PLANEJAR (LLM, JSON): quais buscas na base normativa e quais consultas de PM2.5
     (local, data, horizonte) a pergunta exige
  2. EXECUTAR as ferramentas (buscar_normas, estimar_pm25)
  3. RESPONDER (LLM, JSON): resposta em português, citando os trechos [n] e os
     resultados de ferramenta [F1..]. `encontrado=false` quando a base não cobre
  4. VALIDAR (sem LLM): citações existem? Os números da resposta aparecem nos trechos
     citados, nas ferramentas ou na pergunta? Se não, a resposta vai marcada (e uma
     regeneração é tentada)
  5. REGISTRAR tudo em logs/agent_runs.jsonl
Recusa honesta: se o melhor trecho tem relevância abaixo do limiar e nenhuma
ferramenta foi pedida, o agente recusa sem gerar, para não inventar.
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
from src.rag.tools import buscar_normas_varias, estimar_pm25, reference_today

REFUSAL = ("Não encontrei essa informação na base de normas consultada "
           "(Resoluções CONAMA 491/2018, 506/2024, 03/1990, 05/1989; Lei 6.938/1981; Decreto 99.274/1990; "
           "Lei 9.605/1998). Prefiro não responder sem uma fonte.")
# limiares de recusa sem gerar (calibrados em reports/metrics/rag_limiar_recusa.csv): escore do reranker
# (modo *_rerank) ou maior cosseno da busca vetorial (modos dense/hybrid)
RERANK_MIN = float(os.environ.get("RAG_RERANK_MIN", "0.8"))
COS_MIN = float(os.environ.get("RAG_COS_MIN", "0.675"))
KEEP_MIN = 0.05  # trechos (além dos 2 primeiros) com rerank abaixo disto não vão para o LLM


def gate_score(sources: list[dict], mode: str) -> float | None:
    """Relevância do melhor trecho para a recusa sem gerar. None = modo sem escore calibrável (BM25)."""
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
3. Ao comparar uma estimativa com um padrão legal, use o padrão que valia NA DATA DA ESTIMATIVA (campo padrao_24h_vigente da ferramenta), confirme o valor no trecho da norma nas FONTES e diga claramente se ultrapassa ou não. Uma norma revogada depois dessa data ainda vale para ela.
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
    """Modelos pequenos às vezes devolvem a chamada inteira: 'buscar_normas("padrão MP2,5")'."""
    m = re.match(r'^\s*\w+\(\s*["\']?(.*?)["\']?\s*\)\s*$', q, flags=re.S)
    return (m.group(1) if m else q).strip()


def norm_date(plan_date, question: str) -> str | None:
    """Data em que a norma deve estar vigente. Vem do plano ('AAAA-MM-DD' ou só 'AAAA') ou, se o
    plano não trouxer, de um ano citado na pergunta ('Em 2015, ...'). Sem data = vigente hoje."""
    d = str(plan_date or "").strip()
    # o plano às vezes inventa uma data (ex.: "2018-03-28" para "padrão da CONAMA 491"); só vale se a
    # própria pergunta tiver um ano ou uma data
    if not re.search(r"\b(?:19|20)\d{2}\b(?!\s*\?)", re.sub(r"\b\d+\s*/\s*(?:19|20)?\d{2}\b", "", question)):
        d = ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
        return d
    if re.fullmatch(r"(?:19|20)\d{2}", d):
        return f"{d}-07-01"
    m = re.search(r"\b(\d{2})/(\d{2})/((?:19|20)\d{2})\b", question)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    # ano solto ("em 2015", "no ano de 2020"), mas não o ano no número da norma ("491/2018")
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


def validate(answer: str, cited: list[int], sources: list[dict], tools: list[dict], question: str) -> dict:
    valid_ids = set(range(1, len(sources) + 1))
    bad = [c for c in cited if c not in valid_ids]
    inline = {int(x) for x in re.findall(r"\[(\d+)\]", answer)}
    support = " ".join(sources[i - 1]["texto"] + " " + sources[i - 1]["citacao"] + " " + (sources[i - 1].get("vigencia") or "")
                       for i in (set(cited) | inline) & valid_ids)
    support += " " + json.dumps(tools, ensure_ascii=False) + " " + question
    nums = {n for n in _numbers(re.sub(r"\[(?:F?\d+)\]", "", answer)) if len(n) > 0}
    sup_nums = _numbers(support)
    unsupported = sorted(n for n in nums if n not in sup_nums and not (n.isdigit() and int(n) < 10))
    return {"citacoes_invalidas": bad, "numeros_sem_suporte": unsupported, "cita_algo": bool((set(cited) | inline) & valid_ids),
            "ok": not bad and not unsupported}


class Agent:
    def __init__(self, provider: str | None = None, model: str | None = None, rag_mode: str | None = None,
                 k_per_search: int = 4, max_sources: int | None = None):
        self.llm = get_llm(provider, model)
        self.rag_mode = rag_mode or os.environ.get("RAG_MODE", "hybrid_rerank")
        self.k = k_per_search
        # cada trecho custa ~400-600 tokens de prompt; em CPU o prompt domina o tempo da resposta
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
            buscas.append(question)  # a pergunta original sempre entra como consulta
        date = norm_date(plan.get("data_da_norma"), question)
        # 2) ferramentas
        t1 = time.time()
        # união dos candidatos de todas as consultas; o reranker (se houver) roda uma vez só
        sources, per_query = buscar_normas_varias(buscas[:4], question, data=date, k=self.k, mode=self.rag_mode)
        for q, ids in per_query.items():
            run["ferramentas"].append({"ferramenta": "buscar_normas", "args": {"consulta": q, "data": date},
                                       "resultado": ids})
        sources = sources[: self.max_sources]
        best = gate_score(sources, self.rag_mode)
        # trecho com escore de reranker muito baixo só gasta prompt (o prompt domina o tempo em CPU)
        if best is not None and self.rag_mode.endswith("_rerank"):
            sources = [s for i, s in enumerate(sources) if i < 2 or (s["rerank"] or 0) >= KEEP_MIN]
        tools_out, seen_calls = [], set()
        for c in plan.get("pm25", [])[:3]:
            key = json.dumps(c, sort_keys=True, ensure_ascii=False)
            if key in seen_calls:  # o plano às vezes repete a mesma chamada
                continue
            seen_calls.add(key)
            r = estimar_pm25(c.get("local", ""), c.get("data"), int(c.get("horizonte", 0) or 0))
            tools_out.append(r)
            run["ferramentas"].append({"ferramenta": "estimar_pm25", "args": c, "resultado": r})
        run["tempo_ferramentas_s"] = round(time.time() - t1, 2)
        run["trechos"] = [{"n": i + 1, "id": s["id"], "citacao": s["citacao"], "score": s["score"], "rerank": s["rerank"]}
                          for i, s in enumerate(sources)]
        run["escore_relevancia"] = best
        # recusa sem gerar se a base claramente não cobre e não há ferramenta
        if not tools_out and best is not None and best < (RERANK_MIN if self.rag_mode.endswith("rerank") else COS_MIN):
            return self._finish(run, t0, {"encontrado": False, "resposta": REFUSAL, "citacoes": []}, sources, tools_out,
                                question, gate=True, log=log)
        # 3) resposta
        ctx = "\n\n".join(f"[{i + 1}] {s['citacao']}" + (f" ({s['vigencia']})" if s.get("vigencia") else "") +
                          f"\n{s['texto']}" for i, s in enumerate(sources))
        tctx = "\n".join(f"[F{i + 1}] {json.dumps(t, ensure_ascii=False)}" for i, t in enumerate(tools_out))
        # a data de referência só interessa quando há estimativa de PM2.5 ("hoje"/"amanhã"); nas perguntas
        # só de norma ela distrai o modelo
        today = reference_today()
        ref = (f"Data de referência do sistema: hoje = {today.date()}, amanhã = {(today + pd.Timedelta(days=1)).date()}.\n"
               if tools_out else "")
        user = (f"{ref}PERGUNTA: {question}\n\nFONTES:\n{ctx}\n\n" + (f"FERRAMENTAS:\n{tctx}\n" if tools_out else ""))
        a = self.llm.chat_json(ANSWER_SYS, user, ANSWER_SCHEMA, max_tokens=450)
        run["chamadas_llm"].append(_usage("resposta", a))
        ans = a.data or {"encontrado": False, "resposta": REFUSAL, "citacoes": []}
        v = validate(ans.get("resposta", ""), ans.get("citacoes", []), sources, tools_out, ref + question)
        if ans.get("encontrado") and not v["ok"]:
            # 4) uma regeneração apontando o problema
            fix = (f"{user}\nATENÇÃO: sua resposta anterior tinha problemas: {json.dumps(v, ensure_ascii=False)}. "
                   "Corrija usando apenas números e dispositivos presentes nas FONTES/FERRAMENTAS.")
            a2 = self.llm.chat_json(ANSWER_SYS, fix, ANSWER_SCHEMA, max_tokens=450)
            run["chamadas_llm"].append(_usage("correcao", a2))
            if a2.data:
                v2 = validate(a2.data.get("resposta", ""), a2.data.get("citacoes", []), sources, tools_out, ref + question)
                if v2["ok"] or len(v2["numeros_sem_suporte"]) < len(v["numeros_sem_suporte"]):
                    ans, v = a2.data, v2
        return self._finish(run, t0, ans, sources, tools_out, question, validation=v, log=log)

    def _finish(self, run, t0, ans, sources, tools_out, question, validation=None, gate=False, log=True):
        found = bool(ans.get("encontrado"))
        text = ans.get("resposta", "") if found else REFUSAL
        if found and not ans.get("resposta"):
            text, found = REFUSAL, False
        cited = sorted({c for c in ans.get("citacoes", []) if 1 <= c <= len(sources)} |
                       {int(x) for x in re.findall(r"\[(\d+)\]", text) if 1 <= int(x) <= len(sources)})
        refs = [f"[{c}] {sources[c - 1]['citacao']} — {sources[c - 1].get('url') or ''}".rstrip(" —") for c in cited]
        final = text + ("\n\nFontes:\n" + "\n".join(refs) if found and refs else "")
        if found and tools_out:
            final += ("\n\nObs.: estimativa do modelo da Parte 1 (satélite GOES-16 + ERA5 + rede CETESB); "
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
