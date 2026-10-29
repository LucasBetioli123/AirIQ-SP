"""Interface de chat mínima para o agente (FastAPI + uma página HTML, sem dependência nova).

  uvicorn src.rag.app:app --port 7860      (ou `make chat`)
  abrir http://localhost:7860

POST /ask {"pergunta": "..."} devolve a execução completa do agente (a mesma que vai para o log):
resposta com citações, trechos recuperados, chamadas de ferramenta, tempo e custo.
"""
from __future__ import annotations

import functools
import threading

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Agente de normas ambientais + PM2.5", version="1.0")
_lock = threading.Lock()  # um LLM local atende uma pergunta por vez


@functools.lru_cache(maxsize=1)
def _agent():
    from src.rag.agent import Agent
    return Agent()


class Pergunta(BaseModel):
    pergunta: str


@app.post("/ask")
def ask(p: Pergunta):
    with _lock:
        return _agent().ask(p.pergunta)


PAGE = """<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Agente de normas ambientais</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#1d2433}
 #log{display:flex;flex-direction:column;gap:.8rem;margin-bottom:1rem}
 .q{align-self:flex-end;background:#e8eefc;padding:.6rem .9rem;border-radius:10px;max-width:80%}
 .a{background:#f5f6f8;padding:.8rem 1rem;border-radius:10px;white-space:pre-wrap}
 .meta{font-size:.8rem;color:#5b6475;margin-top:.4rem} details{font-size:.85rem;margin-top:.4rem}
 form{display:flex;gap:.5rem} input{flex:1;padding:.6rem;font-size:1rem} button{padding:.6rem 1rem}
</style></head><body>
<h2>Agente de normas ambientais + PM2.5 (SP)</h2>
<p class="meta">Base: Res. CONAMA 491/2018, 506/2024, 03/1990, 05/1989; Lei 6.938/1981; Decreto 99.274/1990;
Lei 9.605/1998. LLM local em CPU: cada resposta pode levar vários minutos.
Os dados de PM2.5 são históricos (2021–2023); "hoje/amanhã" usam uma data de demonstração,
informada em cada resposta, e não a data atual.</p>
<div id="log"></div>
<form id="f"><input id="q" placeholder="Ex.: qual o PM2.5 previsto para Campinas amanhã e ele ultrapassa o padrão da CONAMA 491?" required>
<button>Perguntar</button></form>
<script>
const log = document.getElementById('log'), f = document.getElementById('f'), q = document.getElementById('q');
function el(cls, text){ const d = document.createElement('div'); d.className = cls; d.textContent = text; return d; }
f.onsubmit = async (e) => {
  e.preventDefault(); const pergunta = q.value; q.value = '';
  log.appendChild(el('q', pergunta)); const a = el('a', 'pensando...'); log.appendChild(a);
  try {
    const r = await (await fetch('/ask', {method:'POST', headers:{'Content-Type':'application/json'},
                                         body: JSON.stringify({pergunta})})).json();
    a.textContent = r.resposta;
    a.appendChild(el('meta', `${r.tempo_total_s} s · ${r.llm} · ${r.rag_mode} · custo US$ ${r.custo_estimado_usd}`));
    const det = document.createElement('details'); det.appendChild(Object.assign(document.createElement('summary'),
      {textContent: 'trechos recuperados e ferramentas'}));
    det.appendChild(el('', JSON.stringify({trechos: r.trechos, ferramentas: r.ferramentas}, null, 1)));
    det.lastChild.style.whiteSpace = 'pre-wrap'; a.appendChild(det);
  } catch (err) { a.textContent = 'erro: ' + err; }
};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE
