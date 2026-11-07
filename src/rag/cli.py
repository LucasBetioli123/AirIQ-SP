"""Ask the agent from the command line (questions in Portuguese: the knowledge base is
Brazilian law).

  python -m src.rag.cli "Em 2021-08-24 a qualidade do ar em Sorocaba atendia ao padrão final (PF) de MP2,5 de 24 horas da CONAMA 491/2018?"
"""
from __future__ import annotations

import json
import sys

from src.rag.agent import Agent


def main():
    q = " ".join(sys.argv[1:]).strip()
    if not q:
        sys.exit("usage: python -m src.rag.cli \"your question (in Portuguese)\"")
    r = Agent().ask(q)
    print(r["resposta"])
    print("\n--- run ---")
    print(json.dumps({k: r[k] for k in ("id", "llm", "rag_mode", "tempo_total_s", "tokens_entrada", "tokens_saida",
                                         "custo_estimado_usd", "trechos", "validacao")}, ensure_ascii=False, indent=1))
    tools = [t for t in r["ferramentas"] if t["ferramenta"] == "estimar_pm25"]
    if tools:
        print(json.dumps(tools, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
