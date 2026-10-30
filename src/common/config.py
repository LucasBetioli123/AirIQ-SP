"""Carregamento da configuração e utilitários de caminho."""
from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_env(path: Path | None = None) -> None:
    """Lê KEY=value do .env local; variáveis já exportadas têm precedência.

    Não executa código, substituição de shell ou expansão de variáveis.
    """
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key.isidentifier():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        os.environ.setdefault(key, value)


def utf8_stdio() -> None:
    """Saída UTF-8 também sem make: no Windows, stdout redirecionado usa cp1252 e quebra em '≤'."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and (stream.encoding or "").lower().replace("-", "") != "utf8":
            stream.reconfigure(encoding="utf-8")


load_env()
utf8_stdio()


@functools.lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    path = path or os.environ.get("AQ_CONFIG", ROOT / "configs" / "config.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_path(key: str, *parts: str) -> Path:
    """Caminho absoluto dentro de uma das pastas de `paths` do config (cria se não existir)."""
    p = ROOT / load_config()["paths"][key]
    p = p.joinpath(*parts)
    (p if not p.suffix else p.parent).mkdir(parents=True, exist_ok=True)
    return p
