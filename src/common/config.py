"""Carregamento da configuração e utilitários de caminho."""
from __future__ import annotations

import functools
import os
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


load_env()


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
