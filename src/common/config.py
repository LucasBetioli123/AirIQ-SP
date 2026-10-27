"""Carregamento da configuração e utilitários de caminho."""
from __future__ import annotations

import functools
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


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
