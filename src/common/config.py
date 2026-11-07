"""Configuration loading and path helpers."""
from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_env(path: Path | None = None) -> None:
    """Read KEY=value pairs from the local .env; variables already exported take precedence.

    Never executes code, shell substitution or variable expansion.
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
    """UTF-8 output even without make: on Windows a redirected stdout uses cp1252 and breaks on '≤'."""
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
    """Absolute path inside one of the config `paths` folders (created if missing)."""
    p = ROOT / load_config()["paths"][key]
    p = p.joinpath(*parts)
    (p if not p.suffix else p.parent).mkdir(parents=True, exist_ok=True)
    return p
