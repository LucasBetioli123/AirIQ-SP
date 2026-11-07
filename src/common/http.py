"""Sessão HTTP com retentativas e utilitários para buckets S3 públicos (acesso anônimo)."""
from __future__ import annotations

import re
import urllib.parse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        retry = Retry(total=5, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504))
        s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=64))
        s.headers["User-Agent"] = "airiq-sp/1.0 (pesquisa; dados publicos)"
        _session = s
    return _session


def s3_list(bucket_url: str, prefix: str, delimiter: str | None = None) -> tuple[list[tuple[str, int]], list[str]]:
    """Lista objetos (key, size) e prefixos comuns de um bucket S3 público via ListObjectsV2."""
    keys: list[tuple[str, int]] = []
    prefixes: list[str] = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        r = session().get(bucket_url + "?" + urllib.parse.urlencode(params), timeout=60)
        r.raise_for_status()
        x = r.text
        keys += [(k, int(s)) for k, s in re.findall(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", x, flags=re.S)]
        prefixes += re.findall(r"<Prefix>([^<]+)</Prefix>", x)[1:] if delimiter else []
        m = re.search(r"<NextContinuationToken>([^<]+)", x)
        if not m:
            return keys, prefixes
        token = m.group(1)


def download(url: str, dest, timeout: int = 300) -> None:
    """Baixa para arquivo de forma atômica (escreve .part e renomeia)."""
    from pathlib import Path

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with session().get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)
