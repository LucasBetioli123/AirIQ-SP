"""Boundaries and municipalities (for the map mask and the agent's geocoding).

- Polygons of São Paulo's 645 municipalities (IBGE mesh converted to GeoJSON by the
  tbrugz/geodata-br project). Dissolving them gives the state outline
- Municipal seats with lat/lon (IBGE, compiled in kelvins/municipios-brasileiros)
"""
from __future__ import annotations

import functools
import unicodedata

import pandas as pd

from src.common.config import data_path
from src.common.http import download

MUN_GEOJSON = "https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-35-mun.json"
MUN_CSV = "https://raw.githubusercontent.com/kelvins/municipios-brasileiros/main/csv/municipios.csv"


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace("-", " ").split())


def fetch():
    for url, name in [(MUN_GEOJSON, "sp_municipios.geojson"), (MUN_CSV, "municipios_br.csv")]:
        p = data_path("raw", "geo", name)
        if not p.exists():
            download(url, p)
    # small versioned copy so the agent/API work with the sample alone
    sp = municipios_sp(refresh=True)
    sp.to_csv(data_path("sample", "municipios_sp.csv"), index=False)
    sp_boundary.cache_clear()
    data_path("sample", "sp_boundary.wkt").write_text(sp_boundary(prefer_raw=True).wkt)


@functools.lru_cache(maxsize=2)
def sp_boundary(prefer_raw: bool = False):
    import geopandas as gpd
    import shapely

    wkt = data_path("sample", "sp_boundary.wkt")
    if wkt.exists() and not prefer_raw:
        return shapely.from_wkt(wkt.read_text())
    p = data_path("raw", "geo", "sp_municipios.geojson")
    if not p.exists():
        fetch()
    g = gpd.read_file(p)
    # buffer(0) fixes small topological inconsistencies in the mesh before dissolving
    return g.geometry.buffer(0).union_all().buffer(0.002).simplify(0.005)


def municipios_sp(refresh: bool = False) -> pd.DataFrame:
    cached = data_path("sample", "municipios_sp.csv")
    if cached.exists() and not refresh:
        return pd.read_csv(cached)
    df = pd.read_csv(data_path("raw", "geo", "municipios_br.csv"))
    df = df[df["codigo_uf"] == 35][["codigo_ibge", "nome", "latitude", "longitude"]]
    return df.rename(columns={"latitude": "lat", "longitude": "lon"}).reset_index(drop=True)


def geocode(nome: str) -> dict | None:
    """São Paulo municipality -> seat coordinates (accent- and case-insensitive match)."""
    df = municipios_sp()
    key = _norm(nome)
    hit = df[df["nome"].map(_norm) == key]
    if hit.empty:
        hit = df[df["nome"].map(_norm).str.startswith(key)]
    if hit.empty:
        return None
    r = hit.iloc[0]
    return {"municipio": r["nome"], "lat": float(r["lat"]), "lon": float(r["lon"]), "codigo_ibge": int(r["codigo_ibge"])}


if __name__ == "__main__":
    fetch()
    print(geocode("Sorocaba"), sp_boundary().bounds)
