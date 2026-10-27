"""Utilitários comuns aos produtos GOES-16 ABI L2 (AOD e focos de queimada).

O ABI usa uma grade fixa em ângulos de varredura (x, y) em radianos. Convertemos
para lat/lon com a fórmula do GOES-R Product User Guide (vol. 5, seção 4.2.8) e
lemos do S3 só a janela de linhas/colunas que cobre o recorte, via HTTP range
requests (o NetCDF4/HDF5 é chunked, então não baixamos os ~40 MB do disco inteiro).
"""
from __future__ import annotations

import functools
import re

import fsspec
import numpy as np
import xarray as xr

from src.common.config import load_config
from src.common.http import s3_list

CFG = load_config()
BUCKET = CFG["goes"]["bucket_url"]


def scan_to_latlon(x: np.ndarray, y: np.ndarray, proj: dict) -> tuple[np.ndarray, np.ndarray]:
    req = proj["semi_major_axis"]
    rpol = proj["semi_minor_axis"]
    H = proj["perspective_point_height"] + req
    lon0 = np.deg2rad(proj["longitude_of_projection_origin"])
    xx, yy = np.meshgrid(x, y)
    a = np.sin(xx) ** 2 + np.cos(xx) ** 2 * (np.cos(yy) ** 2 + (req**2 / rpol**2) * np.sin(yy) ** 2)
    b = -2 * H * np.cos(xx) * np.cos(yy)
    c = H**2 - req**2
    with np.errstate(invalid="ignore"):
        rs = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        sx = rs * np.cos(xx) * np.cos(yy)
        sy = -rs * np.sin(xx)
        sz = rs * np.cos(xx) * np.sin(yy)
        lat = np.rad2deg(np.arctan((req**2 / rpol**2) * sz / np.sqrt((H - sx) ** 2 + sy**2)))
        lon = np.rad2deg(lon0 - np.arctan(sy / (H - sx)))
    return lat, lon


def open_remote(key: str) -> xr.Dataset:
    f = fsspec.open(BUCKET + key, mode="rb", block_size=2**20, cache_type="blockcache").open()
    return xr.open_dataset(f, engine="h5netcdf", mask_and_scale=True)


@functools.lru_cache(maxsize=4)
def window_for_bbox(product: str, bbox: tuple[float, float, float, float]):
    """Calcula (uma vez por produto, com cache em disco) a janela de linhas/colunas que
    cobre o bbox e as coordenadas lat/lon dos pixels dessa janela."""
    from src.common.config import data_path

    cache = data_path("raw", "goes_geom", f"{product}_{'_'.join(f'{b:g}' for b in bbox)}.npz")
    if cache.exists():
        z = np.load(cache)
        return tuple(int(v) for v in z["win"]), z["lat"], z["lon"]
    key = first_key(product, 2022, 200, 15)
    ds = open_remote(key)
    proj = {k: float(v) for k, v in ds["goes_imager_projection"].attrs.items()
            if k in ("semi_major_axis", "semi_minor_axis", "perspective_point_height",
                     "longitude_of_projection_origin")}
    x = ds["x"].values
    y = ds["y"].values
    lat, lon = scan_to_latlon(x, y, proj)
    lon_min, lat_min, lon_max, lat_max = bbox
    inside = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
    rows = np.where(inside.any(axis=1))[0]
    cols = np.where(inside.any(axis=0))[0]
    r0, r1, c0, c1 = int(rows.min()), int(rows.max() + 1), int(cols.min()), int(cols.max() + 1)
    np.savez(cache, win=np.array([r0, r1, c0, c1]), lat=lat[r0:r1, c0:c1], lon=lon[r0:r1, c0:c1])
    return (r0, r1, c0, c1), lat[r0:r1, c0:c1], lon[r0:r1, c0:c1]


def list_hour(product: str, year: int, doy: int, hour: int) -> list[str]:
    keys, _ = s3_list(BUCKET, f"{product}/{year}/{doy:03d}/{hour:02d}/")
    return sorted(k for k, _ in keys if k.endswith(".nc"))


def first_key(product: str, year: int, doy: int, hour: int) -> str:
    ks = list_hour(product, year, doy, hour)
    if not ks:
        raise FileNotFoundError(f"{product} {year}/{doy}/{hour}")
    return ks[0]


def scan_start(key: str):
    """Instante de início da varredura a partir do nome do arquivo (sAAAADDDHHMMSSs)."""
    import pandas as pd

    m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", key)
    y, d, h, mi, s = map(int, m.groups())
    return pd.Timestamp(year=y, month=1, day=1, hour=h, minute=mi, second=s, tz="UTC") + pd.Timedelta(days=d - 1)


def bin_to_grid(lat: np.ndarray, lon: np.ndarray, val: np.ndarray, bbox, res: float):
    """Média de pixels válidos em cada célula de uma grade regular lat/lon (res graus).
    Retorna soma, contagem (para compor várias varreduras depois)."""
    lon_min, lat_min, lon_max, lat_max = bbox
    nx = int(round((lon_max - lon_min) / res))
    ny = int(round((lat_max - lat_min) / res))
    ok = np.isfinite(val) & np.isfinite(lat) & np.isfinite(lon)
    ix = ((lon[ok] - lon_min) / res).astype(int)
    iy = ((lat[ok] - lat_min) / res).astype(int)
    inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    idx = iy[inb] * nx + ix[inb]
    s = np.bincount(idx, weights=val[ok][inb], minlength=nx * ny).reshape(ny, nx)
    n = np.bincount(idx, minlength=nx * ny).reshape(ny, nx)
    return s, n


def grid_coords(bbox, res: float):
    lon_min, lat_min, lon_max, lat_max = bbox
    nx = int(round((lon_max - lon_min) / res))
    ny = int(round((lat_max - lat_min) / res))
    lons = lon_min + res * (np.arange(nx) + 0.5)
    lats = lat_min + res * (np.arange(ny) + 0.5)
    return lats, lons
