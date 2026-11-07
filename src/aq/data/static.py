"""Static context features for each location, on a fine 0.01° (~1 km) grid.

- Land use: ESA WorldCover 2021 v200 (10 m, COG on AWS Open Data). We compute the
  fraction of built-up area (class 50), trees (10), cropland (40) and grassland (30).
  MapBiomas is also available as COG on GCS, but it is a single ~1 GB national GeoTIFF
  per year, and for 1 km land-use fractions both sources are equivalent. WorldCover
  comes in 3° tiles, which makes partial reads simpler.
- Terrain: Copernicus DEM GLO-30 (COG, 1° tiles).

COGs are read at reduced resolution (their internal overviews) to avoid downloading
tens of GB. Sampling pixels at ~80 m is statistically enough for 1 km fractions.

Output: data/interim/static_0p01.nc with variables built, tree, crop, grass and elev.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from tqdm import tqdm

from src.common.config import data_path, load_config

CFG = load_config()
BBOX = CFG["region"]["bbox"]
FINE = 0.01
WC_CLASSES = {"built": 50, "tree": 10, "crop": 40, "grass": 30}


def fine_grid():
    lon_min, lat_min, lon_max, lat_max = BBOX
    nx = int(round((lon_max - lon_min) / FINE))
    ny = int(round((lat_max - lat_min) / FINE))
    return lat_min + FINE * (np.arange(ny) + 0.5), lon_min + FINE * (np.arange(nx) + 0.5)


def _wc_tile_name(lat0: int, lon0: int) -> str:
    ns = "S" if lat0 < 0 else "N"
    ew = "W" if lon0 < 0 else "E"
    return f"ESA_WorldCover_10m_2021_v200_{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}_Map.tif"


def _dem_tile_name(lat0: int, lon0: int) -> str:
    ns = "S" if lat0 < 0 else "N"
    ew = "W" if lon0 < 0 else "E"
    # tiles are named by their lower-left corner; S24 covers -24..-23
    n = f"Copernicus_DSM_COG_10_{ns}{abs(lat0):02d}_00_{ew}{abs(lon0):03d}_00_DEM"
    return f"{n}/{n}.tif"


def _read_block(url: str, bounds, sub: int = 8):
    """Read the part of the COG inside the bbox, downsampled by `sub` (uses the overviews)."""
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        with rasterio.open("/vsicurl/" + url) as src:
            b = (max(bounds[0], src.bounds.left), max(bounds[1], src.bounds.bottom),
                 min(bounds[2], src.bounds.right), min(bounds[3], src.bounds.top))
            if b[0] >= b[2] or b[1] >= b[3]:
                return None
            w = from_bounds(*b, transform=src.transform)
            h, wd = max(1, int(w.height / sub)), max(1, int(w.width / sub))
            arr = src.read(1, window=w, out_shape=(h, wd), resampling=Resampling.nearest)
            ys = np.linspace(b[3], b[1], h, endpoint=False) - (b[3] - b[1]) / h / 2
            xs = np.linspace(b[0], b[2], wd, endpoint=False) + (b[2] - b[0]) / wd / 2
            return arr, ys, xs


def _accumulate(results, value_fn, lats, lons):
    ny, nx = len(lats), len(lons)
    out = {}
    for arr, ys, xs in results:
        X, Y = np.meshgrid(xs, ys)
        ix = ((X - BBOX[0]) / FINE).astype(int)
        iy = ((Y - BBOX[1]) / FINE).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        idx = (iy * nx + ix)[ok]
        for name, v in value_fn(arr[ok]).items():
            s, n = out.setdefault(name, (np.zeros(ny * nx), np.zeros(ny * nx)))
            valid = np.isfinite(v)
            s += np.bincount(idx[valid], weights=v[valid], minlength=ny * nx)
            n += np.bincount(idx[valid], minlength=ny * nx)
    return {k: np.where(n > 0, s / np.maximum(n, 1), np.nan).reshape(ny, nx) for k, (s, n) in out.items()}


def worldcover(lats, lons):
    base = CFG["static"]["worldcover_url"]
    tiles = [(la, lo) for la in range(3 * math.floor(BBOX[1] / 3), math.ceil(BBOX[3]), 3)
             for lo in range(3 * math.floor(BBOX[0] / 3), math.ceil(BBOX[2]), 3)]
    def rd(t):
        try:
            return _read_block(base + _wc_tile_name(*t), BBOX)
        except Exception:
            return None  # ocean-only tiles do not exist

    with ThreadPoolExecutor(6) as ex:
        res = [r for r in tqdm(ex.map(rd, tiles),
                               total=len(tiles), desc="WorldCover") if r is not None]

    def fn(v):
        v = v.astype(float)
        valid = v > 0
        return {k: np.where(valid, (v == c).astype(float), np.nan) for k, c in WC_CLASSES.items()}

    return _accumulate(res, fn, lats, lons)


def dem(lats, lons):
    base = CFG["static"]["dem_url"]
    # 1-tile margin: the effective bounds come from the file itself
    tiles = [(la, lo) for la in range(math.floor(BBOX[1]) - 1, math.ceil(BBOX[3]) + 1)
             for lo in range(math.floor(BBOX[0]) - 1, math.ceil(BBOX[2]) + 1)]

    def rd(t):
        try:
            return _read_block(base + _dem_tile_name(*t), BBOX, sub=4)
        except Exception:
            return None  # ocean-only tiles do not exist

    with ThreadPoolExecutor(8) as ex:
        res = [r for r in tqdm(ex.map(rd, tiles), total=len(tiles), desc="Copernicus DEM") if r is not None]
    return _accumulate(res, lambda v: {"elev": np.where(v > -100, v.astype(float), np.nan)}, lats, lons)


def main():
    lats, lons = fine_grid()
    layers = {**worldcover(lats, lons), **dem(lats, lons)}
    ds = xr.Dataset({k: (("lat", "lon"), v.astype("float32")) for k, v in layers.items()},
                    coords={"lat": lats, "lon": lons})
    ds.to_netcdf(data_path("interim", "static_0p01.nc"))
    print({k: float(np.nanmean(v)) for k, v in layers.items()})


if __name__ == "__main__":
    main()
