"""Tropospheric NO2 from Sentinel-5P TROPOMI (~13:30 local overpass), MEEO mirror on AWS.

The `meeo-s5p` bucket publishes the L2 OFFL product already converted to Cloud-Optimized
GeoTIFF (global 0.035° grid, EPSG:4326, one image per orbit), so only the São Paulo
window is read. Pixels with qa_value >= 75 are kept (the product PUM recommendation:
excludes cloud and snow). NO2 traces combustion emissions (traffic/industry) and
secondary-aerosol precursors (nitrate). It helps tell urban/industrial pollution
from smoke.

Output: data/interim/no2_daily.nc (µmol/m², 0.05° grid)
"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.windows import from_bounds
from tqdm import tqdm

from src.aq.data.goes_common import bin_to_grid, grid_coords
from src.common.config import data_path, load_config
from src.common.http import s3_list

CFG = load_config()
BUCKET = "https://meeo-s5p.s3.amazonaws.com/"
BBOX = tuple(CFG["region"]["bbox"])
RES = CFG["region"]["grid_res_deg"]


def _orbits(date: pd.Timestamp) -> list[str]:
    keys, _ = s3_list(BUCKET, f"COGT/OFFL/L2__NO2___/{date:%Y/%m/%d}/")
    out = []
    for k, _s in keys:
        if not k.endswith("PRODUCT_nitrogendioxide_tropospheric_column_4326.tif"):
            continue
        m = re.search(r"____(\d{8}T\d{6})_(\d{8}T\d{6})_", k)
        t0, t1 = pd.Timestamp(m.group(1)), pd.Timestamp(m.group(2))
        # daytime orbits crossing SP: overpass ~16-17 UTC
        if t1 >= date + pd.Timedelta(hours=15, minutes=30) and t0 <= date + pd.Timedelta(hours=18, minutes=30):
            out.append(k)
    return out


def _read(key: str):
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open("/vsicurl/" + BUCKET + key) as s:
            w = from_bounds(*BBOX, transform=s.transform)
            no2 = s.read(1, window=w, boundless=True, fill_value=s.nodata).astype("float32")
            tr = s.window_transform(w)
            nod = s.nodata
        with rasterio.open("/vsicurl/" + BUCKET + key.replace("nitrogendioxide_tropospheric_column", "qa_value")) as q:
            qa = q.read(1, window=from_bounds(*BBOX, transform=q.transform), boundless=True, fill_value=255)
    ok = (no2 != nod) & (qa >= 75) & (qa != 255)
    no2 = np.where(ok, no2 * 1e6, np.nan)  # mol/m² -> µmol/m²
    rows, cols = np.indices(no2.shape)
    lon = tr.c + (cols + 0.5) * tr.a
    lat = tr.f + (rows + 0.5) * tr.e
    return bin_to_grid(lat, lon, no2, BBOX, RES)


def process_day(date: pd.Timestamp) -> str:
    out = data_path("raw", "s5p_no2", f"no2_{date.date()}.npy")
    if out.exists():
        return "cached"
    lats, lons = grid_coords(BBOX, RES)
    S = np.zeros((len(lats), len(lons)))
    N = np.zeros_like(S)
    for k in _orbits(date):
        try:
            s, n = _read(k)
            S += s
            N += n
        except Exception:
            continue
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(N > 0, S / np.maximum(N, 1), np.nan).astype("float32")
    np.save(out, v)
    return "ok"


def main(start=None, end=None, workers: int = 8):
    dates = pd.date_range(start or CFG["period"]["start"], end or CFG["period"]["end"], freq="D")
    with ThreadPoolExecutor(workers) as ex:
        list(tqdm(ex.map(process_day, dates), total=len(dates), desc="S5P NO2"))
    lats, lons = grid_coords(BBOX, RES)
    arr = np.stack([np.load(data_path("raw", "s5p_no2", f"no2_{d.date()}.npy")) for d in dates])
    ds = xr.Dataset({"no2": (("time", "lat", "lon"), arr)}, coords={"time": dates, "lat": lats, "lon": lons})
    ds.to_netcdf(data_path("interim", "no2_daily.nc"))
    print(f"NO2: mean pixel coverage {np.isfinite(arr).mean():.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    a = ap.parse_args()
    main(a.start, a.end)
