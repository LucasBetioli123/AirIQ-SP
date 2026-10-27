"""AOD (profundidade óptica de aerossóis, 550 nm) do GOES-16 ABI L2 AODF.

Troca em relação ao sugerido (MODIS MAIAC MCD19A2): o MAIAC exige conta no
Earthdata/Earth Engine, e esses hosts não estão acessíveis neste ambiente. O GOES-16
é geoestacionário sobre a América do Sul e publica no AWS Open Data (anônimo) uma
retrieval de AOD a cada 10 min em 2 km. Usar uma varredura por hora das 09h às 16h
locais reduz bastante as falhas por nuvem: basta UMA hora de céu limpo para o dia
ter AOD, enquanto o MODIS tem 1-2 passagens por dia.

Para cada dia (local) e célula da grade de 0.05°, guardamos:
  aod_mean   média das retrievals válidas (DQF <= aod_dqf_max) de todas as varreduras
  aod_max    máximo entre as médias horárias da célula
  aod_nscan  número de varreduras com pelo menos 1 pixel válido na célula
Saída: data/raw/goes_aod/aod_<aaaa-mm-dd>.npz e o cubo data/interim/aod_daily.nc
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from src.aq.data.goes_common import bin_to_grid, grid_coords, list_hour, open_remote, window_for_bbox
from src.common.config import data_path, load_config

CFG = load_config()
PRODUCT = CFG["goes"]["aod_product"]
BBOX = tuple(CFG["region"]["bbox"])
RES = CFG["region"]["grid_res_deg"]


def read_scan(key: str):
    (r0, r1, c0, c1), lat, lon = window_for_bbox(PRODUCT, BBOX)
    ds = open_remote(key)
    aod = ds["AOD"].isel(y=slice(r0, r1), x=slice(c0, c1)).values.astype("float32")
    dqf = ds["DQF"].isel(y=slice(r0, r1), x=slice(c0, c1)).values
    aod[~(dqf <= CFG["goes"]["aod_dqf_max"])] = np.nan
    aod[(aod < -0.05) | (aod > 5)] = np.nan
    return bin_to_grid(lat, lon, aod, BBOX, RES)


def process_day(date: pd.Timestamp) -> str:
    out = data_path("raw", "goes_aod", f"aod_{date.date()}.npz")
    if out.exists():
        return "cached"
    keys = []
    for h in CFG["goes"]["aod_hours_utc"]:
        t = date + pd.Timedelta(hours=h)  # date é meia-noite local ~ 03 UTC; horas UTC no mesmo dia
        try:
            ks = list_hour(PRODUCT, t.year, t.dayofyear, h)
        except Exception:
            ks = []
        if ks:
            keys.append(ks[0])
    sums, cnts, hour_means = [], [], []
    for k in keys:
        try:
            s, n = read_scan(k)
        except Exception:
            continue
        sums.append(s)
        cnts.append(n)
        with np.errstate(invalid="ignore", divide="ignore"):
            hour_means.append(np.where(n > 0, s / np.maximum(n, 1), np.nan))
    lats, lons = grid_coords(BBOX, RES)
    shape = (len(lats), len(lons))
    if sums:
        S, N = np.sum(sums, axis=0), np.sum(cnts, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(N > 0, S / np.maximum(N, 1), np.nan)
        hm = np.stack(hour_means)
        with np.errstate(all="ignore"):
            mx = np.where(np.isfinite(hm).any(0), np.nanmax(np.where(np.isfinite(hm), hm, -np.inf), 0), np.nan)
        nscan = np.isfinite(hm).sum(0)
    else:
        mean = np.full(shape, np.nan)
        mx = np.full(shape, np.nan)
        nscan = np.zeros(shape, int)
    np.savez_compressed(out, aod_mean=mean.astype("float32"), aod_max=mx.astype("float32"),
                        aod_nscan=nscan.astype("int8"), n_files=len(keys))
    return f"{len(keys)} varreduras"


def assemble(dates) -> xr.Dataset:
    lats, lons = grid_coords(BBOX, RES)
    arrs = {"aod_mean": [], "aod_max": [], "aod_nscan": []}
    ok_dates = []
    for d in dates:
        f = data_path("raw", "goes_aod", f"aod_{d.date()}.npz")
        if not f.exists():
            continue
        z = np.load(f)
        for k in arrs:
            arrs[k].append(z[k])
        ok_dates.append(d)
    ds = xr.Dataset({k: (("time", "lat", "lon"), np.stack(v)) for k, v in arrs.items()},
                    coords={"time": pd.DatetimeIndex(ok_dates), "lat": lats, "lon": lons})
    ds.to_netcdf(data_path("interim", "aod_daily.nc"))
    return ds


def main(start=None, end=None, workers: int = 12):
    dates = pd.date_range(start or CFG["period"]["start"], end or CFG["period"]["end"], freq="D")
    window_for_bbox(PRODUCT, BBOX)  # aquece o cache da geometria antes das threads
    with ThreadPoolExecutor(workers) as ex:
        list(tqdm(ex.map(process_day, dates), total=len(dates), desc="GOES AOD"))
    ds = assemble(dates)
    cov = np.isfinite(ds["aod_mean"]).mean().item()
    print(f"AOD diário: {ds.sizes['time']} dias, cobertura média de pixels {cov:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    main(a.start, a.end, a.workers)
