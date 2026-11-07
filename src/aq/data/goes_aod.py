"""AOD (aerosol optical depth, 550 nm) from GOES-16 ABI L2 AODF.

Why not MODIS MAIAC (MCD19A2): MAIAC requires an Earthdata/Earth Engine account.
GOES-16 is geostationary over South America and publishes an AOD retrieval every
10 min at 2 km on AWS Open Data (anonymous). Taking one scan per hour from 09 to 16 h
local time greatly reduces cloud gaps: ONE clear-sky hour is enough for the day to
have AOD, while MODIS has 1-2 overpasses per day.

For each (local) day and 0.05° grid cell we keep:
  aod_mean   mean of valid retrievals (DQF <= aod_dqf_max) over all scans
  aod_max    maximum of the cell's hourly means
  aod_nscan  number of scans with at least 1 valid pixel in the cell
Output: data/raw/goes_aod/aod_<yyyy-mm-dd>.npz and the cube data/interim/aod_daily.nc
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

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
        t = date + pd.Timedelta(hours=h)  # date is local midnight ~ 03 UTC; UTC hours on the same day
        try:
            ks = list_hour(PRODUCT, t.year, t.day_of_year, h)
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
    window_for_bbox(PRODUCT, BBOX)  # warm the geometry cache before starting threads
    with ProcessPoolExecutor(workers) as ex:  # processos: o h5py serializa threads (lock global)
        list(tqdm(ex.map(process_day, dates, chunksize=1), total=len(dates), desc="GOES AOD"))
    ds = assemble(dates)
    cov = np.isfinite(ds["aod_mean"]).mean().item()
    print(f"Daily AOD: {ds.sizes['time']} days, mean pixel coverage {cov:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    main(a.start, a.end, a.workers)
