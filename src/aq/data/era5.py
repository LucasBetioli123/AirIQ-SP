"""ERA5 weather (ECMWF reanalysis, 0.25°, hourly) via NCAR's public mirror on AWS.

Why not Copernicus CDS: the CDS requires an API key. NCAR (RDA dataset ds633.0)
publishes the SAME ERA5 as NetCDF4 in the anonymous bucket `nsf-ncar-era5`, one file
per variable and month. The files are chunked (27 h x 139 lat x 277 lon), so only the
São Paulo window is read through HTTP range requests.

Variables and physical rationale:
  blh   boundary-layer height: the mixing volume. For the same emissions, a shallow
        boundary layer concentrates the pollutant near the surface
  t2m, d2m  temperature and dew point, which give relative humidity (hygroscopic growth
        of aerosol inflates AOD without increasing the dry mass the monitor measures)
  u10, v10  wind: dilution/transport and direction for "upwind fires"
  sp, tcc   pressure (synoptic systems/stability) and cloud cover
  lsp+cp    total precipitation: wet removal (washout)

Aggregated over the LOCAL day (UTC-3): daily means, afternoon BLH (12-18 h local, when
the boundary layer is developed) and minimum BLH (nighttime stability).
Output: data/interim/era5_daily.nc
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from src.common.config import data_path, load_config
from src.common.http import s3_list

CFG = load_config()
E5 = CFG["era5"]
BUCKET = E5["bucket_url"]
# window with a 1-cell margin for bilinear interpolation at the edges
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = CFG["region"]["bbox"]
LAT_SL = slice(LAT_MAX + 0.5, LAT_MIN - 0.5)            # latitude is decreasing in the file
LON_SL = slice(LON_MIN % 360 - 0.5, LON_MAX % 360 + 0.5)  # longitude 0..360
ACCUM = {"lsp": "128_142_lsp", "cp": "128_143_cp"}


def _open(key: str) -> xr.Dataset:
    f = fsspec.open(BUCKET + key, mode="rb", block_size=2**20, cache_type="blockcache").open()
    return xr.open_dataset(f, engine="h5netcdf")


def _main_var(ds: xr.Dataset) -> str:
    return next(v for v in ds.data_vars if ds[v].ndim >= 3)


def _atomic_write(da: xr.DataArray, out) -> None:
    """Write to .tmp and rename: an interrupted process never leaves a truncated file."""
    tmp = out.with_suffix(".tmp")
    da.to_dataset().to_netcdf(tmp)
    tmp.rename(out)


def fetch_an(var: str, code: str, month: pd.Timestamp) -> str:
    out = data_path("raw", "era5", f"{var}_{month:%Y%m}.nc")
    if out.exists():
        return str(out)
    ym = f"{month:%Y%m}"
    keys, _ = s3_list(BUCKET, f"e5.oper.an.sfc/{ym}/e5.oper.an.sfc.{code}.")
    ds = _open(keys[0][0])
    da = ds[_main_var(ds)].sel(latitude=LAT_SL, longitude=LON_SL).load().rename(var)
    _atomic_write(da, out)
    return str(out)


def fetch_accum(var: str, code: str, month: pd.Timestamp) -> str:
    """Accumulated fields come from forecasts (06/18 UTC init, 1-12 h steps) in half-month
    files. They are converted to an hourly series by valid time."""
    out = data_path("raw", "era5", f"{var}_{month:%Y%m}.nc")
    if out.exists():
        return str(out)
    keys, _ = s3_list(BUCKET, f"e5.oper.fc.sfc.accumu/{month:%Y%m}/e5.oper.fc.sfc.accumu.{code}.")
    parts = []
    for k, _ in keys:
        ds = _open(k)
        da = ds[_main_var(ds)].sel(latitude=LAT_SL, longitude=LON_SL).load()
        init = da["forecast_initial_time"].values
        step = da["forecast_hour"].values
        valid = (init[:, None] + step[None, :].astype("timedelta64[h]")).ravel()
        vals = da.values.reshape(len(init) * len(step), da.sizes["latitude"], da.sizes["longitude"])
        parts.append(xr.DataArray(vals, dims=("time", "latitude", "longitude"),
                                  coords={"time": valid, "latitude": da.latitude, "longitude": da.longitude},
                                  name=var))
    da = xr.concat(parts, "time").sortby("time")
    da = da.sel(time=~da.get_index("time").duplicated())
    _atomic_write(da, out)
    return str(out)


def rh_from_t_td(t_k, td_k):
    """Relative humidity (%) from T and Td (K) via the Magnus formula."""
    t, td = t_k - 273.15, td_k - 273.15
    es = np.exp(17.625 * t / (243.04 + t))
    e = np.exp(17.625 * td / (243.04 + td))
    return np.clip(100 * e / es, 0, 100)


def to_daily(h: xr.Dataset) -> xr.Dataset:
    tz_offset = pd.Timedelta(hours=3)  # São Paulo has had no daylight saving since 2019 -> fixed UTC-3
    h = h.assign_coords(time=h.time - tz_offset)
    h["rh"] = rh_from_t_td(h.t2m, h.d2m)
    h["ws"] = np.hypot(h.u10, h.v10)
    h["tp"] = (h.lsp + h.cp) * 1000.0  # m -> mm
    day = h.time.dt.floor("D")
    hour = h.time.dt.hour
    d = xr.Dataset()
    for v in ["t2m", "rh", "ws", "u10", "v10", "sp", "tcc", "blh"]:
        d[v] = h[v].groupby(day).mean()
    d["tp"] = h.tp.groupby(day).sum()
    d["blh_pm"] = h.blh.where((hour >= 12) & (hour <= 18)).groupby(day).mean()
    d["blh_min"] = h.blh.groupby(day).min()
    d["ws_pm"] = h.ws.where((hour >= 12) & (hour <= 18)).groupby(day).mean()
    d["t2m"] = d.t2m - 273.15
    d["sp"] = d.sp / 100.0
    # direction the wind blows FROM (meteorological convention), from the vector mean
    d["wd"] = (np.degrees(np.arctan2(-d.u10, -d.v10)) + 360) % 360
    d = d.rename({"floor": "time"}) if "floor" in d.dims else d
    d = d.rename({"latitude": "lat", "longitude": "lon"})
    d = d.assign_coords(lon=(((d.lon + 180) % 360) - 180)).sortby("lat")
    return d


def _run(job):
    return job[0](*job[1:])


def main(start=None, end=None, workers: int = 6):
    start = pd.Timestamp(start or CFG["period"]["start"])
    end = pd.Timestamp(end or CFG["period"]["end"])
    # the local day ends at 03 UTC the next day -> include the month of end + 1 day
    months = pd.date_range(start.to_period("M").to_timestamp(), end + pd.Timedelta(days=1), freq="MS")
    jobs = [(fetch_an, v, c, m) for m in months for v, c in E5["an_vars"].items()]
    jobs += [(fetch_accum, v, c, m) for m in months for v, c in ACCUM.items()]
    # processes, not threads: HDF5 is not thread-safe (segfaults on concurrent reads)
    with ProcessPoolExecutor(workers) as ex:
        list(tqdm(ex.map(_run, jobs), total=len(jobs), desc="ERA5 (NCAR)"))
    names = list(E5["an_vars"]) + list(ACCUM)
    def series(v):
        da = xr.concat([xr.load_dataset(data_path("raw", "era5", f"{v}_{m:%Y%m}.nc"))[v] for m in months], "time")
        return da.sel(time=~da.get_index("time").duplicated())

    hourly = xr.merge([series(v) for v in names], join="inner")
    daily = to_daily(hourly)
    daily = daily.sel(time=slice(start, end))
    daily.to_netcdf(data_path("interim", "era5_daily.nc"))
    print(daily)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    a = ap.parse_args()
    main(a.start, a.end)
