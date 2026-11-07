"""Feature engineering: point-to-pixel matching and physically motivated features.

The SAME function (`point_features`) builds features for stations (training/validation),
for the map grid and for any coordinate (CLI/API). This prevents training/inference skew.

Spatio-temporal matching:
  - time: everything is aggregated over the LOCAL day (America/Sao_Paulo, UTC-3)
  - AOD (0.05° grid): the cell containing the point, plus means over 3x3 (~15 km)
    and 9x9 (~45 km) windows
  - ERA5 (0.25°): bilinear interpolation
  - static layers (0.01°): 1 km cell and 5x5 km window

Cloud gaps in AOD (in this order):
  1) composite of the day's 8 hourly scans (already done in goes_aod.py)
  2) cell -> 3x3 window -> 9x9 window (spatial neighborhood)
  3) if none exists, it stays NaN and LightGBM learns its own branch for "no AOD".
     Levels: 0=cell, 1=3x3, 2=9x9, 4=missing.
     There is no temporal filling: no future observation is ever used.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter

from src.common.config import data_path, load_config

CFG = load_config()
EARTH_R = 6371.0
GAMMA_HYGRO = 0.6  # typical hygroscopic growth exponent (urban/mixed aerosol)

MET_VARS = ["t2m", "rh", "ws", "ws_pm", "u10", "v10", "sp", "tcc", "blh", "blh_pm", "blh_min", "tp", "wd"]


# ----------------------------------------------------------------------------- contexto
@functools.lru_cache(maxsize=2)
def load_context(sample: bool = False) -> dict:
    """Load the processed daily cubes (AOD, NO2, ERA5, static layers, fire)."""
    base = data_path("sample") if sample else data_path("interim")
    aod = xr.load_dataset(base / "aod_daily.nc")
    era5 = xr.load_dataset(base / "era5_daily.nc")
    static = xr.load_dataset(base / "static_0p01.nc")
    fires = pd.read_parquet(base / "fires_daily.parquet")
    fires["date"] = pd.to_datetime(fires["date"])
    no2 = xr.load_dataset(base / "no2_daily.nc")
    no2["no2_w3"] = (no2["no2"].dims, _nanmean_window(no2["no2"].values, 3))
    no2["no2_w9"] = (no2["no2"].dims, _nanmean_window(no2["no2"].values, 9))
    return {"aod": _aod_windows(aod), "era5": era5, "static": _static_windows(static), "no2": no2,
            "fires": {d: g[["lat", "lon", "frp_max"]].to_numpy() for d, g in fires.groupby("date")}}


def _nanmean_window(a: np.ndarray, size: int) -> np.ndarray:
    """NaN-aware mean over a size x size window, applied to every time slice of the cube."""
    valid = np.isfinite(a).astype("float32")
    filled = np.where(np.isfinite(a), a, 0).astype("float32")
    s = uniform_filter(filled, size=(1, size, size), mode="constant")
    n = uniform_filter(valid, size=(1, size, size), mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 1e-6, s / n, np.nan).astype("float32")


def _aod_windows(aod: xr.Dataset) -> xr.Dataset:
    a = aod["aod_mean"].values
    w3 = _nanmean_window(a, 3)
    w9 = _nanmean_window(a, 9)
    out = aod.copy()
    out["aod_w3"] = (aod["aod_mean"].dims, w3)
    out["aod_w9"] = (aod["aod_mean"].dims, w9)
    return out


def _static_windows(st: xr.Dataset) -> xr.Dataset:
    out = st.copy()
    for v in ["built", "tree", "crop", "grass"]:
        if v in st:
            a = st[v].values[None]
            out[f"{v}_5km"] = (st[v].dims, _nanmean_window(a, 5)[0])
    return out


# ----------------------------------------------------------------------------- geometria
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    """Azimuth (degrees from north) from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


# ----------------------------------------------------------------------------- blocos
def _sel_grid(ds: xr.Dataset, names, dates, lats, lons) -> dict:
    """Value of the cell containing each point (nearest neighbor), for each date."""
    t = xr.DataArray(pd.DatetimeIndex(dates), dims="p")
    la = xr.DataArray(lats, dims="p")
    lo = xr.DataArray(lons, dims="p")
    out = {}
    valid_t = np.isin(pd.DatetimeIndex(dates).values, ds.time.values)
    for n in names:
        v = np.full(len(lats), np.nan, dtype="float32")
        if valid_t.any():
            sub = ds[n].sel(time=t[valid_t], lat=la[valid_t], lon=lo[valid_t], method="nearest").values
            v[valid_t] = sub
        out[n] = v
    return out


def aod_features(ctx, dates, lats, lons, forecast: bool = False) -> pd.DataFrame:
    a = _sel_grid(ctx["aod"], ["aod_mean", "aod_max", "aod_nscan", "aod_w3", "aod_w9"],
                  dates, lats, lons)
    levels = [a["aod_mean"], a["aod_w3"], a["aod_w9"]]
    filled = np.full(len(lats), np.nan, dtype="float32")
    level = np.full(len(lats), 4, dtype="int8")
    for i, arr in enumerate(levels):
        m = np.isnan(filled) & np.isfinite(arr)
        filled[m] = arr[m]
        level[m] = i
    return pd.DataFrame({"aod": a["aod_mean"], "aod_w3": a["aod_w3"], "aod_w9": a["aod_w9"],
                         "aod_max": a["aod_max"], "aod_nscan": a["aod_nscan"],
                         "aod_filled": filled, "aod_fill_level": level})


def no2_features(ctx, dates, lats, lons) -> pd.DataFrame:
    """Tropospheric NO2 (TROPOMI) in the cell and in 3x3/9x9 windows. Cloud gaps: use the larger window."""
    a = _sel_grid(ctx["no2"], ["no2", "no2_w3", "no2_w9"], dates, lats, lons)
    filled = np.where(np.isfinite(a["no2_w3"]), a["no2_w3"], a["no2_w9"])
    return pd.DataFrame({"no2": a["no2"], "no2_w9": a["no2_w9"], "no2_filled": filled})


def _bilinear_index(axis: np.ndarray, x: np.ndarray):
    """Lower index and weight for linear interpolation on a regular increasing grid."""
    step = axis[1] - axis[0]
    f = np.clip((x - axis[0]) / step, 0, len(axis) - 1 - 1e-9)
    i0 = np.floor(f).astype(int)
    return i0, f - i0


def met_features(ctx, dates, lats, lons, prefix: str = "") -> pd.DataFrame:
    """ERA5 by bilinear interpolation (vectorized: xarray's `interp` with pointwise indexers
    grew quadratically with the number of points)."""
    e = ctx["era5"]
    times = pd.DatetimeIndex(e.time.values)
    ti = times.get_indexer(pd.DatetimeIndex(dates))
    valid_t = ti >= 0
    iy, wy = _bilinear_index(e.lat.values, np.asarray(lats))
    ix, wx = _bilinear_index(e.lon.values, np.asarray(lons))
    out = {}
    for v in MET_VARS:
        if v == "wd":  # direction cannot be interpolated linearly; it is derived from u/v below
            continue
        arr = np.full(len(lats), np.nan, dtype="float32")
        if valid_t.any():
            a = e[v].values  # (time, lat, lon)
            t, y0, x0, fy, fx = ti[valid_t], iy[valid_t], ix[valid_t], wy[valid_t], wx[valid_t]
            arr[valid_t] = ((1 - fy) * (1 - fx) * a[t, y0, x0] + (1 - fy) * fx * a[t, y0, x0 + 1] +
                            fy * (1 - fx) * a[t, y0 + 1, x0] + fy * fx * a[t, y0 + 1, x0 + 1])
        out[prefix + v] = arr
    out[prefix + "wd"] = (np.degrees(np.arctan2(-out[prefix + "u10"], -out[prefix + "v10"])) + 360) % 360
    return pd.DataFrame(out)


def static_features(ctx, lats, lons) -> pd.DataFrame:
    st = ctx["static"]
    la = xr.DataArray(lats, dims="p")
    lo = xr.DataArray(lons, dims="p")
    names = [n for n in ["built", "built_5km", "tree_5km", "crop_5km", "grass_5km", "elev"] if n in st]
    return pd.DataFrame({n: st[n].sel(lat=la, lon=lo, method="nearest").values for n in names})


def fire_features(ctx, dates, lats, lons, wd, suffix: str = "") -> pd.DataFrame:
    """Fire detections by radius, and weighted by wind.

    A fire is "upwind" when the azimuth from the point to the fire matches the direction
    the wind blows FROM: air reaches the point coming from the fire. Weight =
    max(0, cos(Δθ)) x FRP / distance, summed within 50/150/400 km. Each input row uses
    the fires of the same date."""
    n = len(lats)
    res = {f"fire_n_{r}km{suffix}": np.zeros(n, "float32") for r in (50, 150, 400)}
    res.update({f"fire_up_{r}km{suffix}": np.zeros(n, "float32") for r in (150, 400)})
    dates = pd.DatetimeIndex(dates)
    for d in np.unique(dates.values):
        idx = np.where(dates.values == d)[0]
        f = ctx["fires"].get(pd.Timestamp(d))
        if f is None or len(f) == 0:
            continue
        plat, plon = lats[idx][:, None], lons[idx][:, None]
        # process in blocks so the full grid fits in memory
        for s in range(0, len(idx), 2000):
            sl = slice(s, s + 2000)
            dist = haversine(plat[sl], plon[sl], f[None, :, 0], f[None, :, 1])
            brg = bearing(plat[sl], plon[sl], f[None, :, 0], f[None, :, 1])
            align = np.clip(np.cos(np.radians(brg - wd[idx][sl][:, None])), 0, None)
            frp = np.clip(f[None, :, 2], 1, None)
            for r in (50, 150, 400):
                res[f"fire_n_{r}km{suffix}"][idx[sl]] = (dist <= r).sum(1)
            for r in (150, 400):
                w = np.where(dist <= r, align * np.log1p(frp) / np.maximum(dist, 5) * 100, 0)
                res[f"fire_up_{r}km{suffix}"][idx[sl]] = w.sum(1)
    return pd.DataFrame(res)


def calendar_features(dates) -> pd.DataFrame:
    d = pd.DatetimeIndex(dates)
    doy = d.day_of_year.values
    return pd.DataFrame({"doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
                         "dow": d.day_of_week.values, "weekend": (d.day_of_week.values >= 5).astype(int)})


def physics_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Physically motivated combinations.

    - aod_blh: AOD is a column integral. Dividing by the mixing height gives a proxy for
      extinction (hence concentration) near the surface
    - aod_dry: hygroscopic correction. At high RH, water swells the particles and raises
      AOD, but the monitor measures dry mass (heated to ~50 °C). Hence
      AOD_dry ≈ AOD·(1-RH)^γ
    - vent: ventilation index (BLH x wind speed), the atmosphere's capacity to dilute
    """
    out = pd.DataFrame(index=df.index)
    blh_km = np.clip(df[prefix + "blh_pm"] / 1000.0, 0.1, None)
    rh = np.clip(df[prefix + "rh"] / 100.0, 0, 0.95)
    aod = df["aod_filled"]
    out["aod_blh"] = aod / blh_km
    out["aod_dry"] = aod * (1 - rh) ** GAMMA_HYGRO
    out["aod_dry_blh"] = out["aod_dry"] / blh_km
    out["vent"] = df[prefix + "blh_pm"] * df[prefix + "ws_pm"]
    out["log_vent"] = np.log1p(out["vent"])
    return out


def point_features(ctx, dates, lats, lons, forecast: bool = False) -> pd.DataFrame:
    """All "context" features (no PM2.5 spatial lag) for (date, point) pairs.

    forecast=True: the target is D+1. Uses satellite, fire and weather up to D, plus the
    known calendar of D+1. Never reads the observed weather of D+1. The forecast is
    issued at the end of D, once the daily aggregates are complete. ERA5 is a
    retrospective reanalysis; its real-time availability is not simulated."""
    dates = pd.DatetimeIndex(dates)
    lats = np.asarray(lats, dtype="float64")
    lons = np.asarray(lons, dtype="float64")
    parts = [aod_features(ctx, dates, lats, lons, forecast=forecast), no2_features(ctx, dates, lats, lons),
             met_features(ctx, dates, lats, lons)]
    met_prev = met_features(ctx, dates - pd.Timedelta(days=1), lats, lons, prefix="prev_")[["prev_tp", "prev_blh_pm"]]
    parts.append(met_prev)
    base = pd.concat(parts, axis=1)
    parts = [base, physics_features(base), static_features(ctx, lats, lons), calendar_features(dates),
             fire_features(ctx, dates, lats, lons, base["wd"].to_numpy()),
             fire_features(ctx, dates - pd.Timedelta(days=1), lats, lons, base["wd"].to_numpy(), suffix="_lag1")]
    if forecast:
        parts.append(calendar_features(dates + pd.Timedelta(days=1)).add_prefix("next_")[["next_dow", "next_weekend"]])
    out = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    out.insert(0, "date", dates)
    out.insert(1, "lat", lats)
    out.insert(2, "lon", lons)
    return out


# ----------------------------------------------------------------------------- lag espacial
def idw(target_lat, target_lon, ref_lat, ref_lon, ref_val, power: float = 2.0, k: int = 8,
        exclude_self_km: float = 0.5):
    """Inverse-distance-weighted interpolation (baseline and spatial-lag feature).
    Ignores references closer than `exclude_self_km` (the station itself)."""
    if len(ref_val) == 0:
        return np.nan, np.nan, 0
    d = haversine(target_lat, target_lon, ref_lat, ref_lon)
    m = d > exclude_self_km
    d, v = d[m], ref_val[m]
    if len(v) == 0:
        return np.nan, np.nan, 0
    o = np.argsort(d)[:k]
    w = 1.0 / np.maximum(d[o], 1.0) ** power
    return float(np.sum(w * v[o]) / np.sum(w)), float(d[o][0]), int(len(o))


def add_spatial_lag(rows: pd.DataFrame, refs: pd.DataFrame, value_col: str = "pm25", prefix: str = "idw") -> pd.DataFrame:
    """For each row (date, lat, lon), IDW of the `refs` values (date, lat, lon, value) on
    the SAME date. The caller decides which stations go into `refs`. In validation, test
    stations NEVER do (prevents leakage)."""
    vals = np.full(len(rows), np.nan)
    dmin = np.full(len(rows), np.nan)
    nref = np.zeros(len(rows), int)
    by_date = {d: g for d, g in refs.groupby("date")}
    rdates = rows["date"].to_numpy()
    rlat, rlon = rows["lat"].to_numpy(), rows["lon"].to_numpy()
    for d in np.unique(rdates):
        g = by_date.get(pd.Timestamp(d))
        if g is None:
            continue
        idx = np.where(rdates == d)[0]
        gl, go, gv = g["lat"].to_numpy(), g["lon"].to_numpy(), g[value_col].to_numpy()
        for i in idx:
            vals[i], dmin[i], nref[i] = idw(rlat[i], rlon[i], gl, go, gv)
    out = rows.copy()
    out[f"{prefix}"] = vals
    out[f"{prefix}_dmin"] = dmin
    out[f"{prefix}_n"] = nref
    return out
