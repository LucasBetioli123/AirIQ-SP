"""Engenharia de variáveis: casamento ponto ↔ pixel e variáveis com justificativa física.

A MESMA função (`point_features`) gera as variáveis para estações (treino/validação),
para a grade do mapa e para uma coordenada qualquer (CLI/API). Isso evita divergência
entre treino e inferência.

Casamento espaço-temporal:
  - tempo: tudo é agregado no dia LOCAL (America/Sao_Paulo, UTC-3)
  - AOD (grade 0.05°): célula que contém o ponto, mais médias em janelas 3x3 (~15 km)
    e 9x9 (~45 km)
  - ERA5 (0.25°): interpolação bilinear
  - estáticos (0.01°): célula de 1 km e janela de 5x5 km

Falhas de AOD por nuvem (nesta ordem):
  1) composição das 8 varreduras horárias do dia (já feita em goes_aod.py)
  2) célula → janela 3x3 → janela 9x9 (vizinhança espacial)
  3) média de D-1 e D+1 na janela 3x3 (persistência temporal do aerossol). Para a
     previsão D+1 usamos só D-1 (sem olhar o futuro)
  4) se nada disso existir, fica NaN, e o LightGBM aprende um ramo próprio para
     "sem AOD". A flag `aod_fill_level` (0-4) diz qual nível foi usado
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
GAMMA_HYGRO = 0.6  # expoente típico de crescimento higroscópico (aerossol urbano/misto)

MET_VARS = ["t2m", "rh", "ws", "ws_pm", "u10", "v10", "sp", "tcc", "blh", "blh_pm", "blh_min", "tp", "wd"]


# ----------------------------------------------------------------------------- contexto
@functools.lru_cache(maxsize=2)
def load_context(sample: bool = False) -> dict:
    """Carrega os cubos diários (AOD, NO2, ERA5, estáticos, fogo) já processados."""
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
    """Média ignorando NaN numa janela size x size, aplicada a cada fatia (tempo) do cubo."""
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
    # temporal: média de D-1 e D+1 (janela 3x3); "prev" usa só D-1
    prev = np.concatenate([np.full_like(w3[:1], np.nan), w3[:-1]])
    nxt = np.concatenate([w3[1:], np.full_like(w3[:1], np.nan)])
    with np.errstate(invalid="ignore"):
        tmean = np.nanmean(np.stack([prev, nxt]), axis=0)
    out = aod.copy()
    out["aod_w3"] = (aod["aod_mean"].dims, w3)
    out["aod_w9"] = (aod["aod_mean"].dims, w9)
    out["aod_t"] = (aod["aod_mean"].dims, tmean.astype("float32"))
    out["aod_tprev"] = (aod["aod_mean"].dims, prev)
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
    """Azimute (graus, a partir do norte) do ponto 1 para o ponto 2."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


# ----------------------------------------------------------------------------- blocos
def _sel_grid(ds: xr.Dataset, names, dates, lats, lons) -> dict:
    """Valor da célula que contém cada ponto (vizinho mais próximo), para cada data."""
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
    a = _sel_grid(ctx["aod"], ["aod_mean", "aod_max", "aod_nscan", "aod_w3", "aod_w9", "aod_t", "aod_tprev"],
                  dates, lats, lons)
    temporal = a["aod_tprev"] if forecast else a["aod_t"]
    levels = [a["aod_mean"], a["aod_w3"], a["aod_w9"], temporal]
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
    """NO2 troposférico (TROPOMI) na célula e em janelas 3x3/9x9. Lacunas de nuvem: usa a janela maior."""
    a = _sel_grid(ctx["no2"], ["no2", "no2_w3", "no2_w9"], dates, lats, lons)
    filled = np.where(np.isfinite(a["no2_w3"]), a["no2_w3"], a["no2_w9"])
    return pd.DataFrame({"no2": a["no2"], "no2_w9": a["no2_w9"], "no2_filled": filled})


def met_features(ctx, dates, lats, lons, prefix: str = "") -> pd.DataFrame:
    e = ctx["era5"]
    t = xr.DataArray(pd.DatetimeIndex(dates), dims="p")
    la = xr.DataArray(lats, dims="p")
    lo = xr.DataArray(lons, dims="p")
    valid_t = np.isin(pd.DatetimeIndex(dates).values, e.time.values)
    out = {}
    for v in MET_VARS:
        arr = np.full(len(lats), np.nan, dtype="float32")
        if valid_t.any():
            src = e[v]
            if v == "wd":  # direção não se interpola linearmente; usa u/v
                continue
            arr[valid_t] = src.sel(time=t[valid_t]).interp(lat=la[valid_t], lon=lo[valid_t]).values
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
    """Focos por raio e ponderados pelo vento.

    Um foco está "a favor do vento" quando o azimute do ponto para o foco coincide com
    a direção DE ONDE o vento sopra: o ar chega ao ponto vindo do foco. Peso =
    max(0, cos(Δθ)) x FRP / distância, somado em 50/150/400 km. Para cada linha de
    entrada, usa os focos da mesma data."""
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
        # processa em blocos para não estourar memória na grade inteira
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
                         "dow": d.dayofweek.values, "weekend": (d.dayofweek.values >= 5).astype(int)})


def physics_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Combinações com motivação física.

    - aod_blh: AOD é coluna integrada. Dividir pela altura de mistura dá um proxy da
      extinção (logo da concentração) perto da superfície
    - aod_dry: correção higroscópica. Com UR alta, a água inflá as partículas e aumenta
      o AOD, mas o monitor mede massa seca (aquecida a ~50 °C). Logo
      AOD_seco ≈ AOD·(1-UR)^γ
    - vent: índice de ventilação (BLH x vento), a capacidade da atmosfera de diluir
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
    """Todas as variáveis "de contexto" (sem lag espacial de PM2.5) para pares (data, ponto).

    forecast=True: o alvo é D+1. Usa satélite e fogo de D (o que se conhece hoje) e
    meteorologia de D e de D+1. A de D+1 vem do ERA5, um "perfect prognosis": na
    operação viria do GFS/WRF, e isso está declarado no relatório."""
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
        nxt = met_features(ctx, dates + pd.Timedelta(days=1), lats, lons, prefix="next_")
        parts.append(nxt[["next_blh_pm", "next_ws_pm", "next_rh", "next_tp", "next_tcc", "next_t2m"]])
        parts.append(pd.DataFrame({"next_log_vent": np.log1p(nxt["next_blh_pm"] * nxt["next_ws_pm"]).values}))
        parts.append(calendar_features(dates + pd.Timedelta(days=1)).add_prefix("next_")[["next_dow", "next_weekend"]])
    out = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    out.insert(0, "date", dates)
    out.insert(1, "lat", lats)
    out.insert(2, "lon", lons)
    return out


# ----------------------------------------------------------------------------- lag espacial
def idw(target_lat, target_lon, ref_lat, ref_lon, ref_val, power: float = 2.0, k: int = 8,
        exclude_self_km: float = 0.5):
    """Interpolação pelo inverso da distância (baseline e variável de lag espacial).
    Ignora referências a menos de `exclude_self_km` (a própria estação)."""
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
    """Para cada linha (date, lat, lon), IDW dos valores de `refs` (date, lat, lon, value)
    da MESMA data. Quem chama decide quais estações entram em `refs`. Na validação, as
    estações de teste NUNCA entram (evita vazamento)."""
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
