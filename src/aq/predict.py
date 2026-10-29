"""Inferência: PM2.5 (estimativa do dia ou previsão D+1) para qualquer coordenada/data.

É o mesmo código usado pelo mapa, pela CLI, pela API e pela ferramenta do agente.
A estimativa usa as variáveis de satélite/meteorologia da data e o IDW das estações
CETESB que mediram naquela data (como seria na operação, com a rede em tempo real).
"""
from __future__ import annotations

import functools

import joblib
import numpy as np
import pandas as pd

from src.aq.aqi import band, legal_standard_24h
from src.aq.features import haversine, load_context, point_features
from src.aq.modeling import FEATURE_SCHEMA, with_lag
from src.common.config import ROOT, data_path


class DataUnavailable(ValueError):
    pass


def _use_sample() -> bool:
    return not (data_path("interim") / "aod_daily.nc").exists()


@functools.lru_cache(maxsize=1)
def models():
    mdir = ROOT / "models"
    if not (mdir / "estimativa.joblib").exists():
        raise FileNotFoundError("modelos não treinados: rode `make train` (ou `make all-sample`)")
    est, fc = joblib.load(mdir / "estimativa.joblib"), joblib.load(mdir / "previsao_d1.joblib")
    if any(m.get("feature_schema") != FEATURE_SCHEMA for m in (est, fc)):
        raise ValueError("modelos antigos: rode `make sample train` para treinar a versão causal")
    return est, fc


@functools.lru_cache(maxsize=1)
def station_obs() -> pd.DataFrame:
    p = data_path("processed", "station_days.parquet")
    if p.exists():
        d = pd.read_parquet(p, columns=["location_id", "name", "date", "lat", "lon", "pm25"])
    else:
        base = data_path("sample")
        d = pd.read_parquet(base / "pm25_daily.parquet").merge(pd.read_parquet(base / "stations.parquet"))
    d["date"] = pd.to_datetime(d["date"])
    return d


@functools.lru_cache(maxsize=1)
def available_index() -> pd.DatetimeIndex:
    ctx = load_context(_use_sample())
    return pd.DatetimeIndex(ctx["aod"].time.values).intersection(pd.DatetimeIndex(ctx["era5"].time.values))


def available_dates() -> tuple[pd.Timestamp, pd.Timestamp]:
    t = available_index()
    return t.min(), t.max()


def predict_points(dates, lats, lons, horizon: int = 0) -> pd.DataFrame:
    """horizon=0: estimativa para `dates`. horizon=1: previsão para `dates`+1, feita com
    o que se sabe em `dates` (satélite, fogo e medições de hoje)."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    if horizon not in (0, 1):
        raise ValueError("horizon deve ser 0 ou 1")
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    idx = available_index()
    need = dates + pd.Timedelta(days=horizon)
    if not dates.isin(idx).all():
        lo_d, hi_d = available_dates()
        extra = " (modo amostra: só algumas janelas de datas; rode `make data` para o período completo)" \
            if _use_sample() else ""
        raise DataUnavailable(f"sem dados de satélite/meteorologia para essa data; período disponível: "
                              f"{lo_d.date()} a {hi_d.date()}{extra}")
    est, fc = models()
    ctx = load_context(_use_sample())
    feats = point_features(ctx, dates, lats, lons, forecast=horizon == 1)
    obs = station_obs()
    feats = with_lag(feats, obs)
    stations = obs[["lat", "lon"]].drop_duplicates()
    d_near = np.array([haversine(la, lo, stations.lat.values, stations.lon.values).min() for la, lo in zip(lats, lons)])
    if horizon == 0:
        m = est
        X = feats[m["features"]]
        point = m["point"].predict(X)
        lo, _, hi = m["quantiles"].predict_log(X)
    else:
        m = fc
        X = feats[m["features_unmon"]]
        point = m["point_unmon"].predict(X)
        # no local de uma estação que mediu hoje: usa o modelo com o PM2.5 de hoje
        today = obs.groupby(["date", "lat", "lon"])["pm25"].mean()
        own = np.array([today.get((d, la, lo), np.nan) if dn < 0.5 else np.nan
                        for d, la, lo, dn in zip(dates, lats, lons, d_near)])
        has = np.isfinite(own)
        lo, _, hi = m["quantiles"].predict_log(X)
        q = np.full(len(X), m["cqr_q"])
        if has.any():
            Xm = feats.loc[has, m["features_mon"][:-1]].assign(pm25_today=own[has])
            point[has] = m["point_mon"].predict(Xm)
            lo[has], _, hi[has] = m["quantiles_mon"].predict_log(Xm)
            q[has] = m["cqr_q_mon"]
    if horizon == 0:
        q = m["cqr_q"]
    lo, hi = np.expm1(lo - q).clip(0), np.expm1(hi + q)
    lo, hi = np.minimum(lo, point), np.maximum(hi, point)
    out = pd.DataFrame({"data_referencia": dates, "data_alvo": need, "lat": lats, "lon": lons,
                        "pm25": point, "ic90_inf": lo, "ic90_sup": hi, "faixa_iqar": band(point),
                        "aod_disponivel": feats["aod_fill_level"].values < 4,
                        "aod_nivel_preenchimento": feats["aod_fill_level"].values,
                        "estacoes_usadas_idw": feats["idw_n"].values, "dist_estacao_mais_proxima_km": d_near})
    return out


def predict_one(lat: float, lon: float, date: str, horizon: int = 0) -> dict:
    r = predict_points([date], [lat], [lon], horizon).iloc[0]
    std = legal_standard_24h(r["data_alvo"])
    return {
        "lat": lat, "lon": lon, "data_referencia": str(r["data_referencia"].date()),
        "data_alvo": str(r["data_alvo"].date()), "horizonte_dias": horizon,
        "pm25_ug_m3": round(float(r["pm25"]), 1),
        "intervalo_90": [round(float(r["ic90_inf"]), 1), round(float(r["ic90_sup"]), 1)],
        "faixa_iqar": r["faixa_iqar"],
        "padrao_24h_vigente": std,
        "ultrapassa_padrao": None if std["valor"] is None else bool(r["pm25"] > std["valor"]),
        "diagnostico": {"aod_disponivel": bool(r["aod_disponivel"]),
                        "aod_nivel_preenchimento": int(r["aod_nivel_preenchimento"]),
                        "estacoes_usadas_idw": int(r["estacoes_usadas_idw"]),
                        "dist_estacao_mais_proxima_km": round(float(r["dist_estacao_mais_proxima_km"]), 1)},
        "fonte": "modelo LightGBM + CQR (Parte 1); satélite GOES-16, ERA5, rede CETESB via OpenAQ",
        "feature_schema": FEATURE_SCHEMA,
        "emissao": "fim do dia de referência; dados históricos, sem feed em tempo real",
    }
