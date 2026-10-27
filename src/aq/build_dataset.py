"""Monta as bases de treino a partir dos dados brutos já processados em data/interim.

Saídas (data/processed):
  station_days.parquet     estimativa do mesmo dia: 1 linha por (estação, dia), alvo pm25
  station_days_fc.parquet  previsão D+1: variáveis de D (+ meteorologia de D+1), alvo pm25_next

  python -m src.aq.build_dataset                # a partir de data/interim (após `make data`)
  python -m src.aq.build_dataset --make-sample  # gera a amostra versionada em data/sample
  python -m src.aq.build_dataset --from-sample  # usa a amostra (roda tudo em poucos minutos)
"""
from __future__ import annotations

import argparse
import shutil

import numpy as np
import pandas as pd
import xarray as xr

from src.aq.features import load_context, point_features
from src.common.config import data_path, load_config

CFG = load_config()


def build(sample: bool = False):
    base = data_path("sample") if sample else data_path("interim")
    st = pd.read_parquet(base / "stations.parquet")
    pm = pd.read_parquet(base / "pm25_daily.parquet").merge(st, on="location_id")
    pm["date"] = pd.to_datetime(pm["date"])
    ctx = load_context(sample)

    feats = point_features(ctx, pm["date"], pm["lat"], pm["lon"])
    est = pd.concat([pm[["location_id", "name", "pm25", "n_hours"]].reset_index(drop=True),
                     feats.reset_index(drop=True)], axis=1)
    est.to_parquet(data_path("processed", "station_days.parquet"), index=False)

    # previsão: para cada (estação, D) com medição em D+1
    nxt = pm[["location_id", "date", "pm25"]].copy()
    nxt["date"] = nxt["date"] - pd.Timedelta(days=1)
    fc = pm.merge(nxt.rename(columns={"pm25": "pm25_next"}), on=["location_id", "date"], how="inner")
    ffc = point_features(ctx, fc["date"], fc["lat"], fc["lon"], forecast=True)
    fcd = pd.concat([fc[["location_id", "name", "pm25", "pm25_next"]].reset_index(drop=True),
                     ffc.reset_index(drop=True)], axis=1)
    fcd.to_parquet(data_path("processed", "station_days_fc.parquet"), index=False)
    print(f"estimativa: {len(est)} linhas x {est.shape[1]} col | previsão D+1: {len(fcd)} linhas")
    print(f"AOD observado na célula: {np.isfinite(est['aod']).mean():.1%} | após preenchimento: "
          f"{np.isfinite(est['aod_filled']).mean():.1%}")
    print(est["aod_fill_level"].value_counts(normalize=True).sort_index().round(3).to_dict())
    return est, fcd


def make_sample():
    """Recorte pequeno (período `sample` do config) de todos os insumos, ~poucos MB."""
    s, e = pd.Timestamp(CFG["sample"]["start"]), pd.Timestamp(CFG["sample"]["end"])
    src, dst = data_path("interim"), data_path("sample")
    shutil.copy(src / "stations.parquet", dst / "stations.parquet")
    pm = pd.read_parquet(src / "pm25_daily.parquet")
    pm[(pm.date >= s) & (pm.date <= e)].to_parquet(dst / "pm25_daily.parquet", index=False)
    pad = pd.Timedelta(days=2)
    enc = lambda ds: {v: {"zlib": True, "complevel": 6} for v in ds.data_vars}  # noqa: E731
    aod = xr.load_dataset(src / "aod_daily.nc").sel(time=slice(s - pad, e + pad))
    aod.to_netcdf(dst / "aod_daily.nc", encoding=enc(aod))
    no2 = xr.load_dataset(src / "no2_daily.nc").sel(time=slice(s - pad, e + pad))
    no2.to_netcdf(dst / "no2_daily.nc", encoding=enc(no2))
    era = xr.load_dataset(src / "era5_daily.nc").sel(time=slice(s - pad, e + pad))
    era.to_netcdf(dst / "era5_daily.nc", encoding=enc(era))
    stc = xr.load_dataset(src / "static_0p01.nc")
    stc.to_netcdf(dst / "static_0p01.nc", encoding=enc(stc))
    f = pd.read_parquet(src / "fires_daily.parquet")
    f["date"] = pd.to_datetime(f["date"])
    f[(f.date >= s - pad) & (f.date <= e + pad)].to_parquet(dst / "fires_daily.parquet", index=False)
    print("amostra gravada em", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-sample", action="store_true")
    ap.add_argument("--make-sample", action="store_true")
    a = ap.parse_args()
    if a.make_sample:
        make_sample()
    else:
        build(sample=a.from_sample)
