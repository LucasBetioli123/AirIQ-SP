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
    """Amostra versionada (poucos MB) que roda TODO o pipeline de modelagem em minutos.

    - bases de treino já com as variáveis (estações x dias do período inteiro):
      benchmark, treino, intervalos, D+1, SHAP e episódios reproduzem os números do relatório
    - cubos de satélite/meteorologia completos SÓ para algumas janelas de datas: mapas,
      CLI, API e agente funcionam nessas datas (lista em data/sample/README.md)
    O caminho completo (download -> base) é `make data`."""
    src, dst, proc = data_path("interim"), data_path("sample"), data_path("processed")
    for f in ("station_days.parquet", "station_days_fc.parquet"):
        pd.read_parquet(proc / f).to_parquet(dst / f, index=False, compression="zstd")
    shutil.copy(src / "stations.parquet", dst / "stations.parquet")
    pd.read_parquet(src / "pm25_daily.parquet").to_parquet(dst / "pm25_daily.parquet", index=False, compression="zstd")
    days = pd.DatetimeIndex([])
    for d0, d1 in CFG["sample"]["context_windows"]:
        days = days.union(pd.date_range(pd.Timestamp(d0) - pd.Timedelta(days=2), pd.Timestamp(d1) + pd.Timedelta(days=2)))
    enc = lambda ds: {v: {"zlib": True, "complevel": 9, "least_significant_digit": 3} for v in ds.data_vars}  # noqa: E731
    for name in ("aod_daily.nc", "no2_daily.nc", "era5_daily.nc"):
        ds = xr.load_dataset(src / name)
        ds = ds.sel(time=ds.time.to_index().isin(days))
        ds.to_netcdf(dst / name, encoding=enc(ds))
    stc = xr.load_dataset(src / "static_0p01.nc")
    stc.to_netcdf(dst / "static_0p01.nc", encoding={v: {"zlib": True, "complevel": 9, "least_significant_digit": 2}
                                                   for v in stc.data_vars})
    f = pd.read_parquet(src / "fires_daily.parquet")
    f["date"] = pd.to_datetime(f["date"])
    f[f["date"].isin(days)].to_parquet(dst / "fires_daily.parquet", index=False, compression="zstd")
    total = sum(p.stat().st_size for p in dst.iterdir()) / 1e6
    print(f"amostra gravada em {dst} ({total:.1f} MB)")


def from_sample():
    """Copia as bases de treino da amostra para data/processed (sem baixar nada)."""
    for f in ("station_days.parquet", "station_days_fc.parquet"):
        shutil.copy(data_path("sample", f), data_path("processed", f))
    print("bases de treino copiadas da amostra para data/processed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-sample", action="store_true")
    ap.add_argument("--make-sample", action="store_true")
    a = ap.parse_args()
    if a.make_sample:
        make_sample()
    elif a.from_sample:
        from_sample()
    else:
        build()
