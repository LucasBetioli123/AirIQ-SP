"""Benchmark de modelos para a ESTIMATIVA diária de PM2.5 em locais sem estação.

Compara, na validação espaço-temporal (estações nunca vistas x período posterior):
  baselines: IDW entre estações vizinhas, krigagem ordinária, climatologia mensal
  modelos:   Ridge, Random Forest, HistGradientBoosting, XGBoost, LightGBM (alvo em log)
  ablações:  LightGBM sem satélite, sem lag espacial (só satélite + meteorologia),
             alvo bruto. E, como referência OTIMISTA, validação aleatória (não é o resultado principal)

Saídas: reports/metrics/benchmark_estimativa.csv (+ _por_fold.csv) e
        data/processed/oof_estimativa.parquet (previsões fora da amostra)
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from src.aq.modeling import (CAL, EST_FEATURES, FIRE, LAG, MET, PHYS, SAT, STATIC, LogTarget, make_model,
                             metrics, period_masks, station_folds, with_lag)
from src.common.config import data_path, load_config

CFG = load_config()
warnings.filterwarnings("ignore")


def kriging_day(test_rows: pd.DataFrame, refs: pd.DataFrame) -> np.ndarray:
    from pykrige.ok import OrdinaryKriging

    if len(refs) < 6:
        return np.full(len(test_rows), np.nan)
    try:
        ok = OrdinaryKriging(refs["lon"].values, refs["lat"].values, np.log1p(refs["pm25"].values),
                             variogram_model="exponential", coordinates_type="geographic", verbose=False,
                             enable_plotting=False)
        z, _ = ok.execute("points", test_rows["lon"].values, test_rows["lat"].values)
        return np.expm1(np.asarray(z)).clip(0)
    except Exception:
        return np.full(len(test_rows), np.nan)


def kriging(te: pd.DataFrame, pool: pd.DataFrame) -> np.ndarray:
    out = np.full(len(te), np.nan)
    by_date = {d: g for d, g in pool.groupby("date")}
    for d, idx in te.groupby("date").indices.items():
        g = by_date.get(d)
        if g is not None:
            out[idx] = kriging_day(te.iloc[idx], g)
    return out


MODELS = {
    "Ridge (log)": ("ridge", EST_FEATURES, True),
    "Random Forest (log)": ("random_forest", EST_FEATURES, True),
    "HistGB (log)": ("hist_gb", EST_FEATURES, True),
    "XGBoost (log)": ("xgboost", EST_FEATURES, True),
    "LightGBM (log)": ("lightgbm", EST_FEATURES, True),
    "LightGBM (alvo bruto)": ("lightgbm", EST_FEATURES, False),
    "LightGBM sem satélite": ("lightgbm", [f for f in EST_FEATURES if f not in SAT + PHYS], True),
    "LightGBM sem lag espacial": ("lightgbm", SAT + MET + PHYS + STATIC + CAL + FIRE, True),
}


def fit_predict(kind, feats, log, Xtr, ytr, Xte):
    m = make_model(kind)
    m = LogTarget(m) if log else m
    m.fit(Xtr[feats], ytr)
    return np.clip(m.predict(Xte[feats]), 0, None)


def run(df: pd.DataFrame | None = None, models: dict | None = None, do_kriging: bool = True) -> pd.DataFrame:
    df = df if df is not None else pd.read_parquet(data_path("processed", "station_days.parquet"))
    df = df.sort_values(["date", "location_id"]).reset_index(drop=True)
    models = models or MODELS
    folds = station_folds(df["location_id"].values)
    df["fold"] = df["location_id"].map(folds)
    per = period_masks(df["date"])
    preds, rows = [], []
    for k in sorted(df["fold"].unique()):
        t0 = time.time()
        pool = df[df["fold"] != k]                     # estações de referência (treino)
        tr = with_lag(pool[per["train"][pool.index]], pool)
        te_mask = (df["fold"] == k) & per["test"]
        te = with_lag(df[te_mask], pool)
        out = te[["location_id", "name", "date", "lat", "lon", "pm25", "fold"]].copy()
        out["IDW (baseline)"] = te["idw"].values
        if do_kriging:
            out["Krigagem ordinária"] = kriging(te.reset_index(drop=True), pool)
        clim = tr.groupby(tr["date"].dt.month)["pm25"].mean()
        out["Climatologia mensal"] = te["date"].dt.month.map(clim).values
        for name, (kind, feats, log) in models.items():
            out[name] = fit_predict(kind, feats, log, tr, tr["pm25"].values, te)
        preds.append(out)
        print(f"fold {k}: {te['location_id'].nunique()} estações de teste, {len(te)} linhas, {time.time() - t0:.0f}s")
    oof = pd.concat(preds, ignore_index=True)
    cols = [c for c in oof.columns if c not in ("location_id", "name", "date", "lat", "lon", "pm25", "fold")]
    ok = oof[cols].notna().all(axis=1)  # compara todos nas MESMAS linhas
    res = []
    for c in cols:
        m = metrics(oof.loc[ok, "pm25"], oof.loc[ok, c], baseline=oof.loc[ok, "IDW (baseline)"])
        fm = [metrics(g["pm25"], g[c])["MAE"] for _, g in oof[ok].groupby("fold")]
        res.append({"modelo": c, **m, "MAE_std_folds": float(np.std(fm))})
    res = pd.DataFrame(res).sort_values("MAE")
    # referência otimista: validação aleatória (linhas embaralhadas), NÃO é o resultado principal
    rnd = random_cv_reference(df)
    res = pd.concat([res, pd.DataFrame([{"modelo": "LightGBM (log) - validação ALEATÓRIA [otimista]", **rnd}])])
    res.to_csv(data_path("metrics", "benchmark_estimativa.csv"), index=False, float_format="%.4f")
    oof.to_parquet(data_path("processed", "oof_estimativa.parquet"), index=False)
    print(res[["modelo", "n", "MAE", "RMSE", "R2", "ganho_MAE_vs_baseline_%", "acerto_faixa_IQAr"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return res


def random_cv_reference(df: pd.DataFrame) -> dict:
    from sklearn.model_selection import KFold

    d = with_lag(df, df)  # lag com todas as estações (inclusive vizinhas do mesmo dia)
    yhat = np.zeros(len(d))
    for tr, te in KFold(5, shuffle=True, random_state=CFG["seed"]).split(d):
        yhat[te] = fit_predict("lightgbm", EST_FEATURES, True, d.iloc[tr], d["pm25"].values[tr], d.iloc[te])
    return metrics(d["pm25"], yhat)


if __name__ == "__main__":
    run()
