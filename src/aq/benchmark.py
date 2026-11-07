"""Model benchmark for the daily PM2.5 ESTIMATE at sites without a station.

Compares, under spatio-temporal validation (never-seen stations x later period):
  baselines: IDW between neighboring stations, ordinary kriging, monthly climatology
  models:    Ridge, Random Forest, HistGradientBoosting, XGBoost, LightGBM (log target)
  ablations: LightGBM without satellite data, without spatial lag (satellite + weather
             only), raw target. Plus, as an OPTIMISTIC reference, random validation (not the
             headline result)

Outputs: reports/metrics/benchmark_estimativa.csv (+ _por_fold.csv) and
         data/processed/oof_estimativa.parquet (out-of-sample predictions)
Model labels stay in Portuguese: they are data keys used by the reports and the dashboard.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from src.aq.features import haversine
from src.aq.modeling import (CAL, EST_FEATURES, FIRE, MET, PHYS, SAT, STATIC, TEMPORAL_FOLDS, LogTarget,
                             OffsetLogTarget, make_model, metrics, period_masks, station_folds, with_lag)
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


# (model, features, target): "log" target = log1p(PM); "offset" = log1p(PM) - log1p(day's IDW)
MODELS = {
    "LightGBM (offset IDW)": ("lightgbm", EST_FEATURES, "offset"),
    "LightGBM regularizado (offset IDW)": ("lightgbm_reg", EST_FEATURES, "offset"),
    "Random Forest (offset IDW)": ("random_forest", EST_FEATURES, "offset"),
    "XGBoost (offset IDW)": ("xgboost", EST_FEATURES, "offset"),
    "XGBoost regularizado (offset IDW)": ("xgboost_reg", EST_FEATURES, "offset"),
    "XGBoost reg. forte (offset IDW)": ("xgboost_forte", EST_FEATURES, "offset"),
    "Ensemble RF + LightGBM reg. (offset IDW)": ("ensemble_rf_lgbm", EST_FEATURES, "offset"),
    "LightGBM (offset) sem satélite": ("lightgbm", [f for f in EST_FEATURES if f not in SAT + PHYS], "offset"),
    "MLP 64-32 (offset IDW)": ("mlp", EST_FEATURES, "offset"),
    "MLP PyTorch 64-32, saída tanh [-0,7; 0,7] (offset IDW)": ("mlp_tanh", EST_FEATURES, "offset"),
    "MLP PyTorch 64-32, saída tanh [-0,7; 0,7] (log)": ("mlp_tanh", EST_FEATURES, True),
    "Ridge (offset IDW)": ("ridge", EST_FEATURES, "offset"),
    "Ridge (log)": ("ridge", EST_FEATURES, True),
    "Random Forest (log)": ("random_forest", EST_FEATURES, True),
    "HistGB (log)": ("hist_gb", EST_FEATURES, True),
    "XGBoost (log)": ("xgboost", EST_FEATURES, True),
    "LightGBM (log)": ("lightgbm", EST_FEATURES, True),
    "LightGBM regularizado (log)": ("lightgbm_reg", EST_FEATURES, True),
    "Ensemble RF + LightGBM reg. (log)": ("ensemble_rf_lgbm", EST_FEATURES, True),
    "LightGBM (alvo bruto)": ("lightgbm", EST_FEATURES, False),
    "LightGBM sem satélite": ("lightgbm", [f for f in EST_FEATURES if f not in SAT + PHYS], True),
    "LightGBM sem lag espacial": ("lightgbm", SAT + MET + PHYS + STATIC + CAL + FIRE, True),
}


def fit_predict(kind, feats, target, Xtr, ytr, Xte):
    m = make_model(kind)
    if target == "offset":
        feats = feats if "idw" in feats else feats + ["idw"]
        m = OffsetLogTarget(m)
    elif target:
        m = LogTarget(m)
    m.fit(Xtr[feats], ytr)
    return np.clip(m.predict(Xte[feats]), 0, None)


def one_fold(df, per, k, tname, models, do_kriging) -> pd.DataFrame:
    """One (temporal split, station fold) pair: train without the fold's stations and predict on
    them over the test period. The IDW (baseline and feature) uses only reference stations."""
    t0 = time.time()
    pool = df[df["fold"] != k]
    tr = with_lag(pool[per["train"][pool.index]], pool)
    te = with_lag(df[(df["fold"] == k) & per["test"]], pool)
    out = te[["location_id", "name", "date", "lat", "lon", "pm25", "fold"]].copy()
    out["tfold"] = tname
    out["IDW (baseline)"] = te["idw"].values
    if do_kriging:
        out["Krigagem ordinária"] = kriging(te.reset_index(drop=True), pool)
    clim = tr.groupby(tr["date"].dt.month)["pm25"].mean()
    # month absent from training (T1 has no Jul-Dec in training) -> overall training mean
    out["Climatologia mensal"] = te["date"].dt.month.map(clim).fillna(tr["pm25"].mean()).values
    for name, (kind, feats, log) in models.items():
        out[name] = fit_predict(kind, feats, log, tr, tr["pm25"].values, te)
    print(f"{tname} fold {k}: {te['location_id'].nunique()} test stations, {len(te)} rows, "
          f"{time.time() - t0:.0f}s", flush=True)
    return out


def run(df: pd.DataFrame | None = None, models: dict | None = None, do_kriging: bool = True) -> pd.DataFrame:
    df = df if df is not None else pd.read_parquet(data_path("processed", "station_days.parquet"))
    df = df.sort_values(["date", "location_id"]).reset_index(drop=True)
    models = models or MODELS
    folds = station_folds(df["location_id"].values)
    df["fold"] = df["location_id"].map(folds)
    preds = []
    for tf in TEMPORAL_FOLDS:
        per = period_masks(df["date"], tf)
        for k in sorted(df["fold"].unique()):
            preds.append(one_fold(df, per, k, tf["name"], models, do_kriging))
    oof = pd.concat(preds, ignore_index=True)
    # distance from each test station to the nearest reference station (outside its fold)
    st = df.groupby("location_id")[["lat", "lon"]].first()
    dmin = {s: min(haversine(st.lat[s], st.lon[s], st.lat[r], st.lon[r]) for r in st.index if folds[r] != folds[s])
            for s in st.index}
    oof["dist_ref_km"] = oof["location_id"].map(dmin)
    cols = [c for c in oof.columns if c not in ("location_id", "name", "date", "lat", "lon", "pm25", "fold", "tfold",
                                                "dist_ref_km")]
    ok = oof[cols].notna().all(axis=1)  # compare every model on the SAME rows
    res = []
    for c in cols:
        m = metrics(oof.loc[ok, "pm25"], oof.loc[ok, c], baseline=oof.loc[ok, "IDW (baseline)"])
        fm = [metrics(g["pm25"], g[c])["MAE"] for _, g in oof[ok].groupby("fold")]
        res.append({"modelo": c, **m, "MAE_std_folds": float(np.std(fm))})
    res = pd.DataFrame(res).sort_values("MAE")
    # by temporal split and by distance to the nearest reference station
    sub = []
    oof["faixa_dist"] = pd.cut(oof["dist_ref_km"], [0, 10, 25, 50, 1000], labels=["<10 km", "10-25 km", "25-50 km", ">50 km"])
    for key in ("tfold", "faixa_dist"):
        for g, gg in oof[ok].groupby(key, observed=True):
            for c in cols:
                m = metrics(gg["pm25"], gg[c], baseline=gg["IDW (baseline)"])
                sub.append({"recorte": key, "grupo": str(g), "modelo": c, **m, "estacoes": gg["location_id"].nunique()})
    pd.DataFrame(sub).to_csv(data_path("metrics", "benchmark_estimativa_recortes.csv"), index=False, float_format="%.4f")
    # optimistic reference: random validation (shuffled rows), NOT the headline result
    rnd = random_cv_reference(df)
    res = pd.concat([res, pd.DataFrame([{"modelo": "LightGBM (offset IDW) - validação ALEATÓRIA [otimista]", **rnd}])])
    res.to_csv(data_path("metrics", "benchmark_estimativa.csv"), index=False, float_format="%.4f")
    oof.to_parquet(data_path("processed", "oof_estimativa.parquet"), index=False)
    print(res[["modelo", "n", "MAE", "RMSE", "R2", "ganho_MAE_vs_baseline_%", "acerto_faixa_IQAr"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return res


def random_cv_reference(df: pd.DataFrame) -> dict:
    from sklearn.model_selection import KFold

    d = with_lag(df.drop(columns=["fold"], errors="ignore"), df)  # lag computed from all stations
    yhat = np.zeros(len(d))
    for tr, te in KFold(5, shuffle=True, random_state=CFG["seed"]).split(d):
        yhat[te] = fit_predict("lightgbm", EST_FEATURES, "offset", d.iloc[tr], d["pm25"].values[tr], d.iloc[te])
    return metrics(d["pm25"], yhat)


if __name__ == "__main__":
    run()
