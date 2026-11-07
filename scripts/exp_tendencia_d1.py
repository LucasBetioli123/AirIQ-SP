"""Experiment: D-1 -> D trend in the D+1 forecast (scenarios A and B), same validation as train.py."""
import sys
import numpy as np
import pandas as pd

from src.aq.modeling import TEMPORAL_FOLDS, metrics, period_masks, station_folds, with_lag
from src.aq.train import FC_FEATURES_MON, FC_FEATURES_UNMON, point_model
from src.aq.evaluate import band_metrics
from src.common.config import data_path

fc = pd.read_parquet(data_path("processed", "station_days_fc.parquet"))
fc = fc.rename(columns={"pm25": "pm25_today"}).sort_values(["date", "location_id"]).reset_index(drop=True)
# the station's own PM2.5 on D-1 (known at the end of D)
prev = fc[["location_id", "date", "pm25_today"]].assign(date=lambda x: x["date"] + pd.Timedelta(days=1))
fc = fc.merge(prev.rename(columns={"pm25_today": "pm25_yday"}), on=["location_id", "date"], how="left")
refs_all = fc.rename(columns={"pm25_today": "pm25"})
refs_yday = refs_all[["date", "lat", "lon", "pm25"]].assign(date=lambda x: x["date"] + pd.Timedelta(days=1))


def add_trend(rows, refs_today, refs_y):
    r = with_lag(rows, refs_today)
    r = with_lag(r, refs_y, prefix="idwy")
    r["idw_trend"] = np.log1p(r["idw"]) - np.log1p(r["idwy"])
    if "pm25_yday" in r:
        r["own_trend"] = np.log1p(r["pm25_today"]) - np.log1p(r["pm25_yday"])
    return r


NEW_U = ["idwy", "idw_trend"]
NEW_M = NEW_U + ["pm25_yday", "own_trend"]
scen = sys.argv[1] if len(sys.argv) > 1 else "A"
rows = []
folds = station_folds(fc["location_id"].values)
fc["fold"] = fc["location_id"].map(folds)
lag_all = add_trend(fc, refs_all, refs_yday)
for tf in TEMPORAL_FOLDS:
    per = period_masks(fc["date"] + pd.Timedelta(days=1), tf)
    if scen == "A":
        tr, te = lag_all[per["train"]], lag_all[per["test"]]
        for name, feats in [("base", FC_FEATURES_MON), ("tendencia", FC_FEATURES_MON + NEW_M)]:
            p = point_model("pm25_today").fit(tr[feats], tr["pm25_next"].values).predict(te[feats])
            rows.append(pd.DataFrame({"m": name, "tf": tf["name"], "y": te["pm25_next"].values, "p": p}))
    else:
        for k in sorted(fc["fold"].unique()):
            pool, pooly = refs_all[fc["fold"] != k], refs_yday[fc["fold"] != k]
            trk = add_trend(fc[(fc["fold"] != k) & per["train"]], pool, pooly)
            tek = add_trend(fc[(fc["fold"] == k) & per["test"]], pool, pooly)
            for name, feats in [("base", FC_FEATURES_UNMON), ("tendencia", FC_FEATURES_UNMON + NEW_U)]:
                p = point_model("idw").fit(trk[feats], trk["pm25_next"].values).predict(tek[feats])
                rows.append(pd.DataFrame({"m": name, "tf": tf["name"], "y": tek["pm25_next"].values, "p": p}))
r = pd.concat(rows)
for name, g in r.groupby("m"):
    g = g.dropna()
    m, b = metrics(g.y, g.p), band_metrics(g.y, g.p)
    print(scen, name, len(g), f"MAE {m['MAE']:.3f} RMSE {m['RMSE']:.3f} R2 {m['R2']:.3f} "
          f"bal {b['acuracia_balanceada']:.3f} sens25 {b['sensibilidade_>25']:.3f} prec25 {b['precisao_>25']:.3f}",
          " | ".join(f"{t}:{metrics(x.y, x.p)['MAE']:.3f}" for t, x in g.groupby("tf")))
