"""Treino do modelo escolhido no benchmark (LightGBM, alvo em log), com intervalo de
confiança e previsão do dia seguinte.

1) INTERVALOS na validação espaço-temporal. Para cada fold de estações, comparamos:
   a) quantis do LightGBM (5%/95%) sem calibração
   b) CQR: os mesmos quantis conformalizados. Os escores de calibração vêm de
      cross-fitting por grupos de estação no período de calibração (o modelo interno
      não viu a estação), o que imita o erro num local sem monitor
   c) conformal do resíduo absoluto em torno da previsão pontual (mesmo cross-fitting)
   e reportamos a cobertura real no teste e a largura média.

2) PREVISÃO D+1, comparada com a persistência (amanhã = hoje):
   A) local COM estação: o modelo sabe o PM2.5 de hoje da própria estação (split temporal)
   B) local SEM estação: estações de teste fora do treino. Hoje só se conhece o IDW,
      então a "persistência" é o IDW de hoje

3) MODELOS FINAIS (para mapa, CLI, API e agente): treinados com todas as estações até o
   fim da calibração, de modo que o período de teste continua fora da amostra no tempo.

Saídas: models/*.joblib, reports/metrics/{intervalos,previsao_d1,...}.csv
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd

from src.aq.modeling import (CQR, EST_FEATURES, FC_EXTRA, SPLIT, LogTarget, QuantileTrio, make_model, metrics,
                             period_masks, station_folds, with_lag)
from src.common.config import data_path

warnings.filterwarnings("ignore")
ALPHA = SPLIT["interval_alpha"]
FC_FEATURES_UNMON = EST_FEATURES + FC_EXTRA
FC_FEATURES_MON = FC_FEATURES_UNMON + ["pm25_today"]


def crossfit_calibration(pool: pd.DataFrame, per_pool: dict, feats: list[str], k_inner: int = 4,
                         target: str = "pm25", lag_col: str = "pm25"):
    """Previsões 'fora da estação' no período de calibração: para cada grupo interno de
    estações, treina sem ele (período de treino) e prevê nele (período de calibração).
    `lag_col` é a medição usada no IDW (na previsão D+1 é a de HOJE, nunca o alvo).
    Devolve (y, lo_log, hi_log, yhat_point)."""
    inner = station_folds(pool["location_id"].values, k_inner)
    g = pool["location_id"].map(inner)
    ys, los, his, pts = [], [], [], []
    for j in range(k_inner):
        ref = pool[g != j]
        tr = with_lag(ref[per_pool["train"][ref.index]], ref, value_col=lag_col)
        ca_rows = pool[(g == j) & per_pool["calib"]]
        if ca_rows.empty:
            continue
        ca = with_lag(ca_rows, ref, value_col=lag_col)
        qt = QuantileTrio(ALPHA).fit(tr[feats], tr[target].values)
        lo, _, hi = qt.predict_log(ca[feats])
        pt = LogTarget(make_model("lightgbm")).fit(tr[feats], tr[target].values).predict(ca[feats])
        ys.append(ca[target].values)
        los.append(lo)
        his.append(hi)
        pts.append(pt)
    return np.concatenate(ys), np.concatenate(los), np.concatenate(his), np.concatenate(pts)


def abs_conformal_q(y, yhat, alpha=ALPHA):
    s = np.abs(np.asarray(y) - np.asarray(yhat))
    n = len(s)
    return float(np.quantile(s, min(1, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


def evaluate_intervals(df: pd.DataFrame) -> pd.DataFrame:
    feats = EST_FEATURES
    folds = station_folds(df["location_id"].values)
    df = df.assign(fold=df["location_id"].map(folds))
    per = period_masks(df["date"])
    out = []
    for k in sorted(df["fold"].unique()):
        pool = df[df["fold"] != k]
        per_pool = {n: m[pool.index] for n, m in per.items()}
        tr = with_lag(pool[per_pool["train"]], pool)
        te = with_lag(df[(df["fold"] == k) & per["test"]], pool)
        qt = QuantileTrio(ALPHA).fit(tr[feats], tr["pm25"].values)
        lo, med, hi = qt.predict_log(te[feats])
        point = LogTarget(make_model("lightgbm")).fit(tr[feats], tr["pm25"].values).predict(te[feats])
        yc, loc, hic, ptc = crossfit_calibration(pool, per_pool, feats)
        cqr = CQR(ALPHA).calibrate(yc, loc, hic)
        clo, chi = cqr.interval(lo, hi)
        qa = abs_conformal_q(yc, ptc)
        r = te[["location_id", "name", "date", "lat", "lon", "pm25", "fold"]].copy()
        r["pred"] = point
        r["q_lo"], r["q_hi"] = np.expm1(lo).clip(0), np.expm1(hi)
        r["cqr_lo"], r["cqr_hi"] = clo, chi
        r["abs_lo"], r["abs_hi"] = (point - qa).clip(0), point + qa
        r["cqr_q"] = cqr.q
        out.append(r)
        print(f"fold {k}: CQR q={cqr.q:.3f} (log) | conformal |res| q={qa:.1f} µg/m³")
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(data_path("processed", "oof_intervalos.parquet"), index=False)
    tab = []
    for name, lo, hi in [("Quantis LightGBM 5-95% (sem calibrar)", "q_lo", "q_hi"),
                         ("CQR (quantis conformalizados)", "cqr_lo", "cqr_hi"),
                         ("Conformal do resíduo absoluto", "abs_lo", "abs_hi")]:
        m = metrics(res["pm25"], res["pred"], res[lo], res[hi])
        tab.append({"metodo": name, "cobertura_nominal": 1 - ALPHA, "cobertura_real": m["cobertura"],
                    "largura_media": m["largura_media"],
                    "cobertura_pm25_>25": float(np.mean(((res.pm25 >= res[lo]) & (res.pm25 <= res[hi]))[res.pm25 > 25]))})
    tab = pd.DataFrame(tab)
    tab.to_csv(data_path("metrics", "intervalos.csv"), index=False, float_format="%.4f")
    print(tab.to_string(index=False))
    return tab


# ----------------------------------------------------------------------------- previsão D+1
def forecast_eval(fc: pd.DataFrame) -> pd.DataFrame:
    fc = fc.rename(columns={"pm25": "pm25_today"}).sort_values(["date", "location_id"]).reset_index(drop=True)
    per = period_masks(fc["date"])
    refs_all = fc.rename(columns={"pm25_today": "pm25"})  # medições de HOJE (D) para o lag
    rows = []

    # A) com estação: split só temporal, todas as estações
    lag = with_lag(fc, refs_all)
    tr, te = lag[per["train"]], lag[per["test"]]
    m = LogTarget(make_model("lightgbm")).fit(tr[FC_FEATURES_MON], tr["pm25_next"].values)
    pa = m.predict(te[FC_FEATURES_MON])
    rows.append({"cenario": "A) local com estação", "metodo": "Persistência (amanhã = hoje)",
                 **metrics(te["pm25_next"], te["pm25_today"])})
    rows.append({"cenario": "A) local com estação", "metodo": "LightGBM D+1",
                 **metrics(te["pm25_next"], pa, baseline=te["pm25_today"])})

    # B) sem estação: folds de estação + período posterior
    folds = station_folds(fc["location_id"].values)
    fc["fold"] = fc["location_id"].map(folds)
    pb, yb, idwb = [], [], []
    for k in sorted(fc["fold"].unique()):
        pool = refs_all[fc["fold"] != k]
        trk = with_lag(fc[(fc["fold"] != k) & per["train"]], pool)
        tek = with_lag(fc[(fc["fold"] == k) & per["test"]], pool)
        mk = LogTarget(make_model("lightgbm")).fit(trk[FC_FEATURES_UNMON], trk["pm25_next"].values)
        pb.append(mk.predict(tek[FC_FEATURES_UNMON]))
        yb.append(tek["pm25_next"].values)
        idwb.append(tek["idw"].values)
    pb, yb, idwb = map(np.concatenate, (pb, yb, idwb))
    ok = np.isfinite(idwb)
    rows.append({"cenario": "B) local sem estação", "metodo": "Persistência do IDW de hoje",
                 **metrics(yb[ok], idwb[ok])})
    rows.append({"cenario": "B) local sem estação", "metodo": "LightGBM D+1",
                 **metrics(yb[ok], pb[ok], baseline=idwb[ok])})
    tab = pd.DataFrame(rows)
    tab.to_csv(data_path("metrics", "previsao_d1.csv"), index=False, float_format="%.4f")
    print(tab[["cenario", "metodo", "n", "MAE", "RMSE", "R2", "acerto_faixa_IQAr"]].to_string(index=False))
    return tab


# ----------------------------------------------------------------------------- modelos finais
def fit_final(df: pd.DataFrame, fc: pd.DataFrame):
    """Modelos de produção: todas as estações, datas até o fim da calibração."""
    cut = pd.Timestamp(SPLIT["calib_end"])
    d = df[df["date"] <= cut].reset_index(drop=True)
    lagd = with_lag(d, d)
    point = LogTarget(make_model("lightgbm")).fit(lagd[EST_FEATURES], lagd["pm25"].values)
    qt = QuantileTrio(ALPHA).fit(lagd[EST_FEATURES], lagd["pm25"].values)
    per = {n: m for n, m in period_masks(d["date"]).items()}
    yc, loc, hic, _ = crossfit_calibration(d, per, EST_FEATURES)
    cqr = CQR(ALPHA).calibrate(yc, loc, hic)

    f = fc.rename(columns={"pm25": "pm25_today"})
    f = f[f["date"] <= cut].reset_index(drop=True)
    lagf = with_lag(f, f.rename(columns={"pm25_today": "pm25"}))
    fc_mon = LogTarget(make_model("lightgbm")).fit(lagf[FC_FEATURES_MON], lagf["pm25_next"].values)
    fc_unmon = LogTarget(make_model("lightgbm")).fit(lagf[FC_FEATURES_UNMON], lagf["pm25_next"].values)
    # intervalo D+1 (local sem estação) via CQR, com o mesmo esquema de cross-fitting
    qt_fc = QuantileTrio(ALPHA).fit(lagf[FC_FEATURES_UNMON], lagf["pm25_next"].values)
    perf = period_masks(f["date"])
    ycf, locf, hicf, _ = crossfit_calibration(f, perf, FC_FEATURES_UNMON, target="pm25_next", lag_col="pm25_today")
    cqr_fc = CQR(ALPHA).calibrate(ycf, locf, hicf)

    mdir = data_path("models", "x").parent
    joblib.dump({"point": point, "quantiles": qt, "cqr_q": cqr.q, "features": EST_FEATURES}, mdir / "estimativa.joblib")
    joblib.dump({"point_mon": fc_mon, "point_unmon": fc_unmon, "quantiles": qt_fc, "cqr_q": cqr_fc.q,
                 "features_mon": FC_FEATURES_MON, "features_unmon": FC_FEATURES_UNMON}, mdir / "previsao_d1.joblib")
    meta = {"treino_ate": str(cut.date()), "alpha": ALPHA, "cqr_q_estimativa": cqr.q, "cqr_q_d1": cqr_fc.q,
            "n_treino": int(len(d)), "estacoes": int(d["location_id"].nunique())}
    (mdir / "meta.json").write_text(json.dumps(meta, indent=2))
    print("modelos finais salvos:", meta)


def main():
    df = pd.read_parquet(data_path("processed", "station_days.parquet")).sort_values(["date", "location_id"])
    df = df.reset_index(drop=True)
    fc = pd.read_parquet(data_path("processed", "station_days_fc.parquet"))
    evaluate_intervals(df)
    forecast_eval(fc)
    fit_final(df, fc)


if __name__ == "__main__":
    main()
