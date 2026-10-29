"""Treino do modelo escolhido no benchmark, com intervalo de confiança e previsão D+1.

Modelo escolhido (reports/metrics/benchmark_estimativa.csv): LightGBM regularizado, com
alvo relativo ao IDW do dia, log1p(PM) - log1p(IDW).

Os folds servem SÓ para medir métricas fora da amostra. Os modelos de produção (mapa,
CLI, API, agente) são treinados com TODOS os dados: todas as estações e todo o período.

1) INTERVALOS na validação espaço-temporal (cortes T1/T2 x folds de estação), comparando:
   a) quantis do LightGBM (5%/95%) sem calibração
   b) CQR: os mesmos quantis conformalizados. Os escores de calibração vêm de
      cross-fitting por grupos de estação no período de calibração (o modelo interno não
      viu a estação), o que imita o erro num local sem monitor
   c) conformal do resíduo absoluto em torno da previsão pontual (mesmo cross-fitting)
2) PREVISÃO D+1, comparada com a persistência (amanhã = hoje):
   A) local COM estação: o modelo conhece o PM2.5 de hoje da própria estação (offset = PM de hoje)
   B) local SEM estação: estações de teste fora do treino; "persistência" = IDW de hoje
3) MODELOS FINAIS com todos os dados. A calibração do CQR de produção usa previsões
   fora da amostra por grupo de estação sobre o período inteiro.
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd

from src.aq.modeling import (CQR, EST_FEATURES, FC_EXTRA, FEATURE_SCHEMA, SPLIT, TEMPORAL_FOLDS, OffsetLogTarget, QuantileTrio,
                             make_model, metrics, period_masks, station_folds, with_lag)
from src.common.config import data_path

warnings.filterwarnings("ignore")
ALPHA = SPLIT["interval_alpha"]
POINT_MODEL = "lightgbm_reg"
POINT_LABEL = "LightGBM reg. (offset IDW)"
FC_FEATURES_UNMON = EST_FEATURES + FC_EXTRA
FC_FEATURES_MON = FC_FEATURES_UNMON + ["pm25_today"]


def point_model(offset_col: str = "idw"):
    return OffsetLogTarget(make_model(POINT_MODEL), offset_col=offset_col)


def quantile_model(offset_col: str = "idw"):
    return QuantileTrio(ALPHA, offset_col=offset_col, base=POINT_MODEL)


# ----------------------------------------------------------------------------- calibração
def crossfit_scores(pool: pd.DataFrame, feats: list[str], fit_mask=None, pred_mask=None, k_inner: int = 4,
                    target: str = "pm25", lag_col: str = "pm25", offset_col: str = "idw"):
    """Previsões 'fora da estação': para cada grupo de estações, treina sem ele (linhas
    `fit_mask`) e prevê nele (linhas `pred_mask`). `lag_col` é a medição usada no IDW
    (na previsão D+1 é a de HOJE, nunca o alvo). Devolve (y, lo_log, hi_log, yhat)."""
    fit_mask = pd.Series(True, index=pool.index) if fit_mask is None else fit_mask
    pred_mask = pd.Series(True, index=pool.index) if pred_mask is None else pred_mask
    inner = station_folds(pool["location_id"].values, k_inner)
    g = pool["location_id"].map(inner)
    ys, los, his, pts = [], [], [], []
    for j in range(k_inner):
        ref = pool[g != j]
        tr = with_lag(ref[fit_mask[ref.index]], ref, value_col=lag_col)
        rows = pool[(g == j) & pred_mask]
        if rows.empty or tr.empty:
            continue
        ca = with_lag(rows, ref, value_col=lag_col)
        lo, _, hi = quantile_model(offset_col).fit(tr[feats], tr[target].values).predict_log(ca[feats])
        pts.append(point_model(offset_col).fit(tr[feats], tr[target].values).predict(ca[feats]))
        ys.append(ca[target].values)
        los.append(lo)
        his.append(hi)
    return np.concatenate(ys), np.concatenate(los), np.concatenate(his), np.concatenate(pts)


def abs_conformal_q(y, yhat, alpha=ALPHA):
    s = np.abs(np.asarray(y) - np.asarray(yhat))
    n = len(s)
    return float(np.quantile(s, min(1, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


# ----------------------------------------------------------------------------- 1) intervalos
def _interval_fold(df, per, k, feats, tname) -> pd.DataFrame:
    pool = df[df["fold"] != k]
    per_pool = {n: m[pool.index] for n, m in per.items()}
    tr = with_lag(pool[per_pool["train"]], pool)
    te = with_lag(df[(df["fold"] == k) & per["test"]], pool)
    lo, _, hi = quantile_model().fit(tr[feats], tr["pm25"].values).predict_log(te[feats])
    point = point_model().fit(tr[feats], tr["pm25"].values).predict(te[feats])
    yc, loc, hic, ptc = crossfit_scores(pool, feats, per_pool["train"], per_pool["calib"])
    cqr = CQR(ALPHA).calibrate(yc, loc, hic)
    clo, chi = cqr.interval(lo, hi)
    qa = abs_conformal_q(yc, ptc)
    r = te[["location_id", "name", "date", "lat", "lon", "pm25", "fold"]].copy()
    r["tfold"] = tname
    r["pred"] = point
    r["q_lo"], r["q_hi"] = np.expm1(lo).clip(0), np.expm1(hi)
    r["cqr_lo"], r["cqr_hi"] = np.minimum(clo, point), np.maximum(chi, point)
    r["abs_lo"], r["abs_hi"] = (point - qa).clip(0), point + qa
    r["cqr_q"] = cqr.q
    print(f"{tname} fold {k}: CQR q={cqr.q:.3f} (log) | conformal |res| q={qa:.1f} µg/m³", flush=True)
    return r


def _cov(g, lo, hi, mask=None):
    inside = (g.pm25 >= g[lo]) & (g.pm25 <= g[hi])
    return float(np.mean(inside if mask is None else inside[mask]))


def evaluate_intervals(df: pd.DataFrame) -> pd.DataFrame:
    folds = station_folds(df["location_id"].values)
    df = df.assign(fold=df["location_id"].map(folds))
    out = []
    for tf in TEMPORAL_FOLDS:
        per = period_masks(df["date"], tf)
        for k in sorted(df["fold"].unique()):
            out.append(_interval_fold(df, per, k, EST_FEATURES, tf["name"]))
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(data_path("processed", "oof_intervalos.parquet"), index=False)
    tab = []
    groups = [("todos", res)] + list(res.groupby("tfold"))
    for gname, g in groups:
        for name, lo, hi in [("Quantis LightGBM 5-95% (sem calibrar)", "q_lo", "q_hi"),
                             ("CQR (quantis conformalizados)", "cqr_lo", "cqr_hi"),
                             ("Conformal do resíduo absoluto", "abs_lo", "abs_hi")]:
            tab.append({"recorte": gname, "metodo": name, "n": len(g), "cobertura_nominal": 1 - ALPHA,
                        "cobertura_real": _cov(g, lo, hi), "largura_media": float(np.mean(g[hi] - g[lo])),
                        "cobertura_pm25_>25": _cov(g, lo, hi, g.pm25 > 25)})
    tab = pd.DataFrame(tab)
    tab.to_csv(data_path("metrics", "intervalos.csv"), index=False, float_format="%.4f")
    print(tab.to_string(index=False))
    return tab


# ----------------------------------------------------------------------------- 2) previsão D+1
def forecast_eval(fc: pd.DataFrame) -> pd.DataFrame:
    fc = fc.rename(columns={"pm25": "pm25_today"}).sort_values(["date", "location_id"]).reset_index(drop=True)
    refs_all = fc.rename(columns={"pm25_today": "pm25"})  # medições de HOJE (D) para o lag
    folds = station_folds(fc["location_id"].values)
    fc["fold"] = fc["location_id"].map(folds)
    lag_all = with_lag(fc, refs_all)
    results = []

    def evaluate(tr, ca, te, feats, offset, scenario, tfold, fold, scores=None):
        point = point_model(offset).fit(tr[feats], tr["pm25_next"].values).predict(te[feats])
        qt = quantile_model(offset).fit(tr[feats], tr["pm25_next"].values)
        lo, _, hi = qt.predict_log(te[feats])
        if scores is None:
            loc, _, hic = qt.predict_log(ca[feats])
            scores = ca["pm25_next"].values, loc, hic
        cqr = CQR(ALPHA).calibrate(*scores)
        lower, upper = cqr.interval(lo, hi)
        r = te[["location_id", "date", "lat", "lon", "pm25_next"]].copy()
        r = r.rename(columns={"pm25_next": "pm25"})
        r["data_alvo"] = r["date"] + pd.Timedelta(days=1)
        r["cenario"], r["tfold"], r["fold"] = scenario, tfold, fold
        r["pred"], r["pers"] = point, te[offset].values
        r["cqr_lo"], r["cqr_hi"] = np.minimum(lower, point), np.maximum(upper, point)
        results.append(r)

    for tf in TEMPORAL_FOLDS:
        # O alvo D+1, inclusive nos limites de treino/calibração, define o split.
        per = period_masks(fc["date"] + pd.Timedelta(days=1), tf)
        # A) local com estação: split só temporal (todas as estações), offset = PM2.5 de hoje
        tr, ca, te = (lag_all[per[p]] for p in ("train", "calib", "test"))
        evaluate(tr, ca, te, FC_FEATURES_MON, "pm25_today", "A) local com estação", tf["name"], -1)
        # B) local sem estação: folds de estação; "persistência" = IDW de hoje
        for k in sorted(fc["fold"].unique()):
            pool = refs_all[fc["fold"] != k]
            trk = with_lag(fc[(fc["fold"] != k) & per["train"]], pool)
            tek = with_lag(fc[(fc["fold"] == k) & per["test"]], pool)
            fp = fc[fc["fold"] != k]
            yc, loc, hic, _ = crossfit_scores(fp, FC_FEATURES_UNMON, per["train"], per["calib"],
                                             target="pm25_next", lag_col="pm25_today")
            evaluate(trk, None, tek, FC_FEATURES_UNMON, "idw", "B) local sem estação", tf["name"], k,
                     (yc, loc, hic))
            print(f"D+1 {tf['name']} fold {k}: ponto e intervalo avaliados", flush=True)
    oof = pd.concat(results, ignore_index=True)
    oof.to_parquet(data_path("processed", "oof_previsao_d1.parquet"), index=False)
    rows = []
    intervals = []
    for cen, g in oof.groupby("cenario"):
        g = g[np.isfinite(g["pers"])]
        pname = "Persistência (amanhã = hoje)" if cen.startswith("A") else "Persistência do IDW de hoje"
        for recorte, s in [("todos", g)] + list(g.groupby("tfold")):
            y = s["pm25"].values
            rows.append({"cenario": cen, "recorte": recorte, "metodo": pname, **metrics(y, s["pers"])})
            rows.append({"cenario": cen, "recorte": recorte, "metodo": f"{POINT_LABEL} D+1",
                         **metrics(y, s["pred"], s["cqr_lo"], s["cqr_hi"], baseline=s["pers"])})
            intervals.append({"cenario": cen, "recorte": recorte, "n": len(s), "cobertura_nominal": 1 - ALPHA,
                              "cobertura_real": _cov(s, "cqr_lo", "cqr_hi"),
                              "largura_media": float((s.cqr_hi - s.cqr_lo).mean()),
                              "cobertura_pm25_>25": _cov(s, "cqr_lo", "cqr_hi", s.pm25 > 25)})
    pd.DataFrame(intervals).to_csv(data_path("metrics", "intervalos_d1.csv"), index=False, float_format="%.4f")
    tab = pd.DataFrame(rows)
    tab.to_csv(data_path("metrics", "previsao_d1.csv"), index=False, float_format="%.4f")
    print(tab[["cenario", "recorte", "metodo", "n", "MAE", "RMSE", "R2", "acerto_faixa_IQAr"]].to_string(index=False))
    return tab


# ----------------------------------------------------------------------------- 3) modelos finais
def fit_final(df: pd.DataFrame, fc: pd.DataFrame):
    """Produção: TODOS os dados (todas as estações, todo o período)."""
    d = df.reset_index(drop=True)
    lagd = with_lag(d, d)
    point = point_model().fit(lagd[EST_FEATURES], lagd["pm25"].values)
    qt = quantile_model().fit(lagd[EST_FEATURES], lagd["pm25"].values)
    yc, loc, hic, _ = crossfit_scores(d, EST_FEATURES, k_inner=5)
    cqr = CQR(ALPHA).calibrate(yc, loc, hic)

    f = fc.rename(columns={"pm25": "pm25_today"}).reset_index(drop=True)
    lagf = with_lag(f, f.rename(columns={"pm25_today": "pm25"}))
    fc_mon = point_model("pm25_today").fit(lagf[FC_FEATURES_MON], lagf["pm25_next"].values)
    fc_unmon = point_model("idw").fit(lagf[FC_FEATURES_UNMON], lagf["pm25_next"].values)
    qt_fc = quantile_model("idw").fit(lagf[FC_FEATURES_UNMON], lagf["pm25_next"].values)
    qt_fc_mon = quantile_model("pm25_today").fit(lagf[FC_FEATURES_MON], lagf["pm25_next"].values)
    ycf, locf, hicf, _ = crossfit_scores(f, FC_FEATURES_UNMON, k_inner=5, target="pm25_next", lag_col="pm25_today")
    cqr_fc = CQR(ALPHA).calibrate(ycf, locf, hicf)
    ycm, lom, him, _ = crossfit_scores(f, FC_FEATURES_MON, k_inner=5, target="pm25_next",
                                      lag_col="pm25_today", offset_col="pm25_today")
    cqr_fc_mon = CQR(ALPHA).calibrate(ycm, lom, him)

    mdir = data_path("models")
    joblib.dump({"point": point, "quantiles": qt, "cqr_q": cqr.q, "features": EST_FEATURES, "modelo": POINT_LABEL,
                 "feature_schema": FEATURE_SCHEMA},
                mdir / "estimativa.joblib")
    joblib.dump({"point_mon": fc_mon, "point_unmon": fc_unmon, "quantiles": qt_fc, "cqr_q": cqr_fc.q,
                 "quantiles_mon": qt_fc_mon, "cqr_q_mon": cqr_fc_mon.q, "feature_schema": FEATURE_SCHEMA,
                 "features_mon": FC_FEATURES_MON, "features_unmon": FC_FEATURES_UNMON, "modelo": POINT_LABEL},
                mdir / "previsao_d1.joblib")
    meta = {"modelo": POINT_LABEL, "feature_schema": FEATURE_SCHEMA,
            "dados_de_treino": f"{d['date'].min().date()} a {d['date'].max().date()}",
            "alpha": ALPHA, "cqr_q_estimativa": cqr.q, "cqr_q_d1": cqr_fc.q,
            "n_treino": int(len(d)), "estacoes": int(d["location_id"].nunique())}
    (mdir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("modelos finais salvos:", meta)


def main():
    df = pd.read_parquet(data_path("processed", "station_days.parquet")).sort_values(["date", "location_id"])
    df = df.reset_index(drop=True)
    fc = pd.read_parquet(data_path("processed", "station_days_fc.parquet"))
    for table in (df, fc):
        if "feature_schema" not in table or not table["feature_schema"].eq(FEATURE_SCHEMA).all():
            raise ValueError("base antiga: rode `make sample` ou `make data` antes do treino")
    evaluate_intervals(df)
    forecast_eval(fc)
    fit_final(df, fc)


if __name__ == "__main__":
    main()
