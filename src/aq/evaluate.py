"""Evaluation and explanation: metric figures, SHAP and analysis of extreme episodes.

Reads the results of `benchmark.py` and `train.py`. Writes to reports/figures and reports/metrics.
CSV labels stay in Portuguese (data keys); figures are labeled in English.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.aq import plotstyle as ps
from src.aq.aqi import LABELS, band
from src.aq.modeling import EST_FEATURES, metrics, period_masks, with_lag
from src.aq.predict import models
from src.common.config import data_path, load_config

import matplotlib.pyplot as plt  # noqa: E402  (after plotstyle)

warnings.filterwarnings("ignore")
CFG = load_config()
FIG = lambda n: data_path("figures", n)  # noqa: E731
BEST = "LightGBM regularizado (offset IDW)"
BASELINES = {"IDW (baseline)", "Krigagem ordinária", "Climatologia mensal"}


def fig_benchmark():
    b = pd.read_csv(data_path("metrics", "benchmark_estimativa.csv"))
    b = b[~b["modelo"].str.contains("ALEATÓRIA")].sort_values("MAE", ascending=True)
    b = b[b["MAE"] < 3 * b["MAE"].min()]  # models that diverged stay in the table only
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(b) + 1.2))
    colors = [ps.GREY if m in BASELINES else (ps.ORANGE if m == BEST else ps.BLUE) for m in b["modelo"]]
    ax.barh([ps.en(m) for m in b["modelo"]], b["MAE"], color=colors, height=0.6, xerr=b["MAE_std_folds"],
            error_kw={"elinewidth": 1, "ecolor": ps.MUTED})
    for y, v in enumerate(b["MAE"]):
        ax.text(v + 0.15, y, f"{v:.2f}", va="center", color=ps.TEXT, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("MAE (µg/m³) on held-out stations, later period\n(whisker = std. dev. across folds)")
    ax.set_title("Daily PM2.5 estimate\ngray = baselines, orange = chosen model")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG("benchmark_mae.png"), dpi=130)
    plt.close(fig)


def fig_scatter():
    oof = pd.read_parquet(data_path("processed", "oof_estimativa.parquet"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    for ax, col, c in zip(axes, ["IDW (baseline)", BEST], [ps.GREY, ps.BLUE]):
        d = oof[["pm25", col]].dropna()
        m = metrics(d["pm25"], d[col])
        ax.scatter(d["pm25"], d[col], s=8, alpha=0.35, color=c, edgecolor="none")
        lim = np.nanpercentile(d.values, 99.5)
        ax.plot([0, lim], [0, lim], color=ps.MUTED, lw=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(f"{ps.en(col)}\nMAE {m['MAE']:.2f} | RMSE {m['RMSE']:.2f} | R² {m['R2']:.2f}")
        ax.set_xlabel("measured PM2.5 (µg/m³)")
    axes[0].set_ylabel("estimated PM2.5 (µg/m³)")
    fig.tight_layout()
    fig.savefig(FIG("dispersao_idw_vs_lgbm.png"), dpi=130)
    plt.close(fig)


def fig_intervals():
    r = pd.read_parquet(data_path("processed", "oof_intervalos.parquet"))
    tab = pd.read_csv(data_path("metrics", "intervalos.csv"))
    tab = tab[tab["recorte"] == "todos"].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.bar(range(len(tab)), tab["cobertura_real"], color=[ps.GREY, ps.BLUE, ps.AQUA], width=0.6)
    ax.axhline(1 - CFG["split"]["interval_alpha"], color=ps.ORANGE, lw=1.5, ls="--", label="nominal 90%")
    for i, (cv, w) in enumerate(zip(tab["cobertura_real"], tab["largura_media"])):
        ax.text(i, cv + 0.01, f"{cv:.1%}\nwidth {w:.1f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(tab)), ["quantiles\n(uncalibrated)", "CQR", "conformal\n|residual|"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("real coverage on test")
    ax.legend(loc="lower right")
    ax.set_title("Coverage of the 90% interval (unseen stations)")
    # time series of one test station with the CQR interval
    st = r.groupby("name").size().idxmax()
    s = r[r["name"] == st].sort_values("date")
    ax = axes[1]
    ax.fill_between(s["date"], s["cqr_lo"], s["cqr_hi"], color=ps.BLUE, alpha=0.18, lw=0, label="90% interval (CQR)")
    ax.plot(s["date"], s["pred"], color=ps.BLUE, lw=1.5, label="estimated")
    ax.plot(s["date"], s["pm25"], color=ps.TEXT, lw=1, marker="o", ms=2.5, label="measured")
    ax.set_title(f"Station '{st}' (held out of training)")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend(loc="upper right", ncols=3, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG("intervalos.png"), dpi=130)
    plt.close(fig)
    # conditional coverage by concentration band
    r["faixa"] = pd.cut(r["pm25"], [0, 15, 25, 35, 50, 500], labels=["0-15", "15-25", "25-35", "35-50", ">50"])
    cond = r.groupby("faixa").apply(lambda g: pd.Series({
        "n": len(g), "cobertura_cqr": float(np.mean((g.pm25 >= g.cqr_lo) & (g.pm25 <= g.cqr_hi))),
        "largura_cqr": float(np.mean(g.cqr_hi - g.cqr_lo))}))
    cond.to_csv(data_path("metrics", "intervalos_por_faixa.csv"), float_format="%.4f")


def shap_analysis(n: int = 3000):
    import shap

    est, _ = models()
    df = pd.read_parquet(data_path("processed", "station_days.parquet"))
    test = with_lag(df, df).sample(min(n, len(df)), random_state=CFG["seed"])
    X = test[EST_FEATURES]
    booster = est["point"].model
    sv = shap.TreeExplainer(booster).shap_values(X)
    imp = pd.DataFrame({"variavel": EST_FEATURES, "shap_medio_abs": np.abs(sv).mean(0)}) \
        .sort_values("shap_medio_abs", ascending=False)
    imp.to_csv(data_path("metrics", "shap_importancia.csv"), index=False, float_format="%.5f")
    plt.figure()
    shap.summary_plot(sv, X, max_display=18, show=False, plot_size=(8, 6.5))
    plt.title("SHAP: contribution to the correction log(1+PM2.5) - log(1+IDW)")
    plt.tight_layout()
    plt.savefig(FIG("shap_resumo.png"), dpi=130)
    plt.close("all")
    # physical dependencies
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, f in zip(axes, ["aod_dry_blh", "blh_pm", "fire_up_400km"]):
        i = EST_FEATURES.index(f)
        ax.scatter(X[f], sv[:, i], s=6, alpha=0.4, color=ps.BLUE, edgecolor="none")
        ax.axhline(0, color=ps.MUTED, lw=0.8)
        ax.set_xlabel(f)
        ax.set_ylabel("SHAP")
        if X[f].max() > 50 * max(X[f].median(), 1e-3):
            ax.set_xscale("symlog")
    axes[0].set_title("Dry AOD / boundary layer")
    axes[1].set_title("Afternoon boundary layer (m)")
    axes[2].set_title("Upwind fire (400 km)")
    fig.tight_layout()
    fig.savefig(FIG("shap_dependencia.png"), dpi=130)
    plt.close(fig)
    # local explanation: the station-day with the highest predicted PM2.5
    j = int(np.argmax(est["point"].predict(X)))
    contrib = pd.Series(sv[j], index=EST_FEATURES).sort_values(key=np.abs, ascending=False).head(10)
    row = test.iloc[j]
    contrib.to_frame("shap").assign(valor=X.iloc[j][contrib.index].values).to_csv(
        data_path("metrics", "shap_exemplo_local.csv"), float_format="%.4f")
    print(f"local example: {row['name']} {row['date'].date()} measured={row['pm25']:.1f}")
    print(imp.head(12).to_string(index=False))


def episodes():
    """Extreme episodes: smoke days (strong upwind fire) or high PM2.5."""
    oof = pd.read_parquet(data_path("processed", "oof_estimativa.parquet"))
    df = pd.read_parquet(data_path("processed", "station_days.parquet"),
                         columns=["location_id", "date", "fire_up_400km", "fire_n_400km", "aod_filled",
                                  "aod_fill_level", "blh_pm", "rh", "tp"])
    d = oof.merge(df, on=["location_id", "date"]).dropna(subset=["IDW (baseline)", BEST])
    p95_fire = d["fire_up_400km"].quantile(0.95)
    p95_pm = d["pm25"].quantile(0.95)
    d["regime"] = np.select([d["fire_up_400km"] >= p95_fire, d["pm25"] >= p95_pm],
                            ["fumaça (fogo a favor do vento ≥ p95)", "PM2.5 alto sem fogo (≥ p95)"], "normal")
    rows = []
    for reg, g in d.groupby("regime"):
        for col in ["IDW (baseline)", BEST]:
            m = metrics(g["pm25"], g[col])
            rows.append({"regime": reg, "modelo": col, "n": m["n"], "PM2.5_medio": g["pm25"].mean(), "MAE": m["MAE"],
                         "vies": m["bias"], "AOD_medio": g["aod_filled"].mean(),
                         "frac_sem_AOD_obs": float((g["aod_fill_level"] > 0).mean())})
    tab = pd.DataFrame(rows)
    tab.to_csv(data_path("metrics", "episodios_extremos.csv"), index=False, float_format="%.3f")
    print(tab.to_string(index=False))
    # worst days
    d["err"] = d[BEST] - d["pm25"]
    worst = d.groupby("date").agg(pm25=("pm25", "mean"), pred=(BEST, "mean"), idw=("IDW (baseline)", "mean"),
                                  err=("err", "mean"),
                                  fogo=("fire_up_400km", "mean"), aod=("aod_filled", "mean"),
                                  sem_aod=("aod_fill_level", lambda s: (s > 0).mean()))
    worst.reindex(worst["err"].abs().sort_values(ascending=False).index).head(15).to_csv(
        data_path("metrics", "piores_dias.csv"), float_format="%.2f")
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True, height_ratios=[2.2, 1])
    t = worst.sort_index()
    ax.plot(t.index, t["pm25"], color=ps.TEXT, lw=1.2, label="measured (mean of test stations)")
    ax.plot(t.index, t["pred"], color=ps.BLUE, lw=1.2, label="model (LightGBM, IDW offset)")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("Test period: measured vs. estimated (held-out stations)")
    ax.legend(loc="upper right", fontsize=8)
    ax2.bar(t.index, t["fogo"], color=ps.ORANGE, width=1)
    ax2.set_ylabel("upwind fire\n(index)")
    fig.tight_layout()
    fig.savefig(FIG("episodios_serie.png"), dpi=130)
    plt.close(fig)


def band_metrics(y, yhat, limiar: float = 25.0) -> dict:
    """IQAr band metrics beyond raw accuracy: ~90% of days are "Boa" (good), so getting the band
    right is almost always easy. Balanced accuracy and macro F1 weight every band equally; recall
    above `limiar` (outside the good band) measures whether polluted days are caught."""
    from sklearn.metrics import balanced_accuracy_score, f1_score

    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    yb, pb = band(y), band(yhat)
    alto, prev_alto = y > limiar, yhat > limiar
    return {"n": len(y), "acerto_faixa": float(np.mean(yb == pb)), "acerto_sempre_Boa": float(np.mean(yb == "Boa")),
            "acuracia_balanceada": balanced_accuracy_score(yb, pb), "f1_macro": f1_score(yb, pb, average="macro"),
            f"n_obs_>{limiar:g}": int(alto.sum()),
            f"sensibilidade_>{limiar:g}": float(np.mean(prev_alto[alto])) if alto.any() else np.nan,
            f"precisao_>{limiar:g}": float(np.mean(alto[prev_alto])) if prev_alto.any() else np.nan}


def band_analysis():
    """IQAr band accuracy for the estimate (same rows as the benchmark) and for the D+1 forecast."""
    oof = pd.read_parquet(data_path("processed", "oof_estimativa.parquet"))
    cols = [c for c in oof.columns if c not in ("location_id", "name", "date", "lat", "lon", "pm25", "fold", "tfold",
                                                "dist_ref_km", "faixa_dist")]
    oof = oof[oof[cols].notna().all(axis=1)]
    # alert on the upper bound of the 90% interval: a fixed rule, no threshold tuned on the test
    # set. Only the recall/precision columns make sense for it.
    alerta = "Alerta: limite superior do IC 90% > 25"
    only_alert = lambda y, hi: {k: (v if k.startswith(("n", "sens", "prec")) else np.nan)  # noqa: E731
                                for k, v in band_metrics(y, hi).items()}
    rows = [{"tarefa": "estimativa D", "metodo": c, **band_metrics(oof["pm25"], oof[c])}
            for c in ["IDW (baseline)", "Krigagem ordinária", BEST]]
    iv = pd.read_parquet(data_path("processed", "oof_intervalos.parquet"))
    rows.append({"tarefa": "estimativa D", "metodo": alerta, **only_alert(iv["pm25"], iv["cqr_hi"])})
    fc = pd.read_parquet(data_path("processed", "oof_previsao_d1.parquet"))
    for cen, g in fc.groupby("cenario"):
        g = g.dropna(subset=["pm25", "pred", "pers"])
        rows += [{"tarefa": f"D+1 {cen}", "metodo": "Persistência", **band_metrics(g["pm25"], g["pers"])},
                 {"tarefa": f"D+1 {cen}", "metodo": "LightGBM reg. (offset IDW) D+1", **band_metrics(g["pm25"], g["pred"])},
                 {"tarefa": f"D+1 {cen}", "metodo": alerta, **only_alert(g["pm25"], g["cqr_hi"])}]
    tab = pd.DataFrame(rows)
    tab.to_csv(data_path("metrics", "faixas_iqar.csv"), index=False, float_format="%.4f")
    # confusion matrix of the chosen model (rows = measured band, columns = estimated band)
    conf = pd.crosstab(pd.Series(band(oof["pm25"].values), name="medida"),
                       pd.Series(band(oof[BEST].values), name="estimada")).reindex(index=LABELS, columns=LABELS,
                                                                                  fill_value=0)
    conf.to_csv(data_path("metrics", "faixas_iqar_confusao.csv"))
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(conf)


def fig_forecast():
    t = pd.read_csv(data_path("metrics", "previsao_d1.csv"))
    t = t[t["recorte"] == "todos"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    scen = {"A": "A) site with station", "B": "B) site without station"}
    labels = [f"{scen[c[0]]}\n{ps.en(m)}" for c, m in zip(t["cenario"], t["metodo"])]
    colors = [ps.GREY if "Persist" in m else ps.BLUE for m in t["metodo"]]
    ax.barh(labels, t["MAE"], color=colors, height=0.55)
    for y, v in enumerate(t["MAE"]):
        ax.text(v + 0.1, y, f"{v:.2f}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("D+1 MAE (µg/m³), test period")
    ax.set_title("Next-day forecast vs. persistence")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG("previsao_d1.png"), dpi=130)
    plt.close(fig)


def main():
    fig_benchmark()
    fig_scatter()
    fig_intervals()
    fig_forecast()
    shap_analysis()
    episodes()
    band_analysis()


if __name__ == "__main__":
    main()
