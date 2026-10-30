"""Avaliação e explicação: figuras de métricas, SHAP e análise dos episódios extremos.

Lê os resultados de `benchmark.py` e `train.py`. Saídas em reports/figures e reports/metrics.
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

import matplotlib.pyplot as plt  # noqa: E402  (depois do plotstyle)

warnings.filterwarnings("ignore")
CFG = load_config()
FIG = lambda n: data_path("figures", n)  # noqa: E731
BEST = "LightGBM regularizado (offset IDW)"
BASELINES = {"IDW (baseline)", "Krigagem ordinária", "Climatologia mensal"}


def fig_benchmark():
    b = pd.read_csv(data_path("metrics", "benchmark_estimativa.csv"))
    b = b[~b["modelo"].str.contains("ALEATÓRIA")].sort_values("MAE", ascending=True)
    b = b[b["MAE"] < 3 * b["MAE"].min()]  # modelos que divergiram ficam só na tabela
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(b) + 1.2))
    colors = [ps.GREY if m in BASELINES else (ps.ORANGE if m == BEST else ps.BLUE) for m in b["modelo"]]
    ax.barh(b["modelo"], b["MAE"], color=colors, height=0.6, xerr=b["MAE_std_folds"],
            error_kw={"elinewidth": 1, "ecolor": ps.MUTED})
    for y, v in enumerate(b["MAE"]):
        ax.text(v + 0.15, y, f"{v:.2f}", va="center", color=ps.TEXT, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("MAE (µg/m³) - estações fora do treino, período posterior\n(traço = desvio entre folds)")
    ax.set_title("Estimativa diária de PM2.5: cinza = baselines, laranja = escolhido")
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
        ax.set_title(f"{col}\nMAE {m['MAE']:.2f} | RMSE {m['RMSE']:.2f} | R² {m['R2']:.2f}")
        ax.set_xlabel("PM2.5 medido (µg/m³)")
    axes[0].set_ylabel("PM2.5 estimado (µg/m³)")
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
        ax.text(i, cv + 0.01, f"{cv:.1%}\nlarg. {w:.1f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(tab)), ["quantis\n(sem calibrar)", "CQR", "conformal\n|resíduo|"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("cobertura real no teste")
    ax.legend(loc="lower right")
    ax.set_title("Cobertura do intervalo de 90% (estações não vistas)")
    # série temporal de uma estação de teste com o intervalo CQR
    st = r.groupby("name").size().idxmax()
    s = r[r["name"] == st].sort_values("date")
    ax = axes[1]
    ax.fill_between(s["date"], s["cqr_lo"], s["cqr_hi"], color=ps.BLUE, alpha=0.18, lw=0, label="IC 90% (CQR)")
    ax.plot(s["date"], s["pred"], color=ps.BLUE, lw=1.5, label="estimado")
    ax.plot(s["date"], s["pm25"], color=ps.TEXT, lw=1, marker="o", ms=2.5, label="medido")
    ax.set_title(f"Estação '{st}' (fora do treino)")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend(loc="upper right", ncols=3, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG("intervalos.png"), dpi=130)
    plt.close(fig)
    # cobertura condicional por faixa de concentração
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
    plt.title("SHAP: contribuição para a correção log(1+PM2.5) - log(1+IDW)")
    plt.tight_layout()
    plt.savefig(FIG("shap_resumo.png"), dpi=130)
    plt.close("all")
    # dependências físicas
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, f in zip(axes, ["aod_dry_blh", "blh_pm", "fire_up_400km"]):
        i = EST_FEATURES.index(f)
        ax.scatter(X[f], sv[:, i], s=6, alpha=0.4, color=ps.BLUE, edgecolor="none")
        ax.axhline(0, color=ps.MUTED, lw=0.8)
        ax.set_xlabel(f)
        ax.set_ylabel("SHAP")
        if X[f].max() > 50 * max(X[f].median(), 1e-3):
            ax.set_xscale("symlog")
    axes[0].set_title("AOD seco / camada limite")
    axes[1].set_title("Camada limite à tarde (m)")
    axes[2].set_title("Fogo a favor do vento (400 km)")
    fig.tight_layout()
    fig.savefig(FIG("shap_dependencia.png"), dpi=130)
    plt.close(fig)
    # explicação local: o dia/estação de maior PM2.5 previsto
    j = int(np.argmax(est["point"].predict(X)))
    contrib = pd.Series(sv[j], index=EST_FEATURES).sort_values(key=np.abs, ascending=False).head(10)
    row = test.iloc[j]
    contrib.to_frame("shap").assign(valor=X.iloc[j][contrib.index].values).to_csv(
        data_path("metrics", "shap_exemplo_local.csv"), float_format="%.4f")
    print(f"exemplo local: {row['name']} {row['date'].date()} medido={row['pm25']:.1f}")
    print(imp.head(12).to_string(index=False))


def episodes():
    """Episódios extremos: dias com fumaça (muito fogo a favor do vento) ou PM2.5 alto."""
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
    # piores dias
    d["err"] = d[BEST] - d["pm25"]
    worst = d.groupby("date").agg(pm25=("pm25", "mean"), pred=(BEST, "mean"), idw=("IDW (baseline)", "mean"),
                                  err=("err", "mean"),
                                  fogo=("fire_up_400km", "mean"), aod=("aod_filled", "mean"),
                                  sem_aod=("aod_fill_level", lambda s: (s > 0).mean()))
    worst.reindex(worst["err"].abs().sort_values(ascending=False).index).head(15).to_csv(
        data_path("metrics", "piores_dias.csv"), float_format="%.2f")
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True, height_ratios=[2.2, 1])
    t = worst.sort_index()
    ax.plot(t.index, t["pm25"], color=ps.TEXT, lw=1.2, label="medido (média das estações de teste)")
    ax.plot(t.index, t["pred"], color=ps.BLUE, lw=1.2, label="modelo (LightGBM offset IDW)")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("Período de teste: medido x estimado (estações fora do treino)")
    ax.legend(loc="upper right", fontsize=8)
    ax2.bar(t.index, t["fogo"], color=ps.ORANGE, width=1)
    ax2.set_ylabel("fogo a favor\ndo vento (índice)")
    fig.tight_layout()
    fig.savefig(FIG("episodios_serie.png"), dpi=130)
    plt.close(fig)


def band_metrics(y, yhat, limiar: float = 25.0) -> dict:
    """Faixa do IQAr além do acerto bruto: ~90% dos dias são "Boa", então acertar a faixa quase
    sempre é fácil. Acurácia balanceada e F1 macro dão o mesmo peso a cada faixa; a sensibilidade
    acima de `limiar` (fora da faixa Boa) mede se os dias poluídos são detectados."""
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
    """Acerto de faixa do IQAr da estimativa (mesmas linhas do benchmark) e da previsão D+1."""
    oof = pd.read_parquet(data_path("processed", "oof_estimativa.parquet"))
    cols = [c for c in oof.columns if c not in ("location_id", "name", "date", "lat", "lon", "pm25", "fold", "tfold",
                                                "dist_ref_km", "faixa_dist")]
    oof = oof[oof[cols].notna().all(axis=1)]
    # alerta pelo limite superior do IC 90%: regra fixa, sem limiar ajustado no teste. Só as colunas
    # de sensibilidade/precisão fazem sentido para ela.
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
    # matriz de confusão do modelo escolhido (linhas = faixa medida, colunas = faixa estimada)
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
    labels = [f"{c.split(')')[0]})\n{m}" for c, m in zip(t["cenario"], t["metodo"])]
    colors = [ps.GREY if "Persist" in m else ps.BLUE for m in t["metodo"]]
    ax.barh(labels, t["MAE"], color=colors, height=0.55)
    for y, v in enumerate(t["MAE"]):
        ax.text(v + 0.1, y, f"{v:.2f}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("MAE D+1 (µg/m³), período de teste")
    ax.set_title("Previsão do dia seguinte x persistência")
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
