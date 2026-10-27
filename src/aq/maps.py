"""Mapa diário de PM2.5 do estado de SP e mapa da incerteza (largura do intervalo de 90%).

  python -m src.aq.maps --date 2022-10-15 [--horizon 0]
  python -m src.aq.maps --auto            # um dia típico e o dia mais poluído do teste
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402

from src.aq import plotstyle as ps  # noqa: E402
from src.aq.aqi import BANDS  # noqa: E402
from src.aq.data.geo import sp_boundary  # noqa: E402
from src.aq.predict import predict_points, station_obs  # noqa: E402
from src.common.config import data_path, load_config  # noqa: E402

CFG = load_config()


def sp_grid(res: float = CFG["region"]["grid_res_deg"]):
    lon_min, lat_min, lon_max, lat_max = CFG["region"]["bbox"]
    lons = np.arange(lon_min + res / 2, lon_max, res)
    lats = np.arange(lat_min + res / 2, lat_max, res)
    LO, LA = np.meshgrid(lons, lats)
    inside = shapely.contains_xy(sp_boundary(), LO, LA)
    return lats, lons, LA, LO, inside


def make_map(date: str, horizon: int = 0, out: str | None = None):
    lats, lons, LA, LO, inside = sp_grid()
    pr = predict_points([date] * int(inside.sum()), LA[inside], LO[inside], horizon)
    pm = np.full(LA.shape, np.nan)
    wid = np.full(LA.shape, np.nan)
    pm[inside] = pr["pm25"].values
    wid[inside] = (pr["ic90_sup"] - pr["ic90_inf"]).values
    target = pd.Timestamp(date) + pd.Timedelta(days=horizon)
    obs = station_obs()
    obs = obs[obs["date"] == target]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    edges = [0] + [b[1] for b in BANDS[:-1]] + [10000]
    ext = [lons[0], lons[-1], lats[0], lats[-1]]
    # magnitude contínua (um só matiz) + contornos nos limites das faixas do IQAr
    im = axes[0].imshow(pm, origin="lower", extent=ext, cmap=ps.SEQ_PM, vmin=0, vmax=max(40, np.nanpercentile(pm, 99)))
    axes[0].contour(LO, LA, np.nan_to_num(pm, nan=-1), levels=[25, 50], colors=[ps.BLUE, "#7d2a8a"], linewidths=1)
    plt.colorbar(im, ax=axes[0], label="PM2.5 (µg/m³)")
    if len(obs):
        axes[0].scatter(obs["lon"], obs["lat"], c=obs["pm25"], cmap=ps.SEQ_PM, vmin=im.norm.vmin, vmax=im.norm.vmax,
                        edgecolor="k", s=40, label="estações CETESB (medido)")
        axes[0].legend(loc="lower left", fontsize=8)
    tipo = "estimativa" if horizon == 0 else f"previsão D+{horizon} (feita em {date})"
    axes[0].set_title(f"PM2.5 {tipo} - {target.date()}\n(contornos: 25 e 50 µg/m³, limites Boa/Moderada/Ruim)")
    im2 = axes[1].imshow(wid, origin="lower", extent=ext, cmap=ps.SEQ_UNC)
    plt.colorbar(im2, ax=axes[1], label="largura do IC 90% (µg/m³)")
    st = obs[["lon", "lat"]].drop_duplicates() if len(obs) else station_obs()[["lon", "lat"]].drop_duplicates()
    axes[1].scatter(st["lon"], st["lat"], marker="^", c="cyan", edgecolor="k", s=30, label="estações")
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].set_title("Incerteza: largura do intervalo de 90% (CQR)")
    for ax in axes:
        b = sp_boundary()
        for g in getattr(b, "geoms", [b]):
            ax.plot(*g.exterior.xy, color="k", lw=0.6)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_aspect("equal")
    out = out or data_path("figures", f"mapa_pm25_{target.date()}_h{horizon}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    # tabela-resumo do mapa (fração da área por faixa)
    bands = pd.cut(pm[inside], bins=edges, labels=[b[2] for b in BANDS], include_lowest=True)
    summary = pd.Series(bands).value_counts(normalize=True).round(4).to_dict()
    print(f"mapa salvo em {out} | área por faixa: {summary} | largura média IC: {np.nanmean(wid):.1f}")
    return out


def auto_days():
    """Um dia típico (mediana) e o dia mais poluído do período de teste."""
    obs = station_obs()
    t = obs[obs["date"] >= CFG["split"]["test_start"]].groupby("date")["pm25"].mean()
    typical = (t - t.median()).abs().idxmin()
    worst = t.idxmax()
    return [str(typical.date()), str(worst.date())]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--horizon", type=int, default=0)
    ap.add_argument("--auto", action="store_true")
    a = ap.parse_args()
    days = auto_days() if (a.auto or not a.date) else [a.date]
    for d in days:
        make_map(d, a.horizon)
