"""Precompute the "any city" map for the dashboard (docs/maps/).

For every day in the sample windows (configs/config.yaml > sample.context_windows) this runs
the trained model (same inference as the CLI/API/maps), both as a same-day estimate and as the
next-day (D+1) forecast issued that day, on:
  - the 0.05° grid covering São Paulo state  -> docs/maps/<date>.json
  - the 645 municipal seats                 -> docs/maps/cities.json
plus the stations' measured values on the day and on the next day (to check the forecast), so
the static page can show estimates anywhere without a server. Requires trained models
(`python -m src.aq.pipeline --quick`).

  python -m scripts.build_city_maps
"""
import json

import numpy as np
import pandas as pd

from src.aq.maps import sp_grid
from src.aq.predict import predict_points, station_obs
from src.common.config import ROOT, load_config

OUT = ROOT / "docs" / "maps"


def window_days() -> list[pd.Timestamp]:
    days = []
    for start, end in load_config()["sample"]["context_windows"]:
        days += list(pd.date_range(start, end, freq="D"))
    return days


def tenths(values) -> list:
    """Compact JSON: values x10 as ints (None where missing)."""
    return [None if not np.isfinite(v) else int(round(v * 10)) for v in np.asarray(values, float)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    days = window_days()
    lats, lons, LA, LO, inside = sp_grid()
    idx = np.flatnonzero(inside.ravel())
    cities = pd.read_csv(ROOT / "data/sample/municipios_sp.csv")
    obs = station_obs()
    stations = obs[["name", "lat", "lon"]].drop_duplicates("name")
    city_rows = {c: {k: [] for k in ("pm", "lo", "hi", "fpm", "flo", "fhi")} for c in cities["nome"]}
    dist = None
    for day in days:
        d = str(day.date())
        grid = predict_points([d] * len(idx), LA.ravel()[idx], LO.ravel()[idx], 0)
        grid_fc = predict_points([d] * len(idx), LA.ravel()[idx], LO.ravel()[idx], 1)
        pts = predict_points([d] * len(cities), cities["lat"], cities["lon"], 0)
        fc = predict_points([d] * len(cities), cities["lat"], cities["lon"], 1)
        dist = pts["dist_estacao_mais_proxima_km"].round(1).tolist()
        for i, name in enumerate(cities["nome"]):
            row = city_rows[name]
            for key, frame, col in (("pm", pts, "pm25"), ("lo", pts, "ic90_inf"), ("hi", pts, "ic90_sup"),
                                    ("fpm", fc, "pm25"), ("flo", fc, "ic90_inf"), ("fhi", fc, "ic90_sup")):
                row[key].append(round(float(frame[col].iloc[i]), 1))

        def measured(when):
            m = obs[obs["date"] == when][["name", "pm25"]]
            return {n: round(float(v), 1) for n, v in zip(m["name"], m["pm25"])}
        (OUT / f"{d}.json").write_text(json.dumps({
            "date": d, "grid": tenths(grid["pm25"]), "grid_fc": tenths(grid_fc["pm25"]),
            "obs": measured(day), "obs_next": measured(day + pd.Timedelta(days=1)),
        }, separators=(",", ":")), encoding="utf-8")
        print(d, f"grid mean {grid['pm25'].mean():.1f}", f"max {grid['pm25'].max():.1f}", flush=True)
    meta = {
        "dates": [str(d.date()) for d in days],
        "windows": load_config()["sample"]["context_windows"],
        "lats": [round(float(v), 3) for v in lats], "lons": [round(float(v), 3) for v in lons],
        "inside": idx.tolist(),
        "stations": [{"name": r.name, "lat": round(r.lat, 4), "lon": round(r.lon, 4)} for r in stations.itertuples()],
        "cities": [{"name": n, "lat": round(float(la), 4), "lon": round(float(lo), 4), "dist_km": dk, **city_rows[n]}
                   for n, la, lo, dk in zip(cities["nome"], cities["lat"], cities["lon"], dist)],
    }
    (OUT / "cities.json").write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(days)} days, {len(idx)} grid cells, {len(cities)} cities -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
