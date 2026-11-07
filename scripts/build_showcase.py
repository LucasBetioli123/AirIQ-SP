"""Gera os dados do dashboard interativo (docs/data.js) e o mapa das estações (docs/estacoes.geojson).

Tudo sai dos CSVs de reports/metrics/ e da amostra versionada em data/sample/:
nenhum número é digitado à mão.

  python -m scripts.build_showcase
"""
import json

import pandas as pd

from src.common.config import ROOT

METRICS = ROOT / "reports" / "metrics"
DOCS = ROOT / "docs"

SHAP_LABELS = {
    "idw": "PM2.5 das estações vizinhas (IDW)",
    "built_5km": "Área construída (5 km)",
    "built": "Área construída (célula)",
    "tree_5km": "Cobertura arbórea (5 km)",
    "prev_tp": "Chuva do dia anterior",
    "crop_5km": "Agricultura (5 km)",
    "elev": "Altitude",
    "doy_cos": "Sazonalidade (dia do ano)",
    "grass_5km": "Pastagem (5 km)",
    "idw_dmin": "Distância à estação mais próxima",
    "no2_filled": "NO₂ troposférico (Sentinel-5P)",
    "blh_min": "Camada limite mínima",
    "no2_w9": "NO₂ regional (9×9)",
    "t2m": "Temperatura a 2 m",
}


def csv(name):
    return pd.read_csv(METRICS / name)


def records(df):
    return json.loads(df.to_json(orient="records", force_ascii=False))


def stations_and_series():
    d = pd.read_parquet(ROOT / "data/sample/station_days.parquet")
    d["date"] = pd.to_datetime(d["date"])
    stats = (d.groupby(["location_id", "name", "lat", "lon"])
               .agg(dias=("pm25", "size"), media=("pm25", "mean"), maximo=("pm25", "max"),
                    acima25=("pm25", lambda s: float((s > 25).mean())))
               .reset_index().sort_values("media", ascending=False))
    series = {}
    for (lid, name), g in d.sort_values("date").groupby(["location_id", "name"]):
        # dias sem medição viram null para a linha não ligar lacunas
        g = g.set_index("date")["pm25"].asfreq("D")
        series[name] = {"x": g.index.strftime("%Y-%m-%d").tolist(),
                        "y": [None if pd.isna(v) else round(float(v), 1) for v in g]}
    return stats, series


def geojson(stats):
    feats = []
    for r in stats.itertuples():
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(r.lon, 4), round(r.lat, 4)]},
            "properties": {
                "estação": r.name,
                "dias com PM2.5": int(r.dias),
                "PM2.5 médio (µg/m³)": round(r.media, 1),
                "máximo diário (µg/m³)": round(r.maximo, 1),
                "dias acima de 25 µg/m³": f"{r.acima25:.0%}",
                "marker-color": "#d95926" if r.media >= 15 else "#2a78d6",
                "marker-size": "medium" if r.acima25 >= 0.1 else "small",
            },
        })
    return {"type": "FeatureCollection", "features": feats}


def outline():
    """Contorno simplificado do estado (linhas separadas por None), para o mapa não depender de tiles externos."""
    from shapely import wkt
    geom = wkt.loads((ROOT / "data/sample/sp_boundary.wkt").read_text()).simplify(0.01)
    lon, lat = [], []
    for poly in getattr(geom, "geoms", [geom]):
        xs, ys = poly.exterior.coords.xy
        lon += [round(x, 3) for x in xs] + [None]
        lat += [round(y, 3) for y in ys] + [None]
    return {"lon": lon, "lat": lat}


def main():
    stats, series = stations_and_series()
    bench = csv("benchmark_estimativa.csv")
    bench = bench[~bench.modelo.str.contains("ALEATÓRIA")]
    recortes = csv("benchmark_estimativa_recortes.csv")
    recortes = recortes[recortes.recorte.eq("faixa_dist") &
                        recortes.modelo.isin(["IDW (baseline)", "Krigagem ordinária", "LightGBM regularizado (offset IDW)"])]
    fc = csv("previsao_d1.csv")
    fc = fc[fc.recorte.eq("todos")]
    rag = csv("rag_eval_resumo.csv")
    data = {
        "stations": records(stats.round(3)),
        "outline": outline(),
        "series": series,
        "benchmark": records(bench[["modelo", "MAE", "R2", "MAE_std_folds"]]),
        "distance": records(recortes[["grupo", "modelo", "n", "MAE"]]),
        "coverage": records(csv("intervalos_por_faixa.csv")),
        "intervals": records(csv("intervalos.csv")),
        "forecast": records(fc[["cenario", "metodo", "MAE", "RMSE", "R2"]]),
        "bands": records(csv("faixas_iqar.csv")),
        "confusion": records(csv("faixas_iqar_confusao.csv")),
        "retrieval": records(csv("rag_retrieval.csv")[["config", "recall@1", "recall@5", "hit@1", "MRR", "latencia_ms"]]),
        "rag": records(rag),
        "shap": [{"var": SHAP_LABELS.get(r.variavel, r.variavel), "valor": r.shap_medio_abs}
                 for r in csv("shap_importancia.csv").head(12).itertuples()],
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "data.js").write_text("window.DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    (DOCS / "estacoes.geojson").write_text(json.dumps(geojson(stats), ensure_ascii=False, indent=1), encoding="utf-8")
    print("docs/data.js e docs/estacoes.geojson atualizados")


if __name__ == "__main__":
    main()
