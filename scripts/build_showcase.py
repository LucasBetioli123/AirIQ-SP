"""Build the interactive dashboard data (docs/data.js) and the station map (docs/stations.geojson).

Everything comes from the CSVs in reports/metrics/ and the versioned sample in data/sample/:
no number is typed by hand.

  python -m scripts.build_showcase
"""
import json

import pandas as pd
import yaml

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
        # days without a measurement become null so the line does not bridge gaps
        g = g.set_index("date")["pm25"].asfreq("D")
        series[name] = {"x": g.index.strftime("%Y-%m-%d").tolist(),
                        "y": [None if pd.isna(v) else round(float(v), 1) for v in g]}
    return stats, series


def agent_runs(tag="qwen3-4b_causal"):
    """Recorded agent runs for the dashboard's "Ask the agent" section (static page: no live LLM).

    Questions, answers and legal passages stay in Portuguese: the knowledge base is Brazilian law."""
    qs = {q["pergunta"]: q for q in yaml.safe_load((ROOT / "eval/rag_questions.yaml").read_text(encoding="utf-8"))["perguntas"]}
    detail = pd.read_csv(METRICS / f"rag_eval_detalhe_{tag}.csv").set_index("id")
    chunks = {}
    for line in (ROOT / "data/sample/rag_chunks.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            chunks[c["id"]] = c
    runs = {}
    for line in (ROOT / f"logs/eval_runs_{tag}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        runs[r["pergunta"]] = r  # latest run of each question
    out = []
    for text, q in qs.items():
        r = runs.get(text)
        if r is None:
            continue
        row = detail.loc[q["id"]]
        ok = row["recusa_correta"] if q["tipo"] == "sem_resposta" else row["acerto"]
        cited = [t for t in r["trechos"] if t["n"] in (r.get("citacoes") or [])]
        tools = []
        for t in r["ferramentas"]:
            res = t["resultado"]
            if t["ferramenta"] == "buscar_normas":
                res = f"{len(res)} passages"
            elif isinstance(res, dict):
                res = {k: res[k] for k in ("data_alvo", "pm25_ug_m3", "intervalo_90", "faixa_iqar", "padrao_24h_vigente",
                                           "etapa_vigente", "valor_vigente") if k in res}
            tools.append({"name": t["ferramenta"], "args": t["args"], "result": res})
        out.append({
            "id": q["id"], "type": q["tipo"], "question": text, "expected": q.get("resposta_esperada", ""),
            "correct": bool(str(ok).lower() in ("true", "1", "1.0")), "found": r["encontrado"],
            "threshold_refusal": bool(r.get("recusa_por_limiar")), "answer": r["resposta"],
            "relevance": r.get("escore_relevancia"), "n_retrieved": len(r["trechos"]),
            "passages": [{"n": t["n"], "citation": t["citacao"],
                          "text": chunks.get(t["id"], {}).get("text", "")[:700]} for t in cited],
            "tools": tools, "seconds": r["tempo_total_s"], "tokens": r["tokens_entrada"] + r["tokens_saida"],
            "validation_ok": bool((r.get("validacao") or {}).get("ok", False)), "llm": r["llm"],
        })
    return out


def geojson(stats):
    feats = []
    for r in stats.itertuples():
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(r.lon, 4), round(r.lat, 4)]},
            "properties": {
                "station": r.name,
                "days measured": int(r.dias),
                "mean PM2.5 (µg/m³)": round(r.media, 1),
                "daily max (µg/m³)": round(r.maximo, 1),
                "days above 25 µg/m³": f"{r.acima25:.0%}",
                "marker-color": "#d95926" if r.media >= 15 else "#2a78d6",
                "marker-size": "medium" if r.acima25 >= 0.1 else "small",
            },
        })
    return {"type": "FeatureCollection", "features": feats}


def outline():
    """Simplified state outline (rings separated by None), so the map needs no external tiles."""
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
        "runs": agent_runs(),
        "shap": [{"var": SHAP_LABELS.get(r.variavel, r.variavel), "valor": r.shap_medio_abs}
                 for r in csv("shap_importancia.csv").head(12).itertuples()],
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "data.js").write_text("window.DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    (DOCS / "stations.geojson").write_text(json.dumps(geojson(stats), ensure_ascii=False, indent=1), encoding="utf-8")
    print("docs/data.js and docs/stations.geojson updated")


if __name__ == "__main__":
    main()
