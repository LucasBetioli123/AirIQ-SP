"""API HTTP (FastAPI): GET /predict?lat=..&lon=..&date=aaaa-mm-dd&horizon=0|1

  uvicorn src.aq.api:app --port 8000
  curl 'localhost:8000/predict?lat=-22.90&lon=-47.06&date=2023-03-10'
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from src.aq.data.geo import geocode
from src.aq.predict import DataUnavailable, available_dates, predict_one

app = FastAPI(title="AirIQ SP - PM2.5 estimate and forecast with confidence interval", version="1.0")


@app.get("/health")
def health():
    lo, hi = available_dates()
    return {"status": "ok", "periodo_disponivel": [str(lo.date()), str(hi.date())]}


@app.get("/predict")
def predict(date: str, lat: float | None = Query(None, ge=-26, le=-19), lon: float | None = Query(None, ge=-54, le=-43),
            municipio: str | None = None, horizon: int = Query(0, ge=0, le=1)):
    if municipio:
        g = geocode(municipio)
        if g is None:
            raise HTTPException(404, f"municipality not found in São Paulo state: {municipio}")
        lat, lon = g["lat"], g["lon"]
    if lat is None or lon is None:
        raise HTTPException(422, "provide lat/lon or municipio")
    try:
        return predict_one(lat, lon, date, horizon)
    except DataUnavailable as e:
        raise HTTPException(404, str(e))
