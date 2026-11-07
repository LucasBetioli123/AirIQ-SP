"""Validate an installation: tests, real inference, API and published metrics; writes UTF-8 evidence.

Usage: python -m scripts.verify_install (after training with `python -m src.aq.pipeline --quick`)."""
import json
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.aq.api import app
from src.aq.modeling import FEATURE_SCHEMA
from src.aq.predict import available_index, predict_one, station_obs
from src.common.config import ROOT


def main():
    path = ROOT / "logs/validacao_instalacao.txt"
    with path.open("w", encoding="utf-8") as log:
        def record(label, value):
            log.write(label + " " + json.dumps(value, ensure_ascii=False, default=str) + "\n")
            log.flush()
        record("inicio_utc", datetime.now(timezone.utc).isoformat())
        record("feature_schema", FEATURE_SCHEMA)
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        for horizon in (0, 1):
            result = predict_one(-22.9053, -47.0659, "2022-10-11", horizon)
            assert result["intervalo_90"][0] <= result["pm25_ug_m3"] <= result["intervalo_90"][1]
            assert result["feature_schema"] == FEATURE_SCHEMA
            record("inferencia", result)
        # The forecast must work even without observations on the target date.
        day = available_index().max()
        record("previsao_sem_contexto_futuro", predict_one(-22.9053, -47.0659, str(day.date()), 1))
        obs = station_obs()
        own = obs[obs.date == pd.Timestamp("2023-04-03")].iloc[0]
        record("previsao_local_monitorado", predict_one(float(own.lat), float(own.lon), "2023-04-03", 1))
        with TestClient(app) as client:
            cases = [({"municipio": "Campinas", "date": "2022-10-11", "horizon": 1}, 200),
                     ({"municipio": "Campinas", "date": "2030-01-01"}, 404),
                     ({"municipio": "Campinas", "date": "2022-10-11", "horizon": 2}, 422)]
            for params, expected in cases:
                response = client.get("/predict", params=params)
                assert response.status_code == expected, response.text
                record("api", {"params": params, "status": response.status_code, "body": response.json()})
        oof = pd.read_parquet(ROOT / "data/processed/oof_previsao_d1.parquet")
        published = pd.read_csv(ROOT / "reports/metrics/intervalos_d1.csv")
        for row in published.itertuples():
            g = oof[(oof.cenario == row.cenario) & np.isfinite(oof.pers)]
            if row.recorte != "todos":
                g = g[g.tfold == row.recorte]
            coverage = ((g.pm25 >= g.cqr_lo) & (g.pm25 <= g.cqr_hi)).mean()
            assert len(g) == row.n and abs(coverage - row.cobertura_real) < 0.000051
            assert (g.data_alvo == g.date + pd.Timedelta(days=1)).all()
            record("cobertura_d1_recalculada", {"cenario": row.cenario, "recorte": row.recorte,
                                               "n": len(g), "cobertura": coverage})
        record("conclusao", "tests, inference, API and D+1 coverage passed")
    print(path)


if __name__ == "__main__":
    main()
