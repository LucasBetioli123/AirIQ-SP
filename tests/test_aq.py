import numpy as np
import pandas as pd

from src.aq.aqi import band, legal_standard_24h
from src.aq.features import add_spatial_lag, bearing, haversine, idw
from src.aq.modeling import CQR, station_folds


def test_bands():
    assert band(10) == "Boa" and band(25) == "Boa" and band(25.1) == "Moderada"
    assert band(60) == "Ruim" and band(100) == "Muito Ruim" and band(200) == "Péssima"
    assert list(band(np.array([5, 55]))) == ["Boa", "Ruim"]


def test_band_metrics_expose_majority_class():
    from src.aq.evaluate import band_metrics

    # 9 dias "Boa" e 1 "Moderada": prever sempre 10 acerta 90% das faixas, mas não detecta o dia poluído
    y = np.array([10.0] * 9 + [40.0])
    m = band_metrics(y, np.full(10, 10.0))
    assert m["acerto_faixa"] == 0.9 and m["acerto_sempre_Boa"] == 0.9
    assert m["sensibilidade_>25"] == 0.0 and m["acuracia_balanceada"] == 0.5


def test_legal_standard_by_date():
    assert legal_standard_24h("2017-01-01")["valor"] is None
    assert legal_standard_24h("2022-06-01")["valor"] == 60
    assert legal_standard_24h("2025-06-01")["valor"] == 50


def test_geometry():
    # São Paulo -> Campinas ~ 84 km, rumo ~ noroeste
    d = haversine(-23.55, -46.63, -22.90, -47.06)
    assert 75 < d < 90
    b = bearing(-23.55, -46.63, -22.90, -47.06)
    assert 300 < b < 340


def test_idw_excludes_self_and_weights_by_distance():
    v, dmin, n = idw(0.0, 0.0, np.array([0.0, 0.0, 0.1]), np.array([0.0, 0.05, 0.0]), np.array([999, 10.0, 20.0]))
    assert n == 2 and 10 < v < 20 and v < 15  # a mais próxima pesa mais; a própria (d=0) é ignorada


def test_spatial_lag_same_date_only():
    rows = pd.DataFrame({"date": pd.to_datetime(["2022-01-01", "2022-01-02"]), "lat": [0.0, 0.0], "lon": [0.0, 0.0]})
    refs = pd.DataFrame({"date": pd.to_datetime(["2022-01-01"]), "lat": [0.1], "lon": [0.1], "pm25": [12.0]})
    out = add_spatial_lag(rows, refs)
    assert out["idw"].iloc[0] == 12.0 and np.isnan(out["idw"].iloc[1])


def test_cqr_coverage_synthetic():
    rng = np.random.default_rng(0)
    n = 4000
    y = np.exp(rng.normal(3, 0.5, n)) - 1
    ly = np.log1p(y)
    # quantis propositalmente estreitos demais: o CQR precisa alargar
    lo, hi = ly - 0.3, ly + 0.3 + rng.normal(0, 0.4, n)
    lo = lo + rng.normal(0, 0.4, n)
    cal, te = slice(0, 2000), slice(2000, None)
    c = CQR(alpha=0.1).calibrate(y[cal], lo[cal], hi[cal])
    a, b = c.interval(lo[te], hi[te])
    cov = np.mean((y[te] >= a) & (y[te] <= b))
    assert 0.87 < cov < 0.93


def test_station_folds_deterministic():
    st = np.arange(30)
    f1, f2 = station_folds(st, 5), station_folds(st, 5)
    assert f1 == f2 and set(f1.values()) == set(range(5))
