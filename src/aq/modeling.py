"""Modeling core: spatio-temporal validation, models, conformal intervals and metrics.

Validation (mimics real use):
  - stations are split into K groups (GroupKFold). In each fold, the test stations stay
    OUT of training and OUT of the spatial lag (IDW): the model never sees them
  - period: train up to `train_end`, calibrate the interval on `calib`, TEST from
    `test_start` on (a later period)
  - the headline metric is computed on (test stations) x (test period)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.aq.aqi import band
from src.aq.features import add_spatial_lag
from src.common.config import load_config

CFG = load_config()
SPLIT = CFG["split"]
SEED = CFG["seed"]

# features of the same-day ESTIMATE model
SAT = ["aod", "aod_w3", "aod_w9", "aod_max", "aod_nscan", "aod_filled", "aod_fill_level", "no2", "no2_w9", "no2_filled"]
MET = ["t2m", "rh", "ws", "ws_pm", "u10", "v10", "sp", "tcc", "blh", "blh_pm", "blh_min", "tp", "prev_tp",
       "prev_blh_pm"]
PHYS = ["aod_blh", "aod_dry", "aod_dry_blh", "vent", "log_vent"]
STATIC = ["built", "built_5km", "tree_5km", "crop_5km", "grass_5km", "elev"]
CAL = ["doy_sin", "doy_cos", "dow", "weekend"]
FIRE = ["fire_n_50km", "fire_n_150km", "fire_n_400km", "fire_up_150km", "fire_up_400km",
        "fire_n_150km_lag1", "fire_up_400km_lag1"]
LAG = ["idw", "idw_dmin", "idw_n"]
EST_FEATURES = SAT + MET + PHYS + STATIC + CAL + FIRE + LAG
# D+1: only observations up to D and tomorrow's known calendar.
FEATURE_SCHEMA = "causal-spatial-aod-v2"
FC_EXTRA = ["next_dow", "next_weekend"]


# ----------------------------------------------------------------------------- metrics
def metrics(y, yhat, lo=None, hi=None, baseline=None) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = {"n": int(len(y)), "MAE": mean_absolute_error(y, yhat), "RMSE": float(np.sqrt(mean_squared_error(y, yhat))),
         "R2": r2_score(y, yhat), "bias": float(np.mean(yhat - y)),
         "acerto_faixa_IQAr": float(np.mean(band(y) == band(yhat)))}
    if baseline is not None:
        bmae = mean_absolute_error(y, baseline)
        m["ganho_MAE_vs_baseline_%"] = 100 * (bmae - m["MAE"]) / bmae
    if lo is not None:
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        m["cobertura"] = float(np.mean((y >= lo) & (y <= hi)))
        m["largura_media"] = float(np.mean(hi - lo))
    return m


# ----------------------------------------------------------------------------- splits
def station_folds(stations: np.ndarray, k: int = SPLIT["n_station_folds"]) -> dict:
    """Assign each station to a fold deterministically (shuffled with a seed)."""
    st = np.array(sorted(set(stations)))
    rng = np.random.default_rng(SEED)
    rng.shuffle(st)
    return {s: i % k for i, s in enumerate(st)}


def period_masks(dates: pd.Series, tf: dict | None = None) -> dict:
    """Train/calibration/test masks for a temporal split `tf` (default: the final split in the config)."""
    tf = tf or SPLIT
    d = pd.to_datetime(dates)
    end = pd.Timestamp(tf.get("test_end", "2100-01-01"))
    return {"train": d <= pd.Timestamp(tf["train_end"]),
            "calib": (d >= pd.Timestamp(tf["calib_start"])) & (d <= pd.Timestamp(tf["calib_end"])),
            "test": (d >= pd.Timestamp(tf["test_start"])) & (d <= end)}


TEMPORAL_FOLDS = SPLIT["temporal_folds"]


def with_lag(rows: pd.DataFrame, ref_pool: pd.DataFrame, value_col: str = "pm25", prefix: str = "idw") -> pd.DataFrame:
    """Add the same-day IDW computed ONLY from the stations in `ref_pool`."""
    refs = ref_pool[["date", "lat", "lon", value_col]].dropna()
    return add_spatial_lag(rows, refs, value_col=value_col, prefix=prefix)


# ----------------------------------------------------------------------------- modelos
def make_model(name: str, quantile: float | None = None):
    """The benchmark model zoo. Every model receives X with NaNs."""
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer

    # continuous models: QuantileTransformer (robust to the long tails of fire/NO2; with
    # StandardScaler, Ridge and MLP extrapolated and diverged during episodes)
    def qnorm():
        return QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=SEED)

    if name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), qnorm(),
                             RidgeCV(alphas=np.logspace(-1, 4, 20)))
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.4,
                                                   n_jobs=-1, random_state=SEED))
    if name == "hist_gb":
        return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.04, min_samples_leaf=30,
                                             l2_regularization=1.0, random_state=SEED)
    if name in ("xgboost", "xgboost_reg", "xgboost_forte"):
        import xgboost as xgb
        params = dict(n_estimators=700, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.7,
                      min_child_weight=5, reg_lambda=2.0, n_jobs=4, random_state=SEED, tree_method="hist")
        if name == "xgboost_reg":  # same logic as lightgbm_reg: shallow trees and larger leaves
            params.update(max_depth=4, min_child_weight=30, colsample_bytree=0.5, reg_lambda=5.0, reg_alpha=0.5)
        if name == "xgboost_forte":  # even shallower/more regularized
            params.update(n_estimators=800, max_depth=3, min_child_weight=60, subsample=0.7, colsample_bytree=0.4,
                          colsample_bynode=0.7, reg_lambda=10.0, reg_alpha=1.0, gamma=0.02)
        if quantile is not None:
            params.update(objective="reg:quantileerror", quantile_alpha=quantile)
        return xgb.XGBRegressor(**params)
    if name in ("lightgbm", "lightgbm_reg"):
        import lightgbm as lgb
        params = dict(n_estimators=900, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                      random_state=SEED, n_jobs=4, verbose=-1)
        if name == "lightgbm_reg":  # more regularized: few stations -> risk of memorizing locations
            params.update(n_estimators=700, num_leaves=15, min_child_samples=60, colsample_bytree=0.5,
                          reg_lambda=5.0, extra_trees=True)
        if quantile is not None:
            params.update(objective="quantile", alpha=quantile)
        return lgb.LGBMRegressor(**params)
    if name == "mlp":
        # small neural net for tabular data: 2 hidden layers, L2 and early stopping
        from sklearn.neural_network import MLPRegressor
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), qnorm(),
                             MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, learning_rate_init=1e-3,
                                          batch_size=128, max_iter=400, early_stopping=True, validation_fraction=0.15,
                                          n_iter_no_change=20, random_state=SEED))
    if name == "mlp_tanh":
        from src.aq.nn_torch import TorchTanhRegressor  # PyTorch: tanh output, target in [-0.7, 0.7]
        return TorchTanhRegressor(hidden=(64, 32), out_range=0.7, weight_decay=1e-2, dropout=0.1, seed=SEED)
    if name == "ensemble_rf_lgbm":
        return Ensemble([make_model("random_forest"), make_model("lightgbm_reg")])
    raise ValueError(name)


class Ensemble:
    """Simple average of predictions (RF and GBM make different errors)."""

    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


class OffsetLogTarget:
    """Target RELATIVE to the same-day IDW: log1p(PM) - log1p(IDW).

    The day's regional level (which varies a lot between dry and wet seasons) comes from
    the station measurements. The model only learns the local correction (urbanization,
    smoke, terrain, weather). This fixes the main weakness of trees: they cannot
    extrapolate to PM2.5 levels unseen in training (dry-season/smoke episodes).
    Without an IDW (no station measured that day), the offset falls back to the training median."""

    def __init__(self, model, offset_col: str = "idw"):
        self.model = model
        self.offset_col = offset_col

    def _offset(self, X):
        o = np.log1p(np.clip(X[self.offset_col].to_numpy(dtype=float), 0, None))
        return np.where(np.isfinite(o), o, self.fallback)

    def fit(self, X, y):
        self.fallback = float(np.nanmedian(np.log1p(np.clip(y, 0, None))))
        self.cap = float(np.log1p(1.5 * np.nanmax(y)))
        self.model.fit(X, np.log1p(np.clip(y, 0, None)) - self._offset(X))
        return self

    def predict(self, X):
        return np.expm1(np.clip(self._offset(X) + self.model.predict(X), 0, self.cap)).clip(0)


class LogTarget:
    """Fit on log1p(PM2.5): the distribution is skewed and the error is multiplicative.
    Back-transforming with expm1 approximates the conditional median, which favors MAE."""

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        # cap on the log back-transform: 1.5x the training max (linear models extrapolate)
        self.cap = float(np.log1p(1.5 * np.nanmax(y)))
        self.model.fit(X, np.log1p(np.clip(y, 0, None)))
        return self

    def predict(self, X):
        return np.expm1(np.clip(self.model.predict(X), 0, self.cap)).clip(0)


# ----------------------------------------------------------------------------- intervalos
class CQR:
    """Conformalized Quantile Regression (Romano et al., 2019) on the log scale.

    1. quantiles q_lo, q_hi (quantile LightGBM) fitted on training data
    2. on calibration data: score E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))
    3. Q = the ceil((n+1)(1-α))/n quantile of the scores
    4. final interval: [q_lo - Q, q_hi + Q]
    Classic split conformal assumes exchangeability. Here there is spatio-temporal
    dependence and cross-fitting across stations, so we report empirical coverage without
    promising an exact guarantee for new places, periods or regimes."""

    def __init__(self, alpha: float = SPLIT["interval_alpha"]):
        self.alpha = alpha
        self.q = 0.0

    def calibrate(self, y, lo_log, hi_log):
        y = np.log1p(np.clip(np.asarray(y, float), 0, None))
        s = np.maximum(lo_log - y, y - hi_log)
        n = len(s)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q = float(np.quantile(s, level, method="higher"))
        return self

    def interval(self, lo_log, hi_log):
        return np.expm1(lo_log - self.q).clip(0), np.expm1(hi_log + self.q)


class QuantileTrio:
    """Three quantile LightGBMs (α/2, 0.5, 1-α/2) on the log scale, with the same IDW offset as
    the point model (the quantiles are of the local correction; log1p(IDW) is added back)."""

    def __init__(self, alpha: float = SPLIT["interval_alpha"], offset_col: str | None = "idw", base: str = "lightgbm"):
        self.qs = (alpha / 2, 0.5, 1 - alpha / 2)
        self.models = [make_model(base, quantile=q) for q in self.qs]
        self.offset_col = offset_col

    def _offset(self, X):
        if self.offset_col is None:
            return 0.0
        o = np.log1p(np.clip(X[self.offset_col].to_numpy(dtype=float), 0, None))
        return np.where(np.isfinite(o), o, self.fallback)

    def fit(self, X, y):
        ly = np.log1p(np.clip(y, 0, None))
        self.fallback = float(np.nanmedian(ly))
        for m in self.models:
            m.fit(X, ly - self._offset(X))
        return self

    def predict_log(self, X):
        off = self._offset(X)
        lo, med, hi = (m.predict(X) + off for m in self.models)
        lo, hi = np.minimum(lo, med), np.maximum(hi, med)  # prevent quantile crossing
        return lo, med, hi
