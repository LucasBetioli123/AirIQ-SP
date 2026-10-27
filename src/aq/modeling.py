"""Núcleo da modelagem: validação espaço-temporal, modelos, intervalos conformais e métricas.

Validação (o que simula o uso real):
  - as estações são divididas em K grupos (GroupKFold). Em cada fold, as estações de
    teste ficam FORA do treino e FORA do lag espacial (IDW): o modelo nunca as vê
  - período: treino até `train_end`, calibração do intervalo em `calib`, TESTE em
    `test_start` em diante (período posterior)
  - a métrica principal é calculada em (estações de teste) x (período de teste)
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

# variáveis do modelo de ESTIMATIVA (mesmo dia)
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
# previsão D+1: o mesmo conjunto (variáveis de D) + meteorologia/calendário de D+1
FC_EXTRA = ["next_blh_pm", "next_ws_pm", "next_rh", "next_tp", "next_tcc", "next_t2m", "next_log_vent",
            "next_dow", "next_weekend"]


# ----------------------------------------------------------------------------- métricas
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
    """Atribui cada estação a um fold de forma determinística (embaralhada com seed)."""
    st = np.array(sorted(set(stations)))
    rng = np.random.default_rng(SEED)
    rng.shuffle(st)
    return {s: i % k for i, s in enumerate(st)}


def period_masks(dates: pd.Series, tf: dict | None = None) -> dict:
    """Máscaras treino/calibração/teste para um corte temporal `tf` (padrão: o corte final do config)."""
    tf = tf or SPLIT
    d = pd.to_datetime(dates)
    end = pd.Timestamp(tf.get("test_end", "2100-01-01"))
    return {"train": d <= pd.Timestamp(tf["train_end"]),
            "calib": (d >= pd.Timestamp(tf["calib_start"])) & (d <= pd.Timestamp(tf["calib_end"])),
            "test": (d >= pd.Timestamp(tf["test_start"])) & (d <= end)}


TEMPORAL_FOLDS = SPLIT["temporal_folds"]


def with_lag(rows: pd.DataFrame, ref_pool: pd.DataFrame, value_col: str = "pm25", prefix: str = "idw") -> pd.DataFrame:
    """Adiciona o IDW do mesmo dia calculado SÓ com as estações de `ref_pool`."""
    refs = ref_pool[["date", "lat", "lon", value_col]].dropna()
    return add_spatial_lag(rows, refs, value_col=value_col, prefix=prefix)


# ----------------------------------------------------------------------------- modelos
def make_model(name: str, quantile: float | None = None):
    """Zoológico de modelos do benchmark. Todos recebem X com NaN."""
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
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
        if name == "xgboost_reg":  # mesma lógica do lightgbm_reg: árvores rasas e folhas com mais amostras
            params.update(max_depth=4, min_child_weight=30, colsample_bytree=0.5, reg_lambda=5.0, reg_alpha=0.5)
        if name == "xgboost_forte":  # ainda mais raso/regularizado (sugestão de teste do usuário)
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
        if name == "lightgbm_reg":  # mais regularizado: poucas estações -> risco de decorar o local
            params.update(n_estimators=700, num_leaves=15, min_child_samples=60, colsample_bytree=0.5,
                          reg_lambda=5.0, extra_trees=True)
        if quantile is not None:
            params.update(objective="quantile", alpha=quantile)
        return lgb.LGBMRegressor(**params)
    if name == "mlp":
        # rede neural pequena para dados tabulares: 2 camadas ocultas, L2 e early stopping
        from sklearn.neural_network import MLPRegressor
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, learning_rate_init=1e-3,
                                          batch_size=128, max_iter=400, early_stopping=True, validation_fraction=0.15,
                                          n_iter_no_change=20, random_state=SEED))
    if name == "ensemble_rf_lgbm":
        return Ensemble([make_model("random_forest"), make_model("lightgbm_reg")])
    raise ValueError(name)


class Ensemble:
    """Média simples das previsões (RF e GBM erram de formas diferentes)."""

    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


class OffsetLogTarget:
    """Alvo RELATIVO ao IDW do mesmo dia: log1p(PM) - log1p(IDW).

    O nível regional do dia (que muda muito entre estação seca e chuvosa) vem das
    medições das estações. O modelo aprende só a correção local (urbanização, fumaça,
    relevo, meteorologia). Isso resolve a principal falha das árvores: elas não
    extrapolam para níveis de PM2.5 que não viram no treino (episódios de seca/fumaça).
    Sem IDW disponível (nenhuma estação mediu no dia), o offset vira a mediana do treino."""

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
    """Ajusta em log1p(PM2.5): a distribuição é assimétrica e o erro é multiplicativo.
    A volta com expm1 aproxima a mediana condicional, o que favorece o MAE."""

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        # limite da volta da escala log: 1,5x o máximo do treino (modelos lineares extrapolam)
        self.cap = float(np.log1p(1.5 * np.nanmax(y)))
        self.model.fit(X, np.log1p(np.clip(y, 0, None)))
        return self

    def predict(self, X):
        return np.expm1(np.clip(self.model.predict(X), 0, self.cap)).clip(0)


# ----------------------------------------------------------------------------- intervalos
class CQR:
    """Conformalized Quantile Regression (Romano et al., 2019) em escala log.

    1. quantis q_lo, q_hi (LightGBM quantílico) ajustados no treino
    2. na calibração: escore E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))
    3. Q = quantil ceil((n+1)(1-α))/n dos escores
    4. intervalo final: [q_lo - Q, q_hi + Q]
    Garantia de cobertura marginal >= 1-α se calibração e teste forem trocáveis. Como a
    calibração usa previsões FORA da estação (cross-fitting por grupos), os escores
    refletem o erro em locais não monitorados, que é o caso de uso."""

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
    """Três LightGBM quantílicos (α/2, 0.5, 1-α/2) em escala log, com o mesmo offset do IDW
    do modelo pontual (os quantis são da correção local; somamos log1p(IDW) de volta)."""

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
        lo, hi = np.minimum(lo, med), np.maximum(hi, med)  # evita cruzamento de quantis
        return lo, med, hi
