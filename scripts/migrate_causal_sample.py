"""Idempotent migration of legacy tables: removes temporal gap-filling and D+1 weather.

Never invents observations: only uses the three spatial features already stored.
The full pipeline produces the same schema directly in build_dataset.
"""
import numpy as np
import pandas as pd

from src.aq.features import physics_features
from src.aq.modeling import FEATURE_SCHEMA
from src.common.config import data_path


def migrate(df):
    df = df.copy()
    filled = np.full(len(df), np.nan, dtype="float32")
    level = np.full(len(df), 4, dtype="int8")
    for i, name in enumerate(("aod", "aod_w3", "aod_w9")):
        values = df[name].to_numpy()
        keep = np.isnan(filled) & np.isfinite(values)
        filled[keep], level[keep] = values[keep], i
    df["aod_filled"], df["aod_fill_level"] = filled, level
    for name, values in physics_features(df).items():
        df[name] = values
    df = df.drop(columns=[c for c in df if c.startswith("next_") and c not in ("next_dow", "next_weekend")])
    df["feature_schema"] = FEATURE_SCHEMA
    return df


if __name__ == "__main__":
    for filename in ("station_days.parquet", "station_days_fc.parquet"):
        path = data_path("sample", filename)
        df = migrate(pd.read_parquet(path))
        df.to_parquet(path, index=False, compression="zstd")
        print(filename, len(df), df.aod_fill_level.value_counts().to_dict())
