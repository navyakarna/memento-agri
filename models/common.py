"""Shared helpers for the ML baselines."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

STATIC_CLIMATE = [
    "temperature", "humidity", "ph", "rainfall", "wind_speed", "solar_radiation",
]
# n_req/p_req/k_req (and total_n/p/k) are NOT used as features: in the source
# dataset they equal crop_yield x a fixed per-crop constant, so they leak the
# same-season yield.


def temporal_split(df):
    """Same rule as experiments/run_temporal_split.py:
    earliest 70% of years -> train, next 20% -> validation, last 10% -> test."""
    years = sorted(pd.to_numeric(df["year"], errors="coerce").dropna().astype(int).unique())
    train_end = years[int(len(years) * 0.70) - 1]
    val_end = years[int(len(years) * 0.90) - 1]
    train = df[df["year"] <= train_end]
    val = df[(df["year"] > train_end) & (df["year"] <= val_end)]
    test = df[df["year"] > val_end]
    years_info = {
        "train": [int(train["year"].min()), int(train["year"].max())],
        "validation": [int(val["year"].min()), int(val["year"].max())],
        "test": [int(test["year"].min()), int(test["year"].max())],
    }
    return train, val, test, years_info


def available(df, columns):
    return [c for c in columns if c in df.columns]


def preprocessor(categorical, numeric):
    """Ordinal-encode categoricals (unknown -> -1); pass numerics through."""
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return ColumnTransformer(
        [("cat", encoder, categorical), ("num", "passthrough", numeric)],
        verbose_feature_names_out=False,
    )


def add_lags(df, column, lags=(1, 2)):
    """Previous seasons' values for the same district and crop (no leakage)."""
    df = df.sort_values(["district", "crop", "year"]).copy()
    grouped = df.groupby(["district", "crop"])[column]
    for lag in lags:
        df[f"{column}_lag{lag}"] = grouped.shift(lag)
    return df
