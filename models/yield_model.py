"""Crop-yield regressor.

Predicts the season's yield (kg/ha) from the district-crop yield baseline
(rolling mean of *previous* seasons), observed regional rainfall for the
season, location, crop and the crop's reference climate profile.
"""

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from config import MODEL_DIR, RANDOM_STATE
from models.common import STATIC_CLIMATE, add_lags, available, preprocessor, temporal_split

MODEL_PATH = MODEL_DIR / "yield_model.joblib"
CATEGORICAL = ["state", "crop"]
NUMERIC = [
    "year", "yield_baseline", "crop_yield_lag1",
    "rainfall_real_mm", "rainfall_anomaly", "climate_risk",
]


def _metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_yield_model(df):
    df = add_lags(df, "crop_yield", lags=(1,))
    df = df.dropna(subset=["crop_yield"])

    categorical = available(df, CATEGORICAL)
    numeric = available(df, NUMERIC + STATIC_CLIMATE)
    features = categorical + numeric
    train, val, test, years = temporal_split(df)

    model = Pipeline([
        ("prep", preprocessor(categorical, numeric)),
        ("reg", HistGradientBoostingRegressor(
            categorical_features=list(range(len(categorical))),
            max_iter=400,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        )),
    ])
    model.fit(train[features], train["crop_yield"])

    results = {"task": "predict crop_yield (kg/ha)", "features": features,
               "split_years": years}
    for name, part in [("validation", val), ("test", test)]:
        y = part["crop_yield"]
        results[name] = {
            "rows": int(len(part)),
            "model": _metrics(y, model.predict(part[features])),
            # Persistence baseline: predict the rolling mean of past seasons.
            "baseline_historical_mean": _metrics(y, part["yield_baseline"]),
        }

    joblib.dump({"model": model, "features": features}, MODEL_PATH)
    results["model_path"] = str(MODEL_PATH)
    return results
