"""Climate-risk classifier.

The risk category for a season is a deterministic function of that season's
rainfall anomaly, so predicting it from same-season rainfall would be trivial
leakage. Instead the model forecasts the category from information available
*before* the season: previous seasons' rainfall anomalies and risk scores,
location, crop and the crop's reference climate profile.
"""

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from config import MODEL_DIR, RANDOM_STATE
from models.common import (
    STATIC_CLIMATE, add_lags, available, preprocessor, temporal_split,
)

MODEL_PATH = MODEL_DIR / "risk_model.joblib"
CATEGORICAL = ["state", "crop"]
LAG_FEATURES = [
    "rainfall_anomaly_lag1", "rainfall_anomaly_lag2",
    "climate_risk_lag1", "climate_risk_lag2",
]


def _metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train_risk_model(df):
    df = add_lags(df, "rainfall_anomaly")
    df = add_lags(df, "climate_risk")
    df = df.dropna(subset=["risk_category"])
    df["risk_category_lag1"] = df.groupby(["district", "crop"])["risk_category"].shift(1)

    categorical = available(df, CATEGORICAL)
    numeric = available(df, ["year"] + LAG_FEATURES + STATIC_CLIMATE)
    train, val, test, years = temporal_split(df)

    model = Pipeline([
        ("prep", preprocessor(categorical, numeric)),
        ("clf", HistGradientBoostingClassifier(
            categorical_features=list(range(len(categorical))),
            class_weight="balanced",
            max_iter=300,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        )),
    ])
    features = categorical + numeric
    model.fit(train[features], train["risk_category"])

    majority = train["risk_category"].mode()[0]
    results = {"task": "forecast season risk category from prior seasons",
               "features": features, "split_years": years,
               "class_distribution_train": train["risk_category"].value_counts().to_dict()}

    for name, part in [("validation", val), ("test", test)]:
        y = part["risk_category"]
        persistence = part["risk_category_lag1"].fillna(majority)
        results[name] = {
            "rows": int(len(part)),
            "model": _metrics(y, model.predict(part[features])),
            "baseline_majority_class": _metrics(y, [majority] * len(y)),
            "baseline_last_season": _metrics(y, persistence),
        }

    results["test_report"] = classification_report(
        test["risk_category"], model.predict(test[features]),
        output_dict=True, zero_division=0,
    )

    joblib.dump({"model": model, "features": features}, MODEL_PATH)
    results["model_path"] = str(MODEL_PATH)
    return results
