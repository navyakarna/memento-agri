import pandas as pd
import numpy as np

from config import PROCESSED_DATA
from data.real_climate import attach_real_rainfall


COLUMN_MAP = {
    "Dist Code": "district_code",
    "Year": "year",
    "State Code": "state_code",
    "State Name": "state",
    "Dist Name": "district",
    "Crop": "crop",
    "Area_ha": "crop_area",
    "Yield_kg_per_ha": "crop_yield",

    "N_req_kg_per_ha": "n_req",
    "P_req_kg_per_ha": "p_req",
    "K_req_kg_per_ha": "k_req",

    "Total_N_kg": "total_n",
    "Total_P_kg": "total_p",
    "Total_K_kg": "total_k",

    "Temperature_C": "temperature",
    "Humidity_%": "humidity",
    "pH": "ph",
    "Rainfall_mm": "rainfall",
    "Wind_Speed_m_s": "wind_speed",
    "Solar_Radiation_MJ_m2_day": "solar_radiation",
}


def clean_column_names(df):

    # Remove accidental ** characters
    df.columns = (
        df.columns
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    # Remove possible spaces around names
    df.columns = df.columns.str.replace(" ", " ", regex=False)

    return df


def load_raw(path):

    df = pd.read_csv(path)

    df = clean_column_names(df)

    # Apply canonical column names
    df = df.rename(columns=COLUMN_MAP)

    required = [
        "year",
        "state",
        "district",
        "crop",
        "crop_yield",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
        "wind_speed",
        "solar_radiation",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    return df


def clean_data(df):

    df = df.copy()

    # Text columns
    text_columns = [
        "state",
        "district",
        "crop"
    ]

    for col in text_columns:

        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    # Numeric columns
    numeric_columns = [
        "year",
        "crop_area",
        "crop_yield",
        "n_req",
        "p_req",
        "k_req",
        "total_n",
        "total_p",
        "total_k",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
        "wind_speed",
        "solar_radiation"
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # Remove impossible observations

    df = df[
        (df["crop_yield"] >= 0) &
        (df["rainfall"] >= 0) &
        (df["humidity"] >= 0) &
        (df["humidity"] <= 100) &
        (df["wind_speed"] >= 0) &
        (df["solar_radiation"] >= 0)
    ]

    return df


def add_climate_features(df):

    df = df.copy()

    # ------------------------------------------------
    # 0. Attach REAL historical rainfall (IMD, by state/year)
    #
    # The source dataset's "rainfall" column is a fixed
    # per-crop reference profile (identical across every
    # district and year), so anomalies computed from it are
    # always zero. We replace the rainfall anomaly input with
    # a genuine (state, year) historical rainfall series
    # merged in from IMD sub-divisional records. See
    # data/real_climate.py for the full explanation and
    # source citation.
    # ------------------------------------------------

    df = attach_real_rainfall(df)

    # ------------------------------------------------
    # 1. State-crop historical rainfall baseline (real data)
    # ------------------------------------------------

    rainfall_mean = (
        df
        .groupby(["state", "crop"])["rainfall_real_mm"]
        .transform("mean")
    )

    rainfall_std = (
        df
        .groupby(["state", "crop"])["rainfall_real_mm"]
        .transform("std")
        .fillna(1)
    )

    # Fractional anomaly (-0.25 = 25% below the state-crop
    # normal). This is the same convention used by the
    # Streamlit app, the ablation experiment and the agent.
    df["rainfall_anomaly"] = (
        df["rainfall_real_mm"] - rainfall_mean
    ) / (rainfall_mean + 1e-6)

    # Standardised (z-score) version, used for the stress score
    # because it accounts for how variable rainfall normally is
    # in that state.
    df["rainfall_anomaly_z"] = (
        df["rainfall_real_mm"] - rainfall_mean
    ) / (rainfall_std + 1e-6)


    # ------------------------------------------------
    # 2. Temperature anomaly
    # ------------------------------------------------

    temp_mean = (
        df
        .groupby(["district", "crop"])["temperature"]
        .transform("mean")
    )

    df["temperature_anomaly"] = (
        df["temperature"] - temp_mean
    ) / (temp_mean.abs() + 1e-6)


    # ------------------------------------------------
    # 3. Humidity anomaly
    # ------------------------------------------------

    humidity_mean = (
        df
        .groupby(["district", "crop"])["humidity"]
        .transform("mean")
    )

    df["humidity_anomaly"] = (
        df["humidity"] - humidity_mean
    ) / (humidity_mean.abs() + 1e-6)


    # ------------------------------------------------
    # 4. Wind anomaly
    # ------------------------------------------------

    wind_mean = (
        df
        .groupby(["district", "crop"])["wind_speed"]
        .transform("mean")
    )

    df["wind_anomaly"] = (
        df["wind_speed"] - wind_mean
    ) / (wind_mean.abs() + 1e-6)


    # ------------------------------------------------
    # 5. Solar radiation anomaly
    # ------------------------------------------------

    solar_mean = (
        df
        .groupby(["district", "crop"])["solar_radiation"]
        .transform("mean")
    )

    df["solar_anomaly"] = (
        df["solar_radiation"] - solar_mean
    ) / (solar_mean.abs() + 1e-6)


    # ------------------------------------------------
    # 6. Historical yield baseline
    # ------------------------------------------------

    df = df.sort_values(
        ["district", "crop", "year"]
    )

    df["yield_baseline"] = (
        df
        .groupby(["district", "crop"])["crop_yield"]
        .transform(
            lambda x:
            x.rolling(
                window=5,
                min_periods=2
            )
            .mean()
            .shift(1)
        )
    )

    # Fallback to group mean

    fallback = (
        df
        .groupby(["district", "crop"])["crop_yield"]
        .transform("mean")
    )

    df["yield_baseline"] = (
        df["yield_baseline"]
        .fillna(fallback)
    )


    # ------------------------------------------------
    # 7. Yield anomaly
    # ------------------------------------------------

    df["yield_anomaly"] = (
        (
            df["crop_yield"]
            -
            df["yield_baseline"]
        )
        /
        df["yield_baseline"].replace(
            0,
            np.nan
        )
    )

    df["yield_anomaly"] = (
        df["yield_anomaly"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # Keep the raw value, but cap extreme ratios (e.g. +6300%
    # caused by near-zero baselines) so they cannot dominate
    # outcome-aware retrieval or model training.
    df["yield_anomaly_raw"] = df["yield_anomaly"]

    df["yield_anomaly"] = df["yield_anomaly"].clip(-1, 3)


    # ------------------------------------------------
    # 8. Climate stress score (0-1)
    # ------------------------------------------------

    # Rainfall: |z| of 3 standard deviations = maximum stress.
    df["rainfall_stress"] = np.clip(
        np.abs(df["rainfall_anomaly_z"]) / 3,
        0,
        1
    )

    # Other variables: fractional anomaly, same rule as the
    # Streamlit app (a 100% deviation = maximum stress).
    for name in ["temperature", "humidity", "wind", "solar"]:

        df[f"{name}_stress"] = np.clip(
            np.abs(df[f"{name}_anomaly"]),
            0,
            1
        )


    # ------------------------------------------------
    # 9. Overall climate risk
    #
    # Weighted average of the stress components. Components
    # that never vary in the data (in the bundled dataset,
    # everything except rainfall is a static per-crop
    # reference value) carry no information, so they are
    # dropped and the remaining weights are renormalised.
    # Otherwise the score would be capped at the rainfall
    # weight (0.30) and could never reach High/Severe.
    # ------------------------------------------------

    weights = {
        "rainfall_stress": 0.30,
        "temperature_stress": 0.25,
        "humidity_stress": 0.15,
        "wind_stress": 0.15,
        "solar_stress": 0.15,
    }

    active = {
        col: w
        for col, w in weights.items()
        if df[col].std() > 1e-9
    }

    if not active:
        active = weights

    total_weight = sum(active.values())

    df["climate_risk"] = (
        sum(
            w * df[col]
            for col, w in active.items()
        )
        / total_weight
    ).clip(0, 1)


    # ------------------------------------------------
    # 10. Risk category
    # ------------------------------------------------

    df["risk_category"] = pd.cut(

        df["climate_risk"],

        bins=[
            -0.001,
            0.25,
            0.50,
            0.75,
            1.0
        ],

        labels=[
            "Low",
            "Moderate",
            "High",
            "Severe"
        ]
    ).astype(str)


    # ------------------------------------------------
    # 11. Nutrient balance
    # ------------------------------------------------

    if all(
        c in df.columns
        for c in [
            "n_req",
            "p_req",
            "k_req"
        ]
    ):

        df["total_npk_requirement"] = (
            df["n_req"]
            +
            df["p_req"]
            +
            df["k_req"]
        )


    # ------------------------------------------------
    # Case IDs
    # ------------------------------------------------

    df["case_id"] = [
        f"C{i:07d}"
        for i in range(1, len(df) + 1)
    ]

    return df


def preprocess(
    input_path,
    output_path=PROCESSED_DATA
):

    print("Loading dataset...")

    df = load_raw(input_path)

    print(
        f"Original dataset: "
        f"{df.shape[0]} rows × "
        f"{df.shape[1]} columns"
    )

    df = clean_data(df)

    print(
        f"After cleaning: "
        f"{df.shape[0]} rows"
    )

    df = add_climate_features(df)

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Processed dataset saved to:\n"
        f"{output_path}"
    )

    return df


if __name__ == "__main__":

    from config import RAW_DATA

    preprocess(
        RAW_DATA,
        PROCESSED_DATA
    )