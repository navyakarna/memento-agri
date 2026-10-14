"""
Real historical climate integration.

Research finding (documented in the Memento-Agri paper):
The source Kaggle "Indian Historical Crop Yield and Weather" dataset
assigns exactly ONE fixed (temperature, humidity, pH, rainfall, wind
speed, solar radiation) profile per crop, repeated identically across
every district and every year. These are crop agro-climatic
*requirement/reference* values, not real historical weather
observations. Consequently, any anomaly computed against a
(district, crop) or (state, crop) baseline from these columns is
identically zero for every row, which silently degenerates the
climate-risk classifier (single class) and removes all predictive
signal from the yield regressor's climate-anomaly features.

This module corrects the RAINFALL component (the highest-weighted
term in the climate-risk formula) by merging genuine year-by-year,
region-by-region historical rainfall from the India Meteorological
Department (IMD) sub-divisional rainfall series (1901-2015), sourced
via https://github.com/chandanverma07/DataSets
("rainfall in india 1901-2015.csv"), itself derived from IMD public
data. Temperature/humidity/pH/wind/solar remain the dataset's static
per-crop reference values; this is documented as a residual
limitation in the paper (Section Limitations) rather than silently
fixed with fabricated data.

Mapping: our dataset's 20 state names are mapped to one or more IMD
sub-divisions (documented below) and averaged per year when a state
spans multiple sub-divisions. IMD coverage ends in 2015; 2016-2017
(the last two years of our 2012-2017 test window) are filled with the
2015 value per state, flagged in the output via `rainfall_real_is_extrapolated`.
"""

from pathlib import Path
import pandas as pd

REAL_RAINFALL_CSV = Path(__file__).resolve().parent / "external" / "india_rainfall_1901_2015.csv"

# State (as it appears in this project's dataset) -> list of IMD sub-divisions to average.
STATE_TO_SUBDIVISION = {
    "Andhra Pradesh": ["COASTAL ANDHRA PRADESH", "RAYALSEEMA"],
    "Assam": ["ASSAM & MEGHALAYA"],
    "Bihar": ["BIHAR"],
    "Chhattisgarh": ["CHHATTISGARH"],
    "Gujarat": ["GUJARAT REGION", "SAURASHTRA & KUTCH"],
    "Haryana": ["HARYANA DELHI & CHANDIGARH"],
    "Himachal Pradesh": ["HIMACHAL PRADESH"],
    "Jharkhand": ["JHARKHAND"],
    "Karnataka": ["COASTAL KARNATAKA", "NORTH INTERIOR KARNATAKA", "SOUTH INTERIOR KARNATAKA"],
    "Kerala": ["KERALA"],
    "Madhya Pradesh": ["WEST MADHYA PRADESH", "EAST MADHYA PRADESH"],
    "Maharashtra": ["MADHYA MAHARASHTRA", "MATATHWADA", "VIDARBHA", "KONKAN & GOA"],
    "Orissa": ["ORISSA"],
    "Punjab": ["PUNJAB"],
    "Rajasthan": ["WEST RAJASTHAN", "EAST RAJASTHAN"],
    "Tamil Nadu": ["TAMIL NADU"],
    "Telangana": ["TELANGANA"],
    "Uttar Pradesh": ["EAST UTTAR PRADESH", "WEST UTTAR PRADESH"],
    "Uttarakhand": ["UTTARAKHAND"],
    "West Bengal": ["GANGETIC WEST BENGAL", "SUB HIMALAYAN WEST BENGAL & SIKKIM"],
}


def _load_state_year_rainfall():
    raw = pd.read_csv(REAL_RAINFALL_CSV)
    raw["SUBDIVISION"] = raw["SUBDIVISION"].str.strip()

    rows = []
    for state, subdivisions in STATE_TO_SUBDIVISION.items():
        sub = raw[raw["SUBDIVISION"].isin(subdivisions)]
        by_year = sub.groupby("YEAR")["ANNUAL"].mean()
        for year, value in by_year.items():
            rows.append({"state": state, "year": int(year), "rainfall_real_mm": float(value)})

    table = pd.DataFrame(rows)

    # Extend 2016-2017 (beyond IMD series) using each state's last available (2015) value.
    max_year = table["year"].max()
    extended = []
    for state in table["state"].unique():
        last_row = table[(table["state"] == state) & (table["year"] == max_year)]
        if last_row.empty:
            continue
        last_value = last_row["rainfall_real_mm"].iloc[0]
        for future_year in range(max_year + 1, 2018):
            extended.append({
                "state": state, "year": future_year,
                "rainfall_real_mm": last_value, "extrapolated": True
            })

    table["extrapolated"] = False
    table = pd.concat([table, pd.DataFrame(extended)], ignore_index=True)
    return table


def attach_real_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    """Merge genuine historical (state, year) rainfall onto the dataframe.

    Adds:
      rainfall_real_mm            - real IMD annual rainfall for that state/year
      rainfall_real_is_extrapolated - True for 2016-2017 (carried-forward 2015 value)
    """
    df = df.copy()
    table = _load_state_year_rainfall()
    df = df.merge(
        table.rename(columns={"extrapolated": "rainfall_real_is_extrapolated"}),
        on=["state", "year"], how="left"
    )
    missing = df["rainfall_real_mm"].isna().sum()
    if missing:
        # Fall back to the crop's static reference rainfall only where no
        # real regional series is available (should be rare / zero given
        # the mapping above covers all 20 states in this dataset).
        df["rainfall_real_mm"] = df["rainfall_real_mm"].fillna(df["rainfall"])
        df["rainfall_real_is_extrapolated"] = df["rainfall_real_is_extrapolated"].fillna(True)
    return df
