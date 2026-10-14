import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st
import pandas as pd


from config import (
    PROCESSED_DATA,
    FAISS_INDEX,
    CASE_META
)

from memory.retriever import CaseRetriever
from agent.agent import MementoAgriAgent


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Memento-Agri",
    page_icon="🌾",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🌾 Memento-Agri")

st.subheader(
    "Personalized Climate Advisory for Farmers"
)

st.write(
    """
    Memento-Agri uses historical crop-climate experiences
    as external memory to generate localized agricultural
    climate advisories without fine-tuning the LLM weights.
    """
)


# ============================================================
# LOAD AGENT
# ============================================================

@st.cache_resource
def load_agent():

    retriever = CaseRetriever(
        index_path=FAISS_INDEX,
        metadata_path=CASE_META
    )

    agent = MementoAgriAgent(
        retriever=retriever
    )

    return agent


# ============================================================
# CHECK PROCESSED DATASET
# ============================================================

if not Path(PROCESSED_DATA).exists():

    st.error(
        "❌ Processed dataset not found."
    )

    st.info(
        "Build the processed dataset first:"
    )

    st.code(
        "python main.py build",
        language="bash"
    )

    st.stop()


# ============================================================
# CHECK MEMENTO MEMORY
# ============================================================

if not Path(FAISS_INDEX).exists():

    st.error(
        "❌ Memento memory has not been built yet."
    )

    st.info(
        "Build the FAISS memory first:"
    )

    st.code(
        "python main.py build-memory",
        language="bash"
    )

    st.stop()


# ============================================================
# LOAD DATASET
# ============================================================

try:

    df = pd.read_csv(
        PROCESSED_DATA
    )

except Exception as e:

    st.error(
        "❌ Could not load processed dataset."
    )

    st.exception(e)

    st.stop()


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "state",
    "district",
    "crop",
    "year",
    "temperature",
    "humidity",
    "rainfall",
    "ph",
    "wind_speed",
    "solar_radiation"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    st.error(
        "❌ Required columns are missing from "
        "the processed dataset."
    )

    st.write(
        "Missing columns:"
    )

    st.code(
        ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# LOAD AGENT
# ============================================================

try:

    agent = load_agent()

except Exception as e:

    st.error(
        "❌ Could not load Memento-Agri agent."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🌱 Farm Information"
)


# ------------------------------------------------------------
# STATE
# ------------------------------------------------------------

states = sorted(
    df["state"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


if not states:

    st.error(
        "No states were found in the dataset."
    )

    st.stop()


state = st.sidebar.selectbox(
    "State",
    states
)


# ------------------------------------------------------------
# DISTRICT
# ------------------------------------------------------------

districts = sorted(
    df.loc[
        df["state"].astype(str) == state,
        "district"
    ]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


if not districts:

    st.warning(
        "No districts found for this state."
    )

    district = "Unknown"

else:

    district = st.sidebar.selectbox(
        "District",
        districts
    )


# ------------------------------------------------------------
# CROP
# ------------------------------------------------------------

crops = sorted(
    df["crop"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


if not crops:

    st.error(
        "No crops were found in the dataset."
    )

    st.stop()


crop = st.sidebar.selectbox(
    "Crop",
    crops
)


# ============================================================
# CURRENT CLIMATE CONDITIONS
# ============================================================

st.sidebar.header(
    "🌦️ Current Conditions"
)


temperature = st.sidebar.number_input(
    "Temperature (°C)",
    min_value=-20.0,
    max_value=60.0,
    value=28.0,
    step=0.1
)


humidity = st.sidebar.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=65.0,
    step=1.0
)


rainfall = st.sidebar.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=10000.0,
    value=800.0,
    step=10.0
)


ph = st.sidebar.number_input(
    "Soil pH",
    min_value=0.0,
    max_value=14.0,
    value=6.5,
    step=0.1
)


wind_speed = st.sidebar.number_input(
    "Wind Speed (m/s)",
    min_value=0.0,
    max_value=100.0,
    value=2.0,
    step=0.1
)


solar_radiation = st.sidebar.number_input(
    "Solar Radiation (MJ/m²/day)",
    min_value=0.0,
    max_value=100.0,
    value=18.0,
    step=0.1
)


# ============================================================
# FARMER QUESTION
# ============================================================

query = st.text_area(
    "💬 What would you like advice about?",
    placeholder=(
        "Example: Rainfall has been lower than usual. "
        "What should I do to reduce climate-related crop risk?"
    ),
    height=120
)


# ============================================================
# HISTORICAL BASELINE
# ============================================================

historical = df[
    (df["state"].astype(str) == state)
    &
    (df["district"].astype(str) == district)
    &
    (df["crop"].astype(str) == crop)
]


# If exact district/crop history isn't available,
# fall back to state + crop.

if len(historical) < 5:

    historical = df[
        (df["state"].astype(str) == state)
        &
        (df["crop"].astype(str) == crop)
    ]


# If still insufficient, fall back to crop history.

if len(historical) < 5:

    historical = df[
        df["crop"].astype(str) == crop
    ]


# ============================================================
# HELPER FOR SAFE MEAN
# ============================================================

def safe_mean(dataframe, column, default=0.0):

    if column not in dataframe.columns:
        return default

    values = pd.to_numeric(
        dataframe[column],
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        return default

    return float(
        values.mean()
    )


# ============================================================
# HISTORICAL CLIMATE BASELINES
# ============================================================

baseline_temperature = safe_mean(
    historical,
    "temperature",
    temperature
)

baseline_humidity = safe_mean(
    historical,
    "humidity",
    humidity
)

baseline_rainfall = safe_mean(
    historical,
    "rainfall",
    rainfall
)

baseline_wind = safe_mean(
    historical,
    "wind_speed",
    wind_speed
)

baseline_solar = safe_mean(
    historical,
    "solar_radiation",
    solar_radiation
)

baseline_ph = safe_mean(
    historical,
    "ph",
    ph
)


# ============================================================
# CALCULATE CLIMATE ANOMALIES
# ============================================================

def percentage_anomaly(
    current,
    baseline
):

    if baseline is None:
        return 0.0

    try:
        baseline = float(baseline)
        current = float(current)
    except (
        TypeError,
        ValueError
    ):
        return 0.0

    if abs(baseline) < 1e-9:
        return 0.0

    return (
        current - baseline
    ) / abs(baseline)


rainfall_anomaly = percentage_anomaly(
    rainfall,
    baseline_rainfall
)

temperature_anomaly = percentage_anomaly(
    temperature,
    baseline_temperature
)

humidity_anomaly = percentage_anomaly(
    humidity,
    baseline_humidity
)

wind_anomaly = percentage_anomaly(
    wind_speed,
    baseline_wind
)

solar_anomaly = percentage_anomaly(
    solar_radiation,
    baseline_solar
)

ph_anomaly = percentage_anomaly(
    ph,
    baseline_ph
)


# ============================================================
# CLIMATE STRESS SCORES
# ============================================================

def stress_score(
    anomaly
):

    try:
        anomaly = abs(
            float(anomaly)
        )
    except (
        TypeError,
        ValueError
    ):
        return 0.0

    # Convert anomaly into 0–1 stress score.
    return min(
        anomaly,
        1.0
    )


rainfall_stress = stress_score(
    rainfall_anomaly
)

temperature_stress = stress_score(
    temperature_anomaly
)

humidity_stress = stress_score(
    humidity_anomaly
)

wind_stress = stress_score(
    wind_anomaly
)

solar_stress = stress_score(
    solar_anomaly
)


# ============================================================
# CLIMATE RISK SCORE
# ============================================================

climate_risk = (
    0.30 * rainfall_stress
    +
    0.25 * temperature_stress
    +
    0.15 * humidity_stress
    +
    0.15 * wind_stress
    +
    0.15 * solar_stress
)


climate_risk = max(
    0.0,
    min(
        1.0,
        climate_risk
    )
)


# ============================================================
# RISK CATEGORY
# ============================================================

if climate_risk < 0.25:

    risk_category = "Low"

elif climate_risk < 0.50:

    risk_category = "Moderate"

elif climate_risk < 0.75:

    risk_category = "High"

else:

    risk_category = "Severe"


# ============================================================
# DISPLAY CURRENT RISK BEFORE GENERATING ADVICE
# ============================================================

st.divider()

st.subheader(
    "📊 Climate Risk Assessment"
)


risk_col1, risk_col2, risk_col3 = st.columns(3)


with risk_col1:

    st.metric(
        "Climate Risk",
        f"{climate_risk:.1%}"
    )


with risk_col2:

    st.metric(
        "Risk Category",
        risk_category
    )


with risk_col3:

    st.metric(
        "Historical Cases",
        len(historical)
    )


# ============================================================
# CURRENT VS HISTORICAL CONDITIONS
# ============================================================

with st.expander(
    "🔎 View Climate Anomalies"
):

    anomaly_col1, anomaly_col2 = st.columns(2)


    with anomaly_col1:

        st.write(
            "**Rainfall**"
        )

        st.write(
            f"Current: {rainfall:.2f} mm"
        )

        st.write(
            f"Historical average: "
            f"{baseline_rainfall:.2f} mm"
        )

        st.write(
            f"Anomaly: "
            f"{rainfall_anomaly:+.1%}"
        )


        st.write(
            "**Temperature**"
        )

        st.write(
            f"Current: {temperature:.2f} °C"
        )

        st.write(
            f"Historical average: "
            f"{baseline_temperature:.2f} °C"
        )

        st.write(
            f"Anomaly: "
            f"{temperature_anomaly:+.1%}"
        )


        st.write(
            "**Humidity**"
        )

        st.write(
            f"Current: {humidity:.2f}%"
        )

        st.write(
            f"Historical average: "
            f"{baseline_humidity:.2f}%"
        )

        st.write(
            f"Anomaly: "
            f"{humidity_anomaly:+.1%}"
        )


    with anomaly_col2:

        st.write(
            "**Wind Speed**"
        )

        st.write(
            f"Current: {wind_speed:.2f} m/s"
        )

        st.write(
            f"Historical average: "
            f"{baseline_wind:.2f} m/s"
        )

        st.write(
            f"Anomaly: "
            f"{wind_anomaly:+.1%}"
        )


        st.write(
            "**Solar Radiation**"
        )

        st.write(
            f"Current: "
            f"{solar_radiation:.2f} MJ/m²/day"
        )

        st.write(
            f"Historical average: "
            f"{baseline_solar:.2f} MJ/m²/day"
        )

        st.write(
            f"Anomaly: "
            f"{solar_anomaly:+.1%}"
        )


        st.write(
            "**Soil pH**"
        )

        st.write(
            f"Current: {ph:.2f}"
        )

        st.write(
            f"Historical average: "
            f"{baseline_ph:.2f}"
        )

        st.write(
            f"Anomaly: "
            f"{ph_anomaly:+.1%}"
        )


# ============================================================
# BUILD QUERY CASE
# ============================================================

row = {

    "state": state,

    "district": district,

    "crop": crop,

    "year": int(
        pd.to_numeric(
            df["year"],
            errors="coerce"
        ).max()
    ),

    # Current conditions
    "temperature": temperature,

    "humidity": humidity,

    "rainfall": rainfall,

    "ph": ph,

    "wind_speed": wind_speed,

    "solar_radiation": solar_radiation,

    # Historical baseline
    "baseline_temperature":
        baseline_temperature,

    "baseline_humidity":
        baseline_humidity,

    "baseline_rainfall":
        baseline_rainfall,

    "baseline_wind_speed":
        baseline_wind,

    "baseline_solar_radiation":
        baseline_solar,

    "baseline_ph":
        baseline_ph,

    # Anomalies
    "rainfall_anomaly":
        rainfall_anomaly,

    "temperature_anomaly":
        temperature_anomaly,

    "humidity_anomaly":
        humidity_anomaly,

    "wind_anomaly":
        wind_anomaly,

    "solar_anomaly":
        solar_anomaly,

    "ph_anomaly":
        ph_anomaly,

    # Stress
    "rainfall_stress":
        rainfall_stress,

    "temperature_stress":
        temperature_stress,

    "humidity_stress":
        humidity_stress,

    "wind_stress":
        wind_stress,

    "solar_stress":
        solar_stress,

    # Overall risk
    "climate_risk":
        climate_risk,

    "risk_category":
        risk_category
}


# ============================================================
# GENERATE ADVISORY
# ============================================================

if st.button(
    "🌾 Generate Advisory",
    type="primary",
    use_container_width=True
):

    if not query.strip():

        st.warning(
            "Please enter your question."
        )

        st.stop()


    with st.spinner(
        "🌾 Memento-Agri is analyzing "
        "historical experiences..."
    ):

        try:

            result = agent.advise(
                query,
                row
            )

        except Exception as e:

            st.error(
                "❌ Error generating advisory."
            )

            st.exception(e)

            st.stop()


    # ========================================================
    # DISPLAY ADVISORY
    # ========================================================

    st.header(
        "🌱 Memento-Agri Advisory"
    )


    if isinstance(
        result,
        dict
    ):

        # ----------------------------------------------------
        # Advice
        # ----------------------------------------------------

        advice = result.get(
            "advice"
        )

        if advice is None:

            advice = result.get(
                "recommendation"
            )

        if advice is None:

            advice = result.get(
                "advisory"
            )

        if advice is None:

            advice = str(
                result
            )


        st.markdown(
            str(advice)
        )


        # ----------------------------------------------------
        # Risk
        # ----------------------------------------------------

        result_risk = result.get(
            "risk"
        )


        if isinstance(
            result_risk,
            dict
        ):

            result_score = result_risk.get(
                "score",
                climate_risk
            )

            result_category = result_risk.get(
                "category",
                risk_category
            )


        elif result_risk is not None:

            try:

                result_score = float(
                    result_risk
                )

            except (
                TypeError,
                ValueError
            ):

                result_score = climate_risk


            result_category = result.get(
                "risk_category",
                risk_category
            )


        else:

            result_score = climate_risk

            result_category = result.get(
                "risk_category",
                risk_category
            )


        # Safely convert risk score

        try:

            result_score = float(
                result_score
            )

        except (
            TypeError,
            ValueError
        ):

            result_score = climate_risk


        result_score = max(
            0.0,
            min(
                1.0,
                result_score
            )
        )


        # ----------------------------------------------------
        # Risk Metrics
        # ----------------------------------------------------

        st.subheader(
            "⚠️ Advisory Risk"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Climate Risk",
                f"{result_score:.1%}"
            )


        with col2:

            st.metric(
                "Risk Category",
                str(result_category)
            )


        # ----------------------------------------------------
        # Retrieved Cases
        # ----------------------------------------------------

        retrieved_cases = result.get(
            "cases",
            result.get(
                "retrieved_cases",
                []
            )
        )


        if retrieved_cases is None:

            retrieved_cases = []


        if isinstance(
            retrieved_cases,
            list
        ) and len(retrieved_cases) > 0:

            st.subheader(
                "🧠 Memento Historical Memory"
            )

            st.write(
                f"{len(retrieved_cases)} "
                "similar historical case(s) "
                "were retrieved."
            )


            with st.expander(
                "View retrieved cases"
            ):

                for i, case in enumerate(
                    retrieved_cases[:5],
                    start=1
                ):

                    st.markdown(
                        f"**Historical Case {i}**"
                    )

                    if isinstance(
                        case,
                        dict
                    ):

                        display_case = {
                            key: value
                            for key, value in case.items()
                            if key not in [
                                "embedding",
                                "vector"
                            ]
                        }

                        st.json(
                            display_case
                        )

                    else:

                        st.write(
                            case
                        )


                    st.divider()


    else:

        st.markdown(
            str(result)
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Memento-Agri | Historical crop-climate memory "
    "+ agentic reasoning + personalized climate advisory. "
    "LLM weights remain unchanged."
)