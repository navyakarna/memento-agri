from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MEMORY_DIR = ROOT / "memory"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

for p in [RAW_DIR, PROCESSED_DIR, MEMORY_DIR, MODEL_DIR, OUTPUT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

RAW_DATA = RAW_DIR / "master_dataset.csv"

PROCESSED_DATA = PROCESSED_DIR / "master_processed.csv"

CASE_BANK = MEMORY_DIR / "cases.json"

FAISS_INDEX = MEMORY_DIR / "cases.faiss"

CASE_META = MEMORY_DIR / "case_metadata.json"

RANDOM_STATE = 42

TOP_K = 4

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4.1-mini"
)

# Canonical names -> accepted aliases found in common CSVs.
COLUMN_ALIASES = {
    "district": ["district", "dist_name", "dist", "district_name"],
    "state": ["state", "state_name"],
    "year": ["year", "yr"],
    "crop": ["crop", "crop_name"],
    "season": ["season"],
    "crop_area": ["crop_area", "area", "crop_area_ha"],
    "crop_production": ["crop_production", "production"],
    "crop_yield": ["crop_yield", "yield", "yield_t_ha"],
    "rainfall": ["rainfall", "annual_rainfall", "rain"],
    "normal_rainfall": ["normal_rainfall", "rainfall_normal"],
    "tmax": ["tmax", "max_temperature", "temperature_max"],
    "tmin": ["tmin", "min_temperature", "temperature_min"],
    "ppt": ["ppt", "precipitation"],
    "aet": ["aet", "actual_evapotranspiration"],
    "pet": ["pet", "potential_evapotranspiration"],
    "def": ["def", "water_deficit", "climatic_water_deficit"],
    "runoff": ["runoff", "q"],
    "wind_speed": ["wind_speed", "ws"],
    "irrigation_percentage": [
        "irrigation_percentage", "irrigated_percentage", "irrigation_pct"
    ],
    "soil_type": ["soil_type", "dominant_soil_type", "soil"],
}

REQUIRED_FOR_CORE = ["district", "state", "year", "crop", "crop_yield"]
