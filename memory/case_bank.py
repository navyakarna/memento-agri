"""Case bank: turns processed historical rows into Memento memory cases.

Each case records an observed situation (context), its climate risk and
the observed outcome. No intervention is attached, because the source
data does not record which interventions farmers used (see README).
"""

import json
import math
from pathlib import Path

import pandas as pd

CONTEXT_KEYS = [
    "state", "district", "crop", "season", "year", "soil_type",
    "rainfall_real_mm", "rainfall_anomaly", "temperature_anomaly",
    "humidity_anomaly", "wind_anomaly", "solar_anomaly",
    "irrigation_percentage",
]

OUTCOME_KEYS = ["crop_yield", "yield_baseline", "yield_anomaly"]


def _clean(value):
    """Convert numpy/pandas scalars to plain JSON-safe Python values."""
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _pick(row, keys):
    return {k: _clean(row.get(k)) for k in keys if k in row}


def describe_anomaly(value):
    """Bucket a fractional anomaly into words the text embedder can match."""
    if value is None:
        return "unknown"
    if value <= -0.30:
        return "severe deficit"
    if value <= -0.10:
        return "deficit"
    if value < 0.10:
        return "near normal"
    if value < 0.30:
        return "surplus"
    return "large surplus"


def case_text(context, risk_category=None):
    """Text used to embed a case (and, identically, a query)."""
    parts = [
        f"crop: {context.get('crop') or 'unknown'}",
        f"state: {context.get('state') or 'unknown'}",
        f"district: {context.get('district') or 'unknown'}",
    ]
    if context.get("season"):
        parts.append(f"season: {context['season']}")
    if context.get("soil_type"):
        parts.append(f"soil: {context['soil_type']}")
    parts.append(f"rainfall: {describe_anomaly(context.get('rainfall_anomaly'))}")
    parts.append(f"temperature: {describe_anomaly(context.get('temperature_anomaly'))}")
    if risk_category:
        parts.append(f"climate risk: {risk_category}")
    return " | ".join(parts)


def row_to_case(row):
    if isinstance(row, pd.Series):
        row = row.to_dict()
    context = _pick(row, CONTEXT_KEYS)
    risk = {
        "score": _clean(row.get("climate_risk")),
        "category": _clean(row.get("risk_category")),
    }
    case = {
        "case_id": str(row.get("case_id")),
        "context": context,
        "risk": risk,
        "outcome": _pick(row, OUTCOME_KEYS),
        "experience": None,
    }
    case["text"] = case_text(context, risk["category"])
    return case


def build_case_bank(df, output_path):
    """Build one case per processed row and save them as JSON."""
    cases = [row_to_case(row) for row in df.to_dict(orient="records")]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cases, f)
    print(f"Saved {len(cases)} cases -> {output_path}")
    return cases
