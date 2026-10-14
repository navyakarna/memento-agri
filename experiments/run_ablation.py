import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import pandas as pd

from config import (
    PROCESSED_DATA,
    FAISS_INDEX,
    CASE_META,
    OUTPUT_DIR,
)

from memory.retriever import CaseRetriever


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def calculate_risk(row):
    """
    Simple deterministic climate-risk score.

    Used for the ablation experiment so that the
    experiment does not depend on an LLM response.
    """

    rainfall_anomaly = abs(
        safe_float(row.get("rainfall_anomaly"))
    )

    temperature_anomaly = abs(
        safe_float(row.get("temperature_anomaly"))
    )

    humidity_anomaly = abs(
        safe_float(row.get("humidity_anomaly"))
    )

    wind_anomaly = abs(
        safe_float(row.get("wind_anomaly"))
    )

    solar_anomaly = abs(
        safe_float(row.get("solar_anomaly"))
    )

    score = (
        0.30 * min(rainfall_anomaly, 1.0)
        +
        0.25 * min(temperature_anomaly, 1.0)
        +
        0.15 * min(humidity_anomaly, 1.0)
        +
        0.15 * min(wind_anomaly, 1.0)
        +
        0.15 * min(solar_anomaly, 1.0)
    )

    return max(
        0.0,
        min(1.0, score)
    )


def risk_category(score):

    if score < 0.25:
        return "Low"

    if score < 0.50:
        return "Moderate"

    if score < 0.75:
        return "High"

    return "Severe"


def run():

    print("=" * 60)
    print("MEMENTO-AGRI ABLATION STUDY")
    print("=" * 60)

    df = pd.read_csv(
        PROCESSED_DATA
    )

    print(
        f"Dataset size: {len(df):,} rows"
    )

    # --------------------------------------------------------
    # Test current Memento memory
    # --------------------------------------------------------

    memory_available = (
        Path(FAISS_INDEX).exists()
        and Path(CASE_META).exists()
    )

    if memory_available:

        try:

            retriever = CaseRetriever(
                index_path=FAISS_INDEX,
                metadata_path=CASE_META
            )

            print(
                "✅ Memento memory loaded."
            )

        except Exception as e:

            retriever = None

            print(
                "⚠️ Memento memory could not be loaded:"
            )

            print(e)

    else:

        retriever = None

        print(
            "⚠️ Memento memory files not found."
        )


    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline_scores = []

    for _, row in df.iterrows():

        score = calculate_risk(
            row.to_dict()
        )

        baseline_scores.append(
            score
        )


    # --------------------------------------------------------
    # Memento condition
    # --------------------------------------------------------

    memento_scores = []

    if retriever is not None:

        # We measure retrieval availability rather
        # than pretending retrieval changes a numeric
        # risk score.

        retrieval_count = 0

        sample_size = min(
            100,
            len(df)
        )

        for _, row in df.head(
            sample_size
        ).iterrows():

            query_case = row.to_dict()

            try:

                cases = retriever.retrieve(
                    query_case
                )

                if cases:
                    retrieval_count += len(cases)

            except Exception:
                pass

        retrieval_average = (
            retrieval_count / sample_size
            if sample_size
            else 0
        )

    else:

        retrieval_average = 0


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = {

        "experiment":
            "ablation",

        "dataset_rows":
            int(len(df)),

        "baseline":

            {
                "mean_risk":
                    float(
                        sum(baseline_scores)
                        /
                        max(
                            len(baseline_scores),
                            1
                        )
                    )
            },

        "memento":

            {
                "memory_available":
                    bool(
                        retriever is not None
                    ),

                "average_retrieved_cases":
                    float(
                        retrieval_average
                    )
            }
    }


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_dir = Path(
        OUTPUT_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        /
        "ablation_results.json"
    )

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )


    print()
    print("RESULTS")
    print("-" * 60)

    print(
        f"Baseline mean risk: "
        f"{results['baseline']['mean_risk']:.4f}"
    )

    print(
        f"Memento memory available: "
        f"{results['memento']['memory_available']}"
    )

    print(
        f"Average retrieved cases: "
        f"{results['memento']['average_retrieved_cases']:.2f}"
    )

    print()
    print(
        f"✅ Saved: {output_file}"
    )


if __name__ == "__main__":
    run()
