import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import pandas as pd

from config import (
    PROCESSED_DATA,
    OUTPUT_DIR,
    EMBEDDING_MODEL,
    FAISS_INDEX,
    CASE_META,
)

from memory.retriever import CaseRetriever


def run():

    print("=" * 60)
    print("MEMENTO-AGRI MEMORY GROWTH EXPERIMENT")
    print("=" * 60)

    df = pd.read_csv(
        PROCESSED_DATA
    )

    print(
        f"Dataset rows: {len(df):,}"
    )

    print(
        f"Embedding model: {EMBEDDING_MODEL}"
    )

    # --------------------------------------------------------
    # Evaluate memory at different sizes
    # --------------------------------------------------------

    memory_sizes = [
        100,
        500,
        1000,
        2500,
        5000
    ]

    memory_sizes = [
        size
        for size in memory_sizes
        if size <= len(df)
    ]

    if not memory_sizes:

        memory_sizes = [
            len(df)
        ]


    results = []


    for size in memory_sizes:

        subset = df.head(
            size
        )

        print()
        print(
            f"Memory size: {size:,}"
        )

        result = {

            "memory_size":
                int(size),

            "dataset_rows":
                int(len(subset)),

            "retriever_available":
                False,

            "retrieved_cases":
                0
        }


        # The currently built FAISS index represents
        # the project's active memory. We therefore
        # test its retrieval separately.

        if (
            Path(FAISS_INDEX).exists()
            and
            Path(CASE_META).exists()
        ):

            try:

                retriever = CaseRetriever(
                    index_path=FAISS_INDEX,
                    metadata_path=CASE_META
                )

                result[
                    "retriever_available"
                ] = True

                sample = subset.head(
                    min(20, len(subset))
                )

                total_retrieved = 0

                for _, row in sample.iterrows():

                    try:

                        cases = retriever.retrieve(
                            row.to_dict()
                        )

                        if cases:
                            total_retrieved += len(
                                cases
                            )

                    except Exception:
                        pass

                result[
                    "retrieved_cases"
                ] = int(
                    total_retrieved
                )

            except Exception as e:

                print(
                    f"⚠️ Retriever error: {e}"
                )


        results.append(
            result
        )

        print(
            f"Retrieved cases: "
            f"{result['retrieved_cases']}"
        )


    # --------------------------------------------------------
    # Save
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
        "memory_growth_results.json"
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
    print(
        "=" * 60
    )

    print(
        f"✅ Saved: {output_file}"
    )


if __name__ == "__main__":
    run()
