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
)


def run():

    print("=" * 60)
    print("MEMENTO-AGRI TEMPORAL SPLIT")
    print("=" * 60)

    df = pd.read_csv(
        PROCESSED_DATA
    )

    if "year" not in df.columns:

        raise ValueError(
            "Dataset does not contain a 'year' column."
        )


    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["year"]
    ).copy()

    df["year"] = df[
        "year"
    ].astype(int)


    min_year = int(
        df["year"].min()
    )

    max_year = int(
        df["year"].max()
    )


    # --------------------------------------------------------
    # Temporal split
    #
    # 70% earliest years -> train
    # 20% -> validation
    # 10% -> test
    # --------------------------------------------------------

    years = sorted(
        df["year"].unique()
    )

    n_years = len(
        years
    )

    train_end = int(
        n_years * 0.70
    )

    validation_end = int(
        n_years * 0.90
    )


    train_years = years[
        :train_end
    ]

    validation_years = years[
        train_end:validation_end
    ]

    test_years = years[
        validation_end:
    ]


    df["split"] = "test"

    df.loc[
        df["year"].isin(
            train_years
        ),
        "split"
    ] = "train"

    df.loc[
        df["year"].isin(
            validation_years
        ),
        "split"
    ] = "validation"


    # --------------------------------------------------------
    # Print split
    # --------------------------------------------------------

    split_summary = (
        df.groupby(
            "split"
        )["year"]
        .agg(
            [
                "min",
                "max",
                "count"
            ]
        )
        .reset_index()
    )


    print()
    print(
        split_summary
    )


    print()
    print(
        f"Train years: "
        f"{min(train_years)}–{max(train_years)}"
    )

    print(
        f"Validation years: "
        f"{min(validation_years)}–"
        f"{max(validation_years)}"
    )

    print(
        f"Test years: "
        f"{min(test_years)}–"
        f"{max(test_years)}"
    )


    # --------------------------------------------------------
    # Save split
    # --------------------------------------------------------

    output_dir = Path(
        OUTPUT_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    split_file = (
        output_dir
        /
        "temporal_split.csv"
    )

    df[
        [
            "year",
            "split"
        ]
    ].drop_duplicates().sort_values(
        "year"
    ).to_csv(
        split_file,
        index=False
    )


    metadata = {

        "min_year":
            min_year,

        "max_year":
            max_year,

        "train_start":
            int(min(train_years)),

        "train_end":
            int(max(train_years)),

        "validation_start":
            int(min(validation_years)),

        "validation_end":
            int(max(validation_years)),

        "test_start":
            int(min(test_years)),

        "test_end":
            int(max(test_years)),

        "train_rows":
            int(
                (
                    df["split"]
                    == "train"
                ).sum()
            ),

        "validation_rows":
            int(
                (
                    df["split"]
                    == "validation"
                ).sum()
            ),

        "test_rows":
            int(
                (
                    df["split"]
                    == "test"
                ).sum()
            )
    }


    metadata_file = (
        output_dir
        /
        "temporal_split_metadata.json"
    )


    with open(
        metadata_file,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )


    print()
    print(
        f"✅ Saved: {split_file}"
    )

    print(
        f"✅ Saved: {metadata_file}"
    )


if __name__ == "__main__":
    run()
