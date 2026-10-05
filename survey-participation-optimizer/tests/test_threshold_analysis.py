import pandas as pd

from src.data.loader import (
    load_config,
    load_source_data,
)

from src.data.assembler import (
    build_modeling_dataset,
)

from src.data.validation import (
    create_temporal_split,
)

from src.features.propensity_features import (
    get_feature_columns,
)

from src.models.propensity import (
    get_propensity_models,
)

from src.evaluation.thresholds import (
    calculate_threshold_table,
)


def main():

    config = load_config()

    source_data = load_source_data()

    data = build_modeling_dataset(
        source_data
    )

    train, validation, test = (
        create_temporal_split(
            data,
            config,
        )
    )

    features = get_feature_columns()

    X_train = train[features]
    y_train = train["completed"]

    X_test = test[features]
    y_test = test["completed"]

    models = get_propensity_models()

    for model_name, model in models.items():

        print(
            f"\n{'=' * 70}"
        )

        print(
            f"THRESHOLD ANALYSIS: {model_name}"
        )

        print(
            f"{'=' * 70}"
        )

        model.fit(
            X_train,
            y_train,
        )

        probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        threshold_table = (
            calculate_threshold_table(
                y_test,
                probabilities,
            )
        )

        print(
            threshold_table.to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.4f}"
                ),
            )
        )

        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
        )

        threshold_table.to_csv(
            f"evaluation/"
            f"{safe_name}_thresholds.csv",
            index=False,
        )

        probability_summary = (
            pd.Series(probabilities)
            .describe(
                percentiles=[
                    0.10,
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                    0.95,
                    0.99,
                ]
            )
        )

        print(
            "\nProbability distribution"
        )

        print(
            probability_summary.to_string()
        )


if __name__ == "__main__":
    main()