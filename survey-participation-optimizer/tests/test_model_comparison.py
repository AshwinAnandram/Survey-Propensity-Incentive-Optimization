import time

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

from src.evaluation.classification import (
    evaluate_classifier,
)

from src.evaluation.ranking import (
    calculate_lift_table,
)


def main():

    print("Loading configuration...")

    config = load_config()

    print("Loading source data...")

    source_data = load_source_data()

    print("Building modeling dataset...")

    data = build_modeling_dataset(
        source_data
    )

    print("Creating temporal split...")

    train, validation, test = (
        create_temporal_split(
            data,
            config,
        )
    )

    features = get_feature_columns()

    X_train = train[features]
    y_train = train["completed"]

    X_validation = validation[features]
    y_validation = validation["completed"]

    X_test = test[features]
    y_test = test["completed"]

    models = get_propensity_models()

    results = []
    lift_tables = {}

    for model_name, model in models.items():

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"Training: {model_name}"
        )

        print(
            f"{'=' * 60}"
        )

        start_time = time.perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        training_time = (
            time.perf_counter()
            - start_time
        )

        start_time = time.perf_counter()

        validation_probabilities = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )

        test_probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        prediction_time = (
            time.perf_counter()
            - start_time
        )

        validation_metrics, _ = (
            evaluate_classifier(
                y_validation,
                validation_probabilities,
            )
        )

        test_metrics, _ = (
            evaluate_classifier(
                y_test,
                test_probabilities,
            )
        )

        lift_table = calculate_lift_table(
            y_test,
            test_probabilities,
        )

        top_decile_lift = (
            lift_table.iloc[0]["lift"]
        )

        top_30_gain = (
            lift_table.iloc[:3][
                "completions"
            ].sum()
            / y_test.sum()
        )

        results.append(
            {
                "model": model_name,
                "validation_roc_auc": (
                    validation_metrics["roc_auc"]
                ),
                "validation_pr_auc": (
                    validation_metrics["pr_auc"]
                ),
                "validation_log_loss": (
                    validation_metrics["log_loss"]
                ),
                "validation_brier": (
                    validation_metrics["brier_score"]
                ),
                "test_roc_auc": (
                    test_metrics["roc_auc"]
                ),
                "test_pr_auc": (
                    test_metrics["pr_auc"]
                ),
                "test_log_loss": (
                    test_metrics["log_loss"]
                ),
                "test_brier": (
                    test_metrics["brier_score"]
                ),
                "test_precision": (
                    test_metrics["precision"]
                ),
                "test_recall": (
                    test_metrics["recall"]
                ),
                "test_f1": (
                    test_metrics["f1"]
                ),
                "top_decile_lift": (
                    top_decile_lift
                ),
                "top_30_gain": (
                    top_30_gain
                ),
                "training_time_seconds": (
                    training_time
                ),
                "prediction_time_seconds": (
                    prediction_time
                ),
            }
        )

        lift_tables[model_name] = lift_table

        print(
            f"Training time: "
            f"{training_time:.4f}s"
        )

        print(
            f"Test ROC-AUC: "
            f"{test_metrics['roc_auc']:.4f}"
        )

        print(
            f"Test PR-AUC: "
            f"{test_metrics['pr_auc']:.4f}"
        )

        print(
            f"Test Log Loss: "
            f"{test_metrics['log_loss']:.4f}"
        )

        print(
            f"Test Brier Score: "
            f"{test_metrics['brier_score']:.4f}"
        )

        print(
            f"Test F1: "
            f"{test_metrics['f1']:.4f}"
        )

        print(
            f"Top Decile Lift: "
            f"{top_decile_lift:.4f}"
        )

        print(
            f"Top 30% Gain: "
            f"{top_30_gain:.4f}"
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            "test_roc_auc",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "MODEL COMPARISON"
    )

    print(
        "=" * 80
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    results_df.to_csv(
        "evaluation/model_comparison.csv",
        index=False,
    )

    for model_name, lift_table in (
        lift_tables.items()
    ):

        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
        )

        lift_table.to_csv(
            f"evaluation/"
            f"{safe_name}_lift.csv",
            index=False,
        )

    print(
        "\nModel comparison completed."
    )


if __name__ == "__main__":
    main()