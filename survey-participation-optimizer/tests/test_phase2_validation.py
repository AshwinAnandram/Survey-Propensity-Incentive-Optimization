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
    create_hist_gradient_boosting_model,
    create_logistic_regression_model,
)

from src.models.tuning import (
    tune_hist_gradient_boosting,
    tune_logistic_regression,
)

from src.evaluation.evaluation import (
    calculate_lift_table,
    evaluate_classifier,
)


def evaluate_model(
    model,
    X,
    y,
):

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    metrics, _ = evaluate_classifier(
        y,
        probabilities,
    )

    lift_table = calculate_lift_table(
        y,
        probabilities,
    )

    return metrics, lift_table


def main():

    print(
        "Loading configuration..."
    )

    config = load_config()

    print(
        "Loading source data..."
    )

    source_data = load_source_data()

    print(
        "Building modeling dataset..."
    )

    data = build_modeling_dataset(
        source_data
    )

    print(
        "Creating temporal split..."
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

    X_validation = validation[features]
    y_validation = validation["completed"]

    print(
        "\nTraining baseline Logistic Regression..."
    )

    baseline_logistic = (
        create_logistic_regression_model()
    )

    baseline_logistic.fit(
        X_train,
        y_train,
    )

    print(
        "\nTraining baseline Hist Gradient Boosting..."
    )

    baseline_hist = (
        create_hist_gradient_boosting_model()
    )

    baseline_hist.fit(
        X_train,
        y_train,
    )

    print(
        "\nTuning Logistic Regression..."
    )

    logistic_search = (
        tune_logistic_regression(
            X_train,
            y_train,
        )
    )

    print(
        "\nTuning Hist Gradient Boosting..."
    )

    hist_search = (
        tune_hist_gradient_boosting(
            X_train,
            y_train,
        )
    )

    models = {
        "Logistic Regression Baseline": (
            baseline_logistic
        ),
        "Logistic Regression Tuned": (
            logistic_search.best_estimator_
        ),
        "Hist Gradient Boosting Baseline": (
            baseline_hist
        ),
        "Hist Gradient Boosting Tuned": (
            hist_search.best_estimator_
        ),
    }

    results = []

    for model_name, model in models.items():

        print(
            f"\nEvaluating {model_name}..."
        )

        metrics, lift_table = (
            evaluate_model(
                model,
                X_validation,
                y_validation,
            )
        )

        top_decile_lift = (
            lift_table.iloc[0]["lift"]
        )

        top_30_gain = (
            lift_table.iloc[:3][
                "completions"
            ].sum()
            / y_validation.sum()
        )

        results.append(
            {
                "model": model_name,
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "log_loss": metrics["log_loss"],
                "brier_score": metrics[
                    "brier_score"
                ],
                "precision": metrics[
                    "precision"
                ],
                "recall": metrics[
                    "recall"
                ],
                "f1": metrics["f1"],
                "top_decile_lift": (
                    top_decile_lift
                ),
                "top_30_gain": (
                    top_30_gain
                ),
            }
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            "roc_auc",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "PHASE II VALIDATION COMPARISON"
    )

    print(
        "=" * 100
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    results_df.to_csv(
        "evaluation/"
        "phase2_validation_comparison.csv",
        index=False,
    )

    print(
        "\nPhase II validation comparison completed."
    )


if __name__ == "__main__":
    main()