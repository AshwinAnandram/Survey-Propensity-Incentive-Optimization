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

from src.models.tuning import (
    tune_hist_gradient_boosting,
    tune_logistic_regression,
)


def print_search_results(
    model_name,
    search,
):

    print(
        f"\n{'=' * 70}"
    )

    print(
        f"{model_name} TUNING RESULTS"
    )

    print(
        f"{'=' * 70}"
    )

    print(
        f"Candidates evaluated: "
        f"{len(search.cv_results_['params'])}"
    )

    print(
        f"Best CV ROC-AUC: "
        f"{search.best_score_:.4f}"
    )

    print(
        "\nBest parameters:"
    )

    for parameter, value in (
        search.best_params_.items()
    ):
        print(
            f"{parameter}: {value}"
        )


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

    print(
        "\nTuning Logistic Regression..."
    )

    start = time.perf_counter()

    logistic_search = (
        tune_logistic_regression(
            X_train,
            y_train,
        )
    )

    logistic_time = (
        time.perf_counter()
        - start
    )

    print_search_results(
        "Logistic Regression",
        logistic_search,
    )

    print(
        f"\nTraining time: "
        f"{logistic_time:.2f}s"
    )

    print(
        "\nTuning Hist Gradient Boosting..."
    )

    start = time.perf_counter()

    hist_search = (
        tune_hist_gradient_boosting(
            X_train,
            y_train,
        )
    )

    hist_time = (
        time.perf_counter()
        - start
    )

    print_search_results(
        "Hist Gradient Boosting",
        hist_search,
    )

    print(
        f"\nTraining time: "
        f"{hist_time:.2f}s"
    )

    results = pd.DataFrame(
        [
            {
                "model": (
                    "Logistic Regression"
                ),
                "best_cv_roc_auc": (
                    logistic_search
                    .best_score_
                ),
                "training_time_seconds": (
                    logistic_time
                ),
                "best_params": (
                    logistic_search
                    .best_params_
                ),
            },
            {
                "model": (
                    "Hist Gradient Boosting"
                ),
                "best_cv_roc_auc": (
                    hist_search
                    .best_score_
                ),
                "training_time_seconds": (
                    hist_time
                ),
                "best_params": (
                    hist_search
                    .best_params_
                ),
            },
        ]
    )

    results.to_csv(
        "evaluation/"
        "phase2_tuning_summary.csv",
        index=False,
    )

    logistic_cv_results = (
        pd.DataFrame(
            logistic_search.cv_results_
        )
    )

    logistic_cv_results.to_csv(
        "evaluation/"
        "logistic_regression_cv_results.csv",
        index=False,
    )

    hist_cv_results = (
        pd.DataFrame(
            hist_search.cv_results_
        )
    )

    hist_cv_results.to_csv(
        "evaluation/"
        "hist_gradient_boosting_cv_results.csv",
        index=False,
    )

    print(
        "\nPhase II tuning completed."
    )


if __name__ == "__main__":
    main()