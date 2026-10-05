import time

import pandas as pd

from src.data.loader import load_source_data
from src.data.assembler import build_modeling_dataset
from src.data.validation import validate_modeling_dataset
from src.features.propensity_features import get_feature_columns
from src.models.propensity import (
    create_hist_gradient_boosting_model,
    create_logistic_regression_model,
)
from src.models.calibration import get_calibration_models
from src.evaluation.evaluation import (
    evaluate_classifier,
    calculate_lift_table,
)


def temporal_split(data):
    data = data.sort_values("invitation_date").reset_index(drop=True)

    train_cutoff = data["invitation_date"].quantile(0.70)
    validation_cutoff = data["invitation_date"].quantile(0.85)

    train = data[data["invitation_date"] <= train_cutoff].copy()
    validation = data[
        (data["invitation_date"] > train_cutoff)
        & (data["invitation_date"] <= validation_cutoff)
    ].copy()
    test = data[data["invitation_date"] > validation_cutoff].copy()

    return train, validation, test


def evaluate_model(name, model, X_train, y_train, X_validation, y_validation):
    start = time.perf_counter()

    model.fit(X_train, y_train)

    train_time = time.perf_counter() - start

    probabilities = model.predict_proba(X_validation)[:, 1]

    metrics, _ = evaluate_classifier(
        y_validation,
        probabilities,
        threshold=0.20,
    )

    lift_table = calculate_lift_table(
        y_validation,
        probabilities,
        bins=10,
    )

    top_decile_lift = lift_table.iloc[0]["lift"]
    top_30_gain = lift_table.iloc[2]["cumulative_gain"]

    return {
        "model": name,
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "log_loss": metrics["log_loss"],
        "brier_score": metrics["brier_score"],
        "top_decile_lift": top_decile_lift,
        "top_30_gain": top_30_gain,
        "train_time_seconds": train_time,
    }


def main():
    source_data = load_source_data()

    data = build_modeling_dataset(source_data)

    validate_modeling_dataset(data)

    train, validation, test = temporal_split(data)

    feature_columns = get_feature_columns()

    X_train = train[feature_columns]
    y_train = train["completed"]

    X_validation = validation[feature_columns]
    y_validation = validation["completed"]

    results = []

    baseline_hgb = create_hist_gradient_boosting_model()

    results.append(
        evaluate_model(
            "HGB Baseline",
            baseline_hgb,
            X_train,
            y_train,
            X_validation,
            y_validation,
        )
    )

    logistic = create_logistic_regression_model()

    results.append(
        evaluate_model(
            "Logistic Regression",
            logistic,
            X_train,
            y_train,
            X_validation,
            y_validation,
        )
    )

    calibration_models = get_calibration_models()

    for name, model in calibration_models.items():
        results.append(
            evaluate_model(
                name,
                model,
                X_train,
                y_train,
                X_validation,
                y_validation,
            )
        )

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 110)
    print("PHASE III CALIBRATION RESULTS")
    print("=" * 110)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


if __name__ == "__main__":
    main()