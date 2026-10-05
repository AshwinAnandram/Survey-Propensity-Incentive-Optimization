import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.data.assembler import build_modeling_dataset
from src.data.loader import load_config, load_source_data
from src.data.validation import create_temporal_split
from src.features.propensity_features import get_feature_columns
from src.models.propensity import create_hist_gradient_boosting_model


def calculate_lift_and_gain(y_true, probabilities, bins=10):
    evaluation_data = pd.DataFrame(
        {
            "actual": np.asarray(y_true),
            "probability": np.asarray(probabilities),
        }
    )

    evaluation_data = evaluation_data.sort_values(
        "probability",
        ascending=False,
    ).reset_index(drop=True)

    evaluation_data["decile"] = pd.qcut(
        evaluation_data.index,
        q=bins,
        labels=False,
        duplicates="drop",
    )

    overall_rate = evaluation_data["actual"].mean()

    grouped = (
        evaluation_data
        .groupby("decile", observed=False)
        .agg(
            observations=("actual", "size"),
            completions=("actual", "sum"),
            completion_rate=("actual", "mean"),
            mean_probability=("probability", "mean"),
        )
        .reset_index()
    )

    grouped["lift"] = (
        grouped["completion_rate"] / overall_rate
    )

    grouped["cumulative_completions"] = (
        grouped["completions"].cumsum()
    )

    total_completions = evaluation_data["actual"].sum()

    grouped["cumulative_gain"] = (
        grouped["cumulative_completions"]
        / total_completions
    )

    return grouped


def calculate_top_percentage_metrics(
    y_true,
    probabilities,
    percentage,
):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    n_selected = max(
        1,
        int(len(y_true) * percentage),
    )

    ranking = np.argsort(
        probabilities
    )[::-1]

    selected_indices = ranking[:n_selected]

    selected_actuals = y_true[selected_indices]

    selected_completion_rate = selected_actuals.mean()

    overall_completion_rate = y_true.mean()

    lift = (
        selected_completion_rate
        / overall_completion_rate
    )

    captured_completions = (
        selected_actuals.sum()
        / y_true.sum()
    )

    return {
        "selected_count": n_selected,
        "selected_pct": percentage,
        "completion_rate": selected_completion_rate,
        "lift": lift,
        "cumulative_gain": captured_completions,
    }


def main():
    source_data = load_source_data()
    data = build_modeling_dataset(source_data)

    config = load_config()

    train_data, validation_data, test_data = create_temporal_split(
        data,
        config,
    )

    features = get_feature_columns()

    X_train = train_data[features]
    y_train = train_data["completed"]

    X_test = test_data[features]
    y_test = test_data["completed"]

    model = create_hist_gradient_boosting_model()

    print("=" * 100)
    print("FINAL PROPENSITY MODEL — UNTOUCHED TEST EVALUATION")
    print("=" * 100)

    print()
    print("Dataset sizes")
    print(f"Training:    {len(train_data):,}")
    print(f"Validation:  {len(validation_data):,}")
    print(f"Test:        {len(test_data):,}")

    print()
    print("Date ranges")

    print(
        f"Training:    "
        f"{train_data['invitation_date'].min().date()} "
        f"to "
        f"{train_data['invitation_date'].max().date()}"
    )

    print(
        f"Validation:  "
        f"{validation_data['invitation_date'].min().date()} "
        f"to "
        f"{validation_data['invitation_date'].max().date()}"
    )

    print(
        f"Test:        "
        f"{test_data['invitation_date'].min().date()} "
        f"to "
        f"{test_data['invitation_date'].max().date()}"
    )

    print()
    print("Completion rates")

    print(
        f"Training:    "
        f"{y_train.mean():.4f}"
    )

    print(
        f"Validation:  "
        f"{validation_data['completed'].mean():.4f}"
    )

    print(
        f"Test:        "
        f"{y_test.mean():.4f}"
    )

    print()
    print("Training final selected HGB model...")

    start_time = time.time()

    model.fit(
        X_train,
        y_train,
    )

    train_time = time.time() - start_time

    print(
        f"Training completed in "
        f"{train_time:.2f} seconds."
    )

    print()
    print("Generating test predictions...")

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    test_log_loss = log_loss(
        y_test,
        probabilities,
    )

    brier = brier_score_loss(
        y_test,
        probabilities,
    )

    print()
    print("=" * 100)
    print("TEST CLASSIFICATION METRICS")
    print("=" * 100)

    print(
        f"ROC-AUC:       {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:        {pr_auc:.4f}"
    )

    print(
        f"Log Loss:      {test_log_loss:.4f}"
    )

    print(
        f"Brier Score:   {brier:.4f}"
    )

    print()
    print("Probability distribution")

    print(
        f"Mean:          {probabilities.mean():.4f}"
    )

    print(
        f"Median:        {np.median(probabilities):.4f}"
    )

    print(
        f"Minimum:       {probabilities.min():.4f}"
    )

    print(
        f"Maximum:       {probabilities.max():.4f}"
    )

    print()
    print("=" * 100)
    print("TOP-PERCENTAGE BUSINESS METRICS")
    print("=" * 100)

    top_metrics = []

    for percentage in [0.10, 0.20, 0.30, 0.50]:
        metrics = calculate_top_percentage_metrics(
            y_test,
            probabilities,
            percentage,
        )

        top_metrics.append(metrics)

        print()
        print(
            f"Top {int(percentage * 100)}%"
        )

        print(
            f"  Selected:          "
            f"{metrics['selected_count']:,}"
        )

        print(
            f"  Completion Rate:   "
            f"{metrics['completion_rate']:.4f}"
        )

        print(
            f"  Lift:              "
            f"{metrics['lift']:.4f}"
        )

        print(
            f"  Cumulative Gain:   "
            f"{metrics['cumulative_gain']:.4f}"
        )

    lift_table = calculate_lift_and_gain(
        y_test,
        probabilities,
    )

    print()
    print("=" * 100)
    print("DECILE LIFT TABLE")
    print("=" * 100)

    display_table = lift_table.copy()

    display_table["decile"] = (
        display_table["decile"] + 1
    )

    print(
        display_table.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()
    print("=" * 100)
    print("FINAL TEST SUMMARY")
    print("=" * 100)

    print(
        f"ROC-AUC:             {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:              {pr_auc:.4f}"
    )

    print(
        f"Log Loss:            {test_log_loss:.4f}"
    )

    print(
        f"Brier Score:         {brier:.4f}"
    )

    print(
        f"Top-Decile Lift:     "
        f"{top_metrics[0]['lift']:.4f}"
    )

    print(
        f"Top-30% Gain:        "
        f"{top_metrics[2]['cumulative_gain']:.4f}"
    )

    print()
    print(
        "The test set was used only for final evaluation."
    )

    output_dir = Path("evaluation")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        Path("models")
        / "propensity_hgb_test_evaluated.joblib"
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        model_path,
    )

    print()
    print(
        f"Model saved to: {model_path}"
    )

    results = pd.DataFrame(
        {
            "metric": [
                "ROC-AUC",
                "PR-AUC",
                "Log Loss",
                "Brier Score",
                "Top-Decile Lift",
                "Top-30% Gain",
            ],
            "test_value": [
                roc_auc,
                pr_auc,
                test_log_loss,
                brier,
                top_metrics[0]["lift"],
                top_metrics[2]["cumulative_gain"],
            ],
        }
    )

    results_path = (
        output_dir
        / "final_propensity_test_results.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    print(
        f"Results saved to: {results_path}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()