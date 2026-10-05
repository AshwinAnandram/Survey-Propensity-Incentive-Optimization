import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(
    y_true,
    probabilities,
    threshold=0.50,
):
    probabilities = np.asarray(probabilities)
    y_true = np.asarray(y_true)

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
    }

    matrix = confusion_matrix(
        y_true,
        predictions,
    )

    return metrics, matrix


def calculate_lift_table(
    y_true,
    probabilities,
    bins=10,
):
    data = pd.DataFrame(
        {
            "actual": np.asarray(y_true),
            "probability": np.asarray(
                probabilities
            ),
        }
    )

    data = data.sort_values(
        "probability",
        ascending=False,
    ).reset_index(drop=True)

    data["decile"] = pd.qcut(
        data.index,
        q=bins,
        labels=False,
        duplicates="drop",
    ) + 1

    overall_rate = data["actual"].mean()

    result = (
        data
        .groupby(
            "decile",
            observed=True,
        )
        .agg(
            observations=("actual", "size"),
            completions=("actual", "sum"),
            completion_rate=("actual", "mean"),
            mean_probability=(
                "probability",
                "mean",
            ),
        )
        .reset_index()
    )

    result["lift"] = (
        result["completion_rate"]
        / overall_rate
    )

    result["cumulative_completions"] = (
        result["completions"].cumsum()
    )

    total_completions = (
        result["completions"].sum()
    )

    result["cumulative_gain"] = (
        result["cumulative_completions"]
        / total_completions
    )

    return result


def calculate_threshold_table(
    y_true,
    probabilities,
    thresholds=None,
):
    if thresholds is None:
        thresholds = np.arange(
            0.10,
            0.55,
            0.05,
        )

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    rows = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        selected = predictions.sum()

        rows.append(
            {
                "threshold": threshold,
                "selected_count": selected,
                "selected_pct": (
                    selected
                    / len(predictions)
                ),
                "precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
            }
        )

    return pd.DataFrame(rows)


def calculate_probability_summary(
    probabilities,
):
    probabilities = np.asarray(
        probabilities
    )

    return (
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