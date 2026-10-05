import matplotlib.pyplot as plt
import numpy as np

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
)


def plot_roc_curves(
    y_true,
    model_probabilities,
    output_path=None,
):
    plt.figure(figsize=(8, 6))

    for model_name, probabilities in (
        model_probabilities.items()
    ):
        false_positive_rate, true_positive_rate, _ = (
            roc_curve(
                y_true,
                probabilities,
            )
        )

        roc_auc = auc(
            false_positive_rate,
            true_positive_rate,
        )

        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=(
                f"{model_name} "
                f"(AUC={roc_auc:.3f})"
            ),
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve Comparison"
    )

    plt.legend()
    plt.grid(True)

    if output_path:
        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

    plt.show()


def plot_precision_recall_curves(
    y_true,
    model_probabilities,
    output_path=None,
):
    plt.figure(figsize=(8, 6))

    for model_name, probabilities in (
        model_probabilities.items()
    ):
        precision, recall, _ = (
            precision_recall_curve(
                y_true,
                probabilities,
            )
        )

        pr_auc = auc(
            recall,
            precision,
        )

        plt.plot(
            recall,
            precision,
            label=(
                f"{model_name} "
                f"(AUC={pr_auc:.3f})"
            ),
        )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Precision-Recall Curve Comparison"
    )

    plt.legend()
    plt.grid(True)

    if output_path:
        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

    plt.show()


def plot_calibration_curves(
    y_true,
    model_probabilities,
    output_path=None,
    n_bins=10,
):
    plt.figure(figsize=(8, 6))

    for model_name, probabilities in (
        model_probabilities.items()
    ):
        observed_rate, mean_probability = (
            calibration_curve(
                y_true,
                probabilities,
                n_bins=n_bins,
                strategy="quantile",
            )
        )

        plt.plot(
            mean_probability,
            observed_rate,
            marker="o",
            label=model_name,
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel(
        "Mean Predicted Probability"
    )

    plt.ylabel(
        "Observed Completion Rate"
    )

    plt.title(
        "Calibration Curve Comparison"
    )

    plt.legend()
    plt.grid(True)

    if output_path:
        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

    plt.show()