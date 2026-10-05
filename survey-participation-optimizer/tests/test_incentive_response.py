import time
from pathlib import Path

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

from src.features.incentive_response_features import (
    get_incentive_response_features,
    prepare_incentive_response_data,
)

from src.models.incentive_response import (
    get_incentive_response_models,
)


OUTPUT_DIR = Path(
    "evaluation/phase3_incentive"
)


def evaluate_model(
    model,
    X,
    y,
):
    probabilities = model.predict_proba(
        X
    )[:, 1]

    return {
        "roc_auc": roc_auc_score(
            y,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y,
            probabilities,
        ),
        "log_loss": log_loss(
            y,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y,
            probabilities,
        ),
        "mean_probability": probabilities.mean(),
        "min_probability": probabilities.min(),
        "max_probability": probabilities.max(),
    }


def create_candidate_incentives():
    return np.array(
        [
            0.50,
            1.00,
            1.50,
            2.00,
            2.50,
            3.00,
            3.50,
            4.00,
            5.00,
            6.00,
            8.00,
            10.00,
            12.00,
            15.00,
        ]
    )


def generate_response_curves(
    model,
    validation_data,
    features,
):
    candidate_incentives = (
        create_candidate_incentives()
    )

    sample_size = min(
        10000,
        len(validation_data),
    )

    sample = validation_data.sample(
        n=sample_size,
        random_state=42,
    ).copy()

    curve_results = []

    for incentive in candidate_incentives:
        scenario = sample.copy()

        scenario["incentive_amount"] = (
            incentive
        )

        scenario[
            "incentive_amount_squared"
        ] = incentive ** 2

        probabilities = model.predict_proba(
            scenario[features]
        )[:, 1]

        curve_results.append(
            {
                "incentive_amount": incentive,
                "mean_predicted_probability": (
                    probabilities.mean()
                ),
                "median_predicted_probability": (
                    np.median(probabilities)
                ),
            }
        )

    return pd.DataFrame(
        curve_results
    )


def create_response_curve_chart(
    curve_results,
):
    import matplotlib.pyplot as plt

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        curve_results[
            "incentive_amount"
        ],
        curve_results[
            "mean_predicted_probability"
        ] * 100,
        marker="o",
    )

    plt.xlabel(
        "Incentive Amount"
    )

    plt.ylabel(
        "Predicted Completion Probability (%)"
    )

    plt.title(
        "Predicted Completion Response to Incentive"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "predicted_incentive_response_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    return output_path


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_data = load_source_data()

    data = build_modeling_dataset(
        source_data
    )

    config = load_config()

    (
        train_data,
        validation_data,
        test_data,
    ) = create_temporal_split(
        data,
        config,
    )

    train_data = (
        prepare_incentive_response_data(
            train_data
        )
    )

    validation_data = (
        prepare_incentive_response_data(
            validation_data
        )
    )

    test_data = (
        prepare_incentive_response_data(
            test_data
        )
    )

    features = (
        get_incentive_response_features()
    )

    X_train = train_data[features]
    y_train = train_data["completed"]

    X_validation = (
        validation_data[features]
    )

    y_validation = (
        validation_data["completed"]
    )

    print("=" * 100)
    print("PHASE IV-B — INCENTIVE RESPONSE MODELING")
    print("=" * 100)

    print()
    print(
        f"Training rows:    {len(train_data):,}"
    )

    print(
        f"Validation rows:  {len(validation_data):,}"
    )

    print(
        f"Test rows:        {len(test_data):,}"
    )

    print()
    print(
        f"Training completion rate: "
        f"{y_train.mean():.4f}"
    )

    print(
        f"Validation completion rate: "
        f"{y_validation.mean():.4f}"
    )

    print()
    print(
        "Candidate incentives:"
    )

    print(
        create_candidate_incentives()
    )

    models = (
        get_incentive_response_models()
    )

    results = []

    fitted_models = {}

    for model_name, model in models.items():
        print()
        print("-" * 100)
        print(
            f"Training: {model_name}"
        )
        print("-" * 100)

        start_time = time.time()

        model.fit(
            X_train,
            y_train,
        )

        train_time = (
            time.time() - start_time
        )

        metrics = evaluate_model(
            model,
            X_validation,
            y_validation,
        )

        metrics["model"] = model_name
        metrics["train_time_seconds"] = (
            train_time
        )

        results.append(
            metrics
        )

        fitted_models[
            model_name
        ] = model

        print()
        print(
            f"ROC-AUC:       "
            f"{metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC:        "
            f"{metrics['pr_auc']:.4f}"
        )

        print(
            f"Log Loss:      "
            f"{metrics['log_loss']:.4f}"
        )

        print(
            f"Brier Score:   "
            f"{metrics['brier_score']:.4f}"
        )

        print(
            f"Mean Probability: "
            f"{metrics['mean_probability']:.4f}"
        )

        print(
            f"Train Time:    "
            f"{train_time:.2f}s"
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df[
        [
            "model",
            "roc_auc",
            "pr_auc",
            "log_loss",
            "brier_score",
            "mean_probability",
            "min_probability",
            "max_probability",
            "train_time_seconds",
        ]
    ]

    print()
    print("=" * 100)
    print("PHASE IV-B VALIDATION RESULTS")
    print("=" * 100)

    print()

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    results_df.to_csv(
        OUTPUT_DIR
        / "incentive_response_model_comparison.csv",
        index=False,
    )

    print()
    print(
        "Generating predicted incentive-response curves..."
    )

    for model_name, model in fitted_models.items():
        curve_results = (
            generate_response_curves(
                model,
                validation_data,
                features,
            )
        )

        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
        )

        curve_results.to_csv(
            OUTPUT_DIR
            / f"{safe_name}_response_curve.csv",
            index=False,
        )

        chart_path = (
            OUTPUT_DIR
            / f"{safe_name}_response_curve.png"
        )

        import matplotlib.pyplot as plt

        plt.figure(
            figsize=(10, 6)
        )

        plt.plot(
            curve_results[
                "incentive_amount"
            ],
            curve_results[
                "mean_predicted_probability"
            ] * 100,
            marker="o",
        )

        plt.xlabel(
            "Incentive Amount"
        )

        plt.ylabel(
            "Predicted Completion Probability (%)"
        )

        plt.title(
            f"{model_name}: "
            "Predicted Incentive Response"
        )

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.tight_layout()

        plt.savefig(
            chart_path,
            dpi=150,
        )

        plt.close()

        print(
            f"Saved: {chart_path}"
        )

    print()
    print("=" * 100)
    print("PHASE IV-B COMPLETE")
    print("=" * 100)

    print()
    print(
        f"Results saved to: {OUTPUT_DIR}"
    )

    print()
    print(
        "Test data was not used for model selection."
    )


if __name__ == "__main__":
    main()