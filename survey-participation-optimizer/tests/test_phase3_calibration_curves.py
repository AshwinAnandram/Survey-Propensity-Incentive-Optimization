import time
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

from src.data.assembler import build_modeling_dataset
from src.data.loader import load_config, load_source_data
from src.data.validation import create_temporal_split
from src.features.propensity_features import get_feature_columns
from src.models.calibration import get_calibration_models
from src.models.propensity import create_hist_gradient_boosting_model


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

    X_validation = validation_data[features]
    y_validation = validation_data["completed"]

    models = {
        "HGB Baseline": create_hist_gradient_boosting_model(),
        **get_calibration_models(),
    }

    predictions = {}

    print("=" * 100)
    print("PHASE III CALIBRATION CURVES")
    print("=" * 100)

    print(f"Training rows:    {len(train_data):,}")
    print(f"Validation rows:  {len(validation_data):,}")
    print(f"Test rows:        {len(test_data):,}")
    print()

    print(
        f"Training completion rate:   "
        f"{y_train.mean():.4f}"
    )

    print(
        f"Validation completion rate: "
        f"{y_validation.mean():.4f}"
    )

    print()

    for name, model in models.items():
        start_time = time.time()

        model.fit(
            X_train,
            y_train,
        )

        probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        elapsed = time.time() - start_time

        predictions[name] = probabilities

        logloss = log_loss(
            y_validation,
            probabilities,
        )

        brier = brier_score_loss(
            y_validation,
            probabilities,
        )

        print(name)
        print(f"  Log Loss:    {logloss:.4f}")
        print(f"  Brier Score: {brier:.4f}")
        print(f"  Train Time:  {elapsed:.2f}s")
        print(
            f"  Mean Probability: "
            f"{probabilities.mean():.4f}"
        )
        print(
            f"  Min Probability:  "
            f"{probabilities.min():.4f}"
        )
        print(
            f"  Max Probability:  "
            f"{probabilities.max():.4f}"
        )
        print()

    output_dir = Path("evaluation")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(10, 7))

    for name, probabilities in predictions.items():
        fraction_positive, mean_predicted = calibration_curve(
            y_validation,
            probabilities,
            n_bins=10,
            strategy="quantile",
        )

        plt.plot(
            mean_predicted,
            fraction_positive,
            marker="o",
            label=name,
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration",
    )

    plt.axhline(
        y=y_validation.mean(),
        linestyle=":",
        label=(
            "Validation Completion Rate "
            f"({y_validation.mean():.3f})"
        ),
    )

    plt.xlabel(
        "Mean Predicted Probability"
    )

    plt.ylabel(
        "Observed Completion Rate"
    )

    plt.title(
        "Phase III Propensity Model Calibration"
    )

    plt.legend()
    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    output_path = (
        output_dir
        / "phase3_calibration_curves.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print("=" * 100)
    print(
        f"Calibration plot saved to: "
        f"{output_path}"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()