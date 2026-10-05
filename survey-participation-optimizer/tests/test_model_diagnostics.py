from pathlib import Path

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

from src.evaluation.plots import (
    plot_calibration_curves,
    plot_precision_recall_curves,
    plot_roc_curves,
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

    model_probabilities = {}

    for model_name, model in models.items():

        print(
            f"Training {model_name}..."
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

        model_probabilities[
            model_name
        ] = probabilities

    output_directory = Path(
        "evaluation/plots"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nCreating ROC curve..."
    )

    plot_roc_curves(
        y_test,
        model_probabilities,
        output_directory
        / "roc_curves.png",
    )

    print(
        "\nCreating Precision-Recall curve..."
    )

    plot_precision_recall_curves(
        y_test,
        model_probabilities,
        output_directory
        / "precision_recall_curves.png",
    )

    print(
        "\nCreating calibration curve..."
    )

    plot_calibration_curves(
        y_test,
        model_probabilities,
        output_directory
        / "calibration_curves.png",
    )

    print(
        "\nModel diagnostics completed."
    )


if __name__ == "__main__":
    main()