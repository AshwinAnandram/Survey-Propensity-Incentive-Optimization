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
    create_logistic_regression_model,
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

    print("\nTraining Logistic Regression...")

    model = create_logistic_regression_model()

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    print("\nGenerating validation predictions...")

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    print("\nGenerating test predictions...")

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    print("\nValidation metrics")

    validation_metrics, validation_matrix = (
        evaluate_classifier(
            y_validation,
            validation_probabilities,
        )
    )

    for metric, value in validation_metrics.items():
        print(
            f"{metric}: "
            f"{value:.4f}"
        )

    print("\nValidation confusion matrix")

    print(validation_matrix)

    print("\nTest metrics")

    test_metrics, test_matrix = (
        evaluate_classifier(
            y_test,
            test_probabilities,
        )
    )

    for metric, value in test_metrics.items():
        print(
            f"{metric}: "
            f"{value:.4f}"
        )

    print("\nTest confusion matrix")

    print(test_matrix)

    print("\nTest lift table")

    lift_table = calculate_lift_table(
        y_test,
        test_probabilities,
    )

    print(
        lift_table.to_string(
            index=False
        )
    )

    print(
        "\nLogistic Regression baseline "
        "completed successfully."
    )


if __name__ == "__main__":
    main()