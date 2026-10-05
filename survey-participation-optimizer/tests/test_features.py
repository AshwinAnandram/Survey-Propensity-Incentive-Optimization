from src.data.loader import load_config, load_source_data
from src.data.assembler import build_modeling_dataset
from src.data.validation import create_temporal_split
from src.features.propensity_features import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    create_preprocessor,
    get_feature_columns,
)


def main():

    config = load_config()

    source_data = load_source_data()

    data = build_modeling_dataset(
        source_data
    )

    train, validation, test = create_temporal_split(
        data,
        config
    )

    features = get_feature_columns()

    X_train = train[features]

    print("Numerical features:")
    for feature in NUMERICAL_FEATURES:
        print(f"  {feature}")

    print("\nCategorical features:")
    for feature in CATEGORICAL_FEATURES:
        print(f"  {feature}")

    print(
        f"\nTotal features before encoding: "
        f"{len(features)}"
    )

    preprocessor = create_preprocessor()

    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    print(
        "\nTransformed training shape:",
        X_train_transformed.shape
    )

    print(
        "\nFeature engineering pipeline "
        "executed successfully."
    )


if __name__ == "__main__":
    main()