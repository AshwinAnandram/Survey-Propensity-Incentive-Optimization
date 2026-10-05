from src.data.loader import load_config, load_source_data
from src.data.assembler import build_modeling_dataset
from src.data.validation import (
    create_temporal_split,
    validate_modeling_dataset,
)


def main():

    print("Loading configuration...")

    config = load_config()

    print("Loading source data...")

    source_data = load_source_data()

    for name, dataframe in source_data.items():
        print(
            f"{name}: "
            f"{dataframe.shape[0]:,} rows x "
            f"{dataframe.shape[1]} columns"
        )

    print("\nBuilding modeling dataset...")

    data = build_modeling_dataset(source_data)

    print(
        f"Modeling dataset: "
        f"{data.shape[0]:,} rows x "
        f"{data.shape[1]} columns"
    )

    print("\nValidating modeling dataset...")

    validate_modeling_dataset(data)

    print("Dataset validation passed.")

    print("\nCreating temporal split...")

    train, validation, test = create_temporal_split(
        data,
        config
    )

    print(
        f"Train:      {len(train):,} rows"
    )

    print(
        f"Validation: {len(validation):,} rows"
    )

    print(
        f"Test:       {len(test):,} rows"
    )

    print("\nDate ranges:")

    print(
        "Train:",
        train["invitation_date"].min(),
        "to",
        train["invitation_date"].max()
    )

    print(
        "Validation:",
        validation["invitation_date"].min(),
        "to",
        validation["invitation_date"].max()
    )

    print(
        "Test:",
        test["invitation_date"].min(),
        "to",
        test["invitation_date"].max()
    )

    print("\nCompletion rates:")

    print(
        "Train:",
        round(train["completed"].mean(), 4)
    )

    print(
        "Validation:",
        round(validation["completed"].mean(), 4)
    )

    print(
        "Test:",
        round(test["completed"].mean(), 4)
    )

    print("\nData pipeline completed successfully.")


if __name__ == "__main__":
    main()