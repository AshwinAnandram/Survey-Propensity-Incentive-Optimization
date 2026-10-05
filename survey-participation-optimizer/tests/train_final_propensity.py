from pathlib import Path

import joblib
import pandas as pd

from src.data.assembler import build_modeling_dataset
from src.data.loader import load_config, load_source_data
from src.data.validation import create_temporal_split
from src.features.propensity_features import get_feature_columns
from src.models.propensity import create_hist_gradient_boosting_model


def main():
    source_data = load_source_data()
    data = build_modeling_dataset(source_data)

    config = load_config()

    train_data, validation_data, test_data = create_temporal_split(
        data,
        config,
    )

    final_training_data = (
        pd.concat(
            [train_data, validation_data],
            ignore_index=True,
        )
        .sort_values("invitation_date")
        .reset_index(drop=True)
    )

    features = get_feature_columns()

    X_final = final_training_data[features]
    y_final = final_training_data["completed"]

    model = create_hist_gradient_boosting_model()

    print("=" * 100)
    print("FINAL PRODUCTION PROPENSITY MODEL")
    print("=" * 100)

    print()
    print(f"Training rows:       {len(train_data):,}")
    print(f"Validation rows:     {len(validation_data):,}")
    print(f"Final training rows: {len(final_training_data):,}")
    print(f"Test rows excluded:  {len(test_data):,}")

    print()
    print(
        f"Final training date range: "
        f"{final_training_data['invitation_date'].min().date()} "
        f"to "
        f"{final_training_data['invitation_date'].max().date()}"
    )

    print()
    print(
        f"Final training completion rate: "
        f"{y_final.mean():.4f}"
    )

    print()
    print("Training final HGB model...")

    model.fit(
        X_final,
        y_final,
    )

    model_path = (
        Path("models")
        / "propensity_hgb_final.joblib"
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
        f"Final production model saved to: "
        f"{model_path}"
    )

    print()
    print(
        "Test data was not used during final training."
    )

    print("=" * 100)


if __name__ == "__main__":
    main()