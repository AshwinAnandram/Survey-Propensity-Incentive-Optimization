from pathlib import Path
from src.data.assembler import build_modeling_dataset
from src.data.loader import load_source_data
from src.features.feature_audit import (
    build_feature_audit,
    validate_feature_audit,
)


def main():

    print("Loading source data...")

    source_data = load_source_data()

    print("Building modeling dataset...")

    data = build_modeling_dataset(
        source_data
    )

    print("Building feature audit...")

    audit = build_feature_audit(data)

    validate_feature_audit(audit)

    print("\nFeature audit")
    print("=" * 80)

    print(
        audit[
            [
                "feature",
                "dtype",
                "model_type",
                "missing_pct",
                "unique_values",
                "sample_values",
            ]
        ].to_string(index=False)
    )

    output_dir = Path("evaluation/propensity")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "feature_audit_v001.csv"
    )

    audit.to_csv(
        output_path,
        index=False,
    )

    print("\nFeature audit validation passed.")
    print(f"Audit saved to: {output_path}")

if __name__ == "__main__":
    main()