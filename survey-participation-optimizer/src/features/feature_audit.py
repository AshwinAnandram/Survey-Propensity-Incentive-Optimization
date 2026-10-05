import pandas as pd

from src.features.propensity_features import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PROPENSITY_FEATURES,
)


def build_feature_audit(data):

    rows = []

    for feature in PROPENSITY_FEATURES:

        series = data[feature]

        if feature in NUMERICAL_FEATURES:
            feature_type = "numerical"
        elif feature in CATEGORICAL_FEATURES:
            feature_type = "categorical"
        else:
            feature_type = "unknown"

        sample_values = (
            series
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )

        rows.append(
            {
                "feature": feature,
                "dtype": str(series.dtype),
                "model_type": feature_type,
                "missing_count": int(series.isna().sum()),
                "missing_pct": round(
                    series.isna().mean() * 100,
                    4,
                ),
                "unique_values": int(
                    series.nunique(dropna=True)
                ),
                "sample_values": sample_values,
            }
        )

    return pd.DataFrame(rows)


def validate_feature_audit(audit):

    if audit.empty:
        raise ValueError(
            "Feature audit is empty."
        )

    if audit["feature"].duplicated().any():
        raise ValueError(
            "Duplicate features found in audit."
        )

    if (audit["model_type"] == "unknown").any():
        unknown_features = audit.loc[
            audit["model_type"] == "unknown",
            "feature",
        ].tolist()

        raise ValueError(
            f"Unknown feature types: {unknown_features}"
        )

    if (audit["missing_pct"] > 100).any():
        raise ValueError(
            "Invalid missing percentage detected."
        )

    return True