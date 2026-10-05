import numpy as np
import pandas as pd


ECONOMIC_SEGMENTS = {
    "Consumer": {
        "segment": "B2C",
        "min_value": 2.0,
        "max_value": 5.0,
    },
    "Retail": {
        "segment": "B2C",
        "min_value": 2.0,
        "max_value": 5.0,
    },
    "B2B": {
        "segment": "B2B",
        "min_value": 15.0,
        "max_value": 30.0,
    },
    "Finance": {
        "segment": "B2B",
        "min_value": 15.0,
        "max_value": 30.0,
    },
    "Technology": {
        "segment": "B2B",
        "min_value": 15.0,
        "max_value": 30.0,
    },
    "Healthcare": {
        "segment": "Healthcare",
        "min_value": 50.0,
        "max_value": 100.0,
    },
}


def calculate_economic_value(data):
    data = data.copy()

    if "survey_category" not in data.columns:
        raise ValueError("survey_category is required")

    if "survey_length_minutes" not in data.columns:
        raise ValueError("survey_length_minutes is required")

    unknown_categories = set(data["survey_category"].dropna().unique()) - set(
        ECONOMIC_SEGMENTS
    )

    if unknown_categories:
        raise ValueError(
            f"Unknown survey categories: {sorted(unknown_categories)}"
        )

    category_min = data["survey_category"].map(
        lambda x: ECONOMIC_SEGMENTS[x]["min_value"]
    )

    category_max = data["survey_category"].map(
        lambda x: ECONOMIC_SEGMENTS[x]["max_value"]
    )

    length_min = data.groupby("survey_category")[
        "survey_length_minutes"
    ].transform("min")

    length_max = data.groupby("survey_category")[
        "survey_length_minutes"
    ].transform("max")

    length_range = (length_max - length_min).replace(0, np.nan)

    length_position = (
        (data["survey_length_minutes"] - length_min) / length_range
    ).fillna(0.5)

    length_position = length_position.clip(0, 1)

    data["economic_segment"] = data["survey_category"].map(
        lambda x: ECONOMIC_SEGMENTS[x]["segment"]
    )

    data["survey_economic_value"] = (
        category_min
        + length_position * (category_max - category_min)
    )

    return data


def get_economic_value_rules():
    return ECONOMIC_SEGMENTS.copy()