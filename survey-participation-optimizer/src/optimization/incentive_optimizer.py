import numpy as np
import pandas as pd


DEFAULT_INCENTIVES = [
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


def build_incentive_candidates(
    data,
    response_model,
    incentives=DEFAULT_INCENTIVES,
):
    base = data.copy().reset_index(drop=True)
    base["decision_id"] = np.arange(len(base))

    candidate_frames = []

    for incentive in incentives:
        candidate = base.copy()
        candidate["incentive_amount"] = incentive
        candidate["incentive_amount_squared"] = incentive ** 2

        probabilities = response_model.predict_proba(
            candidate
        )[:, 1]

        candidate["predicted_probability"] = probabilities

        candidate["expected_value"] = (
            candidate["predicted_probability"]
            * candidate["survey_economic_value"]
            - incentive
        )

        candidate_frames.append(
            candidate[
                [
                    "decision_id",
                    "incentive_amount",
                    "predicted_probability",
                    "survey_economic_value",
                    "expected_value",
                ]
            ]
        )

    return pd.concat(
        candidate_frames,
        ignore_index=True,
    )


def select_optimal_incentive(
    candidate_results,
):
    optimal = (
        candidate_results
        .sort_values(
            [
                "decision_id",
                "expected_value",
                "incentive_amount",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .drop_duplicates(
            "decision_id"
        )
        .sort_values(
            "decision_id"
        )
        .reset_index(
            drop=True
        )
    )

    return optimal