import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.loader import load_source_data, load_config
from src.data.assembler import build_modeling_dataset
from src.data.validation import create_temporal_split, validate_modeling_dataset
from src.features.incentive_response_features import prepare_incentive_response_data
from src.models.incentive_response import create_hist_gradient_response_model
from src.optimization.economic_value import calculate_economic_value
from src.optimization.incentive_optimizer import (
    DEFAULT_INCENTIVES,
    build_incentive_candidates,
    select_optimal_incentive,
)


OUTPUT_DIR = Path("evaluation/phase3_incentive")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_COLUMNS = [
    "age",
    "gender",
    "loc_india",
    "survey_category",
    "survey_length_minutes",
    "survey_complexity",
    "incentive_amount",
    "incentive_amount_squared",
    "device",
    "channel",
    "prior_invitations",
    "prior_clicks",
    "prior_completions",
    "prior_click_rate",
    "prior_completion_rate",
    "days_since_last_invite",
    "days_since_last_completion",
]


def print_section(title):
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


def generate_economic_response_curves(
    validation_data,
    response_model,
):
    print_section("RESPONSE CURVE BY ECONOMIC SEGMENT")

    sample_size = min(10000, len(validation_data))

    sample = validation_data.sample(
        n=sample_size,
        random_state=42,
    ).copy()

    curve_rows = []

    for segment in sample["economic_segment"].unique():

        segment_data = sample[
            sample["economic_segment"] == segment
        ].copy()

        for incentive in DEFAULT_INCENTIVES:

            candidate = segment_data.copy()

            candidate["incentive_amount"] = incentive
            candidate["incentive_amount_squared"] = incentive ** 2

            probabilities = response_model.predict_proba(
                candidate[FEATURE_COLUMNS]
            )[:, 1]

            mean_probability = probabilities.mean()
            mean_survey_value = segment_data[
                "survey_economic_value"
            ].mean()

            expected_value = (
                mean_probability * mean_survey_value
                - incentive
            )

            curve_rows.append(
                {
                    "economic_segment": segment,
                    "incentive_amount": incentive,
                    "mean_predicted_probability": mean_probability,
                    "mean_survey_value": mean_survey_value,
                    "expected_value": expected_value,
                }
            )

    response_curves = pd.DataFrame(curve_rows)

    response_curves.to_csv(
        OUTPUT_DIR / "economic_response_curves.csv",
        index=False,
    )

    for segment in response_curves["economic_segment"].unique():

        curve = response_curves[
            response_curves["economic_segment"] == segment
        ]

        plt.figure(figsize=(10, 6))

        plt.plot(
            curve["incentive_amount"],
            curve["mean_predicted_probability"],
            marker="o",
        )

        plt.xlabel("Incentive Amount")
        plt.ylabel("Mean Predicted Completion Probability")
        plt.title(
            f"Incentive Response Curve — {segment}"
        )

        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        filename = (
            f"{segment.lower()}_incentive_response_curve.png"
        )

        plt.savefig(
            OUTPUT_DIR / filename,
            dpi=150,
        )

        plt.close()

    return response_curves


def main():

    print("=" * 100)
    print("PHASE IV-C — ECONOMIC INCENTIVE OPTIMIZATION")
    print("=" * 100)

    source_data = load_source_data()

    data = build_modeling_dataset(source_data)

    validate_modeling_dataset(data)

    config = load_config()

    train_data, validation_data, test_data = create_temporal_split(
        data,
        config,
    )

    print(f"\nTraining rows:    {len(train_data):,}")
    print(f"Validation rows:  {len(validation_data):,}")
    print(f"Test rows:        {len(test_data):,}")

    print("\nTest data remains untouched.")

    train_data = prepare_incentive_response_data(
        train_data
    )

    validation_data = prepare_incentive_response_data(
        validation_data
    )

    train_data = calculate_economic_value(
        train_data
    )

    validation_data = calculate_economic_value(
        validation_data
    )

    print_section("ECONOMIC VALUE DISTRIBUTION")

    economic_summary = (
        validation_data
        .groupby(
            [
                "economic_segment",
                "survey_category",
            ]
        )
        .agg(
            invitations=("invitation_id", "count"),
            mean_survey_value=(
                "survey_economic_value",
                "mean",
            ),
            min_survey_value=(
                "survey_economic_value",
                "min",
            ),
            max_survey_value=(
                "survey_economic_value",
                "max",
            ),
            mean_survey_length=(
                "survey_length_minutes",
                "mean",
            ),
            completion_rate=(
                "completed",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        economic_summary.to_string(
            index=False
        )
    )

    economic_summary.to_csv(
        OUTPUT_DIR
        / "economic_value_validation_summary.csv",
        index=False,
    )

    print_section("TRAINING HGB RESPONSE MODEL")

    response_model = create_hist_gradient_response_model()

    X_train = train_data[
        FEATURE_COLUMNS
    ]

    y_train = train_data[
        "completed"
    ]

    start_time = time.time()

    response_model.fit(
        X_train,
        y_train,
    )

    train_time = time.time() - start_time

    print(
        f"Training time: {train_time:.2f}s"
    )

    print_section("GENERATING INCENTIVE CANDIDATES")

    print(
        f"\nCandidate incentives:\n"
        f"{DEFAULT_INCENTIVES}"
    )

    candidate_results = build_incentive_candidates(
        validation_data.copy(),
        response_model,
        DEFAULT_INCENTIVES,
    )

    candidate_results["survey_category"] = (
        validation_data[
            "survey_category"
        ]
        .values
        .repeat(len(DEFAULT_INCENTIVES))
    )

    candidate_results["economic_segment"] = (
        validation_data[
            "economic_segment"
        ]
        .values
        .repeat(len(DEFAULT_INCENTIVES))
    )

    candidate_results["survey_length_minutes"] = (
        validation_data[
            "survey_length_minutes"
        ]
        .values
        .repeat(len(DEFAULT_INCENTIVES))
    )

    optimal = select_optimal_incentive(
        candidate_results
    )

    optimal["survey_category"] = (
        validation_data[
            "survey_category"
        ].values
    )

    optimal["economic_segment"] = (
        validation_data[
            "economic_segment"
        ].values
    )

    optimal["survey_length_minutes"] = (
        validation_data[
            "survey_length_minutes"
        ].values
    )

    optimal["completed"] = (
        validation_data[
            "completed"
        ].values
    )

    optimal["survey_economic_value"] = (
        validation_data[
            "survey_economic_value"
        ].values
    )

    print_section("OPTIMAL INCENTIVE SUMMARY")

    incentive_distribution = (
        optimal[
            "incentive_amount"
        ]
        .value_counts()
        .sort_index()
        .rename_axis(
            "recommended_incentive"
        )
        .reset_index(
            name="respondent_count"
        )
    )

    incentive_distribution[
        "percentage"
    ] = (
        incentive_distribution[
            "respondent_count"
        ]
        / len(optimal)
        * 100
    )

    print(
        incentive_distribution.to_string(
            index=False
        )
    )

    incentive_distribution.to_csv(
        OUTPUT_DIR
        / "optimal_incentive_distribution.csv",
        index=False,
    )

    print_section(
        "OPTIMAL INCENTIVE BY SURVEY CATEGORY"
    )

    category_summary = (
        optimal
        .groupby(
            [
                "economic_segment",
                "survey_category",
            ]
        )
        .agg(
            respondents=(
                "decision_id",
                "count",
            ),
            mean_recommended_incentive=(
                "incentive_amount",
                "mean",
            ),
            median_recommended_incentive=(
                "incentive_amount",
                "median",
            ),
            mean_predicted_probability=(
                "predicted_probability",
                "mean",
            ),
            mean_survey_value=(
                "survey_economic_value",
                "mean",
            ),
            mean_expected_value=(
                "expected_value",
                "mean",
            ),
            observed_completion_rate=(
                "completed",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        category_summary.to_string(
            index=False
        )
    )

    category_summary.to_csv(
        OUTPUT_DIR
        / "optimal_incentive_by_category.csv",
        index=False,
    )

    print_section(
        "HISTORICAL VS OPTIMIZED ECONOMIC VALUE"
    )

    historical_data = validation_data.copy()

    historical_probabilities = (
        response_model.predict_proba(
            historical_data[
                FEATURE_COLUMNS
            ]
        )[:, 1]
    )

    historical_data[
        "historical_predicted_probability"
    ] = historical_probabilities

    historical_data[
        "historical_expected_value"
    ] = (
        historical_data[
            "historical_predicted_probability"
        ]
        * historical_data[
            "survey_economic_value"
        ]
        - historical_data[
            "incentive_amount"
        ]
    )

    comparison = optimal[
        [
            "decision_id",
            "incentive_amount",
            "predicted_probability",
            "survey_economic_value",
            "expected_value",
        ]
    ].copy()

    comparison = comparison.rename(
        columns={
            "incentive_amount":
                "optimized_incentive",
            "predicted_probability":
                "optimized_predicted_probability",
            "survey_economic_value":
                "survey_economic_value",
            "expected_value":
                "optimized_expected_value",
        }
    )

    comparison[
        "historical_incentive"
    ] = historical_data[
        "incentive_amount"
    ].values

    comparison[
        "historical_predicted_probability"
    ] = historical_data[
        "historical_predicted_probability"
    ].values

    comparison[
        "historical_expected_value"
    ] = historical_data[
        "historical_expected_value"
    ].values

    comparison[
        "incremental_expected_value"
    ] = (
        comparison[
            "optimized_expected_value"
        ]
        - comparison[
            "historical_expected_value"
        ]
    )

    comparison[
        "incremental_incentive_cost"
    ] = (
        comparison[
            "optimized_incentive"
        ]
        - comparison[
            "historical_incentive"
        ]
    )

    comparison[
        "incremental_probability"
    ] = (
        comparison[
            "optimized_predicted_probability"
        ]
        - comparison[
            "historical_predicted_probability"
        ]
    )

    comparison[
        "survey_category"
    ] = validation_data[
        "survey_category"
    ].values

    comparison[
        "economic_segment"
    ] = validation_data[
        "economic_segment"
    ].values

    comparison[
        "completed"
    ] = validation_data[
        "completed"
    ].values

    historical_total_value = (
        comparison[
            "historical_expected_value"
        ].sum()
    )

    optimized_total_value = (
        comparison[
            "optimized_expected_value"
        ].sum()
    )

    incremental_total_value = (
        comparison[
            "incremental_expected_value"
        ].sum()
    )

    historical_total_cost = (
        comparison[
            "historical_incentive"
        ].sum()
    )

    optimized_total_cost = (
        comparison[
            "optimized_incentive"
        ].sum()
    )

    incremental_total_cost = (
        optimized_total_cost
        - historical_total_cost
    )

    historical_mean_ev = (
        comparison[
            "historical_expected_value"
        ].mean()
    )

    optimized_mean_ev = (
        comparison[
            "optimized_expected_value"
        ].mean()
    )

    incremental_mean_ev = (
        comparison[
            "incremental_expected_value"
        ].mean()
    )

    historical_mean_probability = (
        comparison[
            "historical_predicted_probability"
        ].mean()
    )

    optimized_mean_probability = (
        comparison[
            "optimized_predicted_probability"
        ].mean()
    )

    incremental_mean_probability = (
        comparison[
            "incremental_probability"
        ].mean()
    )

    print(
        f"\nHistorical incentive cost: "
        f"${historical_total_cost:,.2f}"
    )

    print(
        f"Optimized incentive cost:  "
        f"${optimized_total_cost:,.2f}"
    )

    print(
        f"Change in incentive cost:  "
        f"${incremental_total_cost:,.2f}"
    )

    print(
        f"\nHistorical expected value: "
        f"${historical_total_value:,.2f}"
    )

    print(
        f"Optimized expected value:  "
        f"${optimized_total_value:,.2f}"
    )

    print(
        f"Incremental expected value:"
        f" ${incremental_total_value:,.2f}"
    )

    print(
        f"\nHistorical mean EV: "
        f"${historical_mean_ev:.4f}"
    )

    print(
        f"Optimized mean EV:  "
        f"${optimized_mean_ev:.4f}"
    )

    print(
        f"Incremental mean EV:"
        f" ${incremental_mean_ev:.4f}"
    )

    print(
        f"\nHistorical mean probability: "
        f"{historical_mean_probability:.4f}"
    )

    print(
        f"Optimized mean probability:  "
        f"{optimized_mean_probability:.4f}"
    )

    print(
        f"Incremental mean probability:"
        f" {incremental_mean_probability:.4f}"
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "historical_vs_optimized_comparison.csv",
        index=False,
    )

    economic_comparison = pd.DataFrame(
        {
            "metric": [
                "historical_incentive_cost",
                "optimized_incentive_cost",
                "incremental_incentive_cost",
                "historical_expected_value",
                "optimized_expected_value",
                "incremental_expected_value",
                "historical_mean_expected_value",
                "optimized_mean_expected_value",
                "incremental_mean_expected_value",
                "historical_mean_probability",
                "optimized_mean_probability",
                "incremental_mean_probability",
            ],
            "value": [
                historical_total_cost,
                optimized_total_cost,
                incremental_total_cost,
                historical_total_value,
                optimized_total_value,
                incremental_total_value,
                historical_mean_ev,
                optimized_mean_ev,
                incremental_mean_ev,
                historical_mean_probability,
                optimized_mean_probability,
                incremental_mean_probability,
            ],
        }
    )

    economic_comparison.to_csv(
        OUTPUT_DIR
        / "historical_vs_optimized_summary.csv",
        index=False,
    )

    print_section(
        "HISTORICAL VS OPTIMIZED BY SURVEY CATEGORY"
    )

    category_comparison = (
        comparison
        .groupby(
            [
                "economic_segment",
                "survey_category",
            ]
        )
        .agg(
            respondents=(
                "decision_id",
                "count",
            ),
            historical_mean_incentive=(
                "historical_incentive",
                "mean",
            ),
            optimized_mean_incentive=(
                "optimized_incentive",
                "mean",
            ),
            historical_mean_probability=(
                "historical_predicted_probability",
                "mean",
            ),
            optimized_mean_probability=(
                "optimized_predicted_probability",
                "mean",
            ),
            historical_mean_ev=(
                "historical_expected_value",
                "mean",
            ),
            optimized_mean_ev=(
                "optimized_expected_value",
                "mean",
            ),
            incremental_mean_ev=(
                "incremental_expected_value",
                "mean",
            ),
            incremental_incentive_cost=(
                "incremental_incentive_cost",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        category_comparison.to_string(
            index=False
        )
    )

    category_comparison.to_csv(
        OUTPUT_DIR
        / "historical_vs_optimized_by_category.csv",
        index=False,
    )

    print_section(
        "POLICY CHANGE ANALYSIS"
    )

    policy_change_rate = (
        comparison[
            "optimized_incentive"
        ]
        != comparison[
            "historical_incentive"
        ]
    ).mean()

    policy_increase_rate = (
        comparison[
            "optimized_incentive"
        ]
        > comparison[
            "historical_incentive"
        ]
    ).mean()

    policy_decrease_rate = (
        comparison[
            "optimized_incentive"
        ]
        < comparison[
            "historical_incentive"
        ]
    ).mean()

    print(
        f"\nPolicy changed: "
        f"{policy_change_rate * 100:.2f}%"
    )

    print(
        f"Policy increased incentive: "
        f"{policy_increase_rate * 100:.2f}%"
    )

    print(
        f"Policy decreased incentive: "
        f"{policy_decrease_rate * 100:.2f}%"
    )

    policy_change_summary = pd.DataFrame(
        {
            "metric": [
                "policy_change_rate",
                "policy_increase_rate",
                "policy_decrease_rate",
            ],
            "value": [
                policy_change_rate,
                policy_increase_rate,
                policy_decrease_rate,
            ],
        }
    )

    policy_change_summary.to_csv(
        OUTPUT_DIR
        / "policy_change_summary.csv",
        index=False,
    )

    print_section(
        "POLICY CHANGE BY SURVEY CATEGORY"
    )

    category_policy_change = (
        comparison.assign(
            policy_changed=(
                comparison[
                    "optimized_incentive"
                ]
                != comparison[
                    "historical_incentive"
                ]
            ),
            policy_increased=(
                comparison[
                    "optimized_incentive"
                ]
                > comparison[
                    "historical_incentive"
                ]
            ),
            policy_decreased=(
                comparison[
                    "optimized_incentive"
                ]
                < comparison[
                    "historical_incentive"
                ]
            ),
        )
        .groupby(
            [
                "economic_segment",
                "survey_category",
            ]
        )
        .agg(
            respondents=(
                "decision_id",
                "count",
            ),
            policy_changed=(
                "policy_changed",
                "mean",
            ),
            policy_increased=(
                "policy_increased",
                "mean",
            ),
            policy_decreased=(
                "policy_decreased",
                "mean",
            ),
        )
        .reset_index()
    )

    category_policy_change[
        "policy_changed"
    ] *= 100

    category_policy_change[
        "policy_increased"
    ] *= 100

    category_policy_change[
        "policy_decreased"
    ] *= 100

    print(
        category_policy_change.to_string(
            index=False
        )
    )

    category_policy_change.to_csv(
        OUTPUT_DIR
        / "policy_change_by_category.csv",
        index=False,
    )

    response_curves = (
        generate_economic_response_curves(
            validation_data,
            response_model,
        )
    )

    print_section(
        "PHASE IV-C OUTPUTS"
    )

    print(
        f"\nResults saved to: "
        f"{OUTPUT_DIR}"
    )

    print(
        "\nGenerated files:"
    )

    output_files = [
        "economic_value_validation_summary.csv",
        "optimal_incentive_distribution.csv",
        "optimal_incentive_by_category.csv",
        "historical_vs_optimized_comparison.csv",
        "historical_vs_optimized_summary.csv",
        "historical_vs_optimized_by_category.csv",
        "policy_change_summary.csv",
        "policy_change_by_category.csv",
        "economic_response_curves.csv",
        "b2b_incentive_response_curve.png",
        "b2c_incentive_response_curve.png",
        "healthcare_incentive_response_curve.png",
    ]

    for filename in output_files:
        print(
            f"  {OUTPUT_DIR / filename}"
        )

    print("\n" + "=" * 100)
    print("PHASE IV-C COMPLETE")
    print("=" * 100)

    print(
        "\nTest data was not used for model selection."
    )


if __name__ == "__main__":
    main()