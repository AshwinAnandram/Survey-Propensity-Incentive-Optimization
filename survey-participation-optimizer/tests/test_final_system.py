import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.data.loader import load_source_data, load_config
from src.data.assembler import build_modeling_dataset
from src.data.validation import (
    create_temporal_split,
    validate_modeling_dataset,
)

from src.features.incentive_response_features import (
    prepare_incentive_response_data,
)
from src.models.propensity import (
    create_hist_gradient_boosting_model,
)
from src.models.incentive_response import (
    create_hist_gradient_response_model,
)
from src.optimization.incentive_optimizer import (
    DEFAULT_INCENTIVES,
    build_incentive_candidates,
    select_optimal_incentive,
)


OUTPUT_DIR = Path("evaluation/final")
MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


PROPENSITY_FEATURES = [
    "age",
    "gender",
    "loc_india",
    "survey_category",
    "survey_length_minutes",
    "survey_complexity",
    "device",
    "channel",
    "incentive_amount",
    "prior_invitations",
    "prior_clicks",
    "prior_completions",
    "prior_click_rate",
    "prior_completion_rate",
    "days_since_last_invite",
    "days_since_last_completion",
]


RESPONSE_FEATURES = [
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


ECONOMIC_RULES = {
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


def print_section(title):
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


def fit_economic_value_reference(data):
    bounds = (
        data.groupby("survey_category")[
            "survey_length_minutes"
        ]
        .agg(["min", "max"])
        .to_dict("index")
    )

    return bounds


def apply_economic_value(data, length_bounds):
    data = data.copy()

    unknown_categories = (
        set(data["survey_category"].dropna().unique())
        - set(ECONOMIC_RULES)
    )

    if unknown_categories:
        raise ValueError(
            f"Unknown survey categories: "
            f"{sorted(unknown_categories)}"
        )

    data["economic_segment"] = (
        data["survey_category"]
        .map(
            lambda x: ECONOMIC_RULES[x]["segment"]
        )
    )

    category_min = (
        data["survey_category"]
        .map(
            lambda x: ECONOMIC_RULES[x]["min_value"]
        )
    )

    category_max = (
        data["survey_category"]
        .map(
            lambda x: ECONOMIC_RULES[x]["max_value"]
        )
    )

    lower_bounds = (
        data["survey_category"]
        .map(
            lambda x: length_bounds[x]["min"]
        )
    )

    upper_bounds = (
        data["survey_category"]
        .map(
            lambda x: length_bounds[x]["max"]
        )
    )

    length_range = (
        upper_bounds - lower_bounds
    ).replace(0, 1)

    length_position = (
        data["survey_length_minutes"]
        - lower_bounds
    ) / length_range

    length_position = length_position.clip(0, 1)

    data["survey_economic_value"] = (
        category_min
        + length_position
        * (category_max - category_min)
    )

    return data


def evaluate_classifier(
    model,
    X,
    y,
):
    probabilities = model.predict_proba(X)[:, 1]

    results = {
        "roc_auc": roc_auc_score(
            y,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y,
            probabilities,
        ),
        "log_loss": log_loss(
            y,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y,
            probabilities,
        ),
        "mean_probability": probabilities.mean(),
        "min_probability": probabilities.min(),
        "max_probability": probabilities.max(),
    }

    return results, probabilities


def calculate_decile_performance(
    data,
    probabilities,
):
    result = data[
        [
            "completed",
        ]
    ].copy()

    result["probability"] = probabilities

    result["decile"] = pd.qcut(
        result["probability"].rank(
            method="first"
        ),
        10,
        labels=False,
        duplicates="drop",
    ) + 1

    summary = (
        result
        .groupby("decile")
        .agg(
            observations=("completed", "count"),
            completion_rate=("completed", "mean"),
            mean_probability=("probability", "mean"),
        )
        .reset_index()
    )

    overall_rate = result["completed"].mean()

    summary["lift"] = (
        summary["completion_rate"]
        / overall_rate
    )

    summary["decile"] = summary[
        "decile"
    ].astype(int)

    summary = summary.sort_values(
        "decile"
    )

    return summary


def calculate_expected_value(
    probabilities,
    survey_values,
    incentives,
):
    return (
        probabilities
        * survey_values
        - incentives
    )


def evaluate_policy(
    data,
    response_model,
    policy_name,
):
    probabilities = response_model.predict_proba(
        data[RESPONSE_FEATURES]
    )[:, 1]

    expected_value = calculate_expected_value(
        probabilities,
        data["survey_economic_value"].values,
        data["incentive_amount"].values,
    )

    result = data[
        [
            "invitation_id",
            "survey_category",
            "economic_segment",
            "survey_economic_value",
            "incentive_amount",
            "completed",
        ]
    ].copy()

    result["predicted_probability"] = probabilities
    result["expected_value"] = expected_value
    result["policy"] = policy_name

    return result


def main():

    print("=" * 100)
    print("PHASE V — FINAL MODEL FREEZE & TEST EVALUATION")
    print("=" * 100)

    source_data = load_source_data()

    data = build_modeling_dataset(
        source_data
    )

    validate_modeling_dataset(
        data
    )

    config = load_config()

    train_data, validation_data, test_data = (
        create_temporal_split(
            data,
            config,
        )
    )

    print(
        f"\nOriginal temporal split:"
    )

    print(
        f"Training rows:    "
        f"{len(train_data):,}"
    )

    print(
        f"Validation rows:  "
        f"{len(validation_data):,}"
    )

    print(
        f"Test rows:        "
        f"{len(test_data):,}"
    )

    print(
        "\nThe test set has not been used "
        "for model selection."
    )

    development_data = pd.concat(
        [
            train_data,
            validation_data,
        ],
        ignore_index=True,
    )

    development_data = development_data.sort_values(
        "invitation_date"
    ).reset_index(
        drop=True
    )

    print_section(
        "FINAL DEVELOPMENT DATASET"
    )

    print(
        f"Development rows: "
        f"{len(development_data):,}"
    )

    print(
        f"Development completion rate: "
        f"{development_data['completed'].mean():.4f}"
    )

    print(
        f"Test completion rate: "
        f"{test_data['completed'].mean():.4f}"
    )

    development_bounds = (
        fit_economic_value_reference(
            development_data
        )
    )

    development_data = (
        prepare_incentive_response_data(
            development_data
        )
    )

    test_data = (
        prepare_incentive_response_data(
            test_data
        )
    )

    development_data = apply_economic_value(
        development_data,
        development_bounds,
    )

    test_data = apply_economic_value(
        test_data,
        development_bounds,
    )

    print_section(
        "FINAL PROPENSITY MODEL"
    )

    propensity_model = (
        create_hist_gradient_boosting_model()
    )

    X_development_propensity = (
        development_data[
            PROPENSITY_FEATURES
        ]
    )

    y_development = (
        development_data[
            "completed"
        ]
    )

    start_time = time.time()

    propensity_model.fit(
        X_development_propensity,
        y_development,
    )

    propensity_train_time = (
        time.time() - start_time
    )

    print(
        f"Training time: "
        f"{propensity_train_time:.2f}s"
    )

    propensity_results, test_propensity = (
        evaluate_classifier(
            propensity_model,
            test_data[
                PROPENSITY_FEATURES
            ],
            test_data[
                "completed"
            ],
        )
    )

    print(
        f"\nTest ROC-AUC: "
        f"{propensity_results['roc_auc']:.4f}"
    )

    print(
        f"Test PR-AUC: "
        f"{propensity_results['pr_auc']:.4f}"
    )

    print(
        f"Test Log Loss: "
        f"{propensity_results['log_loss']:.4f}"
    )

    print(
        f"Test Brier Score: "
        f"{propensity_results['brier_score']:.4f}"
    )

    print(
        f"Mean Probability: "
        f"{propensity_results['mean_probability']:.4f}"
    )

    print_section(
        "FINAL PROPENSITY DECILE PERFORMANCE"
    )

    propensity_deciles = (
        calculate_decile_performance(
            test_data,
            test_propensity,
        )
    )

    print(
        propensity_deciles.to_string(
            index=False
        )
    )

    propensity_deciles.to_csv(
        OUTPUT_DIR
        / "final_propensity_test_deciles.csv",
        index=False,
    )

    joblib.dump(
        propensity_model,
        MODEL_DIR
        / "final_propensity_hgb.joblib",
    )

    print(
        "\nSaved:"
        " models/final_propensity_hgb.joblib"
    )

    print_section(
        "FINAL RESPONSE MODEL"
    )

    response_model = (
        create_hist_gradient_response_model()
    )

    X_development_response = (
        development_data[
            RESPONSE_FEATURES
        ]
    )

    start_time = time.time()

    response_model.fit(
        X_development_response,
        y_development,
    )

    response_train_time = (
        time.time() - start_time
    )

    print(
        f"Training time: "
        f"{response_train_time:.2f}s"
    )

    response_results, test_response = (
        evaluate_classifier(
            response_model,
            test_data[
                RESPONSE_FEATURES
            ],
            test_data[
                "completed"
            ],
        )
    )

    print(
        f"\nTest ROC-AUC: "
        f"{response_results['roc_auc']:.4f}"
    )

    print(
        f"Test PR-AUC: "
        f"{response_results['pr_auc']:.4f}"
    )

    print(
        f"Test Log Loss: "
        f"{response_results['log_loss']:.4f}"
    )

    print(
        f"Test Brier Score: "
        f"{response_results['brier_score']:.4f}"
    )

    print(
        f"Mean Probability: "
        f"{response_results['mean_probability']:.4f}"
    )

    joblib.dump(
        response_model,
        MODEL_DIR
        / "final_incentive_response_hgb.joblib",
    )

    print(
        "\nSaved:"
        " models/final_incentive_response_hgb.joblib"
    )

    print_section(
        "HISTORICAL TEST POLICY"
    )

    historical_test = (
        evaluate_policy(
            test_data.copy(),
            response_model,
            "historical",
        )
    )

    historical_cost = (
        historical_test[
            "incentive_amount"
        ].sum()
    )

    historical_expected_value = (
        historical_test[
            "expected_value"
        ].sum()
    )

    historical_mean_probability = (
        historical_test[
            "predicted_probability"
        ].mean()
    )

    print(
        f"\nHistorical incentive cost: "
        f"${historical_cost:,.2f}"
    )

    print(
        f"Historical expected value: "
        f"${historical_expected_value:,.2f}"
    )

    print(
        f"Historical mean predicted probability: "
        f"{historical_mean_probability:.4f}"
    )

    print_section(
        "FINAL INCENTIVE OPTIMIZATION"
    )

    print(
        "\nProduction candidate incentives:"
    )

    print(
        DEFAULT_INCENTIVES
    )

    candidate_results = (
        build_incentive_candidates(
            test_data.copy(),
            response_model,
            DEFAULT_INCENTIVES,
        )
    )

    optimal_test = (
        select_optimal_incentive(
            candidate_results
        )
    )

    optimal_test[
        "survey_category"
    ] = test_data[
        "survey_category"
    ].values

    optimal_test[
        "economic_segment"
    ] = test_data[
        "economic_segment"
    ].values

    optimal_test[
        "survey_economic_value"
    ] = test_data[
        "survey_economic_value"
    ].values

    optimal_test[
        "completed"
    ] = test_data[
        "completed"
    ].values

    optimal_test[
        "historical_incentive"
    ] = test_data[
        "incentive_amount"
    ].values

    optimized_cost = (
        optimal_test[
            "incentive_amount"
        ].sum()
    )

    optimized_expected_value = (
        optimal_test[
            "expected_value"
        ].sum()
    )

    optimized_mean_probability = (
        optimal_test[
            "predicted_probability"
        ].mean()
    )

    print(
        f"\nOptimized incentive cost: "
        f"${optimized_cost:,.2f}"
    )

    print(
        f"Optimized expected value: "
        f"${optimized_expected_value:,.2f}"
    )

    print(
        f"Optimized mean predicted probability: "
        f"{optimized_mean_probability:.4f}"
    )

    incremental_cost = (
        optimized_cost
        - historical_cost
    )

    incremental_value = (
        optimized_expected_value
        - historical_expected_value
    )

    print(
        f"\nChange in incentive cost: "
        f"${incremental_cost:,.2f}"
    )

    print(
        f"Incremental expected value: "
        f"${incremental_value:,.2f}"
    )

    print_section(
        "FINAL TEST INCENTIVE DISTRIBUTION"
    )

    incentive_distribution = (
        optimal_test[
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
        / len(optimal_test)
        * 100
    )

    print(
        incentive_distribution.to_string(
            index=False
        )
    )

    incentive_distribution.to_csv(
        OUTPUT_DIR
        / "final_test_optimal_incentive_distribution.csv",
        index=False,
    )

    print_section(
        "FINAL TEST INCENTIVE BY CATEGORY"
    )

    category_summary = (
        optimal_test
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
            historical_mean_incentive=(
                "historical_incentive",
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
        / "final_test_incentive_by_category.csv",
        index=False,
    )

    print_section(
        "FINAL TEST POLICY CHANGE ANALYSIS"
    )

    policy_changed = (
        optimal_test[
            "incentive_amount"
        ]
        != optimal_test[
            "historical_incentive"
        ]
    )

    policy_increased = (
        optimal_test[
            "incentive_amount"
        ]
        > optimal_test[
            "historical_incentive"
        ]
    )

    policy_decreased = (
        optimal_test[
            "incentive_amount"
        ]
        < optimal_test[
            "historical_incentive"
        ]
    )

    print(
        f"\nPolicy changed: "
        f"{policy_changed.mean() * 100:.2f}%"
    )

    print(
        f"Policy increased incentive: "
        f"{policy_increased.mean() * 100:.2f}%"
    )

    print(
        f"Policy decreased incentive: "
        f"{policy_decreased.mean() * 100:.2f}%"
    )

    final_summary = pd.DataFrame(
        {
            "metric": [
                "test_rows",
                "test_completion_rate",
                "propensity_test_roc_auc",
                "propensity_test_pr_auc",
                "propensity_test_log_loss",
                "propensity_test_brier",
                "response_test_roc_auc",
                "response_test_pr_auc",
                "response_test_log_loss",
                "response_test_brier",
                "historical_incentive_cost",
                "optimized_incentive_cost",
                "change_in_incentive_cost",
                "historical_expected_value",
                "optimized_expected_value",
                "incremental_expected_value",
                "historical_mean_probability",
                "optimized_mean_probability",
                "policy_changed_rate",
                "policy_increased_rate",
                "policy_decreased_rate",
            ],
            "value": [
                len(test_data),
                test_data[
                    "completed"
                ].mean(),
                propensity_results[
                    "roc_auc"
                ],
                propensity_results[
                    "pr_auc"
                ],
                propensity_results[
                    "log_loss"
                ],
                propensity_results[
                    "brier_score"
                ],
                response_results[
                    "roc_auc"
                ],
                response_results[
                    "pr_auc"
                ],
                response_results[
                    "log_loss"
                ],
                response_results[
                    "brier_score"
                ],
                historical_cost,
                optimized_cost,
                incremental_cost,
                historical_expected_value,
                optimized_expected_value,
                incremental_value,
                historical_mean_probability,
                optimized_mean_probability,
                policy_changed.mean(),
                policy_increased.mean(),
                policy_decreased.mean(),
            ],
        }
    )

    final_summary.to_csv(
        OUTPUT_DIR
        / "final_system_test_summary.csv",
        index=False,
    )

    optimal_test.to_csv(
        OUTPUT_DIR
        / "final_test_optimal_decisions.csv",
        index=False,
    )

    historical_test.to_csv(
        OUTPUT_DIR
        / "final_test_historical_policy.csv",
        index=False,
    )

    print_section(
        "FINAL SYSTEM OUTPUTS"
    )

    print(
        "\nModels:"
    )

    print(
        "  models/final_propensity_hgb.joblib"
    )

    print(
        "  models/final_incentive_response_hgb.joblib"
    )

    print(
        "\nEvaluation:"
    )

    print(
        "  evaluation/final/"
    )

    print(
        "\nFinal test summary:"
    )

    print(
        "  evaluation/final/"
        "final_system_test_summary.csv"
    )

    print("\n" + "=" * 100)
    print(
        "PHASE V COMPLETE"
    )
    print("=" * 100)

    print(
        "\nThe final test set was used only for final evaluation."
    )


if __name__ == "__main__":
    main()