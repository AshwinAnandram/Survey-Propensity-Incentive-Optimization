from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.assembler import build_modeling_dataset
from src.data.loader import load_config, load_source_data
from src.data.validation import create_temporal_split


OUTPUT_DIR = Path("evaluation/phase3_incentive")


def main():
    source_data = load_source_data()
    data = build_modeling_dataset(source_data)

    config = load_config()

    train_data, validation_data, test_data = (
        create_temporal_split(
            data,
            config,
        )
    )

    assessment_data = pd.concat(
        [
            train_data,
            validation_data,
        ],
        ignore_index=True,
    )

    assessment_data = (
        assessment_data
        .sort_values("invitation_date")
        .reset_index(drop=True)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("PHASE IV-A — INCENTIVE ASSIGNMENT ANALYSIS")
    print("=" * 100)

    print()
    print(
        f"Training rows:       {len(train_data):,}"
    )

    print(
        f"Validation rows:     {len(validation_data):,}"
    )

    print(
        f"Assignment analysis: {len(assessment_data):,}"
    )

    print(
        f"Test rows excluded:  {len(test_data):,}"
    )

    print()
    print(
        f"Assessment date range: "
        f"{assessment_data['invitation_date'].min().date()} "
        f"to "
        f"{assessment_data['invitation_date'].max().date()}"
    )

    analyze_numeric_assignment(
        assessment_data
    )

    analyze_categorical_assignment(
        assessment_data,
        "survey_category",
    )

    analyze_categorical_assignment(
        assessment_data,
        "channel",
    )

    analyze_categorical_assignment(
        assessment_data,
        "device",
    )

    analyze_categorical_assignment(
        assessment_data,
        "gender",
    )

    analyze_categorical_assignment(
        assessment_data,
        "loc_india",
    )

    analyze_time_assignment(
        assessment_data
    )

    create_assignment_charts(
        assessment_data
    )

    print()
    print("=" * 100)
    print("PHASE IV-A COMPLETE")
    print("=" * 100)

    print()
    print(
        f"Outputs saved to: {OUTPUT_DIR}"
    )

    print()
    print(
        "Test period was excluded from assignment analysis."
    )


def analyze_numeric_assignment(data):
    print()
    print("=" * 100)
    print("INCENTIVE ASSOCIATION WITH NUMERIC VARIABLES")
    print("=" * 100)

    numeric_columns = [
        "age",
        "survey_length_minutes",
        "survey_complexity",
        "prior_invitations",
        "prior_clicks",
        "prior_completions",
        "prior_click_rate",
        "prior_completion_rate",
        "days_since_last_invite",
        "days_since_last_completion",
    ]

    correlations = (
        data[
            numeric_columns + ["incentive_amount"]
        ]
        .corr(numeric_only=True)
        ["incentive_amount"]
        .drop("incentive_amount")
        .sort_values(
            key=lambda series: series.abs(),
            ascending=False,
        )
    )

    result = correlations.reset_index()

    result.columns = [
        "feature",
        "correlation_with_incentive",
    ]

    print()

    print(
        result.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    result.to_csv(
        OUTPUT_DIR
        / "incentive_numeric_associations.csv",
        index=False,
    )


def analyze_categorical_assignment(
    data,
    column,
):
    print()
    print("=" * 100)
    print(
        f"INCENTIVE ASSIGNMENT BY {column.upper()}"
    )
    print("=" * 100)

    summary = (
        data
        .groupby(column, dropna=False)
        .agg(
            invitations=(
                "invitation_id",
                "count",
            ),
            mean_incentive=(
                "incentive_amount",
                "mean",
            ),
            median_incentive=(
                "incentive_amount",
                "median",
            ),
            completion_rate=(
                "completed",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["completion_rate_pct"] = (
        summary["completion_rate"] * 100
    )

    summary = summary.sort_values(
        "mean_incentive",
        ascending=False,
    )

    print()

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    safe_name = column.lower()

    summary.to_csv(
        OUTPUT_DIR
        / f"incentive_assignment_by_{safe_name}.csv",
        index=False,
    )


def analyze_time_assignment(data):
    print()
    print("=" * 100)
    print("INCENTIVE ASSIGNMENT OVER TIME")
    print("=" * 100)

    monthly = data.copy()

    monthly["month"] = (
        monthly["invitation_date"]
        .dt.to_period("M")
        .astype(str)
    )

    summary = (
        monthly
        .groupby("month")
        .agg(
            invitations=(
                "invitation_id",
                "count",
            ),
            mean_incentive=(
                "incentive_amount",
                "mean",
            ),
            median_incentive=(
                "incentive_amount",
                "median",
            ),
            incentive_std=(
                "incentive_amount",
                "std",
            ),
            completion_rate=(
                "completed",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["completion_rate_pct"] = (
        summary["completion_rate"] * 100
    )

    print()

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    summary.to_csv(
        OUTPUT_DIR
        / "incentive_assignment_over_time.csv",
        index=False,
    )


def create_assignment_charts(data):
    numeric_columns = [
        "prior_completion_rate",
        "prior_click_rate",
        "prior_invitations",
        "prior_completions",
        "survey_length_minutes",
        "survey_complexity",
        "age",
    ]

    for column in numeric_columns:
        plt.figure(figsize=(9, 6))

        plt.scatter(
            data[column],
            data["incentive_amount"],
            alpha=0.15,
            s=8,
        )

        plt.xlabel(column)
        plt.ylabel("Incentive Amount")
        plt.title(
            f"Incentive Assignment vs {column}"
        )

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.tight_layout()

        filename = (
            f"incentive_vs_{column}.png"
        )

        plt.savefig(
            OUTPUT_DIR / filename,
            dpi=150,
        )

        plt.close()

    monthly = data.copy()

    monthly["month"] = (
        monthly["invitation_date"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly_summary = (
        monthly
        .groupby("month")
        .agg(
            mean_incentive=(
                "incentive_amount",
                "mean",
            ),
            median_incentive=(
                "incentive_amount",
                "median",
            ),
        )
        .reset_index()
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        monthly_summary["month"],
        monthly_summary["mean_incentive"],
        marker="o",
        label="Mean incentive",
    )

    plt.plot(
        monthly_summary["month"],
        monthly_summary["median_incentive"],
        marker="o",
        label="Median incentive",
    )

    plt.xlabel("Month")
    plt.ylabel("Incentive Amount")
    plt.title(
        "Historical Incentive Assignment Over Time"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "incentive_assignment_over_time.png",
        dpi=150,
    )

    plt.close()


if __name__ == "__main__":
    main()