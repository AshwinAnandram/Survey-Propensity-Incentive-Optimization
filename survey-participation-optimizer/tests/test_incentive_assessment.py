from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


CONFIG_PATH = Path("configs/config.yaml")
OUTPUT_DIR = Path("evaluation/phase3_incentive")


def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_source_data(config_path=CONFIG_PATH):
    config = load_config(config_path)

    respondents = pd.read_excel(
        Path(config["data"]["respondents"])
    )

    surveys = pd.read_excel(
        Path(config["data"]["surveys"])
    )

    invitations = pd.read_excel(
        Path(config["data"]["invitations"])
    )

    responses = pd.read_excel(
        Path(config["data"]["responses"])
    )

    invitations["invitation_date"] = pd.to_datetime(
        invitations["invitation_date"],
        errors="coerce",
    )

    responses["click_date"] = pd.to_datetime(
        responses["click_date"],
        errors="coerce",
    )

    responses["completion_date"] = pd.to_datetime(
        responses["completion_date"],
        errors="coerce",
    )

    return {
        "respondents": respondents,
        "surveys": surveys,
        "invitations": invitations,
        "responses": responses,
    }


def build_modeling_dataset(source_data):
    respondents = source_data["respondents"]
    surveys = source_data["surveys"]
    invitations = source_data["invitations"]
    responses = source_data["responses"]

    response_outcomes = responses[
        ["invitation_id", "completed"]
    ].copy()

    data = (
        invitations
        .merge(
            respondents,
            on="user_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            surveys,
            on="survey_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            response_outcomes,
            on="invitation_id",
            how="left",
            validate="one_to_one",
        )
    )

    data["completed"] = (
        data["completed"]
        .fillna(0)
        .astype(int)
    )

    return data


def create_training_data(data, config):
    date_column = config["split"]["date_column"]

    data = data.sort_values(
        date_column
    ).reset_index(drop=True)

    train_pct = config["split"]["train_pct"]
    validation_pct = config["split"]["validation_pct"]

    train_cutoff = data[date_column].quantile(
        train_pct
    )

    validation_cutoff = data[date_column].quantile(
        train_pct + validation_pct
    )

    training_data = data[
        data[date_column] <= train_cutoff
    ].copy()

    validation_data = data[
        (data[date_column] > train_cutoff)
        & (data[date_column] <= validation_cutoff)
    ].copy()

    test_data = data[
        data[date_column] > validation_cutoff
    ].copy()

    return (
        training_data,
        validation_data,
        test_data,
        train_cutoff,
        validation_cutoff,
    )


def print_dataset_summary(
    data,
    training_data,
    validation_data,
    test_data,
):
    print()
    print("=" * 100)
    print("DATASET SUMMARY")
    print("=" * 100)

    print()
    print(f"Total invitations:      {len(data):,}")
    print(f"Training invitations:   {len(training_data):,}")
    print(f"Validation invitations: {len(validation_data):,}")
    print(f"Test invitations:       {len(test_data):,}")

    print()
    print(
        f"Overall completion rate: "
        f"{data['completed'].mean() * 100:.2f}%"
    )

    print(
        f"Training completion rate: "
        f"{training_data['completed'].mean() * 100:.2f}%"
    )

    print(
        f"Validation completion rate: "
        f"{validation_data['completed'].mean() * 100:.2f}%"
    )

    print(
        f"Test completion rate: "
        f"{test_data['completed'].mean() * 100:.2f}%"
    )


def print_incentive_distribution(data):
    print()
    print("=" * 100)
    print("INCENTIVE DISTRIBUTION")
    print("=" * 100)

    summary = data["incentive_amount"].describe()

    print()
    print(summary.to_string())

    print()
    print(
        f"Unique incentive values: "
        f"{data['incentive_amount'].nunique():,}"
    )

    print(
        f"Minimum incentive: "
        f"{data['incentive_amount'].min():.4f}"
    )

    print(
        f"Maximum incentive: "
        f"{data['incentive_amount'].max():.4f}"
    )

    print(
        f"Mean incentive: "
        f"{data['incentive_amount'].mean():.4f}"
    )

    print(
        f"Median incentive: "
        f"{data['incentive_amount'].median():.4f}"
    )


def create_quantile_incentive_buckets(data):
    print()
    print("=" * 100)
    print("COMPLETION RATE BY INCENTIVE QUANTILE")
    print("=" * 100)

    data = data.copy()

    data["incentive_quantile"] = pd.qcut(
        data["incentive_amount"],
        q=10,
        duplicates="drop",
    )

    summary = (
        data
        .groupby(
            "incentive_quantile",
            observed=True,
        )
        .agg(
            invitations=("invitation_id", "count"),
            completions=("completed", "sum"),
            mean_incentive=("incentive_amount", "mean"),
            median_incentive=("incentive_amount", "median"),
            completion_rate=("completed", "mean"),
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

    return summary


def create_incentive_quantile_chart(
    quantile_summary,
    output_dir,
):
    plt.figure(figsize=(10, 6))

    x = range(len(quantile_summary))

    plt.plot(
        x,
        quantile_summary["completion_rate_pct"],
        marker="o",
    )

    plt.xticks(
        x,
        [
            f"Q{i}"
            for i in range(
                1,
                len(quantile_summary) + 1,
            )
        ],
    )

    plt.xlabel("Incentive Quantile")
    plt.ylabel("Completion Rate (%)")
    plt.title(
        "Completion Rate by Incentive Quantile"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    output_path = (
        output_dir
        / "incentive_quantile_completion.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print()
    print(
        f"Saved chart: {output_path}"
    )


def create_incentive_amount_scatter(
    data,
    output_dir,
):
    plt.figure(figsize=(12, 7))

    plt.scatter(
        data["incentive_amount"],
        data["completed"] * 100,
        alpha=0.15,
        s=10,
    )

    plt.xlabel("Incentive Amount")
    plt.ylabel("Completion (%)")
    plt.title(
        "Historical Completion by Incentive Amount"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    output_path = (
        output_dir
        / "historical_incentive_scatter.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print()
    print(
        f"Saved chart: {output_path}"
    )


def create_monthly_incentive_summary(data):
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
            invitations=("invitation_id", "count"),
            mean_incentive=("incentive_amount", "mean"),
            median_incentive=("incentive_amount", "median"),
            completion_rate=("completed", "mean"),
        )
        .reset_index()
    )

    summary["completion_rate_pct"] = (
        summary["completion_rate"] * 100
    )

    return summary


def create_monthly_incentive_chart(
    monthly_summary,
    output_dir,
):
    plt.figure(figsize=(12, 7))

    plt.plot(
        monthly_summary["month"],
        monthly_summary["mean_incentive"],
        marker="o",
    )

    plt.xlabel("Month")
    plt.ylabel("Mean Incentive")
    plt.title(
        "Historical Mean Incentive Over Time"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    output_path = (
        output_dir
        / "historical_mean_incentive_over_time.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print()
    print(
        f"Saved chart: {output_path}"
    )


def create_monthly_completion_chart(
    monthly_summary,
    output_dir,
):
    plt.figure(figsize=(12, 7))

    plt.plot(
        monthly_summary["month"],
        monthly_summary["completion_rate_pct"],
        marker="o",
    )

    plt.xlabel("Month")
    plt.ylabel("Completion Rate (%)")
    plt.title(
        "Historical Completion Rate Over Time"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    output_path = (
        output_dir
        / "historical_completion_rate_over_time.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print()
    print(
        f"Saved chart: {output_path}"
    )


def save_results(
    quantile_summary,
    monthly_summary,
    output_dir,
):
    quantile_path = (
        output_dir
        / "incentive_quantile_summary.csv"
    )

    monthly_path = (
        output_dir
        / "monthly_incentive_summary.csv"
    )

    quantile_summary.to_csv(
        quantile_path,
        index=False,
    )

    monthly_summary.to_csv(
        monthly_path,
        index=False,
    )

    print()
    print(
        f"Saved results: {quantile_path}"
    )

    print(
        f"Saved results: {monthly_path}"
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = load_config()

    source_data = load_source_data(
        CONFIG_PATH
    )

    data = build_modeling_dataset(
        source_data
    )

    (
        training_data,
        validation_data,
        test_data,
        train_cutoff,
        validation_cutoff,
    ) = create_training_data(
        data,
        config,
    )

    print_dataset_summary(
        data,
        training_data,
        validation_data,
        test_data,
    )

    print()
    print("=" * 100)
    print("TEMPORAL SPLIT")
    print("=" * 100)

    print()
    print(
        f"Training cutoff:   {train_cutoff}"
    )

    print(
        f"Validation cutoff: {validation_cutoff}"
    )

    print(
        f"Training period:   "
        f"{training_data['invitation_date'].min()} "
        f"to "
        f"{training_data['invitation_date'].max()}"
    )

    print(
        f"Validation period: "
        f"{validation_data['invitation_date'].min()} "
        f"to "
        f"{validation_data['invitation_date'].max()}"
    )

    print(
        f"Test period:       "
        f"{test_data['invitation_date'].min()} "
        f"to "
        f"{test_data['invitation_date'].max()}"
    )

    print_incentive_distribution(
        training_data
    )

    quantile_summary = (
        create_quantile_incentive_buckets(
            training_data
        )
    )

    monthly_summary = (
        create_monthly_incentive_summary(
            training_data
        )
    )

    print()
    print("=" * 100)
    print("MONTHLY INCENTIVE SUMMARY")
    print("=" * 100)

    print()

    print(
        monthly_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    create_incentive_quantile_chart(
        quantile_summary,
        OUTPUT_DIR,
    )

    create_incentive_amount_scatter(
        training_data,
        OUTPUT_DIR,
    )

    create_monthly_incentive_chart(
        monthly_summary,
        OUTPUT_DIR,
    )

    create_monthly_completion_chart(
        monthly_summary,
        OUTPUT_DIR,
    )

    save_results(
        quantile_summary,
        monthly_summary,
        OUTPUT_DIR,
    )

    print()
    print("=" * 100)
    print("INCENTIVE ASSESSMENT COMPLETE")
    print("=" * 100)

    print()
    print(
        f"Outputs saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()