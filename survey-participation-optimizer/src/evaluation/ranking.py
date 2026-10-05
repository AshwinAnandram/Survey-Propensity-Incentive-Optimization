import pandas as pd
import numpy as np


def calculate_lift_table(
    y_true,
    probabilities,
    bins=10,
):

    data = pd.DataFrame(
        {
            "actual": np.asarray(y_true),
            "probability": np.asarray(
                probabilities
            ),
        }
    )

    data = data.sort_values(
        "probability",
        ascending=False,
    ).reset_index(drop=True)

    data["decile"] = pd.qcut(
        data.index,
        q=bins,
        labels=False,
        duplicates="drop",
    ) + 1

    overall_rate = data["actual"].mean()

    result = (
        data
        .groupby("decile", observed=True)
        .agg(
            observations=("actual", "size"),
            completions=("actual", "sum"),
            completion_rate=("actual", "mean"),
            mean_probability=(
                "probability",
                "mean",
            ),
        )
        .reset_index()
    )

    result["lift"] = (
        result["completion_rate"]
        / overall_rate
    )

    result["cumulative_completions"] = (
        result["completions"].cumsum()
    )

    total_completions = (
        result["completions"].sum()
    )

    result["cumulative_gain"] = (
        result["cumulative_completions"]
        / total_completions
    )

    return result