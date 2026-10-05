import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)


def calculate_threshold_table(
    y_true,
    probabilities,
    thresholds=None,
):

    if thresholds is None:
        thresholds = np.arange(
            0.10,
            0.55,
            0.05,
        )

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    rows = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        selected = predictions.sum()

        rows.append(
            {
                "threshold": threshold,
                "selected_count": selected,
                "selected_pct": (
                    selected
                    / len(predictions)
                ),
                "precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
            }
        )

    return pd.DataFrame(rows)