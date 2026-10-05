from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.incentive_response_features import (
    create_incentive_response_preprocessor,
)


def create_logistic_response_model():
    preprocessor = (
        create_incentive_response_preprocessor()
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight=None,
        random_state=42,
    )

    return Pipeline(
        [
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


def create_hist_gradient_response_model():
    preprocessor = (
        create_incentive_response_preprocessor()
    )

    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42,
    )

    return Pipeline(
        [
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


def get_incentive_response_models():
    return {
        "Logistic Response": (
            create_logistic_response_model()
        ),
        "HGB Response": (
            create_hist_gradient_response_model()
        ),
    }