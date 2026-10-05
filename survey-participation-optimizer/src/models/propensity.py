from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.features.propensity_features import (
    create_preprocessor,
)


def create_logistic_regression_model():

    preprocessor = create_preprocessor()

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline(
        steps=[
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


def create_decision_tree_model():

    preprocessor = create_preprocessor()

    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline(
        steps=[
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


def create_random_forest_model():

    preprocessor = create_preprocessor()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
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


def create_gradient_boosting_model():

    preprocessor = create_preprocessor()

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=20,
        random_state=42,
    )

    return Pipeline(
        steps=[
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


def create_hist_gradient_boosting_model():

    preprocessor = create_preprocessor()

    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42,
    )

    return Pipeline(
        steps=[
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


def get_propensity_models():

    return {
        "Logistic Regression": (
            create_logistic_regression_model()
        ),
        "Decision Tree": (
            create_decision_tree_model()
        ),
        "Random Forest": (
            create_random_forest_model()
        ),
        "Gradient Boosting": (
            create_gradient_boosting_model()
        ),
        "Hist Gradient Boosting": (
            create_hist_gradient_boosting_model()
        ),
    }