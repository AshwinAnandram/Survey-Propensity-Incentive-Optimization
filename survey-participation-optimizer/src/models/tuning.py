from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.models.propensity import (
    create_hist_gradient_boosting_model,
    create_logistic_regression_model,
)


def tune_logistic_regression(
    X_train,
    y_train,
):

    model = create_logistic_regression_model()

    parameter_grid = {
        "model__C": [
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
        ],
        "model__solver": [
            "lbfgs",
        ],
        "model__class_weight": [
            "balanced",
            None,
        ],
    }

    cv = TimeSeriesSplit(
        n_splits=3
    )

    search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )

    search.fit(
        X_train,
        y_train,
    )

    return search


def tune_hist_gradient_boosting(
    X_train,
    y_train,
):

    model = create_hist_gradient_boosting_model()

    parameter_grid = {
        "model__learning_rate": [
            0.03,
            0.05,
            0.10,
        ],
        "model__max_iter": [
            100,
            200,
            300,
        ],
        "model__max_leaf_nodes": [
            15,
            31,
            63,
        ],
        "model__min_samples_leaf": [
            20,
            50,
            100,
        ],
        "model__l2_regularization": [
            0.0,
            0.1,
            1.0,
        ],
    }

    cv = TimeSeriesSplit(
        n_splits=3
    )

    search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )

    search.fit(
        X_train,
        y_train,
    )

    return search