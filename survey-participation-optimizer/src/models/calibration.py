from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

from src.models.propensity import create_hist_gradient_boosting_model


def create_calibrated_hgb(method="sigmoid"):
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("method must be 'sigmoid' or 'isotonic'")

    base_model = create_hist_gradient_boosting_model()
    temporal_cv = TimeSeriesSplit(n_splits=3)

    return CalibratedClassifierCV(
        estimator=base_model,
        method=method,
        cv=temporal_cv,
        ensemble=True,
    )


def get_calibration_models():
    return {
        "HGB + Sigmoid": create_calibrated_hgb("sigmoid"),
        "HGB + Isotonic": create_calibrated_hgb("isotonic"),
    }