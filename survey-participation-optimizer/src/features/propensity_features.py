from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERICAL_FEATURES = [
    "age",
    "survey_length_minutes",
    "survey_complexity",
    "incentive_amount",
    "prior_invitations",
    "prior_clicks",
    "prior_completions",
    "prior_click_rate",
    "prior_completion_rate",
    "days_since_last_invite",
    "days_since_last_completion",
]


CATEGORICAL_FEATURES = [
    "gender",
    "loc_india",
    "survey_category",
    "device",
    "channel",
]


PROPENSITY_FEATURES = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
)


def get_feature_columns():
    return PROPENSITY_FEATURES.copy()


def create_preprocessor():

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "cat",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor