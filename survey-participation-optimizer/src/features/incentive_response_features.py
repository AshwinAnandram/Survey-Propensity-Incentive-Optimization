from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERICAL_FEATURES = [
    "age",
    "survey_length_minutes",
    "survey_complexity",
    "incentive_amount",
    "incentive_amount_squared",
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

INCENTIVE_RESPONSE_FEATURES = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
)


def prepare_incentive_response_data(data):
    data = data.copy()

    data["incentive_amount_squared"] = (
        data["incentive_amount"] ** 2
    )

    return data


def get_incentive_response_features():
    return INCENTIVE_RESPONSE_FEATURES.copy()


def create_incentive_response_preprocessor():
    numerical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
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

    return ColumnTransformer(
        [
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