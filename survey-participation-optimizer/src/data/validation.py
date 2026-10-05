FEATURES = [
    "age",
    "gender",
    "loc_india",
    "survey_category",
    "survey_length_minutes",
    "survey_complexity",
    "device",
    "channel",
    "incentive_amount",
    "prior_invitations",
    "prior_clicks",
    "prior_completions",
    "prior_click_rate",
    "prior_completion_rate",
    "days_since_last_invite",
    "days_since_last_completion",
]


FORBIDDEN_FEATURES = {
    "completed",
    "click_date",
    "completion_date",
    "invitation_id",
    "user_id",
    "survey_id",
    "invitation_date",
    "date_of_join",
    "base_engagement",
    "incentive_sensitivity",
    "fatigue_sensitivity",
}


def validate_modeling_dataset(data):

    required_columns = set(FEATURES) | {
        "completed",
        "invitation_date",
        "invitation_id",
        "user_id",
        "survey_id",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    leakage = FORBIDDEN_FEATURES.intersection(FEATURES)

    if leakage:
        raise ValueError(
            f"Potential leakage features detected: {sorted(leakage)}"
        )

    if data["invitation_id"].duplicated().any():
        raise ValueError(
            "Duplicate invitation_id values detected."
        )

    if data["user_id"].isna().any():
        raise ValueError(
            "Missing user_id values detected."
        )

    if data["survey_id"].isna().any():
        raise ValueError(
            "Missing survey_id values detected."
        )

    if data["invitation_date"].isna().any():
        raise ValueError(
            "Missing invitation_date values detected."
        )

    if not data["completed"].isin([0, 1]).all():
        raise ValueError(
            "Target completed must contain only 0 and 1."
        )

    return True


def create_temporal_split(data, config):

    date_column = config["split"]["date_column"]

    train_pct = config["split"]["train_pct"]
    validation_pct = config["split"]["validation_pct"]
    test_pct = config["split"]["test_pct"]

    total = train_pct + validation_pct + test_pct

    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            "Train, validation and test percentages must sum to 1."
        )

    data = data.sort_values(date_column).reset_index(drop=True)

    train_cutoff = data[date_column].quantile(train_pct)

    validation_cutoff = data[date_column].quantile(
        train_pct + validation_pct
    )

    train = data[
        data[date_column] <= train_cutoff
    ].copy()

    validation = data[
        (data[date_column] > train_cutoff)
        & (data[date_column] <= validation_cutoff)
    ].copy()

    test = data[
        data[date_column] > validation_cutoff
    ].copy()

    if train.empty:
        raise ValueError("Training dataset is empty.")

    if validation.empty:
        raise ValueError("Validation dataset is empty.")

    if test.empty:
        raise ValueError("Test dataset is empty.")

    if train[date_column].max() >= validation[date_column].min():
        raise ValueError(
            "Training and validation periods overlap."
        )

    if validation[date_column].max() >= test[date_column].min():
        raise ValueError(
            "Validation and test periods overlap."
        )

    return train, validation, test