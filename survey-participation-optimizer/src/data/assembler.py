import pandas as pd


def build_modeling_dataset(source_data):

    respondents = source_data["respondents"]
    surveys = source_data["surveys"]
    invitations = source_data["invitations"]
    responses = source_data["responses"]

    response_outcomes = responses[
        [
            "invitation_id",
            "completed",
        ]
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