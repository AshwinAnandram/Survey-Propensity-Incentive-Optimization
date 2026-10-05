from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path="configs/config.yaml"):
    config_path = Path(config_path)

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_source_data(config_path="configs/config.yaml"):
    config = load_config(config_path)

    respondents = pd.read_excel(
        Path(config["data"]["respondents"])
    )

    surveys = pd.read_excel(
        Path(config["data"]["surveys"])
    )

    invitations = pd.read_excel(
        Path(config["data"]["invitations"])
    )

    responses = pd.read_excel(
        Path(config["data"]["responses"])
    )

    invitations["invitation_date"] = pd.to_datetime(
        invitations["invitation_date"],
        errors="coerce"
    )

    responses["click_date"] = pd.to_datetime(
        responses["click_date"],
        errors="coerce"
    )

    responses["completion_date"] = pd.to_datetime(
        responses["completion_date"],
        errors="coerce"
    )

    return {
        "respondents": respondents,
        "surveys": surveys,
        "invitations": invitations,
        "responses": responses,
    }