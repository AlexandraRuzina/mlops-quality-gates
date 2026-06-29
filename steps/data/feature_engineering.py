from zenml import step
from typing import Annotated
from scripts import calculate_features
import pandas as pd


@step
def feature_engineering(
    df: pd.DataFrame,
) -> Annotated[pd.DataFrame, "feature_engineered_data"]:

    """
    Creates additional domain-specific features for the German Credit dataset.
    """

    df = df.copy()

    df = calculate_features.calculate_features(df)

    def extract_sex(personal_status):
        personal_status = str(personal_status).lower()

        if "female" in personal_status:
            return "female"
        elif "male" in personal_status:
            return "male"
        else:
            return None

    df["sex"] = df["personal_status"].apply(extract_sex)

    # Remove sensitive attributes from training dataset
    df = df.drop(
        columns=[
            "personal_status",
            "foreign_worker",
            "own_telephone",
            "age",
        ]
    )

    return df