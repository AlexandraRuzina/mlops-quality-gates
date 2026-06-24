from zenml import step
from typing import Annotated
import pandas as pd


@step
def feature_engineering(
    df: pd.DataFrame,
) -> tuple[
    Annotated[pd.DataFrame, "feature_engineered_data"],
    Annotated[list[str], "sensitive_attributes"],
]:
    """
    Creates additional domain-specific features for the German Credit dataset.
    """

    df = df.copy()

    # 1. Monthly payment
    # Approximate monthly repayment amount
    df["monthly_payment"] = df["credit_amount"] / df["duration"]

    # 2. Age group
    # Groups applicants into age categories
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 24, 40, 60, float("inf")],
        labels=["<25", "25-40", "40-60", "60+"]
    )

    # 3. Credit burden ratio
    # Credit amount in relation to number of existing credits
    df["credit_burden_ratio"] = (
        df["credit_amount"] / df["existing_credits"]
    )

    # 4. Negative checking account
    # Indicates whether the checking account balance is below 0 DM
    #1 = Konto ist überzogen / negativer Kontostand, sonst 0
    df["is_negative_checking"] = (
        df["checking_status"] == "<0"
    ).astype(int)

    df["has_additional_security"] = (
            df["other_parties"] != "none"
    ).astype(int)

    def extract_sex(personal_status):
        personal_status = str(personal_status).lower()

        if "female" in personal_status:
            return "female"
        elif "male" in personal_status:
            return "male"
        else:
            return None

    df["sex"] = df["personal_status"].apply(extract_sex)

    # Sensitive attributes for fairness
    sensitive_attributes = df[
        ["foreign_worker", "sex"]
    ].copy()

    # Remove sensitive attributes from training dataset
    df = df.drop(
        columns=[
            "personal_status",
            "foreign_worker",
            "own_telephone",
            "sex",
            "age"
        ]
    )

    return df, sensitive_attributes