from zenml import step
from typing import Annotated
import pandas as pd


@step
def feature_engineering(
    df: pd.DataFrame,
) -> tuple[
    Annotated[pd.DataFrame, "feature_engineered_data"],
    Annotated[pd.DataFrame, "sensitive_attributes"],
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
        labels=["<25", "25-40", "40-60", "60+"],
    )

    # 5. Additional security
    # Indicates whether another party, such as a guarantor or co-applicant, exists
    df["has_additional_security"] = (
        df["other_parties"] != "none"
    ).astype(int)

    # 6. High-risk financial status
    # Indicates a combination of negative checking account and low/no savings
    df["high_risk_financial_status"] = (
        (df["checking_status"] == "<0")
        & (df["savings_status"].isin(["<100", "no known savings"]))
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
            "age",
        ]
    )

    return df, sensitive_attributes