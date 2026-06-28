import pandas as pd
from typing import Tuple

def generate_duration_increase_data(
    X_test: pd.DataFrame,
    months: int = 12,
) -> pd.DataFrame:
    """
    MR1: Higher duration -> P(bad) should not decrease.
    """
    X_duration = X_test.copy()

    X_duration["duration"] = X_duration["duration"] + months
    X_duration["monthly_payment"] = (
        X_duration["credit_amount"] / X_duration["duration"]
    )

    return X_duration

def generate_checking_status_improvement_data(
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    MR4:
    Better checking account status -> P(bad) should not increase.

    'no checking' is excluded because it does not represent
    an ordered checking account balance category.

    '>=200' is already the best category and therefore cannot
    be improved further.
    """

    X_checking = X_test.copy()

    evaluation_mask = X_test["checking_status"].isin(
        ["<0", "0<=X<200"]
    )

    checking_mapping = {
        "<0": "0<=X<200",
        "0<=X<200": ">=200",
        ">=200": ">=200",
        "no checking": "no checking",
    }

    X_checking["checking_status"] = (
        X_checking["checking_status"].replace(checking_mapping)
    )

    return X_checking, evaluation_mask


def generate_savings_status_deterioration_data(
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    MR5:
    Lower savings -> P(bad) should not decrease.

    'no known savings' is excluded because it does not
    represent an ordered savings category.

    '<100' is already the lowest category and therefore
    cannot be decreased further.
    """

    X_savings = X_test.copy()

    evaluation_mask = X_test["savings_status"].isin(
        ["100<=X<500", "500<=X<1000", ">=1000"]
    )

    savings_mapping = {
        ">=1000": "500<=X<1000",
        "500<=X<1000": "100<=X<500",
        "100<=X<500": "<100",
        "<100": "<100",
        "no known savings": "no known savings",
    }

    X_savings["savings_status"] = (
        X_savings["savings_status"].replace(savings_mapping)
    )

    return X_savings, evaluation_mask
