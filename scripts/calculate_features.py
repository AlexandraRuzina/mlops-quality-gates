import pandas as pd

def calculate_features(df: pd.DataFrame):

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

    return df