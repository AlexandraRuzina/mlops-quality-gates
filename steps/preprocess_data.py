from zenml import step
import pandas as pd


@step
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic data preprocessing for the DataCo dataset.
    """

    # Remove rows with missing values
    df = df.dropna()
    # Remove duplicate rows
    df = df.drop_duplicates()

    # Convert numeric features to numeric
    numeric_features = [
        "duration",
        "credit_amount",
        "installment_commitment",
        "residence_since",
        "age",
        "existing_credits",
        "num_dependents"
    ]

    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_features)

    df = df[
        (df["duration"] >= 0) &
        (df["credit_amount"] >= 0) &
        (df["installment_commitment"] >= 0) &
        (df["residence_since"] >= 0) &
        (df["age"] >= 18) &
        (df["existing_credits"] >= 0) &
        (df["num_dependents"] >= 1)
        ]

    # Remove rows with invalid categorical values
    allowed_categories = {
        "checking_status": {
            "<0",
            "0<=X<200",
            ">=200",
            "no checking"
        },
        "credit_history": {
            "no credits/all paid",
            "all paid",
            "existing paid",
            "delayed previously",
            "critical/other existing credit"
        },
        "purpose": {
            "new car",
            "used car",
            "furniture/equipment",
            "radio/tv",
            "domestic appliance",
            "repairs",
            "education",
            "retraining",
            "business",
            "other"
        },
        "savings_status": {
            "<100",
            "100<=X<500",
            "500<=X<1000",
            ">=1000",
            "no known savings"
        },
        "employment": {
            "unemployed",
            "<1",
            "1<=X<4",
            "4<=X<7",
            ">=7"
        },
        "other_parties": {
            "none",
            "co applicant",
            "guarantor"
        },
        "property_magnitude": {
            "real estate",
            "life insurance",
            "car",
            "no known property"
        },
        "other_payment_plans": {
            "bank",
            "stores",
            "none"
        },
        "housing": {
            "rent",
            "own",
            "for free"
        },
        "job": {
            "unemp/unskilled non res",
            "unskilled resident",
            "skilled",
            "high qualif/self emp/mgmt"
        },
        "personal_status": {
            "male div/sep",
            "female div/dep/mar",
            "male single",
            "male mar/wid",
            "female single"
        },
        "foreign_worker": {
            "yes",
            "no"
        }
    }

    for column, valid_values in allowed_categories.items():
        df[column] = df[column].astype(str).str.strip()
        df = df[df[column].isin(valid_values)]

    # Remove invalid target values
    valid_target_values = ["good", "bad"]
    df = df[df["class"].isin(valid_target_values)]

    df = df.reset_index(drop=True)
    return df
