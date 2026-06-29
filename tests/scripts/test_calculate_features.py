import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from scripts.calculate_features import calculate_features


@pytest.fixture
def sample_input_df():
    return pd.DataFrame(
        [
            {
                "credit_amount": 2400,
                "duration": 24,
                "age": 23,
                "other_parties": "none",
                "checking_status": "<0",
                "savings_status": "<100",
            },
            {
                "credit_amount": 6000,
                "duration": 30,
                "age": 35,
                "other_parties": "guarantor",
                "checking_status": "0<=X<200",
                "savings_status": "500<=X<1000",
            },
            {
                "credit_amount": 12000,
                "duration": 48,
                "age": 61,
                "other_parties": "co applicant",
                "checking_status": "<0",
                "savings_status": "no known savings",
            },
        ]
    )


def test_calculate_features_returns_dataframe(sample_input_df):
    result = calculate_features(sample_input_df.copy())

    assert isinstance(result, pd.DataFrame)


def test_monthly_payment_is_calculated_correctly(sample_input_df):
    result = calculate_features(sample_input_df.copy())

    assert result.loc[0, "monthly_payment"] == 100
    assert result.loc[1, "monthly_payment"] == 200
    assert result.loc[2, "monthly_payment"] == 250


def test_age_group_is_created_correctly(sample_input_df):
    result = calculate_features(sample_input_df.copy())

    assert str(result.loc[0, "age_group"]) == "<25"
    assert str(result.loc[1, "age_group"]) == "25-40"
    assert str(result.loc[2, "age_group"]) == "60+"


def test_has_additional_security_is_created_correctly(sample_input_df):
    result = calculate_features(sample_input_df.copy())

    assert result.loc[0, "has_additional_security"] == 0
    assert result.loc[1, "has_additional_security"] == 1
    assert result.loc[2, "has_additional_security"] == 1


def test_high_risk_financial_status_is_created_correctly(sample_input_df):
    result = calculate_features(sample_input_df.copy())

    assert result.loc[0, "high_risk_financial_status"] == 1
    assert result.loc[1, "high_risk_financial_status"] == 0
    assert result.loc[2, "high_risk_financial_status"] == 1


def test_original_rows_are_preserved(sample_input_df):
    result = calculate_features(sample_input_df.copy())

    assert len(result) == len(sample_input_df)


def test_expected_feature_columns_are_added(sample_input_df):
    result = calculate_features(sample_input_df.copy())

    expected_new_columns = {
        "monthly_payment",
        "age_group",
        "has_additional_security",
        "high_risk_financial_status",
    }

    assert expected_new_columns.issubset(set(result.columns))