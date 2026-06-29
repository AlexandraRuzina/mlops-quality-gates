import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from scripts.generate_robustness_test_data import (
    generate_perturbation,
    generate_inflation_test_data,
)


@pytest.fixture
def sample_X_test():
    return pd.DataFrame(
        [
            {
                "duration": 12,
                "credit_amount": 1200.0,
                "monthly_payment": 100.0,
                "checking_status": "<0",
            },
            {
                "duration": 24,
                "credit_amount": 2400.0,
                "monthly_payment": 100.0,
                "checking_status": "0<=X<200",
            },
            {
                "duration": 36,
                "credit_amount": 3600.0,
                "monthly_payment": 100.0,
                "checking_status": ">=200",
            },
        ]
    )


def test_generate_perturbation_returns_dataframe(sample_X_test):
    result = generate_perturbation(sample_X_test)

    assert isinstance(result, pd.DataFrame)


def test_generate_perturbation_preserves_number_of_rows(sample_X_test):
    result = generate_perturbation(sample_X_test)

    assert len(result) == len(sample_X_test)


def test_generate_perturbation_does_not_modify_original_dataframe(sample_X_test):
    original_df = sample_X_test.copy(deep=True)

    _ = generate_perturbation(sample_X_test)

    pd.testing.assert_frame_equal(sample_X_test, original_df)


def test_generate_perturbation_changes_numeric_features(sample_X_test):
    result = generate_perturbation(sample_X_test)

    assert not result["duration"].equals(sample_X_test["duration"])
    assert not result["credit_amount"].equals(sample_X_test["credit_amount"])


def test_generate_perturbation_keeps_values_non_negative(sample_X_test):
    result = generate_perturbation(sample_X_test)

    assert (result["duration"] >= 0).all()
    assert (result["credit_amount"] >= 0).all()
    assert (result["monthly_payment"] >= 0).all()


def test_generate_perturbation_recalculates_monthly_payment(sample_X_test):
    result = generate_perturbation(sample_X_test)

    expected_monthly_payment = (
        result["credit_amount"] / result["duration"]
    )

    pd.testing.assert_series_equal(
        result["monthly_payment"],
        expected_monthly_payment,
        check_names=False,
    )


def test_generate_perturbation_is_reproducible(sample_X_test):
    result_1 = generate_perturbation(sample_X_test)
    result_2 = generate_perturbation(sample_X_test)

    pd.testing.assert_frame_equal(result_1, result_2)


def test_generate_perturbation_preserves_non_numeric_columns(sample_X_test):
    result = generate_perturbation(sample_X_test)

    pd.testing.assert_series_equal(
        result["checking_status"],
        sample_X_test["checking_status"],
        check_names=False,
    )


def test_generate_perturbation_handles_missing_numeric_columns():
    df = pd.DataFrame(
        [
            {
                "checking_status": "<0",
                "credit_amount": 1200.0,
                "duration": 12,
                "monthly_payment": 100.0,
            }
        ]
    )

    result = generate_perturbation(df)

    assert "monthly_payment" in result.columns


def test_generate_inflation_test_data_returns_dataframe(sample_X_test):
    result = generate_inflation_test_data(sample_X_test)

    assert isinstance(result, pd.DataFrame)


def test_generate_inflation_test_data_preserves_number_of_rows(sample_X_test):
    result = generate_inflation_test_data(sample_X_test)

    assert len(result) == len(sample_X_test)


def test_generate_inflation_test_data_increases_credit_amount_by_three_percent(sample_X_test):
    result = generate_inflation_test_data(sample_X_test)

    expected_credit_amount = sample_X_test["credit_amount"] * 1.03

    pd.testing.assert_series_equal(
        result["credit_amount"],
        expected_credit_amount,
        check_names=False,
    )


def test_generate_inflation_test_data_recalculates_monthly_payment(sample_X_test):
    result = generate_inflation_test_data(sample_X_test)

    expected_monthly_payment = (
        result["credit_amount"] / result["duration"]
    )

    pd.testing.assert_series_equal(
        result["monthly_payment"],
        expected_monthly_payment,
        check_names=False,
    )


def test_generate_inflation_test_data_does_not_modify_original_dataframe(sample_X_test):
    original_df = sample_X_test.copy(deep=True)

    _ = generate_inflation_test_data(sample_X_test)

    pd.testing.assert_frame_equal(sample_X_test, original_df)


def test_generate_inflation_test_data_preserves_unrelated_columns(sample_X_test):
    result = generate_inflation_test_data(sample_X_test)

    pd.testing.assert_series_equal(
        result["checking_status"],
        sample_X_test["checking_status"],
        check_names=False,
    )