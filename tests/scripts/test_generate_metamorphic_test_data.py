import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from scripts.generate_metamorphic_test_data import (
    generate_duration_increase_data,
    generate_checking_status_improvement_data,
    generate_savings_status_deterioration_data,
)


@pytest.fixture
def sample_X_test():
    return pd.DataFrame(
        [
            {
                "duration": 12,
                "credit_amount": 1200,
                "monthly_payment": 100,
                "checking_status": "<0",
                "savings_status": "100<=X<500",
            },
            {
                "duration": 24,
                "credit_amount": 2400,
                "monthly_payment": 100,
                "checking_status": "0<=X<200",
                "savings_status": "500<=X<1000",
            },
            {
                "duration": 36,
                "credit_amount": 3600,
                "monthly_payment": 100,
                "checking_status": ">=200",
                "savings_status": ">=1000",
            },
            {
                "duration": 48,
                "credit_amount": 4800,
                "monthly_payment": 100,
                "checking_status": "no checking",
                "savings_status": "<100",
            },
            {
                "duration": 60,
                "credit_amount": 6000,
                "monthly_payment": 100,
                "checking_status": "<0",
                "savings_status": "no known savings",
            },
        ]
    )


def test_generate_duration_increase_data_returns_dataframe(sample_X_test):
    result = generate_duration_increase_data(sample_X_test)

    assert isinstance(result, pd.DataFrame)


def test_generate_duration_increase_data_increases_duration_by_default_months(sample_X_test):
    result = generate_duration_increase_data(sample_X_test)

    expected_duration = sample_X_test["duration"] + 12

    pd.testing.assert_series_equal(
        result["duration"],
        expected_duration,
        check_names=False,
    )


def test_generate_duration_increase_data_uses_custom_months(sample_X_test):
    result = generate_duration_increase_data(
        sample_X_test,
        months=6,
    )

    expected_duration = sample_X_test["duration"] + 6

    pd.testing.assert_series_equal(
        result["duration"],
        expected_duration,
        check_names=False,
    )


def test_generate_duration_increase_data_recalculates_monthly_payment(sample_X_test):
    result = generate_duration_increase_data(sample_X_test)

    expected_monthly_payment = (
        result["credit_amount"] / result["duration"]
    )

    pd.testing.assert_series_equal(
        result["monthly_payment"],
        expected_monthly_payment,
        check_names=False,
    )


def test_generate_duration_increase_data_does_not_modify_original_dataframe(sample_X_test):
    original_df = sample_X_test.copy(deep=True)

    _ = generate_duration_increase_data(sample_X_test)

    pd.testing.assert_frame_equal(sample_X_test, original_df)


def test_generate_checking_status_improvement_data_returns_dataframe_and_mask(sample_X_test):
    transformed_df, mask = generate_checking_status_improvement_data(sample_X_test)

    assert isinstance(transformed_df, pd.DataFrame)
    assert isinstance(mask, pd.Series)


def test_generate_checking_status_improvement_data_improves_expected_categories(sample_X_test):
    transformed_df, _ = generate_checking_status_improvement_data(sample_X_test)

    expected_values = [
        "0<=X<200",
        ">=200",
        ">=200",
        "no checking",
        "0<=X<200",
    ]

    assert transformed_df["checking_status"].tolist() == expected_values


def test_generate_checking_status_improvement_mask_selects_only_improvable_categories(sample_X_test):
    _, mask = generate_checking_status_improvement_data(sample_X_test)

    expected_mask = pd.Series(
        [True, True, False, False, True],
        index=sample_X_test.index,
    )

    pd.testing.assert_series_equal(
        mask,
        expected_mask,
        check_names=False,
    )


def test_generate_checking_status_improvement_data_does_not_modify_original_dataframe(sample_X_test):
    original_df = sample_X_test.copy(deep=True)

    _ = generate_checking_status_improvement_data(sample_X_test)

    pd.testing.assert_frame_equal(sample_X_test, original_df)


def test_generate_savings_status_deterioration_data_returns_dataframe_and_mask(sample_X_test):
    transformed_df, mask = generate_savings_status_deterioration_data(sample_X_test)

    assert isinstance(transformed_df, pd.DataFrame)
    assert isinstance(mask, pd.Series)


def test_generate_savings_status_deterioration_data_deteriorates_expected_categories(sample_X_test):
    transformed_df, _ = generate_savings_status_deterioration_data(sample_X_test)

    expected_values = [
        "<100",
        "100<=X<500",
        "500<=X<1000",
        "<100",
        "no known savings",
    ]

    assert transformed_df["savings_status"].tolist() == expected_values


def test_generate_savings_status_deterioration_mask_selects_only_deterioratable_categories(sample_X_test):
    _, mask = generate_savings_status_deterioration_data(sample_X_test)

    expected_mask = pd.Series(
        [True, True, True, False, False],
        index=sample_X_test.index,
    )

    pd.testing.assert_series_equal(
        mask,
        expected_mask,
        check_names=False,
    )


def test_generate_savings_status_deterioration_data_does_not_modify_original_dataframe(sample_X_test):
    original_df = sample_X_test.copy(deep=True)

    _ = generate_savings_status_deterioration_data(sample_X_test)

    pd.testing.assert_frame_equal(sample_X_test, original_df)