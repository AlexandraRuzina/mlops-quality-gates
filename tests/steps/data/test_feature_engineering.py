import sys
import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FEATURE_ENGINEERING_PATH = (
    PROJECT_ROOT / "steps" / "data" / "feature_engineering.py"
)

sys.path.append(str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "feature_engineering",
    FEATURE_ENGINEERING_PATH,
)
feature_engineering_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature_engineering_module)


def create_input_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "credit_amount": 2400,
                "duration": 24,
                "age": 23,
                "checking_status": "<0",
                "savings_status": "<100",
                "other_parties": "none",
                "personal_status": "female div/dep/mar",
                "foreign_worker": "yes",
                "own_telephone": "none",
            },
            {
                "credit_amount": 6000,
                "duration": 30,
                "age": 35,
                "checking_status": "0<=X<200",
                "savings_status": "500<=X<1000",
                "other_parties": "guarantor",
                "personal_status": "male single",
                "foreign_worker": "no",
                "own_telephone": "yes",
            },
        ]
    )


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    return feature_engineering_module.feature_engineering.entrypoint(df)


def test_feature_engineering_returns_dataframe():
    df = create_input_df()

    result = run_feature_engineering(df)

    assert isinstance(result, pd.DataFrame)


def test_feature_engineering_adds_calculated_feature_columns():
    df = create_input_df()

    result = run_feature_engineering(df)

    expected_columns = {
        "monthly_payment",
        "age_group",
        "has_additional_security",
        "high_risk_financial_status",
    }

    assert expected_columns.issubset(set(result.columns))


def test_feature_engineering_calculates_monthly_payment():
    df = create_input_df()

    result = run_feature_engineering(df)

    assert result.loc[0, "monthly_payment"] == 100
    assert result.loc[1, "monthly_payment"] == 200


def test_feature_engineering_extracts_female_sex():
    df = create_input_df()

    result = run_feature_engineering(df)

    assert result.loc[0, "sex"] == "female"


def test_feature_engineering_extracts_male_sex():
    df = create_input_df()

    result = run_feature_engineering(df)

    assert result.loc[1, "sex"] == "male"


def test_feature_engineering_sets_sex_to_none_for_unknown_personal_status():
    df = create_input_df()
    df.loc[0, "personal_status"] = "unknown status"

    result = run_feature_engineering(df)

    assert pd.isna(result.loc[0, "sex"])


def test_feature_engineering_removes_sensitive_columns():
    df = create_input_df()

    result = run_feature_engineering(df)

    removed_columns = {
        "personal_status",
        "foreign_worker",
        "own_telephone",
        "age",
    }

    for column in removed_columns:
        assert column not in result.columns


def test_feature_engineering_keeps_sex_column_for_fairness_gate():
    df = create_input_df()

    result = run_feature_engineering(df)

    assert "sex" in result.columns


def test_feature_engineering_does_not_modify_original_dataframe():
    df = create_input_df()
    original_df = df.copy(deep=True)

    _ = run_feature_engineering(df)

    pd.testing.assert_frame_equal(df, original_df)


def test_feature_engineering_preserves_number_of_rows():
    df = create_input_df()

    result = run_feature_engineering(df)

    assert len(result) == len(df)


def test_feature_engineering_creates_expected_binary_features():
    df = create_input_df()

    result = run_feature_engineering(df)

    assert result.loc[0, "has_additional_security"] == 0
    assert result.loc[1, "has_additional_security"] == 1

    assert result.loc[0, "high_risk_financial_status"] == 1
    assert result.loc[1, "high_risk_financial_status"] == 0