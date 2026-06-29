import sys
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_POST_PROCESSING_GATE_PATH = (
    PROJECT_ROOT / "steps" / "data" / "data_post_processing_gate.py"
)

sys.path.append(str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "data_post_processing_gate",
    DATA_POST_PROCESSING_GATE_PATH,
)
data_post_processing_gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_post_processing_gate)


def create_valid_post_processed_dataframe() -> pd.DataFrame:
    rows = []

    for i in range(800):
        rows.append(
            {
                "class": 0 if i < 560 else 1,
                "duration": 24,
                "credit_amount": 3000,
                "checking_status": "<0",
                "savings_status": "<100",
                "monthly_payment": 125.0,
                "age_group": "25-40",
                "has_additional_security": 0,
                "high_risk_financial_status": 1,
            }
        )

    return pd.DataFrame(rows)


def run_gate(df: pd.DataFrame) -> pd.DataFrame:
    return data_post_processing_gate.data_post_processing_gate.entrypoint(df)


def test_data_post_processing_gate_passes_for_valid_dataframe():
    df = create_valid_post_processed_dataframe()

    result = run_gate(df)

    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_data_post_processing_gate_fails_when_row_count_too_low():
    df = create_valid_post_processed_dataframe().iloc[:799].copy()

    with pytest.raises(ValueError, match="expect_table_row_count_to_be_between"):
        run_gate(df)


def test_data_post_processing_gate_fails_when_target_column_missing():
    df = create_valid_post_processed_dataframe().drop(columns=["class"])

    with pytest.raises(ValueError, match="Target column 'class' is missing"):
        run_gate(df)


def test_data_post_processing_gate_fails_when_target_contains_invalid_values():
    df = create_valid_post_processed_dataframe()
    df.loc[0, "class"] = 2

    with pytest.raises(ValueError, match="expect_column_values_to_be_in_set"):
        run_gate(df)


def test_data_post_processing_gate_fails_when_class_distribution_is_too_imbalanced():
    df = create_valid_post_processed_dataframe()
    df["class"] = [0] * 760 + [1] * 40

    with pytest.raises(ValueError, match="Class distribution is too imbalanced"):
        run_gate(df)


def test_data_post_processing_gate_fails_when_expected_feature_is_missing():
    df = create_valid_post_processed_dataframe().drop(
        columns=["monthly_payment"]
    )

    with pytest.raises(ValueError, match="Expected feature 'monthly_payment' is missing"):
        run_gate(df)


def test_data_post_processing_gate_fails_when_generated_feature_contains_nulls():
    df = create_valid_post_processed_dataframe()
    df.loc[0, "monthly_payment"] = None

    with pytest.raises(ValueError, match="expect_column_values_to_not_be_null"):
        run_gate(df)


def test_data_post_processing_gate_fails_when_monthly_payment_is_zero():
    df = create_valid_post_processed_dataframe()
    df.loc[0, "monthly_payment"] = 0

    with pytest.raises(ValueError, match="expect_column_values_to_be_between"):
        run_gate(df)


def test_data_post_processing_gate_fails_when_has_additional_security_invalid():
    df = create_valid_post_processed_dataframe()
    df.loc[0, "has_additional_security"] = 3

    with pytest.raises(ValueError, match="expect_column_values_to_be_in_set"):
        run_gate(df)


def test_data_post_processing_gate_fails_when_high_risk_financial_status_invalid():
    df = create_valid_post_processed_dataframe()
    df.loc[0, "high_risk_financial_status"] = 3

    with pytest.raises(ValueError, match="expect_column_values_to_be_in_set"):
        run_gate(df)


def test_data_post_processing_gate_fails_when_age_group_invalid():
    df = create_valid_post_processed_dataframe()
    df.loc[0, "age_group"] = "invalid"

    with pytest.raises(ValueError, match="expect_column_values_to_be_in_set"):
        run_gate(df)


def test_data_post_processing_gate_fails_when_sensitive_attribute_is_present():
    df = create_valid_post_processed_dataframe()
    df["personal_status"] = "male single"

    with pytest.raises(ValueError, match="Sensitive/deleted attributes are still present"):
        run_gate(df)


def test_data_post_processing_gate_fails_when_foreign_worker_is_present():
    df = create_valid_post_processed_dataframe()
    df["foreign_worker"] = "yes"

    with pytest.raises(ValueError, match="Sensitive/deleted attributes are still present"):
        run_gate(df)


def test_data_post_processing_gate_fails_when_own_telephone_is_present():
    df = create_valid_post_processed_dataframe()
    df["own_telephone"] = "yes"

    with pytest.raises(ValueError, match="Sensitive/deleted attributes are still present"):
        run_gate(df)


def test_data_post_processing_gate_fails_when_age_is_present():
    df = create_valid_post_processed_dataframe()
    df["age"] = 35

    with pytest.raises(ValueError, match="Sensitive/deleted attributes are still present"):
        run_gate(df)