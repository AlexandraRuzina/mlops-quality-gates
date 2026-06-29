import sys
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_VALIDATION_GATE_PATH = (
    PROJECT_ROOT / "steps" / "data" / "data_validation_gate.py"
)

sys.path.append(str(PROJECT_ROOT))


spec = importlib.util.spec_from_file_location(
    "data_validation_gate",
    DATA_VALIDATION_GATE_PATH,
)
data_validation_gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_validation_gate)


def create_valid_dataframe() -> pd.DataFrame:
    rows = []

    for i in range(1000):
        rows.append(
            {
                "checking_status": "<0",
                "duration": 24,
                "credit_amount": 3000,
                "age": 35,
                "class": "good" if i < 700 else "bad",
            }
        )

    return pd.DataFrame(rows)


def test_data_validation_gate_returns_dataframe_for_valid_input():
    df = create_valid_dataframe()

    result = data_validation_gate.data_validation_gate.entrypoint(df)

    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_data_validation_gate_fails_for_none_input():
    with pytest.raises(ValueError, match="Dataset is empty or not loaded"):
        data_validation_gate.data_validation_gate.entrypoint(None)


def test_data_validation_gate_fails_for_empty_dataframe():
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="Dataset is empty or not loaded"):
        data_validation_gate.data_validation_gate.entrypoint(df)


def test_data_validation_gate_fails_when_target_column_is_missing():
    df = create_valid_dataframe().drop(columns=["class"])

    with pytest.raises(ValueError, match="expect_column_to_exist"):
        data_validation_gate.data_validation_gate.entrypoint(df)


def test_data_validation_gate_fails_when_row_count_is_too_low():
    df = create_valid_dataframe().iloc[:999].copy()

    with pytest.raises(ValueError, match="expect_table_row_count_to_be_between"):
        data_validation_gate.data_validation_gate.entrypoint(df)


def test_data_validation_gate_fails_when_column_has_too_many_null_values():
    df = create_valid_dataframe()

    null_indices = df.index[:60]
    df.loc[null_indices, "credit_amount"] = None

    with pytest.raises(ValueError, match="expect_column_values_to_not_be_null"):
        data_validation_gate.data_validation_gate.entrypoint(df)


def test_data_validation_gate_passes_when_column_has_null_values_below_threshold():
    df = create_valid_dataframe()

    null_indices = df.index[:40]
    df.loc[null_indices, "credit_amount"] = None

    result = data_validation_gate.data_validation_gate.entrypoint(df)

    assert isinstance(result, pd.DataFrame)


def test_data_validation_gate_fails_when_class_distribution_is_too_imbalanced():
    df = create_valid_dataframe()
    df["class"] = ["good"] * 950 + ["bad"] * 50

    with pytest.raises(ValueError, match="Class distribution is too imbalanced"):
        data_validation_gate.data_validation_gate.entrypoint(df)