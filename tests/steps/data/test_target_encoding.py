import sys
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TARGET_ENCODING_PATH = PROJECT_ROOT / "steps" / "data" / "target_encoding.py"

sys.path.append(str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "target_encoding",
    TARGET_ENCODING_PATH,
)
target_encoding = importlib.util.module_from_spec(spec)
spec.loader.exec_module(target_encoding)


def run_encode_target(df: pd.DataFrame) -> pd.DataFrame:
    return target_encoding.encode_target.entrypoint(df)


def test_encode_target_returns_dataframe():
    df = pd.DataFrame(
        {
            "class": ["good", "bad"],
            "duration": [12, 24],
        }
    )

    result = run_encode_target(df)

    assert isinstance(result, pd.DataFrame)


def test_encode_target_maps_good_to_zero_and_bad_to_one():
    df = pd.DataFrame(
        {
            "class": ["good", "bad", "good", "bad"],
        }
    )

    result = run_encode_target(df)

    assert result["class"].tolist() == [0, 1, 0, 1]


def test_encode_target_class_column_is_integer():
    df = pd.DataFrame(
        {
            "class": ["good", "bad"],
        }
    )

    result = run_encode_target(df)

    assert pd.api.types.is_integer_dtype(result["class"])


def test_encode_target_preserves_other_columns():
    df = pd.DataFrame(
        {
            "class": ["good", "bad"],
            "duration": [12, 24],
            "credit_amount": [1000, 2000],
        }
    )

    result = run_encode_target(df)

    assert "duration" in result.columns
    assert "credit_amount" in result.columns
    assert result["duration"].tolist() == [12, 24]
    assert result["credit_amount"].tolist() == [1000, 2000]


def test_encode_target_does_not_modify_original_dataframe():
    df = pd.DataFrame(
        {
            "class": ["good", "bad"],
            "duration": [12, 24],
        }
    )
    original_df = df.copy(deep=True)

    _ = run_encode_target(df)

    pd.testing.assert_frame_equal(df, original_df)


def test_encode_target_raises_error_for_invalid_target_value():
    df = pd.DataFrame(
        {
            "class": ["good", "invalid"],
        }
    )

    with pytest.raises(ValueError, match="Target Encoding fehlgeschlagen"):
        run_encode_target(df)


def test_encode_target_raises_error_for_missing_target_value():
    df = pd.DataFrame(
        {
            "class": ["good", None],
        }
    )

    with pytest.raises(ValueError, match="Target Encoding fehlgeschlagen"):
        run_encode_target(df)


def test_encode_target_preserves_number_of_rows():
    df = pd.DataFrame(
        {
            "class": ["good", "bad", "good"],
        }
    )

    result = run_encode_target(df)

    assert len(result) == 3